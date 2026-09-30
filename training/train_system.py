"""Training back end (runs on any Linux VM with enough compute, ideally an NVIDIA GPU).

From a user prompt in any language, such as "recognise skin cancer" or
"Quiero reconocer el cáncer de piel", it:
  1. understands the prompt (local language model, or translation + keywords) and picks the
     cancer type and its public dataset                        (prompt_parser, registry)
  2. downloads and splits the dataset                          (datasets)
  3. screens the candidate models - curated pool, optionally the newest models on the Hugging
     Face Hub (--pool latest) and any model added with --add_model: each is fine-tuned briefly
     on a stratified subset and scored on validation balanced accuracy
  4. checks ONNX exportability down the ranking and fully trains the TOP_K best exportable
     candidates                                                (trainer, export_onnx)
  5. chooses the voting configuration and referral threshold on validation, reports on test
  6. exports the members to ONNX, adds INT8 versions for CPU-only diagnosis computers when they
     keep the validation accuracy, and writes the model package

Everything is written to artifacts/<task_id>/; the file to copy to the diagnosis computer is
artifacts/<task_id>_package.zip. Re-running resumes: finished steps and members are skipped.

Usage:
    python train_system.py "recognise skin cancer"
    python train_system.py "Hirntumor im MRT erkennen" --pool quick --top_k 3
    python train_system.py --task lung --pool latest --add_model timm:iformer_l.in1k --add_model hf:facebook/dinov2-base
"""
import argparse
import json
import os
import time

import numpy as np
import torch

import paths
from datasets import prepare, summary
from ensemble_config import configure
from export_onnx import build_package, is_exportable
from models import POOL_CHOICES, ModelSpec, candidate_pool
from prompt_parser import PromptError, understand_prompt
from registry import malignant_indices, task as get_task
from trainer import balanced_accuracy, fit, stratified_indices
from translation import TranslationError, to_default_language

INT8_CHECK_IMAGES = 600  # validation images used to compare FP32 and INT8 members


def load_json(path, default):
    return json.load(open(path, encoding="utf-8")) if os.path.exists(path) else default


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("prompt", nargs="?", help="what to recognise, in any language, e.g. 'recognise skin cancer'")
    ap.add_argument("--task", help="task id instead of a prompt (skin, skin_phone, breast, colorectal, lung, brain)")
    ap.add_argument("--pool", choices=POOL_CHOICES, default="core",
                    help="candidates: core (curated, incl. latest foundation models), quick (10 strongest), "
                         "latest (core + newest timm models on the Hugging Face Hub)")
    ap.add_argument("--latest_n", type=int, default=8, help="newest model families added by --pool latest")
    ap.add_argument("--add_model", action="append", default=[], metavar="SOURCE:ID",
                    help="extra candidate, e.g. timm:<timm id> or hf:<transformers repo> (repeatable)")
    ap.add_argument("--top_k", type=int, default=5, help="members kept after screening")
    ap.add_argument("--screen_epochs", type=int, default=1)
    ap.add_argument("--screen_train", type=int, default=3000, help="max training images per screening run")
    ap.add_argument("--screen_val", type=int, default=1500, help="max validation images per screening run")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--weight_decay", type=float, default=0.05)
    ap.add_argument("--no_class_weights", action="store_true", help="plain (unweighted) cross-entropy")
    ap.add_argument("--quantize", choices=["auto", "int8", "fp32"], default="auto",
                    help="INT8 members for CPU-only diagnosis computers: auto keeps them if validation "
                         "balanced accuracy drops by <= 1 point")
    ap.add_argument("--prompt_model", choices=["llm", "translate", "keywords"], default="llm",
                    help="prompt understanding: local language model (default), translation + keywords, "
                         "or keywords only (English)")
    ap.add_argument("--translator", choices=["local", "aws"], default="local",
                    help="translation for --prompt_model translate: local model or Amazon Translate")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--s3_uri", help="also upload the package, e.g. s3://my-bucket/cancer-models/")
    args = ap.parse_args()

    if args.task:
        task_id, understood = args.task, None
    elif args.prompt:
        try:
            understood = understand_prompt(args.prompt, method=args.prompt_model,
                                           translator=lambda p: to_default_language(p, args.translator))
        except (PromptError, TranslationError) as e:
            raise SystemExit(str(e))
        task_id = understood["task_id"]
        print(f"prompt: {understood['prompt']}")
        if understood["english"] != understood["prompt"]:
            print(f"in English: {understood['english']}  (understood by: {understood['method']})")
    else:
        raise SystemExit("give a prompt, e.g.  python train_system.py \"recognise skin cancer\"")
    t = get_task(task_id)
    classes, mal_idx = t["classes"], malignant_indices(t)
    out = os.path.join(paths.ARTIFACTS_DIR, task_id)
    wdir = os.path.join(out, "members")
    os.makedirs(wdir, exist_ok=True)
    if understood:
        save_json(understood, os.path.join(out, "prompt.json"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base = dict(seed=args.seed, workers=args.workers, class_weights=not args.no_class_weights,
                weight_decay=args.weight_decay, batch_size=args.batch_size)
    t_start = time.time()
    print(f"task: {task_id} - {t['name']} ({t['modality']}) | device {device}")

    # 1-2. dataset
    print("\n[1/5] dataset")
    splits = prepare(task_id)
    summary(task_id, splits)

    # 3. screening. The candidate list is frozen in candidates.json so that a resumed run
    #    screens and trains the same models even if newer ones have been published since.
    cand_path = os.path.join(out, "candidates.json")
    saved = {d["name"]: ModelSpec.from_dict(d) for d in load_json(cand_path, [])}
    for spec in candidate_pool(args.pool, args.add_model, args.latest_n):
        saved.setdefault(spec.name, spec)
    specs = saved
    save_json([s.to_dict() for s in specs.values()], cand_path)
    print(f"\n[2/5] screening {len(specs)} candidates (pool '{args.pool}'"
          + (f", added: {', '.join(args.add_model)}" if args.add_model else "") + ")")
    screen_path = os.path.join(out, "screening.json")
    screening = load_json(screen_path, {})
    tr_idx = stratified_indices(splits["train"].targets, args.screen_train, args.seed)
    va_idx = stratified_indices(splits["val"].targets, args.screen_val, args.seed)
    for name, spec in specs.items():
        if name in screening:
            continue
        try:
            r = fit(spec, splits, len(classes), t["flips"], {**base, "epochs": args.screen_epochs}, device,
                    train_idx=tr_idx, eval_sets=("val",), eval_idx=va_idx)
            p, y = r["val"]
            screening[name] = balanced_accuracy(p.argmax(1), y)
            print(f"  [{name}] screening val balanced acc {screening[name]:.4f}")
        except Exception as e:  # not downloadable (gated), unsupported by this timm version, ...
            print(f"  [{name}] skipped: {type(e).__name__}: {str(e).splitlines()[0][:160]}")
            screening[name] = None
        save_json(screening, screen_path)
    ranked = sorted((n for n, s in screening.items() if s is not None and n in specs), key=lambda n: -screening[n])
    print("  ranking:", ", ".join(f"{n} {screening[n]:.3f}" for n in ranked))

    # 4. exportability down the ranking, then full training of the top_k exportable candidates
    export_path = os.path.join(out, "exportable.json")
    exportable = load_json(export_path, {})
    members = []
    for name in ranked:
        if name not in exportable:
            ok, info = is_exportable(specs[name], len(classes))
            exportable[name] = {"ok": ok, "info": info}
            save_json(exportable, export_path)
            if not ok:
                print(f"  [{name}] not exportable to ONNX, skipped: {info}")
        if exportable[name]["ok"]:
            members.append(name)
        if len(members) == args.top_k:
            break
    print("  selected:", members)

    print(f"\n[3/5] training {len(members)} members for {args.epochs} epochs")
    for name in members:
        files = [os.path.join(wdir, f) for f in (f"{name}.pt", f"{name}_val.npy", f"{name}_test.npy")]
        if all(os.path.exists(f) for f in files):
            print(f"  [{name}] already trained, skipped")
            continue
        t0 = time.time()
        r = fit(specs[name], splits, len(classes), t["flips"], {**base, "epochs": args.epochs}, device,
                weights_path=files[0])
        np.save(files[1], r["val"][0])
        np.save(files[2], r["test"][0])
        np.save(os.path.join(wdir, "val_labels.npy"), r["val"][1])
        np.save(os.path.join(wdir, "test_labels.npy"), r["test"][1])
        print(f"  [{name}] val balanced acc {balanced_accuracy(r['val'][0].argmax(1), r['val'][1]):.4f} "
              f"({(time.time() - t0) / 60:.1f} min)")

    # 5. ensemble configuration (validation) + report (test)
    print("\n[4/5] ensemble configuration")
    P_val = np.stack([np.load(os.path.join(wdir, f"{n}_val.npy")) for n in members])
    P_test = np.stack([np.load(os.path.join(wdir, f"{n}_test.npy")) for n in members])
    y_val, y_test = np.load(os.path.join(wdir, "val_labels.npy")), np.load(os.path.join(wdir, "test_labels.npy"))
    member_bal = [balanced_accuracy(p.argmax(1), y_val) for p in P_val]
    config, rep = configure(P_val, y_val, P_test, y_test, classes, mal_idx, member_bal)
    rep["members_test_bal_acc"] = {n: balanced_accuracy(p.argmax(1), y_test) for n, p in zip(members, P_test)}
    save_json({"task_id": task_id, "members": members, **config, "test_report": rep}, os.path.join(out, "config.json"))
    print(f"  test: accuracy {rep['acc'] * 100:.2f}%, balanced accuracy {rep['bal_acc'] * 100:.2f}%, "
          f"referral sensitivity {100 * (rep['referral_sensitivity'] or 0):.1f}% / "
          f"specificity {100 * (rep['referral_specificity'] or 0):.1f}%")

    # 6. ONNX export (+ INT8) and package
    print("\n[5/5] ONNX export and model package")
    samples = [splits["test"][i][0] for i in np.linspace(0, len(splits["test"]) - 1, 4).astype(int)]
    check_idx = stratified_indices(splits["val"].targets, INT8_CHECK_IMAGES, args.seed)
    val_images = [splits["val"][int(i)][0] for i in check_idx] if args.quantize != "fp32" else None
    val_labels = [splits["val"].targets[int(i)] for i in check_idx] if args.quantize != "fp32" else None
    zip_path = build_package(task_id, t, [specs[n] for n in members], config, rep, wdir, samples, out,
                             val_images, val_labels, args.quantize)
    if args.s3_uri:
        import boto3  # credentials: IAM role, AWS profile or environment variables

        bucket, _, prefix = args.s3_uri.removeprefix("s3://").partition("/")
        key = f"{prefix.rstrip('/')}/{os.path.basename(zip_path)}".lstrip("/")
        boto3.client("s3").upload_file(zip_path, bucket, key)
        print(f"uploaded to s3://{bucket}/{key} - restart the diagnosis service to load it")
    print(f"\ndone in {(time.time() - t_start) / 3600:.1f} h - model package: {zip_path}")


if __name__ == "__main__":
    main()
