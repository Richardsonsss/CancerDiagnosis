"""Export trained members to ONNX, optionally quantize them, and build the model package.

Package layout (artifacts/<task_id>/package/, zipped to artifacts/<task_id>_package.zip):
    package.json               task info, class names, advice texts, voting configuration,
                               referral threshold, members with preprocessing, licence, precision
    members/<name>.onnx        FP32 model (used when the diagnosis computer has a GPU)
    members/<name>.int8.onnx   INT8 model, only for members where it is faster on the CPU (typically
                               transformers: ~2x faster; convolutional networks get slower, so they stay FP32)

Export uses PyTorch's current torch.export-based ONNX exporter (dynamo=True); the legacy
TorchScript exporter (deprecated since PyTorch 2.9) is only a fallback for models the new
exporter cannot handle yet.

Every member is verified before it is packaged:
  1. common/preprocess.py (used by the service) == the training framework's eval transform
  2. ONNX Runtime output == PyTorch output
INT8 versions are kept per member only if they run faster than FP32 on the CPU, and then checked
on validation images: with --quantize auto they are kept only if the ensemble's validation balanced
accuracy (INT8 where available, FP32 otherwise - exactly what a CPU-only computer runs) drops by at
most MAX_INT8_DROP.
"""
import json
import os
import shutil
import tempfile
import time
import warnings
import zipfile

import numpy as np
import torch

import paths  # noqa: F401  (makes common/ importable)
from models import build_model, build_transforms, data_config, load_member
from preprocess import preprocess
from voting import vote_scores

OPSET = 18
MAX_INT8_DROP = 0.01    # 1 percentage point of balanced accuracy
MIN_INT8_SPEEDUP = 1.2  # an INT8 member must be at least this much faster than FP32 on the CPU


def _export(model, path, input_size):
    """Write `model` to ONNX with a dynamic batch dimension; returns the exporter used."""
    x = torch.randn(2, 3, input_size[1], input_size[2])
    names = dict(input_names=["input"], output_names=["logits"])
    try:
        batch = torch.export.Dim("batch", min=1, max=256)
        program = torch.onnx.export(model, (x,), dynamo=True, dynamic_shapes=({0: batch},),
                                    opset_version=OPSET, external_data=False, **names)
        program.save(path)
        return "dynamo"
    except Exception as e:  # operator not supported by torch.export yet -> legacy exporter
        first_error = e
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # the legacy exporter warns that it is deprecated
        try:
            torch.onnx.export(model, (x,), path, dynamo=False, opset_version=17,
                              dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}}, **names)
            return "torchscript"
        except Exception as e:
            raise RuntimeError(f"ONNX export failed (dynamo: {first_error}; torchscript: {e})") from e


def is_exportable(spec, num_classes):
    """Check (random weights, temp file) that a candidate can be exported; returns (ok, exporter / reason)."""
    try:
        model = build_model(spec, num_classes, pretrained=False).eval()
        with tempfile.TemporaryDirectory() as tmp:
            exporter = _export(model, os.path.join(tmp, "m.onnx"), data_config(model)["input_size"])
        return True, exporter
    except Exception as e:
        return False, f"{type(e).__name__}: {str(e).splitlines()[0][:160]}"


def _softmax(logits):
    p = np.exp(logits - logits.max(1, keepdims=True))
    return p / p.sum(1, keepdims=True)


def _session(path):
    import onnxruntime as ort

    return ort.InferenceSession(path, providers=["CPUExecutionProvider"])


def ort_probs(session, images, pcfg, batch=32):
    """Softmax outputs of an ONNX model for a list of PIL images."""
    out = []
    for i in range(0, len(images), batch):
        x = np.concatenate([preprocess(img, pcfg) for img in images[i:i + batch]])
        out.append(_softmax(session.run(None, {"input": x})[0]))
    return np.concatenate(out)


def export_member(spec, num_classes, weights_path, onnx_path, sample_images):
    model = load_member(spec, num_classes, weights_path, "cpu")
    pcfg = data_config(model)
    _, test_tf = build_transforms(model)

    # 1. service preprocessing (common/preprocess.py) == training eval transform
    for img in sample_images:
        diff = np.abs(preprocess(img, pcfg)[0] - test_tf(img.convert("RGB")).numpy()).max()
        assert diff < 1e-4, f"{spec.name}: preprocessing mismatch {diff:.2e}"

    # 2. ONNX Runtime == PyTorch
    exporter = _export(model, onnx_path, pcfg["input_size"])
    x = np.concatenate([preprocess(img, pcfg) for img in sample_images])
    with torch.no_grad():
        ref = torch.softmax(model(torch.from_numpy(x)), 1).numpy()
    got = _softmax(_session(onnx_path).run(None, {"input": x})[0])
    err = np.abs(ref - got).max()
    assert err < 1e-3, f"{spec.name}: ONNX output differs from PyTorch by {err:.2e}"
    print(f"  [{spec.name}] exported with the {exporter} exporter, verified (max prob diff {err:.1e})")
    return pcfg, exporter


def quantize_member(fp32_path, int8_path):
    """Dynamic INT8 quantization (weights int8, activations quantized at run time).

    The torch.export-based exporter stores intermediate shape annotations (value_info) that
    ONNX Runtime's quantizer re-infers and rejects as conflicting; they are only hints, so
    they are removed from the copy that gets quantized.
    """
    import logging

    import onnx
    from onnxruntime.quantization import QuantType, quantize_dynamic

    model = onnx.load(fp32_path)
    del model.graph.value_info[:]
    with tempfile.TemporaryDirectory() as tmp:
        clean = os.path.join(tmp, "clean.onnx")
        onnx.save(model, clean)
        logging.getLogger().setLevel(logging.ERROR)  # silence the quantizer's pre-processing hint
        quantize_dynamic(clean, int8_path, weight_type=QuantType.QInt8)


def cpu_latency(path, input_size, runs=5):
    """Seconds per single-image inference on the CPU."""
    session = _session(path)
    x = np.random.default_rng(0).standard_normal((1, 3, input_size[1], input_size[2])).astype(np.float32)
    session.run(None, {"input": x})
    t0 = time.perf_counter()
    for _ in range(runs):
        session.run(None, {"input": x})
    return (time.perf_counter() - t0) / runs


def _balanced_accuracy(pred, y):
    return float(np.mean([(pred[y == c] == c).mean() for c in np.unique(y)]))


def check_int8(entries, pkg_dir, config, val_images, val_labels):
    """Validation balanced accuracy of the ensemble: all FP32 vs the CPU mix (INT8 where available)."""
    result = {}
    for precision in ("fp32", "int8"):
        t0 = time.perf_counter()
        P = np.stack([ort_probs(_session(os.path.join(pkg_dir, e["onnx_int8"] if precision == "int8" and e["onnx_int8"]
                                                        else e["onnx"])), val_images, e["preprocess"])
                      for e in entries])
        seconds = (time.perf_counter() - t0) / len(val_images)
        pred = vote_scores(P, config["method"], config["alpha"], config["weights"]).argmax(1)
        result[precision] = {"val_bal_acc": _balanced_accuracy(pred, val_labels), "sec_per_image": seconds}
    result["drop"] = result["fp32"]["val_bal_acc"] - result["int8"]["val_bal_acc"]
    result["speedup"] = result["fp32"]["sec_per_image"] / max(result["int8"]["sec_per_image"], 1e-9)
    return result


def build_package(task_id, t, specs, config, report, weights_dir, sample_images, out_dir,
                  val_images=None, val_labels=None, quantize="auto"):
    pkg = os.path.join(out_dir, "package")
    shutil.rmtree(pkg, ignore_errors=True)
    os.makedirs(os.path.join(pkg, "members"))
    entries = []
    for spec in specs:
        rel = f"members/{spec.name}.onnx"
        pcfg, exporter = export_member(spec, len(t["classes"]), os.path.join(weights_dir, f"{spec.name}.pt"),
                                       os.path.join(pkg, rel), sample_images)
        entries.append({"name": spec.name, "model_id": spec.id, "source": spec.source, "license": spec.license,
                        "onnx": rel, "onnx_int8": None, "exporter": exporter, "preprocess": pcfg})

    quantization = {"mode": quantize, "used": False, "members": {}}
    if quantize != "fp32":
        for e in entries:
            fp32, int8 = os.path.join(pkg, e["onnx"]), os.path.join(pkg, e["onnx"].replace(".onnx", ".int8.onnx"))
            quantize_member(fp32, int8)
            t32, t8 = cpu_latency(fp32, e["preprocess"]["input_size"]), cpu_latency(int8, e["preprocess"]["input_size"])
            keep_member = quantize == "int8" or t32 / t8 >= MIN_INT8_SPEEDUP
            quantization["members"][e["name"]] = {"fp32_ms": round(t32 * 1000, 1), "int8_ms": round(t8 * 1000, 1),
                                                  "int8": keep_member}
            print(f"  [{e['name']}] CPU latency FP32 {t32 * 1000:.0f} ms, INT8 {t8 * 1000:.0f} ms -> "
                  + ("INT8 kept" if keep_member else "FP32 only (INT8 not faster)"))
            if keep_member:
                e["onnx_int8"] = os.path.relpath(int8, pkg).replace(os.sep, "/")
            else:
                os.remove(int8)
        if any(e["onnx_int8"] for e in entries):
            if val_images is not None and len(val_images):
                quantization.update(check_int8(entries, pkg, config, val_images, np.asarray(val_labels)))
                print(f"  INT8 check on {len(val_images)} validation images: balanced accuracy "
                      f"{quantization['fp32']['val_bal_acc']:.4f} (FP32) vs {quantization['int8']['val_bal_acc']:.4f} "
                      f"(CPU mix), {quantization['speedup']:.1f}x faster")
            keep = quantize == "int8" or ("drop" in quantization and quantization["drop"] <= MAX_INT8_DROP)
            if not keep:
                reason = (f"it lowered validation balanced accuracy by more than {MAX_INT8_DROP:.0%}"
                          if "drop" in quantization else "it could not be checked on validation images")
                print(f"  INT8 not used because {reason} - package keeps FP32 only")
                for e in entries:
                    if e["onnx_int8"]:
                        os.remove(os.path.join(pkg, e["onnx_int8"]))
                        e["onnx_int8"] = None
            quantization["used"] = keep
        else:
            print("  INT8 is not faster for any member - package keeps FP32 only")

    package = {
        "format": 3, "task_id": task_id, "created": time.strftime("%Y-%m-%d %H:%M"),
        "name": t["name"], "modality": t["modality"],
        "classes": t["classes"], "class_names": t["class_names"], "malignant": t["malignant"],
        "positive_term": t["positive_term"], "advice_malignant": t["advice_malignant"],
        "advice_benign": t["advice_benign"], "input_hint": t["input_hint"],
        "ensemble": {"method": config["method"], "alpha": config["alpha"],
                     "weights": config["weights"], "referral_threshold": config["referral_threshold"]},
        "members": entries,
        "quantization": quantization,
        "test_report": report,  # documentation only - the web app never displays numbers
    }
    with open(os.path.join(pkg, "package.json"), "w", encoding="utf-8") as f:
        json.dump(package, f, ensure_ascii=False, indent=2)

    zip_path = os.path.join(os.path.dirname(out_dir), f"{task_id}_package.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(pkg):
            for f in files:
                full = os.path.join(root, f)
                zf.write(full, os.path.join(task_id, os.path.relpath(full, pkg)))
    print(f"package: {zip_path} ({os.path.getsize(zip_path) / 1e6:.0f} MB)")
    return zip_path
