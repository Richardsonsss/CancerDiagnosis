"""Diagnosis engine: ONNX inference of the ensemble members, voting, and the result text.

Every installed model package gets one DiagnosisEngine. The members run with ONNX
Runtime; their outputs are combined with the package's voting configuration and turned
into plain-language text. Neither numbers nor model / voting details leave this module.

Hardware: the fastest available ONNX Runtime execution provider is used - NVIDIA CUDA, then
DirectML (any DirectX 12 GPU on Windows, with the onnxruntime-directml package), then the CPU.
On the CPU each member uses its INT8 version when the package has one (kept during training only
where it is faster and keeps the validation accuracy); on a GPU all members use FP32. ORT_PROVIDERS (comma-separated) overrides the choice.

Latency: the members run in parallel threads (ONNX Runtime releases the GIL), each with
its share of the CPU cores, so a request takes about as long as the slowest member rather
than the sum of all members. One warm-up inference runs when the package is loaded.
"""
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from PIL import Image

import numpy as np

from preprocess import preprocess  # shared with training/ (common/)
from voting import vote_scores     # shared with training/ (common/)

log = logging.getLogger(__name__)

UNCERTAIN_AGREEMENT = 0.5  # below this share of agreeing members the result is reported as uncertain


PREFERRED_PROVIDERS = ["CUDAExecutionProvider", "DmlExecutionProvider", "CPUExecutionProvider"]


def select_providers():
    """Execution providers to use, fastest first (always ending with the CPU)."""
    import onnxruntime as ort

    available = ort.get_available_providers()
    wanted = [p.strip() for p in os.getenv("ORT_PROVIDERS", "").split(",") if p.strip()] or PREFERRED_PROVIDERS
    chosen = [p for p in wanted if p in available]
    return chosen if "CPUExecutionProvider" in chosen else chosen + ["CPUExecutionProvider"]


@dataclass
class Diagnosis:
    level: str      # "high" (refer to a doctor), "low", or "uncertain"
    verdict: str
    advice: str


class DiagnosisEngine:
    def __init__(self, package_dir):
        import onnxruntime as ort

        with open(os.path.join(package_dir, "package.json"), encoding="utf-8") as f:
            self.pkg = json.load(f)
        n = len(self.pkg["members"])
        self.providers = select_providers()
        gpu = self.providers[0] != "CPUExecutionProvider"
        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        opts.intra_op_num_threads = max(1, (os.cpu_count() or 1) // n)  # members share the cores
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if "DmlExecutionProvider" in self.providers:  # DirectML requirements
            opts.enable_mem_pattern = False
            opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        # On the CPU each member uses its INT8 version when the package has one (only kept where it is
        # faster); on a GPU all members use FP32 (INT8 kernels mostly fall back to the CPU there).
        files = [m.get("onnx_int8") if not gpu and m.get("onnx_int8") else m["onnx"] for m in self.pkg["members"]]
        n_int8 = sum(f != m["onnx"] for f, m in zip(files, self.pkg["members"]))
        self.precision = f"{n_int8}/{n} INT8" if n_int8 else "FP32"
        self.members = [(ort.InferenceSession(os.path.join(package_dir, f), opts, providers=self.providers),
                         m["preprocess"]) for f, m in zip(files, self.pkg["members"])]
        self.pool = ThreadPoolExecutor(max_workers=n, thread_name_prefix=f"{self.pkg['task_id']}-member")
        e = self.pkg["ensemble"]
        self.method, self.alpha, self.weights, self.threshold = e["method"], e["alpha"], e["weights"], e["referral_threshold"]
        self.classes = self.pkg["classes"]
        self.malignant_idx = [self.classes.index(c) for c in self.pkg["malignant"]]

    @property
    def info(self):
        p = self.pkg
        return {"id": p["task_id"], "name": p["name"], "modality": p["modality"], "input_hint": p["input_hint"]}

    def warm_up(self):
        """First inference allocates memory and picks kernels; do it before the first user."""
        self.member_probs(Image.new("RGB", (448, 448), (128, 100, 90)))

    @staticmethod
    def _run(member, image):
        sess, pcfg = member
        logits = sess.run(None, {"input": preprocess(image, pcfg)})[0]
        p = np.exp(logits - logits.max(1, keepdims=True))
        return p / p.sum(1, keepdims=True)

    def member_probs(self, image):
        """(n_members, 1, n_classes) softmax outputs for one PIL image, members in parallel."""
        return np.stack(list(self.pool.map(lambda m: self._run(m, image), self.members)))

    def diagnose(self, image):
        P = self.member_probs(image)
        scores = vote_scores(P, self.method, self.alpha, self.weights)[0]
        agreement = float((P[:, 0].argmax(1) == scores.argmax()).mean())
        return self.describe(scores, agreement)

    def describe(self, scores, agreement):
        """Plain-language result (no numbers)."""
        pkg = self.pkg
        if agreement < UNCERTAIN_AGREEMENT:
            return Diagnosis("uncertain", "Result: no clear conclusion is possible from this image.",
                             "Advice: the image is unclear or atypical. Retake the photo with better focus, lighting "
                             "and angle; if you are concerned, please see a doctor.")
        top = self.classes[int(scores.argmax())]
        name = pkg["class_names"][top]
        risky = float(scores[self.malignant_idx].sum()) >= self.threshold
        if top in pkg["malignant"]:
            return Diagnosis("high", f"Result: suspected {name}, which is {pkg['positive_term']}.",
                             f"Advice: {pkg['advice_malignant']}")
        if risky:
            return Diagnosis("high", f"Result: most likely {name}, but {pkg['positive_term']} cannot be ruled out.",
                             f"Advice: {pkg['advice_malignant']}")
        return Diagnosis("low", f"Result: most likely {name}; no clear signs of {pkg['positive_term']}.",
                         f"Advice: {pkg['advice_benign']}")


def load_engines(models_dir):
    """{task_id: DiagnosisEngine} for every package in models_dir (broken packages are skipped)."""
    engines = {}
    if not os.path.isdir(models_dir):
        return engines
    for d in sorted(os.listdir(models_dir)):
        path = os.path.join(models_dir, d)
        if d.startswith(".") or not os.path.exists(os.path.join(path, "package.json")):
            continue
        try:
            engines[d] = DiagnosisEngine(path)
            engines[d].warm_up()
            log.info("loaded model package %s (%d members, %s on %s)", d, len(engines[d].members),
                     engines[d].precision, engines[d].providers[0])
        except Exception:
            log.exception("could not load model package %s", d)
    return engines
