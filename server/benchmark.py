"""Measure diagnosis latency with the installed model packages on this machine.

Run it on the server that will host the service to check the 2-3 second target:
    docker compose exec diagnosis python benchmark.py
    python benchmark.py --models-dir ../models --runs 20        # without Docker

The time a user waits after taking a photo is roughly
    photo upload (~100-200 KB, well under 1 s on 4G / Wi-Fi) + inference measured here.
"""
import argparse
import os
import sys
import time

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "common"))
from app.diagnosis import load_engines  # noqa: E402

TARGET_SECONDS = 2.0  # leaves time for the upload within a 2-3 s total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-dir", default=os.getenv("MODELS_DIR", "/models"))
    ap.add_argument("--runs", type=int, default=10)
    args = ap.parse_args()

    engines = load_engines(args.models_dir)  # includes one warm-up inference per package
    if not engines:
        raise SystemExit(f"no model packages in {args.models_dir}")
    rng = np.random.default_rng(0)
    photo = Image.fromarray(rng.integers(0, 255, (768, 1024, 3), dtype=np.uint8))  # size the web app uploads
    print(f"CPU cores: {os.cpu_count()}")
    ok = True
    for task_id, engine in engines.items():
        times = []
        for _ in range(args.runs):
            t0 = time.perf_counter()
            engine.diagnose(photo)
            times.append(time.perf_counter() - t0)
        p50, p95 = np.percentile(times, 50), np.percentile(times, 95)
        verdict = "OK" if p95 <= TARGET_SECONDS else "TOO SLOW - use a machine with more CPU cores or fewer members"
        ok &= p95 <= TARGET_SECONDS
        print(f"{task_id:12s} {len(engine.members)} members  p50 {p50:.2f} s  p95 {p95:.2f} s  {verdict}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
