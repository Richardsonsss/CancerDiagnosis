"""Model input preprocessing with PIL + NumPy only (no torch / timm), shared by the training
exporter and the diagnosis service (server/).

Reproduces the evaluation transform of the model's pre-training recipe:
    crop_mode "center": resize shorter side to floor(size / crop_pct) -> center crop to size
                        (timm transforms_imagenet_eval; also Hugging Face "shortest_edge + crop")
    crop_mode "squash": resize to exactly floor(size / crop_pct) (both sides) -> center crop to size
                        (Hugging Face processors that resize to a fixed height x width, e.g. SigLIP)
    then scale to [0, 1] and normalise with the model's mean / std.
The exporter checks this against the training framework's own transform for every member,
so the service feeds each ONNX model exactly what it saw during validation.
"""
import math

import numpy as np
from PIL import Image

INTERPOLATION = {"bicubic": Image.BICUBIC, "bilinear": Image.BILINEAR, "nearest": Image.NEAREST,
                 "lanczos": Image.LANCZOS}


def preprocess_config(cfg):
    """JSON-serialisable preprocessing settings (from timm.data.resolve_data_config or a HF processor)."""
    return {"input_size": [int(v) for v in cfg["input_size"]], "interpolation": cfg["interpolation"],
            "mean": list(map(float, cfg["mean"])), "std": list(map(float, cfg["std"])),
            "crop_pct": float(cfg.get("crop_pct") or 0.875), "crop_mode": cfg.get("crop_mode") or "center"}


def preprocess(image, cfg):
    """PIL image -> float32 array (1, 3, H, W) ready for the ONNX model."""
    mode = cfg.get("crop_mode") or "center"
    if mode not in ("center", "squash"):
        raise ValueError(f"unsupported crop_mode {mode}")
    img = image.convert("RGB")
    h_out, w_out = cfg["input_size"][1], cfg["input_size"][2]
    interp = INTERPOLATION[cfg["interpolation"]]

    if mode == "squash":
        new_w, new_h = math.floor(w_out / cfg["crop_pct"]), math.floor(h_out / cfg["crop_pct"])
    else:  # torchvision Resize(int): shorter side -> scale, longer side scaled and truncated
        scale = math.floor(h_out / cfg["crop_pct"])
        w, h = img.size
        if w <= h:
            new_w, new_h = scale, int(scale * h / w)
        else:
            new_w, new_h = int(scale * w / h), scale
    img = img.resize((new_w, new_h), interp)

    # torchvision CenterCrop
    top = int(round((new_h - h_out) / 2.0))
    left = int(round((new_w - w_out) / 2.0))
    img = img.crop((left, top, left + w_out, top + h_out))

    x = np.asarray(img, dtype=np.float32) / 255.0
    x = (x - np.array(cfg["mean"], dtype=np.float32)) / np.array(cfg["std"], dtype=np.float32)
    return x.transpose(2, 0, 1)[None].astype(np.float32)
