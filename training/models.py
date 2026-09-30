"""Candidate model pool, model building / saving / loading, and per-model preprocessing.

Every candidate is a ModelSpec. Candidates come from three places:
  1. CORE           curated list: established ImageNet backbones plus the latest foundation
                    models (DINOv2, DINOv3, SigLIP 2, ConvNeXt V2, EUPE, AIMv2, iFormer, EfficientViM,
                    MedSigLIP, ...)
  2. discover_latest()  the newest timm models published on the Hugging Face Hub, so models
                    released after this code was written join the pool automatically
                    (--pool latest)
  3. parse_model()  any timm or Hugging Face transformers image model given on the command line
                    (--add_model timm:<id> / hf:<repo>)
Two sources are supported: timm (incl. "hf-hub:<repo>" ids) and Hugging Face transformers
(AutoModelForImageClassification). Gated models (e.g. MedSigLIP) need the HF_TOKEN environment
variable and accepted terms on huggingface.co; without them the candidate is skipped.
Licences differ between models (see ModelSpec.license) and are recorded in the model package.
"""
import os
import re
from dataclasses import asdict, dataclass

import timm
import timm.data
import torch
import torch.nn as nn

import paths  # noqa: F401  (makes common/ importable)
from preprocess import preprocess_config

VIT_LR, CNN_LR = 3e-5, 1e-4  # convolutional networks fine-tune better with a larger learning rate
CNN_PATTERN = re.compile(r"resnet|convnext|efficientnet|efficientvim|mobilenet|regnet|lcnet|cpubone|ghostnet|rexnet|"
                         r"dla|densenet|inception|nfnet|repvgg|fastvit|iformer|lowformer|mambaout|starnet", re.I)


@dataclass(frozen=True)
class ModelSpec:
    name: str              # unique short name used in logs, file names and the package
    id: str                # timm model id (or "hf-hub:<repo>") / Hugging Face transformers repo id
    source: str = "timm"   # "timm" | "hf"
    lr: float = VIT_LR
    img_size: int = 0      # 0 = the model's own input size; set to shrink e.g. DINOv2 518 -> 224
    license: str = ""
    added_by: str = "core"  # core | latest | user

    def to_dict(self):
        return asdict(self)

    @staticmethod
    def from_dict(d):
        return ModelSpec(**d)


def _t(name, model_id, lr=VIT_LR, img_size=0, license=""):
    return ModelSpec(name, model_id, "timm", lr, img_size, license)


CORE = [
    # established ImageNet-pretrained backbones
    _t("ViT", "vit_base_patch16_224.augreg2_in21k_ft_in1k", license="apache-2.0"),
    _t("ViT-CLIP", "vit_base_patch16_clip_224.openai_ft_in1k", license="apache-2.0"),
    _t("DeiT", "deit_base_patch16_224.fb_in1k", license="apache-2.0"),
    _t("DeiT3", "deit3_base_patch16_224.fb_in22k_ft_in1k", license="apache-2.0"),
    _t("BEiT", "beit_base_patch16_224.in22k_ft_in22k_in1k", license="apache-2.0"),
    _t("EVA02", "eva02_base_patch14_224.mim_in22k", license="mit"),
    _t("Swin", "swin_base_patch4_window7_224.ms_in22k_ft_in1k", license="mit"),
    _t("SwinV2-CR", "swinv2_cr_small_ns_224.sw_in1k", license="apache-2.0"),
    _t("MaxViT", "maxvit_small_tf_224.in1k", license="apache-2.0"),
    _t("CaiT", "cait_s24_224.fb_dist_in1k", license="apache-2.0"),
    _t("ConViT", "convit_base.fb_in1k", license="apache-2.0"),
    _t("PiT", "pit_b_224.in1k", license="apache-2.0"),
    _t("Twins", "twins_svt_base.in1k", license="apache-2.0"),
    _t("DaViT", "davit_base.msft_in1k", license="apache-2.0"),
    _t("XCiT", "xcit_small_24_p16_224.fb_in1k", license="apache-2.0"),
    _t("MViTv2", "mvitv2_base.fb_in1k", license="apache-2.0"),
    _t("VOLO", "volo_d1_224.sail_in1k", license="apache-2.0"),
    _t("LeViT", "levit_384.fb_dist_in1k", license="apache-2.0"),
    _t("CoaT", "coat_lite_small.in1k", license="apache-2.0"),
    _t("NesT", "nest_base_jx.goog_in1k", license="apache-2.0"),
    _t("CrossViT", "crossvit_base_240.in1k", license="apache-2.0"),
    _t("FlexiViT", "flexivit_base.1200ep_in1k", license="apache-2.0"),
    _t("Hiera", "hiera_base_224.mae_in1k_ft_in1k", license="cc-by-nc-4.0"),
    _t("ResNet50", "resnet50.a1_in1k", CNN_LR, license="apache-2.0"),
    _t("ConvNeXt-S", "convnext_small.fb_in22k_ft_in1k", CNN_LR, license="apache-2.0"),
    _t("EfficientNetB2", "efficientnet_b2.ra_in1k", CNN_LR, license="apache-2.0"),
    # latest foundation models (self-supervised / image-text pre-training, 2023-2026)
    _t("DINOv2", "vit_base_patch14_dinov2.lvd142m", img_size=224, license="apache-2.0"),
    _t("DINOv2-reg", "vit_base_patch14_reg4_dinov2.lvd142m", img_size=224, license="apache-2.0"),
    _t("DINOv3", "vit_base_patch16_dinov3.lvd1689m", license="dinov3-license"),
    _t("DINOv3-ConvNeXt", "convnext_base.dinov3_lvd1689m", CNN_LR, license="dinov3-license"),
    _t("SigLIP2", "vit_base_patch16_siglip_224.v2_webli", license="apache-2.0"),
    _t("ConvNeXtV2", "convnextv2_base.fcmae_ft_in22k_in1k", CNN_LR, license="cc-by-nc-4.0"),
    _t("EUPE-ConvNeXt", "convnext_base.eupe_lvd1689m", CNN_LR, license="fair-noncommercial-research-license"),
    _t("AIMv2", "aimv2_large_patch14_224.apple_pt", license="apple-ascl"),
    _t("iFormer", "iformer_m.in1k", CNN_LR, license="mit"),
    _t("EfficientViM", "efficientvim_m4.e450_in1k", CNN_LR, license="mit"),
    # medical foundation model (gated on Hugging Face: needs HF_TOKEN + accepted terms)
    ModelSpec("MedSigLIP", "google/medsiglip-448", "hf", VIT_LR, 0, "health-ai-developer-foundations"),
]

POOLS = {
    "core": [s.name for s in CORE],
    # smaller screening pool for a quick run: strongest families only
    "quick": ["DINOv2", "DINOv3", "SigLIP2", "EVA02", "Swin", "BEiT", "ConvNeXtV2", "DINOv3-ConvNeXt",
              "EfficientNetB2", "iFormer"],
}
POOL_CHOICES = ["core", "quick", "latest"]


def guess_lr(model_id):
    return CNN_LR if CNN_PATTERN.search(model_id) else VIT_LR


def parse_model(text):
    """'timm:<id>', 'hf:<repo>' or a bare timm id -> ModelSpec (for --add_model)."""
    source, _, model_id = text.partition(":") if text.split(":")[0] in ("timm", "hf") else ("timm", "", text)
    name = model_id.split("/")[-1].split(".")[0]
    return ModelSpec(name, model_id, source, guess_lr(model_id), 0, "", "user")


def discover_latest(n=8, known_ids=(), min_params=5e6, max_params=350e6, max_input=448):
    """Newest timm model families on the Hugging Face Hub that the installed timm can build.

    One checkpoint per family (the newest), 5-350M parameters, input size <= max_input.
    Needs internet access; returns [] (with a message) when the Hub is not reachable.
    """
    import json

    from huggingface_hub import HfApi, hf_hub_download

    try:
        models = list(HfApi().list_models(author="timm", sort="created_at", limit=400,
                                          expand=["safetensors", "cardData", "createdAt"]))
    except Exception as e:
        print(f"  model discovery skipped (Hugging Face Hub not reachable: {type(e).__name__})")
        return []
    known = set(known_ids)
    found, families = [], set()
    for m in models:
        repo = m.id.split("/", 1)[1]
        arch = repo.split(".")[0]
        family = arch.split("_")[0]  # e.g. iformer_l2_distilled -> iformer: one checkpoint per family
        params = (m.safetensors.total if getattr(m, "safetensors", None) else 0) or 0
        if repo in known or family in families or not timm.is_model(arch) or not (min_params <= params <= max_params):
            continue
        try:
            cfg = json.load(open(hf_hub_download(m.id, "config.json")))
            size = max(cfg.get("pretrained_cfg", {}).get("input_size", [3, 224, 224])[1:])
        except Exception:
            continue
        if size > max_input:
            continue
        card = getattr(m, "card_data", None)
        license_ = str(getattr(card, "license", "") or "") if card is not None else ""
        families.add(family)
        found.append(ModelSpec(arch, repo, "timm", guess_lr(repo), 0, license_, "latest"))
        if len(found) >= n:
            break
    return found


def candidate_pool(pool, add_models=(), latest_n=8):
    """ModelSpecs to screen for --pool / --add_model."""
    by_name = {s.name: s for s in CORE}
    specs = list(CORE) if pool in ("core", "latest") else [by_name[n] for n in POOLS["quick"]]
    if pool == "latest":
        specs += discover_latest(latest_n, known_ids={s.id for s in CORE})
    for text in add_models:
        spec = parse_model(text)
        if spec.name not in {s.name for s in specs}:
            specs.append(spec)
    return specs


# ---------------------------------------------------------------- Hugging Face transformers backbones
class HFImageClassifier(nn.Module):
    """AutoModelForImageClassification wrapped to take a pixel tensor and return logits."""

    def __init__(self, repo, num_classes, pretrained=True):
        super().__init__()
        from transformers import AutoConfig, AutoImageProcessor, AutoModelForImageClassification

        token = os.getenv("HF_TOKEN") or None
        if pretrained:
            self.net = AutoModelForImageClassification.from_pretrained(
                repo, num_labels=num_classes, ignore_mismatched_sizes=True, token=token)
        else:
            cfg = AutoConfig.from_pretrained(repo, num_labels=num_classes, token=token)
            self.net = AutoModelForImageClassification.from_config(cfg)
        self.data_cfg = _hf_data_config(AutoImageProcessor.from_pretrained(repo, token=token))

    def forward(self, x):
        return self.net(pixel_values=x).logits


def _hf_data_config(proc):
    """Hugging Face image processor settings -> preprocess() config (dict or transformers-5 SizeDict)."""
    def get(obj, key):
        if obj is None:
            return None
        return obj.get(key) if isinstance(obj, dict) else getattr(obj, key, None)

    size = proc.size if not isinstance(proc.size, int) else {"shortest_edge": proc.size}
    crop = getattr(proc, "crop_size", None) if getattr(proc, "do_center_crop", False) else None
    crop_hw = (get(crop, "height"), get(crop, "width")) if get(crop, "height") else None
    resample = {0: "nearest", 1: "lanczos", 2: "bilinear", 3: "bicubic"}.get(int(getattr(proc, "resample", 3) or 3), "bicubic")
    if get(size, "height"):  # fixed height x width (e.g. SigLIP), optionally followed by a centre crop
        h, w = get(size, "height"), get(size, "width")
        out = crop_hw or (h, w)
        crop_pct, mode = out[0] / h, "squash"
    else:  # shortest edge (+ optional centre crop), e.g. DINOv2
        edge = get(size, "shortest_edge")
        out = crop_hw or (edge, edge)
        crop_pct, mode = out[0] / edge, "center"
    return preprocess_config({"input_size": (3, out[0], out[1]), "interpolation": resample,
                              "mean": proc.image_mean, "std": proc.image_std, "crop_pct": crop_pct,
                              "crop_mode": mode})


# ---------------------------------------------------------------- building / preprocessing
def build_model(spec, num_classes, pretrained=True):
    """Backbone with a new `num_classes` head (pre-trained weights when `pretrained`)."""
    if spec.source == "hf":
        return HFImageClassifier(spec.id, num_classes, pretrained)
    kwargs = {"img_size": spec.img_size} if spec.img_size else {}
    model = timm.create_model(spec.id, pretrained=pretrained, num_classes=num_classes, **kwargs)
    if spec.img_size:  # make the data config follow the overridden input size
        model.pretrained_cfg = {**model.pretrained_cfg, "input_size": (3, spec.img_size, spec.img_size)}
    return model


def data_config(model):
    """preprocess() config of a built model (what the diagnosis service must feed it)."""
    if isinstance(model, HFImageClassifier):
        return model.data_cfg
    return preprocess_config(timm.data.resolve_data_config({}, model=model))


def build_transforms(model, flips="horizontal"):
    """(train, eval) torchvision transforms; eval matches common/preprocess.py exactly."""
    vflip = 0.5 if flips == "both" else 0.0
    if not isinstance(model, HFImageClassifier):
        cfg = timm.data.resolve_data_config({}, model=model)
        return (timm.data.create_transform(**cfg, is_training=True, hflip=0.5, vflip=vflip),
                timm.data.create_transform(**cfg, is_training=False))
    import math

    from torchvision import transforms as T

    c = model.data_cfg
    h, w = c["input_size"][1:]
    interp = {"bicubic": T.InterpolationMode.BICUBIC, "bilinear": T.InterpolationMode.BILINEAR,
              "nearest": T.InterpolationMode.NEAREST, "lanczos": T.InterpolationMode.LANCZOS}[c["interpolation"]]
    norm = [T.ToTensor(), T.Normalize(c["mean"], c["std"])]
    train = T.Compose([T.RandomResizedCrop((h, w), scale=(0.3, 1.0), interpolation=interp),
                       T.RandomHorizontalFlip(), T.RandomVerticalFlip(vflip), T.ColorJitter(0.2, 0.2, 0.2)] + norm)
    if c["crop_mode"] == "squash":
        resize = T.Resize((math.floor(h / c["crop_pct"]), math.floor(w / c["crop_pct"])), interpolation=interp)
    else:
        resize = T.Resize(math.floor(h / c["crop_pct"]), interpolation=interp)
    return train, T.Compose([resize, T.CenterCrop((h, w))] + norm)


def save_weights(model, path):
    """Fine-tuned weights in half precision (half the disk space; loaded back as float32)."""
    torch.save({k: v.half() if v.is_floating_point() else v for k, v in model.state_dict().items()}, path)


def load_member(spec, num_classes, path, device="cpu"):
    """Rebuild a fine-tuned member from its saved weights (for ONNX export)."""
    model = build_model(spec, num_classes, pretrained=False)
    state = torch.load(path, map_location="cpu")
    model.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in state.items()})
    return model.to(device).eval()
