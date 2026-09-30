"""Fine-tuning loop shared by model screening and full training."""
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from datasets import with_transform
from models import build_model, build_transforms, save_weights


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def balanced_accuracy(pred, labels):
    """Mean per-class recall (robust to class imbalance)."""
    return float(np.mean([(pred[labels == c] == c).mean() for c in np.unique(labels)]))


def stratified_indices(targets, max_n, seed):
    """At most max_n indices, keeping the class proportions (used for fast screening)."""
    targets = np.asarray(targets)
    if len(targets) <= max_n:
        return np.arange(len(targets))
    rng = np.random.default_rng(seed)
    idx = []
    for c in np.unique(targets):
        members = np.flatnonzero(targets == c)
        idx += list(rng.choice(members, max(1, round(max_n * len(members) / len(targets))), replace=False))
    return np.sort(np.array(idx))


def _subset(ds, idx):
    if idx is None:
        return ds
    sub = Subset(ds, idx)
    sub.targets = [ds.targets[i] for i in idx]
    return sub


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, accum_steps=1):
    model.train()
    total, correct, loss_sum = 0, 0, 0.0
    optimizer.zero_grad(set_to_none=True)
    for i, (x, y) in enumerate(loader, 1):
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            logits = model(x)
            loss = criterion(logits, y)
        scaler.scale(loss / accum_steps).backward()
        if i % accum_steps == 0 or i == len(loader):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        loss_sum += loss.item() * y.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += y.size(0)
    return loss_sum / total, correct / total


@torch.no_grad()
def predict_probs(model, loader, device):
    model.eval()
    probs, labels = [], []
    for x, y in loader:
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            logits = model(x.to(device))
        probs.append(torch.softmax(logits.float(), dim=1).cpu())
        labels.append(y)
    return torch.cat(probs).numpy(), torch.cat(labels).numpy()


def _fit(spec, splits, num_classes, flips, cfg, device, batch_size, accum_steps,
         train_idx=None, eval_sets=("val", "test"), eval_idx=None, weights_path=None):
    set_seed(cfg["seed"])
    model = build_model(spec, num_classes).to(device)
    train_tf, test_tf = build_transforms(model, flips)
    train_ds = _subset(with_transform(splits["train"], train_tf), train_idx)
    g = torch.Generator().manual_seed(cfg["seed"])
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=g, drop_last=len(train_ds) > batch_size,
                        num_workers=cfg["workers"], pin_memory=True)

    weight = None
    if cfg["class_weights"]:  # inverse class frequency, normalised to mean 1
        counts = np.maximum(np.bincount(train_ds.targets, minlength=num_classes), 1)
        weight = torch.tensor(counts.sum() / (num_classes * counts), dtype=torch.float32, device=device)
    params = [p for p in model.parameters() if p.requires_grad]
    lr = spec.lr
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"])
    criterion = nn.CrossEntropyLoss(weight=weight, label_smoothing=0.1)
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")

    for epoch in range(1, cfg["epochs"] + 1):
        t0 = time.time()
        loss, acc = train_one_epoch(model, loader, criterion, optimizer, scaler, device, accum_steps)
        scheduler.step()
        print(f"  [{spec.name}] epoch {epoch}/{cfg['epochs']}  loss {loss:.4f}  train acc {acc:.4f}  "
              f"({time.time() - t0:.0f}s, batch {batch_size}x{accum_steps}, lr {lr:g})")

    out = {}
    for s in eval_sets:
        ds = _subset(with_transform(splits[s], test_tf), eval_idx)
        out[s] = predict_probs(model, DataLoader(ds, batch_size=batch_size, shuffle=False,
                                                 num_workers=cfg["workers"], pin_memory=True), device)
    if weights_path:
        save_weights(model, weights_path)
    del model
    return out


def fit(spec, splits, num_classes, flips, cfg, device, **kw):
    """Fine-tune one model; on CUDA out-of-memory retry with half the batch and twice the
    gradient accumulation (same effective batch). Returns {split: (probs, labels)}."""
    batch_size, accum_steps = cfg["batch_size"], 1
    while True:
        try:
            return _fit(spec, splits, num_classes, flips, cfg, device, batch_size, accum_steps, **kw)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if batch_size <= 4:
                raise
            batch_size, accum_steps = batch_size // 2, accum_steps * 2
            print(f"  [{spec.name}] CUDA out of memory -> retry with batch {batch_size} x {accum_steps}")
        finally:
            if device.type == "cuda":
                torch.cuda.empty_cache()
