"""Download and load the public dataset of a registered task as train / val / test sets.

Four source types (see registry.py):
  ham10000        Harvard Dataverse zips; lesion-grouped stratified split (no lesion on two sides)
  pad_ufes20      Mendeley Data zips (smartphone photos); patient-grouped stratified split
  medmnist        MedMNIST+ .npz from Zenodo; official train / val / test split. Arrays are
                  converted once to .npy so they can be memory-mapped (PathMNIST is ~5 GB raw)
  kaggle_folders  zip from Kaggle's public download API; class folders located by name;
                  split 'folders' (dataset's own test folder + 15% val) or 'random' (70/15/15)
Splits are written once (split.csv / .npy files) and reused by every model.

Data lives in <project>/data/<task_id>/.
"""
import copy
import csv
import os
import random
import urllib.error
import urllib.request
import zipfile
from collections import defaultdict

import numpy as np

from paths import PROJECT_DIR
from registry import task as get_task

DATA_ROOT = os.path.join(PROJECT_DIR, "data")
# Some servers (e.g. Harvard Dataverse) answer 403 to Python's default User-Agent.
USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) CancerDiagnosisSystem/1.0"
IMAGE_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
SPLIT_SEED = 42


# ---------------------------------------------------------------- download
def download(url, path, retries=5, chunk=1 << 20):
    """Download with a browser-like User-Agent; resumes a partial .part file
    (HTTP Range) after a dropped connection instead of starting over."""
    if os.path.exists(path):
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    part = path + ".part"
    print(f"downloading {os.path.basename(path)}")
    for attempt in range(1, retries + 1):
        done = os.path.getsize(part) if os.path.exists(part) else 0
        headers = {"User-Agent": USER_AGENT}
        if done:
            headers["Range"] = f"bytes={done}-"
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=60) as r:
                if done and r.status != 206:  # server ignored Range: start over
                    done = 0
                total = done + int(r.headers.get("Content-Length", 0))
                with open(part, "ab" if done else "wb") as f:
                    while True:
                        block = r.read(chunk)
                        if not block:
                            break
                        f.write(block)
                        done += len(block)
                        if total:
                            print(f"\r  {done / 1e6:.0f}/{total / 1e6:.0f} MB", end="")
            if total and done < total:
                raise IOError(f"incomplete download: {done}/{total} bytes")
            os.replace(part, path)
            print()
            return
        except (urllib.error.URLError, IOError, TimeoutError) as e:
            print(f"\n  attempt {attempt}/{retries} failed: {e}")
            if attempt == retries:
                raise


# ---------------------------------------------------------------- dataset classes
class ImageListDataset:
    """[(image path, label)] -> (transformed RGB image, label)."""

    def __init__(self, samples, transform=None):
        self.samples, self.targets, self.transform = samples, [y for _, y in samples], transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        from PIL import Image

        path, y = self.samples[i]
        img = Image.open(path).convert("RGB")
        return (self.transform(img) if self.transform else img), y


class ArrayDataset:
    """uint8 image array (N, H, W[, C]) + labels -> (transformed RGB image, label)."""

    def __init__(self, images, labels, transform=None):
        self.images, self.targets, self.transform = images, [int(y) for y in labels], transform

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, i):
        from PIL import Image

        img = Image.fromarray(np.asarray(self.images[i])).convert("RGB")
        return (self.transform(img) if self.transform else img), self.targets[i]


def with_transform(ds, transform):
    """Shallow copy of a dataset with another transform (data is shared, not copied)."""
    ds = copy.copy(ds)
    ds.transform = transform
    return ds


def _write_split(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "label", "split"])
        w.writerows(rows)


def _read_split(path, root):
    out = defaultdict(list)
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            out[r["split"]].append((os.path.join(root, r["path"]), int(r["label"])))
    return {s: ImageListDataset(out[s]) for s in ("train", "val", "test")}


def _stratified(items, fractions, rng):
    """Split [(key, label)] per label into consecutive parts of the given fractions."""
    by_label = defaultdict(list)
    for key, y in items:
        by_label[y].append(key)
    parts = [[] for _ in fractions]
    for y in sorted(by_label):
        keys = sorted(by_label[y])
        rng.shuffle(keys)
        start = 0
        for i, frac in enumerate(fractions):
            end = len(keys) if i == len(fractions) - 1 else start + round(frac * len(keys))
            parts[i] += [(k, y) for k in keys[start:end]]
            start = end
    return parts


# ---------------------------------------------------------------- HAM10000
DATAVERSE = "https://dataverse.harvard.edu/api/access/datafile"


def _load_ham10000(t, root):
    meta = os.path.join(root, "HAM10000_metadata.csv")
    download(f"{DATAVERSE}/4338392?format=original", meta)
    img_dir = os.path.join(root, "images")
    if not os.path.isdir(img_dir) or len(os.listdir(img_dir)) < 10015:
        os.makedirs(img_dir, exist_ok=True)
        for name, fid in (("HAM10000_images_part_1.zip", 3172585), ("HAM10000_images_part_2.zip", 3172584)):
            zpath = os.path.join(root, name)
            download(f"{DATAVERSE}/{fid}", zpath)
            print(f"extracting {name}")
            with zipfile.ZipFile(zpath) as zf:
                for m in zf.infolist():
                    if m.filename.lower().endswith(".jpg"):
                        with zf.open(m) as src, open(os.path.join(img_dir, os.path.basename(m.filename)), "wb") as dst:
                            dst.write(src.read())
            os.remove(zpath)

    split_path = os.path.join(root, "split.csv")
    if not os.path.exists(split_path):
        with open(meta, newline="") as f:
            rows = list(csv.DictReader(f))
        lesions = {}  # lesion_id -> (label, [image_id])
        for r in rows:
            lesions.setdefault(r["lesion_id"], (t["classes"].index(r["dx"]), []))[1].append(r["image_id"])
        rng = random.Random(SPLIT_SEED)
        # split lesions (not images): 20% test, then 15% of the rest val
        test, rest = _stratified([(k, v[0]) for k, v in lesions.items()], [0.2, 0.8], rng)
        val, train = _stratified(rest, [0.15, 0.85], rng)
        side = {k: s for s, part in (("train", train), ("val", val), ("test", test)) for k, _ in part}
        _write_split(split_path, [(f"images/{img}.jpg", y, side[lid])
                                  for lid, (y, imgs) in sorted(lesions.items()) for img in sorted(imgs)])
    return _read_split(split_path, root)


# ---------------------------------------------------------------- PAD-UFES-20 (smartphone photos)
MENDELEY = "https://data.mendeley.com/public-files/datasets/zr7vgbcyr2/files"
PAD_FILES = {"metadata.csv": "fa850265-57da-48f0-ba3e-998b3e44b1f6",
             "imgs_part_1.zip": "1cc2f71f-20a2-412d-b746-a9b9bc20c966",
             "imgs_part_2.zip": "559a60ed-5504-475d-996c-6a8bc253b5e7",
             "imgs_part_3.zip": "34dcdf8e-e5f1-4b35-aa0b-5135051ff852"}


def _load_pad_ufes20(t, root):
    meta = os.path.join(root, "metadata.csv")
    download(f"{MENDELEY}/{PAD_FILES['metadata.csv']}/file_downloaded", meta)
    img_dir = os.path.join(root, "images")
    with open(meta, newline="") as f:
        rows = list(csv.DictReader(f))
    if not os.path.isdir(img_dir) or len(os.listdir(img_dir)) < len(rows):
        os.makedirs(img_dir, exist_ok=True)
        for name in ("imgs_part_1.zip", "imgs_part_2.zip", "imgs_part_3.zip"):
            zpath = os.path.join(root, name)
            download(f"{MENDELEY}/{PAD_FILES[name]}/file_downloaded", zpath)
            print(f"extracting {name}")
            with zipfile.ZipFile(zpath) as zf:
                for m in zf.infolist():
                    if m.filename.lower().endswith(IMAGE_EXT):
                        with zf.open(m) as src, open(os.path.join(img_dir, os.path.basename(m.filename)), "wb") as dst:
                            dst.write(src.read())
            os.remove(zpath)

    split_path = os.path.join(root, "split.csv")
    if not os.path.exists(split_path):
        patients = {}  # patient_id -> (label of the first lesion, [(image, label)])
        for r in rows:
            y = t["classes"].index(r["diagnostic"])
            patients.setdefault(r["patient_id"], (y, []))[1].append((r["img_id"], y))
        rng = random.Random(SPLIT_SEED)
        # split patients (not images): no patient appears on two sides
        test, rest = _stratified([(k, v[0]) for k, v in patients.items()], [0.2, 0.8], rng)
        val, train = _stratified(rest, [0.15, 0.85], rng)
        side = {k: s for s, part in (("train", train), ("val", val), ("test", test)) for k, _ in part}
        _write_split(split_path, [(f"images/{img}", y, side[pid])
                                  for pid, (_, imgs) in sorted(patients.items()) for img, y in sorted(imgs)])
    return _read_split(split_path, root)


# ---------------------------------------------------------------- MedMNIST+
ZENODO = "https://zenodo.org/api/records/10519652/files"


def _load_medmnist(t, root):
    ds = t["dataset"]
    stem = f"{ds['name']}_{ds['size']}" if ds["size"] != 28 else ds["name"]
    npy = {s: os.path.join(root, f"{stem}_{s}_images.npy") for s in ("train", "val", "test")}
    if not all(os.path.exists(p) for p in npy.values()):
        npz_path = os.path.join(root, f"{stem}.npz")
        download(f"{ZENODO}/{stem}.npz/content", npz_path)
        print(f"converting {stem}.npz to memory-mappable .npy")
        with np.load(npz_path) as z:
            for s in ("train", "val", "test"):
                np.save(os.path.join(root, f"{stem}_{s}_labels.npy"), z[f"{s}_labels"].ravel())
                np.save(npy[s], z[f"{s}_images"])
        os.remove(npz_path)
    return {s: ArrayDataset(np.load(npy[s], mmap_mode="r"), np.load(os.path.join(root, f"{stem}_{s}_labels.npy")))
            for s in ("train", "val", "test")}


# ---------------------------------------------------------------- Kaggle folders
KAGGLE = "https://www.kaggle.com/api/v1/datasets/download"


def _load_kaggle_folders(t, root):
    ds = t["dataset"]
    split_path = os.path.join(root, "split.csv")
    if not os.path.exists(split_path):
        raw = os.path.join(root, "raw")
        if not os.path.isdir(raw):
            zpath = os.path.join(root, "archive.zip")
            download(f"{KAGGLE}/{ds['slug']}", zpath)
            print("extracting archive.zip")
            with zipfile.ZipFile(zpath) as zf:
                zf.extractall(raw)
            os.remove(zpath)
        items = []  # (relative path, label, is_test)
        for dirpath, _, files in os.walk(raw):
            cls = ds["class_dirs"].get(os.path.basename(dirpath))
            if cls is None:
                continue
            is_test = ds.get("test_dir") in os.path.relpath(dirpath, raw).split(os.sep)
            for f in sorted(files):
                if f.lower().endswith(IMAGE_EXT):
                    rel = os.path.relpath(os.path.join(dirpath, f), root).replace(os.sep, "/")
                    items.append((rel, t["classes"].index(cls), is_test))
        missing = set(ds["class_dirs"]) - {os.path.basename(os.path.dirname(os.path.join(root, p))) for p, _, _ in items}
        assert not missing, f"class folders not found in the archive: {missing}"
        rng = random.Random(SPLIT_SEED)
        if ds["split"] == "folders":
            test = [(p, y) for p, y, is_t in items if is_t]
            val, train = _stratified([(p, y) for p, y, is_t in items if not is_t], [0.15, 0.85], rng)
        else:
            train, val, test = _stratified([(p, y) for p, y, _ in items], [0.70, 0.15, 0.15], rng)
        _write_split(split_path, [(p, y, s) for s, part in (("train", train), ("val", val), ("test", test))
                                  for p, y in part])
    return _read_split(split_path, root)


LOADERS = {"ham10000": _load_ham10000, "pad_ufes20": _load_pad_ufes20, "medmnist": _load_medmnist,
           "kaggle_folders": _load_kaggle_folders}


def prepare(task_id):
    """Download (once) and return {'train', 'val', 'test'} datasets without transforms."""
    t = get_task(task_id)
    root = os.path.join(DATA_ROOT, task_id)
    os.makedirs(root, exist_ok=True)
    return LOADERS[t["dataset"]["type"]](t, root)


def summary(task_id, splits):
    t = get_task(task_id)
    for s, ds in splits.items():
        counts = np.bincount(ds.targets, minlength=len(t["classes"]))
        print(f"  {s:5s} {len(ds):6d} |", " ".join(f"{c} {n}" for c, n in zip(t["classes"], counts)))


if __name__ == "__main__":
    import sys

    for tid in sys.argv[1:] or ["skin"]:
        print(tid)
        summary(tid, prepare(tid))
