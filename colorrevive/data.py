"""Datasets yielding (L, ab) tensors.

Two sources:
  * folder : any photos you put in  data/images/  (JPG/PNG, sub-folders allowed)  <- works offline
  * stl10  : STL-10 via torchvision (needs access to cs.stanford.edu, or place the extracted
             'stl10_binary' folder inside data/ yourself)
'auto' (default) uses the folder if it has images, otherwise STL-10.
"""
import random
from pathlib import Path
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, ConcatDataset, Subset
from .utils import rgb_to_lab_tensors

EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MIN_IMAGES = 20

FOLDER_HELP = """
Found only {n} image(s) in '{folder}'. Training needs far more (aim for 2,000-10,000+).
Put many colour photos in that folder (sub-folders are fine), then re-run. Good offline-friendly sources:
  * COCO val2017: https://images.cocodataset.org/zips/val2017.zip  (5,000 photos, unzip into data/images)
  * STL-10 / any Kaggle photo dataset (landscapes, flowers, animals) downloaded in your browser
Colour photos only: the grayscale input is generated from them automatically.
"""

STL_HELP = """
Could not download STL-10 (the server is unreachable from this network).
Options: (a) download any photo collection in your browser and unzip it into  data/images/
         (e.g. COCO val2017: https://images.cocodataset.org/zips/val2017.zip), then re-run; or
         (b) run on another network (home Wi-Fi / mobile hotspot / Google Colab).
"""


class FolderBase(Dataset):
    def __init__(self, paths):
        self.paths = list(paths)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return Image.open(self.paths[i]).convert("RGB"), 0


def _prepare(img, size, train):
    """Resize shorter side to `size`, then (random|centre) crop to size x size; random flip when training."""
    w, h = img.size
    s = size / min(w, h)
    img = img.resize((max(size, round(w * s)), max(size, round(h * s))), Image.BICUBIC)
    w, h = img.size
    if train:
        x, y = np.random.randint(0, w - size + 1), np.random.randint(0, h - size + 1)
    else:
        x, y = (w - size) // 2, (h - size) // 2
    img = img.crop((x, y, x + size, y + size))
    if train and np.random.rand() < 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    return img


class LabDataset(Dataset):
    def __init__(self, base, size=128, train=True):
        self.base, self.size, self.train = base, size, train

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        img = _prepare(self.base[i][0].convert("RGB"), self.size, self.train)
        return rgb_to_lab_tensors(np.asarray(img))


def _stl10(root, split):
    from torchvision.datasets import STL10
    try:
        return STL10(root, split=split, download=False)      # use local copy if present
    except RuntimeError:
        pass
    try:
        return STL10(root, split=split, download=True)
    except Exception:
        raise SystemExit(STL_HELP)


def get_datasets(dataset="auto", root="data", size=128, n_unlabeled=20000, n_val=500, max_train=None):
    """Returns (train, val, test) LabDatasets. val is a small slice of test, used only for monitoring."""
    img_dir = Path(root) / "images"
    paths = sorted(p for p in img_dir.rglob("*") if p.suffix.lower() in EXT) if img_dir.exists() else []

    if dataset == "auto":
        if 0 < len(paths) < MIN_IMAGES:
            raise SystemExit(FOLDER_HELP.format(n=len(paths), folder=img_dir))
        dataset = "folder" if paths else "stl10"

    if dataset == "folder":
        if len(paths) < MIN_IMAGES:
            raise SystemExit(FOLDER_HELP.format(n=len(paths), folder=img_dir))
        random.Random(0).shuffle(paths)                       # fixed split -> train/eval agree
        n_test = min(8000, max(2, len(paths) // 10))
        test_p, train_p = paths[:n_test], paths[n_test:]
        if max_train:
            train_p = train_p[:max_train]
        if len(train_p) < 1000:
            print(f"[warn] only {len(train_p)} training images - results will be weak; 2,000+ recommended.")
        tr, te = FolderBase(train_p), FolderBase(test_p)
        print(f"dataset=folder  train={len(tr)}  test={len(te)}")
    elif dataset == "stl10":
        parts = [_stl10(root, "train")]
        if n_unlabeled > 0:
            unl = _stl10(root, "unlabeled")
            parts.append(Subset(unl, np.random.RandomState(0).choice(len(unl), n_unlabeled, replace=False)))
        tr = ConcatDataset(parts)
        if max_train:
            tr = Subset(tr, range(min(max_train, len(tr))))
        te = _stl10(root, "test")
        print(f"dataset=stl10  train={len(tr)}  test={len(te)}")
    else:
        raise SystemExit(f"unknown dataset '{dataset}'")

    return (LabDataset(tr, size, True),
            LabDataset(Subset(te, range(min(n_val, len(te)))), size, False),
            LabDataset(te, size, False))
