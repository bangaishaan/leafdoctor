"""Fine-tune an ImageNet-pretrained MobileNetV3-Large (timm) on PlantVillage.

Training data is pv_train, plus an extra image folder for the non-standard arms:

  standard   pv_train only
  cutpaste   pv_train + --extra-train-dir (cut-and-paste composites)
  diffusion  pv_train + --extra-train-dir (diffusion-generated images)

--extra-train-dir must contain one sub-folder per class_id (from class_map.csv).

Recipe: 224x224, AdamW, linear warmup then cosine LR, label smoothing 0.1,
mixed precision on CUDA, class-balanced sampling, and RandomResizedCrop, flips,
rotation +-30 deg, ColorJitter and Gaussian blur augmentation.

The best epoch is chosen by pv_val macro-F1 only. PlantDoc (field_val,
field_test) is never trained on and plays no part in choosing the epoch.

Outputs in <out>/<arm>_seed<seed>/:
  best.pt                              best checkpoint (state_dict + metadata)
  logits_<split>.npy, labels_<split>.npy  for pv_val, pv_test, field_val, field_test
  metrics.json                         accuracy, macro-F1, per-class F1, history

Example:
  python src/train.py --data-root data/processed --arm standard --seed 1
  python src/train.py --arm cutpaste --extra-train-dir data/cutpaste --seed 1
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import random
import sys
import time
from pathlib import Path

import numpy as np
import timm
import torch
from PIL import Image, ImageOps
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms as T

REPO_ROOT = Path(__file__).resolve().parents[1]
ARMS = ("standard", "cutpaste", "diffusion")
EVAL_SPLITS = ("pv_val", "pv_test", "field_val", "field_test")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", type=Path, default=REPO_ROOT / "data" / "processed",
                   help="folder holding plantvillage/ and plantdoc/ (the extracted Kaggle dataset)")
    p.add_argument("--arm", choices=ARMS, default="standard")
    p.add_argument("--extra-train-dir", type=Path, help="extra training images, one sub-folder per class_id")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--out", type=Path, default=REPO_ROOT / "results" / "runs",
                   help="run folder is <out>/<arm>_seed<seed>/")
    p.add_argument("--class-map", type=Path, help="default: <data-root>/class_map.csv, else data/class_map.csv")
    p.add_argument("--split-manifest", type=Path,
                   help="default: <data-root>/split_manifest.csv, else data/split_manifest.csv")
    p.add_argument("--pv-config", default="color", choices=["color", "segmented", "grayscale"])
    p.add_argument("--model", default="mobilenetv3_large_100")
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--warmup-epochs", type=float, default=1.0)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--crop-scale", type=float, nargs=2, default=[0.5, 1.0], help="RandomResizedCrop area range")
    p.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    p.add_argument("--no-amp", action="store_true", help="disable mixed precision on CUDA")
    p.add_argument("--max-per-class", type=int, default=0, help="subsample every split (smoke tests; 0 = all)")
    a = p.parse_args()

    if a.arm == "standard" and a.extra_train_dir:
        p.error("--extra-train-dir is only for the cutpaste and diffusion arms")
    if a.arm != "standard" and not a.extra_train_dir:
        p.error(f"--arm {a.arm} needs --extra-train-dir")
    if a.extra_train_dir and not a.extra_train_dir.is_dir():
        p.error(f"--extra-train-dir {a.extra_train_dir} is not a folder")
    a.class_map = a.class_map or first_existing(a.data_root / "class_map.csv", REPO_ROOT / "data" / "class_map.csv")
    a.split_manifest = a.split_manifest or first_existing(a.data_root / "split_manifest.csv",
                                                          REPO_ROOT / "data" / "split_manifest.csv")
    return a


def first_existing(*paths: Path) -> Path:
    return next((p for p in paths if p.exists()), paths[-1])


def log(msg: str) -> None:
    print(msg, flush=True)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def seed_worker(_: int) -> None:
    s = torch.initial_seed() % 2**32
    np.random.seed(s)
    random.seed(s)


# --------------------------------------------------------------------------- data

Item = tuple[Path, int]


def load_class_ids(path: Path) -> list[str]:
    if not path.exists():
        sys.exit(f"class map not found: {path}")
    with open(path, newline="", encoding="utf-8-sig") as f:
        return [r["class_id"] for r in csv.DictReader(f)]


def load_splits(args, class_ids: list[str]) -> dict[str, list[Item]]:
    if not args.split_manifest.exists():
        sys.exit(f"split manifest not found: {args.split_manifest}")
    index = {c: i for i, c in enumerate(class_ids)}
    splits: dict[str, list[Item]] = {s: [] for s in ("pv_train",) + EVAL_SPLITS}
    pv_prefix = f"plantvillage/{args.pv_config}/"
    with open(args.split_manifest, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            split = r["split"]
            if split not in splits or (split.startswith("pv_") and not r["path"].startswith(pv_prefix)):
                continue
            splits[split].append((args.data_root / r["path"], index[r["class_id"]]))
    for split, items in splits.items():
        if not items:
            sys.exit(f"no images for {split} in {args.split_manifest}")
        if not items[0][0].exists():
            sys.exit(f"{items[0][0]} not found; is --data-root ({args.data_root}) the processed data folder?")
    if args.max_per_class:
        for split, items in splits.items():
            rng = random.Random(f"{args.seed}:{split}")
            by_class: dict[int, list[Item]] = {}
            for it in items:
                by_class.setdefault(it[1], []).append(it)
            splits[split] = [it for v in by_class.values() for it in rng.sample(v, min(len(v), args.max_per_class))]
    return splits


def load_extra(folder: Path, class_ids: list[str]) -> list[Item]:
    index = {c: i for i, c in enumerate(class_ids)}
    items = []
    for sub in sorted(p for p in folder.iterdir() if p.is_dir()):
        if sub.name not in index:
            sys.exit(f"{sub} is not a class_id from the class map")
        items += [(f, index[sub.name]) for f in sorted(sub.rglob("*")) if f.suffix.lower() in IMAGE_EXTS]
    if not items:
        sys.exit(f"no images found in {folder}")
    return items


class ImageList(Dataset):
    def __init__(self, items: list[Item], transform, draft_size: int):
        self.items, self.transform, self.draft_size = items, transform, draft_size

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int):
        path, label = self.items[i]
        with Image.open(path) as im:
            im.draft("RGB", (self.draft_size, self.draft_size))  # fast decode of large field photos
            im = ImageOps.exif_transpose(im).convert("RGB")
        return self.transform(im), label


def build_transforms(args):
    normalize = [T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    train = T.Compose([
        T.RandomRotation(30),
        T.RandomResizedCrop(args.image_size, scale=tuple(args.crop_scale)),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
        T.RandomApply([T.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))], p=0.3),
        *normalize,
    ])
    evaluate = T.Compose([T.Resize((args.image_size, args.image_size)), *normalize])
    return train, evaluate


# --------------------------------------------------------------------------- training

def cosine_with_warmup(warmup_steps: int, total_steps: int):
    def factor(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))
    return factor


def param_groups(model: nn.Module, weight_decay: float) -> list[dict]:
    """No weight decay on biases and normalisation parameters."""
    decay, no_decay = [], []
    for p in model.parameters():
        if p.requires_grad:
            (no_decay if p.ndim <= 1 else decay).append(p)
    return [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]


@torch.inference_mode()
def predict(model, loader, device, use_amp: bool) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    logits, labels = [], []
    for x, y in loader:
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            out = model(x.to(device, non_blocking=True))
        logits.append(out.float().cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(logits), np.concatenate(labels)


def metrics(logits: np.ndarray, labels: np.ndarray, class_ids: list[str]) -> dict:
    pred = logits.argmax(1)
    ks = list(range(len(class_ids)))
    per_class = f1_score(labels, pred, labels=ks, average=None, zero_division=0)
    return {"n": int(len(labels)), "accuracy": float(accuracy_score(labels, pred)),
            "macro_f1": float(f1_score(labels, pred, labels=ks, average="macro", zero_division=0)),
            "per_class_f1": {c: float(f) for c, f in zip(class_ids, per_class)}}


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and not args.no_amp
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    run_dir = args.out / f"{args.arm}_seed{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    class_ids = load_class_ids(args.class_map)
    splits = load_splits(args, class_ids)
    train_items = list(splits["pv_train"])
    extra = load_extra(args.extra_train_dir, class_ids) if args.extra_train_dir else []
    train_items += extra
    log(f"device={device} amp={use_amp} arm={args.arm} seed={args.seed} run_dir={run_dir}")
    log(f"train: {len(splits['pv_train'])} pv_train + {len(extra)} extra; "
        + ", ".join(f"{s}={len(splits[s])}" for s in EVAL_SPLITS)
        + (f"  (SMOKE TEST: max {args.max_per_class}/class)" if args.max_per_class else ""))

    train_tf, eval_tf = build_transforms(args)
    labels = np.array([y for _, y in train_items])
    counts = np.bincount(labels, minlength=len(class_ids))
    if (counts == 0).any():
        sys.exit(f"no training images for {[c for c, n in zip(class_ids, counts) if n == 0]}")
    generator = torch.Generator().manual_seed(args.seed)
    sampler = WeightedRandomSampler(torch.as_tensor(1.0 / counts[labels], dtype=torch.double),
                                    num_samples=len(labels), replacement=True, generator=generator)
    loader_kw = dict(num_workers=args.workers, pin_memory=device.type == "cuda",
                     persistent_workers=args.workers > 0, worker_init_fn=seed_worker)
    train_loader = DataLoader(ImageList(train_items, train_tf, 2 * args.image_size), batch_size=args.batch_size,
                              sampler=sampler, drop_last=True, generator=generator, **loader_kw)
    eval_loaders = {s: DataLoader(ImageList(splits[s], eval_tf, args.image_size), batch_size=args.batch_size * 2,
                                  shuffle=False, **loader_kw) for s in EVAL_SPLITS}

    model = timm.create_model(args.model, pretrained=True, num_classes=len(class_ids)).to(device)
    optimizer = torch.optim.AdamW(param_groups(model, args.weight_decay), lr=args.lr)
    steps_per_epoch = len(train_loader)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, cosine_with_warmup(int(args.warmup_epochs * steps_per_epoch), args.epochs * steps_per_epoch))
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    best_f1, best_epoch, history = -1.0, 0, []
    ckpt_path = run_dir / "best.pt"
    t_start = time.time()
    for epoch in range(1, args.epochs + 1):
        t = time.time()
        model.train()
        loss_sum, correct, seen = 0.0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                out = model(x)
                loss = criterion(out, y)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            loss_sum += loss.item() * len(y)
            correct += (out.argmax(1) == y).sum().item()
            seen += len(y)

        val = metrics(*predict(model, eval_loaders["pv_val"], device, use_amp), class_ids)
        improved = val["macro_f1"] > best_f1
        if improved:
            best_f1, best_epoch = val["macro_f1"], epoch
            torch.save({"model": args.model, "state_dict": model.state_dict(), "class_ids": class_ids,
                        "image_size": args.image_size, "mean": IMAGENET_MEAN, "std": IMAGENET_STD,
                        "epoch": epoch, "pv_val_macro_f1": best_f1, "arm": args.arm, "seed": args.seed}, ckpt_path)
        history.append({"epoch": epoch, "train_loss": loss_sum / seen, "train_acc": correct / seen,
                        "pv_val_accuracy": val["accuracy"], "pv_val_macro_f1": val["macro_f1"],
                        "lr": optimizer.param_groups[0]["lr"], "seconds": round(time.time() - t, 1)})
        log(f"epoch {epoch:>3}/{args.epochs}  loss {loss_sum / seen:.4f}  train acc {correct / seen:.3f}  "
            f"pv_val acc {val['accuracy']:.4f}  macro-F1 {val['macro_f1']:.4f}"
            f"{'  *best' if improved else ''}  ({time.time() - t:.0f}s)")

    log(f"best epoch {best_epoch} (pv_val macro-F1 {best_f1:.4f}); evaluating all splits")
    model.load_state_dict(torch.load(ckpt_path, map_location=device)["state_dict"])
    results = {}
    for split in EVAL_SPLITS:
        logits, y = predict(model, eval_loaders[split], device, use_amp)
        np.save(run_dir / f"logits_{split}.npy", logits)
        np.save(run_dir / f"labels_{split}.npy", y)
        results[split] = metrics(logits, y, class_ids)
        log(f"  {split:<11} acc {results[split]['accuracy']:.4f}  macro-F1 {results[split]['macro_f1']:.4f}")

    report = {
        "arm": args.arm, "seed": args.seed, "model": args.model,
        "selection": "best epoch by pv_val macro-F1; PlantDoc not used for training or selection",
        "best_epoch": best_epoch, "splits": results, "history": history, "classes": class_ids,
        "train_images": {"pv_train": len(splits["pv_train"]), "extra": len(extra)},
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "env": {"device": str(device), "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
                "torch": torch.__version__, "timm": timm.__version__, "python": platform.python_version()},
        "seconds": round(time.time() - t_start, 1),
    }
    (run_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    log(f"saved {run_dir}")


if __name__ == "__main__":
    main()
