"""Two LeafDoctor baselines, evaluated on lab (pv_test) and field (field_test) data.

  classical   128x128 RGB -> HSV colour histogram (16 bins/channel) + uniform LBP
              histogram on grayscale -> standardised -> RBF SVM.
  pretrained  frozen ImageNet MobileNetV3-Large (timm) features at 224x224 ->
              standardised -> logistic regression.

Hyperparameters are chosen on pv_val only (macro-F1). field_test is used only
for the final report. Features are cached as .npy in --features-dir.

Reads the splits written by src/prepare_data.py (data/processed/manifest.csv).
Runs on a CPU laptop; a CUDA GPU is used for feature extraction if present.

Example:
  python src/baselines.py
  python src/baselines.py --models pretrained --max-per-class 50   # quick smoke test
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

REPO_ROOT = Path(__file__).resolve().parents[1]
SPLITS = ("pv_train", "pv_val", "pv_test", "field_test")
EVAL_SPLITS = ("pv_test", "field_test")
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data")
    p.add_argument("--processed-dir", type=Path, help="default: <data-dir>/processed")
    p.add_argument("--class-map", type=Path, help="default: <data-dir>/class_map.csv")
    p.add_argument("--features-dir", type=Path, default=REPO_ROOT / "features")
    p.add_argument("--results-dir", type=Path, default=REPO_ROOT / "results")
    p.add_argument("--models", nargs="+", default=["classical", "pretrained"], choices=["classical", "pretrained"])
    p.add_argument("--pv-config", default="color", choices=["color", "segmented", "grayscale"],
                   help="PlantVillage config used for train/val/test")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-per-class", type=int, default=0,
                   help="subsample each split to N images per class (smoke tests; 0 = use all)")
    p.add_argument("--recompute-features", action="store_true", help="ignore cached features")
    p.add_argument("--jobs", type=int, default=-1, help="CPU workers for classical features / data loading")
    # classical
    p.add_argument("--classical-size", type=int, default=128)
    p.add_argument("--hsv-bins", type=int, default=16)
    p.add_argument("--lbp-points", type=int, default=8)
    p.add_argument("--lbp-radius", type=float, default=1.0)
    p.add_argument("--svm-C", type=float, nargs="+", default=[1, 10, 100])
    p.add_argument("--svm-gamma", type=float, nargs="+", default=[0.003, 0.01, 0.03, 0.1])
    p.add_argument("--svm-tune-samples", type=int, default=6000,
                   help="train images used per grid point (stratified subsample) to keep tuning CPU-friendly; "
                        "the chosen setting is refit on all of pv_train (0 = all)")
    # pretrained
    p.add_argument("--backbone", default="mobilenetv3_large_100", help="timm model name")
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr-C", type=float, nargs="+", default=[0.01, 0.1, 1.0])
    a = p.parse_args()
    a.processed_dir = a.processed_dir or a.data_dir / "processed"
    a.class_map = a.class_map or a.data_dir / "class_map.csv"
    if a.jobs < 1:
        a.jobs = os.cpu_count() or 1
    return a


def log(msg: str) -> None:
    print(msg, flush=True)


# --------------------------------------------------------------------------- data

def load_splits(args) -> tuple[list[str], dict[str, tuple[list[str], np.ndarray]]]:
    """Returns class ids (class-map order) and {split: (relative paths, labels)}."""
    with open(args.class_map, newline="", encoding="utf-8-sig") as f:
        class_ids = [r["class_id"] for r in csv.DictReader(f)]
    index = {c: i for i, c in enumerate(class_ids)}
    manifest = args.processed_dir / "manifest.csv"
    if not manifest.exists():
        sys.exit(f"{manifest} not found; run src/prepare_data.py first")
    rows = defaultdict(list)
    with open(manifest, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["dataset"] == "plantvillage" and r["config"] == args.pv_config:
                rows[f"pv_{r['split']}"].append(r)
            elif r["dataset"] == "plantdoc" and r["split"] == "field_test":
                rows["field_test"].append(r)
    missing = [s for s in SPLITS if not rows[s]]
    if missing:
        sys.exit(f"no images for {missing} in {manifest}; rerun src/prepare_data.py (pv_val needs --pv-val-frac > 0)")
    splits = {}
    for s in SPLITS:
        items = sorted(rows[s], key=lambda r: r["path"])
        if args.max_per_class:
            by_class = defaultdict(list)
            for r in items:
                by_class[r["class_id"]].append(r)
            rng = random.Random(f"{args.seed}:{s}")
            items = sorted((r for v in by_class.values() for r in rng.sample(v, min(len(v), args.max_per_class))),
                           key=lambda r: r["path"])
        splits[s] = ([r["path"] for r in items], np.array([index[r["class_id"]] for r in items]))
    return class_ids, splits


def load_rgb(path: Path, size: int) -> Image.Image:
    """Decode (fast JPEG draft for big field photos), apply EXIF rotation, resize to size x size."""
    with Image.open(path) as im:
        im.draft("RGB", (size, size))
        im = ImageOps.exif_transpose(im).convert("RGB")
        return im.resize((size, size), Image.Resampling.BILINEAR)


def cached_features(name: str, split: str, paths: list[str], compute, args) -> np.ndarray:
    """Load features from <features-dir>/<name>/<split>.npy if they match `paths`, else compute and save."""
    folder = args.features_dir / name
    feat_file, paths_file = folder / f"{split}.npy", folder / f"{split}_paths.txt"
    if (not args.recompute_features and feat_file.exists() and paths_file.exists()
            and paths_file.read_text(encoding="utf-8").splitlines() == paths):
        log(f"  {split}: cached {feat_file}")
        return np.load(feat_file)
    t = time.time()
    feats = compute([args.processed_dir / p for p in paths])
    folder.mkdir(parents=True, exist_ok=True)
    np.save(feat_file, feats)
    paths_file.write_text("\n".join(paths), encoding="utf-8")
    log(f"  {split}: {feats.shape} in {time.time() - t:.0f}s -> {feat_file}")
    return feats


# --------------------------------------------------------------------------- classical

def classical_vector(path: Path, size: int, bins: int, points: int, radius: float) -> np.ndarray:
    from skimage.feature import local_binary_pattern

    im = load_rgb(path, size)
    hsv = np.asarray(im.convert("HSV"))
    parts = [np.histogram(hsv[..., c], bins=bins, range=(0, 256))[0] for c in range(3)]
    lbp = local_binary_pattern(np.asarray(im.convert("L")), points, radius, method="uniform")
    parts.append(np.histogram(lbp, bins=points + 2, range=(0, points + 2))[0])
    return np.concatenate([h / max(h.sum(), 1) for h in parts]).astype(np.float32)


def run_classical(args, splits) -> dict:
    from joblib import Parallel, delayed
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    name = f"classical_{args.classical_size}px_hsv{args.hsv_bins}_lbp{args.lbp_points}r{args.lbp_radius:g}"
    log(f"[classical] features ({name})")

    def compute(paths):
        vecs = Parallel(n_jobs=args.jobs, batch_size=64)(
            delayed(classical_vector)(p, args.classical_size, args.hsv_bins, args.lbp_points, args.lbp_radius)
            for p in paths)
        return np.stack(vecs)

    X = {s: cached_features(name, s, splits[s][0], compute, args) for s in SPLITS}
    y = {s: splits[s][1] for s in SPLITS}

    X_tune, y_tune = X["pv_train"], y["pv_train"]
    if 0 < args.svm_tune_samples < len(y_tune):
        X_tune, _, y_tune, _ = train_test_split(X_tune, y_tune, train_size=args.svm_tune_samples,
                                                stratify=y_tune, random_state=args.seed)
    log(f"[classical] tuning C x gamma on pv_val ({len(y_tune)} train images per fit)")
    grid = []
    for C in args.svm_C:
        for gamma in args.svm_gamma:
            clf = make_pipeline(StandardScaler(), SVC(kernel="rbf", C=C, gamma=gamma)).fit(X_tune, y_tune)
            score = macro_f1(y["pv_val"], clf.predict(X["pv_val"]))
            grid.append({"C": C, "gamma": gamma, "pv_val_macro_f1": score})
            log(f"  C={C:<6g} gamma={gamma:<6g} pv_val macro-F1={score:.4f}")
    best = max(grid, key=lambda g: g["pv_val_macro_f1"])
    log(f"[classical] refit C={best['C']:g} gamma={best['gamma']:g} on all {len(y['pv_train'])} pv_train images")
    t = time.time()
    clf = make_pipeline(StandardScaler(), SVC(kernel="rbf", C=best["C"], gamma=best["gamma"]))
    clf.fit(X["pv_train"], y["pv_train"])
    return {"description": "HSV hist (16 bins/channel) + uniform LBP hist on grayscale, 128x128, RBF SVM",
            "feature_dim": int(X["pv_train"].shape[1]), "feature_cache": name,
            "selected": {"C": best["C"], "gamma": best["gamma"]}, "grid": grid,
            "fit_seconds": round(time.time() - t, 1),
            "predictions": {s: clf.predict(X[s]) for s in ("pv_val",) + EVAL_SPLITS}}


# --------------------------------------------------------------------------- pretrained

class ImageDataset:
    """Picklable dataset (DataLoader workers on Windows need this at module level)."""

    def __init__(self, paths: list[Path], size: int):
        self.paths, self.size = paths, size

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        import torch

        x = np.asarray(load_rgb(self.paths[i], self.size), dtype=np.float32) / 255.0
        x = (x - IMAGENET_MEAN) / IMAGENET_STD
        return torch.from_numpy(x.transpose(2, 0, 1).copy())


def run_pretrained(args, splits) -> dict:
    import timm
    import torch
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    name = f"{args.backbone}_{args.image_size}px"
    log(f"[pretrained] features ({name}, frozen ImageNet weights)")
    state = {}

    def compute(paths):
        if "model" not in state:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = timm.create_model(args.backbone, pretrained=True, num_classes=0).eval().to(device)
            state.update(model=model, device=device)
            log(f"  backbone on {device}")
        loader = torch.utils.data.DataLoader(ImageDataset(paths, args.image_size), batch_size=args.batch_size,
                                             num_workers=min(args.jobs, 8), shuffle=False)
        out = []
        with torch.inference_mode():
            for i, batch in enumerate(loader, 1):
                out.append(state["model"](batch.to(state["device"])).float().cpu().numpy())
                if i % 50 == 0:
                    log(f"    {i * args.batch_size}/{len(paths)}")
        return np.concatenate(out)

    X = {s: cached_features(name, s, splits[s][0], compute, args) for s in SPLITS}
    y = {s: splits[s][1] for s in SPLITS}

    log("[pretrained] tuning logistic-regression C on pv_val")
    grid, models = [], {}
    for C in args.lr_C:
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000))
        clf.fit(X["pv_train"], y["pv_train"])
        score = macro_f1(y["pv_val"], clf.predict(X["pv_val"]))
        grid.append({"C": C, "pv_val_macro_f1": score})
        models[C] = clf
        log(f"  C={C:<6g} pv_val macro-F1={score:.4f}")
    best = max(grid, key=lambda g: g["pv_val_macro_f1"])
    clf = models[best["C"]]
    return {"description": f"frozen ImageNet {args.backbone} (timm) at {args.image_size}x{args.image_size}, "
                           "logistic regression",
            "feature_dim": int(X["pv_train"].shape[1]), "feature_cache": name,
            "selected": {"C": best["C"]}, "grid": grid,
            "predictions": {s: clf.predict(X[s]) for s in ("pv_val",) + EVAL_SPLITS}}


# --------------------------------------------------------------------------- metrics and reporting

def macro_f1(y_true, y_pred) -> float:
    from sklearn.metrics import f1_score

    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def metrics(y_true, y_pred, class_ids: list[str]) -> dict:
    from sklearn.metrics import accuracy_score, f1_score

    labels = list(range(len(class_ids)))
    per_class = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    return {"n": int(len(y_true)), "accuracy": float(accuracy_score(y_true, y_pred)),
            "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
            "per_class_f1": {c: float(f) for c, f in zip(class_ids, per_class)}}


def save_confusion(y_true, y_pred, class_ids: list[str], title: str, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_ids))), normalize="true")
    fig, ax = plt.subplots(figsize=(11, 10))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(class_ids)), class_ids, rotation=90, fontsize=8)
    ax.set_yticks(range(len(class_ids)), class_ids, fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Fraction of true class")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def print_summary(results: dict) -> None:
    head = f"{'model':<12}{'lab acc':>10}{'field acc':>11}{'gap':>9}{'lab F1':>9}{'field F1':>10}"
    log("\nLab = pv_test (PlantVillage), field = field_test (PlantDoc). F1 is macro-F1.")
    log(head)
    log("-" * len(head))
    for model, r in results.items():
        lab, field = r["pv_test"], r["field_test"]
        log(f"{model:<12}{lab['accuracy']:>10.3f}{field['accuracy']:>11.3f}"
            f"{field['accuracy'] - lab['accuracy']:>+9.3f}{lab['macro_f1']:>9.3f}{field['macro_f1']:>10.3f}")


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    class_ids, splits = load_splits(args)
    log("images per split: " + ", ".join(f"{s}={len(splits[s][0])}" for s in SPLITS)
        + (f"  (SMOKE TEST: max {args.max_per_class}/class)" if args.max_per_class else ""))

    out_json = args.results_dir / "baselines.json"
    report = json.loads(out_json.read_text(encoding="utf-8")) if out_json.exists() else {}
    report.setdefault("models", {})
    runners = {"classical": run_classical, "pretrained": run_pretrained}
    for model in args.models:
        t = time.time()
        r = runners[model](args, splits)
        preds = r.pop("predictions")
        r["seconds"] = round(time.time() - t, 1)
        r["pv_val"] = metrics(splits["pv_val"][1], preds["pv_val"], class_ids)
        for s in EVAL_SPLITS:
            r[s] = metrics(splits[s][1], preds[s], class_ids)
            save_confusion(splits[s][1], preds[s], class_ids,
                           f"{model} on {s}: acc {r[s]['accuracy']:.3f}, macro-F1 {r[s]['macro_f1']:.3f}",
                           args.results_dir / "figures" / f"{model}_{s}_confusion.png")
        report["models"][model] = r

    report["setup"] = {"pv_config": args.pv_config, "seed": args.seed, "max_per_class": args.max_per_class,
                       "split_sizes": {s: len(splits[s][0]) for s in SPLITS}, "classes": class_ids,
                       "tuning": "hyperparameters chosen on pv_val macro-F1; field_test never used for tuning",
                       "python": platform.python_version()}
    args.results_dir.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print_summary({m: report["models"][m] for m in report["models"]})
    log(f"\nresults: {out_json}\nfigures: {args.results_dir / 'figures'}")


if __name__ == "__main__":
    main()
