"""Analyse a finished training run from its saved logits (no model inference needed).

For pv_test (lab) and field_test (field) it reports:
  1. top-1 / top-3 accuracy and macro-F1
  2. crop-level accuracy (each class mapped to its plant species, e.g. all
     tomato classes -> Tomato; a prediction counts if the species is right)
  3. a row-normalised confusion matrix PNG for field_test
  4. the 5 most common confusions
  5. a grid of random field_test images exactly as the model saw them
     (eval preprocessing), titled with true and predicted labels
and a comparison table against results/baselines.json.

It first checks that the logits' class order matches class_map.csv and that
each logit row lines up with the right image (labels rebuilt from
split_manifest.csv must equal the saved labels); it stops if not.

Outputs go to <run>/analysis/: analysis.json, report.md, confusion_field_test.png,
field_test_grid.png.

Example:
  python src/analyze_run.py results/runs/standard_seed1
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train  # noqa: E402  (reuse the exact split loading and eval preprocessing)

REPO_ROOT = Path(__file__).resolve().parents[1]
ALL_SPLITS = ("pv_val", "pv_test", "field_val", "field_test")
REPORT_SPLITS = ("pv_test", "field_test")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir", type=Path, help="e.g. results/runs/standard_seed1")
    p.add_argument("--data-root", type=Path, default=REPO_ROOT / "data" / "processed",
                   help="local processed images (for the image grid)")
    p.add_argument("--class-map", type=Path, default=REPO_ROOT / "data" / "class_map.csv")
    p.add_argument("--split-manifest", type=Path, default=REPO_ROOT / "data" / "split_manifest.csv")
    p.add_argument("--baselines", type=Path, default=REPO_ROOT / "results" / "baselines.json")
    p.add_argument("--out", type=Path, help="default: <run_dir>/analysis")
    p.add_argument("--grid", type=int, default=24, help="number of field_test images in the grid")
    p.add_argument("--seed", type=int, default=0, help="seed for picking grid images")
    a = p.parse_args()
    a.out = a.out or a.run_dir / "analysis"
    return a


def log(msg: str) -> None:
    print(msg, flush=True)


def load_class_map(path: Path) -> tuple[list[str], list[str]]:
    """class_ids in file order, and the crop (plant species) of each, from the label 'Crop — condition'."""
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    return [r["class_id"] for r in rows], [r["label"].split(" — ")[0].strip() for r in rows]


# --------------------------------------------------------------------------- checks

def check_alignment(args, run_metrics: dict, class_ids: list[str], logits: dict, labels: dict) -> tuple[dict, dict]:
    """Verify class order and row alignment. Returns (checks, splits rebuilt like train.py)."""
    run_args = run_metrics["args"]
    split_args = SimpleNamespace(split_manifest=args.split_manifest, data_root=args.data_root,
                                 pv_config=run_args["pv_config"], max_per_class=run_args["max_per_class"],
                                 seed=run_args["seed"])
    splits = train.load_splits(split_args, class_ids)
    checks = {"metrics.json classes == class_map.csv order": run_metrics["classes"] == class_ids}
    ckpt_path = args.run_dir / "best.pt"
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        checks["best.pt class_ids == class_map.csv order"] = ckpt["class_ids"] == class_ids
    for s in ALL_SPLITS:
        checks[f"{s}: logits have {len(class_ids)} columns"] = logits[s].shape == (len(labels[s]), len(class_ids))
        rebuilt = np.array([y for _, y in splits[s]])
        checks[f"{s}: saved labels == labels rebuilt from split_manifest (row order)"] = (
            len(rebuilt) == len(labels[s]) and bool((rebuilt == labels[s]).all()))
        saved = run_metrics["splits"][s]
        pred = logits[s].argmax(1)
        checks[f"{s}: accuracy/macro-F1 recomputed from logits == metrics.json"] = bool(
            np.isclose(accuracy_score(labels[s], pred), saved["accuracy"])
            and np.isclose(f1_score(labels[s], pred, labels=range(len(class_ids)), average="macro",
                                    zero_division=0), saved["macro_f1"]))
    # If logit columns were permuted relative to class_map, these classes would have ~0 recall in the lab.
    recall = confusion_matrix(labels["pv_test"], logits["pv_test"].argmax(1),
                              labels=range(len(class_ids)), normalize="true").diagonal()
    checks["pv_test: every class's recall > 0.9 (logit column i is class_map class i)"] = bool(recall.min() > 0.9)
    return checks, splits


# --------------------------------------------------------------------------- metrics

def split_report(logits: np.ndarray, y: np.ndarray, class_ids: list[str], crops: list[str]) -> dict:
    k = len(class_ids)
    pred = logits.argmax(1)
    top3 = np.argsort(-logits, axis=1)[:, :3]
    crop_names = sorted(set(crops))
    crop_of = np.array([crop_names.index(c) for c in crops])
    true_crop, pred_crop = crop_of[y], crop_of[pred]
    per_crop = {c: {"n": int((true_crop == i).sum()),
                    "accuracy": float((pred_crop[true_crop == i] == i).mean())}
                for i, c in enumerate(crop_names) if (true_crop == i).any()}
    cm = confusion_matrix(y, pred, labels=range(k))
    off = [(cm[i, j], i, j) for i in range(k) for j in range(k) if i != j and cm[i, j] > 0]
    top_confusions = [{"true": class_ids[i], "predicted": class_ids[j], "count": int(n),
                       "share_of_true_class": float(n / cm[i].sum())}
                      for n, i, j in sorted(off, key=lambda t: (-t[0], t[1], t[2]))[:5]]
    return {"n": int(len(y)),
            "top1_accuracy": float((pred == y).mean()),
            "top3_accuracy": float((top3 == y[:, None]).any(1).mean()),
            "macro_f1": float(f1_score(y, pred, labels=range(k), average="macro", zero_division=0)),
            "crop_accuracy": float((pred_crop == true_crop).mean()),
            "per_crop_accuracy": per_crop,
            "top5_confusions": top_confusions}


# --------------------------------------------------------------------------- figures

def save_confusion(logits, y, class_ids, title: str, path: Path) -> None:
    cm = confusion_matrix(y, logits.argmax(1), labels=range(len(class_ids)), normalize="true")
    fig, ax = plt.subplots(figsize=(11, 10))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(class_ids)), class_ids, rotation=90, fontsize=8)
    ax.set_yticks(range(len(class_ids)), class_ids, fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Fraction of true class")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def save_grid(items, logits, y, class_ids, image_size: int, n: int, seed: int, path: Path) -> list[dict]:
    """Show images after the run's eval preprocessing (what the model actually received)."""
    _, eval_tf = train.build_transforms(SimpleNamespace(image_size=image_size, crop_scale=[0.5, 1.0]))
    dataset = train.ImageList(items, eval_tf, image_size)
    idx = sorted(random.Random(seed).sample(range(len(items)), min(n, len(items))))
    probs = torch.softmax(torch.from_numpy(logits), 1).numpy()
    mean, std = np.array(train.IMAGENET_MEAN), np.array(train.IMAGENET_STD)
    cols = 6
    rows = (len(idx) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.1, rows * 3.6))
    shown = []
    for ax, i in zip(axes.flat, idx):
        x, label = dataset[i]
        assert label == y[i], "image/label row mismatch"
        img = np.clip(x.numpy().transpose(1, 2, 0) * std + mean, 0, 1)
        p = int(probs[i].argmax())
        ok = p == y[i]
        ax.imshow(img)
        ax.set_title(f"T: {class_ids[y[i]]}\nP: {class_ids[p]} ({probs[i, p]:.0%})",
                     fontsize=8.5, color="#1a7f37" if ok else "#c62828")
        ax.set_xlabel(Path(items[i][0]).name, fontsize=7)
        ax.set_xticks([])
        ax.set_yticks([])
        shown.append({"file": Path(items[i][0]).as_posix(), "true": class_ids[y[i]], "predicted": class_ids[p],
                      "confidence": float(probs[i, p]), "correct": bool(ok)})
    for ax in list(axes.flat)[len(idx):]:
        ax.axis("off")
    n_ok = sum(s["correct"] for s in shown)
    fig.suptitle(f"field_test: {len(idx)} random images after eval preprocessing "
                 f"({image_size}x{image_size} resize) - {n_ok}/{len(idx)} correct (green)", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return shown


# --------------------------------------------------------------------------- report

def comparison_rows(baselines_path: Path, run_name: str, report: dict) -> list[dict]:
    rows = []
    if baselines_path.exists():
        names = {"classical": "Classical: HSV + LBP, RBF SVM", "pretrained": "Frozen MobileNetV3 + logistic reg."}
        for key, b in json.loads(baselines_path.read_text(encoding="utf-8"))["models"].items():
            rows.append({"model": names.get(key, key), "lab_acc": b["pv_test"]["accuracy"],
                         "field_acc": b["field_test"]["accuracy"], "lab_f1": b["pv_test"]["macro_f1"],
                         "field_f1": b["field_test"]["macro_f1"], "lab_top3": None, "field_top3": None})
    rows.append({"model": f"Fine-tuned MobileNetV3 ({run_name})", "lab_acc": report["pv_test"]["top1_accuracy"],
                 "field_acc": report["field_test"]["top1_accuracy"], "lab_f1": report["pv_test"]["macro_f1"],
                 "field_f1": report["field_test"]["macro_f1"], "lab_top3": report["pv_test"]["top3_accuracy"],
                 "field_top3": report["field_test"]["top3_accuracy"]})
    return rows


def fmt(v) -> str:
    return "–" if v is None else f"{v:.3f}"


def write_report(path: Path, run_name: str, run_metrics: dict, checks: dict, report: dict,
                 comparison: list[dict], grid: list[dict]) -> None:
    env = run_metrics.get("env", {})
    lines = [f"# Run analysis: {run_name}", "",
             f"Arm `{run_metrics['arm']}`, seed {run_metrics['seed']}, best epoch {run_metrics['best_epoch']} "
             f"of {len(run_metrics['history'])} (chosen by pv_val macro-F1), trained on {env.get('gpu') or env.get('device')} "
             f"in {run_metrics['seconds'] / 60:.0f} min.", "",
             "## Class order and row alignment", "", "| Check | Result |", "|---|---|"]
    lines += [f"| {name} | {'pass' if ok else '**FAIL**'} |" for name, ok in checks.items()]
    lines += ["", "## Metrics", "", "| Split | n | Top-1 acc | Top-3 acc | Macro-F1 | Crop-level acc |",
              "|---|---|---|---|---|---|"]
    for s in REPORT_SPLITS:
        r = report[s]
        lines.append(f"| {s} | {r['n']} | {r['top1_accuracy']:.3f} | {r['top3_accuracy']:.3f} | "
                     f"{r['macro_f1']:.3f} | {r['crop_accuracy']:.3f} |")
    lines += ["", "Crop-level accuracy counts a prediction as right when its plant species matches "
                  "(e.g. any tomato class for a tomato leaf).", "",
              "## Comparison with baselines", "", "Lab = pv_test (PlantVillage), field = field_test (PlantDoc).", "",
              "| Model | Lab acc | Field acc | Lab→field gap | Lab macro-F1 | Field macro-F1 | Field top-3 |",
              "|---|---|---|---|---|---|---|"]
    for c in comparison:
        lines.append(f"| {c['model']} | {c['lab_acc']:.3f} | {c['field_acc']:.3f} | "
                     f"{c['field_acc'] - c['lab_acc']:+.3f} | {c['lab_f1']:.3f} | {c['field_f1']:.3f} | "
                     f"{fmt(c['field_top3'])} |")
    for s in REPORT_SPLITS:
        lines += ["", f"## 5 most common confusions: {s}", "", "| True | Predicted | Count | Share of true class |",
                  "|---|---|---|---|"]
        lines += [f"| {c['true']} | {c['predicted']} | {c['count']} | {c['share_of_true_class']:.0%} |"
                  for c in report[s]["top5_confusions"]]
    lines += ["", "## Crop-level accuracy by species", "", "| Crop | pv_test n | pv_test acc | field_test n | field_test acc |",
              "|---|---|---|---|---|"]
    for crop in sorted(report["field_test"]["per_crop_accuracy"]):
        lab = report["pv_test"]["per_crop_accuracy"].get(crop, {"n": 0, "accuracy": float("nan")})
        fld = report["field_test"]["per_crop_accuracy"][crop]
        lines.append(f"| {crop} | {lab['n']} | {lab['accuracy']:.3f} | {fld['n']} | {fld['accuracy']:.3f} |")
    n_ok = sum(g["correct"] for g in grid)
    lines += ["", "## Figures", "",
              "- `confusion_field_test.png`: row-normalised confusion matrix on field_test.",
              f"- `field_test_grid.png`: {len(grid)} random field_test images after eval preprocessing; "
              f"{n_ok}/{len(grid)} correct (green = correct, red = wrong).", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    run_name = args.run_dir.name
    run_metrics = json.loads((args.run_dir / "metrics.json").read_text(encoding="utf-8"))
    class_ids, crops = load_class_map(args.class_map)
    logits = {s: np.load(args.run_dir / f"logits_{s}.npy") for s in ALL_SPLITS}
    labels = {s: np.load(args.run_dir / f"labels_{s}.npy") for s in ALL_SPLITS}

    checks, splits = check_alignment(args, run_metrics, class_ids, logits, labels)
    for name, ok in checks.items():
        log(f"  [{'ok' if ok else 'FAIL'}] {name}")
    if not all(checks.values()):
        sys.exit("class order / row alignment check failed; not writing an analysis")

    report = {s: split_report(logits[s], labels[s], class_ids, crops) for s in REPORT_SPLITS}
    args.out.mkdir(parents=True, exist_ok=True)
    save_confusion(logits["field_test"], labels["field_test"], class_ids,
                   f"{run_name} on field_test: acc {report['field_test']['top1_accuracy']:.3f}, "
                   f"macro-F1 {report['field_test']['macro_f1']:.3f}", args.out / "confusion_field_test.png")
    grid = save_grid(splits["field_test"], logits["field_test"], labels["field_test"], class_ids,
                     run_metrics["args"]["image_size"], args.grid, args.seed, args.out / "field_test_grid.png")
    comparison = comparison_rows(args.baselines, run_name, report)

    (args.out / "analysis.json").write_text(json.dumps(
        {"run": run_name, "checks": checks, "splits": report, "comparison": comparison, "grid": grid},
        indent=2), encoding="utf-8")
    write_report(args.out / "report.md", run_name, run_metrics, checks, report, comparison, grid)

    log(f"\n{'split':<12}{'top-1':>8}{'top-3':>8}{'macro-F1':>10}{'crop acc':>10}")
    for s in REPORT_SPLITS:
        r = report[s]
        log(f"{s:<12}{r['top1_accuracy']:>8.3f}{r['top3_accuracy']:>8.3f}{r['macro_f1']:>10.3f}{r['crop_accuracy']:>10.3f}")
    log(f"\n{'model':<44}{'lab acc':>9}{'field acc':>11}{'lab F1':>9}{'field F1':>10}")
    for c in comparison:
        log(f"{c['model']:<44}{c['lab_acc']:>9.3f}{c['field_acc']:>11.3f}{c['lab_f1']:>9.3f}{c['field_f1']:>10.3f}")
    for s in REPORT_SPLITS:
        log(f"\ntop confusions on {s}:")
        for c in report[s]["top5_confusions"]:
            log(f"  {c['true']:>24} -> {c['predicted']:<24} {c['count']:>4}  ({c['share_of_true_class']:.0%} of true class)")
    log(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
