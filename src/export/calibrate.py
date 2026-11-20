"""Temperature scaling and an "unsure" threshold for the web app.

- Temperature T: minimises the NLL of softmax(logits / T) on the run's pv_val logits.
- Threshold t on the calibrated top-1 probability, chosen on field_val (PlantDoc) logits: the lowest
  t whose accepted predictions reach --target-acc while keeping at least --min-coverage of images;
  if no t reaches the target, the t with the highest accepted accuracy at --min-coverage or more.
  Below t the app shows "Unsure". field_test is only reported, never used to choose anything.

Writes calibration.json: temperature, threshold, and coverage / accepted accuracy on field_val
(plus field_test and pv_val for reporting).

Example:
  python src/export/calibrate.py --run-dir results/runs/standard_ft25_seed1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import log_softmax, softmax

REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, default=REPO_ROOT / "web" / "model" / "calibration.json")
    p.add_argument("--target-acc", type=float, default=0.5)
    p.add_argument("--min-coverage", type=float, default=0.1)
    return p.parse_args()


def load(run: Path, split: str):
    return np.load(run / f"logits_{split}.npy"), np.load(run / f"labels_{split}.npy")


def fit_temperature(z: np.ndarray, y: np.ndarray) -> float:
    nll = lambda lt: -log_softmax(z / np.exp(lt), axis=1)[np.arange(len(y)), y].mean()
    return float(np.exp(minimize_scalar(nll, bounds=(-3, 3), method="bounded").x))


def stats(z, y, T, t) -> dict:
    p = softmax(z / T, axis=1)
    conf, ok = p.max(1), p.argmax(1) == y
    keep = conf >= t
    return {"n": int(len(y)), "coverage": float(keep.mean()),
            "accepted_accuracy": float(ok[keep].mean()) if keep.any() else None,
            "overall_accuracy": float(ok.mean())}


def choose_threshold(z, y, T, target, min_cov) -> tuple[float, str]:
    conf = softmax(z / T, axis=1).max(1)
    cands = np.unique(np.round(np.concatenate([[0.0], conf]), 4))
    rows = [(t, stats(z, y, T, t)) for t in cands]
    rows = [(t, s) for t, s in rows if s["coverage"] >= min_cov and s["accepted_accuracy"] is not None]
    hit = [(t, s) for t, s in rows if s["accepted_accuracy"] >= target]
    if hit:
        return float(min(hit, key=lambda r: r[0])[0]), f"lowest threshold with field_val accepted accuracy >= {target}"
    best = max(rows, key=lambda r: (r[1]["accepted_accuracy"], -r[0]))
    return float(best[0]), (f"no threshold reaches accepted accuracy {target} on field_val; "
                            f"highest accepted accuracy with coverage >= {min_cov}")


def main() -> None:
    a = parse_args()
    T = fit_temperature(*load(a.run_dir, "pv_val"))
    zf, yf = load(a.run_dir, "field_val")
    t, rule = choose_threshold(zf, yf, T, a.target_acc, a.min_coverage)
    out = {
        "run": a.run_dir.name,
        "temperature": round(T, 4),
        "threshold": round(t, 4),
        "threshold_rule": rule,
        "coverage": round(stats(zf, yf, T, t)["coverage"], 4),
        "accepted_accuracy": round(stats(zf, yf, T, t)["accepted_accuracy"], 4),
        "fitted_on": {"temperature": "pv_val logits (NLL)", "threshold": "field_val logits"},
        "report": {s: stats(*load(a.run_dir, s), T, t) for s in ("pv_val", "field_val", "field_test")},
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
