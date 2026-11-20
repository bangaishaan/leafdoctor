"""One-shot generative experiment: make arms B and C, train arms A, B, C, print the comparison.

  (b) arm B: src/gen/cutpaste.py on the first --per-class leaves per class of data/gen/selection.csv
  (c) arm C: src/gen/diffusion_inpaint.py on exactly the same leaves and placements (arm B's manifest),
      on all GPUs
  (d) train arms A (standard), B (+cut-paste), C (+diffusion) with train.py and an identical recipe
      (--epochs, --lp-epochs, --seed)
  (e) print a table of pv_test / field_test accuracy and macro-F1 per arm, and each metrics.json

Scheduling for a T4 x2 Kaggle session: arm A needs no synthetic data, so it starts at once on
the last GPU while B and C are generated; B and C then train in parallel, one per GPU.

Example (Kaggle):
  python src/gen/run_experiment.py --data-root /kaggle/input/datasets/ishaanbanga/leafdoctor-data \\
      --work /kaggle/working
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
ARMS = {"A": ("standard", None), "B": ("cutpaste", "cutpaste"), "C": ("diffusion", "diffusion")}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--work", type=Path, required=True, help="writable output folder, e.g. /kaggle/working")
    p.add_argument("--selection", type=Path, default=REPO_ROOT / "data" / "gen" / "selection.csv")
    p.add_argument("--per-class", type=int, default=15)
    p.add_argument("--steps", type=int, default=20, help="diffusion steps")
    p.add_argument("--diffusion-batch", type=int, default=4)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lp-epochs", type=int, default=1)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--train-workers", type=int, default=3, help="data-loader workers per training run")
    p.add_argument("--max-per-class", type=int, default=0,
                   help="cap PlantVillage train images per class for all arms (0 = no cap; use if time is short)")
    p.add_argument("--gpus", type=int, help="default: all visible GPUs")
    return p.parse_args()


def start(tag: str, cmd: list[str], gpu: int | None = None) -> subprocess.Popen:
    """Start a subprocess and stream its output with a [tag] prefix."""
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    print(f"[{tag}] $ {' '.join(cmd)}" + (f"  (GPU {gpu})" if gpu is not None else ""), flush=True)
    proc = subprocess.Popen(cmd, env=env, cwd=REPO_ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1)

    def pump():
        for line in proc.stdout:
            print(f"[{tag}] {line}", end="", flush=True)
    proc.pump = threading.Thread(target=pump, daemon=True)
    proc.pump.start()
    return proc


def wait(tag: str, proc: subprocess.Popen) -> bool:
    code = proc.wait()
    proc.pump.join()
    print(f"[{tag}] exited with code {code}", flush=True)
    return code == 0


def train_cmd(a, arm: str, extra: Path | None) -> list[str]:
    cmd = [sys.executable, "src/train.py", "--data-root", str(a.data_root), "--out", str(a.work / "results"),
           "--arm", arm, "--seed", str(a.seed), "--epochs", str(a.epochs), "--lp-epochs", str(a.lp_epochs),
           "--workers", str(a.train_workers)]
    if a.max_per_class:
        cmd += ["--max-per-class", str(a.max_per_class)]
    if extra is not None:
        cmd += ["--extra-train-dir", str(extra)]
    return cmd


def main() -> None:
    a = parse_args()
    t0 = time.time()
    n_gpus = a.gpus
    if n_gpus is None:
        import torch
        n_gpus = torch.cuda.device_count()
    if n_gpus < 1:
        sys.exit("no GPU found")
    synth, previews = a.work / "synthetic", a.work / "previews"
    previews.mkdir(parents=True, exist_ok=True)
    (a.work / "results").mkdir(parents=True, exist_ok=True)
    extra = {"cutpaste": synth / "cutpaste", "diffusion": synth / "diffusion"}
    ok = {}

    # Arm A trains on the last GPU while the synthetic data is made.
    procs = {"A": start("A", train_cmd(a, "standard", None), gpu=n_gpus - 1)}

    # (b) arm B
    ok["gen_B"] = wait("gen B", start("gen B", [
        sys.executable, "src/gen/cutpaste.py", "--data-root", str(a.data_root), "--selection", str(a.selection),
        "--limit-per-class", str(a.per_class), "--out", str(extra["cutpaste"]),
        "--preview", str(previews / "cutpaste_preview.png"), "--overwrite", "--workers", "2"]))

    # (c) arm C, same leaves and placements as arm B
    if ok["gen_B"]:
        ok["gen_C"] = wait("gen C", start("gen C", [
            sys.executable, "src/gen/diffusion_inpaint.py", "--data-root", str(a.data_root),
            "--cutpaste-manifest", str(extra["cutpaste"] / "cutpaste_manifest.csv"), "--selection", str(a.selection),
            "--out", str(extra["diffusion"]), "--preview", str(previews / "diffusion_preview.png"),
            "--steps", str(a.steps), "--batch-size", str(a.diffusion_batch), "--gpus", str(n_gpus)]))
    print(f"generation finished after {(time.time() - t0) / 60:.1f} min", flush=True)

    # (d) arms B and C in parallel, one per GPU
    if ok.get("gen_B"):
        procs["B"] = start("B", train_cmd(a, "cutpaste", extra["cutpaste"]), gpu=0)
    # A few safety-checker skips make diffusion_inpaint exit 1; arm C still trains on what was made.
    if (extra["diffusion"] / "diffusion_manifest.csv").exists():
        procs["C"] = start("C", train_cmd(a, "diffusion", extra["diffusion"]), gpu=min(1, n_gpus - 1))
    for tag, proc in procs.items():
        ok[tag] = wait(tag, proc)

    # (e) results
    rows, metrics = [], {}
    for tag, (arm, _) in ARMS.items():
        f = a.work / "results" / f"{arm}_seed{a.seed}" / "metrics.json"
        if f.exists():
            m = json.loads(f.read_text())
            metrics[tag] = m
            s = m["splits"]
            rows.append((tag, arm, m["train_images"].get("extra", 0), s["pv_test"]["accuracy"],
                         s["pv_test"]["macro_f1"], s["field_test"]["accuracy"], s["field_test"]["macro_f1"]))
        else:
            rows.append((tag, arm, None, None, None, None, None))
    for tag, m in metrics.items():
        print(f"\n===== metrics.json arm {tag} ({m['arm']}) =====")
        print(json.dumps(m, indent=1), flush=True)

    fmt = lambda v: "   -  " if v is None else f"{v:.4f}"
    print("\n===== Generative experiment: final results =====")
    print(f"recipe: --epochs {a.epochs} --lp-epochs {a.lp_epochs} --seed {a.seed}"
          + (f" --max-per-class {a.max_per_class}" if a.max_per_class else "")
          + f"; {a.per_class} synthetic leaves per class for B and C")
    print(f"{'arm':4s} {'name':10s} {'extra':>6s} {'pv_test acc':>12s} {'pv_test F1':>11s} {'field acc':>10s} {'field F1':>9s}")
    for tag, arm, ex, pa, pf, fa, ff in rows:
        print(f"{tag:4s} {arm:10s} {ex if ex is not None else '-':>6} {fmt(pa):>12s} {fmt(pf):>11s} {fmt(fa):>10s} {fmt(ff):>9s}")
    summary = {"recipe": vars(a) | {"data_root": str(a.data_root), "work": str(a.work), "selection": str(a.selection)},
               "results": {r[0]: {"arm": r[1], "extra_images": r[2], "pv_test_accuracy": r[3], "pv_test_macro_f1": r[4],
                                  "field_test_accuracy": r[5], "field_test_macro_f1": r[6]} for r in rows},
               "steps_ok": ok, "minutes": round((time.time() - t0) / 60, 1)}
    (a.work / "results" / "experiment_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    print(f"total {(time.time() - t0) / 60:.1f} min", flush=True)
    if not all(ok.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
