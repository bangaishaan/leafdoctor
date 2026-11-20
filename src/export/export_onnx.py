"""Export the trained classifier to an int8 ONNX model for the browser app.

1. Load <run-dir>/best.pt (timm model) and check its class order equals data/class_map.csv.
2. Export to ONNX (input "input": 1x3x224x224 float, ImageNet-normalised RGB; output "logits": 1x27).
3. Static int8 quantisation (QDQ, per-channel weights) calibrated on --calib-images pv_val images.
4. Compare PyTorch, fp32 ONNX and int8 ONNX top-1 on --test-images pv_test images.
5. Refuse to write the model if it is over --max-mb (15 MB project limit).

Writes <out-dir>/leafdoctor_int8.onnx and export_report.json.
Preprocessing is train.py's evaluation transform: EXIF-rotate, RGB, resize to 224x224 (bilinear),
ImageNet mean/std. The web app reproduces it.

Example:
  python src/export/export_onnx.py --run-dir results/runs/standard_ft25_seed1
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
from train import ImageList, build_transforms  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", type=Path, help="default: the run in --runs-root with the best pv_val macro-F1")
    p.add_argument("--runs-root", type=Path, default=REPO_ROOT / "results" / "runs")
    p.add_argument("--data-root", type=Path, default=REPO_ROOT / "data" / "processed")
    p.add_argument("--class-map", type=Path, default=REPO_ROOT / "data" / "class_map.csv")
    p.add_argument("--split-manifest", type=Path, default=REPO_ROOT / "data" / "split_manifest.csv")
    p.add_argument("--pv-config", default="color")
    p.add_argument("--out-dir", type=Path, default=REPO_ROOT / "web" / "model")
    p.add_argument("--calib-images", type=int, default=200)
    p.add_argument("--test-images", type=int, default=500)
    p.add_argument("--max-mb", type=float, default=15.0)
    p.add_argument("--opset", type=int, default=17)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def best_run(root: Path) -> Path:
    runs = []
    for ck in root.glob("*/best.pt"):
        m = ck.parent / "metrics.json"
        f1 = json.loads(m.read_text())["pv_val"]["macro_f1"] if m.exists() else -1
        runs.append((f1, ck.parent))
    if not runs:
        sys.exit(f"no best.pt under {root}; download the checkpoint from Kaggle first")
    return max(runs)[1]


def sample_split(a, split: str, n: int, class_ids: list[str]) -> list[tuple[Path, int]]:
    index = {c: i for i, c in enumerate(class_ids)}
    with open(a.split_manifest, newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f)
                if r["split"] == split and r["path"].startswith(f"plantvillage/{a.pv_config}/")]
    rng = np.random.default_rng(a.seed)
    pick = rng.choice(len(rows), min(n, len(rows)), replace=False)
    return [(a.data_root / rows[i]["path"], index[rows[i]["class_id"]]) for i in sorted(pick)]


def batches(items, size=64):
    _, tf = build_transforms(SimpleNamespace(image_size=224, crop_scale=(0.5, 1.0)))
    ds = ImageList(items, tf, 224)
    for s in range(0, len(ds), size):
        xs, ys = zip(*(ds[i] for i in range(s, min(s + size, len(ds)))))
        yield torch.stack(xs).numpy(), np.array(ys)


def onnx_logits(sess, x: np.ndarray) -> np.ndarray:
    return np.concatenate([sess.run(None, {"input": x[i:i + 1]})[0] for i in range(len(x))])


def main() -> None:
    a = parse_args()
    import onnx
    import onnxruntime as ort
    import timm
    from onnxruntime.quantization import (CalibrationDataReader, QuantFormat, QuantType, quantize_static)
    from onnxruntime.quantization.shape_inference import quant_pre_process

    run_dir = a.run_dir or best_run(a.runs_root)
    ck = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
    with open(a.class_map, newline="", encoding="utf-8") as f:
        cmap = list(csv.DictReader(f))
    class_ids = [r["class_id"] for r in cmap]
    if list(ck["class_ids"]) != class_ids:
        sys.exit("checkpoint class order differs from class_map.csv")
    model = timm.create_model(ck["model"], pretrained=False, num_classes=len(class_ids))
    model.load_state_dict(ck["state_dict"])
    model.eval()
    print(f"run {run_dir.name}: {ck['model']}, epoch {ck['epoch']}, pv_val macro-F1 {ck['pv_val_macro_f1']:.4f}", flush=True)

    a.out_dir.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix="leafdoctor_export_"))
    fp32, pre, int8 = tmp / "fp32.onnx", tmp / "fp32_pre.onnx", a.out_dir / "leafdoctor_int8.onnx"
    torch.onnx.export(model, torch.zeros(1, 3, 224, 224), fp32, input_names=["input"], output_names=["logits"],
                      opset_version=a.opset, dynamo=False)
    onnx.checker.check_model(onnx.load(fp32))
    quant_pre_process(str(fp32), str(pre))

    calib = sample_split(a, "pv_val", a.calib_images, class_ids)
    calib_x = np.concatenate([x for x, _ in batches(calib)])

    class Reader(CalibrationDataReader):
        def __init__(self):
            self.it = iter(calib_x[i:i + 1] for i in range(len(calib_x)))

        def get_next(self):
            x = next(self.it, None)
            return None if x is None else {"input": x}

    t = time.time()
    quantize_static(str(pre), str(int8), Reader(), quant_format=QuantFormat.QDQ, per_channel=True,
                    activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8)
    print(f"quantised with {len(calib_x)} pv_val images in {time.time() - t:.0f}s", flush=True)
    mb = int8.stat().st_size / 1e6
    if mb > a.max_mb:
        int8.unlink()
        sys.exit(f"int8 model is {mb:.1f} MB > {a.max_mb} MB; not shipping it")

    test = sample_split(a, "pv_test", a.test_images, class_ids)
    s32 = ort.InferenceSession(str(fp32), providers=["CPUExecutionProvider"])
    s8 = ort.InferenceSession(str(int8), providers=["CPUExecutionProvider"])
    pt, o32, o8, ys = [], [], [], []
    with torch.no_grad():
        for x, y in batches(test):
            pt.append(model(torch.from_numpy(x)).numpy())
            o32.append(onnx_logits(s32, x))
            o8.append(onnx_logits(s8, x))
            ys.append(y)
    pt, o32, o8, ys = map(np.concatenate, (pt, o32, o8, ys))
    acc = {k: float((v.argmax(1) == ys).mean()) for k, v in (("pytorch", pt), ("onnx_fp32", o32), ("onnx_int8", o8))}
    report = {
        "run": run_dir.name, "model": ck["model"], "epoch": ck["epoch"], "opset": a.opset,
        "input": {"name": "input", "shape": [1, 3, 224, 224], "mean": list(ck["mean"]), "std": list(ck["std"]),
                  "resize": "direct resize to 224x224, bilinear, after EXIF rotation"},
        "int8_mb": round(mb, 2), "fp32_mb": round(fp32.stat().st_size / 1e6, 2),
        "calibration_images": len(calib_x), "test_images": len(ys), "pv_test_top1": acc,
        "int8_vs_pytorch_top1_agreement": float((o8.argmax(1) == pt.argmax(1)).mean()),
        "fp32_vs_pytorch_max_abs_logit_diff": float(np.abs(o32 - pt).max()),
    }
    (a.out_dir / "export_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
