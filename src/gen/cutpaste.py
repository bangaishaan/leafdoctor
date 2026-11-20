"""Arm B of the generative experiment: cut-and-paste leaves onto procedural backgrounds (no generative AI).

For every leaf in data/gen/selection.csv (written by select_leaves.py, shared with arm A):

  1. cut the leaf out of its *color* image with the mask made from its segmented copy,
  2. rotate it by a random angle and scale it so its longest side is 40-80% of a 512x512 canvas,
  3. place it at a random position on a random background, chosen uniformly from
       solid colour | linear/radial colour gradient | fractal Perlin noise | random crop of a tiled texture
     (all generated in code, so there are no licence issues),
  4. add a soft drop shadow, feather the mask edge by 2-3 px, then apply random brightness/contrast.

Output: <out>/<class_id>/<color stem>_<k>.jpg, ready for `train.py --arm cutpaste --extra-train-dir <out>`,
plus <out>/cutpaste_manifest.csv with every random parameter, and a 6x6 preview grid PNG.
Every image has its own seeded RNG, so results do not depend on --workers.

Example:
  python src/gen/cutpaste.py
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent))
from select_leaves import REPO_ROOT, disk, leaf_mask, load_rgb  # noqa: E402

BACKGROUNDS = ("solid", "gradient", "perlin", "texture")
TEXTURES = ("checker", "stripes", "dots", "bricks", "grid", "weave")
MANIFEST_NAME = "cutpaste_manifest.csv"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", type=Path, default=REPO_ROOT / "data" / "processed")
    p.add_argument("--selection", type=Path, default=REPO_ROOT / "data" / "gen" / "selection.csv")
    p.add_argument("--out", type=Path, default=REPO_ROOT / "data" / "synthetic" / "cutpaste")
    p.add_argument("--preview", type=Path, default=REPO_ROOT / "results" / "figures" / "cutpaste_preview.png")
    p.add_argument("--preview-grid", type=int, default=6, help="preview is N x N images")
    p.add_argument("--per-leaf", type=int, default=1, help="images generated per selected leaf")
    p.add_argument("--canvas", type=int, default=512)
    p.add_argument("--scale", type=float, nargs=2, default=(0.4, 0.8), help="leaf longest side / canvas side")
    p.add_argument("--feather", type=float, nargs=2, default=(2.0, 3.0), help="mask edge feather in px")
    p.add_argument("--brightness", type=float, nargs=2, default=(0.8, 1.2))
    p.add_argument("--contrast", type=float, nargs=2, default=(0.75, 1.25))
    p.add_argument("--threshold", type=int, default=25, help="mask threshold, as in select_leaves.py")
    p.add_argument("--jpeg-quality", type=int, default=95)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    p.add_argument("--overwrite", action="store_true", help="replace an existing cut-paste output folder")
    return p.parse_args()


# ---------------------------------------------------------------- backgrounds (float32 HxWx3 in [0, 1])

def random_colour(rng: np.random.Generator) -> np.ndarray:
    h, s, v = rng.random(), rng.uniform(0.0, 1.0), rng.uniform(0.15, 1.0)
    i, f = int(h * 6) % 6, h * 6 - int(h * 6)
    p, q, t = v * (1 - s), v * (1 - f * s), v * (1 - (1 - f) * s)
    return np.array([(v, t, p), (q, v, p), (p, v, t), (p, q, v), (t, p, v), (v, p, q)][i], np.float32)


def colour_ramp(t: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Map t in [0, 1] (HxW) through 2-3 random colour stops."""
    n = int(rng.integers(2, 4))
    stops = np.linspace(0, 1, n)
    colours = np.stack([random_colour(rng) for _ in range(n)])
    return np.stack([np.interp(t, stops, colours[:, c]) for c in range(3)], axis=-1).astype(np.float32)


def bg_solid(size: int, rng: np.random.Generator) -> np.ndarray:
    return np.broadcast_to(random_colour(rng), (size, size, 3)).copy()


def bg_gradient(size: int, rng: np.random.Generator) -> np.ndarray:
    y, x = np.mgrid[0:size, 0:size].astype(np.float32) / (size - 1)
    if rng.random() < 0.7:  # linear, random direction
        a = rng.uniform(0, 2 * np.pi)
        t = x * np.cos(a) + y * np.sin(a)
    else:                   # radial, random centre
        cx, cy = rng.uniform(0, 1, 2)
        t = np.hypot(x - cx, y - cy)
    t = (t - t.min()) / (np.ptp(t) + 1e-6)
    return colour_ramp(t, rng)


def perlin(size: int, res: int, rng: np.random.Generator) -> np.ndarray:
    """2-D gradient (Perlin) noise on a size x size grid with res x res cells; size % res == 0."""
    d = size // res
    coords = np.arange(size, dtype=np.float32) * res / size
    gy, gx = np.meshgrid(coords % 1, coords % 1, indexing="ij")
    angles = rng.uniform(0, 2 * np.pi, (res + 1, res + 1))
    grads = np.stack([np.cos(angles), np.sin(angles)], -1).repeat(d, 0).repeat(d, 1)
    g00, g10 = grads[:-d, :-d], grads[d:, :-d]
    g01, g11 = grads[:-d, d:], grads[d:, d:]
    n00 = gy * g00[..., 0] + gx * g00[..., 1]
    n10 = (gy - 1) * g10[..., 0] + gx * g10[..., 1]
    n01 = gy * g01[..., 0] + (gx - 1) * g01[..., 1]
    n11 = (gy - 1) * g11[..., 0] + (gx - 1) * g11[..., 1]
    ty, tx = (6 * g**5 - 15 * g**4 + 10 * g**3 for g in (gy, gx))
    return (1 - tx) * ((1 - ty) * n00 + ty * n10) + tx * ((1 - ty) * n01 + ty * n11)


def bg_perlin(size: int, rng: np.random.Generator) -> np.ndarray:
    res = int(rng.choice([2, 4, 8]))
    octaves = int(rng.integers(3, 6))
    persistence = rng.uniform(0.35, 0.65)
    t = np.zeros((size, size), np.float32)
    for o in range(octaves):
        r = res * 2**o
        if size % r or r > size // 4:
            break
        t += persistence**o * perlin(size, r, rng)
    t = (t - t.min()) / (np.ptp(t) + 1e-6)
    return colour_ramp(t, rng)


def texture_tile(kind: str, n: int, rng: np.random.Generator) -> np.ndarray:
    """One n x n tile (float HxW in [0, 1]) that repeats seamlessly."""
    y, x = np.mgrid[0:n, 0:n].astype(np.float32) / n
    if kind == "checker":
        return ((x < 0.5) ^ (y < 0.5)).astype(np.float32)
    if kind == "stripes":
        return (y < rng.uniform(0.2, 0.8)).astype(np.float32)
    if kind == "dots":
        return (np.hypot(x - 0.5, y - 0.5) < rng.uniform(0.15, 0.45)).astype(np.float32)
    if kind == "bricks":
        mortar = rng.uniform(0.04, 0.12)
        row = (y >= 0.5).astype(np.float32)
        xs = (x + 0.5 * row) % 1
        return ((np.minimum(y % 0.5, 0.5 - y % 0.5) > mortar / 2) & (np.minimum(xs, 1 - xs) > mortar / 2)).astype(np.float32)
    if kind == "grid":
        w = rng.uniform(0.04, 0.2)
        return ((x < w) | (y < w)).astype(np.float32)
    # weave: smooth periodic over/under pattern
    k = int(rng.integers(1, 4))
    return (0.5 + 0.25 * np.sin(2 * np.pi * k * x) * np.cos(2 * np.pi * k * y)
            + 0.25 * np.sin(2 * np.pi * (x + y) * k)).astype(np.float32)


def bg_texture(size: int, rng: np.random.Generator) -> np.ndarray:
    kind = TEXTURES[int(rng.integers(len(TEXTURES)))]
    n = int(rng.integers(16, 97))
    big = int(math.ceil(size * 1.5))  # room for rotation: inscribed square of a 1.5x tile sheet is > size
    sheet = np.tile(texture_tile(kind, n, rng), (big // n + 1, big // n + 1))[:big, :big]
    img = Image.fromarray((sheet * 255).astype(np.uint8)).rotate(rng.uniform(0, 360), resample=Image.BILINEAR)
    half = (big - size) // 2
    jitter = int(big / math.sqrt(2) - size) // 2 - 1  # random crop, but always inside the rotated area
    ox, oy = (half + rng.integers(-jitter, jitter + 1, 2)) if jitter > 0 else (half, half)
    t = np.asarray(img, np.float32)[oy:oy + size, ox:ox + size] / 255
    t = ndimage.gaussian_filter(t, rng.uniform(0.5, 1.5))  # soften aliasing
    out = colour_ramp(t, rng) + rng.normal(0, 0.015, (size, size, 1)).astype(np.float32)  # mild grain
    return out, kind


def make_background(size: int, rng: np.random.Generator) -> tuple[np.ndarray, str]:
    kind = BACKGROUNDS[int(rng.integers(len(BACKGROUNDS)))]
    if kind == "texture":
        img, tex = bg_texture(size, rng)
        return img, f"texture:{tex}"
    return {"solid": bg_solid, "gradient": bg_gradient, "perlin": bg_perlin}[kind](size, rng), kind


# ---------------------------------------------------------------- leaf composite

def prepare_leaf(color: np.ndarray, mask: np.ndarray, angle: float, target: int) -> tuple[np.ndarray, np.ndarray]:
    """Rotate and scale the leaf so its longest side is `target` px. Returns (RGB float HxWx3, hard mask)."""
    # Replace background pixels with the nearest leaf colour, so later resampling/feathering
    # cannot pull the original PlantVillage backdrop into the edge as a halo.
    idx = ndimage.distance_transform_edt(~mask, return_distances=False, return_indices=True)
    filled = color[idx[0], idx[1]]
    ys, xs = np.nonzero(mask)
    pad = 4
    y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad + 1, mask.shape[0])
    x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad + 1, mask.shape[1])
    rgb_im = Image.fromarray(filled[y0:y1, x0:x1])
    m_im = Image.fromarray(mask[y0:y1, x0:x1].astype(np.uint8) * 255)
    fill = tuple(int(c) for c in color[mask].mean(0))
    rgb_im = rgb_im.rotate(angle, resample=Image.BICUBIC, expand=True, fillcolor=fill)
    m_im = m_im.rotate(angle, resample=Image.BILINEAR, expand=True)
    box = Image.fromarray((np.asarray(m_im) > 127).astype(np.uint8)).getbbox()
    rgb_im, m_im = rgb_im.crop(box), m_im.crop(box)
    f = target / max(rgb_im.size)
    new = (max(1, round(rgb_im.width * f)), max(1, round(rgb_im.height * f)))
    rgb = np.asarray(rgb_im.resize(new, Image.LANCZOS), np.float32) / 255
    hard = np.asarray(m_im.resize(new, Image.BILINEAR)) > 127
    return rgb, hard


def feathered_alpha(hard: np.ndarray, feather: float) -> np.ndarray:
    """Soft alpha whose ramp is ~`feather` px wide and stays inside the leaf outline."""
    sigma = feather / 2
    inner = ndimage.binary_erosion(np.pad(hard, 1), structure=disk(max(1, round(sigma))))[1:-1, 1:-1]
    return ndimage.gaussian_filter(inner.astype(np.float32), sigma)


def paste(canvas: np.ndarray, patch: np.ndarray, x: int, y: int) -> np.ndarray:
    """Place a 2-D patch into a zero canvas of `canvas` shape at (x, y), clipping at the borders."""
    out = np.zeros(canvas.shape[:2], np.float32)
    h, w = patch.shape
    H, W = out.shape
    xa, ya, xb, yb = max(x, 0), max(y, 0), min(x + w, W), min(y + h, H)
    if xa < xb and ya < yb:
        out[ya:yb, xa:xb] = patch[ya - y:yb - y, xa - x:xb - x]
    return out


def compose(task: tuple) -> dict:
    i, k, row, cfg = task
    rng = np.random.default_rng([cfg["seed"], i, k])
    size = cfg["canvas"]
    color = load_rgb(cfg["data_root"] / row["color_path"])
    mask = leaf_mask(load_rgb(cfg["data_root"] / row["segmented_path"]), cfg["threshold"])

    bg, bg_kind = make_background(size, rng)
    scale = rng.uniform(*cfg["scale"])
    angle = rng.uniform(0, 360)
    leaf, hard = prepare_leaf(color, mask, angle, round(scale * size))
    feather = rng.uniform(*cfg["feather"])
    alpha = feathered_alpha(hard, feather)
    h, w = alpha.shape
    x, y = int(rng.integers(0, size - w + 1)), int(rng.integers(0, size - h + 1))

    # Soft drop shadow: offset, blurred copy of the alpha that darkens the background.
    dist, direction = rng.uniform(4, 16), rng.uniform(0, 2 * np.pi)
    dx, dy = round(dist * np.cos(direction)), round(dist * np.sin(direction))
    blur, opacity = rng.uniform(3, 9), rng.uniform(0.25, 0.6)
    shadow = ndimage.gaussian_filter(paste(bg, alpha, x + dx, y + dy), blur)
    img = bg * (1 - opacity * shadow[..., None])

    a = paste(bg, alpha, x, y)[..., None]
    leaf_full = np.zeros_like(img)
    leaf_full[y:y + h, x:x + w] = leaf
    img = img * (1 - a) + leaf_full * a

    contrast, brightness = rng.uniform(*cfg["contrast"]), rng.uniform(*cfg["brightness"])
    mean = img.mean()
    img = np.clip(((img - mean) * contrast + mean) * brightness, 0, 1)

    rel = Path(row["class_id"]) / f"{Path(row['color_path']).stem}_{k}.jpg"
    out = cfg["out"] / rel
    out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((img * 255 + 0.5).astype(np.uint8)).save(out, quality=cfg["jpeg_quality"])
    return {"path": rel.as_posix(), "class_id": row["class_id"], "color_path": row["color_path"], "k": k,
            "background": bg_kind, "scale": round(scale, 4), "angle": round(angle, 2), "x": x, "y": y,
            "feather": round(feather, 3), "shadow_dx": dx, "shadow_dy": dy, "shadow_blur": round(blur, 2),
            "shadow_opacity": round(opacity, 3), "contrast": round(contrast, 4), "brightness": round(brightness, 4)}


# ---------------------------------------------------------------- main

def prepare_out_dir(out: Path, overwrite: bool) -> None:
    if out.exists() and any(out.iterdir()):
        if not overwrite:
            sys.exit(f"{out} is not empty; pass --overwrite to replace it")
        if not (out / MANIFEST_NAME).exists():
            sys.exit(f"{out} has no {MANIFEST_NAME}, so it does not look like cut-paste output; not deleting it")
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)


def write_preview(records: list[dict], out_dir: Path, path: Path, n: int, seed: int) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(seed)
    by_class: dict[str, list[dict]] = {}
    for r in records:
        by_class.setdefault(r["class_id"], []).append(r)
    classes = list(by_class)
    # One image per class first (as many classes as fit), then random extras.
    picks = [by_class[c][int(rng.integers(len(by_class[c])))] for c in classes[:n * n]]
    rest = [r for r in records if r not in picks]
    picks += [rest[j] for j in rng.choice(len(rest), min(n * n - len(picks), len(rest)), replace=False)]
    order = {c: i for i, c in enumerate(classes)}
    picks.sort(key=lambda r: (order[r["class_id"]], r["path"]))

    fig, axes = plt.subplots(n, n, figsize=(2.2 * n, 2.35 * n))
    for ax in axes.ravel():
        ax.axis("off")
    for ax, r in zip(axes.ravel(), picks):
        with Image.open(out_dir / r["path"]) as im:
            ax.imshow(im.convert("RGB").resize((256, 256), Image.LANCZOS))
        ax.set_title(f"{r['class_id']}\n{r['background']}", fontsize=7)
    fig.suptitle("Arm B: cut-and-paste on procedural backgrounds", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=100)
    plt.close(fig)


def main() -> None:
    a = parse_args()
    if not a.selection.exists():
        sys.exit(f"{a.selection} not found; run src/gen/select_leaves.py first")
    with open(a.selection, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    missing = [p for r in rows for p in (r["color_path"], r["segmented_path"]) if not (a.data_root / p).is_file()]
    if missing:
        sys.exit(f"{len(missing)} selected images missing under {a.data_root}, e.g. {missing[:2]}")
    if not 0 < a.scale[0] <= a.scale[1] < 1:
        sys.exit("--scale must satisfy 0 < min <= max < 1")

    prepare_out_dir(a.out, a.overwrite)
    cfg = {"seed": a.seed, "canvas": a.canvas, "scale": a.scale, "feather": a.feather, "threshold": a.threshold,
           "contrast": a.contrast, "brightness": a.brightness, "jpeg_quality": a.jpeg_quality,
           "data_root": a.data_root, "out": a.out}
    tasks = [(i, k, r, cfg) for i, r in enumerate(rows) for k in range(a.per_leaf)]
    print(f"generating {len(tasks)} images from {len(rows)} leaves with {a.workers} workers", flush=True)

    records = []
    if a.workers > 1:
        with ProcessPoolExecutor(a.workers) as ex:
            for j, rec in enumerate(ex.map(compose, tasks, chunksize=16), 1):
                records.append(rec)
                if j % 500 == 0:
                    print(f"  {j}/{len(tasks)}", flush=True)
    else:
        for j, t in enumerate(tasks, 1):
            records.append(compose(t))
            if j % 500 == 0:
                print(f"  {j}/{len(tasks)}", flush=True)

    with open(a.out / MANIFEST_NAME, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(records[0]))
        w.writeheader()
        w.writerows(records)
    write_preview(records, a.out, a.preview, a.preview_grid, a.seed)

    counts = {}
    for r in records:
        counts[r["background"].split(":")[0]] = counts.get(r["background"].split(":")[0], 0) + 1
    print(f"backgrounds: {counts}")
    print(f"wrote {len(records)} images to {a.out}, manifest {a.out / MANIFEST_NAME}, preview {a.preview}")


if __name__ == "__main__":
    main()
