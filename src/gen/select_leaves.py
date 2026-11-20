"""Step 4a: pick the PlantVillage leaves that both generative-experiment arms start from.

From pv_train only (never val, test or PlantDoc), pick --per-class images per class whose
color image has a segmented counterpart, and check that a clean binary leaf mask can be made
from the segmented version:

  1. threshold the black background (max RGB channel > --threshold),
  2. morphological open (removes JPEG speckle) then close (bridges small gaps),
  3. keep the largest connected component and fill interior holes (dark lesions),
  4. reject masks covering < --min-mask-frac or > --max-mask-frac of the image, masks that
     needed a hole larger than --max-hole-frac filled (unreliable segmentation), masks that
     kept part of the blue-grey backdrop (> --max-backdrop-frac bluish pixels), and pairs
     whose color and segmented images are not aligned (so the mask would not fit the color leaf).

The mask is meant for the color image: the segmented copies are colour-adjusted, so arm B
pastes color-image pixels, which match what the classifier is trained on.

Picks are seeded and spread over physical leaves: one image per leaf group first, then a
second image per group, and so on, so a class with few leaves still gets every leaf.

The result, data/gen/selection.csv (class_id, color_path, segmented_path, mask_fraction), is
committed so arm A (diffusion) and arm B (src/gen/cutpaste.py) use exactly the same leaves.
Paths are relative to --data-root. Other scripts import leaf_mask() from here.

Example:
  python src/gen/select_leaves.py
"""

from __future__ import annotations

import argparse
import collections
import csv
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
from prepare_data import photo_identifier  # noqa: E402  (one pairing rule for the whole repo)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", type=Path, default=REPO_ROOT / "data" / "processed")
    p.add_argument("--manifest", type=Path, help="default: <data-root>/manifest.csv (written by prepare_data.py)")
    p.add_argument("--class-map", type=Path, default=REPO_ROOT / "data" / "class_map.csv")
    p.add_argument("--out", type=Path, default=REPO_ROOT / "data" / "gen" / "selection.csv")
    p.add_argument("--per-class", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--threshold", type=int, default=25, help="background if max RGB channel <= this")
    p.add_argument("--min-mask-frac", type=float, default=0.05)
    p.add_argument("--max-mask-frac", type=float, default=0.95)
    p.add_argument("--max-hole-frac", type=float, default=0.01,
                   help="reject if a filled mask hole exceeds this fraction of the image: such holes are either "
                        "leaf tissue the segmentation dropped or real gaps between leaflets, and cannot be told apart")
    p.add_argument("--max-backdrop-frac", type=float, default=0.02,
                   help="reject if more than this share of the mask has the bluish backdrop colour")
    p.add_argument("--align-shift", type=int, default=4,
                   help="reject a pair if shifting the color image by this many px matches the segmented one better")
    a = p.parse_args()
    a.manifest = a.manifest or a.data_root / "manifest.csv"
    return a


def disk(radius: int) -> np.ndarray:
    y, x = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    return x * x + y * y <= radius * radius


def _morph(op, mask: np.ndarray, radius: int) -> np.ndarray:
    # Edge padding so leaves that run off the frame are not eaten away at the border.
    padded = np.pad(mask, radius + 1, mode="edge")
    return op(padded, structure=disk(radius))[radius + 1:-radius - 1, radius + 1:-radius - 1]


def leaf_mask_info(seg_rgb: np.ndarray, threshold: int = 25, open_radius: int = 2,
                   close_radius: int = 3) -> tuple[np.ndarray, float]:
    """Binary leaf mask from a PlantVillage segmented image (H, W, 3 uint8), plus the largest
    filled hole as a fraction of the image (large holes mean unreliable segmentation)."""
    fg = seg_rgb.max(axis=2) > threshold
    fg = _morph(ndimage.binary_opening, fg, open_radius)
    fg = _morph(ndimage.binary_closing, fg, close_radius)
    labels, n = ndimage.label(fg)
    if n == 0:
        return fg, 0.0
    sizes = ndimage.sum_labels(fg, labels, index=np.arange(1, n + 1))
    fg = labels == (int(np.argmax(sizes)) + 1)
    filled = ndimage.binary_fill_holes(fg)
    holes, n_holes = ndimage.label(filled & ~fg)
    largest = ndimage.sum_labels(filled, holes, np.arange(1, n_holes + 1)).max() / fg.size if n_holes else 0.0
    return filled, float(largest)


def leaf_mask(seg_rgb: np.ndarray, threshold: int = 25, open_radius: int = 2, close_radius: int = 3) -> np.ndarray:
    return leaf_mask_info(seg_rgb, threshold, open_radius, close_radius)[0]


def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"))


def pair_key(r: dict) -> tuple[str, str]:
    """Same key for a color image and its segmented copy: class folder + original photo name.

    File names differ between configs (e.g. corn rust color 'RS_Rust 2234.JPG' vs segmented
    '<uuid>___RS_Rust 2234_final_masked.jpg'), so use prepare_data's photo_identifier rule.
    """
    folder = r["source"].split("/")[2]  # raw/<config>/<PlantVillage folder>/<file>
    return folder, photo_identifier(r["original_filename"]).lower()


def unique_by_key(rows: list[dict]) -> dict[tuple[str, str], dict]:
    """Index rows by pair_key, dropping keys shared by several images (ambiguous pairing)."""
    groups = collections.defaultdict(list)
    for r in rows:
        groups[pair_key(r)].append(r)
    return {k: v[0] for k, v in groups.items() if len(v) == 1}


def train_pairs(manifest: Path) -> dict[str, list[tuple[dict, dict]]]:
    with open(manifest, newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["dataset"] == "plantvillage" and r["split"] == "train"]
    color = unique_by_key([r for r in rows if r["config"] == "color"])
    seg = unique_by_key([r for r in rows if r["config"] == "segmented"])
    by_class = collections.defaultdict(list)
    for k in sorted(color.keys() & seg.keys()):
        by_class[color[k]["class_id"]].append((color[k], seg[k]))
    return by_class


def candidate_order(pairs: list[tuple[dict, dict]], rng: np.random.Generator) -> list[tuple[dict, dict]]:
    """Shuffle, then interleave leaf groups so picks cover as many physical leaves as possible."""
    groups = collections.defaultdict(list)
    for pair in pairs:
        groups[pair[0]["group"]].append(pair)
    names = sorted(groups)
    rng.shuffle(names)
    queues = []
    for g in names:
        q = groups[g]
        queues.append([q[i] for i in rng.permutation(len(q))])
    order = []
    for depth in range(max(len(q) for q in queues)):
        order += [q[depth] for q in queues if depth < len(q)]
    return order


def backdrop_fraction(color_rgb: np.ndarray, mask: np.ndarray) -> float:
    """Share of the mask (eroded 2 px) that looks like PlantVillage's blue-grey backdrop.

    Where the segmentation kept part of the backdrop attached to the leaf (common in
    tomato_leaf_mold), those pixels are bluish (B clearly above G, B >= R), which leaf
    tissue, whether green, yellow, brown or white mildew, almost never is.
    """
    c = color_rgb.astype(np.int16)
    r, g, b = c[..., 0], c[..., 1], c[..., 2]
    inner = ndimage.binary_erosion(mask, structure=disk(2))
    return float(((b - g > 8) & (b >= r))[inner].mean()) if inner.any() else 1.0


def check_pair(color_rgb: np.ndarray, seg_rgb: np.ndarray, a: argparse.Namespace) -> tuple[str | None, float]:
    if color_rgb.shape != seg_rgb.shape:
        return "size mismatch", 0.0
    mask, hole = leaf_mask_info(seg_rgb, a.threshold)
    frac = float(mask.mean())
    if not a.min_mask_frac <= frac <= a.max_mask_frac:
        return "mask size", frac
    if hole > a.max_hole_frac:
        return "large hole in segmentation", frac
    if backdrop_fraction(color_rgb, mask) > a.max_backdrop_frac:
        return "backdrop inside mask", frac
    # The segmented copies are colour/brightness-adjusted, so compare structure, not values:
    # an aligned pair correlates best with no shift between the two images.
    inner = ndimage.binary_erosion(mask, structure=disk(a.align_shift + 2))
    if inner.sum() < 100:
        return "mask size", frac
    c, s = color_rgb.mean(axis=2), seg_rgb.mean(axis=2)[inner]
    k = a.align_shift
    score = {(dy, dx): np.corrcoef(np.roll(c, (dy, dx), (0, 1))[inner], s)[0, 1]
             for dy in (-k, 0, k) for dx in (-k, 0, k)}
    if max(score, key=score.get) != (0, 0):
        return "color/segmented misaligned", frac
    return None, frac


def main() -> None:
    a = parse_args()
    for f in (a.manifest, a.class_map):
        if not f.exists():
            sys.exit(f"{f} not found; run src/prepare_data.py first")
    with open(a.class_map, newline="", encoding="utf-8") as f:
        class_ids = [r["class_id"] for r in csv.DictReader(f)]

    by_class = train_pairs(a.manifest)
    out_rows, rejected = [], collections.Counter()
    print(f"{'class_id':28s} {'pairs':>5s} {'picked':>6s} {'leaves':>6s} {'rejected':>8s}", flush=True)
    for ci, cid in enumerate(class_ids):
        rng = np.random.default_rng([a.seed, ci])
        picked, n_rej = [], 0
        if not by_class.get(cid):
            sys.exit(f"{cid}: no pv_train image has both a color and a segmented version")
        for color_r, seg_r in candidate_order(by_class[cid], rng):
            reason, frac = check_pair(load_rgb(a.data_root / color_r["path"]), load_rgb(a.data_root / seg_r["path"]), a)
            if reason:
                rejected[reason] += 1
                n_rej += 1
                continue
            picked.append((color_r, seg_r, frac))
            if len(picked) == a.per_class:
                break
        if len(picked) < a.per_class:
            sys.exit(f"{cid}: only {len(picked)} usable pv_train pairs, need {a.per_class}")
        leaves = len({c["group"] for c, _, _ in picked})
        print(f"{cid:28s} {len(by_class[cid]):5d} {len(picked):6d} {leaves:6d} {n_rej:8d}", flush=True)
        out_rows += [{"class_id": cid, "color_path": c["path"], "segmented_path": s["path"],
                      "mask_fraction": f"{frac:.4f}"} for c, s, frac in picked]

    a.out.parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["class_id", "color_path", "segmented_path", "mask_fraction"])
        w.writeheader()
        w.writerows(out_rows)
    print(f"rejected: {dict(rejected) or 'none'}")
    print(f"wrote {a.out} ({len(out_rows)} leaves, {len(class_ids)} classes)")


if __name__ == "__main__":
    main()
