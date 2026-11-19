"""Download and prepare the LeafDoctor datasets.

Steps:
  1. Download PlantVillage (Hugging Face: mohanty/PlantVillage) and clone
     PlantDoc (github.com/pratikkayal/PlantDoc-Dataset) into the raw dir.
  2. Keep only the classes in data/class_map.csv, renaming folders to class_id.
  3. PlantVillage: start from the dataset's leaf-grouped train/test splits,
     make the split consistent per physical leaf across all configs, then cap
     each class at --pv-cap images (seeded, whole leaves at a time).
  4. PlantDoc: pool its train+test, group near-duplicates (pHash Hamming
     distance <= --phash-threshold), and do a stratified, group-aware split into
     field_val / field_test. PlantDoc is never used for training.
  5. Write images under data/processed/ and counts to data/split_summary.csv.

Example (Colab/Kaggle):
  pip install -r requirements.txt
  python src/prepare_data.py --overwrite
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import shutil
import subprocess
import sys
import zipfile
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path

import imagehash
import numpy as np
from PIL import Image, ImageOps
from sklearn.model_selection import StratifiedGroupKFold

REPO_ROOT = Path(__file__).resolve().parents[1]

PV_REPO_ID = "mohanty/PlantVillage"
PV_REVISION = "9e97599868962bd0079b8db4b7f1efa9185fa1e7"  # pinned for reproducibility
PV_SPLITS = ("train", "test")
PD_GIT_URL = "https://github.com/pratikkayal/PlantDoc-Dataset.git"
PD_ORIG_SPLITS = ("train", "test")
FIELD_SPLITS = ("field_val", "field_test")
INFO_FILE = "prepare_info.json"
PIL_EXT = {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp", "BMP": ".bmp", "GIF": ".gif", "TIFF": ".tif"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data", help="base data directory")
    p.add_argument("--class-map", type=Path, help="default: <data-dir>/class_map.csv")
    p.add_argument("--raw-dir", type=Path, help="downloads go here (default: <data-dir>/raw)")
    p.add_argument("--out-dir", type=Path, help="processed images (default: <data-dir>/processed)")
    p.add_argument("--summary", type=Path, help="default: <data-dir>/split_summary.csv")
    p.add_argument("--pv-configs", nargs="+", default=["color", "segmented"],
                   choices=["color", "segmented", "grayscale"], help="PlantVillage configs to export")
    p.add_argument("--pv-revision", default=PV_REVISION, help="Hugging Face dataset revision")
    p.add_argument("--pv-cap", type=int, default=1500, help="max PlantVillage images per class per config (0 = no cap)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--phash-threshold", type=int, default=6, help="max pHash Hamming distance for near-duplicates")
    p.add_argument("--field-val-frac", type=float, default=0.3, help="fraction of PlantDoc that goes to field_val")
    p.add_argument("--skip-download", action="store_true", help="use files already in --raw-dir")
    p.add_argument("--overwrite", action="store_true", help="replace an existing --out-dir from a previous run")
    a = p.parse_args()
    a.class_map = a.class_map or a.data_dir / "class_map.csv"
    a.raw_dir = a.raw_dir or a.data_dir / "raw"
    a.out_dir = a.out_dir or a.data_dir / "processed"
    a.summary = a.summary or a.data_dir / "split_summary.csv"
    if not 0 < a.field_val_frac < 1:
        p.error("--field-val-frac must be between 0 and 1")
    return a


def log(msg: str) -> None:
    print(msg, flush=True)


# --------------------------------------------------------------------------- class map

def load_class_map(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    for col in ("class_id", "label", "plantvillage_folder", "plantdoc_folder"):
        if rows and col not in rows[0]:
            sys.exit(f"{path} is missing column '{col}'")
    for col in ("class_id", "plantvillage_folder", "plantdoc_folder"):
        dupes = [k for k, n in Counter(r[col] for r in rows).items() if n > 1]
        if dupes:
            sys.exit(f"{path}: duplicate {col}: {dupes}")
    log(f"class map: {len(rows)} classes from {path}")
    return rows


# --------------------------------------------------------------------------- downloads

def download_plantvillage(raw_dir: Path, configs: list[str], revision: str, skip: bool) -> Path:
    dest = raw_dir / "plantvillage"
    files = ["data.zip", "leaf_grouping/leaf-map.json"]
    files += [f"splits/{c}_{s}.txt" for c in configs for s in PV_SPLITS]
    if not skip:
        from huggingface_hub import hf_hub_download

        for name in files:
            log(f"PlantVillage: fetching {name} (data.zip is ~2.2 GB)" if name == "data.zip"
                else f"PlantVillage: fetching {name}")
            hf_hub_download(PV_REPO_ID, name, repo_type="dataset", revision=revision, local_dir=dest)
    missing = [n for n in files if not (dest / n).exists()]
    if missing:
        sys.exit(f"PlantVillage files missing in {dest}: {missing}")
    return dest


def clone_plantdoc(raw_dir: Path, skip: bool) -> tuple[Path, str]:
    # --no-checkout: some PlantDoc file names contain characters such as '?' that
    # Windows cannot create, so files are read from git objects in extract_plantdoc().
    dest = raw_dir / "PlantDoc-Dataset"
    if not (dest / ".git").exists():
        if skip:
            sys.exit(f"PlantDoc not found at {dest}")
        log("PlantDoc: cloning (~1 GB)")
        dest.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--depth", "1", "--no-checkout", PD_GIT_URL, str(dest)], check=True)
    commit = subprocess.run(["git", "-C", str(dest), "rev-parse", "HEAD"],
                            capture_output=True, text=True, check=True).stdout.strip()
    return dest, commit


def extract_plantdoc(repo: Path, classes: list[dict], dest: Path) -> Path:
    """Write the kept classes from git HEAD to dest/<orig_split>/<plantdoc_folder>/ with portable names."""
    if dest.exists():
        shutil.rmtree(dest)
    wanted = {(s, c["plantdoc_folder"]) for s in PD_ORIG_SPLITS for c in classes}
    listing = subprocess.run(["git", "-C", str(repo), "ls-tree", "-r", "-z", "HEAD"],
                             capture_output=True, check=True).stdout
    entries = []
    for rec in filter(None, listing.split(b"\0")):
        meta, raw_path = rec.split(b"\t", 1)
        _, kind, sha = meta.split()
        parts = raw_path.decode("utf-8").split("/")
        if kind == b"blob" and len(parts) == 3 and (parts[0], parts[1]) in wanted:
            entries.append((sha, parts))
    used: set[Path] = set()
    proc = subprocess.Popen(["git", "-C", str(repo), "cat-file", "--batch"],
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    for sha, (split, folder, name) in entries:
        proc.stdin.write(sha + b"\n")
        proc.stdin.flush()
        size = int(proc.stdout.readline().split()[2])
        data = proc.stdout.read(size)
        proc.stdout.read(1)  # trailing newline
        safe = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", name).strip(" .") or "file"
        target, k = dest / split / folder / safe, 1
        while target in used:
            k += 1
            target = dest / split / folder / f"{Path(safe).stem}_{k}{Path(safe).suffix}"
        used.add(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    proc.stdin.close()
    proc.wait()
    log(f"PlantDoc: extracted {len(entries)} files from git into {dest}")
    return dest


# --------------------------------------------------------------------------- PlantVillage

def photo_identifier(filename: str) -> str:
    """Original photo name, shared by all configs of one photo (same rule as the HF loader)."""
    s = filename.replace("_final_masked", "")
    if "___" in s:
        s = s.split("___")[-1]
    s = s.split("copy")[0]
    s = re.sub(r"\.(jpe?g|png)$", "", s, flags=re.IGNORECASE)
    return s.strip()


def leaf_group(pv_class: str, filename: str, leaf_map: dict) -> str:
    """Physical-leaf id from leaf-map.json; falls back to the photo name. Always within one class."""
    ident = photo_identifier(filename)
    for candidate in leaf_map.get(ident.lower(), []):
        if candidate.startswith(pv_class + ":::"):
            return candidate
    return f"{pv_class}:::photo:{ident.lower()}"


def load_plantvillage_records(pv_dir: Path, configs: list[str], classes: list[dict]) -> list[dict]:
    pv_to_id = {c["plantvillage_folder"]: c["class_id"] for c in classes}
    leaf_map = json.loads((pv_dir / "leaf_grouping" / "leaf-map.json").read_text(encoding="utf-8"))
    records = []
    for config in configs:
        for split in PV_SPLITS:
            lines = (pv_dir / "splits" / f"{config}_{split}.txt").read_text(encoding="utf-8").splitlines()
            for member in filter(None, (l.strip() for l in lines)):
                _, _, pv_class, filename = member.split("/", 3)
                if pv_class not in pv_to_id:
                    continue
                records.append({
                    "config": config, "orig_split": split, "pv_class": pv_class,
                    "class_id": pv_to_id[pv_class], "filename": filename, "member": member,
                    "group": leaf_group(pv_class, filename, leaf_map),
                })
    found = {r["pv_class"] for r in records}
    absent = sorted(set(pv_to_id) - found)
    if absent:
        sys.exit(f"PlantVillage folders from class map not found in splits: {absent}")
    return records


def make_splits_leaf_consistent(records: list[dict], ref_config: str) -> None:
    """Give every image of one leaf the same split in every config.

    The published per-config splits put the same leaf (and the colour and
    segmented copies of one photo) on both sides of train/test. Each leaf goes
    to its majority split in the reference config (ties go to test).
    """
    ref_votes: dict[str, Counter] = defaultdict(Counter)
    other_votes: dict[str, Counter] = defaultdict(Counter)
    for r in records:
        (ref_votes if r["config"] == ref_config else other_votes)[r["group"]][r["orig_split"]] += 1
    final = {}
    for g in set(ref_votes) | set(other_votes):
        v = ref_votes.get(g) or other_votes[g]  # leaves missing from the reference config use the others
        final[g] = "test" if v["test"] >= v["train"] else "train"
    moved = Counter()
    for r in records:
        r["split"] = final[r["group"]]
        if r["split"] != r["orig_split"]:
            moved[r["config"]] += 1
    for config, n in sorted(moved.items()):
        log(f"PlantVillage {config}: moved {n} images to their leaf's split to stop train/test leakage")


def select_capped_groups(records: list[dict], cap: int, seed: int) -> set[tuple[str, str]]:
    """Pick whole leaves per class, up to `cap` images in every config, keeping the train/test ratio."""
    sizes: dict[str, dict[tuple[str, str], Counter]] = defaultdict(lambda: defaultdict(Counter))
    for r in records:
        sizes[r["class_id"]][(r["split"], r["group"])][r["config"]] += 1
    selected: set[tuple[str, str]] = set()
    for class_id in sorted(sizes):
        groups = sizes[class_id]
        size = {k: max(c.values()) for k, c in groups.items()}
        total = sum(size.values())
        if cap <= 0 or total <= cap:
            selected.update((class_id, g) for _, g in groups)
            continue
        train_total = sum(n for (s, _), n in size.items() if s == "train")
        quotas = {"train": round(cap * train_total / total)}
        quotas["test"] = cap - quotas["train"]
        for split in PV_SPLITS:
            keys = sorted(k for k in groups if k[0] == split)
            random.Random(f"{seed}:{class_id}:{split}").shuffle(keys)
            used = 0
            for k in keys:
                if used + size[k] <= quotas[split]:
                    selected.add((class_id, k[1]))
                    used += size[k]
    return selected


def export_plantvillage(pv_dir: Path, records: list[dict], out_dir: Path) -> list[dict]:
    manifest = []
    with zipfile.ZipFile(pv_dir / "data.zip") as zf:
        index = {}
        for name in zf.namelist():
            i = name.find("raw/")
            if i >= 0 and not name.endswith("/"):
                index[name[i:]] = name
        missing = 0
        for n, r in enumerate(records, 1):
            src = index.get(r["member"])
            if src is None:
                missing += 1
                continue
            rel = Path("plantvillage") / r["config"] / r["split"] / r["class_id"] / r["filename"]
            dst = out_dir / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(src) as fsrc, open(dst, "wb") as fdst:
                shutil.copyfileobj(fsrc, fdst)
            manifest.append({"dataset": "plantvillage", "config": r["config"], "split": r["split"],
                             "class_id": r["class_id"], "path": rel.as_posix(),
                             "source": r["member"], "group": r["group"]})
            if n % 10000 == 0:
                log(f"PlantVillage: extracted {n}/{len(records)}")
    if missing:
        log(f"WARNING: {missing} PlantVillage files listed in the splits were not in data.zip")
    return manifest


# --------------------------------------------------------------------------- PlantDoc

def collect_plantdoc(pd_dir: Path, classes: list[dict]) -> list[dict]:
    items = []
    for orig_split in PD_ORIG_SPLITS:
        for c in classes:
            folder = pd_dir / orig_split / c["plantdoc_folder"]
            if not folder.is_dir():
                log(f"WARNING: PlantDoc folder missing: {folder}")
                continue
            for path in sorted(folder.iterdir()):
                if path.is_file():
                    items.append({"path": path, "orig_split": orig_split, "class_id": c["class_id"]})
    return items


def hash_images(items: list[dict]) -> tuple[list[dict], np.ndarray]:
    kept, hashes, bad = [], [], []
    for n, it in enumerate(items, 1):
        try:
            with Image.open(it["path"]) as im:
                it["format"] = im.format
                im = ImageOps.exif_transpose(im).convert("RGB")
                hashes.append(imagehash.phash(im).hash.flatten())
            kept.append(it)
        except Exception as e:  # unreadable/corrupt files are skipped and reported
            bad.append(f"{it['path']} ({type(e).__name__})")
        if n % 500 == 0:
            log(f"PlantDoc: hashed {n}/{len(items)}")
    if bad:
        log(f"WARNING: skipped {len(bad)} unreadable PlantDoc files:\n  " + "\n  ".join(bad))
    return kept, np.array(hashes, dtype=bool)


def group_near_duplicates(hashes: np.ndarray, threshold: int, block: int = 2048) -> np.ndarray:
    """Union-find over all pairs with Hamming distance <= threshold. Returns a group id per image."""
    n = len(hashes)
    parent = np.arange(n)

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    bits = hashes.astype(np.float32)
    inv = 1.0 - bits
    for start in range(0, n, block):
        a, ai = bits[start:start + block], inv[start:start + block]
        dist = hashes.shape[1] - (a @ bits.T + ai @ inv.T)
        rows, cols = np.nonzero(dist <= threshold + 0.5)
        for i, j in zip(rows + start, cols):
            if i < j:
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[max(ri, rj)] = min(ri, rj)
    return np.array([find(i) for i in range(n)])


def stratified_group_split(labels: list[str], groups: np.ndarray, val_frac: float, seed: int) -> np.ndarray:
    """True = field_val. Uses k of n StratifiedGroupKFold folds, where k/n ~= val_frac."""
    frac = Fraction(val_frac).limit_denominator(20)
    sgkf = StratifiedGroupKFold(n_splits=frac.denominator, shuffle=True, random_state=seed)
    is_val = np.zeros(len(labels), dtype=bool)
    for fold, (_, idx) in enumerate(sgkf.split(np.zeros(len(labels)), labels, groups)):
        if fold < frac.numerator:
            is_val[idx] = True
    return is_val


def export_plantdoc(items: list[dict], groups: np.ndarray, is_val: np.ndarray, out_dir: Path) -> list[dict]:
    manifest, used = [], set()
    for it, g, val in zip(items, groups, is_val):
        split = FIELD_SPLITS[0] if val else FIELD_SPLITS[1]
        stem = re.sub(r"[^A-Za-z0-9._-]+", "_", it["path"].stem).strip("_") or "img"
        ext = PIL_EXT.get(it["format"], it["path"].suffix.lower() or ".img")
        base = Path("plantdoc") / split / it["class_id"]
        rel, k = base / f"{it['orig_split']}__{stem}{ext}", 1
        while rel in used:
            k += 1
            rel = base / f"{it['orig_split']}__{stem}_{k}{ext}"
        used.add(rel)
        (out_dir / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(it["path"], out_dir / rel)
        manifest.append({"dataset": "plantdoc", "config": "field", "split": split, "class_id": it["class_id"],
                         "path": rel.as_posix(), "source": f"{it['orig_split']}/{it['path'].parent.name}/{it['path'].name}",
                         "group": f"phash:{g}"})
    return manifest


# --------------------------------------------------------------------------- outputs

def prepare_out_dir(out_dir: Path, overwrite: bool) -> None:
    if out_dir.exists() and any(out_dir.iterdir()):
        if not overwrite:
            sys.exit(f"{out_dir} is not empty; pass --overwrite to replace it")
        if not (out_dir / INFO_FILE).exists():
            sys.exit(f"{out_dir} has no {INFO_FILE}, so it was not made by this script; refusing to delete it")
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Marker so a later --overwrite can tell this folder was made by this script, even after a crash.
    (out_dir / INFO_FILE).write_text(json.dumps({"status": "incomplete"}), encoding="utf-8")


def write_summary(path: Path, classes: list[dict], manifest: list[dict], configs: list[str]) -> list[str]:
    columns = [f"pv_{c}_{s}" for c in configs for s in PV_SPLITS] + list(FIELD_SPLITS)
    counts = Counter()
    for m in manifest:
        col = m["split"] if m["dataset"] == "plantdoc" else f"pv_{m['config']}_{m['split']}"
        counts[(m["class_id"], col)] += 1
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["class_id", "label"] + columns)
        for c in classes:
            w.writerow([c["class_id"], c["label"]] + [counts[(c["class_id"], col)] for col in columns])
        w.writerow(["TOTAL", ""] + [sum(counts[(c["class_id"], col)] for c in classes) for col in columns])
    return columns


def main() -> None:
    args = parse_args()
    classes = load_class_map(args.class_map)
    prepare_out_dir(args.out_dir, args.overwrite)

    # 1. downloads
    pv_dir = download_plantvillage(args.raw_dir, args.pv_configs, args.pv_revision, args.skip_download)
    pd_dir, pd_commit = clone_plantdoc(args.raw_dir, args.skip_download)

    # 2-3. PlantVillage
    records = load_plantvillage_records(pv_dir, args.pv_configs, classes)
    make_splits_leaf_consistent(records, ref_config=args.pv_configs[0])
    keep = select_capped_groups(records, args.pv_cap, args.seed)
    records = [r for r in records if (r["class_id"], r["group"]) in keep]
    log(f"PlantVillage: exporting {len(records)} images (cap {args.pv_cap}/class/config)")
    manifest = export_plantvillage(pv_dir, records, args.out_dir)

    # 4. PlantDoc
    pd_files = extract_plantdoc(pd_dir, classes, args.raw_dir / "plantdoc_files")
    items = collect_plantdoc(pd_files, classes)
    log(f"PlantDoc: {len(items)} files in kept classes; hashing")
    items, hashes = hash_images(items)
    groups = group_near_duplicates(hashes, args.phash_threshold)
    sizes = Counter(groups)
    multi = [g for g, n in sizes.items() if n > 1]
    mixed = sum(len({items[i]["class_id"] for i in np.nonzero(groups == g)[0]}) > 1 for g in multi)
    log(f"PlantDoc: {len(sizes)} groups; {len(multi)} near-duplicate groups covering "
        f"{sum(sizes[g] for g in multi)} images; {mixed} groups mix classes")
    is_val = stratified_group_split([it["class_id"] for it in items], groups, args.field_val_frac, args.seed)
    manifest += export_plantdoc(items, groups, is_val, args.out_dir)

    # 5. outputs
    with open(args.out_dir / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["dataset", "config", "split", "class_id", "path", "source", "group"])
        w.writeheader()
        w.writerows(manifest)
    columns = write_summary(args.summary, classes, manifest, args.pv_configs)
    info = {"pv_repo": PV_REPO_ID, "pv_revision": args.pv_revision, "pv_configs": args.pv_configs,
            "pv_cap": args.pv_cap, "plantdoc_repo": PD_GIT_URL, "plantdoc_commit": pd_commit,
            "seed": args.seed, "phash_threshold": args.phash_threshold, "field_val_frac": args.field_val_frac,
            "num_classes": len(classes), "num_images": len(manifest)}
    (args.out_dir / INFO_FILE).write_text(json.dumps(info, indent=2), encoding="utf-8")

    totals = Counter(m["split"] if m["dataset"] == "plantdoc" else f"pv_{m['config']}_{m['split']}" for m in manifest)
    log("done: " + ", ".join(f"{c}={totals[c]}" for c in columns))
    log(f"images: {args.out_dir}\nsummary: {args.summary}")


if __name__ == "__main__":
    main()
