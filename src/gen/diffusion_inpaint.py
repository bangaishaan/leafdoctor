"""Arm C of the generative experiment: Stable Diffusion inpaints a realistic scene around each leaf.

For every row of data/gen/cutpaste_manifest.csv (arm B), this script:

  1. rebuilds the same leaf placement (rotation, scale, position) on a 512x512 canvas, using
     the leaf mask from data/gen/selection.csv (same masking code as arm B),
  2. inpaints everything except the leaf: the repaint mask is the inverse of the leaf mask,
     dilated by --edge-px (8) so it reaches into the leaf rim and diffusion can blend the edge
     and add contact shadows,
  3. prompts with a random scene + lighting + "photo, realistic, blurred background",
  4. pastes the original leaf pixels back with arm B's feathered mask, so disease symptoms
     are never altered by diffusion.

Model: stable-diffusion-v1-5/stable-diffusion-inpainting (the RunwayML SD 1.5 inpainting
checkpoint; runwayml/stable-diffusion-inpainting redirects to it), pinned to revision
8a4288a76071f7280aedbdb3253bdb9e9d5d84bb, fp16, DPM-Solver++ with 25 steps.
Licence: CreativeML OpenRAIL-M (https://huggingface.co/spaces/CompVis/stable-diffusion-license),
not gated, free to download. The licence allows research use and use of the outputs,
subject to its use-based restrictions (Attachment A). The safety checker is kept on; a flagged
image is retried with a new seed rather than saved black.

Output: <out>/<class_id>/<same name as arm B>.jpg (usable with train.py --extra-train-dir) and
<out>/diffusion_manifest.csv (prompt, seed, model, revision, steps, ...). Existing images are
skipped, so a crashed run resumes where it stopped. Every image's prompt and seed depend only
on --seed and its manifest row, not on GPU count or batch size.

Multi-GPU: --gpus N (default: all visible GPUs) starts N worker processes, one per GPU, each
taking every N-th row. Then the outputs are zipped, the manifest is summarised and a 6x6
preview grid is written.

Examples:
  python src/gen/diffusion_inpaint.py --data-root /kaggle/input/datasets/ishaanbanga/leafdoctor-data \\
      --out /kaggle/working/synthetic/diffusion --preview /kaggle/working/diffusion_preview.png
  python src/gen/diffusion_inpaint.py --limit-per-class 2 ...   # quick test, 54 images
  python src/gen/diffusion_inpaint.py --dry-run ...             # no model: check placement/masks on CPU
"""

from __future__ import annotations

import argparse
import collections
import csv
import os
import subprocess
import sys
import threading
import time
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cutpaste import feathered_alpha, paste, prepare_leaf, write_preview  # noqa: E402
from select_leaves import REPO_ROOT, disk, leaf_mask, load_rgb  # noqa: E402

MODEL_ID = "stable-diffusion-v1-5/stable-diffusion-inpainting"
MODEL_REVISION = "8a4288a76071f7280aedbdb3253bdb9e9d5d84bb"
MODEL_LICENCE = "CreativeML OpenRAIL-M"
SCENES = ("held in a hand", "on bare soil", "among green foliage in a farm field",
          "on a wooden table", "in a greenhouse")
LIGHTING = ("harsh midday sun", "overcast", "golden hour", "dappled shade")
PROMPT_SUFFIX = "photo, realistic, blurred background"
NEGATIVE_PROMPT = "cartoon, painting, text, watermark, multiple leaves in focus"
MANIFEST_NAME = "diffusion_manifest.csv"
LOG_GLOB = "diffusion_log.shard*.csv"
LOG_FIELDS = ["path", "class_id", "color_path", "scene", "lighting", "prompt", "negative_prompt", "seed",
              "attempts", "model", "revision", "scheduler", "steps", "guidance_scale", "edge_px", "seconds"]


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", type=Path, default=REPO_ROOT / "data" / "processed",
                   help="folder holding plantvillage/... (e.g. the Kaggle dataset)")
    p.add_argument("--cutpaste-manifest", type=Path, default=REPO_ROOT / "data" / "gen" / "cutpaste_manifest.csv")
    p.add_argument("--selection", type=Path, default=REPO_ROOT / "data" / "gen" / "selection.csv")
    p.add_argument("--out", type=Path, default=REPO_ROOT / "data" / "synthetic" / "diffusion")
    p.add_argument("--zip", type=Path, help="default: <out>.zip")
    p.add_argument("--preview", type=Path, default=REPO_ROOT / "results" / "figures" / "diffusion_preview.png")
    p.add_argument("--preview-grid", type=int, default=6)
    p.add_argument("--limit-per-class", type=int, default=0, help="only the first N manifest rows per class (0 = all)")
    p.add_argument("--model", default=MODEL_ID)
    p.add_argument("--revision", default=MODEL_REVISION)
    p.add_argument("--steps", type=int, default=25)
    p.add_argument("--guidance-scale", type=float, default=7.5)
    p.add_argument("--edge-px", type=int, default=8, help="repaint mask reaches this far into the leaf")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--max-attempts", type=int, default=3, help="new seeds to try if the safety checker flags an image")
    p.add_argument("--threshold", type=int, default=25, help="leaf mask threshold, as in select_leaves.py")
    p.add_argument("--canvas", type=int, default=512)
    p.add_argument("--jpeg-quality", type=int, default=95)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gpus", type=int, help="worker processes, one per GPU (default: all visible GPUs, at least 1)")
    p.add_argument("--log-every", type=int, default=20, help="print progress every N images per worker")
    p.add_argument("--dry-run", action="store_true",
                   help="no diffusion: save the masked input with the repaint area grey (checks placement on CPU)")
    p.add_argument("--finalize-only", action="store_true", help="skip generation; merge logs, zip, summarise, preview")
    # internal: set by the launcher for each worker process
    p.add_argument("--shard", type=int, default=0, help=argparse.SUPPRESS)
    p.add_argument("--num-shards", type=int, default=1, help=argparse.SUPPRESS)
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    a = p.parse_args(argv)
    a.zip = a.zip or a.out.with_suffix(".zip")
    return a


# ---------------------------------------------------------------- plan

def load_plan(a: argparse.Namespace) -> list[dict]:
    """One job per arm-B manifest row, with a prompt and seed that depend only on --seed and the row."""
    for f in (a.cutpaste_manifest, a.selection):
        if not f.exists():
            sys.exit(f"{f} not found")
    with open(a.selection, newline="", encoding="utf-8") as f:
        seg_of = {r["color_path"]: r["segmented_path"] for r in csv.DictReader(f)}
    with open(a.cutpaste_manifest, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    plan, per_class = [], collections.Counter()
    for i, r in enumerate(rows):
        if a.limit_per_class and per_class[r["class_id"]] >= a.limit_per_class:
            continue
        per_class[r["class_id"]] += 1
        if r["color_path"] not in seg_of:
            sys.exit(f"{r['color_path']} is in {a.cutpaste_manifest.name} but not in {a.selection.name}")
        rng = np.random.default_rng([a.seed, i])
        scene, lighting = SCENES[int(rng.integers(len(SCENES)))], LIGHTING[int(rng.integers(len(LIGHTING)))]
        plan.append({**r, "row": i, "segmented_path": seg_of[r["color_path"]], "scene": scene, "lighting": lighting,
                     "prompt": f"a leaf {scene}, {lighting}, {PROMPT_SUFFIX}",
                     "seed": int(rng.integers(2**31 - 1))})
    return plan


# ---------------------------------------------------------------- leaf placement (same as arm B)

def place_leaf(job: dict, a: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rebuild arm B's leaf on an empty canvas. Returns (leaf RGB float HxWx3, feathered alpha, hard mask)."""
    size = a.canvas
    color = load_rgb(a.data_root / job["color_path"])
    mask = leaf_mask(load_rgb(a.data_root / job["segmented_path"]), a.threshold)
    leaf, hard = prepare_leaf(color, mask, float(job["angle"]), round(float(job["scale"]) * size))
    alpha = feathered_alpha(hard, float(job["feather"]))
    x, y = int(job["x"]), int(job["y"])
    h, w = hard.shape
    if x < 0 or y < 0 or x + w > size or y + h > size:
        raise ValueError(f"{job['path']}: rebuilt leaf ({w}x{h} at {x},{y}) does not fit the canvas; "
                         "the leaf masking code or data differ from the arm B run")
    canvas = np.zeros((size, size, 3), np.float32)
    leaf_full = np.zeros_like(canvas)
    leaf_full[y:y + h, x:x + w] = leaf
    return leaf_full, paste(canvas, alpha, x, y), paste(canvas, hard.astype(np.float32), x, y) > 0.5


def inpaint_inputs(leaf_full: np.ndarray, alpha: np.ndarray, hard: np.ndarray, edge_px: int):
    """Init image (leaf on mid-grey) and repaint mask (white = repaint) for the inpainting pipeline."""
    keep = ndimage.binary_erosion(hard, structure=disk(edge_px))  # = inverse of (inverse mask dilated)
    init = leaf_full * alpha[..., None] + 0.5 * (1 - alpha[..., None])
    init_im = Image.fromarray((init * 255 + 0.5).astype(np.uint8))
    mask_im = Image.fromarray(np.where(keep, 0, 255).astype(np.uint8))
    return init_im, mask_im


def finish(generated: Image.Image, leaf_full: np.ndarray, alpha: np.ndarray) -> Image.Image:
    """Paste the original leaf pixels back with the feathered mask."""
    g = np.asarray(generated.convert("RGB"), np.float32) / 255
    out = g * (1 - alpha[..., None]) + leaf_full * alpha[..., None]
    return Image.fromarray((np.clip(out, 0, 1) * 255 + 0.5).astype(np.uint8))


# ---------------------------------------------------------------- worker

def load_pipeline(a: argparse.Namespace):
    import torch
    from diffusers import DPMSolverMultistepScheduler, StableDiffusionInpaintPipeline

    if not torch.cuda.is_available():
        sys.exit("no CUDA GPU found; use a GPU runtime (or --dry-run to test placement on CPU)")
    pipe = StableDiffusionInpaintPipeline.from_pretrained(a.model, revision=a.revision, torch_dtype=torch.float16,
                                                          variant="fp16")
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to("cuda")
    pipe.set_progress_bar_config(disable=True)
    return pipe


def generate_batch(pipe, jobs: list[dict], inputs: list[tuple], seeds: list[int], a: argparse.Namespace):
    """Returns a list of (PIL image, flagged) for the batch."""
    import torch

    out = pipe(prompt=[j["prompt"] for j in jobs], negative_prompt=[NEGATIVE_PROMPT] * len(jobs),
               image=[i[0] for i in inputs], mask_image=[i[1] for i in inputs],
               height=a.canvas, width=a.canvas, num_inference_steps=a.steps, guidance_scale=a.guidance_scale,
               strength=1.0, generator=[torch.Generator("cpu").manual_seed(s) for s in seeds])
    flags = out.nsfw_content_detected or [False] * len(jobs)
    return list(zip(out.images, flags))


def save_atomic(img: Image.Image, path: Path, quality: int) -> Path:
    tmp = path.with_name(path.name + ".tmp")
    img.save(tmp, format="JPEG", quality=quality)
    return tmp


def run_worker(a: argparse.Namespace, plan: list[dict]) -> None:
    jobs = [j for j in plan if j["row"] % a.num_shards == a.shard and not (a.out / j["path"]).exists()]
    tag = f"[worker {a.shard}/{a.num_shards}]"
    print(f"{tag} {len(jobs)} images to generate "
          f"({sum(1 for j in plan if j['row'] % a.num_shards == a.shard) - len(jobs)} already done)", flush=True)
    if not jobs:
        return
    pipe = None if a.dry_run else load_pipeline(a)
    scheduler = "none (dry run)" if a.dry_run else type(pipe.scheduler).__name__
    log_path = a.out / f"diffusion_log.shard{a.shard}.csv"
    new_log = not log_path.exists()
    log = open(log_path, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(log, fieldnames=LOG_FIELDS)
    if new_log:
        writer.writeheader()

    t0, done, skipped = time.time(), 0, 0
    for b in range(0, len(jobs), a.batch_size):
        batch = jobs[b:b + a.batch_size]
        tb = time.time()
        placed = [place_leaf(j, a) for j in batch]
        inputs = [inpaint_inputs(*p, a.edge_px) for p in placed]
        seeds = [j["seed"] for j in batch]
        attempts = [1] * len(batch)
        if a.dry_run:
            results = [(Image.composite(Image.new("RGB", im.size, (128, 128, 128)), im, m), False) for im, m in inputs]
        else:
            results = generate_batch(pipe, batch, inputs, seeds, a)
            for k, (_, flagged) in enumerate(results):  # retry flagged images one by one with new seeds
                while flagged and attempts[k] < a.max_attempts:
                    seeds[k] += 1_000_003
                    attempts[k] += 1
                    results[k] = generate_batch(pipe, [batch[k]], [inputs[k]], [seeds[k]], a)[0]
                    flagged = results[k][1]
        secs = (time.time() - tb) / len(batch)
        for j, p, (img, flagged), seed, n_att in zip(batch, placed, results, seeds, attempts):
            if flagged:
                skipped += 1
                print(f"{tag} skipped {j['path']}: safety checker flagged all {n_att} attempts", flush=True)
                continue
            dest = a.out / j["path"]
            dest.parent.mkdir(parents=True, exist_ok=True)
            tmp = save_atomic(finish(img, p[0], p[1]), dest, a.jpeg_quality)
            writer.writerow({"path": j["path"], "class_id": j["class_id"], "color_path": j["color_path"],
                             "scene": j["scene"], "lighting": j["lighting"], "prompt": j["prompt"],
                             "negative_prompt": NEGATIVE_PROMPT, "seed": seed, "attempts": n_att,
                             "model": a.model, "revision": a.revision, "scheduler": scheduler, "steps": a.steps,
                             "guidance_scale": a.guidance_scale, "edge_px": a.edge_px, "seconds": round(secs, 2)})
            log.flush()
            os.fsync(log.fileno())
            tmp.replace(dest)  # the image only "exists" (and is skipped on resume) once its log row is written
            done += 1
            if done % a.log_every == 0 or done == len(jobs) - skipped:
                rate = done / (time.time() - t0)
                eta = (len(jobs) - done - skipped) / rate if rate else 0
                print(f"{tag} {done}/{len(jobs)} done, {rate:.2f} img/s, ETA {eta / 60:.1f} min", flush=True)
    log.close()
    print(f"{tag} finished: {done} generated, {skipped} skipped, {(time.time() - t0) / 60:.1f} min", flush=True)


# ---------------------------------------------------------------- launcher and finalisation

def launch_workers(a: argparse.Namespace, n: int, argv: list[str]) -> bool:
    """Run n worker processes, one per GPU, streaming their output. Returns True if all succeeded."""
    procs, threads = [], []
    for k in range(n):
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(k), "PYTHONUNBUFFERED": "1"}
        cmd = [sys.executable, str(Path(__file__).resolve()), *argv, "--worker", "--shard", str(k), "--num-shards", str(n)]
        proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)

        def pump(p=proc):
            for line in p.stdout:
                print(line, end="", flush=True)
        t = threading.Thread(target=pump, daemon=True)
        t.start()
        procs.append(proc)
        threads.append(t)
    codes = [p.wait() for p in procs]
    for t in threads:
        t.join()
    for k, c in enumerate(codes):
        if c:
            print(f"worker {k} exited with code {c}", flush=True)
    return not any(codes)


def finalize(a: argparse.Namespace, plan: list[dict]) -> list[dict]:
    """Merge worker logs into the manifest, zip outputs, print a summary and write the preview."""
    logged = {}
    for log in sorted(a.out.glob(LOG_GLOB)):
        with open(log, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                logged[r["path"]] = r  # last write wins (a resumed image is logged again)
    records = [logged[j["path"]] for j in plan if (a.out / j["path"]).exists() and j["path"] in logged]
    with open(a.out / MANIFEST_NAME, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        w.writeheader()
        w.writerows(records)

    missing = [j["path"] for j in plan if j["path"] not in {r["path"] for r in records}]
    print(f"\n==== {MANIFEST_NAME}: {len(records)}/{len(plan)} images ====", flush=True)
    if records:
        print(f"model: {records[0]['model']} @ {records[0]['revision'][:7]} ({MODEL_LICENCE}), "
              f"scheduler {records[0]['scheduler']}, {records[0]['steps']} steps, cfg {records[0]['guidance_scale']}")
        per_class = collections.Counter(r["class_id"] for r in records)
        print(f"per class: min {min(per_class.values())}, max {max(per_class.values())}, classes {len(per_class)}")
        print("scenes:  ", dict(collections.Counter(r["scene"] for r in records)))
        print("lighting:", dict(collections.Counter(r["lighting"] for r in records)))
        retried = sum(1 for r in records if int(r["attempts"]) > 1)
        secs = [float(r["seconds"]) for r in records]
        print(f"safety-checker retries: {retried}; mean {np.mean(secs):.2f} s/image per GPU")
    if missing:
        print(f"missing {len(missing)} images, e.g. {missing[:3]}; rerun to resume", flush=True)

    if records:
        tmp = a.zip.with_suffix(".zip.partial")
        with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_STORED) as zf:
            zf.write(a.out / MANIFEST_NAME, MANIFEST_NAME, compress_type=zipfile.ZIP_DEFLATED)
            for r in records:
                zf.write(a.out / r["path"], r["path"])
        tmp.replace(a.zip)
        print(f"zipped {len(records)} images to {a.zip} ({a.zip.stat().st_size / 1e6:.0f} MB)")
        write_preview(records, a.out, a.preview, a.preview_grid, a.seed,
                      caption=lambda r: f"{r['scene']}\n{r['lighting']}",
                      title="Arm C: SD 1.5 inpainting around the original leaf")
        print(f"preview: {a.preview}", flush=True)
    return records


def main() -> None:
    argv = sys.argv[1:]
    a = parse_args(argv)
    plan = load_plan(a)
    a.out.mkdir(parents=True, exist_ok=True)
    if a.worker:
        run_worker(a, plan)
        return

    ok = True
    if not a.finalize_only:
        n = a.gpus
        if n is None:
            try:
                import torch
                n = torch.cuda.device_count()
            except ImportError:
                n = 0
        n = max(1, n)
        print(f"{len(plan)} images planned ({len(plan) - sum((a.out / j['path']).exists() for j in plan)} to do) "
              f"on {n} worker(s); model {a.model} @ {a.revision[:7]} ({MODEL_LICENCE})", flush=True)
        ok = launch_workers(a, n, argv) if n > 1 else (run_worker(a, plan) is None)
    records = finalize(a, plan)
    if not ok or len(records) < len(plan):
        sys.exit(1)


if __name__ == "__main__":
    main()
