"""Fine-tune a YOLOv8 person detector on OmniSearch's own render distribution.

Stock COCO YOLO is trained on eye-level photos and is unreliable on small,
top-down aerial survivors (confidence ~0.2-0.7, often missed). Because we
*control* the renderer, we can generate labelled training data that matches
exactly what the drone camera produces — SARD survivor assets composited on
terrain-like backgrounds with the same wildfire smoke/flame/burn effects — and
fine-tune YOLO on it. The detector then sees in-distribution data and reaches
high confidence (~0.9) on survivors, including those under fire and smoke.

Usage:
    python scripts/train_survivor_detector.py                 # quick default run
    python scripts/train_survivor_detector.py --epochs 25 --n-train 800 --model yolov8s.pt

Output weights: models/survivor_yolov8n.pt (gitignored). Point the CV adapter
at them with --cv-person-model models/survivor_yolov8n.pt.
"""

from __future__ import annotations

import argparse
import glob
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from detection.wildfire_effects import (
    WildfireEffectConfig,
    WildfireMasks,
    apply_wildfire_effects_to_pil,
)


def _procedural_background(size: int, rng: np.random.Generator) -> Image.Image:
    """Terrain-like background: low-frequency green/brown/tan colour field."""
    low = rng.integers(40, 150, size=(8, 8, 3)).astype(np.float32)
    # Bias toward vegetation/soil hues.
    low[..., 1] *= rng.uniform(1.0, 1.4)   # green
    low[..., 0] *= rng.uniform(0.8, 1.2)   # red/soil
    img = Image.fromarray(np.clip(low, 0, 255).astype("uint8")).resize((size, size), Image.Resampling.BICUBIC)
    arr = np.asarray(img).astype(np.float32)
    arr += rng.normal(0, 12, arr.shape)    # fine texture
    return Image.fromarray(np.clip(arr, 0, 255).astype("uint8")).convert("RGB")


def _disc(size: int, cx: float, cy: float, r: float) -> np.ndarray:
    yy, xx = np.mgrid[0:size, 0:size]
    return np.clip(1.0 - np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / max(r, 1.0), 0, 1).astype(np.float32)


def _wildfire_masks(size, rng, centers):
    """Build burn/flame/smoke masks, biased to sit ON survivors.

    ``centers`` are pixel (cx, cy) of placed survivors; fire is centered on a
    random subset so the model sees plenty of survivors *inside* active fire
    and smoke — the hard case stock YOLO fails on.
    """
    burned = np.zeros((size, size), np.float32)
    active = np.zeros((size, size), np.float32)
    smoke = np.zeros((size, size), np.float32)
    # 1-2 fire blobs, each centered on a survivor (mostly) or random (sometimes).
    for _ in range(int(rng.integers(1, 3))):
        if centers and rng.random() < 0.8:
            cx, cy = centers[int(rng.integers(0, len(centers)))]
            cx += rng.uniform(-size * 0.05, size * 0.05); cy += rng.uniform(-size * 0.05, size * 0.05)
        else:
            cx, cy = rng.uniform(0, size), rng.uniform(0, size)
        burned = np.maximum(burned, _disc(size, cx, cy, size * rng.uniform(0.18, 0.5)) * rng.uniform(0.5, 1.0))
        active = np.maximum(active, _disc(size, cx, cy, size * rng.uniform(0.12, 0.38)) * rng.uniform(0.5, 1.0))
        smoke = np.maximum(smoke, _disc(size, cx, cy, size * rng.uniform(0.25, 0.6)) * rng.uniform(0.4, 1.0))
    return WildfireMasks(burned=burned, active=active, intensity=active.copy(), smoke=smoke)


def _generate_split(out_dir: Path, n: int, assets, size, rng, cfg, fire_frac=0.65, surv_px=(45, 230)):
    img_dir = out_dir / "images"; lbl_dir = out_dir / "labels"
    img_dir.mkdir(parents=True, exist_ok=True); lbl_dir.mkdir(parents=True, exist_ok=True)
    lo, hi = int(surv_px[0]), int(surv_px[1])
    for i in range(n):
        bg = _procedural_background(size, rng)
        # Decide survivor placements first so fire can be centered on them.
        placements = []
        for _ in range(int(rng.integers(1, 4))):  # 1-3 survivors
            asset = assets[int(rng.integers(0, len(assets)))]
            w = int(rng.integers(lo, hi)); h = int(w * asset.height / asset.width)
            x = int(rng.integers(0, max(1, size - w))); y = int(rng.integers(0, max(1, size - h)))
            placements.append((asset, w, h, x, y))
        centers = [(x + w / 2, y + h / 2) for (_, w, h, x, y) in placements]

        has_fire = rng.random() < fire_frac
        mask = _wildfire_masks(size, rng, centers) if has_fire else None
        if mask is not None:  # burn + flame UNDER survivors (production order)
            bg, _ = apply_wildfire_effects_to_pil(bg, mask, config=cfg, include_burn=True, include_flame=True, include_smoke=False)

        labels = []
        for asset, w, h, x, y in placements:
            s = asset.resize((w, h), Image.Resampling.LANCZOS)
            bg.paste(s, (x, y), s)
            labels.append(f"0 {(x+w/2)/size:.6f} {(y+h/2)/size:.6f} {w/size:.6f} {h/size:.6f}")

        if mask is not None:  # smoke drifts OVER survivors
            bg, _ = apply_wildfire_effects_to_pil(bg, mask, config=cfg, include_burn=False, include_flame=False, include_smoke=True)
        bg.save(img_dir / f"{i:05d}.jpg", quality=92)
        (lbl_dir / f"{i:05d}.txt").write_text("\n".join(labels) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolov8n.pt")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--n-train", type=int, default=500)
    ap.add_argument("--n-val", type=int, default=80)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--size", type=int, default=640, help="Generated training image size.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fire-frac", type=float, default=0.65,
                    help="Fraction of training images with fire/smoke centered on survivors.")
    ap.add_argument("--min-surv-px", type=int, default=45,
                    help="Min survivor width in the generated image. Use larger (e.g. 120) for a ground-robot close-range model.")
    ap.add_argument("--max-surv-px", type=int, default=230,
                    help="Max survivor width in the generated image.")
    ap.add_argument("--assets-dir", default=str(ROOT / "data/cv_assets/sard_grabcut"))
    ap.add_argument("--data-dir", default=str(ROOT / "data/cv_train/survivor"))
    ap.add_argument("--out", default=str(ROOT / "models/survivor_yolov8n.pt"))
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    random.seed(args.seed)
    cfg = WildfireEffectConfig()
    asset_paths = sorted(glob.glob(str(Path(args.assets_dir) / "*.png")))
    if not asset_paths:
        raise SystemExit(f"No survivor assets in {args.assets_dir}")
    assets = [Image.open(p).convert("RGBA") for p in asset_paths]
    # Hold out some assets for val so we test pose generalization, not memorization.
    n_hold = max(1, len(assets) // 5)
    train_assets, val_assets = assets[:-n_hold], assets[-n_hold:]

    data_dir = Path(args.data_dir)
    print(f"Generating {args.n_train} train / {args.n_val} val composites at {args.size}px "
          f"({int(args.fire_frac*100)}% with fire/smoke on survivors) ...")
    surv_px = (args.min_surv_px, args.max_surv_px)
    _generate_split(data_dir / "train", args.n_train, train_assets, args.size, rng, cfg, fire_frac=args.fire_frac, surv_px=surv_px)
    _generate_split(data_dir / "val", args.n_val, val_assets, args.size, rng, cfg, fire_frac=args.fire_frac, surv_px=surv_px)

    yaml = data_dir / "survivor.yaml"
    yaml.write_text(
        f"path: {data_dir}\ntrain: train/images\nval: val/images\nnames:\n  0: person\n",
        encoding="utf-8",
    )

    from ultralytics import YOLO
    model = YOLO(args.model)
    print(f"Fine-tuning {args.model} for {args.epochs} epochs ...")
    model.train(
        data=str(yaml), epochs=args.epochs, imgsz=args.imgsz, batch=16,
        device="cpu", verbose=True, seed=args.seed, project=str(data_dir.resolve() / "runs"),
        name="survivor", exist_ok=True, plots=False,
    )
    # Ultralytics records the best-weights path on the trainer; fall back to a
    # search (relative project paths get re-rooted under runs/detect/).
    best = Path(getattr(model.trainer, "best", data_dir / "runs" / "survivor" / "weights" / "best.pt"))
    if not best.exists():
        found = list(ROOT.rglob("survivor/weights/best.pt"))
        if found:
            best = max(found, key=lambda p: p.stat().st_mtime)
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    import shutil
    shutil.copy(best, out)
    print(f"\nWrote fine-tuned survivor detector to: {out}")
    print(f"Use it with: python scripts/export_trajectories.py --enable-cv --cv-detector yolo --cv-person-model {out}")


if __name__ == "__main__":
    main()
