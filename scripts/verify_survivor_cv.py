"""Visual + numeric check of survivor CV detection under fire and smoke.

Renders survivor composites (clean, light smoke, heavy smoke, flames+burn,
flames+burn+smoke) using the real wildfire effects, runs the SimulationCvAdapter
YOLO detector (tiled small-object inference) over each, draws the detected boxes
with confidence, and saves a montage. Prints a per-scenario detection table.

    python scripts/verify_survivor_cv.py
    python scripts/verify_survivor_cv.py --model models/survivor_yolov8n.pt
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from detection.simulation_adapter import SimulationCvAdapter
from detection.wildfire_effects import WildfireEffectConfig, WildfireMasks, apply_wildfire_effects_to_pil


def _disc(size, cx, cy, r):
    yy, xx = np.mgrid[0:size, 0:size]
    return np.clip(1.0 - np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / r, 0, 1).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolov8n.pt", help="YOLO weights (stock or fine-tuned).")
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--surv-px", type=int, default=150)
    ap.add_argument("--conf", type=float, default=0.12)
    ap.add_argument("--out", default=str(ROOT / "docs" / "survivor_cv_detection.png"))
    args = ap.parse_args()

    SZ = args.size
    det = object.__new__(SimulationCvAdapter)
    det.detector_backend = "yolo"; det.person_model_name = args.model; det.person_conf = args.conf
    det.person_iou = 0.6; det.person_imgsz = 1280; det.person_tiled = True; det.person_tile_grid = 2
    det.person_tile_overlap = 0.25; det.person_match_iou = 0.15; det.person_device = None
    det._person_detector = None; det.image_size = SZ
    cfg = WildfireEffectConfig()

    asset = Image.open(sorted(glob.glob(str(ROOT / "data/cv_assets/sard_grabcut/*.png")))[0]).convert("RGBA")

    def bg():
        b = Image.fromarray(np.random.default_rng(2).integers(70, 120, (SZ, SZ, 3)).astype("uint8")).convert("RGB")
        a = np.asarray(b).astype(float); a[..., 1] *= 1.2
        return Image.fromarray(np.clip(a, 0, 255).astype("uint8"))

    def paste(b):
        h = int(args.surv_px * asset.height / asset.width)
        s = asset.resize((args.surv_px, h), Image.Resampling.LANCZOS)
        b = b.copy(); b.paste(s, (SZ // 2 - args.surv_px // 2, SZ // 2 - h // 2), s)
        return b

    def masks(s=0, f=0, bn=0):
        c = SZ // 2
        return WildfireMasks(burned=_disc(SZ, c, c, SZ * 0.4) * bn, active=_disc(SZ, c, c, SZ * 0.3) * f,
                             intensity=_disc(SZ, c, c, SZ * 0.3) * f, smoke=_disc(SZ, c, c, SZ * 0.45) * s)

    scenes = {}
    scenes["clean"] = paste(bg())
    v = paste(bg()); v, _ = apply_wildfire_effects_to_pil(v, masks(s=0.5), config=cfg, include_burn=False, include_flame=False, include_smoke=True); scenes["light smoke"] = v
    v = paste(bg()); v, _ = apply_wildfire_effects_to_pil(v, masks(s=1.0), config=cfg, include_burn=False, include_flame=False, include_smoke=True); scenes["heavy smoke"] = v
    m = masks(f=1.0, bn=1.0); b = bg(); b, _ = apply_wildfire_effects_to_pil(b, m, config=cfg, include_burn=True, include_flame=True, include_smoke=False); scenes["flames+burn"] = paste(b)
    m = masks(s=1.0, f=1.0, bn=1.0); b = bg(); b, _ = apply_wildfire_effects_to_pil(b, m, config=cfg, include_burn=True, include_flame=True, include_smoke=False)
    v = paste(b); v, _ = apply_wildfire_effects_to_pil(v, m, config=cfg, include_burn=False, include_flame=False, include_smoke=True); scenes["fire+smoke"] = v

    print(f"model: {args.model}")
    print(f"{'scenario':14} {'detections':>10} {'max_conf':>9}")
    tiles = []
    for name, view in scenes.items():
        dets = det._detect_people_cv(view)
        draw = ImageDraw.Draw(view)
        for box, conf in dets:
            draw.rectangle([box[0], box[1], box[2], box[3]], outline=(0, 255, 0), width=4)
            draw.text((box[0], max(0, box[1] - 14)), f"{conf:.2f}", fill=(0, 255, 0))
        maxc = max([c for _, c in dets], default=0.0)
        print(f"{name:14} {len(dets):10d} {maxc:9.2f}")
        lbl = view.resize((360, 360)); d2 = ImageDraw.Draw(lbl); d2.text((6, 6), name, fill=(255, 255, 0))
        tiles.append(lbl)

    montage = Image.new("RGB", (360 * len(tiles), 360), (10, 10, 14))
    for i, t in enumerate(tiles):
        montage.paste(t, (i * 360, 0))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    montage.save(args.out)
    print(f"\nSaved montage: {args.out}")


if __name__ == "__main__":
    main()
