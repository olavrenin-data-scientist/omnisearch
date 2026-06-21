"""Download VisDrone2019-DET-val and extract non-person aerial vehicle crops as
RGBA hard-negative decoy assets for YOLO training.

Why VisDrone?
-------------
VisDrone images are captured by consumer drones at 30–100 m altitude — the
same vantage point as our SAR drone.  Non-person categories (cars, vans,
trucks, buses, motorcycles, bicycles) appear at exactly the same angular size
and perspective as SARD survivors, but are clearly not people.  Compositing
them onto NAIP backgrounds through the same GrabCut → harmonize → blur
pipeline as survivor assets creates hard negatives that are indistinguishable
from real composites except in shape — exactly what Ann-Kathrin described.

Dataset details
---------------
* Source:    VisDrone2019-DET (detection in images task)
* Val split: ~500 images, ~200 MB
* License:   Research/non-commercial use only (aiskyeye.com)
* Citation:  Du et al., "VisDrone-DET2019: The Vision Meets Drone
             Object Detection in Image Challenge Results," ICCV 2019 Workshop.

VisDrone annotation format (one .txt per image, CSV per line):
    <bbox_left>,<bbox_top>,<bbox_width>,<bbox_height>,<score>,<object_category>,<truncation>,<occlusion>

Category mapping (1-indexed):
    1=pedestrian  2=person(sitting/lying)  — EXCLUDED (these are people!)
    3=car  4=van  5=truck  6=bus  7=motor  8=bicycle  9=awning-tricycle  10=tricycle

score=0 → annotator flagged as ignored; skip those boxes.

Usage
-----
    python scripts/extract_visdrone_decoys.py            # auto-downloads val set
    python scripts/extract_visdrone_decoys.py --zip path/to/VisDrone2019-DET-val.zip

    # After extraction:
    python scripts/train_survivor_detector.py \\
        --decoy-assets-dir data/cv_assets/visdrone_decoys \\
        --naip-dir data/source_cache/naip/...
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

try:
    import cv2
except ImportError:
    cv2 = None

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Ultralytics mirrors the VisDrone datasets for public access.
_VISDRONE_VAL_URL = (
    "https://github.com/ultralytics/assets/releases/download/v0.0.0/"
    "VisDrone2019-DET-val.zip"
)

# VisDrone categories we want as hard negatives (non-person).
_VEHICLE_CATEGORIES = {3, 4, 5, 6, 7, 8, 9, 10}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


# ---------------------------------------------------------------------------
# GrabCut helpers (shared pattern with extract_sard_assets.py)
# ---------------------------------------------------------------------------

def _grabcut_rgba(
    crop: Image.Image,
    *,
    foreground_box: tuple[int, int, int, int],
    fallback_threshold: float = 30.0,
) -> Image.Image:
    """Extract foreground from *crop* using GrabCut; fall back to border-diff mask."""
    if cv2 is None:
        return _border_diff_rgba(crop, threshold=fallback_threshold)

    rgb = np.asarray(crop.convert("RGB"), dtype=np.uint8)
    h, w = rgb.shape[:2]
    x1, y1, x2, y2 = foreground_box
    x1 = max(1, min(w - 2, int(x1)))
    y1 = max(1, min(h - 2, int(y1)))
    x2 = max(x1 + 2, min(w - 1, int(x2)))
    y2 = max(y1 + 2, min(h - 1, int(y2)))

    mask = np.full((h, w), cv2.GC_BGD, dtype=np.uint8)
    mask[y1:y2, x1:x2] = cv2.GC_PR_FGD
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    try:
        cv2.grabCut(bgr, mask, (x1, y1, x2 - x1, y2 - y1), bgd, fgd, 5, cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        return _border_diff_rgba(crop, threshold=fallback_threshold)

    alpha = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    if int(np.count_nonzero(alpha)) < max(1, (x2 - x1) * (y2 - y1)) * 0.10:
        return _border_diff_rgba(crop, threshold=fallback_threshold)

    pil_alpha = Image.fromarray(alpha)
    pil_alpha = pil_alpha.filter(ImageFilter.MaxFilter(size=3))
    pil_alpha = pil_alpha.filter(ImageFilter.GaussianBlur(radius=max(1, min(w, h) // 80)))
    rgba = crop.convert("RGBA")
    rgba.putalpha(pil_alpha)
    return rgba


def _border_diff_rgba(crop: Image.Image, threshold: float = 30.0) -> Image.Image:
    """Simple border-subtraction fallback alpha mask."""
    rgb = np.asarray(crop.convert("RGB"), dtype=np.float32)
    h, w = rgb.shape[:2]
    b = max(2, min(w, h) // 10)
    bg = np.median(
        np.concatenate([rgb[:b].reshape(-1, 3), rgb[-b:].reshape(-1, 3),
                        rgb[:, :b].reshape(-1, 3), rgb[:, -b:].reshape(-1, 3)], axis=0),
        axis=0,
    )
    diff = np.linalg.norm(rgb - bg, axis=2)
    alpha = (diff > threshold).astype(np.uint8) * 255
    pil_alpha = Image.fromarray(alpha).filter(ImageFilter.GaussianBlur(radius=max(1, min(w, h) // 45)))
    rgba = crop.convert("RGBA")
    rgba.putalpha(pil_alpha)
    return rgba


# ---------------------------------------------------------------------------
# VisDrone parsing
# ---------------------------------------------------------------------------

def _parse_visdrone_ann(ann_path: Path) -> list[tuple[int, int, int, int, int]]:
    """Return list of (x, y, w, h, category) tuples from a VisDrone .txt file."""
    boxes = []
    for line in ann_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.strip().split(",")
        if len(parts) < 6:
            continue
        try:
            x, y, w, h, score, cat = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4]), int(parts[5])
        except ValueError:
            continue
        if score == 0:
            continue   # annotator-flagged "ignored" region
        boxes.append((x, y, w, h, cat))
    return boxes


# ---------------------------------------------------------------------------
# Download helper
# ---------------------------------------------------------------------------

def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url} → {dest} …")

    def _progress(count: int, block: int, total: int) -> None:
        if total > 0:
            pct = min(100, 100 * count * block / total)
            print(f"\r  {pct:.1f}%", end="", flush=True)

    try:
        urllib.request.urlretrieve(url, dest, reporthook=_progress)
    except Exception as exc:
        raise SystemExit(
            f"\nDownload failed: {exc}\n\n"
            "Manual alternative: download VisDrone2019-DET-val.zip from\n"
            "  https://drive.google.com/file/d/1bxK5zgLn0_L8x276eKkuYA_FzwCIjb59\n"
            "and pass it with --zip path/to/VisDrone2019-DET-val.zip"
        ) from exc
    print()   # newline after progress


# ---------------------------------------------------------------------------
# Main extraction logic
# ---------------------------------------------------------------------------

def extract_decoys(
    visdrone_dir: Path,
    out_dir: Path,
    min_px: int = 20,
    max_assets: int = 500,
    padding: float = 0.15,
) -> int:
    """Extract non-person vehicle crops from an extracted VisDrone DET folder."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # VisDrone layout: images/ and annotations/ side-by-side.
    img_dir = visdrone_dir / "images"
    ann_dir = visdrone_dir / "annotations"
    if not img_dir.exists() or not ann_dir.exists():
        # Some zips nest under VisDrone2019-DET-val/
        for sub in visdrone_dir.iterdir():
            if sub.is_dir() and (sub / "images").exists():
                img_dir = sub / "images"
                ann_dir = sub / "annotations"
                break

    if not img_dir.exists():
        raise SystemExit(f"Could not find 'images/' under {visdrone_dir}")

    asset_idx = 0
    skipped_small = 0
    skipped_person = 0
    manifest = []

    for img_path in sorted(img_dir.iterdir()):
        if asset_idx >= max_assets:
            break
        if img_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        ann_path = ann_dir / (img_path.stem + ".txt")
        if not ann_path.exists():
            continue

        try:
            image = Image.open(img_path).convert("RGB")
        except OSError:
            continue
        iw, ih = image.size

        for x, y, bw, bh, cat in _parse_visdrone_ann(ann_path):
            if asset_idx >= max_assets:
                break
            if cat not in _VEHICLE_CATEGORIES:
                skipped_person += 1
                continue
            if bw < min_px or bh < min_px:
                skipped_small += 1
                continue

            # Add padding around the bbox.
            pad_x = int(bw * padding); pad_y = int(bh * padding)
            cx1 = max(0, x - pad_x); cy1 = max(0, y - pad_y)
            cx2 = min(iw, x + bw + pad_x); cy2 = min(ih, y + bh + pad_y)
            if cx2 - cx1 < min_px or cy2 - cy1 < min_px:
                continue

            crop = image.crop((cx1, cy1, cx2, cy2))
            fg_box = (x - cx1, y - cy1, x - cx1 + bw, y - cy1 + bh)
            rgba = _grabcut_rgba(crop, foreground_box=fg_box)

            asset_idx += 1
            name = f"visdrone_decoy_{asset_idx:04d}.png"
            rgba.save(out_dir / name)
            manifest.append({
                "asset": name,
                "source": img_path.name,
                "visdrone_category": cat,
                "bbox_xywh": [x, y, bw, bh],
            })

    import json
    (out_dir / "manifest.json").write_text(
        json.dumps({"source": "VisDrone2019-DET-val", "assets": manifest}, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Extracted {asset_idx} vehicle decoy assets → {out_dir}")
    print(f"  Skipped {skipped_small} boxes below {min_px} px minimum")
    print(f"  Skipped {skipped_person} person/pedestrian boxes (as intended)")
    return asset_idx


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--zip",
        default=None,
        help="Path to a pre-downloaded VisDrone2019-DET-val.zip. "
             "If omitted the script tries to download it automatically.",
    )
    ap.add_argument(
        "--out-dir",
        default=str(ROOT / "data/cv_assets/visdrone_decoys"),
        help="Output directory for extracted RGBA decoy PNGs.",
    )
    ap.add_argument(
        "--cache-dir",
        default=str(ROOT / "data/source_cache/visdrone"),
        help="Where to store the downloaded zip and extracted files.",
    )
    ap.add_argument("--min-px", type=int, default=20,
                    help="Minimum width and height (px) for a vehicle bbox to be kept.")
    ap.add_argument("--max-assets", type=int, default=500,
                    help="Maximum number of decoy PNGs to produce.")
    ap.add_argument("--padding", type=float, default=0.15,
                    help="Padding added around each bbox before cropping.")
    args = ap.parse_args()

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    if args.zip:
        zip_path = Path(args.zip)
        if not zip_path.exists():
            raise SystemExit(f"Provided zip not found: {zip_path}")
    else:
        zip_path = cache_dir / "VisDrone2019-DET-val.zip"
        if not zip_path.exists():
            _download(_VISDRONE_VAL_URL, zip_path)
        else:
            print(f"Using cached zip: {zip_path}")

    extract_dir = cache_dir / "VisDrone2019-DET-val"
    if not extract_dir.exists():
        print(f"Extracting {zip_path} …")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(cache_dir)
        # Locate the extracted folder (zip root name varies).
        candidates = [d for d in cache_dir.iterdir() if d.is_dir() and "visdrone" in d.name.lower()]
        if candidates:
            extract_dir = candidates[0]
        else:
            extract_dir = cache_dir

    n = extract_decoys(
        visdrone_dir=extract_dir,
        out_dir=Path(args.out_dir),
        min_px=args.min_px,
        max_assets=args.max_assets,
        padding=args.padding,
    )

    if n > 0:
        print(
            f"\nNext step: pass --decoy-assets-dir {args.out_dir} to train_survivor_detector.py\n"
            f"  python scripts/train_survivor_detector.py \\\n"
            f"      --decoy-assets-dir {args.out_dir} \\\n"
            f"      --naip-dir data/source_cache/naip/..."
        )


if __name__ == "__main__":
    main()
