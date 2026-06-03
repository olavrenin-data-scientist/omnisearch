"""Extract small demo survivor assets from the public NII-CU MAPD sample image."""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

from PIL import Image, ImageDraw


SAMPLE_RGB_URL = "https://www.nii-cu-multispectral.org/samples/flight2_frame12641_rgb.jpg"

# Crops are from the public annotated sample. The boxes are tightened slightly
# to avoid most of the blue annotation rectangle while keeping the real person.
PERSON_CROPS = [
    (386, 313, 399, 338),
    (492, 284, 504, 313),
    (592, 176, 604, 204),
    (837, 240, 849, 263),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="data/cv_assets/nii_cu_sample")
    parser.add_argument("--force-download", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    sample_path = out_dir / "flight2_frame12641_rgb.jpg"
    if args.force_download or not sample_path.exists():
        request = urllib.request.Request(SAMPLE_RGB_URL, headers={"User-Agent": "omnisearch-cv-demo/0.1"})
        with urllib.request.urlopen(request, timeout=120) as response:
            sample_path.write_bytes(response.read())

    image = Image.open(sample_path).convert("RGB")
    asset_paths = []
    for idx, crop_box in enumerate(PERSON_CROPS, start=1):
        crop = image.crop(crop_box).resize((40, 90), Image.Resampling.BILINEAR)
        asset = _soft_capsule_rgba(crop)
        asset_path = out_dir / f"nii_cu_person_{idx:02d}.png"
        asset.save(asset_path)
        asset_paths.append(asset_path)

    print(f"Wrote source sample: {sample_path}")
    for path in asset_paths:
        print(f"Wrote asset: {path}")


def _soft_capsule_rgba(crop: Image.Image) -> Image.Image:
    """Apply a soft body-shaped alpha mask to a tight person crop."""

    rgba = crop.convert("RGBA")
    mask = Image.new("L", rgba.size, 0)
    draw = ImageDraw.Draw(mask)
    w, h = rgba.size
    draw.ellipse((w * 0.25, 0, w * 0.75, h * 0.22), fill=255)
    draw.rounded_rectangle((w * 0.18, h * 0.14, w * 0.82, h * 0.98), radius=max(2, w // 5), fill=255)
    rgba.putalpha(mask)
    return rgba


if __name__ == "__main__":
    main()
