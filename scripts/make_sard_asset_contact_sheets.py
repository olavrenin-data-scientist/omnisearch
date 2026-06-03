"""Create review contact sheets for extracted SARD survivor assets."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets-dir", default="data/cv_assets/sard")
    parser.add_argument("--out-dir", default="results/cv_demo_sard_asset_review")
    parser.add_argument("--prefix", default="sard_assets")
    parser.add_argument("--max-items", type=int, default=24)
    args = parser.parse_args()

    assets_dir = Path(args.assets_dir)
    if not assets_dir.is_absolute():
        assets_dir = ROOT / assets_dir
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    records = _asset_records(assets_dir)
    if not records:
        raise SystemExit(f"No PNG assets found in {assets_dir}")

    groups = {
        "largest": sorted(records, key=lambda item: item["area"], reverse=True)[: args.max_items],
        "wide_lying": sorted(
            [item for item in records if item["aspect"] >= 1.25],
            key=lambda item: item["area"],
            reverse=True,
        )[: args.max_items],
        "tall_standing": sorted(
            [item for item in records if item["aspect"] <= 0.8],
            key=lambda item: item["area"],
            reverse=True,
        )[: args.max_items],
        "medium_balanced": sorted(
            [item for item in records if 0.8 < item["aspect"] < 1.25],
            key=lambda item: item["area"],
            reverse=True,
        )[: args.max_items],
    }

    manifest = {}
    for name, items in groups.items():
        if not items:
            continue
        path = out_dir / f"{args.prefix}_{name}.png"
        _write_sheet(items, path)
        manifest[name] = str(path)

    manifest_path = out_dir / f"{args.prefix}_contact_sheets_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(manifest)} contact sheets to {out_dir}")
    print(f"Wrote manifest: {manifest_path}")


def _asset_records(assets_dir: Path) -> list[dict]:
    records = []
    for path in sorted(assets_dir.glob("*.png")):
        with Image.open(path) as image:
            rgba = image.convert("RGBA")
            alpha = rgba.getchannel("A")
            bbox = alpha.getbbox()
            if bbox is None:
                continue
            visible_width = bbox[2] - bbox[0]
            visible_height = bbox[3] - bbox[1]
            records.append(
                {
                    "path": path,
                    "size": rgba.size,
                    "visible_size": (visible_width, visible_height),
                    "area": visible_width * visible_height,
                    "aspect": visible_width / max(1, visible_height),
                }
            )
    return records


def _write_sheet(items: list[dict], out_path: Path) -> None:
    columns = 6
    thumb_w = 150
    thumb_h = 120
    label_h = 54
    pad = 16
    rows = math.ceil(len(items) / columns)
    sheet = Image.new("RGB", (columns * (thumb_w + pad) + pad, rows * (thumb_h + label_h + pad) + pad), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()

    checker = _checkerboard((thumb_w, thumb_h))
    for index, item in enumerate(items):
        col = index % columns
        row = index // columns
        x = pad + col * (thumb_w + pad)
        y = pad + row * (thumb_h + label_h + pad)
        tile = checker.copy()
        with Image.open(item["path"]) as raw:
            asset = raw.convert("RGBA")
        asset.thumbnail((thumb_w, thumb_h), Image.Resampling.NEAREST)
        ax = x + (thumb_w - asset.width) // 2
        ay = y + (thumb_h - asset.height) // 2
        tile.alpha_composite(asset, (ax - x, ay - y))
        sheet.paste(tile.convert("RGB"), (x, y))
        label = f"{item['path'].name}\n{item['visible_size'][0]}x{item['visible_size'][1]} area={item['area']}"
        draw.multiline_text((x, y + thumb_h + 6), label, fill=(20, 20, 20), font=font, spacing=2)

    sheet.save(out_path)


def _checkerboard(size: tuple[int, int]) -> Image.Image:
    image = Image.new("RGBA", size, (238, 238, 238, 255))
    draw = ImageDraw.Draw(image)
    square = 10
    for y in range(0, size[1], square):
        for x in range(0, size[0], square):
            if (x // square + y // square) % 2:
                draw.rectangle((x, y, x + square - 1, y + square - 1), fill=(214, 214, 214, 255))
    return image


if __name__ == "__main__":
    main()
