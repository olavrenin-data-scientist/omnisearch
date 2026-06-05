"""Analyze extracted survivor PNG assets for CV simulation review.

The goal is not to perfectly judge realism automatically. It is to narrow a
large asset folder into practical review groups: high-detail candidates,
tiny/low-detail assets, masks that touch crop edges, and low-occupancy masks.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets-dir", default="data/cv_assets/sard_grabcut")
    parser.add_argument("--out-dir", default="results/cv_asset_analysis")
    parser.add_argument("--max-sheet-items", type=int, default=36)
    parser.add_argument("--preview-altitude-m", type=float, default=20.0)
    parser.add_argument("--fov-deg", type=float, default=90.0)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--survivor-width-m", type=float, default=2.4)
    parser.add_argument("--survivor-height-m", type=float, default=1.4)
    args = parser.parse_args()

    assets_dir = _resolve(args.assets_dir)
    out_dir = _resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    records = analyze_assets(
        assets_dir,
        preview_altitude_m=args.preview_altitude_m,
        fov_deg=args.fov_deg,
        image_size=args.image_size,
        survivor_width_m=args.survivor_width_m,
        survivor_height_m=args.survivor_height_m,
    )
    if not records:
        raise SystemExit(f"No PNG assets found in {assets_dir}")

    csv_path = out_dir / "sard_grabcut_asset_metrics.csv"
    _write_csv(records, csv_path)
    summary_path = out_dir / "sard_grabcut_asset_summary.json"
    _write_summary(records, summary_path, args)
    sheet_manifest = _write_contact_sheets(records, out_dir, max_items=args.max_sheet_items)

    print(f"Analyzed {len(records)} survivor assets")
    print(f"Wrote metrics: {csv_path}")
    print(f"Wrote summary: {summary_path}")
    for name, path in sheet_manifest.items():
        print(f"Wrote {name}: {path}")


def analyze_assets(
    assets_dir: Path,
    *,
    preview_altitude_m: float,
    fov_deg: float,
    image_size: int,
    survivor_width_m: float,
    survivor_height_m: float,
) -> list[dict]:
    footprint_m = 2.0 * float(preview_altitude_m) * math.tan(math.radians(float(fov_deg)) / 2.0)
    target_width_px = survivor_width_m / footprint_m * image_size
    target_height_px = survivor_height_m / footprint_m * image_size
    records: list[dict] = []
    for path in sorted(assets_dir.glob("*.png")):
        with Image.open(path) as raw:
            rgba = raw.convert("RGBA")
        record = _asset_record(
            path=path,
            image=rgba,
            target_width_px=target_width_px,
            target_height_px=target_height_px,
        )
        if record is not None:
            records.append(record)
    return records


def _asset_record(
    *,
    path: Path,
    image: Image.Image,
    target_width_px: float,
    target_height_px: float,
) -> dict | None:
    alpha = np.asarray(image.getchannel("A"))
    visible_mask = alpha > 16
    if not visible_mask.any():
        return None

    ys, xs = np.where(visible_mask)
    left, right = int(xs.min()), int(xs.max()) + 1
    top, bottom = int(ys.min()), int(ys.max()) + 1
    visible_w = right - left
    visible_h = bottom - top
    bbox_area = visible_w * visible_h
    visible_pixels = int(visible_mask.sum())
    occupancy = visible_pixels / max(1, bbox_area)
    mean_alpha = float(alpha[visible_mask].mean())
    edge_touch = bool(
        visible_mask[0, :].any()
        or visible_mask[-1, :].any()
        or visible_mask[:, 0].any()
        or visible_mask[:, -1].any()
    )
    aspect = visible_w / max(1, visible_h)
    source_to_target_min_ratio = min(
        visible_w / max(target_width_px, 1e-6),
        visible_h / max(target_height_px, 1e-6),
    )

    flags = []
    if bbox_area < 1_500 or min(visible_w, visible_h) < 18:
        flags.append("tiny_source")
    if source_to_target_min_ratio < 1.0:
        flags.append("below_20m_target_resolution")
    if occupancy < 0.35:
        flags.append("low_alpha_occupancy")
    if mean_alpha < 180.0:
        flags.append("soft_or_incomplete_mask")
    if edge_touch:
        flags.append("mask_touches_crop_edge")
    if aspect < 0.22 or aspect > 4.0:
        flags.append("extreme_aspect")

    score = _quality_score(
        bbox_area=bbox_area,
        occupancy=occupancy,
        mean_alpha=mean_alpha,
        source_to_target_min_ratio=source_to_target_min_ratio,
        edge_touch=edge_touch,
    )
    status = "candidate"
    if "tiny_source" in flags or "below_20m_target_resolution" in flags:
        status = "reject_or_resample"
    elif flags:
        status = "review"

    return {
        "file": path.name,
        "path": str(path),
        "image_width": image.width,
        "image_height": image.height,
        "visible_width": visible_w,
        "visible_height": visible_h,
        "visible_bbox_area": bbox_area,
        "visible_alpha_pixels": visible_pixels,
        "alpha_occupancy": round(float(occupancy), 4),
        "mean_alpha_visible": round(mean_alpha, 2),
        "aspect": round(float(aspect), 4),
        "source_to_20m_target_min_ratio": round(float(source_to_target_min_ratio), 3),
        "edge_touch": edge_touch,
        "score": round(score, 2),
        "status": status,
        "flags": flags,
    }


def _quality_score(
    *,
    bbox_area: int,
    occupancy: float,
    mean_alpha: float,
    source_to_target_min_ratio: float,
    edge_touch: bool,
) -> float:
    detail_score = min(math.sqrt(max(bbox_area, 0)) / math.sqrt(12_000), 1.0)
    occupancy_score = min(max(occupancy / 0.55, 0.0), 1.0)
    alpha_score = min(max(mean_alpha / 220.0, 0.0), 1.0)
    resolution_score = min(max(source_to_target_min_ratio / 2.0, 0.0), 1.0)
    edge_penalty = 0.78 if edge_touch else 1.0
    return 100.0 * detail_score * occupancy_score * alpha_score * resolution_score * edge_penalty


def _write_csv(records: list[dict], path: Path) -> None:
    fieldnames = [
        "file",
        "status",
        "score",
        "image_width",
        "image_height",
        "visible_width",
        "visible_height",
        "visible_bbox_area",
        "visible_alpha_pixels",
        "alpha_occupancy",
        "mean_alpha_visible",
        "aspect",
        "source_to_20m_target_min_ratio",
        "edge_touch",
        "flags",
        "path",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            row = dict(record)
            row["flags"] = ";".join(record["flags"])
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _write_summary(records: list[dict], path: Path, args: argparse.Namespace) -> None:
    by_status: dict[str, int] = {}
    by_flag: dict[str, int] = {}
    for record in records:
        by_status[record["status"]] = by_status.get(record["status"], 0) + 1
        for flag in record["flags"]:
            by_flag[flag] = by_flag.get(flag, 0) + 1

    def percentile(field: str, q: float) -> float:
        return round(float(np.percentile([record[field] for record in records], q)), 3)

    summary = {
        "assets_dir": str(_resolve(args.assets_dir)),
        "count": len(records),
        "preview_model": {
            "altitude_m": args.preview_altitude_m,
            "fov_deg": args.fov_deg,
            "image_size_px": args.image_size,
            "survivor_width_m": args.survivor_width_m,
            "survivor_height_m": args.survivor_height_m,
        },
        "status_counts": dict(sorted(by_status.items())),
        "flag_counts": dict(sorted(by_flag.items())),
        "visible_bbox_area": {
            "p10": percentile("visible_bbox_area", 10),
            "median": percentile("visible_bbox_area", 50),
            "p90": percentile("visible_bbox_area", 90),
        },
        "source_to_20m_target_min_ratio": {
            "p10": percentile("source_to_20m_target_min_ratio", 10),
            "median": percentile("source_to_20m_target_min_ratio", 50),
            "p90": percentile("source_to_20m_target_min_ratio", 90),
        },
        "top_candidates": [
            {
                "file": record["file"],
                "score": record["score"],
                "visible_size": [record["visible_width"], record["visible_height"]],
                "flags": record["flags"],
            }
            for record in sorted(records, key=lambda item: item["score"], reverse=True)[:20]
        ],
    }
    path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def _write_contact_sheets(records: list[dict], out_dir: Path, *, max_items: int) -> dict[str, str]:
    groups = {
        "best_candidates": sorted(
            [record for record in records if record["status"] == "candidate"],
            key=lambda item: item["score"],
            reverse=True,
        )[:max_items],
        "needs_review": sorted(
            [record for record in records if record["status"] == "review"],
            key=lambda item: item["score"],
            reverse=True,
        )[:max_items],
        "tiny_or_low_resolution": sorted(
            [record for record in records if record["status"] == "reject_or_resample"],
            key=lambda item: item["score"],
        )[:max_items],
        "mask_edge_touch": sorted(
            [record for record in records if "mask_touches_crop_edge" in record["flags"]],
            key=lambda item: item["score"],
            reverse=True,
        )[:max_items],
        "largest_visible": sorted(records, key=lambda item: item["visible_bbox_area"], reverse=True)[:max_items],
    }
    manifest = {}
    for name, items in groups.items():
        if not items:
            continue
        out_path = out_dir / f"{name}.png"
        _write_sheet(items, out_path)
        manifest[name] = str(out_path)
    manifest_path = out_dir / "contact_sheets_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def _write_sheet(items: list[dict], out_path: Path) -> None:
    columns = 6
    thumb_w = 160
    thumb_h = 130
    label_h = 64
    pad = 14
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
        ax = (thumb_w - asset.width) // 2
        ay = (thumb_h - asset.height) // 2
        tile.alpha_composite(asset, (ax, ay))
        sheet.paste(tile.convert("RGB"), (x, y))
        label = (
            f"{item['file']}\n"
            f"score={item['score']} {item['status']}\n"
            f"{item['visible_width']}x{item['visible_height']} ratio={item['source_to_20m_target_min_ratio']}"
        )
        draw.multiline_text((x, y + thumb_h + 5), label, fill=(20, 20, 20), font=font, spacing=2)

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


def _resolve(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return ROOT / path


if __name__ == "__main__":
    main()
