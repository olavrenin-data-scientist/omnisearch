"""Generate Plotly.js EDA plots for the drone perception model.

The input is an exported trajectory JSON from ``scripts/export_trajectories.py``.
The output is a standalone HTML report with actual analysis plots, not a replay
viewer:

    python scripts/perception_eda.py \
      --trajectory web/trajectories/nearest_candidate.json \
      --out results/eda/nearest_candidate_perception_eda.html
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


PLOTLY_JS = "https://cdn.plot.ly/plotly-2.35.2.min.js"
LAND_COVER_FALLBACK = ["road", "open", "brush", "forest", "rock", "water"]
COLORS = [
    "#60a5fa",
    "#22c55e",
    "#fbbf24",
    "#f97316",
    "#a855f7",
    "#14b8a6",
    "#ef4444",
]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--trajectory",
        default=str(ROOT / "web" / "trajectories" / "nearest_candidate.json"),
        help="Trajectory JSON exported by scripts/export_trajectories.py.",
    )
    p.add_argument("--out", default=None, help="Output HTML path.")
    p.add_argument(
        "--sample-points",
        type=int,
        default=12_000,
        help="Maximum drone-survivor samples shown in scatter plots.",
    )
    args = p.parse_args()

    trajectory_path = Path(args.trajectory)
    out = Path(args.out) if args.out else (
        ROOT / "results" / "eda" / f"{trajectory_path.stem}_perception_eda.html"
    )
    generate_perception_eda_html(
        trajectory_path=trajectory_path,
        out=out,
        sample_points=args.sample_points,
    )


def generate_perception_eda_html(
    *,
    trajectory_path: Path,
    out: Path,
    sample_points: int = 12_000,
) -> Path:
    data = _load_json(trajectory_path)
    records = _collect_records(data)
    if not records:
        raise SystemExit(
            "trajectory does not contain drone_perception records; rerun scripts/export_trajectories.py first"
        )

    meta = data.get("metadata", {})
    frames = data.get("frames", [])
    cover_names = meta.get("terrain", {}).get("cover_names") or LAND_COVER_FALLBACK
    sampled = _sample_records(records, max_points=sample_points)
    plots = [
        _plot_probability_vs_distance(sampled),
        _plot_component_boxes(records),
        _plot_probability_by_land_cover(records, cover_names),
        _plot_component_time_series(records),
        _plot_survivor_probability_timeline(data),
        _plot_footprint_coverage(data),
    ]
    summary = _summary(trajectory_path, meta, frames, records)
    html = _build_html(summary=summary, plots=plots)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html)
    print(f"Wrote {out}")
    print(f"Open with: open {out}")
    return out


def _load_json(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"trajectory not found: {path}")
    return json.loads(path.read_text())


def _collect_records(data: dict) -> list[dict]:
    records = []
    for frame in data.get("frames", []):
        step = int(frame.get("step", 0))
        for drone in frame.get("drone_perception", []):
            base = {
                "step": step,
                "drone": drone.get("name", "drone"),
                "footprint": _float_or_nan(drone.get("footprint")),
                "altitude_agl": _float_or_nan(drone.get("altitude_agl")),
                "altitude_msl": _float_or_nan(drone.get("altitude_msl")),
                "altitude_level": int(drone.get("altitude_level", -1)),
            }
            for survivor in drone.get("survivors", []):
                record = dict(base)
                record.update({
                    "survivor": int(survivor.get("index", -1)),
                    "distance": _float_or_nan(survivor.get("distance")),
                    "visible": bool(survivor.get("visible", False)),
                    "probability": _float_or_nan(survivor.get("probability")),
                    "distance_factor": _float_or_nan(survivor.get("distance_factor")),
                    "environment_factor": _float_or_nan(
                        survivor.get("environment_factor", survivor.get("cover_factor")),
                    ),
                    "cover_factor": _float_or_nan(survivor.get("cover_factor")),
                    "fire_smoke_factor": _float_or_nan(survivor.get("fire_smoke_factor")),
                    "altitude_quality": _float_or_nan(survivor.get("altitude_quality")),
                    "land_cover": int(survivor.get("land_cover", -1)),
                })
                records.append(record)
    return records


def _plot_probability_vs_distance(records: list[dict]) -> dict:
    traces = []
    levels = sorted({r["altitude_level"] for r in records})
    for i, level in enumerate(levels):
        group = [r for r in records if r["altitude_level"] == level]
        color = COLORS[i % len(COLORS)]
        traces.append({
            "type": "scattergl",
            "mode": "markers",
            "name": f"level {level} samples",
            "x": _values(group, "distance"),
            "y": _values(group, "probability"),
            "text": [
                f"{r['drone']} -> survivor {r['survivor']}<br>"
                f"visible={r['visible']}<br>footprint={r['footprint']:.3f}"
                for r in group
            ],
            "marker": {"size": 4, "opacity": 0.38, "color": color},
            "hovertemplate": "distance=%{x:.3f}<br>p=%{y:.3f}<br>%{text}<extra></extra>",
        })
        xs, ys = _binned_mean(group, "distance", "probability", bins=28)
        if xs:
            traces.append({
                "type": "scatter",
                "mode": "lines",
                "name": f"level {level} binned mean",
                "x": xs,
                "y": ys,
                "line": {"width": 3, "color": color},
                "hovertemplate": "distance bin=%{x:.3f}<br>mean p=%{y:.3f}<extra></extra>",
            })
    return _plot(
        "probability-distance",
        "Detection Probability vs Drone-Survivor Distance",
        traces,
        {
            "xaxis": {"title": "distance in world units"},
            "yaxis": {"title": "final detection probability", "range": [-0.03, 1.03]},
        },
        "Shows the actual probability used before the stochastic detection draw. "
        "Samples outside the camera footprint appear at probability zero.",
    )


def _plot_component_boxes(records: list[dict]) -> dict:
    visible = [r for r in records if r["visible"]]
    source = visible or records
    components = [
        ("altitude_quality", "altitude quality"),
        ("distance_factor", "distance factor"),
        ("environment_factor", "environment factor"),
        ("fire_smoke_factor", "fire/smoke factor"),
        ("probability", "final probability"),
    ]
    traces = [
        {
            "type": "box",
            "name": label,
            "y": _values(source, key),
            "boxpoints": "outliers",
            "marker": {"color": COLORS[i % len(COLORS)]},
        }
        for i, (key, label) in enumerate(components)
    ]
    subtitle = "visible drone-survivor pairs only" if visible else "all pairs; no visible pairs in this trajectory"
    return _plot(
        "component-boxes",
        "Perception Component Distributions",
        traces,
        {
            "yaxis": {"title": "component value", "range": [-0.03, 1.03]},
            "showlegend": False,
        },
        f"Factor breakdown for {subtitle}. Final probability is their product, clipped to [0, 1].",
    )


def _plot_probability_by_land_cover(records: list[dict], cover_names: list[str]) -> dict:
    visible = [r for r in records if r["visible"]]
    source = visible or records
    traces = []
    for cover in sorted({r["land_cover"] for r in source if r["land_cover"] >= 0}):
        group = [r for r in source if r["land_cover"] == cover]
        name = cover_names[cover] if cover < len(cover_names) else f"cover {cover}"
        traces.append({
            "type": "box",
            "name": name,
            "y": _values(group, "probability"),
            "boxpoints": "outliers",
        })
    return _plot(
        "probability-cover",
        "Detection Probability by Survivor Land Cover",
        traces,
        {
            "xaxis": {"title": "survivor land cover cell"},
            "yaxis": {"title": "final detection probability", "range": [-0.03, 1.03]},
            "showlegend": False,
        },
        "Uses survivor cell land cover. This quickly exposes whether brush/forest/rock penalties dominate detection.",
    )


def _plot_component_time_series(records: list[dict]) -> dict:
    by_step: dict[int, list[dict]] = defaultdict(list)
    for record in records:
        if record["visible"]:
            by_step[record["step"]].append(record)
    if not by_step:
        for record in records:
            by_step[record["step"]].append(record)
    steps = sorted(by_step)
    components = [
        ("probability", "final p"),
        ("distance_factor", "distance"),
        ("environment_factor", "environment"),
        ("fire_smoke_factor", "fire/smoke"),
        ("altitude_quality", "altitude"),
    ]
    traces = []
    for i, (key, label) in enumerate(components):
        traces.append({
            "type": "scatter",
            "mode": "lines",
            "name": label,
            "x": steps,
            "y": [_mean(_values(by_step[step], key)) for step in steps],
            "line": {"width": 2.5, "color": COLORS[i % len(COLORS)]},
        })
    return _plot(
        "component-time",
        "Mean Perception Factors Over Time",
        traces,
        {
            "xaxis": {"title": "step"},
            "yaxis": {"title": "mean factor value", "range": [-0.03, 1.03]},
        },
        "A dip in the fire/smoke line means the fire environment, not the search geometry, is degrading visibility.",
    )


def _plot_survivor_probability_timeline(data: dict) -> dict:
    frames = data.get("frames", [])
    meta = data.get("metadata", {})
    n_survivors = int(meta.get("n_survivors", 0))
    if n_survivors <= 0:
        n_survivors = 1 + max(
            (s.get("index", -1) for f in frames for d in f.get("drone_perception", []) for s in d.get("survivors", [])),
            default=-1,
        )
    steps = [int(f.get("step", i)) for i, f in enumerate(frames)]
    heat = np.zeros((n_survivors, len(frames)), dtype=float)
    first_scout: dict[int, int] = {}
    first_found: dict[int, int] = {}

    for col, frame in enumerate(frames):
        best = np.zeros(n_survivors, dtype=float)
        for drone in frame.get("drone_perception", []):
            for survivor in drone.get("survivors", []):
                idx = int(survivor.get("index", -1))
                if 0 <= idx < n_survivors:
                    best[idx] = max(best[idx], _float_or_nan(survivor.get("probability"), default=0.0))
        heat[:, col] = best
        for idx, survivor in enumerate(frame.get("survivors", [])):
            if survivor.get("scouted") and idx not in first_scout:
                first_scout[idx] = int(frame.get("step", col))
            if survivor.get("found") and idx not in first_found:
                first_found[idx] = int(frame.get("step", col))

    labels = [f"survivor {i}" for i in range(n_survivors)]
    traces = [{
        "type": "heatmap",
        "x": steps,
        "y": labels,
        "z": _round_grid(heat),
        "zmin": 0.0,
        "zmax": 1.0,
        "colorscale": "Viridis",
        "colorbar": {"title": "best p"},
        "hovertemplate": "step=%{x}<br>%{y}<br>best p=%{z:.3f}<extra></extra>",
    }]
    if first_scout:
        traces.append({
            "type": "scatter",
            "mode": "markers",
            "name": "first scouted",
            "x": [first_scout[i] for i in sorted(first_scout)],
            "y": [labels[i] for i in sorted(first_scout)],
            "marker": {"symbol": "x", "size": 11, "color": "#fbbf24", "line": {"width": 2}},
            "hovertemplate": "first scouted at step %{x}<br>%{y}<extra></extra>",
        })
    if first_found:
        traces.append({
            "type": "scatter",
            "mode": "markers",
            "name": "first confirmed",
            "x": [first_found[i] for i in sorted(first_found)],
            "y": [labels[i] for i in sorted(first_found)],
            "marker": {"symbol": "circle", "size": 9, "color": "#84cc16"},
            "hovertemplate": "first confirmed at step %{x}<br>%{y}<extra></extra>",
        })
    return _plot(
        "survivor-timeline",
        "Best Drone Detection Probability per Survivor Over Time",
        traces,
        {
            "xaxis": {"title": "step"},
            "yaxis": {"title": "survivor"},
        },
        "The heatmap uses the best probability across drones before the random detection draw.",
    )


def _plot_footprint_coverage(data: dict) -> dict:
    meta = data.get("metadata", {})
    terrain = meta.get("terrain", {})
    land_cover = np.asarray(terrain.get("land_cover", []), dtype=int)
    frames = data.get("frames", [])
    if land_cover.ndim != 2 or land_cover.size == 0:
        return _plot(
            "footprint-coverage",
            "Drone Footprint Coverage of Searchable Land",
            [],
            {},
            "No terrain land-cover grid was available in the trajectory metadata.",
        )

    x_semidim = float(meta.get("world", {}).get("x_semidim", 1.0))
    y_semidim = float(meta.get("world", {}).get("y_semidim", 1.0))
    height, width = land_cover.shape
    xs = np.linspace(-x_semidim, x_semidim, width, endpoint=False) + x_semidim / width
    ys = np.linspace(-y_semidim, y_semidim, height, endpoint=False) + y_semidim / height
    xx, yy = np.meshgrid(xs, ys)
    land_mask = land_cover != 5
    land_x = xx[land_mask]
    land_y = yy[land_mask]
    n_land = max(int(land_x.size), 1)

    steps = []
    coverage_pct = []
    mean_footprint = []
    drone_count = []
    for idx, frame in enumerate(frames):
        drones = frame.get("drone_perception", [])
        covered = np.zeros(n_land, dtype=bool)
        footprints = []
        for drone in drones:
            radius = _float_or_nan(drone.get("footprint"), default=0.0)
            if radius <= 0:
                continue
            footprints.append(radius)
            dx = land_x - _float_or_nan(drone.get("x"), default=0.0)
            dy = land_y - _float_or_nan(drone.get("y"), default=0.0)
            covered |= (dx * dx + dy * dy) <= radius * radius
        steps.append(int(frame.get("step", idx)))
        coverage_pct.append(round(float(100.0 * covered.mean()), 4))
        mean_footprint.append(round(float(np.mean(footprints)) if footprints else 0.0, 4))
        drone_count.append(len(footprints))

    traces = [
        {
            "type": "scatter",
            "mode": "lines",
            "name": "searchable land inside any footprint",
            "x": steps,
            "y": coverage_pct,
            "line": {"color": "#60a5fa", "width": 3},
            "hovertemplate": "step=%{x}<br>coverage=%{y:.2f}%<extra></extra>",
        },
        {
            "type": "scatter",
            "mode": "lines",
            "name": "mean drone footprint radius",
            "x": steps,
            "y": mean_footprint,
            "yaxis": "y2",
            "line": {"color": "#fbbf24", "width": 2, "dash": "dot"},
            "hovertemplate": "step=%{x}<br>mean radius=%{y:.3f}<extra></extra>",
        },
        {
            "type": "scatter",
            "mode": "lines",
            "name": "active drone footprints",
            "x": steps,
            "y": drone_count,
            "yaxis": "y3",
            "line": {"color": "#22c55e", "width": 1.5, "dash": "dash"},
            "hovertemplate": "step=%{x}<br>drones=%{y}<extra></extra>",
        },
    ]
    return _plot(
        "footprint-coverage",
        "Drone Footprint Coverage of Searchable Land",
        traces,
        {
            "xaxis": {"title": "step"},
            "yaxis": {"title": "land cells covered (%)", "range": [0, 100]},
            "yaxis2": {"title": "mean radius", "overlaying": "y", "side": "right"},
            "yaxis3": {
                "title": "drones",
                "overlaying": "y",
                "side": "right",
                "anchor": "free",
                "position": 0.94,
                "showgrid": False,
            },
            "legend": {"orientation": "h"},
        },
        "Coverage excludes water/ocean cells. This is geometric footprint coverage, not guaranteed detection.",
    )


def _summary(path: Path, meta: dict, frames: list[dict], records: list[dict]) -> dict:
    visible = [r for r in records if r["visible"]]
    nonzero = [r for r in records if r["probability"] > 0]
    probabilities = _values(nonzero or records, "probability")
    return {
        "trajectory": str(path),
        "strategy": meta.get("strategy", "unknown"),
        "seed": meta.get("seed"),
        "steps": len(frames),
        "records": len(records),
        "visible_pairs": len(visible),
        "nonzero_probability_pairs": len(nonzero),
        "mean_nonzero_probability": round(_mean(probabilities), 4),
        "max_probability": round(max(probabilities) if probabilities else 0.0, 4),
    }


def _plot(plot_id: str, title: str, data: list[dict], layout: dict, description: str) -> dict:
    base_layout = {
        "title": title,
        "paper_bgcolor": "#10141d",
        "plot_bgcolor": "#10141d",
        "font": {"color": "#e6e9ef"},
        "margin": {"l": 70, "r": 70, "t": 70, "b": 70},
    }
    merged = {**base_layout, **layout}
    return {
        "id": plot_id,
        "title": title,
        "description": description,
        "data": data,
        "layout": merged,
    }


def _build_html(*, summary: dict, plots: list[dict]) -> str:
    cards = "\n".join(
        f"""
<section class="card">
  <div id="{_escape_html(plot['id'])}" class="plot"></div>
  <p>{_escape_html(plot['description'])}</p>
</section>
"""
        for plot in plots
    )
    summary_rows = "\n".join(
        f"<tr><th>{_escape_html(str(key))}</th><td>{_escape_html(str(value))}</td></tr>"
        for key, value in summary.items()
    )
    payload = {"plots": plots}
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Drone Perception EDA</title>
<script src="{PLOTLY_JS}"></script>
<style>
  :root {{
    --bg: #0e1118;
    --panel: #151a23;
    --border: #263043;
    --text: #e6e9ef;
    --muted: #9aa5ba;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0;
    background: var(--bg);
    color: var(--text);
    font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", "Segoe UI", sans-serif;
  }}
  header {{
    padding: 24px 28px 12px;
    border-bottom: 1px solid var(--border);
    background: var(--panel);
  }}
  h1 {{ margin: 0 0 8px; font-size: 24px; }}
  .subtitle {{ color: var(--muted); }}
  main {{ padding: 22px 28px 40px; display: grid; gap: 22px; }}
  .card {{
    border: 1px solid var(--border);
    border-radius: 12px;
    background: var(--panel);
    padding: 16px;
  }}
  .plot {{ width: 100%; height: 520px; }}
  p {{ color: var(--muted); margin: 10px 4px 0; line-height: 1.45; }}
  table {{ border-collapse: collapse; margin-top: 14px; }}
  th, td {{ padding: 6px 16px 6px 0; text-align: left; border-bottom: 1px solid var(--border); }}
  th {{ color: var(--muted); font-weight: 500; }}
</style>
</head>
<body>
<header>
  <h1>Drone Perception EDA</h1>
  <div class="subtitle">Actual analysis plots from exported trajectory perception probabilities.</div>
  <table>{summary_rows}</table>
</header>
<main>
{cards}
</main>
<script>
const payload = {json.dumps(payload, separators=(",", ":"))};
for (const plot of payload.plots) {{
  Plotly.newPlot(plot.id, plot.data, plot.layout, {{responsive: true}});
}}
</script>
</body>
</html>
"""


def _sample_records(records: list[dict], *, max_points: int) -> list[dict]:
    if max_points <= 0 or len(records) <= max_points:
        return records
    idx = np.linspace(0, len(records) - 1, max_points, dtype=int)
    return [records[int(i)] for i in idx]


def _binned_mean(records: list[dict], x_key: str, y_key: str, *, bins: int) -> tuple[list[float], list[float]]:
    xs = np.asarray(_values(records, x_key), dtype=float)
    ys = np.asarray(_values(records, y_key), dtype=float)
    mask = np.isfinite(xs) & np.isfinite(ys)
    xs = xs[mask]
    ys = ys[mask]
    if len(xs) < 2:
        return [], []
    edges = np.linspace(float(xs.min()), float(xs.max()), max(bins, 2) + 1)
    out_x = []
    out_y = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        in_bin = (xs >= lo) & (xs < hi if hi < edges[-1] else xs <= hi)
        if not np.any(in_bin):
            continue
        out_x.append(round(float((lo + hi) / 2), 4))
        out_y.append(round(float(ys[in_bin].mean()), 4))
    return out_x, out_y


def _values(records: Iterable[dict], key: str) -> list[float]:
    values = []
    for record in records:
        value = _float_or_nan(record.get(key))
        if math.isfinite(value):
            values.append(round(value, 4))
    return values


def _float_or_nan(value, *, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else float("nan")


def _round_grid(array: np.ndarray) -> list[list[float]]:
    return [[round(float(value), 4) for value in row] for row in array.tolist()]


def _escape_html(value: str) -> str:
    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


if __name__ == "__main__":
    main()
