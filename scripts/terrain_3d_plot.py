"""Generate an interactive Plotly.js 3D terrain map from a terrain cache.

Example:

    python scripts/terrain_3d_plot.py \
      --terrain-cache data/terrain_cache/big_sur_128.npz \
      --out results/eda/big_sur_3d.html

    python scripts/terrain_3d_plot.py \
      --terrain-cache data/terrain_cache/big_sur_128.npz \
      --trajectory web/trajectories/lawnmower.json \
      --out results/eda/big_sur_lawnmower_3d.html
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


LAND_COVER_NAMES = ["road", "open", "brush", "forest", "rock", "water"]
LAND_COVER_COLORS = ["#b59665", "#47783d", "#315a2e", "#203d24", "#555963", "#2563eb"]
DRONE_FLIGHT_COLORS = ["#3b82f6", "#22d3ee", "#a855f7", "#f59e0b", "#ef4444", "#10b981"]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--terrain-cache", required=True, help="Path to a terrain .npz cache.")
    p.add_argument("--trajectory", default=None, help="Optional trajectory JSON with drone flight paths.")
    p.add_argument("--out", default="results/eda/terrain_3d.html", help="Output .html path.")
    p.add_argument("--vertical-exaggeration", type=float, default=4.0)
    p.add_argument("--flight-stride", type=int, default=1, help="Use every Nth trajectory frame for flight paths.")
    p.add_argument(
        "--color-by",
        choices=("land_cover", "elevation", "slope", "fuel_density", "moisture", "rockiness"),
        default="land_cover",
    )
    p.add_argument(
        "--no-flatten-water",
        action="store_true",
        help="Keep cached water elevations instead of drawing water as a flat low plane.",
    )
    p.add_argument(
        "--hide-obstacles",
        action="store_true",
        help="Do not overlay tree/house obstacle markers.",
    )
    args = p.parse_args()

    generate_terrain_3d_html(
        terrain_cache=Path(args.terrain_cache),
        trajectory=Path(args.trajectory) if args.trajectory else None,
        out=Path(args.out),
        vertical_exaggeration=float(args.vertical_exaggeration),
        color_by=args.color_by,
        flight_stride=max(int(args.flight_stride), 1),
        flatten_water=not args.no_flatten_water,
        show_obstacles=not args.hide_obstacles,
    )


def generate_terrain_3d_html(
    *,
    terrain_cache: Path,
    out: Path,
    trajectory: Path | None = None,
    vertical_exaggeration: float = 4.0,
    color_by: str = "land_cover",
    flight_stride: int = 1,
    flatten_water: bool = True,
    show_obstacles: bool = True,
) -> Path:
    cache_path = Path(terrain_cache)
    out_path = Path(out)
    terrain = _load_cache(cache_path)
    trajectory_payload = _load_trajectory(Path(trajectory)) if trajectory is not None else None
    html = _build_html(
        terrain=terrain,
        trajectory=trajectory_payload,
        title=cache_path.name,
        vertical_exaggeration=float(vertical_exaggeration),
        color_by=color_by,
        flight_stride=max(int(flight_stride), 1),
        flatten_water=flatten_water,
        show_obstacles=show_obstacles,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html)
    print(f"Wrote {out_path}")
    return out_path


def _load_cache(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"terrain cache not found: {path}")
    with np.load(path, allow_pickle=False) as data:
        required = ("land_cover", "elevation", "slope", "fuel_density", "moisture", "rockiness")
        missing = [name for name in required if name not in data]
        if missing:
            raise SystemExit(f"terrain cache is missing arrays: {', '.join(missing)}")
        terrain = {name: np.asarray(data[name]) for name in required}
        terrain["obstacle_type"] = np.asarray(data["obstacle_type"]) if "obstacle_type" in data else None
        terrain["obstacle_height"] = np.asarray(data["obstacle_height"]) if "obstacle_height" in data else None
        terrain["source"] = str(data["source"].item()) if "source" in data else str(path)
    return terrain


def _load_trajectory(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"trajectory not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    if not isinstance(payload.get("metadata"), dict):
        raise SystemExit(f"trajectory is missing metadata: {path}")
    if not isinstance(payload.get("frames"), list):
        raise SystemExit(f"trajectory is missing frames: {path}")
    return payload


def _build_html(
    *,
    terrain: dict,
    trajectory: dict | None,
    title: str,
    vertical_exaggeration: float,
    color_by: str,
    flight_stride: int,
    flatten_water: bool,
    show_obstacles: bool,
) -> str:
    land_cover = terrain["land_cover"].astype(int)
    elevation = _finite(terrain["elevation"])
    z = elevation.copy()
    if flatten_water:
        land = land_cover != 5
        low = float(np.nanmin(z[land])) if np.any(land) else float(np.nanmin(z))
        z = np.where(land_cover == 5, low, z)
    z = z * max(vertical_exaggeration, 0.0)

    grid_size = z.shape[0]
    world = _trajectory_world(trajectory)
    x_semidim = world.get("x_semidim", 1.0)
    y_semidim = world.get("y_semidim", 1.0)
    x = np.linspace(-x_semidim, x_semidim, grid_size)
    y = np.linspace(-y_semidim, y_semidim, grid_size)
    surface_color = _surface_color(terrain, color_by)
    surface = {
        "type": "surface",
        "x": _round_list(x),
        "y": _round_list(y),
        "z": _round_grid(z),
        "surfacecolor": _round_grid(surface_color),
        "colorscale": _colorscale(color_by),
        "cmin": 0.0 if color_by == "land_cover" else float(np.nanmin(surface_color)),
        "cmax": 5.0 if color_by == "land_cover" else float(np.nanmax(surface_color)),
        "colorbar": _colorbar(color_by),
        "showscale": True,
        "lighting": {"ambient": 0.62, "diffuse": 0.72, "roughness": 0.85, "specular": 0.08},
        "hovertemplate": "x=%{x:.2f}<br>y=%{y:.2f}<br>z=%{z:.3f}<extra></extra>",
    }
    traces = [surface]
    if show_obstacles:
        traces.extend(_obstacle_traces(terrain, x, y, z))
    if trajectory is not None:
        traces.extend(_drone_flight_traces(trajectory, vertical_exaggeration, flight_stride))

    land_cells = int(np.count_nonzero(land_cover != 5))
    water_cells = int(np.count_nonzero(land_cover == 5))
    source = terrain.get("source", title)
    strategy = _trajectory_strategy(trajectory)
    flight_subtitle = f"<br>drone flight: {strategy}" if strategy else ""
    layout = {
        "title": {
            "text": (
                f"{title} - 3D terrain ({color_by})"
                f"<br><sup>{source}{flight_subtitle}"
                f"<br>land cells: {land_cells}, water cells: {water_cells}</sup>"
            )
        },
        "scene": {
            "xaxis": {"title": "world x"},
            "yaxis": {"title": "world y"},
            "zaxis": {"title": f"height x {vertical_exaggeration:g}"},
            "aspectmode": "manual",
            "aspectratio": {"x": 1.0, "y": 1.0, "z": 0.35},
            "camera": {"eye": {"x": 1.35, "y": -1.55, "z": 0.85}},
        },
        "margin": {"l": 0, "r": 0, "t": 80, "b": 0},
        "paper_bgcolor": "#0e1118",
        "plot_bgcolor": "#0e1118",
        "font": {"color": "#e6e9ef"},
    }
    payload = {"data": traces, "layout": layout}
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>{_escape_html(title)} 3D terrain</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
  html, body, #plot {{ width: 100%; height: 100%; margin: 0; background: #0e1118; }}
</style>
</head>
<body>
<div id="plot"></div>
<script>
const payload = {json.dumps(payload, separators=(",", ":"))};
Plotly.newPlot('plot', payload.data, payload.layout, {{responsive: true}});
</script>
</body>
</html>
"""


def _surface_color(terrain: dict, color_by: str) -> np.ndarray:
    if color_by == "land_cover":
        return terrain["land_cover"].astype(float)
    return _finite(terrain[color_by])


def _colorscale(color_by: str):
    if color_by != "land_cover":
        return "Viridis"
    max_value = len(LAND_COVER_COLORS) - 1
    stops = []
    for idx, color in enumerate(LAND_COVER_COLORS):
        lo = max(0.0, (idx - 0.5) / max_value)
        hi = min(1.0, (idx + 0.5) / max_value)
        stops.append([lo, color])
        stops.append([hi, color])
    return stops


def _colorbar(color_by: str) -> dict:
    if color_by != "land_cover":
        return {"title": color_by}
    return {
        "title": "land cover",
        "tickmode": "array",
        "tickvals": list(range(len(LAND_COVER_NAMES))),
        "ticktext": LAND_COVER_NAMES,
    }


def _obstacle_traces(terrain: dict, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> list[dict]:
    obstacle_type = terrain.get("obstacle_type")
    obstacle_height = terrain.get("obstacle_height")
    if obstacle_type is None or obstacle_height is None:
        return []

    traces = []
    for value, name, color, size in (
        (1, "tree canopy", "#1d5128", 2.8),
        (2, "house", "#a34d3d", 3.6),
    ):
        ys, xs = (obstacle_type == value).nonzero()
        if len(xs) == 0:
            continue
        marker_z = z[ys, xs] + np.maximum(obstacle_height[ys, xs], 0.02)
        traces.append({
            "type": "scatter3d",
            "mode": "markers",
            "name": name,
            "x": _round_list(x[xs]),
            "y": _round_list(y[ys]),
            "z": _round_list(marker_z),
            "marker": {"size": size, "color": color, "opacity": 0.86},
        })
    return traces


def _drone_flight_traces(trajectory: dict, vertical_exaggeration: float, flight_stride: int) -> list[dict]:
    frames = trajectory.get("frames", [])
    scale = max(float(vertical_exaggeration), 0.0)
    paths: dict[str, dict[str, list[float]]] = {}

    for frame_index, frame in enumerate(frames):
        if frame_index % flight_stride != 0 and frame_index != len(frames) - 1:
            continue
        step = int(frame.get("step", frame_index))
        for agent in frame.get("agents", []):
            if agent.get("type") != "drone":
                continue
            name = str(agent.get("name", f"drone_{len(paths)}"))
            altitude_msl = agent.get("altitude_msl", agent.get("altitude", 0.0))
            altitude_agl = agent.get("altitude_agl", agent.get("altitude", 0.0))
            paths.setdefault(name, {"x": [], "y": [], "z": [], "step": [], "msl": [], "agl": []})
            paths[name]["x"].append(float(agent.get("x", 0.0)))
            paths[name]["y"].append(float(agent.get("y", 0.0)))
            paths[name]["z"].append(float(altitude_msl) * scale)
            paths[name]["step"].append(float(step))
            paths[name]["msl"].append(float(altitude_msl))
            paths[name]["agl"].append(float(altitude_agl))

    traces = []
    for index, (name, path) in enumerate(sorted(paths.items())):
        if len(path["x"]) < 2:
            continue
        color = DRONE_FLIGHT_COLORS[index % len(DRONE_FLIGHT_COLORS)]
        traces.append({
            "type": "scatter3d",
            "mode": "lines+markers",
            "name": f"{name} flight",
            "x": _round_list(np.asarray(path["x"])),
            "y": _round_list(np.asarray(path["y"])),
            "z": _round_list(np.asarray(path["z"])),
            "customdata": [
                [int(step), round(msl, 4), round(agl, 4)]
                for step, msl, agl in zip(path["step"], path["msl"], path["agl"])
            ],
            "line": {"color": color, "width": 6},
            "marker": {"size": 2.2, "color": color, "opacity": 0.92},
            "hovertemplate": (
                f"{_escape_html(name)}"
                "<br>step=%{customdata[0]}"
                "<br>x=%{x:.3f}<br>y=%{y:.3f}"
                "<br>altitude MSL=%{customdata[1]:.3f}"
                "<br>altitude AGL=%{customdata[2]:.3f}<extra></extra>"
            ),
        })
        traces.extend(_flight_endpoint_traces(name, path, color))
    return traces


def _flight_endpoint_traces(name: str, path: dict[str, list[float]], color: str) -> list[dict]:
    return [
        {
            "type": "scatter3d",
            "mode": "markers",
            "name": f"{name} start",
            "x": [round(float(path["x"][0]), 5)],
            "y": [round(float(path["y"][0]), 5)],
            "z": [round(float(path["z"][0]), 5)],
            "marker": {"size": 5.5, "color": "#ffffff", "line": {"color": color, "width": 2}},
            "showlegend": False,
            "hovertemplate": f"{_escape_html(name)} start<extra></extra>",
        },
        {
            "type": "scatter3d",
            "mode": "markers",
            "name": f"{name} end",
            "x": [round(float(path["x"][-1]), 5)],
            "y": [round(float(path["y"][-1]), 5)],
            "z": [round(float(path["z"][-1]), 5)],
            "marker": {"size": 6.5, "color": color, "symbol": "diamond"},
            "showlegend": False,
            "hovertemplate": f"{_escape_html(name)} end<extra></extra>",
        },
    ]


def _trajectory_world(trajectory: dict | None) -> dict[str, float]:
    if trajectory is None:
        return {"x_semidim": 1.0, "y_semidim": 1.0}
    world = trajectory.get("metadata", {}).get("world", {})
    return {
        "x_semidim": float(world.get("x_semidim", 1.0)),
        "y_semidim": float(world.get("y_semidim", 1.0)),
    }


def _trajectory_strategy(trajectory: dict | None) -> str | None:
    if trajectory is None:
        return None
    metadata = trajectory.get("metadata", {})
    strategy = metadata.get("strategy")
    seed = metadata.get("seed")
    steps = metadata.get("actual_n_steps", metadata.get("n_steps"))
    pieces = [str(strategy)] if strategy else []
    if seed is not None:
        pieces.append(f"seed {seed}")
    if steps is not None:
        pieces.append(f"{steps} steps")
    return ", ".join(pieces) if pieces else "trajectory"


def _finite(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array, dtype=float)
    fill = float(np.nanmean(array)) if np.isfinite(array).any() else 0.0
    return np.nan_to_num(array, nan=fill, posinf=fill, neginf=fill)


def _round_grid(array: np.ndarray) -> list[list[float]]:
    return [[round(float(value), 5) for value in row] for row in array.tolist()]


def _round_list(array: np.ndarray) -> list[float]:
    return [round(float(value), 5) for value in np.asarray(array).tolist()]


def _escape_html(value: str) -> str:
    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


if __name__ == "__main__":
    main()
