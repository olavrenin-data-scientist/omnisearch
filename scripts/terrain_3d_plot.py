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
    p.add_argument("--vertical-exaggeration", type=float, default=1.0)
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
        help="Do not overlay tree canopy markers and house footprint/roof meshes.",
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
    vertical_exaggeration: float = 1.0,
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
        traces.extend(_obstacle_traces(terrain, x, y, z, vertical_exaggeration))
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
            "zaxis": {"title": _z_axis_title(vertical_exaggeration)},
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


def _obstacle_traces(
    terrain: dict,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    vertical_exaggeration: float,
) -> list[dict]:
    obstacle_type = terrain.get("obstacle_type")
    obstacle_height = terrain.get("obstacle_height")
    if obstacle_type is None or obstacle_height is None:
        return []

    traces = []
    height_scale = max(float(vertical_exaggeration), 0.0)
    tree_trace = _tree_canopy_trace(obstacle_type, obstacle_height, x, y, z, height_scale)
    if tree_trace is not None:
        traces.append(tree_trace)
    traces.extend(_house_mesh_traces(obstacle_type, obstacle_height, x, y, z, height_scale))
    return traces


def _tree_canopy_trace(
    obstacle_type: np.ndarray,
    obstacle_height: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    height_scale: float,
) -> dict | None:
    ys, xs = (obstacle_type == 1).nonzero()
    if len(xs) == 0:
        return None

    marker_z = z[ys, xs] + np.maximum(obstacle_height[ys, xs] * height_scale, 0.02 * height_scale)
    return {
        "type": "scatter3d",
        "mode": "markers",
        "name": "tree canopy",
        "x": _round_list(x[xs]),
        "y": _round_list(y[ys]),
        "z": _round_list(marker_z),
        "marker": {"size": 2.8, "color": "#1d5128", "opacity": 0.82},
        "hovertemplate": "tree canopy<br>x=%{x:.2f}<br>y=%{y:.2f}<br>z=%{z:.3f}<extra></extra>",
    }


def _house_mesh_traces(
    obstacle_type: np.ndarray,
    obstacle_height: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    height_scale: float,
) -> list[dict]:
    house_mask = obstacle_type == 2
    if not np.any(house_mask):
        return []

    x_edges = _cell_edges(x)
    y_edges = _cell_edges(y)
    base_lift = 0.0002 * height_scale
    min_roof_height = 0.0008 * height_scale
    house_height = np.maximum(obstacle_height * height_scale, min_roof_height)

    footprints = _empty_mesh()
    roofs = _empty_mesh()
    walls = _empty_mesh()
    for row, col in np.argwhere(house_mask):
        x0 = float(x_edges[col])
        x1 = float(x_edges[col + 1])
        y0 = float(y_edges[row])
        y1 = float(y_edges[row + 1])
        base_z = float(z[row, col] + base_lift)
        roof_z = float(base_z + house_height[row, col])

        _add_horizontal_quad(footprints, x0, x1, y0, y1, base_z)
        _add_horizontal_quad(roofs, x0, x1, y0, y1, roof_z)
        _add_exposed_house_walls(walls, house_mask, row, col, x0, x1, y0, y1, base_z, roof_z)

    traces = []
    if footprints["x"]:
        traces.append(_mesh_trace(footprints, "house footprints", "#6f342c", 0.46))
    if walls["x"]:
        traces.append(_mesh_trace(walls, "house walls", "#8e493e", 0.74))
    if roofs["x"]:
        traces.append(_mesh_trace(roofs, "house roofs", "#b45a48", 0.90))
    return traces


def _cell_edges(coords: np.ndarray) -> np.ndarray:
    coords = np.asarray(coords, dtype=float)
    if len(coords) == 0:
        return np.asarray([0.0, 1.0], dtype=float)
    if len(coords) == 1:
        return np.asarray([coords[0] - 0.5, coords[0] + 0.5], dtype=float)
    mids = 0.5 * (coords[:-1] + coords[1:])
    first = coords[0] - (mids[0] - coords[0])
    last = coords[-1] + (coords[-1] - mids[-1])
    return np.concatenate(([first], mids, [last]))


def _empty_mesh() -> dict[str, list[float] | list[int]]:
    return {"x": [], "y": [], "z": [], "i": [], "j": [], "k": []}


def _add_horizontal_quad(mesh: dict, x0: float, x1: float, y0: float, y1: float, z_value: float) -> None:
    start = len(mesh["x"])
    mesh["x"].extend([x0, x1, x1, x0])
    mesh["y"].extend([y0, y0, y1, y1])
    mesh["z"].extend([z_value, z_value, z_value, z_value])
    mesh["i"].extend([start, start])
    mesh["j"].extend([start + 1, start + 2])
    mesh["k"].extend([start + 2, start + 3])


def _add_exposed_house_walls(
    mesh: dict,
    house_mask: np.ndarray,
    row: int,
    col: int,
    x0: float,
    x1: float,
    y0: float,
    y1: float,
    base_z: float,
    roof_z: float,
) -> None:
    max_row, max_col = house_mask.shape[0] - 1, house_mask.shape[1] - 1
    if row == 0 or not house_mask[row - 1, col]:
        _add_vertical_quad(mesh, x0, x1, y0, y0, base_z, roof_z)
    if row == max_row or not house_mask[row + 1, col]:
        _add_vertical_quad(mesh, x1, x0, y1, y1, base_z, roof_z)
    if col == 0 or not house_mask[row, col - 1]:
        _add_vertical_quad(mesh, x0, x0, y1, y0, base_z, roof_z)
    if col == max_col or not house_mask[row, col + 1]:
        _add_vertical_quad(mesh, x1, x1, y0, y1, base_z, roof_z)


def _add_vertical_quad(
    mesh: dict,
    x0: float,
    x1: float,
    y0: float,
    y1: float,
    base_z: float,
    roof_z: float,
) -> None:
    start = len(mesh["x"])
    mesh["x"].extend([x0, x1, x1, x0])
    mesh["y"].extend([y0, y1, y1, y0])
    mesh["z"].extend([base_z, base_z, roof_z, roof_z])
    mesh["i"].extend([start, start])
    mesh["j"].extend([start + 1, start + 2])
    mesh["k"].extend([start + 2, start + 3])


def _mesh_trace(mesh: dict, name: str, color: str, opacity: float) -> dict:
    return {
        "type": "mesh3d",
        "name": name,
        "x": _round_list(np.asarray(mesh["x"], dtype=float)),
        "y": _round_list(np.asarray(mesh["y"], dtype=float)),
        "z": _round_list(np.asarray(mesh["z"], dtype=float)),
        "i": mesh["i"],
        "j": mesh["j"],
        "k": mesh["k"],
        "color": color,
        "opacity": opacity,
        "flatshading": True,
        "lighting": {"ambient": 0.65, "diffuse": 0.72, "roughness": 0.88, "specular": 0.06},
        "hovertemplate": f"{name}<br>x=%{{x:.2f}}<br>y=%{{y:.2f}}<br>z=%{{z:.3f}}<extra></extra>",
    }


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


def _z_axis_title(vertical_exaggeration: float) -> str:
    if abs(float(vertical_exaggeration) - 1.0) < 1e-6:
        return "height (true scale)"
    return f"height x {vertical_exaggeration:g}"


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
