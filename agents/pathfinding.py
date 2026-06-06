"""Grid route planning helpers for ground robots."""

from __future__ import annotations

from typing import Sequence
import math

import numpy as np


GridCell = tuple[int, int]


def find_ground_route(
    *,
    traversable,
    movement_cost,
    start: GridCell,
    goal: GridCell,
    fully_connected: bool = True,
) -> list[GridCell]:
    """Find a weighted grid route with NetworkX A*.

    Cells are represented as ``(x, y)`` tuples. ``movement_cost`` is treated as
    cost per cell-distance, so diagonal moves are weighted by ``sqrt(2)``.
    Non-traversable cells and non-finite cost cells are omitted from the graph.
    """

    try:
        import networkx as nx
    except ImportError as exc:
        raise ImportError(
            "Ground-route planning now uses NetworkX. Install it with "
            "`pip install networkx`."
        ) from exc

    traversable_np = _as_numpy(traversable).astype(bool)
    cost_np = _as_numpy(movement_cost).astype(np.float64)
    if traversable_np.shape != cost_np.shape or traversable_np.ndim != 2:
        raise ValueError(
            "traversable and movement_cost must be 2D arrays with matching shapes"
        )

    height, width = traversable_np.shape
    start = _clamp_cell(start, width, height)
    goal = _clamp_cell(goal, width, height)
    if not _is_open(traversable_np, cost_np, start) or not _is_open(traversable_np, cost_np, goal):
        return []

    graph = nx.Graph()
    for y in range(height):
        for x in range(width):
            if not _is_open(traversable_np, cost_np, (x, y)):
                continue
            graph.add_node((x, y))
            for dx, dy, step_len in _neighbor_offsets(fully_connected):
                nx_cell, ny_cell = x + dx, y + dy
                if not (0 <= nx_cell < width and 0 <= ny_cell < height):
                    continue
                if not _is_open(traversable_np, cost_np, (nx_cell, ny_cell)):
                    continue
                if dx != 0 and dy != 0:
                    # A diagonal UGV move cannot squeeze between two blocked
                    # axial neighbors even when both diagonal cells are open.
                    if not _is_open(traversable_np, cost_np, (x + dx, y)):
                        continue
                    if not _is_open(traversable_np, cost_np, (x, y + dy)):
                        continue
                edge_cost = step_len * (cost_np[y, x] + cost_np[ny_cell, nx_cell]) * 0.5
                graph.add_edge((x, y), (nx_cell, ny_cell), weight=float(edge_cost))

    min_cost = _minimum_positive_cost(cost_np, traversable_np)

    def heuristic(cell: GridCell, target: GridCell) -> float:
        return math.hypot(target[0] - cell[0], target[1] - cell[1]) * min_cost

    try:
        return list(nx.astar_path(graph, start, goal, heuristic=heuristic, weight="weight"))
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return []


def _as_numpy(array) -> np.ndarray:
    if hasattr(array, "detach"):
        return array.detach().cpu().numpy()
    return np.asarray(array)


def _clamp_cell(cell: Sequence[int], width: int, height: int) -> GridCell:
    x, y = int(cell[0]), int(cell[1])
    return max(0, min(width - 1, x)), max(0, min(height - 1, y))


def _is_open(traversable: np.ndarray, cost: np.ndarray, cell: GridCell) -> bool:
    x, y = cell
    return bool(traversable[y, x]) and bool(np.isfinite(cost[y, x]))


def _neighbor_offsets(fully_connected: bool) -> tuple[tuple[int, int, float], ...]:
    axial = ((-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0))
    if not fully_connected:
        return axial
    return axial + (
        (-1, -1, math.sqrt(2.0)),
        (-1, 1, math.sqrt(2.0)),
        (1, -1, math.sqrt(2.0)),
        (1, 1, math.sqrt(2.0)),
    )


def _minimum_positive_cost(cost: np.ndarray, traversable: np.ndarray) -> float:
    valid = cost[traversable & np.isfinite(cost)]
    if valid.size == 0:
        return 0.0
    positive = valid[valid > 0.0]
    if positive.size == 0:
        return 0.0
    return float(positive.min())
