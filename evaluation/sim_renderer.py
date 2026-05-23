"""
Synthetic UAV view of a VMAS WildfireSearchScenario.

VMAS renders abstract 2D circles, but our detection pipeline expects real
camera frames. This module bridges the two: take the scenario's internal
state (fire grid, survivor positions, agent positions) and paint a
bird's-eye image that:

  - looks like an aerial forest photo to a fire detector (orange where
    fire cells burn, dark green elsewhere)
  - contains real person sprites at survivor positions so YOLOv8 will
    actually fire on them (a colored circle wouldn't be recognised as
    a person)

This is a deliberate proxy for what a deployed UAV camera feed would
deliver — it's not photoreal, but it's adversarial to the same detectors
the production system uses. So the closed-loop demo it powers is a
genuine validation of the perception pipeline, not just a tautology.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image

from envs.wildfire_search import WildfireSearchScenario, X, Y


# Colors in RGB
FOREST  = (40, 78, 38)         # dark forest green
FIRE    = (240, 110, 30)       # saturated orange — gets detected by HSV fire detector
DRONE   = (60, 130, 250)       # blue marker
GROUND  = (90, 200, 90)        # green marker (different shade from forest)


@lru_cache(maxsize=1)
def _person_sprite(target_height_px: int) -> Image.Image:
    """
    A real person crop, scaled to `target_height_px` tall. Cached.
    Cropped from Ultralytics' bundled bus.jpg so YOLOv8 will detect it.
    """
    from ultralytics.utils import ASSETS
    bus = Image.open(ASSETS / "bus.jpg").convert("RGB")
    # The leftmost person box from our earlier detection runs:
    # (48, 398, 245, 902) — 197×504 in the source.
    person = bus.crop((48, 398, 245, 902))
    aspect = person.width / person.height
    new_h = target_height_px
    new_w = max(int(round(target_height_px * aspect)), 4)
    return person.resize((new_w, new_h), Image.LANCZOS)


@dataclass
class GroundTruth:
    """Per-frame ground truth from the simulator — used to score detections."""
    survivor_pixel_boxes: List[Tuple[int, int, int, int]]
    fire_pixel_boxes:     List[Tuple[int, int, int, int]]
    image_size:           Tuple[int, int]   # (H, W)


class UAVRenderer:
    """
    Renders a single env's state as a synthetic UAV top-down image.

    Parameters
    ----------
    size :
        Output image side (square). Default 512.
    person_sprite_height :
        Rendered survivor height in pixels (smaller = harder for YOLOv8).
    show_agents :
        Paint drone/ground markers on top. Off by default so the detector
        isn't confused by extra colored shapes (drones aren't survivors).
    """

    def __init__(
        self,
        size: int = 512,
        person_sprite_height: int = 90,
        show_agents: bool = False,
    ):
        self.size = size
        self.person_sprite_height = person_sprite_height
        self.show_agents = show_agents

    # ------------------------------------------------------------------
    # Render
    # ------------------------------------------------------------------
    def render(
        self,
        scenario: WildfireSearchScenario,
        env_index: int = 0,
    ) -> Tuple[np.ndarray, GroundTruth]:
        """Return (image_rgb, ground_truth) for the given env index."""
        H = W = self.size
        img = np.full((H, W, 3), FOREST, dtype=np.uint8)

        # --- Paint fire cells ---
        fire_grid = scenario.fire_grid[env_index].cpu().numpy()      # (G, G) bool
        G = scenario.fire_grid_size
        cell_h = H / G
        cell_w = W / G
        fire_boxes: List[Tuple[int, int, int, int]] = []
        ys, xs = np.where(fire_grid)
        for gy, gx in zip(ys, xs):
            # Note: image-y is flipped vs world-y, but the fire grid maps
            # world (gx, gy) → grid indices the same way the scenario does.
            x1 = int(round(gx * cell_w));  x2 = int(round((gx + 1) * cell_w))
            y1 = int(round((G - 1 - gy) * cell_h));  y2 = int(round((G - gy) * cell_h))
            img[y1:y2, x1:x2] = FIRE
            fire_boxes.append((x1, y1, x2, y2))

        # --- Paste survivor sprites ---
        survivor_boxes: List[Tuple[int, int, int, int]] = []
        sprite = _person_sprite(self.person_sprite_height)
        sprite_np = np.asarray(sprite)
        sh, sw = sprite_np.shape[:2]
        pil_img = Image.fromarray(img)
        for surv in scenario._survivors:
            wx, wy = self._world_xy(surv, env_index)
            px, py = self._world_to_pixel(wx, wy)
            # Center sprite horizontally, anchor feet at the point
            x1 = px - sw // 2
            y1 = py - sh
            pil_img.paste(sprite, (x1, y1))
            survivor_boxes.append((max(x1, 0), max(y1, 0),
                                   min(x1 + sw, W), min(y1 + sh, H)))
        img = np.asarray(pil_img).copy()

        # --- Optional agent markers ---
        if self.show_agents:
            for a in scenario.world.agents:
                wx, wy = self._world_xy(a, env_index)
                px, py = self._world_to_pixel(wx, wy)
                color = DRONE if getattr(a, "is_drone", False) else GROUND
                self._draw_disc(img, px, py, r=6, color=color)

        return img, GroundTruth(
            survivor_pixel_boxes=survivor_boxes,
            fire_pixel_boxes=fire_boxes,
            image_size=(H, W),
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _world_xy(self, entity, env_index: int) -> Tuple[float, float]:
        pos = entity.state.pos[env_index]
        return float(pos[X]), float(pos[Y])

    def _world_to_pixel(self, wx: float, wy: float) -> Tuple[int, int]:
        # World ∈ [-1, 1]; flip y so up-in-world is up-in-image.
        u = (wx + 1.0) / 2.0
        v = (1.0 - (wy + 1.0) / 2.0)
        return int(round(u * (self.size - 1))), int(round(v * (self.size - 1)))

    @staticmethod
    def _draw_disc(img: np.ndarray, cx: int, cy: int, r: int, color):
        H, W = img.shape[:2]
        ys, xs = np.ogrid[:H, :W]
        m = (xs - cx) ** 2 + (ys - cy) ** 2 <= r * r
        img[m] = color


def render_uav_view(scenario, env_index: int = 0, **kwargs) -> Tuple[np.ndarray, GroundTruth]:
    """Convenience wrapper for one-shot renders."""
    return UAVRenderer(**kwargs).render(scenario, env_index)
