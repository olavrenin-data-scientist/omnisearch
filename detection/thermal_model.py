"""Simulated thermal infrared (TIR) sensor model for wildfire SAR.

Models the physics of thermal detection in wildfire environments:
- Smoke penetration (TIR sees through light/moderate smoke)
- Thermal crossover near active fire (ΔT → 0 when ambient ≈ body temp)
- Altitude-dependent thermal resolution
- Distance-based detection probability within footprint

This is a *simulated* sensor — no real thermal imagery is generated.
Detection decisions are stochastic, based on physical parameters.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

import numpy as np


@dataclass
class ThermalSensorConfig:
    """Configuration for the simulated thermal sensor."""

    # Human body temperature (Kelvin)
    body_temp_k: float = 310.0  # ~37°C

    # Ambient temperature without fire (Kelvin)
    ambient_temp_k: float = 293.0  # ~20°C

    # Thermal sensitivity (NETD) — minimum ΔT detectable (Kelvin)
    netd_k: float = 0.05  # 50 mK, typical uncooled LWIR

    # Sensor resolution (pixels)
    sensor_resolution: int = 640  # 640x512 typical

    # Sensor horizontal FOV (degrees)
    sensor_fov_deg: float = 40.0  # Narrower than RGB (typical TIR lens)

    # Base detection probability at optimal ΔT (>10K contrast, close range)
    base_detection_prob: float = 0.92

    # Smoke transmittance coefficient (Beer-Lambert, per unit smoke_load)
    # TIR penetrates smoke better than visible: lower attenuation coefficient
    smoke_attenuation_coeff: float = 0.4  # vs ~1.4 for visible

    # Smoke transmittance floor (even in heavy smoke, some TIR gets through)
    smoke_transmittance_floor: float = 0.70

    # Fire thermal radiation model
    # Ground temperature near active fire (Kelvin) — makes ΔT vanish
    fire_ground_temp_k: float = 450.0  # ~177°C near active flame front

    # Burned ground temperature (cooling after fire passes)
    burned_ground_temp_k: float = 330.0  # ~57°C — close to body temp!

    # Thermal crossover threshold: if |ΔT| < this, detection fails
    crossover_threshold_k: float = 2.0

    # Altitude quality curve (same operational meaning as RGB)
    min_altitude_m: float = 20.0
    max_altitude_m: float = 50.0
    quality_at_min_alt: float = 0.95
    quality_at_max_alt: float = 0.65  # TIR has lower resolution than RGB

    # False positive rate (warm animals, heated rocks, embers)
    false_positive_rate: float = 0.08

    seed: int = 42


class ThermalSensorModel:
    """Simulated thermal infrared sensor for survivor detection.

    Computes detection probability based on thermal contrast (ΔT),
    smoke conditions, fire proximity, and altitude. Does NOT generate
    images — operates on simulator state directly.
    """

    def __init__(self, config: ThermalSensorConfig | None = None):
        self.config = config or ThermalSensorConfig()
        self.rng = random.Random(self.config.seed)
        self._frame_count = 0

    def reset(self):
        """Reset state between episodes."""
        self._frame_count = 0

    def detect_survivors(
        self,
        *,
        drone_xy: tuple[float, float],
        drone_altitude_m: float,
        fov_deg: float,
        survivors: list[dict],
        fire_grid: np.ndarray | None = None,
        fire_intensity_grid: np.ndarray | None = None,
        burned_grid: np.ndarray | None = None,
        smoke_grid: np.ndarray | None = None,
        sim_units_per_meter: float = 1.0,
        grid_size: int = 100,
    ) -> list[dict]:
        """Run thermal detection on all survivors within the sensor footprint.

        Parameters
        ----------
        drone_xy : world coordinates of drone
        drone_altitude_m : altitude above ground in meters
        fov_deg : sensor FOV in degrees
        survivors : list of dicts with keys 'index', 'world_xy'
        fire_grid, fire_intensity_grid, burned_grid, smoke_grid : simulation grids
        sim_units_per_meter : conversion factor
        grid_size : grid dimension for coordinate mapping

        Returns
        -------
        List of detection dicts with thermal-specific metadata.
        """
        self._frame_count += 1
        cfg = self.config

        footprint_m = 2.0 * drone_altitude_m * math.tan(math.radians(fov_deg) / 2.0)
        footprint_world = footprint_m * sim_units_per_meter

        detections = []

        for survivor in survivors:
            sx, sy = survivor["world_xy"]
            dx = sx - drone_xy[0]
            dy = sy - drone_xy[1]
            dist_world = math.sqrt(dx * dx + dy * dy)

            # Check if survivor is within thermal sensor footprint
            if dist_world > footprint_world * 0.5:
                continue

            # Compute local environmental conditions at survivor position
            local_fire = self._sample_grid_at(
                fire_intensity_grid, sx, sy, sim_units_per_meter, grid_size
            )
            local_burned = self._sample_grid_at(
                burned_grid, sx, sy, sim_units_per_meter, grid_size
            )
            local_smoke = self._sample_grid_at(
                smoke_grid, sx, sy, sim_units_per_meter, grid_size
            )

            # Compute effective ground temperature at survivor location
            ground_temp = self._effective_ground_temp(local_fire, local_burned)

            # Compute thermal contrast (ΔT)
            delta_t = abs(cfg.body_temp_k - ground_temp)

            # Detection probability computation
            prob = self._compute_detection_prob(
                delta_t=delta_t,
                smoke_load=local_smoke,
                distance_fraction=dist_world / max(footprint_world * 0.5, 1e-6),
                altitude_m=drone_altitude_m,
            )

            detected = self.rng.random() < prob

            detection_record = {
                "survivor_index": survivor["index"],
                "true_world_xy": [sx, sy],
                "detected": detected,
                "detection_probability": round(prob, 4),
                "delta_t_k": round(delta_t, 2),
                "ground_temp_k": round(ground_temp, 1),
                "local_fire_intensity": round(local_fire, 3),
                "local_burned": round(local_burned, 3),
                "local_smoke": round(local_smoke, 3),
                "thermal_crossover": delta_t < cfg.crossover_threshold_k,
                "confidence": round(self._thermal_confidence(delta_t, prob), 4),
                "sensor": "thermal",
            }

            if detected:
                # Add position estimate with thermal-specific noise
                pos_noise_m = self._position_noise(drone_altitude_m, delta_t)
                est_x = sx + self.rng.gauss(0, pos_noise_m) * sim_units_per_meter
                est_y = sy + self.rng.gauss(0, pos_noise_m) * sim_units_per_meter
                detection_record["estimated_world_xy"] = [round(est_x, 6), round(est_y, 6)]

            detections.append(detection_record)

        # False positives from warm objects (animals, embers, heated rocks)
        n_fp = self._generate_false_positives(footprint_world, drone_xy, fire_grid,
                                               sim_units_per_meter, grid_size)
        for fp in n_fp:
            detections.append(fp)

        return detections

    def _effective_ground_temp(self, fire_intensity: float, burned_fraction: float) -> float:
        """Compute effective ground temperature based on fire/burn state."""
        cfg = self.config

        # Linear interpolation: ambient → fire_temp based on fire intensity
        fire_contribution = fire_intensity * (cfg.fire_ground_temp_k - cfg.ambient_temp_k)
        # Burned areas are warm but cooling
        burn_contribution = burned_fraction * (cfg.burned_ground_temp_k - cfg.ambient_temp_k)

        # Take the maximum effect (fire dominates where active)
        ground_temp = cfg.ambient_temp_k + max(fire_contribution, burn_contribution)
        return ground_temp

    def _compute_detection_prob(
        self,
        *,
        delta_t: float,
        smoke_load: float,
        distance_fraction: float,
        altitude_m: float,
    ) -> float:
        """Compute thermal detection probability from physical factors."""
        cfg = self.config

        # Factor 1: Thermal contrast
        # If ΔT < crossover threshold → detection nearly impossible
        if delta_t < cfg.crossover_threshold_k:
            contrast_factor = 0.05  # Near-zero but not absolute zero
        elif delta_t < 5.0:
            # Poor contrast zone (2-5K): linear ramp
            contrast_factor = 0.05 + 0.65 * (delta_t - cfg.crossover_threshold_k) / (5.0 - cfg.crossover_threshold_k)
        elif delta_t < 10.0:
            # Moderate contrast (5-10K)
            contrast_factor = 0.70 + 0.25 * (delta_t - 5.0) / 5.0
        else:
            # Good contrast (>10K)
            contrast_factor = 0.95

        # Factor 2: Smoke transmittance (TIR penetrates smoke well)
        smoke_transmittance = max(
            math.exp(-cfg.smoke_attenuation_coeff * smoke_load),
            cfg.smoke_transmittance_floor,
        )

        # Factor 3: Distance within footprint (quadratic falloff)
        distance_factor = max(0.2, 1.0 - 0.8 * distance_fraction ** 2)

        # Factor 4: Altitude quality (thermal resolution degrades with altitude)
        alt_frac = (altitude_m - cfg.min_altitude_m) / max(
            cfg.max_altitude_m - cfg.min_altitude_m, 1e-6
        )
        alt_frac = max(0.0, min(1.0, alt_frac))
        altitude_quality = cfg.quality_at_min_alt + alt_frac * (
            cfg.quality_at_max_alt - cfg.quality_at_min_alt
        )

        prob = cfg.base_detection_prob * contrast_factor * smoke_transmittance * distance_factor * altitude_quality
        return max(0.0, min(1.0, prob))

    def _thermal_confidence(self, delta_t: float, detection_prob: float) -> float:
        """Estimate confidence score for a thermal detection."""
        # Confidence correlates with ΔT — higher contrast = more certain
        base_conf = min(1.0, delta_t / 15.0) * 0.7 + 0.3 * detection_prob
        jitter = self.rng.gauss(0, 0.05)
        return max(0.1, min(0.99, base_conf + jitter))

    def _position_noise(self, altitude_m: float, delta_t: float) -> float:
        """Compute position estimation noise in meters.

        Thermal has lower resolution than RGB, so position noise is higher.
        Also worse when ΔT is low (blob is diffuse).
        """
        base_noise = 1.5 * (altitude_m / 30.0)  # ~1.5m at 30m altitude
        contrast_penalty = max(1.0, 5.0 / max(delta_t, 0.5))
        return base_noise * contrast_penalty

    def _generate_false_positives(
        self,
        footprint_world: float,
        drone_xy: tuple[float, float],
        fire_grid: np.ndarray | None,
        sim_units_per_meter: float,
        grid_size: int,
    ) -> list[dict]:
        """Generate thermal false positives (warm animals, embers, heated rocks)."""
        cfg = self.config
        fps = []

        if self.rng.random() < cfg.false_positive_rate:
            # One false positive: random position within footprint
            angle = self.rng.uniform(0, 2 * math.pi)
            dist = self.rng.uniform(0, footprint_world * 0.4)
            fp_x = drone_xy[0] + dist * math.cos(angle)
            fp_y = drone_xy[1] + dist * math.sin(angle)

            # FPs are more likely near fire (embers, heated debris)
            local_fire = self._sample_grid_at(
                fire_grid, fp_x, fp_y, sim_units_per_meter, grid_size
            )
            fp_boost = 1.0 + 2.0 * local_fire  # More FPs near fire

            if self.rng.random() < min(1.0, cfg.false_positive_rate * fp_boost):
                fps.append({
                    "survivor_index": None,
                    "true_world_xy": [round(fp_x, 6), round(fp_y, 6)],
                    "detected": True,
                    "detection_probability": None,
                    "delta_t_k": round(self.rng.uniform(2.0, 8.0), 2),
                    "ground_temp_k": None,
                    "local_fire_intensity": round(local_fire, 3),
                    "thermal_crossover": False,
                    "confidence": round(self.rng.uniform(0.3, 0.65), 4),
                    "estimated_world_xy": [round(fp_x, 6), round(fp_y, 6)],
                    "is_false_positive": True,
                    "sensor": "thermal",
                })

        return fps

    def _sample_grid_at(
        self,
        grid: np.ndarray | None,
        world_x: float,
        world_y: float,
        sim_units_per_meter: float,
        grid_size: int,
    ) -> float:
        """Sample a simulation grid value at a world coordinate."""
        if grid is None:
            return 0.0
        # World coords are in [-1, 1] normalized space
        col = int(np.clip((world_x + 1.0) * 0.5 * grid_size, 0, grid_size - 1))
        row = int(np.clip((world_y + 1.0) * 0.5 * grid_size, 0, grid_size - 1))
        val = float(grid[row, col])
        return val

    @property
    def frame_count(self) -> int:
        return self._frame_count


def fuse_cv_thermal(
    cv_detections: list[dict],
    thermal_detections: list[dict],
    *,
    fusion_mode: str = "union",
    match_radius_world: float = 0.05,
) -> list[dict]:
    """Fuse CV and thermal detection results.

    Parameters
    ----------
    cv_detections : detections from CV pipeline (have 'matched_survivor_index')
    thermal_detections : detections from thermal model (have 'survivor_index')
    fusion_mode : 'union' (either detects → candidate) or
                  'intersection' (both must detect → confirmed) or
                  'weighted' (combine confidences)
    match_radius_world : maximum distance to consider detections as the same target

    Returns
    -------
    Fused detection list with source attribution.
    """
    fused = []

    # Index thermal detections by survivor_index for matching
    thermal_by_survivor = {}
    thermal_fp = []
    for td in thermal_detections:
        if td.get("detected", False):
            sid = td.get("survivor_index")
            if sid is not None:
                thermal_by_survivor[sid] = td
            else:
                thermal_fp.append(td)

    # Index CV detections by matched_survivor_index
    cv_by_survivor = {}
    cv_unmatched = []
    for cd in cv_detections:
        sid = cd.get("matched_survivor_index")
        if sid is not None:
            cv_by_survivor[sid] = cd
        else:
            cv_unmatched.append(cd)

    # All survivor indices seen by either sensor
    all_survivors = set(cv_by_survivor.keys()) | set(thermal_by_survivor.keys())

    for sid in all_survivors:
        cv_det = cv_by_survivor.get(sid)
        th_det = thermal_by_survivor.get(sid)

        if fusion_mode == "union":
            # Either sensor detecting is enough
            if cv_det or th_det:
                fused.append(_merge_detection(cv_det, th_det, sid, mode="union"))

        elif fusion_mode == "intersection":
            # Both sensors must detect
            if cv_det and th_det:
                fused.append(_merge_detection(cv_det, th_det, sid, mode="intersection"))

        elif fusion_mode == "weighted":
            # Weighted combination of confidence scores
            if cv_det or th_det:
                fused.append(_merge_detection(cv_det, th_det, sid, mode="weighted"))

    # Add unmatched CV detections (potential FPs or survivors without thermal)
    for cd in cv_unmatched:
        fused.append({
            **cd,
            "fusion_source": "cv_only",
            "thermal_confirmed": False,
        })

    # Add thermal false positives
    for fp in thermal_fp:
        fused.append({
            **fp,
            "fusion_source": "thermal_fp",
            "cv_confirmed": False,
        })

    return fused


def _merge_detection(
    cv_det: dict | None,
    th_det: dict | None,
    survivor_index: int,
    mode: str,
) -> dict:
    """Merge a CV and thermal detection for the same survivor."""
    result = {
        "matched_survivor_index": survivor_index,
        "fusion_mode": mode,
    }

    if cv_det and th_det:
        # Both sensors detected
        cv_conf = cv_det.get("confidence", 0.0)
        th_conf = th_det.get("confidence", 0.0)

        if mode == "weighted":
            # Weighted average: CV gets higher weight (has spatial precision)
            fused_conf = 0.6 * cv_conf + 0.4 * th_conf
        else:
            fused_conf = max(cv_conf, th_conf)

        result.update({
            "confidence": round(fused_conf, 4),
            "cv_confidence": round(cv_conf, 4),
            "thermal_confidence": round(th_conf, 4),
            "fusion_source": "both",
            "cv_confirmed": True,
            "thermal_confirmed": True,
            "estimated_world_xy": cv_det.get("estimated_world_xy"),  # CV has better spatial precision
            "bbox_xyxy": cv_det.get("bbox_xyxy"),
            "delta_t_k": th_det.get("delta_t_k"),
            "thermal_crossover": th_det.get("thermal_crossover", False),
        })

    elif cv_det:
        result.update({
            "confidence": cv_det.get("confidence", 0.0),
            "cv_confidence": cv_det.get("confidence", 0.0),
            "thermal_confidence": None,
            "fusion_source": "cv_only",
            "cv_confirmed": True,
            "thermal_confirmed": False,
            "estimated_world_xy": cv_det.get("estimated_world_xy"),
            "bbox_xyxy": cv_det.get("bbox_xyxy"),
        })

    elif th_det:
        result.update({
            "confidence": th_det.get("confidence", 0.0),
            "cv_confidence": None,
            "thermal_confidence": th_det.get("confidence", 0.0),
            "fusion_source": "thermal_only",
            "cv_confirmed": False,
            "thermal_confirmed": True,
            "estimated_world_xy": th_det.get("estimated_world_xy"),
            "delta_t_k": th_det.get("delta_t_k"),
            "thermal_crossover": th_det.get("thermal_crossover", False),
        })

    return result
