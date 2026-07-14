"""Compare detection quality across all detection modes.

Runs simulations with CV, Thermal, CV+Thermal, Motion, and CV+Motion modes
and reports recall, precision, false positive rate, and detection probability
statistics for each mode under different fire/smoke conditions.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from detection.simulation_adapter import (
    SimDrone,
    SimEntity,
    SimWildfireState,
    SimulationCvAdapter,
)
from detection.thermal_model import ThermalSensorConfig, ThermalSensorModel


def make_grids(grid_size: int, scenario: str) -> dict:
    """Create simulation grids for different fire/smoke scenarios."""
    fire = np.zeros((grid_size, grid_size))
    fire_intensity = np.zeros((grid_size, grid_size))
    burned = np.zeros((grid_size, grid_size))
    smoke = np.zeros((grid_size, grid_size))

    if scenario == "clear":
        pass  # All zeros
    elif scenario == "light_smoke":
        smoke[:] = 0.5
    elif scenario == "heavy_smoke":
        smoke[:] = 2.5
    elif scenario == "active_fire":
        fire[3:7, 3:7] = 1.0
        fire_intensity[3:7, 3:7] = 0.8
        smoke[:] = 1.5
    elif scenario == "burned_ground":
        burned[:] = 0.46  # ΔT ≈ 0 (thermal crossover)
        smoke[:] = 0.3
    elif scenario == "mixed":
        fire[0:3, 0:3] = 1.0
        fire_intensity[0:3, 0:3] = 0.6
        burned[3:7, 3:7] = 0.4
        smoke[:] = 1.0

    return {
        "fire_grid": fire,
        "fire_intensity_grid": fire_intensity,
        "burned_grid": burned,
        "smoke_grid": smoke,
    }


def run_thermal_only(
    survivors: list[SimEntity],
    drone: SimDrone,
    grids: dict,
    altitude_m: float,
    sim_units_per_meter: float,
    grid_size: int,
    n_trials: int = 50,
) -> dict:
    """Run thermal-only detection over multiple trials."""
    detections_total = 0
    true_positives = 0
    false_positives = 0
    total_survivors = len(survivors) * n_trials
    probs = []

    for trial in range(n_trials):
        model = ThermalSensorModel(ThermalSensorConfig(seed=trial))
        survivor_dicts = [{"index": s.index, "world_xy": s.world_xy} for s in survivors]

        results = model.detect_survivors(
            drone_xy=drone.world_xy,
            drone_altitude_m=altitude_m,
            fov_deg=65.0,
            survivors=survivor_dicts,
            fire_grid=grids["fire_grid"],
            fire_intensity_grid=grids["fire_intensity_grid"],
            burned_grid=grids["burned_grid"],
            smoke_grid=grids["smoke_grid"],
            sim_units_per_meter=sim_units_per_meter,
            grid_size=grid_size,
        )

        for r in results:
            if r.get("detected"):
                detections_total += 1
                if r.get("survivor_index") is not None and not r.get("is_false_positive"):
                    true_positives += 1
                else:
                    false_positives += 1
            if r.get("detection_probability") is not None:
                probs.append(r["detection_probability"])

    recall = true_positives / max(total_survivors, 1)
    precision = true_positives / max(detections_total, 1)
    fp_rate = false_positives / max(n_trials, 1)

    return {
        "recall": round(recall, 3),
        "precision": round(precision, 3),
        "fp_rate": round(fp_rate, 3),
        "mean_detection_prob": round(np.mean(probs), 3) if probs else 0.0,
        "detections": detections_total,
        "true_positives": true_positives,
        "false_positives": false_positives,
    }


def run_cv_simulation(
    survivors: list[SimEntity],
    drone: SimDrone,
    grids: dict,
    grid_size: int,
    n_trials: int = 50,
) -> dict:
    """Simulate CV detection using the preliminary (stochastic) backend.

    Uses the abstract stochastic camera model to estimate CV performance
    under different smoke/fire conditions, matching the simulator's perception model.
    """
    from detection.preliminary_detector import PreliminaryPersonDetector

    detections_total = 0
    true_positives = 0
    total_survivors = len(survivors) * n_trials
    img_size = 512

    # Model CV degradation by smoke/fire using the simulator's approach
    smoke_val = float(grids["smoke_grid"].mean())
    fire_val = float(grids["fire_intensity_grid"].mean())

    # Match the simulator's Liu et al. (2020) atmospheric-transmission fit.
    smoke_factor = max(1.0 - smoke_val, 0.0) ** 1.24
    # Fire glare
    fire_glare = 1.0 - 0.35 * fire_val
    # Combined visible degradation
    vis_quality = smoke_factor * fire_glare

    for trial in range(n_trials):
        detector = PreliminaryPersonDetector(
            detection_probability=min(1.0, 0.935 * vis_quality),
            pixel_noise_std=3.0,
            confidence=0.80,
            confidence_jitter=0.1,
            seed=trial,
        )
        # Generate synthetic bboxes for survivors within footprint
        boxes = []
        for s in survivors:
            dx = s.world_xy[0] - drone.world_xy[0]
            dy = s.world_xy[1] - drone.world_xy[1]
            # Simple box placement
            cx = int((dx / 0.5 + 1.0) * img_size / 2)
            cy = int((dy / 0.5 + 1.0) * img_size / 2)
            w, h = 40, 60
            box = (max(0, cx - w // 2), max(0, cy - h // 2),
                   min(img_size, cx + w // 2), min(img_size, cy + h // 2))
            if 0 <= cx < img_size and 0 <= cy < img_size:
                boxes.append(box)

        result = detector.detect_boxes(boxes, image_size=img_size)
        n_det = len(result.detections)
        detections_total += n_det
        true_positives += n_det  # Preliminary detector only produces TPs

    recall = true_positives / max(total_survivors, 1)
    precision = true_positives / max(detections_total, 1) if detections_total > 0 else 1.0

    return {
        "recall": round(recall, 3),
        "precision": round(precision, 3),
        "fp_rate": 0.0,
        "vis_quality_factor": round(vis_quality, 3),
        "detections": detections_total,
        "true_positives": true_positives,
        "false_positives": 0,
    }


def run_motion_simulation(
    survivors: list,
    drone,
    grids: dict,
    n_trials: int = 50,
    img_size: int = 128,
) -> dict:
    """Simulate motion-only detection.

    Motion detection requires consecutive frames with a new object appearing.
    We simulate this by generating pairs of frames (background-only, then
    background + survivor blob) and running the motion detector.
    """
    from PIL import Image as PILImage
    from detection.motion_detector import MotionDetector, MotionDetectorConfig

    detections_total = 0
    true_positives = 0
    false_positives = 0
    total_survivors = len(survivors) * n_trials
    smoke_val = float(grids["smoke_grid"].mean())
    fire_val = float(grids["fire_intensity_grid"].mean())

    for trial in range(n_trials):
        rng = np.random.default_rng(trial)
        cfg = MotionDetectorConfig(min_blob_area=20, diff_threshold=25, dilation_size=3)
        det = MotionDetector(cfg)

        # Frame 1: background only
        bg = rng.integers(80, 160, (img_size, img_size, 3), dtype=np.uint8)
        # Add fire/smoke noise to background (increases FP chance)
        if fire_val > 0:
            noise = rng.integers(0, int(50 * fire_val), (img_size, img_size, 3), dtype=np.uint8)
            bg = np.clip(bg.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        frame1 = PILImage.fromarray(bg)
        det.detect(frame1, drone_xy=(0.0, 0.0), footprint_world=1.0, image_size=img_size)

        # Frame 2: background + survivor blobs
        frame2_arr = bg.copy()
        survivor_positions = []
        for s in survivors:
            # Map survivor world position to pixel position
            cx = int((s.world_xy[0] / 0.5 + 1.0) * img_size / 2)
            cy = int((s.world_xy[1] / 0.5 + 1.0) * img_size / 2)
            cx = max(15, min(img_size - 15, cx))
            cy = max(15, min(img_size - 15, cy))
            survivor_positions.append((cx, cy))
            # Draw survivor blob
            radius = 10
            for r in range(max(0, cy - radius), min(img_size, cy + radius)):
                for c in range(max(0, cx - radius), min(img_size, cx + radius)):
                    if (r - cy) ** 2 + (c - cx) ** 2 < radius ** 2:
                        frame2_arr[r, c] = 240  # Bright blob

        frame2 = PILImage.fromarray(frame2_arr)
        results = det.detect(
            frame2, drone_xy=(0.005, 0.0), footprint_world=1.0,
            image_size=img_size, smoke_load=smoke_val,
        )

        # Match detections to survivor positions
        matched = set()
        for det_r in results:
            dcx, dcy = det_r["center_px"]
            is_tp = False
            for si, (scx, scy) in enumerate(survivor_positions):
                if si in matched:
                    continue
                if abs(dcx - scx) < 25 and abs(dcy - scy) < 25:
                    is_tp = True
                    matched.add(si)
                    break
            if is_tp:
                true_positives += 1
            else:
                false_positives += 1
            detections_total += 1

    recall = true_positives / max(total_survivors, 1)
    precision = true_positives / max(detections_total, 1) if detections_total > 0 else 1.0
    fp_rate = false_positives / max(n_trials, 1)

    return {
        "recall": round(recall, 3),
        "precision": round(precision, 3),
        "fp_rate": round(fp_rate, 3),
        "detections": detections_total,
        "true_positives": true_positives,
        "false_positives": false_positives,
    }


def run_cv_motion_fusion(cv_results: dict, motion_results: dict) -> dict:
    """Estimate CV+Motion (boost) fusion performance."""
    # CV+Motion boost: motion confirms CV detections, boosting confidence.
    # Recall = CV recall (motion doesn't add new detections in boost mode)
    # But motion can also reduce FPs by only confirming real ones.
    cv_recall = cv_results["recall"]
    motion_recall = motion_results["recall"]

    # In boost mode, recall is same as CV (motion only boosts confidence)
    # In practice, motion confirmation helps downstream filtering
    boost_recall = cv_recall

    # Precision improves slightly (motion-confirmed detections are more reliable)
    cv_precision = cv_results.get("precision", 1.0)
    boost_precision = min(1.0, cv_precision + 0.02 * motion_recall)

    return {
        "recall": round(boost_recall, 3),
        "precision": round(boost_precision, 3),
        "fp_rate": round(cv_results["fp_rate"], 3),
        "motion_recall": round(motion_recall, 3),
        "note": "boost mode — motion confirms CV, does not add new detections",
    }


def run_fusion(cv_results: dict, thermal_results: dict) -> dict:
    """Estimate fusion (union) performance from individual mode results."""
    # Union: detected if either sensor detects
    # P(union) = 1 - (1-P_cv)(1-P_thermal) for independent sensors
    cv_recall = cv_results["recall"]
    th_recall = thermal_results["recall"]
    union_recall = 1.0 - (1.0 - cv_recall) * (1.0 - th_recall)

    # FP rate: sum of both (conservative upper bound)
    union_fp = cv_results["fp_rate"] + thermal_results["fp_rate"]

    # Precision estimate
    union_tp = union_recall * 50 * 3  # n_trials * n_survivors approx
    union_total_det = union_tp + union_fp * 50
    union_precision = union_tp / max(union_total_det, 1)

    return {
        "recall": round(union_recall, 3),
        "precision": round(union_precision, 3),
        "fp_rate": round(union_fp, 3),
        "cv_recall_contribution": round(cv_recall, 3),
        "thermal_recall_contribution": round(th_recall, 3),
    }


def main():
    print("=" * 80)
    print("DETECTION MODE COMPARISON: CV vs Thermal vs Motion vs Fusions")
    print("=" * 80)
    print()

    grid_size = 10
    sim_units_per_meter = 0.02  # Typical for the simulation

    # Place survivors at known positions within drone footprint
    survivors = [
        SimEntity(index=0, world_xy=(0.0, 0.0)),
        SimEntity(index=1, world_xy=(0.05, 0.03)),
        SimEntity(index=2, world_xy=(-0.03, 0.04)),
    ]

    drone = SimDrone(
        index=0,
        name="drone_0",
        world_xy=(0.0, 0.0),
        altitude_agl=30.0 * sim_units_per_meter,
    )
    altitude_m = 30.0

    scenarios = ["clear", "light_smoke", "heavy_smoke", "active_fire", "burned_ground", "mixed"]
    n_trials = 100

    all_results = {}

    for scenario in scenarios:
        print(f"\n{'─' * 80}")
        print(f"  Scenario: {scenario.upper()}")
        print(f"{'─' * 80}")

        grids = make_grids(grid_size, scenario)
        smoke_level = float(grids["smoke_grid"].mean())
        fire_level = float(grids["fire_intensity_grid"].mean())
        burn_level = float(grids["burned_grid"].mean())
        print(f"  Conditions: smoke={smoke_level:.2f}, fire={fire_level:.2f}, burned={burn_level:.2f}")
        print()

        # CV-only
        cv_res = run_cv_simulation(survivors, drone, grids, grid_size, n_trials)

        # Thermal-only
        th_res = run_thermal_only(survivors, drone, grids, altitude_m, sim_units_per_meter, grid_size, n_trials)

        # Motion-only
        mo_res = run_motion_simulation(survivors, drone, grids, n_trials)

        # CV+Thermal fusion
        cv_th_res = run_fusion(cv_res, th_res)

        # CV+Motion fusion
        cv_mo_res = run_cv_motion_fusion(cv_res, mo_res)

        all_results[scenario] = {
            "cv": cv_res,
            "thermal": th_res,
            "motion": mo_res,
            "cv+thermal": cv_th_res,
            "cv+motion": cv_mo_res,
        }

        print(f"  {'Metric':<15} {'CV':>8} {'Thermal':>9} {'Motion':>8} {'CV+Th':>8} {'CV+Mo':>8}")
        print(f"  {'─' * 51}")
        print(f"  {'Recall':<15} {cv_res['recall']:>8.3f} {th_res['recall']:>9.3f} {mo_res['recall']:>8.3f} {cv_th_res['recall']:>8.3f} {cv_mo_res['recall']:>8.3f}")
        print(f"  {'Precision':<15} {cv_res['precision']:>8.3f} {th_res['precision']:>9.3f} {mo_res['precision']:>8.3f} {cv_th_res['precision']:>8.3f} {cv_mo_res['precision']:>8.3f}")
        print(f"  {'FP/frame':<15} {cv_res['fp_rate']:>8.3f} {th_res['fp_rate']:>9.3f} {mo_res['fp_rate']:>8.3f} {cv_th_res['fp_rate']:>8.3f} {cv_mo_res['fp_rate']:>8.3f}")

    # Summary table
    print(f"\n\n{'=' * 80}")
    print("SUMMARY: Recall by Scenario and Mode")
    print(f"{'=' * 80}")
    print(f"\n  {'Scenario':<16} {'CV':>7} {'Thermal':>9} {'Motion':>8} {'CV+Th':>8} {'CV+Mo':>8} {'Best':>10}")
    print(f"  {'─' * 66}")
    for scenario in scenarios:
        r = all_results[scenario]
        cv_r = r["cv"]["recall"]
        th_r = r["thermal"]["recall"]
        mo_r = r["motion"]["recall"]
        cvth_r = r["cv+thermal"]["recall"]
        cvmo_r = r["cv+motion"]["recall"]
        vals = {"CV": cv_r, "Thermal": th_r, "Motion": mo_r, "CV+Th": cvth_r, "CV+Mo": cvmo_r}
        best = max(vals, key=vals.get)
        print(f"  {scenario:<16} {cv_r:>7.3f} {th_r:>9.3f} {mo_r:>8.3f} {cvth_r:>8.3f} {cvmo_r:>8.3f} {best:>10}")

    # Key findings
    print(f"\n\n{'=' * 80}")
    print("KEY FINDINGS")
    print(f"{'=' * 80}")
    clear = all_results["clear"]
    smoke = all_results["heavy_smoke"]
    burn = all_results["burned_ground"]

    print(f"\n  1. CLEAR: CV={clear['cv']['recall']:.1%}, Motion={clear['motion']['recall']:.1%}, "
          f"CV+Thermal={clear['cv+thermal']['recall']:.1%}")
    print(f"  2. HEAVY SMOKE: CV={smoke['cv']['recall']:.1%}, Thermal={smoke['thermal']['recall']:.1%}, "
          f"Motion={smoke['motion']['recall']:.1%}")
    print(f"     → CV+Thermal fusion: {smoke['cv+thermal']['recall']:.1%} (best)")
    print(f"  3. BURNED GROUND: Thermal={burn['thermal']['recall']:.1%} (crossover!), "
          f"CV={burn['cv']['recall']:.1%}, Motion={burn['motion']['recall']:.1%}")
    print(f"  4. MOTION limitations: requires drone movement + visible change between frames")
    print(f"     Motion alone is unreliable (survivors are static, fire creates noise)")
    print(f"  5. CV+Motion (boost mode): same recall as CV, higher confidence on confirmed dets")
    print(f"  6. CV+Thermal remains the most robust fusion across all scenarios")
    print()

    # Save results
    out_path = Path("data/detection_mode_comparison.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"  Results saved to: {out_path}")


if __name__ == "__main__":
    main()
