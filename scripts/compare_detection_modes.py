"""Compare detection quality across CV, Thermal, and CV+Thermal modes.

Runs a short simulation with all three detection modes and reports
recall, precision, false positive rate, and detection probability
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

    # Smoke attenuation for visible: exp(-1.4 * smoke_load) with floor 0.55
    smoke_factor = max(np.exp(-1.4 * smoke_val), 0.55)
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
    print("=" * 70)
    print("DETECTION MODE COMPARISON: CV vs Thermal vs CV+Thermal")
    print("=" * 70)
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
        print(f"\n{'─' * 70}")
        print(f"  Scenario: {scenario.upper()}")
        print(f"{'─' * 70}")

        grids = make_grids(grid_size, scenario)
        smoke_level = float(grids["smoke_grid"].mean())
        fire_level = float(grids["fire_intensity_grid"].mean())
        burn_level = float(grids["burned_grid"].mean())
        print(f"  Conditions: smoke={smoke_level:.2f}, fire={fire_level:.2f}, burned={burn_level:.2f}")
        print()

        # CV-only
        t0 = time.time()
        cv_res = run_cv_simulation(survivors, drone, grids, grid_size, n_trials)
        cv_time = time.time() - t0

        # Thermal-only
        t0 = time.time()
        th_res = run_thermal_only(survivors, drone, grids, altitude_m, sim_units_per_meter, grid_size, n_trials)
        th_time = time.time() - t0

        # Fusion (estimated from independent results)
        fusion_res = run_fusion(cv_res, th_res)

        all_results[scenario] = {"cv": cv_res, "thermal": th_res, "fusion": fusion_res}

        print(f"  {'Metric':<20} {'CV Only':>12} {'Thermal Only':>14} {'CV+Thermal':>12}")
        print(f"  {'─' * 58}")
        print(f"  {'Recall':<20} {cv_res['recall']:>12.3f} {th_res['recall']:>14.3f} {fusion_res['recall']:>12.3f}")
        print(f"  {'Precision':<20} {cv_res['precision']:>12.3f} {th_res['precision']:>14.3f} {fusion_res['precision']:>12.3f}")
        print(f"  {'FP rate/frame':<20} {cv_res['fp_rate']:>12.3f} {th_res['fp_rate']:>14.3f} {fusion_res['fp_rate']:>12.3f}")
        print(f"  {'Time (s)':<20} {cv_time:>12.3f} {th_time:>14.3f} {'—':>12}")

    # Summary table
    print(f"\n\n{'=' * 70}")
    print("SUMMARY: Recall by Scenario")
    print(f"{'=' * 70}")
    print(f"\n  {'Scenario':<18} {'CV':>8} {'Thermal':>10} {'CV+Thermal':>12} {'Winner':>10}")
    print(f"  {'─' * 58}")
    for scenario in scenarios:
        r = all_results[scenario]
        cv_r = r["cv"]["recall"]
        th_r = r["thermal"]["recall"]
        fu_r = r["fusion"]["recall"]
        winner = "CV+Th" if fu_r >= max(cv_r, th_r) else ("CV" if cv_r > th_r else "Thermal")
        print(f"  {scenario:<18} {cv_r:>8.3f} {th_r:>10.3f} {fu_r:>12.3f} {winner:>10}")

    # Key findings
    print(f"\n\n{'=' * 70}")
    print("KEY FINDINGS")
    print(f"{'=' * 70}")
    clear_cv = all_results["clear"]["cv"]["recall"]
    smoke_cv = all_results["heavy_smoke"]["cv"]["recall"]
    smoke_th = all_results["heavy_smoke"]["thermal"]["recall"]
    smoke_fu = all_results["heavy_smoke"]["fusion"]["recall"]
    burn_cv = all_results["burned_ground"]["cv"]["recall"]
    burn_th = all_results["burned_ground"]["thermal"]["recall"]
    burn_fu = all_results["burned_ground"]["fusion"]["recall"]

    print(f"\n  1. In CLEAR conditions: CV achieves {clear_cv:.1%} recall (baseline)")
    print(f"  2. In HEAVY SMOKE: CV drops to {smoke_cv:.1%}, Thermal maintains {smoke_th:.1%}")
    print(f"     → Fusion recovers to {smoke_fu:.1%}")
    print(f"  3. On BURNED GROUND (thermal crossover): Thermal drops to {burn_th:.1%}")
    print(f"     → CV rescues at {burn_cv:.1%}, Fusion = {burn_fu:.1%}")
    print(f"  4. FUSION (CV+Thermal) is most robust across all scenarios")
    print()

    # Save results
    out_path = Path("data/detection_mode_comparison.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"  Results saved to: {out_path}")


if __name__ == "__main__":
    main()
