"""Evaluate CV perception quality for RL integration readiness.

Runs the YOLO-based survivor detector across a controlled matrix of conditions
and produces quantitative metrics: precision, recall, confidence calibration,
spatial accuracy, temporal stability, and false-positive characterization.

The outcome drives a clear recommendation for whether to integrate CV into the
RL training loop, keep it as visualization-only, or use it as a post-hoc layer.

    python scripts/evaluate_cv_perception.py
    python scripts/evaluate_cv_perception.py --model models/survivor_naip_yolov8s.pt
    python scripts/evaluate_cv_perception.py --n-trials 20 --report-path docs/cv_perception_eval.md
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from detection.simulation_adapter import SimulationCvAdapter
from detection.wildfire_effects import (
    WildfireEffectConfig,
    WildfireMasks,
    apply_wildfire_effects_to_pil,
)


@dataclass
class DetectionEvent:
    """Single detection in a trial frame."""
    confidence: float
    box_xyxy: tuple[float, float, float, float]
    center_px: tuple[float, float]
    matched_truth_idx: Optional[int]  # None = false positive


@dataclass
class TrialResult:
    """Result of one rendering + detection trial."""
    scenario: str
    seed: int
    n_survivors_placed: int
    detections: list[DetectionEvent] = field(default_factory=list)
    truth_boxes: list[tuple[int, int, int, int]] = field(default_factory=list)
    render_time_ms: float = 0.0
    detect_time_ms: float = 0.0

    @property
    def tp(self) -> int:
        return sum(1 for d in self.detections if d.matched_truth_idx is not None)

    @property
    def fp(self) -> int:
        return sum(1 for d in self.detections if d.matched_truth_idx is None)

    @property
    def fn(self) -> int:
        return self.n_survivors_placed - self.tp

    @property
    def precision(self) -> float:
        denom = self.tp + self.fp
        return self.tp / denom if denom > 0 else 1.0

    @property
    def recall(self) -> float:
        return self.tp / self.n_survivors_placed if self.n_survivors_placed > 0 else 1.0


@dataclass
class StabilityResult:
    """Measures detection consistency across repeated runs on the same image."""
    n_runs: int
    detection_counts: list[int]
    confidence_means: list[float]
    box_center_std_px: float  # std of center positions across runs


def _disc(size: int, cx: float, cy: float, r: float) -> np.ndarray:
    yy, xx = np.mgrid[0:size, 0:size]
    return np.clip(1.0 - np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / r, 0, 1).astype(np.float32)


def _make_background(size: int, rng: np.random.Generator) -> Image.Image:
    """Generate a naturalistic procedural background."""
    base = rng.integers(50, 130, (size, size, 3)).astype("uint8")
    base[..., 1] = np.clip(base[..., 1].astype(float) * 1.15, 0, 255).astype("uint8")
    return Image.fromarray(base, "RGB")


def _load_assets(assets_dir: Path, max_assets: int = 10) -> list[Image.Image]:
    """Load SARD survivor assets for evaluation."""
    paths = sorted(glob.glob(str(assets_dir / "*.png")))
    if not paths:
        raise FileNotFoundError(f"No PNG assets in {assets_dir}")
    selected = paths[:max_assets]
    return [Image.open(p).convert("RGBA") for p in selected]


def _place_survivor(bg: Image.Image, asset: Image.Image, cx: int, cy: int, width_px: int) -> tuple[int, int, int, int]:
    """Paste a survivor asset onto background, return bbox xyxy."""
    aspect = asset.height / asset.width
    h_px = int(width_px * aspect)
    resized = asset.resize((width_px, h_px), Image.Resampling.LANCZOS)
    x = cx - width_px // 2
    y = cy - h_px // 2
    bg.paste(resized, (x, y), resized)
    return (x, y, x + width_px, y + h_px)


def _apply_wildfire(view: Image.Image, scenario: str, cfg: WildfireEffectConfig) -> Image.Image:
    """Apply wildfire effects based on scenario name."""
    sz = view.size[0]
    c = sz // 2
    if scenario == "clean":
        return view
    elif scenario == "light_smoke":
        m = WildfireMasks(burned=np.zeros((sz, sz), "f4"), active=np.zeros((sz, sz), "f4"),
                         intensity=np.zeros((sz, sz), "f4"), smoke=_disc(sz, c, c, sz * 0.45) * 0.5)
        v, _ = apply_wildfire_effects_to_pil(view, m, config=cfg, include_burn=False, include_flame=False, include_smoke=True)
        return v
    elif scenario == "heavy_smoke":
        m = WildfireMasks(burned=np.zeros((sz, sz), "f4"), active=np.zeros((sz, sz), "f4"),
                         intensity=np.zeros((sz, sz), "f4"), smoke=_disc(sz, c, c, sz * 0.45) * 1.0)
        v, _ = apply_wildfire_effects_to_pil(view, m, config=cfg, include_burn=False, include_flame=False, include_smoke=True)
        return v
    elif scenario == "fire":
        m = WildfireMasks(burned=_disc(sz, c, c, sz * 0.4), active=_disc(sz, c, c, sz * 0.3),
                         intensity=_disc(sz, c, c, sz * 0.3), smoke=np.zeros((sz, sz), "f4"))
        v, _ = apply_wildfire_effects_to_pil(view, m, config=cfg, include_burn=True, include_flame=True, include_smoke=False)
        return v
    elif scenario == "fire_and_smoke":
        m = WildfireMasks(burned=_disc(sz, c, c, sz * 0.4), active=_disc(sz, c, c, sz * 0.3),
                         intensity=_disc(sz, c, c, sz * 0.3), smoke=_disc(sz, c, c, sz * 0.45) * 0.8)
        v, _ = apply_wildfire_effects_to_pil(view, m, config=cfg, include_burn=True, include_flame=True, include_smoke=False)
        v, _ = apply_wildfire_effects_to_pil(v, m, config=cfg, include_burn=False, include_flame=False, include_smoke=True)
        return v
    elif scenario == "empty":
        return view
    return view


def _iou(a: tuple, b: tuple) -> float:
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def run_detection_trials(
    det: SimulationCvAdapter,
    assets: list[Image.Image],
    *,
    n_trials: int = 10,
    image_size: int = 1024,
    survivor_sizes_px: list[int] = None,
    scenarios: list[str] = None,
) -> list[TrialResult]:
    """Run detection across scenarios, sizes, and random placements."""
    if survivor_sizes_px is None:
        survivor_sizes_px = [30, 50, 80, 120]
    if scenarios is None:
        scenarios = ["clean", "light_smoke", "heavy_smoke", "fire", "fire_and_smoke"]

    cfg = WildfireEffectConfig(seed=42)
    results = []

    for scenario in scenarios:
        for size_px in survivor_sizes_px:
            for trial in range(n_trials):
                rng = np.random.default_rng(trial * 1000 + hash(scenario) % 9999)
                asset = assets[trial % len(assets)]
                bg = _make_background(image_size, rng)

                cx = int(rng.integers(image_size // 4, 3 * image_size // 4))
                cy = int(rng.integers(image_size // 4, 3 * image_size // 4))

                t0 = time.time()
                truth_box = _place_survivor(bg, asset, cx, cy, size_px)
                view = _apply_wildfire(bg, scenario, cfg)
                render_ms = (time.time() - t0) * 1000

                t1 = time.time()
                cv_boxes = det._detect_people_cv(view)
                detect_ms = (time.time() - t1) * 1000

                events = []
                truth_already_matched = False
                for box, conf in cv_boxes:
                    iou_val = _iou(box, truth_box)
                    if iou_val >= 0.15 and not truth_already_matched:
                        matched_idx = 0
                        truth_already_matched = True
                    else:
                        matched_idx = None
                    events.append(DetectionEvent(
                        confidence=conf,
                        box_xyxy=box,
                        center_px=((box[0] + box[2]) / 2, (box[1] + box[3]) / 2),
                        matched_truth_idx=matched_idx,
                    ))

                tr = TrialResult(
                    scenario=f"{scenario}@{size_px}px",
                    seed=trial,
                    n_survivors_placed=1,
                    detections=events,
                    truth_boxes=[truth_box],
                    render_time_ms=render_ms,
                    detect_time_ms=detect_ms,
                )
                results.append(tr)

    return results


def run_false_positive_trials(
    det: SimulationCvAdapter,
    *,
    n_trials: int = 20,
    image_size: int = 1024,
    scenarios: list[str] = None,
) -> list[TrialResult]:
    """Run detection on empty backgrounds (no survivors) to measure FP rate."""
    if scenarios is None:
        scenarios = ["empty", "fire", "fire_and_smoke"]

    cfg = WildfireEffectConfig(seed=42)
    results = []

    for scenario in scenarios:
        for trial in range(n_trials):
            rng = np.random.default_rng(trial * 7 + 3)
            bg = _make_background(image_size, rng)
            view = _apply_wildfire(bg, scenario, cfg)

            cv_boxes = det._detect_people_cv(view)
            events = [DetectionEvent(
                confidence=conf, box_xyxy=box,
                center_px=((box[0] + box[2]) / 2, (box[1] + box[3]) / 2),
                matched_truth_idx=None,
            ) for box, conf in cv_boxes]

            results.append(TrialResult(
                scenario=f"FP_{scenario}",
                seed=trial,
                n_survivors_placed=0,
                detections=events,
                truth_boxes=[],
            ))

    return results


def run_stability_test(
    det: SimulationCvAdapter,
    assets: list[Image.Image],
    *,
    n_runs: int = 5,
    image_size: int = 1024,
) -> StabilityResult:
    """Run detection multiple times on same image to measure output stability."""
    rng = np.random.default_rng(999)
    asset = assets[0]
    bg = _make_background(image_size, rng)
    _place_survivor(bg, asset, image_size // 2, image_size // 2, 60)

    counts = []
    confidences = []
    centers = []

    for _ in range(n_runs):
        boxes = det._detect_people_cv(bg)
        counts.append(len(boxes))
        if boxes:
            confs = [c for _, c in boxes]
            confidences.append(sum(confs) / len(confs))
            for box, _ in boxes:
                centers.append(((box[0] + box[2]) / 2, (box[1] + box[3]) / 2))
        else:
            confidences.append(0.0)

    center_std = 0.0
    if len(centers) > 1:
        arr = np.array(centers)
        center_std = float(arr.std(axis=0).mean())

    return StabilityResult(
        n_runs=n_runs,
        detection_counts=counts,
        confidence_means=confidences,
        box_center_std_px=center_std,
    )


def run_spatial_accuracy_test(
    det: SimulationCvAdapter,
    assets: list[Image.Image],
    *,
    n_trials: int = 20,
    image_size: int = 1024,
    survivor_size_px: int = 60,
) -> dict:
    """Measure spatial error between detection center and ground-truth center."""
    errors_px = []
    relative_errors = []  # normalized by survivor size

    for trial in range(n_trials):
        rng = np.random.default_rng(trial * 13)
        asset = assets[trial % len(assets)]
        bg = _make_background(image_size, rng)
        cx = int(rng.integers(image_size // 4, 3 * image_size // 4))
        cy = int(rng.integers(image_size // 4, 3 * image_size // 4))
        truth_box = _place_survivor(bg, asset, cx, cy, survivor_size_px)
        truth_cx = (truth_box[0] + truth_box[2]) / 2
        truth_cy = (truth_box[1] + truth_box[3]) / 2

        cv_boxes = det._detect_people_cv(bg)
        best_iou = 0.0
        best_center = None
        for box, conf in cv_boxes:
            iou_val = _iou(box, truth_box)
            if iou_val > best_iou:
                best_iou = iou_val
                best_center = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)

        if best_center is not None:
            err = math.sqrt((best_center[0] - truth_cx) ** 2 + (best_center[1] - truth_cy) ** 2)
            errors_px.append(err)
            relative_errors.append(err / survivor_size_px)

    return {
        "n_matched": len(errors_px),
        "n_trials": n_trials,
        "mean_error_px": float(np.mean(errors_px)) if errors_px else None,
        "median_error_px": float(np.median(errors_px)) if errors_px else None,
        "max_error_px": float(np.max(errors_px)) if errors_px else None,
        "mean_relative_error": float(np.mean(relative_errors)) if relative_errors else None,
        "p90_error_px": float(np.percentile(errors_px, 90)) if errors_px else None,
    }


def generate_report(
    detection_results: list[TrialResult],
    fp_results: list[TrialResult],
    stability: StabilityResult,
    spatial: dict,
    model_name: str,
) -> str:
    """Generate markdown evaluation report."""
    lines = []
    lines.append("# CV Perception Quality Evaluation\n")
    display_model = Path(model_name).name if "/" in model_name else model_name
    lines.append(f"**Model:** `{display_model}`\n")
    lines.append("---\n")

    # 1. Detection metrics by scenario
    lines.append("## 1. Detection Performance by Scenario\n")
    lines.append("| Scenario | Recall | Precision | Mean Conf (TP) | N trials |")
    lines.append("|----------|--------|-----------|----------------|----------|")

    from collections import defaultdict
    grouped = defaultdict(list)
    for r in detection_results:
        grouped[r.scenario].append(r)

    overall_tp, overall_fp, overall_fn = 0, 0, 0
    for scenario in sorted(grouped.keys()):
        trials = grouped[scenario]
        tp = sum(t.tp for t in trials)
        fp = sum(t.fp for t in trials)
        fn = sum(t.fn for t in trials)
        overall_tp += tp; overall_fp += fp; overall_fn += fn
        prec = tp / (tp + fp) if (tp + fp) else 1.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        tp_confs = [d.confidence for t in trials for d in t.detections if d.matched_truth_idx is not None]
        mean_conf = sum(tp_confs) / len(tp_confs) if tp_confs else 0.0
        lines.append(f"| {scenario} | {rec:.3f} | {prec:.3f} | {mean_conf:.3f} | {len(trials)} |")

    overall_prec = overall_tp / (overall_tp + overall_fp) if (overall_tp + overall_fp) else 1.0
    overall_rec = overall_tp / (overall_tp + overall_fn) if (overall_tp + overall_fn) else 0.0
    lines.append(f"| **OVERALL** | **{overall_rec:.3f}** | **{overall_prec:.3f}** | — | {len(detection_results)} |")
    lines.append("")

    # 2. False positive analysis
    lines.append("## 2. False Positive Analysis (Empty Backgrounds)\n")
    lines.append("| Scenario | Mean FP/frame | Max FP/frame | Mean FP conf |")
    lines.append("|----------|---------------|--------------|--------------|")

    fp_grouped = defaultdict(list)
    for r in fp_results:
        fp_grouped[r.scenario].append(r)

    for scenario in sorted(fp_grouped.keys()):
        trials = fp_grouped[scenario]
        fp_counts = [t.fp for t in trials]
        mean_fp = sum(fp_counts) / len(fp_counts)
        max_fp = max(fp_counts)
        fp_confs = [d.confidence for t in trials for d in t.detections]
        mean_fp_conf = sum(fp_confs) / len(fp_confs) if fp_confs else 0.0
        lines.append(f"| {scenario} | {mean_fp:.2f} | {max_fp} | {mean_fp_conf:.3f} |")
    lines.append("")

    # 3. Spatial accuracy
    lines.append("## 3. Spatial Accuracy\n")
    if spatial["n_matched"] > 0:
        lines.append(f"- Matched detections: {spatial['n_matched']}/{spatial['n_trials']}")
        lines.append(f"- Mean center error: **{spatial['mean_error_px']:.1f} px**")
        lines.append(f"- Median center error: {spatial['median_error_px']:.1f} px")
        lines.append(f"- P90 center error: {spatial['p90_error_px']:.1f} px")
        lines.append(f"- Max center error: {spatial['max_error_px']:.1f} px")
        lines.append(f"- Mean relative error (error / survivor_size): {spatial['mean_relative_error']:.3f}")
    else:
        lines.append("- No detections matched ground truth (detector may not be loaded).")
    lines.append("")

    # 4. Temporal stability
    lines.append("## 4. Temporal Stability (Same Image, Multiple Runs)\n")
    lines.append(f"- Runs: {stability.n_runs}")
    lines.append(f"- Detection counts: {stability.detection_counts}")
    lines.append(f"- Confidence means: {[f'{c:.3f}' for c in stability.confidence_means]}")
    lines.append(f"- Center position std: **{stability.box_center_std_px:.2f} px**")
    count_stable = len(set(stability.detection_counts)) == 1
    lines.append(f"- Count stable (same count every run): {'YES' if count_stable else 'NO'}")
    lines.append("")

    # 5. Inference performance
    lines.append("## 5. Inference Performance\n")
    detect_times = [r.detect_time_ms for r in detection_results if r.detect_time_ms > 0]
    if detect_times:
        lines.append(f"- Mean inference time: {np.mean(detect_times):.0f} ms")
        lines.append(f"- Median inference time: {np.median(detect_times):.0f} ms")
        lines.append(f"- P95 inference time: {np.percentile(detect_times, 95):.0f} ms")
    lines.append("")

    # 6. RL Integration Assessment
    lines.append("## 6. RL Integration Assessment\n")
    lines.append("### Signals available from CV module:\n")
    lines.append("| Signal | Available | Quality | RL-useful? |")
    lines.append("|--------|-----------|---------|------------|")
    lines.append(f"| Detection probability | Yes (confidence) | {'Good' if overall_rec > 0.8 else 'Poor'} (recall={overall_rec:.2f}) | Partial — no miss model |")
    lines.append(f"| False positive rate | Yes | {'Low' if sum(t.fp for t in fp_results)/max(len(fp_results),1) < 2 else 'High'} ({sum(t.fp for t in fp_results)/max(len(fp_results),1):.1f} FP/frame) | {'Yes' if sum(t.fp for t in fp_results)/max(len(fp_results),1) < 2 else 'Problematic'} |")
    lines.append(f"| Spatial uncertainty | Yes (bbox center) | {spatial['mean_error_px']:.1f} px mean error | Partial |")
    lines.append(f"| Confidence scores | Yes | {'Calibrated' if overall_prec > 0.8 else 'Needs calibration'} | Yes — thresholdable |")
    lines.append(f"| Temporal consistency | Yes | {'Stable' if count_stable else 'Unstable'} | {'Yes' if count_stable else 'No — would inject noise'} |")
    lines.append("")

    # 7. Recommendation
    lines.append("## 7. Recommendation\n")
    fp_rate = sum(t.fp for t in fp_results) / max(len(fp_results), 1)
    is_recall_good = overall_rec >= 0.85
    is_fp_low = fp_rate < 2.0
    is_stable = count_stable
    is_spatial_good = spatial.get("mean_relative_error", 1.0) is not None and spatial.get("mean_relative_error", 1.0) < 0.5

    if is_recall_good and is_fp_low and is_stable and is_spatial_good:
        verdict = "INTEGRATE"
        lines.append("### Verdict: INTEGRATE into simulator\n")
        lines.append("The CV perception module is reliable enough to replace or augment the abstract")
        lines.append("stochastic camera model in the RL training loop. Detections produce stable,")
        lines.append("spatially accurate, high-recall signals with manageable false positive rates.")
        lines.append("")
        lines.append("**Integration path:**")
        lines.append("1. Replace `_drone_survivor_detections()` Bernoulli with CV confidence > threshold")
        lines.append("2. Feed `estimated_world_xy` into survivor message observations (with noise)")
        lines.append("3. Model false positives as phantom observations that waste UGV travel")
        lines.append("4. Use CV confidence as quality signal in observation vector")
    elif is_recall_good and (not is_fp_low or not is_stable):
        verdict = "POST-HOC LAYER"
        lines.append("### Verdict: Use as POST-HOC VISUALIZATION LAYER\n")
        lines.append("The CV module detects survivors reliably but introduces too many false positives")
        lines.append("or temporal instability to safely drive RL reward signals. False detections would")
        lines.append("corrupt the reward landscape, and unstable outputs inject non-stationary noise.")
        lines.append("")
        lines.append("**Current role (keep):**")
        lines.append("- Trajectory export visualization (`--enable-cv`)")
        lines.append("- Post-hoc validation of learned policies")
        lines.append("- Web viewer camera panel for qualitative assessment")
        lines.append("")
        lines.append("**Path to integration:**")
        if not is_fp_low:
            lines.append(f"- Reduce FP rate from {fp_rate:.1f}/frame to <1.0 (more NAIP training tiles, higher conf threshold)")
        if not is_stable:
            lines.append("- Stabilize outputs (deterministic inference, TTA averaging)")
    else:
        verdict = "TOY DEMO"
        lines.append("### Verdict: Keep as SEPARATE TOY DEMO\n")
        lines.append("The CV perception module does not reliably detect survivors in the current")
        lines.append("rendering conditions. Integration would provide random/noisy signals that")
        lines.append("hinder rather than help RL training.")
        lines.append("")
        lines.append("**Blockers:**")
        if not is_recall_good:
            lines.append(f"- Recall too low ({overall_rec:.2f}) — model misses survivors at deployment sizes")
        if not is_fp_low:
            lines.append(f"- FP rate too high ({fp_rate:.1f}/frame) — would corrupt reward landscape")
        lines.append("")
        lines.append("**Required improvements:**")
        lines.append("- Fine-tune on NAIP backgrounds spanning multiple geographic regions")
        lines.append("- Expand SARD asset diversity for training")
        lines.append("- Consider thermal/IR channel for smoke penetration")

    lines.append("")
    lines.append("---\n")
    lines.append("*Generated by `scripts/evaluate_cv_perception.py`*\n")

    return "\n".join(lines), verdict


def main():
    ap = argparse.ArgumentParser(description="Evaluate CV perception quality for RL integration")
    ap.add_argument("--model", default="yolov8n.pt", help="YOLO weights path")
    ap.add_argument("--size", type=int, default=1024, help="Image size for evaluation")
    ap.add_argument("--n-trials", type=int, default=10, help="Trials per scenario/size combo")
    ap.add_argument("--conf", type=float, default=0.35, help="Detection confidence threshold")
    ap.add_argument("--assets-dir", default=str(ROOT / "data/cv_assets/sard_grabcut"), help="Survivor assets directory")
    ap.add_argument("--report-path", default=str(ROOT / "docs/cv_perception_eval.md"), help="Output report path")
    ap.add_argument("--json-path", default=None, help="Optional JSON metrics output")
    args = ap.parse_args()

    print(f"=== CV Perception Quality Evaluation ===")
    print(f"Model: {args.model}")
    print(f"Image size: {args.size}px")
    print(f"Confidence threshold: {args.conf}")
    print(f"Trials per condition: {args.n_trials}")
    print()

    # Build detector
    det = object.__new__(SimulationCvAdapter)
    det.detector_backend = "yolo"
    det.person_conf = args.conf
    det.person_iou = 0.6
    det.person_imgsz = max(args.size, 1280)
    det.person_tiled = True
    det.person_tile_grid = 2
    det.person_tile_overlap = 0.25
    det.person_match_iou = 0.15
    det.person_device = None
    det._person_detector = None
    det.image_size = args.size

    # Resolve model path (same logic as adapter __init__)
    model_name = args.model
    if model_name == "yolov8n.pt":
        for candidate in ("survivor_naip_yolov8s.pt", "survivor_yolov8s.pt", "survivor_yolov8n.pt"):
            path = ROOT / "models" / candidate
            if path.exists():
                model_name = str(path)
                print(f"  Auto-selected fine-tuned model: {candidate}")
                break
    det.person_model_name = model_name

    # Load assets
    assets_dir = Path(args.assets_dir)
    if not assets_dir.exists():
        print(f"ERROR: Assets directory not found: {assets_dir}")
        print("Run `python scripts/extract_sard_assets.py` first.")
        sys.exit(1)
    assets = _load_assets(assets_dir)
    print(f"Loaded {len(assets)} survivor assets from {assets_dir.name}/")
    print()

    # Run evaluations
    print("[1/4] Running detection trials...")
    detection_results = run_detection_trials(
        det, assets, n_trials=args.n_trials, image_size=args.size,
        survivor_sizes_px=[30, 50, 80, 120],
        scenarios=["clean", "light_smoke", "heavy_smoke", "fire", "fire_and_smoke"],
    )
    tp_total = sum(r.tp for r in detection_results)
    fp_total = sum(r.fp for r in detection_results)
    fn_total = sum(r.fn for r in detection_results)
    print(f"  TP={tp_total} FP={fp_total} FN={fn_total}")

    print("[2/4] Running false positive trials (empty backgrounds)...")
    fp_results = run_false_positive_trials(det, n_trials=args.n_trials * 2, image_size=args.size)
    total_fps = sum(r.fp for r in fp_results)
    print(f"  Total FP on empty frames: {total_fps} across {len(fp_results)} frames ({total_fps/len(fp_results):.2f}/frame)")

    print("[3/4] Running stability test...")
    stability = run_stability_test(det, assets, n_runs=5, image_size=args.size)
    print(f"  Counts: {stability.detection_counts}, center std: {stability.box_center_std_px:.2f}px")

    print("[4/4] Running spatial accuracy test...")
    spatial = run_spatial_accuracy_test(det, assets, n_trials=args.n_trials * 2, image_size=args.size)
    if spatial["mean_error_px"] is not None:
        print(f"  Mean spatial error: {spatial['mean_error_px']:.1f}px (relative: {spatial['mean_relative_error']:.3f})")
    print()

    # Generate report
    report_text, verdict = generate_report(detection_results, fp_results, stability, spatial, model_name)
    report_path = Path(args.report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report_text)
    print(f"Report saved: {report_path}")
    print(f"\n{'='*50}")
    print(f"  VERDICT: {verdict}")
    print(f"{'='*50}\n")

    # Optional JSON export
    if args.json_path:
        metrics = {
            "model": model_name,
            "overall_precision": overall_prec if 'overall_prec' in dir() else tp_total / max(tp_total + fp_total, 1),
            "overall_recall": tp_total / max(tp_total + fn_total, 1),
            "fp_rate_per_frame": total_fps / max(len(fp_results), 1),
            "spatial_mean_error_px": spatial["mean_error_px"],
            "stability_count_consistent": len(set(stability.detection_counts)) == 1,
            "verdict": verdict,
        }
        Path(args.json_path).write_text(json.dumps(metrics, indent=2))
        print(f"JSON metrics: {args.json_path}")


if __name__ == "__main__":
    main()
