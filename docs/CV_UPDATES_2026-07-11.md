# Weekly Progress Report — July 5–11, 2026

**Project:** OmniSearch — Heterogeneous Air-Ground Robotic Swarms for Wildfire Survivor Search
**Author:** Oleksii Lavrenin
**Period:** Sunday Jul 5 – Saturday Jul 11, 2026

---

## Results

| Milestone | Status |
|-----------|--------|
| Real thermal-infrared dataset ingested (HIT-UAV, 2,898 images) | **Achieved** |
| Drone detector retrained natively at 1280px on a larger, fire-biased dataset | **Achieved** |
| Root cause of the tiled-inference recall gap found and fixed | **Achieved** |
| Full eval suite re-run and compared before/after (recall-by-size, decoy FP, scenario heatmap) | **Achieved** |
| Presentation deck heatmaps and figures refreshed with measured results | **Achieved** |
| All 7 CV datasets re-annotated with physical-scale overlays and exported | **Achieved** |
| Colorful hard-negative decoy false positives reduced | **Not resolved** |

---

## What Was Done

### 1. New training data generated

| Dataset | Images | What's new | Why |
|---|---|---|---|
| `survivor_6k/` | 6,800 (6,000 train / 800 val) | Fire/smoke fraction raised 65%→75%; decoy fraction 20%→35%; 25% oblique views; all prior physical-sizing fixes carried forward | Bias training toward the hardest scenario (burned ground) and get large-enough per-bucket samples for tight confidence intervals |
| `thermal_real/` | 2,898 (2,029 train / 290 val / 579 test) | Real infrared drone imagery ingested from the public **HIT-UAV** dataset (Suo et al., *Scientific Data* 2023, CC-BY-4.0), person class only — vehicle/bicycle classes stripped | Address reviewer feedback that our synthetic thermal blobs had no human shape; this is real sensor data with real human silhouettes at 60–130 m altitude, day and night |
| **Total new images this week** | **9,698** | | |

New script: `scripts/ingest_hit_uav.py` — converts HIT-UAV to our YOLO format with per-frame altitude/camera-angle metadata parsed from filenames.

### 2. Model training

| Model | Data | Config | Result |
|---|---|---|---|
| `survivor_yolov8s_1280.pt` (new) | `survivor_6k` (6,800 img) | YOLOv8s, imgsz 1280, 30 epochs, Apple GPU (MPS) — first run moved off CPU (~7h vs an estimated 30h+) | P 0.92 · R 0.77 · mAP50 0.86 (val) |
| `thermal_real_yolov8n.pt` (new) | `thermal_real` (2,898 img) | YOLOv8n, imgsz 640, 30 epochs, CPU | P 0.89 · R 0.86 · mAP50 0.92 (held-out test split) |

Also added a `--device` flag to `scripts/train_survivor_detector.py` so future runs can use `mps` instead of being hardcoded to `cpu`.

### 3. Detector quality — before vs. after (measured, not simulated)

| Eval | Metric | Old model | New model | Change |
|---|---|---|---|---|
| Recall by size — **deployment config** (1280px, tiled 2×2) | Overall recall | 71.8% | **95.1%** | +23.3 pp |
| Recall by size — deployment config | Sub-8px recall | 62.5% | **89.9%** | +27.4 pp |
| Scenario heatmap | Clear recall | 69.4% | **81.9%** | +12.5 pp |
| Scenario heatmap | Burned-ground recall (hardest) | 50.2% | **65.8%** | +15.6 pp |
| Fire-condition eval | Fire/smoke recall | 84% | **87.8%** | +3.8 pp |
| Decoy false positives | Colorful-object FP rate | 15.4% | 14.3% | ~flat (unresolved) |
| Decoy false positives | Animal / vehicle FP rate | 1.5% / 0.0% | 2.0% / 1.0% | ~flat, negligible |

**Root cause found:** the previously deployed model was trained and validated at 640px, but the simulator's deployment code path runs tiled inference at 1280px. That mismatch — not a lack of data — was the dominant source of lost recall, especially on small (<8px) targets. Training natively at 1280px on more data closed the gap almost completely.

**Still open:** colorful hard-negative decoys still trigger a false "person" detection ~14% of the time. This did not move with more data or higher resolution, confirming it needs more decoy *diversity* specifically, not more scale.

New reusable eval scripts (all save JSON reports under `reports/` for repeatable before/after comparisons): `eval_recall_by_size.py`, `eval_decoy_fp.py`, `eval_cv_by_scenario.py`, `eval_cv_by_fire_condition.py`, `validate_labels.py`.

### 4. Presentation deck updated

`docs/slides/omnisearch_cv_results.pptx` — all three heatmap slides (modality comparison, recall-by-size, decoy false-positive) and the directly dependent bullets/cards (fusion-wins bars, "how the real detector performs," CV-perception-by-condition, executive-summary metric cards) were refreshed to the new measured numbers so the deck is internally consistent end to end.

### 5. Dataset visualizations exported

Re-annotated all 7 CV datasets with real-world-scale overlays (bounding boxes with px + metre dimensions, GSD/altitude/camera-mode header, color-coded decoy classes, scale bar):

| Dataset | Images annotated |
|---|---|
| `survivor` | 2,400 |
| `survivor_6k` | 6,800 |
| `survivor_naip` | 590 |
| `thermal` (synthetic) | 1,200 |
| `thermal_real` (HIT-UAV) | 2,898 |
| `ugv/front` + `ugv/mast` | 3,600 |
| **Total** | **17,488** |

The real HIT-UAV thermal frames are flagged "NO SCALE" in these visualizations — that public dataset publishes altitude and camera angle but not per-frame focal length, so a trustworthy ground-sample-distance cannot be derived for it the way it can for our own generators (where GSD is exact by construction).

---

## Progress Assessment

| Project Component | Progress | Notes |
|-------------------|----------|-------|
| CV perception pipeline (drone) | **95%** | Root-cause of the tiling/imgsz recall gap found and fixed; deployment-config recall now 95% |
| CV perception pipeline (thermal) | **90%** | Real sensor data (HIT-UAV) now included alongside synthetic; not yet fused with drone/UGV pipeline |
| Hard-negative robustness | **75%** | Human-scale decoys added; colorful-object false positives still ~14%, unresolved |
| Documentation | **95%** | EDA, composition report, and slide deck all current with this week's numbers |

**Overall CV workstream: ~90% complete.**
