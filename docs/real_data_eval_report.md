# Real-Data Evaluation: Sim-to-Real Gap Measurement

**Date:** 2026-07-12 (updated 2026-07-13 with fine-tune + FLAME results; updated
2026-07-13 with the real-Malibu-terrain UGV set replacing COCO)
**Purpose:** Quantify how the synthetic-trained OmniSearch detectors perform on
*real* imagery, per modality (drone RGB, UGV ground-level, thermal IR). This
directly answers the critique that synthetic composite data may not transfer
to real-world detection.

---

## 1. Summary

### Synthetic-only models (the measured gap)

| Modality | Model (training data) | Real test set | Synthetic val | Real recall | Verdict |
|---|---|---|---|---|---|
| Thermal IR | `thermal_yolov8n` (synthetic) | HIT-UAV test (579 img) | R 0.96 | **R 0.31** | Large gap |
| Drone RGB | `survivor_yolov8s_1280` (synthetic) | HERIDAL test (101 img, 337 GT) | R 0.95 | **R 0.22–0.28** | Large gap |
| UGV front | `ugv_front_yolov8s` (synthetic) | COCO val2017 persons (2,668 img) | R 0.33 | **R 0.005** | Near-total failure |
| UGV mast | `ugv_mast_yolov8n` (synthetic) | COCO val2017 persons (2,668 img) | R 0.23 | **R 0.015** | Near-total failure |
| (baseline) | stock `yolov8s` (real COCO) | COCO val2017 persons | — | R 0.73, mAP50 0.83 | Reference |

### After real-data fine-tuning (the fix, now applied to all three modalities)

| Modality | Model (training data) | Real test set | Real recall before → after | Synthetic val after |
|---|---|---|---|---|
| Thermal IR | `thermal_real_yolov8n` (real HIT-UAV) | HIT-UAV test | 0.31 → **0.86** | — |
| Drone RGB | `survivor_heridal_yolov8s_1280` (synthetic + HERIDAL mix) | HERIDAL test | 0.22 → **0.82** (P 0.64, conf 0.15) | R 0.93 (was 0.95) |
| UGV front | `ugv_front_yolov8s_real` (synthetic + COCO mix) | COCO holdout | 0.005 → **0.57** (mAP50 0.64) | R 0.95 (was 0.33) |
| UGV front | `ugv_front_yolov8s_real` (synthetic + COCO mix) | Real Malibu terrain photos + real person cutouts (185 img) | 0.035 → **0.63** (mAP50 0.62, P 0.89) | — |

**Headline:** the synthetic-only models lose roughly 60–95% of their recall
when moved from their own synthetic validation sets to real imagery. Mixing
real data into training recovers most of that gap in every modality we tried
— drone RGB recall on real SAR imagery went from 22% to 82% — while keeping
(or even improving) performance on the synthetic simulator distribution.

---

## 2. Thermal IR — HIT-UAV test split (579 real drone IR images)

| Model | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|
| `thermal_yolov8n` (synthetic Gaussian-blob renderer) | 0.287 | 0.314 | 0.167 | 0.041 |
| `thermal_real_yolov8n` (trained on HIT-UAV train) | 0.889 | 0.861 | 0.923 | 0.495 |

- The synthetic thermal renderer draws survivors as radially symmetric warm
  Gaussian blobs; real IR people are small anisotropic silhouettes with limbs,
  partial occlusion, and background thermal clutter (roads, vehicles, warm
  rocks). The synthetic model finds only ~1 in 3 real people and produces
  ~2.5 false alarms per true detection.
- The real-trained model confirms the task itself is very learnable
  (mAP50 0.92) — the gap is a *data* problem, not a model-capacity problem.

## 3. Drone RGB — HERIDAL test split (101 real 4000x3000 wilderness SAR photos, 337 person boxes)

HERIDAL is the closest public dataset to our deployment domain: real drone
photos at ~50 m over Mediterranean wilderness, annotated for person search
and rescue. Match IoU 0.15 (deployment `person_match_iou`).

| Inference config | conf 0.05 | conf 0.15 | conf 0.35 |
|---|---|---|---|
| Full frame @1280 | R 27.9% / P 11.8% | R 21.7% / P 25.3% | R 14.5% / P 34.8% |
| Tiled 2x2 @1280 | R 15.1% / P 2.6% | R 11.6% / P 5.7% | R 7.7% / P 11.3% |
| Tiled 4x4 @1280 | R 5.3% / P 0.3% | R 3.0% / P 0.5% | R 2.1% / P 1.0% |

- Synthetic val recall for the same model/config is 95%; on real imagery it
  is 22–28% at usable confidence levels — a ~70-point drop.
- Tiling *hurts* on HERIDAL, the opposite of the synthetic result. Cause:
  HERIDAL GT boxes have a median long axis of 66 px at 4000 px, so a
  full-frame resize to 1280 px puts people at ~21 px — inside the model's
  trained size range (5–60 px) — while 4x4 tiles put them at ~77 px, *outside*
  it. The model does not generalize above its trained size band, which is
  itself a symptom of overfitting to the synthetic size distribution.
- Precision is low across the board: real terrain (rocks, bushes, shadows,
  logs) generates many false positives never seen in the composited
  backgrounds.

### 3b. After fine-tuning on synthetic + HERIDAL mix (`survivor_heridal_yolov8s_1280`)

15 epochs at 1280 px on a mixed dataset (synthetic survivor composites +
HERIDAL train split), HERIDAL val as validation. Same test protocol.

| Inference config | conf 0.05 | conf 0.15 | conf 0.35 |
|---|---|---|---|
| Full frame @1280 | R 85.8% / P 50.3% | R 81.6% / P 64.1% | R 71.8% / P 78.8% |
| Tiled 2x2 @1280 | R 87.8% / P 42.2% | R 84.0% / P 56.2% | R 79.5% / P 70.5% |

- Real-SAR recall at usable confidence went from **22% to 82–84%**, precision
  from 25% to 56–64%. The mixed recipe avoided catastrophic forgetting: on the
  synthetic val split the fine-tuned model still scores R 92.6% @1280 full
  (pre-fine-tune: 95.1%) with <8 px bucket at 79.6%.
- Tiling no longer hurts (it slightly helps at 2x2), because HERIDAL-sized
  people are now inside the trained size distribution.

## 3c. Drone RGB through REAL smoke — FLAME 3 composites (230 frames, 483 people)

No public dataset has people annotated under wildfire smoke, so we built one:
real SARD person cutouts composited onto real prescribed-burn drone frames
(FLAME 3 / Sycan Marsh, DJI M30T at 55–100 m). Person size is computed from
per-image EXIF altitude and gimbal pitch; per-paste real-smoke opacity is
estimated from the scene and the person's visibility attenuated accordingly
(a person behind dense smoke is faint, as in reality). 40 person-free fire
frames serve as a false-positive control. Detections on the DJI on-screen
overlay text are excluded. Build script: `scripts/build_flame_composites.py`.

At conf 0.35, tiled 4x4 @1280 (best recall config on 4000 px frames):

| Model | Recall overall | clear | light smoke | heavy smoke | Precision | FP per person-free fire frame |
|---|---|---|---|---|---|---|
| `survivor_yolov8s_1280` (synthetic only) | 67.5% | 66% | 68% | 81% | 15.1% | 8.1 |
| `survivor_heridal_yolov8s_1280` (fine-tuned) | 68.5% | 68% | 68% | 77% | **36.3%** | **2.9** |

- **Recall through real smoke is respectable and roughly flat across smoke
  density** — light/heavy-smoke placements are attenuated but detectable, and
  the model finds ~2/3 of them. (The heavy-smoke bucket is small, n=26.)
- **The real cost of fire scenes is false positives, not misses**: the
  synthetic-only model hallucinates ~8 "survivors" per person-free fire frame
  at conf 0.35 (burning brush, hot spots, smoke wisps). Real-data fine-tuning
  cuts that to ~2.9 and more than doubles precision at equal recall.
- Full-frame @1280 recall is much lower (~10–15%) on these 4000 px frames
  because people shrink to a few pixels — tiled inference is mandatory at
  M30T resolutions, matching the deployment configuration.

## 4. UGV — real ground-level Malibu-area terrain (185 composite images, 226 people)

COCO (used below in 4a) has almost no visual relationship to a wilderness UGV
deployment — it is dominated by indoor rooms, city streets and kitchens.
To get an eval domain that actually matches the deployment context, we built
a second real-image UGV set: real SARD person-cutout photos composited onto
**37 real ground-level photographs of Malibu Canyon, Malibu Creek State Park,
Topanga State Park and Point Mugu State Park** (Wikimedia Commons, CC-licensed
— chaparral hillsides, dry grass, oak-lined canyons, fire roads, one set of
frames after a burn). People are sized with the exact pinhole model used to
generate synthetic UGV training data (5–30 m range, front camera, 70° HFOV) —
only the *background texture* is now a real photo instead of a NAIP zoom-crop
or procedural gradient. Build script: `scripts/build_ugv_malibu_composites.py`.

| Model | Precision | Recall | mAP50 |
|---|---|---|---|
| `ugv_front_yolov8s` (synthetic only) | 0.201 | 0.035 | 0.021 |
| `ugv_front_yolov8s_real` (synthetic + COCO mix) | **0.888** | **0.628** | **0.621** |

- The synthetic-only model collapses on real terrain texture (3.5% recall) —
  worse than its own synthetic val (32.6%) because real chaparral has much
  richer color/lighting variation than the procedural/NAIP-crop backgrounds
  it was trained on, even though the *person* compositing recipe is identical.
- The COCO-mix fine-tune, despite never seeing a single wilderness photo
  during fine-tuning, generalizes well here: 62.8% recall at 88.8% precision.
  This is a more encouraging and more relevant number than the COCO result in
  4a, because the background domain now matches deployment.
- Annotated TP (green) / FN (yellow) / FP (red) images for all 185 frames are
  exported to `~/Documents/omnisearch_capstone/ugv_malibu_annotated/` — most
  misses are people almost fully occluded by brush at the far end of the
  range band (25–30 m); false positives are mostly clumps of burned/dark
  branches with a person-like silhouette.

### 4a. UGV — COCO val2017 person subset (2,668 real ground-level images, boxes ≥ 8 px)

Kept as a secondary, larger-scale (but domain-mismatched) real-image sanity
check, and as the source of the fine-tuning mix in 4b below.

| Model | Precision | Recall | mAP50 |
|---|---|---|---|
| `ugv_front_yolov8s` (synthetic) | 0.010 | 0.005 | 0.001 |
| `ugv_mast_yolov8n` (synthetic) | 0.015 | 0.015 | 0.001 |
| stock `yolov8s` (COCO-pretrained, reference) | 0.831 | 0.733 | 0.830 |
| `ugv_front_yolov8s` on its own synthetic val | 0.356 | 0.326 | 0.236 |
| `ugv_mast_yolov8n` on its own synthetic val | 0.233 | 0.233 | 0.123 |

- The UGV models essentially do not detect real people at all (< 2% recall).
  Fine-tuning on synthetic composites *destroyed* the COCO person knowledge
  the base model started with (catastrophic forgetting): the stock model
  scores mAP50 0.83 on the same images.
- Caveat: COCO is general-purpose imagery (urban/indoor scenes included), not
  wilderness UGV footage, so this is a domain-shifted test for a
  wilderness-trained model. But the stock baseline shows the images are not
  intrinsically hard — and the UGV models are also the weakest of our models
  on their *own* synthetic val (R 0.23–0.33), so they were fragile to begin
  with. See section 4 above for the Malibu-terrain replacement of this test.

### 4b. After fine-tuning on synthetic + COCO mix (`ugv_front_yolov8s_real`)

| Test set | Precision | Recall | mAP50 |
|---|---|---|---|
| COCO person holdout (real) | 0.671 | 0.572 | 0.640 |
| Real Malibu terrain (real) | 0.888 | 0.628 | 0.621 |
| Own synthetic UGV val | 0.975 | 0.948 | 0.980 |

- Real recall went from 0.5% to **57–63%** on both real-image tests (COCO
  holdout and real Malibu terrain), and the synthetic val score *also* jumped
  from 33% to 95% — the original UGV model was undertrained, and the mixed
  data fixed both problems at once.

## 5. Interpretation

1. **The simulation-loop use case is unaffected.** Inside the simulator, the
   detectors run on frames rendered by the same pipeline they were trained
   on; the measured degradation curves (altitude, smoke, fire) remain valid
   as a *sensor model* for the RL policy.
2. **Real-world detection claims are not supported by the synthetic-only
   models.** The gap is 60–95 points of recall depending on modality.
3. **Real data fixes it — now demonstrated on all three modalities.**
   - Thermal: HIT-UAV retrain holds 86% recall on real IR imagery.
   - Drone RGB: synthetic + HERIDAL mix goes 22% → 82% real recall while
     keeping 93% on the synthetic simulator distribution.
   - UGV: synthetic + COCO mix goes 0.5% → 57% recall on COCO and 3.5% → 63%
     recall on real Malibu terrain photos (the domain-relevant test), and
     33% → 95% on synthetic val.
4. **Real wildfire scenes stress precision, not recall.** On FLAME 3
   composites the models find people through real light/heavy smoke at ~2/3
   recall, but the synthetic-only model produces ~8 phantom detections per
   person-free fire frame; real-data fine-tuning cuts that ~3x. Any real
   deployment claim should quote the FP-per-frame number, and thermal should
   remain the lead sensor inside active fire/smoke.
5. **Honest framing for the report/presentation:** "synthetic-trained CV as a
   sensor model for RL simulation, with measured sim-to-real gaps and a
   demonstrated fix — mixed synthetic + real fine-tuning — validated on
   thermal (HIT-UAV), drone RGB (HERIDAL, FLAME 3), and UGV (real Malibu
   terrain photos, COCO)."

## 6. Artifacts

| File | Contents |
|---|---|
| `reports/thermal_real_eval.json` | Both thermal models on HIT-UAV test |
| `reports/heridal_real_eval.json` | Drone model on HERIDAL, full + 4x4 tiled |
| `reports/heridal_real_eval_2x2.json` | Drone model on HERIDAL, full + 2x2 tiled |
| `reports/ugv_real_eval.json` | UGV models on COCO persons |
| `reports/ugv_synthetic_val.json` | UGV models on own synthetic val (contrast) |
| `reports/ugv_malibu_real_eval_synthetic_only.json` | Synthetic-only UGV model on real Malibu terrain composites |
| `reports/ugv_malibu_real_eval_finetuned_coco_mix.json` | Fine-tuned UGV model on real Malibu terrain composites |
| `reports/stock_coco_person_eval.json` | Stock YOLOv8s baseline on COCO persons |
| `reports/heridal_real_eval_finetuned.json` | Fine-tuned drone model on HERIDAL |
| `reports/ugv_real_eval_finetuned.json` | Fine-tuned UGV model on COCO holdout + synthetic val |
| `reports/flame_composite_eval_synthetic.json` | Synthetic-only drone model on FLAME composites |
| `reports/flame_composite_eval_finetuned.json` | Fine-tuned drone model on FLAME composites |
| `reports/recall_by_size_heridal_ft.json` | Fine-tuned drone model on synthetic val (forgetting check) |
| `models/survivor_heridal_yolov8s_1280.pt` | Drone model fine-tuned on synthetic + HERIDAL |
| `models/ugv_front_yolov8s_real.pt` | UGV model fine-tuned on synthetic + COCO |
| `scripts/eval_heridal.py` | HERIDAL evaluation script (VOC XML, full/tiled) |
| `scripts/build_flame_composites.py` | FLAME 3 real-smoke composite builder |
| `scripts/eval_flame_composites.py` | FLAME composite eval (recall by smoke band, FP control) |
| `scripts/build_ugv_malibu_composites.py` | Real-Malibu-terrain UGV composite builder |
| `data/source_cache/heridal/` | HERIDAL dataset (8.3 GB, train+test) |
| `data/cv_train/ugv_real_coco/` | COCO person subset in YOLO format |
| `data/cv_train/flame_composites/` | FLAME 3 composites (230 frames, 483 people, per-paste metadata) |
| `data/cv_assets/malibu_terrain_photos/` | 37 real Malibu-area terrain photos (Wikimedia Commons, `SOURCES.tsv` has attribution) |
| `data/cv_train/ugv_malibu_real/` | UGV Malibu-terrain composites (185 images, 226 people, per-paste metadata) |
