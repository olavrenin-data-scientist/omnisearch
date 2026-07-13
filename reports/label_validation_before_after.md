# Physical-Scale Correctness — Before/After Validation

Validator: `scripts/validate_labels.py` (rules R1–R4, see script docstring).
CSVs: `reports/label_validation_baseline.csv` (before), `reports/label_validation_after.csv` (after).

## Rules

| Rule | Meaning |
|---|---|
| R1 size | box long axis > 2.0 m, or nadir short axis > 0.7 m |
| R2 coherence | aerial: frame max/min survivor pixel-size ratio > 2.0; UGV: implied person long axis outside [1.3, 2.2] m |
| R3 metadata | frame missing GSD / per-object scale metadata |
| R4 loose box | stored box-vs-alpha-mask IoU < 0.9 |

## Offender counts (flagged frames)

| Dataset | Frames | R1 before → after | R2 before → after | R3 before → after | R4 before → after |
|---|---|---|---|---|---|
| survivor (drone) | 2,400 | 964 → **0** | 649 → **0** | 0 → 0 | n/a¹ → **0** |
| survivor_naip (drone) | 590 | 0 → **0** | 175 → **0** | 590 → **0** | n/a¹ → **0** |
| thermal (validated, unchanged) | 1,200 | 0² → 0 | 0 → 0 | 0 → 0 | n/a¹ → n/a¹ |
| survivor_ground (legacy UGV) | 480 | 0 | 0 | 480 → *removed*³ | n/a¹ |
| ugv front+mast | 3,600 | 0 | 0 | 3,264 → **0** | n/a¹ → **0** |
| **Total flags** | | | | | **6,122 → 0** |

¹ Legacy frames had no stored mask IoU; the validator reports them as "unknown" rather than offending. All regenerated visual datasets now persist `mask_iou` per box (all = 1.0 by construction). Thermal boxes are physics-derived warm spots with no sprite alpha, so IoU stays "unknown" there by design.
² Thermal warm-spot boxes legitimately spread beyond the 0.7 m body cross-section; the nadir short-axis rule applies only to visual aerial boxes.
³ `survivor_ground/` was superseded by the regenerated `ugv/front` + `ugv/mast` sets and deleted per the fix plan.

## What changed in the generators

**Drone (`scripts/train_survivor_detector.py`)**
- Labels are the tight bbox of the sprite's final (visible, in-frame) alpha mask — never the padded paste rectangle. `mask_iou` is persisted per box.
- Aspect-preserving scaling: sprites are uniformly scaled to fit the physics box, never stretched.
- Nadir viewpoint realism: standing survivors use a synthesized top-down head+shoulders blob; prone survivors use the full-body sprite rotated to its ground heading (near-axis-aligned at nadir so the AABB stays ~0.5 x 1.8 m).
- Per-frame coherence: within-frame max/min sqrt(box-area) ratio capped (generator bound 1.7 < audit rule 2.0).
- Non-overlapping labels via rejection sampling.
- Metadata unconditional: every frame gets a JSON sidecar with `gsd_m`, `altitude_m`, `view`, and per-box `{pose, heading_deg, px, w_m, h_m, mask_iou}` (legacy mode writes `gsd_m: null`, which the validator flags).

**UGV (`scripts/train_ugv_detector.py`)**
- Per-object scale metadata: `range_m`, `m_per_px`, `implied_long_m`, `foreshortening`, `mask_iou` per box; camera params per frame.
- Single scalar body jitter (aspect never distorted); sprite long axis anchored to the 1.75 m body length through the pinhole range physics.
- Acceptance check: a sprite is only placed if its final box implies a 1.35–2.15 m person (de-foreshortened for the mast camera).
- Physically sized decoys: bush/rock 0.5–2 m, vehicle 3–8 m, projected through the same range model.
- `--epochs 0` = generate-only mode.

## Test status
- `tests/test_object_scale_physics.py` — 24 passed (incl. new tight-box, frame-coherence, nadir-short-axis, and UGV implied-size tests).
- `tests/test_cv_compositing.py`, `tests/test_cv_cropping.py`, `tests/test_cv_person_detection.py`, `tests/test_thermal_model.py` — 100 passed total.

## Annotated exports
Refreshed at `exports/cv_annotated_all/{survivor, survivor_naip, thermal, ugv}` — every frame shows per-box real-world sizes, GSD/camera header, and a terrain scale bar; frames lacking scale metadata would get a red warning banner (none remain).

Note: the datasets were regenerated with corrected labels but the deployed YOLO weights have **not** been retrained on them (per plan).
