# Survivor Detection with Computer Vision

How OmniSearch detects survivors in rendered drone imagery, why the naive
approach failed, and what was done to fix it. Honest about what is real vs.
simulated.

---

## Pipeline

Each drone frame is rendered and (optionally) run through a person detector in
[detection/simulation_adapter.py](../detection/simulation_adapter.py):

```
NAIP aerial background crop (real imagery, per terrain bbox)
   → render burn scar + flames           (detection/wildfire_effects.py)
   → paste SARD survivor cutout(s)        (real people, drone-view)
   → render smoke over the scene
   → detect people  ──►  PreliminaryPersonDetector  OR  YOLOv8
```

The detection backend is selectable:

- **`preliminary`** (default) — echoes renderer ground-truth boxes with optional
  noise. Fast, deterministic, no model. For wiring/tests.
- **`yolo`** — real YOLOv8 person detection over the rendered crop.

---

## The problem the naive approach hit

1. **Detection bypassed the image.** The pipeline only ever used the
   ground-truth stub — no actual computer vision ran on the rendered crop.
2. **Survivors are tiny.** At realistic drone altitudes (20–50 m), a person is
   only ~20–48 px in a 512 px crop — **below YOLO's reliable detection floor**.
   Stock YOLOv8 detected ~0 survivors at that size.
3. **Stock YOLO is out-of-distribution.** COCO weights are trained on eye-level
   photos; top-down aerial survivors detect at noisy 0.2–0.7 confidence, often
   missed entirely.

## What was done

1. **Real YOLO backend** wired into the adapter (`--cv-detector yolo`).
2. **Tiled small-object inference** — the frame is sliced into an overlapping
   grid and each tile run at high `imgsz`, giving each tiny survivor more
   effective resolution; boxes are mapped back and de-duplicated with NMS.
   This alone makes ~48 px survivors detectable where single-pass found none.
3. **IoU matching to ground truth** so each detection is tied to a survivor
   index (and unmatched boxes count as false positives).
4. **Fine-tuning on the render distribution**
   ([scripts/train_survivor_detector.py](../scripts/train_survivor_detector.py)).
   Because we control the renderer, we generate labelled composites — SARD
   survivors on terrain-like backgrounds with the same wildfire smoke/flame/burn
   effects, fire centered on survivors — and fine-tune YOLOv8. The detector then
   sees in-distribution data. The adapter auto-uses these weights when present.

## Results

> ⚠️ The table below is on **synthetic (procedural) backgrounds**. These numbers
> are good but **did not transfer to real NAIP** — see "The procedural-background
> trap (and the NAIP fix)" below for the deployment-accurate eval (recall 0.95,
> ~1.15 false positives/crop with the NAIP-trained model, which the adapter now
> auto-prefers).

Fine-tuned **YOLOv8s** (balanced fire data), robust eval — mean over 10 random
assets/positions per scenario (not a single lucky image):

| Scenario | Drone (tiled, aerial) | Ground confirm (single-pass, close) |
|---|---|---|
| Clean | recall 1.00, conf **0.84** | recall 1.00, conf **0.83** |
| Smoke | recall 1.00, conf **0.76** | — |
| Flames | recall 1.00, conf **0.75** | — |
| Fire + smoke | recall 1.00, conf **0.75** | recall 1.00, conf **0.80** |

**Recall is ~1.0 in every condition** — survivors are essentially never missed,
which is the metric that matters for SAR — at a mean confidence of 0.75–0.84
(peaks 0.91–0.95). This is up from stock COCO YOLO's noisy 0.24–0.63 and from
the nano fine-tune's ~0.55–0.63. Verify/visualize with
[scripts/verify_survivor_cv.py](../scripts/verify_survivor_cv.py); montage saved
to `docs/survivor_cv_detection.png`.

**One model serves both agents.** A dedicated nano model trained only on large
(close-range) survivors scored *worse* on ground confirmation (~0.60) than the
general yolov8s (~0.83): model capacity beat size-specialization, and yolov8s's
training scale already overlaps the close-range size. So both the drone scout
and the ground confirm use the same yolov8s weights.

---

## Two-stage detection: drone scouts, ground robot confirms

Both agent types now use real computer vision, matching the project's
scout→confirm story:

| Stage | Agent | View | Why it works |
|---|---|---|---|
| **Scout** | Drone | Top-down aerial, survivor ~20–100 px (tiny) | Tiled high-`imgsz` inference + fine-tuned model |
| **Confirm** | Ground robot | Close-range, survivor 25–40% of frame (large) | Single-pass inference; survivor is already big |

`SimulationCvAdapter.render_ground_confirmation(robot, survivor, ...)` renders a
small-footprint (`2 * view_radius_m`) view centred on the survivor and runs the
detector. The export records these as `cv_ground_confirmations` per frame,
firing only when a ground robot is within ~1.5× its confirmation range of a
scouted-but-unconfirmed survivor (so the extra inference is rare).

**Key subtlety — confirmation uses single-pass, not tiling.** Tiling helps the
drone's tiny survivors but *hurts* the ground robot's large ones: it up-scales
an already-large survivor past the detector's training scale and lowers
confidence (measured 0.29 tiled vs 0.70–0.73 single-pass, peaks ~0.88). So
`render_ground_confirmation` disables tiling for the close-range view.

---

## Where the images come from (real vs. simulated)

| Element | Source | Real or simulated? |
|---|---|---|
| Survivors (people) | **SARD** Search-And-Rescue UAV dataset, GrabCut-segmented cutouts in `data/cv_assets/sard_grabcut/` (54 reviewed) | **Real** drone-view people |
| Terrain background (deployment) | **NAIP** USDA aerial ortho-imagery, fetched per bbox ([detection/naip.py](../detection/naip.py)) | **Real** aerial imagery |
| Terrain background (training) | Procedural color-field noise | **Simulated** |
| Fire / smoke / burn | Rendered by [detection/wildfire_effects.py](../detection/wildfire_effects.py) | **Simulated** overlay |

## The procedural-background trap (and the NAIP fix)

A model fine-tuned on **procedural** (synthetic noise) backgrounds looked great on
synthetic eval (~0.84) but **failed on the real pipeline**: when tested on real
NAIP imagery it fired **~23 false positives per empty crop** — it never learned
what real terrain looks like, so it saw "people" all over rocks and brush. In a
short export run where survivors were rarely in the camera footprint, the result
was *all false positives, zero real survivors detected.*

The fix: generate training composites on **real NAIP crops** as backgrounds
(`--naip-dir`), so the model learns real terrain. Honest eval on real NAIP:

| Model | Real survivor recall | False positives / empty crop |
|---|---|---|
| Procedural-trained | 1.00 | **23.0** |
| **NAIP-trained @ conf 0.40** | 0.95 | **1.15** |

A ~20× drop in false positives at the same recall. **Lesson: always train the
detector on the same imagery it will be deployed on.** The adapter auto-prefers
`models/survivor_naip_yolov8s.pt` and the default confidence threshold is 0.35.

> Caveat: only 4 NAIP tiles (one 1 km area) were available, so the false-positive
> eval is partly in-distribution. On an unseen area FP would be higher; the proper
> next step is NAIP training data spanning many regions.

## Domain-gap limitations (be honest about these)

- **No real people-in-fire imagery exists** in the dataset. "A survivor in fire"
  is a real drone-view person with a *synthetic* flame/smoke layer drawn over
  them — it does not reproduce soot, real smoke turbulence, or thermal effects.
- **NAIP training is in (`--naip-dir`), but only one ~1 km area was available**
  (4 tiles), so the false-positive eval is partly in-distribution. On an unseen
  region false positives would be higher; the next step is NAIP training data
  spanning many regions.
- **Visible-light only.** Real SAR drones see people through smoke with
  **thermal/IR**, not RGB. A thermal channel would be the largest real-world
  robustness gain but needs thermal survivor data we do not have.
- Detection confidence numbers reflect our **simulated** fire, not field wildfire
  conditions.

## Usage

```bash
# Fine-tune the survivor detector on the render distribution (CPU-friendly):
python scripts/train_survivor_detector.py --epochs 30 --n-train 400 --fire-frac 0.7

# Export trajectories with real CV detection:
python scripts/export_trajectories.py --enable-cv --cv-detector yolo \
  --terrain-cache-path data/terrain_cache/<cache>.npz --cv-image-size 1024

# Visual + numeric check under fire and smoke (uses the NAIP-trained model):
python scripts/verify_survivor_cv.py --model models/survivor_naip_yolov8s.pt
```

Generated training data (`data/cv_train/`) and weights (`models/`, `*.pt`) are
git-ignored.
