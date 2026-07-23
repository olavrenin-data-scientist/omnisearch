# Detection Modality — Data & Camera Report

---

## 1. Do we have side angles for drones?

**Yes — now implemented.** The drone camera supports both **nadir (top-down)** and **oblique (side-angle)** views.

| Mode | Camera angle | What survivors look like | Training fraction |
|------|:----------:|--------------------------|:-----------------:|
| **Nadir** | 90° down | Small circular/foreshortened blobs (20–60 px) | 75% of images |
| **Oblique** | 45–75° from horizontal (15–45° tilt from nadir) | Elongated/partially upright (taller than wide) | 25% of images |

### Implementation (`scripts/train_survivor_detector.py`)

- New `--oblique-frac` CLI argument (default 0.25 = 25% of training images).
- Physics function `oblique_survivor_size()` computes pixel width and height using:
  - `width_px = body_width_m × (image_size / footprint_m)` — same as nadir
  - `height_px = (person_height × sin(tilt) + body_width × cos(tilt)) × px_per_m` — foreshortened
- Tilt angle sampled uniformly from 15°–45° per image.
- Metadata sidecar includes `"oblique": true/false` and `"tilt_deg"` for analysis.

**Retrained drone detector with 25% oblique views** (2,000 train / 400 val, 50 epochs, NAIP backgrounds + VisDrone decoys):

| Metric | Value |
|--------|:-----:|
| Precision | 0.863 |
| Recall | 0.778 |
| mAP50 | 0.818 |
| mAP50-95 | 0.348 |

Weights: `models/survivor_yolov8n.pt`

| Platform | Camera angle | What survivors look like |
|----------|:----------:|--------------------------|
| **Drone (nadir)** | 90° down | Small circular blobs (20–60 px) |
| **Drone (oblique)** | 45–75° from horizontal | Elongated upright shapes (wider bounding box) |
| UGV Front | 0° (horizontal) | Full upright person (30–350 px tall) |
| UGV Mast | ~45° oblique | Partially foreshortened (40–250 px) |

### Simulation support (`detection/simulation_adapter.py`)

The oblique view is also wired into the simulation, so oblique training is
exercised at deployment, not just in the dataset:

- New `camera_tilt_deg` parameter on `SimulationCvAdapter` (0 = nadir, up to 60°).
- When tilted, rendered survivors use the same foreshortening physics as
  training: `apparent_height = person_height × sin(tilt) + nadir_height × cos(tilt)`.
- Exposed in trajectory export: `--cv-camera-tilt 30` renders a 30°-tilted view.

---

## 2. Which data is used by UGV CV?

The UGV computer vision detector uses the **same source assets** as the drone but with different compositing:

| Data source | How it's used |
|-------------|---------------|
| **SARD** (54 real person cutouts) | Pasted **upright** (full height visible), not top-down |
| **NAIP** (67 train / 4 val tiles) | Tight crop + heavy upscale to simulate ground-level terrain (~1m height) |
| **Procedural backgrounds** | Color noise + brush texture (~30% of images when NAIP not used) |
| **VisDrone decoys** (500 vehicle crops) | Hard negatives — composited identically but unlabeled |

Training set: 1,500 images per camera (front + mast). Validation: 300 images per camera on **geographically unseen terrain**.

---

## 3. Which data does thermal use?

The thermal model now has **two components**:

### 3a. Detection Model (Physics-Based — no images needed)

The detection model is entirely physics-based — closed-form equations on simulator state:

| Input | Source | Formula |
|-------|--------|---------|
| Body temperature | Constant (310K / 37°C) | — |
| Ground temperature | Computed from fire/burn grids | `ambient + fire_intensity × (450K − 293K)` |
| Smoke transmittance | Computed from smoke grid | `max(exp(−0.4 × smoke_load), 0.70)` |
| Distance/altitude | From drone position | Quadratic falloff within footprint |

No training data, no neural network. Detection decisions are stochastic draws against physically computed probability.

### 3b. Thermal Image Renderer (NEW — `detection/thermal_renderer.py`)

A new **simulated thermal image renderer** generates grayscale TIR-like visualizations:

| Feature | Implementation |
|---------|---------------|
| Human body | Bright Gaussian blob (310 Kelvin / 37°C) against cooler terrain |
| Active fire | Intense white hotspots (600 Kelvin / 327°C) |
| Burned ground | Warm gray (330 Kelvin / 57°C) |
| Ambient terrain | Dark gray (293 Kelvin / 20°C) |
| Sensor noise | Gaussian noise (σ = 2 Kelvin) |
| Point spread function | 1.2px Gaussian blur |
| False-color options | Iron, white-hot, black-hot colormaps |

These images serve as:
1. **Visualization** in the web viewer (thermal overlay on drone view)
2. **Training data** for the thermal-specific YOLO detector (below)
3. **Report figures** for comparison between modalities

### 3c. Generated TIR Dataset & Trained Thermal Detector

A dedicated dataset generator (`scripts/generate_thermal_dataset.py`) produced
**1,200 labeled TIR images** (1,000 train / 200 val, 512×512 grayscale, YOLO format)
covering six scenarios: clear, fire, burned, smoke, fire+smoke, burned+smoke.
Survivors are placed uniformly across the camera footprint at physics-derived
sizes (blob radius scales with altitude), and the renderer applies scene-adaptive
contrast (AGC) like a real thermal camera.

A YOLOv8-nano was trained on this dataset (30 epochs):

| Metric | Value |
|--------|:-----:|
| Precision | 1.000 |
| Recall | 0.963 |
| mAP50 | 0.965 |
| mAP50-95 | 0.843 |

Weights: `models/thermal_yolov8n.pt`. The near-perfect scores reflect the
simplicity of the simulated task (bright Gaussian blobs on cooler terrain);
real TIR imagery would be substantially harder.

### 3d. Thermal YOLO Wired Into the Simulation

The trained thermal detector is now usable inside the simulation as an
alternative to the physics probability model:

- New `thermal_detector` parameter on `SimulationCvAdapter`:
  - `"physics"` (default) — closed-form probability model, no images.
  - `"yolo"` — renders a simulated TIR frame each step and runs
    `models/thermal_yolov8n.pt` on it; boxes map back to world coordinates
    and feed the same fusion pipeline (`cv+thermal`) unchanged.
- Exposed in trajectory export: `--detection-mode thermal --thermal-detector yolo`.
- Covered by an end-to-end test (`TestThermalYoloBackend`) that renders a
  frame, runs the detector, and verifies the world-coordinate mapping.

Usage:
```python
from detection.thermal_renderer import render_thermal_frame, render_thermal_with_colormap

gray = render_thermal_frame(
    image_size=512,
    drone_xy=(0.0, 0.0),
    footprint_world=50.0,
    survivors=[{"world_xy": (0.1, -0.2)}],
    fire_intensity_grid=fire_grid,
    seed=42,
)
colored = render_thermal_with_colormap(gray, colormap="iron")
colored.save("thermal_view.png")
```

---

## Files Modified/Created

| File | Change |
|------|--------|
| `scripts/train_survivor_detector.py` | Added oblique camera physics (`oblique_survivor_size`), `--oblique-frac` CLI arg, and oblique compositing in `_generate_split` |
| `detection/thermal_renderer.py` | **NEW** — Simulated thermal image renderer with colormaps |
| `tests/test_cv_compositing.py` | Added `TestThermalRenderer` and oblique angle tests |
