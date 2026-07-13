# Weekly Progress Report — June 16–25, 2026

**Project:** OmniSearch — Heterogeneous Air-Ground Robotic Swarms for Wildfire Survivor Search  
**Author:** Oleksii Lavrenin  
**Period:** Monday Jun 16 – Wednesday Jun 25, 2026

---

## Results

| Milestone | Status |
|-----------|--------|
| CV detector evaluated on realistic wildfire scenes | **Achieved** |
| CV integrated as alternative RL perception backend | **Achieved** |
| False-positive perception model implemented and tested | **Achieved** |
| Training data quality improved (compositing, hard negatives, augmentation) | **Achieved** |
| Altitude-aware survivor sizing (physics-based camera model) | **Achieved** |

---

## What Was Done

### Computer Vision — Full pipeline from data quality to RL integration

---

### Dataset Construction

We create our own synthetic training dataset by compositing human cutouts and fire effects onto aerial backgrounds. The dataset is generated programmatically — the script produces a new dataset from source assets each time it runs, so it is fully reproducible from the source data + a random seed.

**Source data (all publicly available):**

| Source | What it provides | License |
|--------|-----------------|---------|
| [SARD](https://github.com/BerkeleyAutomation/SARD) (Search And Rescue Dataset) | Top-down UAV images of people in wilderness terrain. We extract RGBA person cutouts using GrabCut segmentation. | Academic / open |
| [NAIP](https://naip-usdaonline.hub.arcgis.com/) (USDA National Agriculture Imagery Program) | 0.6 m/px aerial ortho-imagery of real terrain (Malibu, CA). Used as backgrounds. | US Government / public domain |
| [VisDrone](https://github.com/VisDrone/VisDrone-Dataset) | Drone images with annotated vehicles, pedestrians, etc. We extract non-person vehicle crops as hard-negative decoys. | Academic / open |

**Compositing pipeline (what happens for each generated image):**

1. **Background selection** — 85% real NAIP aerial tiles (random crop, random 90° rotation, color jitter), 15% procedural terrain-like noise for diversity
2. **Altitude sampling** — each image simulates a specific drone altitude drawn uniformly from the operational flight envelope (20–50 m). This determines the camera footprint and survivor pixel size via the pinhole camera model.
3. **Survivor placement** — 1–3 SARD person cutouts placed at random positions. Pixel size is derived from the sampled altitude (24–60 px at 640px image, 65° FOV) rather than a uniform random range — this matches the actual size distribution the drone sees in deployment.
4. **Artifact reduction** — three techniques applied to each placed survivor:
   - *Color harmonization* — shifts the survivor's color mean toward the local background patch (reduces "bright blob on muted terrain" artifact)
   - *Alpha-edge erosion* — erodes and feathers the GrabCut hard boundary (removes telltale edge ring)
   - *Altitude-scaled resolution blur* — Gaussian blur derived from ground sample distance at the sampled altitude (higher altitude = more blur, matching the coarser effective resolution)
5. **Wildfire effects** — 65% of images get procedural fire overlays:
   - Burn scars and active flames rendered UNDER survivors
   - Smoke rendered OVER survivors (the hard occlusion case)
   - 20% of fire images get heavy smoke columns directly on each survivor
6. **Hard negatives** — 20% of images include 1–3 non-human decoy objects (VisDrone vehicles, SARD rejected cutouts, or procedural rock/debris blobs) composited with the same pipeline but carrying NO label
7. **Negative-only images** — 12% of images are pure background with no survivors at all (teaches that terrain alone does not imply a person)
8. **Boundary cases** — 10% of survivor placements overlap the image edge; labels use the clipped visible bounding box

**Dataset splits:**
- NAIP tiles are geographically split (80% train / 20% val by tile index) so the model is validated on terrain it has never trained on
- SARD assets are split (80% train / 20% val) so it is validated on body poses it has never trained on
- Current generated dataset: 800 train + 100 val images (configurable via `--n-train` / `--n-val`)

**Output format:** Standard YOLO detection format (images + label .txt files with normalized bounding boxes, class 0 = person).

**Dataset publication plan:** The generated dataset (images + labels + source asset manifest) will be uploaded to **Zenodo** (free, DOI-citable, CC-licensed) alongside the generation script so anyone can reproduce or extend it. The Zenodo record will include:
- Generated training/validation image sets
- The source SARD cutouts used (with provenance)
- The NAIP tile crops used (public domain)
- The generation script and config (for reproducibility)
- A datasheet documenting composition, intended use, and limitations

---

### Training Curriculum

The training uses **transfer learning** — we do NOT train from scratch:

1. **Start:** Pre-trained YOLOv8n weights (`yolov8n.pt`) from Ultralytics, trained on the COCO dataset (80 classes including "person"). This gives the model general object detection knowledge learned from millions of eye-level photos.

2. **Fine-tune:** We fine-tune the full model on our generated synthetic dataset (top-down aerial survivors on NAIP terrain with fire/smoke). This adapts the model from COCO's eye-level distribution to our specific deployment distribution:
   - Top-down viewing angle (not side-view)
   - Altitude-aware small objects (24–60 px at realistic flight levels, not 100–500+ px)
   - Aerial background (terrain, vegetation, not streets/rooms)
   - Fire/smoke occlusion (unique to our domain)

3. **Result:** The fine-tuned model (`models/survivor_yolov8n.pt`) produces high-confidence detections (0.9+) on in-distribution data where stock COCO YOLO only gives 0.2–0.7 confidence and frequently misses small survivors entirely.

**Why this approach works:**
- Training from scratch would require 10–100x more data
- COCO pre-training provides robust low-level features (edges, textures, body parts) that transfer well to the aerial domain
- Fine-tuning on our specific distribution closes the domain gap cheaply (20 epochs, ~500 images, runs on CPU in minutes)

---

### Quantitative Evaluation

Built a systematic evaluation (`scripts/evaluate_cv_perception.py`) measuring detector quality across scenarios:

| Metric | Clear weather | Fire/smoke | Notes |
|--------|:---:|:---:|-------|
| Precision | 0.97 | 0.97 | Very few false positives |
| Recall | 0.94 | 0.71 | Smoke occlusion is the hard case |
| FP rate | 0.1/frame | 0.1/frame | Manageable for RL |
| Spatial error | 3.6 px mean | — | Bounding box center accuracy |
| Temporal stability | Stable | — | Consistent across consecutive frames |

**Conclusion:** Detector is reliable enough for RL integration. The remaining gap (0.71 recall under heavy smoke) is addressed by the augmentation improvements.

---

### CV-RL Integration

- Added a switchable `detection_backend="cv"` to the RL scenario so real YOLOv8 inference can replace the abstract probabilistic camera model during training
- Allows the policy to train against real CV noise characteristics instead of a hand-tuned Bernoulli model

### False-Positive Perception Model (Decoy Landmarks)

- Added non-survivor decoy objects (rocks, debris, animals) that drones can misclassify as survivors with configurable probability
- Ground robots waste trips investigating false reports — decoys can never be confirmed
- Enables evaluation of coordination robustness: "how does the strategy degrade under noisy perception?"
- Configurable: `n_decoys`, `drone_false_positive_rate`, `r_decoy_pursuit_penalty`
- OFF by default — existing trained policies are unaffected

---

## Progress Assessment

| Project Component | Progress | Notes |
|-------------------|----------|-------|
| CV perception pipeline | **85%** | Training, evaluation, RL integration, noise model all done |
| False-positive robustness | **90%** | Model built and tested; ready for training experiments |
| RL policy convergence | 95% | No change this period (achieved last week) |
| Comms dropout evaluation | 80% | Sweep scheduled next |
| Web viewer | 70% | No change this period |
| Final report / paper | 40% | Metrics infrastructure ready |

**Overall: ~85% complete**

---
