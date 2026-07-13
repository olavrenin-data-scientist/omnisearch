# CV Dataset Composition Report — Hard Negatives, Decoys & Wildfire Conditions

**Scope:** All generated CV training/validation datasets under `data/cv_train/`.
**Method:** Counts are read directly from the per-image metadata JSON sidecars
(`*/labels/*.json`). Where a property is not stored per-image, the generation-time
probability is reported instead (and flagged as an **estimate**).

---

## 1. Summary

| Question | Answer |
|----------|--------|
| Hard-negative decoy classes | **3**: vehicles (VisDrone crops), animals (procedural), colourful human-scale objects (procedural) |
| Images containing decoys (drone) | **839** exact — 687 train (34%) + 152 val (38%), recorded per-image as `n_decoys` |
| Negative-only images (no survivor) | **725** across all datasets (see §2) |
| Images with fire/smoke (drone) | **1,576** (train+val, `has_fire` flag) |
| Thermal images by wildfire scenario | 6 scenarios, balanced (~1/6 each) — see §4 |

---

## 2. Negative Images (no survivor / empty label)

Exact counts from metadata (`n_survivors == 0` for drone/thermal, `is_negative` for UGV):

| Dataset | Split | Images | Negatives | % |
|---------|-------|-------:|----------:|----:|
| Drone survivor | train | 2,000 | 227 | 11.4% |
| Drone survivor | val | 400 | 42 | 10.5% |
| UGV front | train | 1,500 | 147 | 9.8% |
| UGV front | val | 300 | 26 | 8.7% |
| UGV mast | train | 1,500 | 126 | 8.4% |
| UGV mast | val | 300 | 37 | 12.3% |
| Thermal TIR | train | 1,000 | 98 | 9.8% |
| Thermal TIR | val | 200 | 22 | 11.0% |
| **Total** | | **7,200** | **725** | **~10%** |

> Note: negative-only images are pure background (or wildfire scene with no person).
> They teach the detector that "terrain / fire / vehicles ≠ person".

---

## 3. Hard-Negative Decoys

The drone dataset uses **three decoy classes**, all composited with the **same
pipeline** as survivors (color harmonization, alpha-edge erosion, resolution blur)
but carrying **no label**. Reviewer feedback (A.-K. Schuetz) motivated the two
human-scale classes: vehicles alone are large and mostly white/gray, so a model
could pass by learning "colourful human-sized blob = person" or by keying on
compositing artifacts. The human-scale decoys share the survivors' size band,
saturated clothing palette and contrast, so silhouette is the only remaining
separator. Decoys are also placed so they never overlap a labeled survivor box.

| Class | Source | Physical size | Share of decoys |
|-------|--------|---------------|----------------:|
| Vehicle | `data/cv_assets/visdrone_decoys/` (500 RGBA crops) | cars 3.5–5.2 m (70%), vans/pickups 5.2–8 m (22%), bus/truck 8–13 m (8%) | ~50% |
| Animal | procedural (body + head blob, fur palette, dorsal stripe) | 0.5–1.8 m long axis | ~25% |
| Colourful object | procedural (tarp/jacket/gear: saturated 1–2-tone irregular blob) | 0.5–1.8 m long axis | ~25% |

All sizes are scaled through each frame's GSD, so decoys obey the same physics as
survivors. Vehicle sizing was rebalanced from the earlier uniform 3–8 m (+15%
buses) mix, which read as visually oversized next to 0.5–2 m people.

### Images containing decoys (exact, from `n_decoys` metadata)

| Dataset | `decoy_frac` | Images | Images w/ ≥1 decoy | Decoy objects (veh / animal / colourful) |
|---------|-------------:|-------:|-------------------:|------------------------------------------|
| Drone survivor train | 0.35 | 2,000 | 687 (34%) | 702 / 352 / 323 |
| Drone survivor val | 0.35 | 400 | 152 (38%) | 162 / 75 / 66 |
| UGV front + mast | 0.15 | 3,600 | ~540 (estimate) | bushes + vehicles (see UGV generator) |
| Thermal TIR | — (no decoys) | 1,200 | 0 | — |

Every decoy's type, pixel bbox and physical long axis are stored in the per-image
metadata (`"decoys"` list), so these counts are fully auditable. A companion
script `scripts/eval_decoy_fp.py` measures how often each decoy class triggers a
false 'person' detection — the artifact-shortcut test the reviewer asked for.

---

## 4. Wildfire Conditions (fire / smoke / burned)

### 4a. Drone survivor dataset — `has_fire` flag

| Split | Images | With fire/smoke | % |
|-------|-------:|----------------:|----:|
| train | 2,000 | 1,314 | 65.7% |
| val | 400 | 262 | 65.5% |

Procedural wildfire effects (fire glow, smoke plumes) are overlaid at
`--fire-frac 0.65`.

### 4b. Thermal TIR dataset — scenario label + continuous `smoke_load`

The thermal generator samples one of **6 wildfire scenarios** per image (near-balanced):

| Scenario | Train | Val |
|----------|------:|----:|
| clear | 173 | 35 |
| smoke | 165 | 42 |
| fire | 150 | 35 |
| burned | 159 | 28 |
| fire+smoke | 173 | 35 |
| burned+smoke | 180 | 25 |
| **Total** | **1,000** | **200** |

Smoke intensity distribution (train split, bucketed by `smoke_load`):

| Level | `smoke_load` | Images | % |
|-------|--------------|-------:|----:|
| none | < 0.05 | 482 | 48.2% |
| light | 0.05–0.3 | 151 | 15.1% |
| moderate | 0.3–0.6 | 287 | 28.7% |
| heavy | > 0.6 | 80 | 8.0% |

### 4c. UGV datasets

UGV metadata stores only `camera`, `n_persons`, and `is_negative` — there is **no
per-image fire/smoke flag**, so exact counts are unavailable. The UGV generator applies
fire/smoke at `fire_frac = 0.40` (≈40% of images, estimated).

---

## 5. Generation Parameters (defaults)

| Parameter | Drone | UGV | Thermal |
|-----------|------:|----:|--------:|
| `neg_frac` (negative-only) | 0.12 | 0.10 | 0.10 |
| `decoy_frac` (hard negatives) | 0.35 (current drone set) | 0.15 | — |
| `fire_frac` (fire/smoke) | 0.65 | 0.40 | scenario-based |

Sources: `scripts/train_survivor_detector.py`, `scripts/train_ugv_detector.py`,
`scripts/generate_thermal_dataset.py`.

---

## 6. Data-Quality Note / Recommendation

The drone generator now persists `"n_decoys"` and a full `"decoys"` list (type,
bbox, physical size) plus `"has_fire"` in every metadata sidecar, so drone counts
in this report are exact. The UGV generator still applies decoys and fire/smoke
stochastically without persisting them — adding the same fields there would make
those counts exact as well.
