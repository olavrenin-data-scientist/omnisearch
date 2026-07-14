# Probabilistic UAV Perception: Background and Occlusion

## Purpose

This document defines a literature-informed treatment of background and physical
occlusion for the abstract probabilistic UAV survivor-perception model. The goal
is not to reproduce a particular computer-vision detector. Instead, the model
should provide an interpretable probability that a survivor would be detected
during one UAV observation while avoiding unsupported terrain-specific
constants.

The central design decision is to model **background clutter** and **physical
occlusion** separately:

- **Background** changes how visually distinct a visible person is from the
  surrounding scene. Its effect depends on sensor modality, temperature,
  lighting, clothing, season, and detector training.
- **Occlusion** is the fraction of the person's visible projection hidden by
  vegetation or another object. It can be estimated spatially from vegetation
  or canopy data and converted into a detection multiplier.

Conflating the two effects in a single land-cover factor makes values difficult
to interpret and can double-count vegetation. For example, applying both a low
`forest` background factor and a canopy-occlusion factor would penalize the same
forest twice.

## Per-Step Detection Model

For UAV $i$, time step $t$, and map cell or survivor location $x$, use

\[
p_{i,t}(x) =
\mathbb{1}[x \in F_{i,t}]
q_{\mathrm{base}}
q_{\mathrm{altitude}}
q_{\mathrm{range}}
q_{\mathrm{background}}
q_{\mathrm{occlusion}}
q_{\mathrm{fire}},
\]

where $F_{i,t}$ is the camera footprint. All factors are constrained to
$[0,1]$. This decomposition has two useful properties:

1. The reason for a difficult observation remains visible in diagnostics.
2. The same probability can be used by both the survivor Bernoulli detection
   and the confidence-map update, preventing the observation and reward models
   from disagreeing.

This document changes only the interpretation and proposed values of
`q_background` and `q_occlusion`. Altitude, footprint-relative range, and
fire/smoke terms require their own calibration.

## Evidence From Existing Studies

No study below supplies a universal set of multipliers for road, open ground,
brush, forest, rock, and water. The studies instead provide three kinds of
evidence: measured target occlusion, operational probability of detection, and
performance changes between visual backgrounds or sensor modalities.

### Most Relevant Studies

| Study | Setting | Quantitative evidence | Use and limitation |
|---|---|---|---|
| [Baur et al. (2024), *Modeling the Effect of Vegetation Coverage on UAV-Based Object Detection*](https://doi.org/10.3390/rs16122046) | RGB UAV imagery with synthetic grass covering a PFM-1 landmine | Recall was approximately 0.99, 0.89, 0.68, 0.37, 0.13, and 0.04 at 0%, 20%, 30%, 40%, 50%, and 60% occlusion. The fitted sigmoid had $R^2=0.996$. | Provides an explicit vegetation-occlusion curve. It is a useful conservative shape prior, but the target is a small landmine rather than a person. |
| [Song et al. (2025), *An Infrared Dataset for Partially Occluded Person Detection in Complex Environment for Search and Rescue*](https://doi.org/10.1038/s41597-025-04600-0) | Nadir thermal UAV imagery of lying people at 30, 50, and 70 m with measured body occlusion | Median detection AP was 0.786 at 70% occlusion, 0.616 at 80%, 0.523 at 90%, and 0 at 100%. | The closest direct UAV-human occlusion experiment. AP is not a per-frame probability, and the detector was trained specifically on thermal occlusion examples. |
| [Pagacz and Witczuk (2023), *Estimating Ground Surface Visibility on Thermal Images From Drone Wildlife Surveys in Forests*](https://doi.org/10.1016/j.ecoinf.2023.102379) | Thermal drone wildlife surveys across forest habitats | Detectability was described as proportional to the percentage of ground visible from above; detection of large warm animals over open terrain was nearly certain. | Supports spatial visible-ground or canopy-gap modeling instead of one fixed forest multiplier. The targets were animals and the sensor was thermal. |
| [Schedl, Kurmi, and Bimber (2021), *An Autonomous Drone for Search and Rescue in Forests Using Airborne Optical Sectioning*](https://doi.org/10.1126/scirobotics.abg1188) | Thermal search for hidden people in different forest types using multi-view optical sectioning | The system found 38 of 42 hidden people. Predefined paths found 30 of 34; adaptive resampling found 8 of 8. | Demonstrates that repeated multi-view processing can overcome forest occlusion. It should not be interpreted as ordinary single-image detection probability. |
| [Nilsen et al. (2026), *Beyond Platform Type: Effects of Vegetation Density, Sensor Modality, and Search Strategy on Aerial SAR Performance*](https://doi.org/10.66050/sja2vn07) | 48 professional SAR sorties with 251 human targets | Mean probability of detection exceeded 83%; UAS probability of detection in high-density forest was 73.3%. | A valuable mission-level sanity check. It combines search path, operator, sensor, and repeated observations, so 0.733 is not a per-step forest factor. |
| [Broyles, Hayner, and Leung (2022), *WiSARD*](https://doi.org/10.1109/IROS47612.2022.9981298) | Paired visual and thermal UAV imagery across forest, fields, rock, coast, and snow | Aggregate YOLOv5 recall was 0.845 for RGB and 0.959 for thermal. The authors show thermal misses over rock when target and background temperatures become similar. | Confirms that background effects are sensor- and condition-dependent. The paper does not report calibrated per-terrain recall values. |
| [Riz et al. (2023), *The MONET Dataset*](https://openaccess.thecvf.com/content/CVPR2023W/MULA/papers/Riz_The_MONET_Dataset_Multimodal_Drone_Thermal_Dataset_Recorded_in_Rural_CVPRW_2023_paper.pdf) | Thermal UAV people and vehicles over dirt-road and runway scenes | Same-scene person AP was approximately 33.0-45.6, while cross-scene person AP dropped to approximately 15.0-19.2. | Shows a large background-domain effect even with the same modality. These AP values cannot be copied directly into a Bernoulli probability. |
| [Varga et al. (2022), *SeaDronesSee*](https://openaccess.thecvf.com/content/WACV2022/papers/Varga_SeaDronesSee_A_Maritime_Benchmark_for_Detecting_Humans_in_Open_Water_WACV_2022_paper.pdf) | RGB UAV detection of swimmers and floaters in open water | The strongest reported baseline obtained AP50 of 78.1 for swimmers and 82.4 for floaters; accuracy also varied substantially with altitude and viewing angle. | Water is not intrinsically represented by one low background factor. Submersion, glare, target type, altitude, and detector training must be modeled separately. |

## Occlusion Model From Baur et al.

Baur et al. measured binary detection recall after placing synthetic grass over
the target segmentation mask. Their fitted recall curve can be written as

\[
q_{\mathrm{Baur}}(O) =
\frac{1}{1 + \exp\left(0.132379(O - 35.80863)\right)},
\]

where $O$ is the percentage of the target projection that is occluded. The
curve changes most rapidly around 36% occlusion and approaches zero after about
60% occlusion.

Because the fitted value at zero occlusion is 0.991 rather than exactly 1, use a
normalized multiplier:

\[
q_{\mathrm{occlusion}}(O) =
\min\left(1,
\frac{q_{\mathrm{Baur}}(O)}{q_{\mathrm{Baur}}(0)}
\right).
\]

This leaves unoccluded detection quality to `q_base`, altitude, and range rather
than silently reducing every observation by 0.9%.

| Occlusion $O$ | Normalized multiplier $q_{\mathrm{occlusion}}$ |
|---:|---:|
| 0% | 1.000 |
| 5% | 0.992 |
| 10% | 0.977 |
| 20% | 0.898 |
| 25% | 0.814 |
| 30% | 0.689 |
| 35% | 0.531 |
| 40% | 0.368 |
| 50% | 0.134 |
| 60% | 0.039 |

This curve should be described as a **conservative RGB vegetation proxy**, not
as a human-specific calibration. A person has different size, pose, color, and
salient features from a landmine. Song et al. show that an occlusion-trained
thermal person detector can remain effective at much higher occlusion levels.

## Recommended Spatial Representation

### Preferred approach: continuous occlusion map

If vegetation-height or canopy data are available, estimate
$O(x)$ directly for every terrain cell. Ideally, $O(x)$ is the fraction of
the nadir target projection blocked by vegetation taller than the relevant
target profile. Then calculate

\[
q_{\mathrm{occlusion}}(x) =
q_{\mathrm{occlusion}}(O(x)).
\]

Canopy cover is a practical approximation, but it is not identical to body
occlusion. It is more appropriate for a lying or injured person than for a
standing person whose upper body may remain visible above brush. If pose is not
modeled, the resulting uncertainty should be acknowledged.

### Compatibility fallback: land-cover-derived occlusion

Existing terrain caches may contain only categorical land cover. For those
caches, use the following explicit fallback assumptions:

| Land-cover class | Assumed occlusion | Resulting multiplier | Reasoning |
|---|---:|---:|---|
| Road | 0% | 1.000 | The road surface does not physically cover the target. Visual contrast belongs to the background term. |
| Open ground | 5% | 0.992 | Represents sparse incidental vegetation while remaining effectively unoccluded. This is an engineering prior, not a measured class mean. |
| Brush | 25% | 0.814 | Places brush below the terrain builder's forest canopy threshold and within Baur's still-detectable transition region. This is a fallback prior. |
| Forest | 35% | 0.531 | Matches the approximate canopy threshold used to identify forest and the midpoint of Baur's steep transition. Dense forest should use a larger measured value instead. |
| Rock | 0% | 1.000 | Rock may create camouflage or thermal crossover, but it does not itself occlude a nadir-visible target. |
| Water | 0% background occlusion | 1.000 | Water background is not body occlusion. A person in water requires a separate submersion/visibility term. |

These values are deliberately separated into an assumed occlusion percentage
and a literature-derived conversion. The assumed percentages must not be cited
as measurements from Baur et al.

## Recommended Background Factors

For the current RGB abstract model, use neutral background factors until a
terrain-stratified human dataset has been evaluated:

| Land-cover class | Proposed `q_background` |
|---|---:|
| Road | 1.00 |
| Open ground | 1.00 |
| Brush | 1.00 |
| Forest | 1.00 |
| Rock | 1.00 |
| Water | 1.00 |

This does **not** claim that backgrounds are equally difficult. It states that
the available studies do not justify stable, sensor-independent multipliers.
Vegetation difficulty is already represented by `q_occlusion`. Assigning
additional values such as `forest=0.55` would double-count vegetation unless
that factor had been calibrated specifically as residual clutter after
controlling for occlusion.

Future background calibration should stratify held-out aerial human detections
by terrain class and fit relative recall after controlling for occlusion,
altitude, footprint position, pose, smoke, and lighting. Thermal operation must
have a separate model: WiSARD and MONET show that hot rock, sand, road, or other
surfaces can reduce thermal contrast, while the same backgrounds may remain
usable in RGB.

## Proposed Values To Use

The recommended first implementation is:

```text
occlusion_curve                 = baur_normalized
occlusion_percent_road          = 0
occlusion_percent_open          = 5
occlusion_percent_brush         = 25
occlusion_percent_forest        = 35
occlusion_percent_rock          = 0
occlusion_percent_water         = 0

background_factor_road          = 1.00
background_factor_open          = 1.00
background_factor_brush         = 1.00
background_factor_forest        = 1.00
background_factor_rock          = 1.00
background_factor_water         = 1.00
```

The resulting compatibility multipliers are approximately:

```text
road=1.000, open=0.992, brush=0.814,
forest=0.531, rock=1.000, water=1.000
```

The source and confidence of each part are different:

- The **curve coefficients** come directly from Baur et al.
- The **land-cover occlusion percentages** are transparent engineering priors
  for old categorical caches. They should be replaced by spatial canopy or
  vegetation visibility whenever that data is available.
- The **background factors of 1.0** are intentional neutral values. They avoid
  inserting unsupported terrain penalties and prevent double-counting.
- The **forest result of approximately 0.53** should be treated as a
  conservative single-pass RGB prior. Operational search probability can be
  much higher after repeated views, adaptive revisits, thermal sensing, or
  occlusion-removal methods.

At minimum, report sensitivity experiments with brush and forest occlusion
shifted by plus or minus 10 percentage points. The policy should not be claimed
to be robust to vegetation if conclusions change substantially within that
plausible uncertainty range.

## Implementation Requirements

1. Use the same `q_occlusion` and `q_background` in survivor detection and the
   confidence-map update.
2. Store or expose the estimated occlusion percentage separately from the final
   probability for diagnostics.
3. Do not apply both the legacy land-cover detection factors and the new
   occlusion multiplier.
4. Preserve a categorical fallback for old terrain caches, but prefer a
   continuous canopy/visibility array in newly built caches.
5. Report detection, misses, and confidence gain by occlusion bin and terrain
   class so the assumptions can later be replaced by measured calibration.
6. Treat repeated observations as correlated in future refinements; multiplying
   independent per-step Bernoulli misses can otherwise make cumulative
   confidence rise too quickly during nearly identical consecutive frames.
