# Probabilistic UAV Perception: Environment Factor

## Purpose

This document defines the literature-informed terrain multiplier for the
abstract probabilistic UAV survivor-perception model. The goal is not to
reproduce a particular computer-vision detector. Instead, the model should
provide an interpretable probability that a survivor would be detected during
one UAV observation while avoiding unsupported terrain-specific constants.

The implemented simulator uses a single categorical multiplier,
`q_environment`, for land-cover effects. It intentionally bundles terrain,
background clutter, concealment, vegetation, and water-surface ambiguity into
one factor because the current terrain cache does not expose separate canopy,
submersion, glare, pose, or contrast fields. This is less mechanistic than a
future spatial occlusion model, but it keeps the current RGB abstract perception
model calibrated and avoids multiplying several weakly supported penalties.

The simulator exposes two abstract UAV perception modes: `rgb` and
`rgb_thermal`. In the first RGB+thermal implementation step, `rgb_thermal` is a
named sensor-stack mode that intentionally reuses the same equations and values
as `rgb`. Thermal-specific terms for smoke penetration, body/background thermal
contrast, and heat crossover should be calibrated separately before changing the
probability model.

## Per-Step Detection Model

For UAV $i$, time step $t$, and map cell or survivor location $x$, use

\[
p_{i,t}(x) =
\mathbb{1}[x \in F_{i,t}]
q_{\mathrm{base}}
q_{\mathrm{altitude}}
q_{\mathrm{range}}
q_{\mathrm{environment}}
q_{\mathrm{fire}},
\]

where $F_{i,t}$ is the camera footprint. All factors are constrained to
$[0,1]$. This decomposition has two useful properties:

1. The reason for a difficult observation remains visible in diagnostics.
2. The same probability can be used by both the survivor Bernoulli detection
   and the confidence-map update, preventing the observation and reward models
   from disagreeing.

This document covers `q_environment` only. Altitude, footprint-relative range,
and fire/smoke terms are calibrated separately.

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

## Implemented Categorical `q_environment`

The current terrain cache stores categorical land cover rather than measured
canopy, submersion, glare, or body-visibility fields. For this reason, the
implemented RGB abstract model uses one categorical environment factor:

| Land-cover class | Evidence mapping | Implemented `q_environment` |
|---|---|---:|
| Road | Low vegetation / unobstructed reference | 1.00 |
| Open ground | Low vegetation / unobstructed reference | 1.00 |
| Brush | SAVIOUR 2024 medium vegetation probability-of-detection proxy | 0.71 |
| Forest | SAVIOUR 2024 high vegetation probability-of-detection proxy | 0.56 |
| Rock | Average of low and medium clutter classes | 0.86 |
| Water | SeaDronesSee swimmer AP50 reference | 0.78 |

The simulator stores these values in land-cover order
`road, open, brush, forest, rock, water`:

```text
q_environment = (1.00, 1.00, 0.71, 0.56, 0.86, 0.78)
```

SAVIOUR-style probability of detection should be interpreted as an operational
terrain-quality proxy rather than pure physical occlusion. It likely includes
altitude, viewing geometry, camera/operator performance, repeated viewing, and
environmental concealment. SeaDronesSee likewise reports detector AP rather
than a Bernoulli detection probability; the water value is therefore a compact
proxy for glare, wave clutter, partial submersion, and maritime background
ambiguity.

The Baur occlusion curve above remains useful for a future spatial canopy model,
but it is not multiplied into the current `q_environment` values. Applying both
would double-count vegetation difficulty.

At minimum, report sensitivity experiments with brush and forest occlusion
shifted by plus or minus 10 percentage points. The policy should not be claimed
to be robust to vegetation if conclusions change substantially within that
plausible uncertainty range.

## Implementation Requirements

1. Use the same `q_environment` in survivor detection and the confidence-map
   update.
2. Expose `q_environment` separately from the final probability in diagnostics.
3. Preserve the legacy `drone_cover_detection_factors` config key as an alias
   for old scenario files, but use `drone_environment_detection_factors` as the
   canonical name.
4. Do not additionally multiply the Baur occlusion curve into the current
   categorical environment factors.
5. Prefer measured canopy, submersion, glare, or visibility arrays in future
   terrain caches, then replace the categorical multiplier with a spatial model.
6. Treat repeated observations as correlated in future refinements; multiplying
   independent per-step Bernoulli misses can otherwise make cumulative
   confidence rise too quickly during nearly identical consecutive frames.
