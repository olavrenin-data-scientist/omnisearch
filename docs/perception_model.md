# Probabilistic UAV Perception Model

## Purpose

OmniSearch uses an abstract probabilistic UAV perception model during MARL
training. The model does not try to reproduce a particular computer-vision
detector frame by frame. Instead, it provides an interpretable probability that
a survivor at a location would be detected by a UAV observation, using factors
that correspond to altitude, footprint position, land cover, smoke, and fire.

The same probability is used in two places:

1. **Survivor scouting:** each UAV-survivor pair draws a Bernoulli detection.
2. **Confidence-map updates:** each grid cell accumulates the probability that
   a survivor there would already have been detected.

This keeps the detection model, observation confidence, and confidence-based
reward terms aligned. A policy is therefore rewarded for moving through areas
where the model predicts a real increase in detection probability, not merely
for touching new binary coverage cells.

The default training mode is `rgb_thermal` mode as a conservative RGB+thermal stack approximation. The simulator also exposes an `rgb` only mode.

## Implemented Per-Pass Probability

For UAV $i$, time step $t$, and a survivor or grid cell at location $x$,
the instantaneous detection probability is

$$
p_{i,t}(x)
=
\mathbb{1}[x \in F_{i,t}]
\;q_{\mathrm{alt}}(h_{i,t})
\;q_{\mathrm{range}}(d_{i,t}(x))
\;q_{\mathrm{env}}(x)
\;q_{\mathrm{fire/smoke}}(i,t,x),
$$

with multiplication between the four quality factors. $F_{i,t}$ is the UAV
camera footprint, $h_{i,t}$ is altitude above ground level, and
$d_{i,t}(x)\in[0,1]$ is the normalized distance from the footprint center to
the footprint edge. The implemented code clamps the final product to $[0,1]$
and sets it to zero outside the footprint.

In code this is implemented by `_drone_detection_components()` for survivor
detections and `_uav_cell_detection_probability()` for dense confidence-map
updates in `envs/wildfire_search.py`.

## Camera Footprint

The UAV footprint is modeled as a circle:

$$
r_{\mathrm{footprint}}(h)
=
h \tan(\theta / 2),
$$

where $h$ is the current AGL altitude and $\theta$ is the camera field of
view. The canonical default is

```text
DRONE_CAMERA_FOV_DEG = 90.0
```

so the default footprint radius is approximately equal to altitude in meters.
For example, a UAV at $20\,\mathrm{m}$, $35\,\mathrm{m}$, or
$50\,\mathrm{m}$ AGL has an approximate footprint radius of
$20\,\mathrm{m}$, $35\,\mathrm{m}$, or $50\,\mathrm{m}$, respectively.
The environment also supports a minimum footprint floor for training stability
when configured.

This circular footprint is an abstraction. It ignores rectangular sensor shape,
camera yaw, lens distortion, motion blur, and perspective variation across the
image.

## Factor 1: Altitude Quality

Altitude affects detection because higher flights produce fewer pixels per
meter and smaller apparent target size. If no legacy override is supplied, the
simulator uses a fitted altitude-quality curve:

$$
q_{\mathrm{alt}}(h)
=
\exp\left(-0.00238255\;\max(h-30,0)^{1.26540149}\right).
$$

The implementation labels this model `sambolek_ivasickos_2021_fit`. It is a
shifted-Weibull fit to the altitude-dependent detection values reported by
[Sambolek and Ivasic-Kos (2021)](https://doi.org/10.1109/ACCESS.2021.3063681)
for UAV person detection at 15, 30, 45, 60, and 75 m. The shift keeps quality
at 1.0 up to 30 m and then decays smoothly with altitude.

Example values:

| Altitude AGL | $q_{\mathrm{alt}}$ |
|---:|---:|
| 20 m | 1.000 |
| 30 m | 1.000 |
| 35 m | 0.982 |
| 50 m | 0.900 |
| 75 m | 0.745 |
| 90 m | 0.655 |

The default flight levels are currently:

```text
DRONE_FLIGHT_LEVELS_M = (20.0, 35.0, 50.0)
```

so the trained UAVs operate in the high-quality part of the fitted curve while
still trading off altitude, footprint size, and movement-energy cost.

## Factor 2: Footprint-Range Quality

Detection is strongest near nadir and weaker near the footprint edge. The code
uses a quadratic radial model:

$$
q_{\mathrm{range}}(d)
=
1 - (1-q_{\min})d^2,
\qquad
q_{\min}=0.70.
$$

Here $d=0$ at the footprint center and $d=1$ at the footprint boundary.
Thus a central pass has quality 1.0, while an edge pass retains 70% of the
central quality.

The 70% edge floor is motivated by
[*Zone Evaluation: Revealing Spatial Bias in Object Detection*](https://doi.org/10.1109/TPAMI.2024.3409416),
which reports that outer image regions often retain roughly 70-80% of center
performance depending on detector and dataset. That study reports region-wise
AP rather than a radial probability law, so the quadratic interpolation is a
simulator assumption.

## Factor 3: Land-Cover / Environment Quality

The terrain cache currently stores categorical land cover, not measured canopy
occlusion, submersion, glare, pose, or target-background contrast. OmniSearch
therefore uses a single categorical multiplier
$q_{\mathrm{env}}$ that bundles vegetation, clutter, concealment, and
background ambiguity:

| Land-cover class | Evidence mapping | Implemented $q_{\mathrm{env}}$ |
|---|---|---:|
| Road | unobstructed reference | 1.00 |
| Open ground | unobstructed reference | 1.00 |
| Brush | medium vegetation terrain-quality proxy | 0.71 |
| Forest | high vegetation terrain-quality proxy | 0.56 |
| Rock | low-to-medium clutter proxy | 0.86 |
| Water | maritime swimmer/background proxy | 0.78 |

The values are stored in land-cover order:

```text
road, open, brush, forest, rock, water
q_environment = (1.00, 1.00, 0.71, 0.56, 0.86, 0.78)
```

### Derivation and interpretation

No reviewed study supplies universal Bernoulli multipliers for every land-cover
class used by OmniSearch. The implemented values are therefore calibrated as
interpretable terrain-quality proxies:

- Road and open ground are treated as the unobstructed reference.
- Brush and forest use SAVIOUR-style medium/high vegetation probability of
  detection as an operational terrain-quality proxy. These values should not be
  interpreted as pure physical occlusion; they likely include sensor quality,
  search path, viewing geometry, repeated observations, and operator effects.
- Rock is assigned an intermediate value between low-clutter and medium-clutter
  conditions, representing broken visual background and reduced contrast.
- Water uses the SeaDronesSee swimmer AP50 reference as a compact proxy for
  glare, wave clutter, partial submersion, and maritime background ambiguity.

The legacy config key `drone_cover_detection_factors` is still accepted as an
alias, but `drone_environment_detection_factors` is the canonical name.

## Factor 4: Smoke and Fire Quality

The smoke/fire factor is sampled along the straight optical path from UAV to
survivor or grid cell using `drone_perception_path_samples` points
(default: 8). The code combines mean path smoke, target smoke, flame glare, and
heat distortion:

$$
q_{\mathrm{fire/smoke}}
=
q_{\mathrm{smoke}}
\;q_{\mathrm{glare}}
\;q_{\mathrm{heat}},
$$

again with multiplication between factors.

### Smoke

The RGB smoke-quality model is

$$
q_{\mathrm{smoke}}^{\mathrm{RGB}}(s)
=
(1-s)^{1.24},
$$

where $s\in[0,1]$ is the combined smoke load:

$$
s = 0.65\,\overline{s}_{\mathrm{path}} + 0.35\,s_{\mathrm{target}}.
$$

The exponent 1.24 is fitted to the clear-normalized Faster R-CNN recalls
reported by Liu et al. (2020),
[*Analysis of the Influence of Foggy Weather Environment on the Detection Effect
of Machine Vision Obstacles*](https://doi.org/10.3390/s20020349). The model
uses the fog/smoke visibility result as a contrast-loss proxy; it is not a
wildfire-plume radiative-transfer model.

Example values:

| Smoke load $s$ | $q_{\mathrm{smoke}}^{\mathrm{RGB}}$ |
|---:|---:|
| 0.00 | 1.000 |
| 0.25 | 0.700 |
| 0.50 | 0.423 |
| 0.75 | 0.179 |
| 1.00 | 0.000 |

### Fire glare and heat shimmer

Fire affects RGB perception in two additional ways:

$$
q_{\mathrm{glare}}
=
1 - 0.35\,
\max(\mathrm{fire}_{\mathrm{path,max}},
     \mathrm{fire}_{\mathrm{near\ target}}),
$$

$$
q_{\mathrm{heat}}
=
1 - 0.20\,
\overline{\mathrm{fire}}_{\mathrm{path}}.
$$

The glare term represents saturated, visually confusing pixels around active
flame. The heat term is a simple proxy for heat-induced refractive distortion.
Both are clamped to $[0,1]$. These are simulator-level heuristics rather than
direct fits to one dataset.

## RGB+Thermal Mode

The `rgb_thermal` mode keeps the same altitude and range factors as RGB but
adds a conservative thermal benefit for smoke and background ambiguity:

$$
q_{\mathrm{env}}^{\mathrm{RGB+Thermal}}
=
\min(1,\;1.30\,q_{\mathrm{env}}^{\mathrm{RGB}}),
$$

$$
q_{\mathrm{smoke}}^{\mathrm{RGB+Thermal}}
=
\eta + (1-\eta)q_{\mathrm{smoke}}^{\mathrm{RGB}},
\qquad
\eta=0.6.
$$

The environment boost is motivated by multimodal RGB/thermal detection gains
reported in arXiv:2203.04567 and IEEE LRA DOI 10.1109/LRA.2019.2900907. The
smoke blend follows standard sensor-reliability fusion intuition: the thermal
channel preserves a residual sensing path when RGB contrast is degraded by
visible smoke.

This mode is intentionally conservative. It does not yet include
thermal-specific body/background contrast, thermal crossover, sensor NETD, or
false positives from warm non-human objects. Those effects are represented in
the separate simulated thermal sensor backend, not in the default MARL
probability product.

## Stochastic Survivor Detection

For a true survivor $k$, the simulator computes
$p_{i,t}(x_k)$ for every UAV. A survivor is detected by UAV $i$ when

$$
z_{i,t,k} \sim \mathrm{Bernoulli}(p_{i,t}(x_k)).
$$

If a detection occurs, the recorded detection confidence for that UAV-survivor
pair is the probability value $p_{i,t}(x_k)$; otherwise it is zero. The
probability components are exposed through `drone_perception_debug()` and the
diagnostics under fields such as `probability`, `distance_factor`,
`environment_factor`, `fire_smoke_factor`, and `altitude_quality`.

## Confidence-Map Update

For every grid cell $x$, the confidence map $C_t(x)\in[0,1]$ is interpreted
as the cumulative probability that a survivor at $x$ would already have been
detected by at least one UAV. Given per-UAV probabilities $p_{i,t}(x)$, the
team update is

$$
C_{t+1}(x)
=
1 -
(1-C_t(x))
\prod_i (1-p_{i,t}(x)).
$$

Equivalently, the residual miss probability is multiplied by the miss
probability of each current UAV observation. This lets repeated observations
increase confidence, but with diminishing returns as $C_t(x)$ approaches 1.

When communication-aware per-agent maps are enabled, each UAV first updates its
own confidence memory using the same per-cell probabilities:

$$
C_{i,t+1}(x)
=
1-(1-C_{i,t}(x))(1-p_{i,t}(x)).
$$

Connected agents then synchronize their map memories. This is why communication
dropout changes the information available to each policy without changing the
underlying physical detection probability.

## Literature Evidence

The following studies motivate the current factors and, equally important,
their limitations. Most report AP, recall, or mission-level probability of
detection rather than a simulator-ready per-pass Bernoulli probability.

| Study | Setting | Quantitative evidence | Use and limitation |
|---|---|---|---|
| [Sambolek and Ivasic-Kos (2021), *Automatic Person Detection in Search and Rescue Operations Using Deep CNN Detectors*](https://doi.org/10.1109/ACCESS.2021.3063681) | UAV person detection across flight altitudes | Detection results at 15, 30, 45, 60, and 75 m; detections below or equal to 30 m are reported as accurate. | Used to fit the shifted-Weibull altitude-quality curve. The exact fit is a simulator proxy for resolution loss, not a camera-specific guarantee. |
| [Liu et al. (2020), *Analysis of the Influence of Foggy Weather Environment on the Detection Effect of Machine Vision Obstacles*](https://doi.org/10.3390/s20020349) | Machine-vision obstacle detection in foggy conditions | Clear-normalized Faster R-CNN recall decreases with fog density. | Used to fit the RGB smoke exponent $1.24$. Fog is used as a visibility/contrast proxy for smoke. |
| [Zheng et al. (2024), *Zone Evaluation: Revealing Spatial Bias in Object Detection*](https://doi.org/10.1109/TPAMI.2024.3409416) | Object detector performance by image region | Outer image regions commonly retain roughly 70-80% of center AP. | Motivates the 0.70 footprint-edge floor. The radial quadratic form is our interpolation. |
| [Baur et al. (2024), *Modeling the Effect of Vegetation Coverage on UAV-Based Object Detection*](https://doi.org/10.3390/rs16122046) | RGB UAV imagery with synthetic grass covering a PFM-1 landmine | Recall was approximately 0.99, 0.89, 0.68, 0.37, 0.13, and 0.04 at 0%, 20%, 30%, 40%, 50%, and 60% occlusion. The fitted sigmoid had $R^2=0.996$. | Provides a useful vegetation-occlusion curve. It is not directly multiplied into the current categorical factors to avoid double counting. |
| [Song et al. (2025), *An Infrared Dataset for Partially Occluded Person Detection in Complex Environment for Search and Rescue*](https://doi.org/10.1038/s41597-025-04600-0) | Nadir thermal UAV imagery of lying people at 30, 50, and 70 m with measured body occlusion | Median detection AP was 0.786 at 70% occlusion, 0.616 at 80%, 0.523 at 90%, and 0 at 100%. | Closest direct UAV-human occlusion experiment. It supports future thermal occlusion modeling but uses AP, not per-pass Bernoulli probability. |
| [Pagacz and Witczuk (2023), *Estimating Ground Surface Visibility on Thermal Images From Drone Wildlife Surveys in Forests*](https://doi.org/10.1016/j.ecoinf.2023.102379) | Thermal drone wildlife surveys across forest habitats | Detectability was described as proportional to percentage of ground visible from above. | Supports a future spatial canopy/visible-ground map instead of one fixed forest multiplier. |
| [Schedl, Kurmi, and Bimber (2021), *An Autonomous Drone for Search and Rescue in Forests Using Airborne Optical Sectioning*](https://doi.org/10.1126/scirobotics.abg1188) | Thermal search for hidden people in different forest types using multi-view optical sectioning | The system found 38 of 42 hidden people. Predefined paths found 30 of 34; adaptive resampling found 8 of 8. | Shows repeated multi-view thermal processing can overcome forest occlusion. It should not be interpreted as ordinary single-image probability. |
| [Nilsen et al. (2026), *Beyond Platform Type: Effects of Vegetation Density, Sensor Modality, and Search Strategy on Aerial SAR Performance*](https://doi.org/10.66050/sja2vn07) | Professional SAR sorties with human targets | Mean probability of detection exceeded 83%; UAS probability of detection in high-density forest was 73.3%. | Mission-level sanity check, but it combines search strategy, repeated views, operators, and sensor modality. |
| [Broyles, Hayner, and Leung (2022), *WiSARD*](https://doi.org/10.1109/IROS47612.2022.9981298) | Paired visual and thermal UAV imagery across forest, fields, rock, coast, and snow | Aggregate YOLOv5 recall was 0.845 for RGB and 0.959 for thermal. | Supports RGB+thermal improvement, but does not provide calibrated per-terrain Bernoulli probabilities. |
| [Riz et al. (2023), *The MONET Dataset*](https://openaccess.thecvf.com/content/CVPR2023W/MULA/papers/Riz_The_MONET_Dataset_Multimodal_Drone_Thermal_Dataset_Recorded_in_Rural_CVPRW_2023_paper.pdf) | Thermal UAV people and vehicles over dirt-road and runway scenes | Same-scene person AP was approximately 33.0-45.6, while cross-scene AP dropped to approximately 15.0-19.2. | Shows strong background-domain effects even in thermal imagery. |
| [Varga et al. (2022), *SeaDronesSee*](https://openaccess.thecvf.com/content/WACV2022/papers/Varga_SeaDronesSee_A_Maritime_Benchmark_for_Detecting_Humans_in_Open_Water_WACV_2022_paper.pdf) | RGB UAV detection of swimmers and floaters in open water | Strong baseline AP50 was 78.1 for swimmers and 82.4 for floaters. | Motivates a water/background factor, but water detectability depends on glare, submersion, altitude, and target state. |

## Vegetation Occlusion Curve for Future Spatial Models

Baur et al. measured binary detection recall after placing synthetic grass over
the target segmentation mask. Their fitted recall curve can be written as

$$
q_{\mathrm{Baur}}(O)
=
\frac{1}{1+\exp(0.132379(O-35.80863))},
$$

where $O$ is the percentage of the target projection that is occluded. Because
the fitted value at zero occlusion is 0.991 rather than exactly 1, a normalized
occlusion multiplier is

$$
q_{\mathrm{occlusion}}(O)
=
\min\left(1,\frac{q_{\mathrm{Baur}}(O)}{q_{\mathrm{Baur}}(0)}\right).
$$

| Occlusion $O$ | Normalized multiplier |
|---:|---:|
| 0% | 1.000 |
| 10% | 0.977 |
| 20% | 0.898 |
| 30% | 0.689 |
| 40% | 0.368 |
| 50% | 0.134 |
| 60% | 0.039 |

This curve should be treated as a conservative RGB vegetation proxy, not a
human-specific calibration. It is not multiplied into the current categorical
environment factors because doing so would double-count vegetation difficulty.
A future terrain cache with canopy, submersion, glare, or visible-ground arrays
should replace the categorical multiplier with a spatial field $q_{\mathrm{env}}(x)$.

## Assumptions and Limitations

- The multiplicative factorization assumes conditional independence between
  altitude, range, environment, smoke, and fire. Real detectors have nonlinear
  interactions between these effects.
- Literature values often report AP, recall, or mission probability of
  detection. The simulator maps them to probability multipliers only after
  interpreting them as relative quality factors.
- Repeated observations are treated as independent Bernoulli misses in the
  confidence update. This is convenient and interpretable but may overestimate
  cumulative confidence when consecutive frames are nearly identical.
- The default abstract model does not simulate bounding boxes, false positives,
  detector thresholds, image texture, motion blur, or target pose.
- The current land-cover multiplier is categorical. Future terrain caches should
  expose spatial canopy, visible-ground, submersion, glare, and contrast fields
  so the model can use measured $q_{\mathrm{env}}(x)$ values.
