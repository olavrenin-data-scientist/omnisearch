# OmniSearch Simulation Overview

OmniSearch is a wildfire search-and-rescue simulator built on top of
[VMAS](https://github.com/proroklab/vectorizedmultiagentsimulator), a vectorized
2D multi-agent simulator for efficient MARL experiments. VMAS supplies the
batched 2D physics engine; OmniSearch adds a physically scaled wildfire mission
layer with terrain, fire, smoke, aerial perception, ground traversal, and
survivor scouting/confirmation.

The simulator is intentionally not a full wildfire or robotics digital twin. It
is a **2.5D mission simulator**: robots move in continuous 2D, terrain and fire
live on raster grids, and UAV altitude is tracked as an above-ground scalar. The
goal is to preserve the mission-relevant structure of wildfire SAR while keeping
the environment fast enough for reinforcement learning.

---

## 1. World Model

The environment combines two coordinate systems:

- **Continuous VMAS plane:** UAVs, UGVs, and survivors have continuous
  positions and velocities.
- **Raster map layers:** fire, smoke, land cover, elevation, slope, fuel, and
  inspection confidence are stored on grid cells.

Canonical task defaults are:

| Quantity | Default value |
|---|---:|
| UAVs | 3 |
| UGVs | 2 |
| Survivors | 5 |
| Fire / terrain grid | 128 x 128 |
| Reference terrain grid | 16 x 16 |
| Episode horizon | 500 steps |
| Simulation step | 2 s |
| Default horizon in simulated time | 1000 s |

Many diagnostic and paper runs use a compact stress-test setting:

| Quantity | Diagnostic value |
|---|---:|
| Search area | 500 m x 500 m |
| Grid | 128 x 128 |
| Cell size | 3.9 m |
| Episode horizon | 300 steps |
| Simulated time | 600 s / 10 min |

The compact setting is useful because coverage saturates quickly. That makes it
easier to study overlap, late-stage sparse low-confidence areas, and UAV/UGV
handoff behavior.

## 2. Agent Types

### UAVs

UAVs are aerial search agents. Their horizontal motion is continuous and
holonomic, while altitude is updated by an automatic terrain-following
controller.

| Parameter | Value |
|---|---:|
| Speed | 10 m/s |
| Default flight levels | 20, 35, 50 m AGL |
| Camera field of view | 90 deg |
| Climb rate | 10 m/step |
| Descent rate | 8 m/step |
| Minimum safety clearance | 3 m |
| Fire/smoke clearance thresholds | 25 m, smoke threshold 0.20 |

The controller chooses an altitude that clears terrain and obstacles while
respecting the configured flight range. Horizontal control remains 2D; the
altitude affects footprint size, detection quality, and energy diagnostics.

### UGVs

UGVs are ground confirmation agents. They move on the terrain surface and are
constrained by traversability, terrain speed, slope, and fire-aware route
planning.

| Parameter | Value |
|---|---:|
| Speed | 1.6 m/s |
| Acceleration | 2.0 m/s^2 |
| Lidar range | 20 m |
| Survivor confirmation range | 10 m |
| Arrival slowdown radius | 10 m |
| Arrival damping | 0.6 |

UGVs use local and global A* route hints in several evaluation settings. Route
costs incorporate land cover, slope, water/rock/building blockage, fire, and
smoke. The learned controller still outputs continuous motion; the planner
provides structured navigation information and diagnostics.

## 3. Terrain and GIS Layers

OmniSearch terrain caches are derived from real geospatial data:

- [USGS 3DEP](https://www.usgs.gov/3d-elevation-program/about-3dep-products-services)
  digital elevation products provide meter-valued terrain elevation. The
  commonly used 1/3 arc-second product has approximately 10 m spacing.
- OpenStreetMap contributes roads, water bodies, and building footprints.
- [LANDFIRE](https://www.landfire.gov/fuel) fuel and vegetation products are
  used when available; otherwise the simulator derives fuel density from land
  cover.

The simulator collapses the terrain into six land-cover classes:

| Class | Fire fuel | UGV cost | UGV speed | Interpretation |
|---|---:|---:|---:|---|
| Road | 0.05 | 0.65 | 1.00 | road or trail |
| Open | 0.40 | 1.00 | 0.95 | grass, clearing, sparse cover |
| Brush | 1.10 | 1.50 | 0.80 | chaparral or dense shrub |
| Forest | 1.35 | 2.20 | 0.70 | wooded terrain |
| Rock | 0.00 | 4.00 | 0.00 | impassable rock |
| Water | 0.00 | 8.00 | 0.00 | impassable water |

Slope is derived from the elevation grid and affects both fire spread and ground
mobility. UGVs cannot traverse non-road cells above a maximum slope of 0.70
grade. Below that threshold, slope increases traversal cost and reduces speed:

```text
cost multiplier  = 1 + 2.0 * slope
speed multiplier = terrain_speed / (1 + 0.5 * slope)
```

Moisture is a static cell field derived from terrain and land cover. It is used
as a coarse proxy for fuel wetness, with wetter cells suppressing ignition and
fire intensity.

## 4. Fire Model

Fire is modeled as a stochastic cellular automaton on the raster grid. Each
cell can be unburned, burning, burned out, or non-burning. At every fire update,
currently burning cells send ignition pressure to nearby unburned cells. The
target cell then ignites with a probability determined by the accumulated
pressure from its burning neighbors.

Conceptually, ignition pressure has four physical ingredients:

- **Exposure:** stronger and closer burning neighbors create more heat exposure.
- **Fuel:** brush and forest cells ignite more readily than road or open ground.
- **Moisture:** wetter cells suppress spread and reduce fire intensity.
- **Directionality:** wind and uphill slope bias spread toward downwind and
  upslope neighbors.

The simulator evaluates the eight neighboring cells around each target. For
each burning neighbor, it builds an effective contribution:

```text
neighbor contribution
  = fire_intensity
  * wind_alignment_factor
  * uphill_slope_factor
```

These neighbor contributions are summed and then multiplied by target-cell
factors for fuel, fuel density, moisture, and stochastic variability. The final
spread pressure is converted into an ignition probability using a saturating
Bernoulli form:

```text
p_ignite = 1 - (1 - base_spread_probability) ^ effective_rate
```

This form has a useful interpretation: many weak contributors or one strong
contributor can both raise ignition probability, but the probability remains
bounded by 1.

Canonical values:

| Parameter | Value |
|---|---:|
| Base spread probability | 0.03 |
| Fire update interval | every 5 env steps |
| Wind direction | (1, 0) |
| Wind strength | 0.06 |
| Wind spread weight | 1.25 |
| Slope spread weight | 1.65 |
| Moisture damping | 1.15 |
| Spread variability | 0.55 |
| Initial fire area | 2.5% of grid |

### Wind and slope

Wind is represented as a fixed 2D vector. The default wind direction `(1, 0)`
pushes spread and smoke eastward, with strength `0.06`. For each neighbor
direction, the simulator computes alignment with the wind vector. Downwind
directions receive an exponential boost; upwind directions receive less spread
pressure.

Slope is computed from the elevation grid. Uphill spread receives an exponential
boost, reflecting the fact that flames and preheating tend to accelerate fire
upslope. The model is directional: a cell upslope from a burning neighbor is
easier to ignite than a cell downslope from that same neighbor.

### Fuel, moisture, and intensity

Land cover controls the amount of burnable material. Fuel-rich classes such as
brush and forest have higher fuel factors; rock and water do not burn. Moisture
reduces both ignition probability and fire intensity, acting as a static proxy
for fuel wetness.

Burning cells carry a continuous intensity value in `[0, 1]`. Intensity is not
flame length or heat release rate, but it plays the same conceptual role inside
the simulator: more intense cells spread more strongly and emit more smoke.
Intensity evolves over the cell's burn lifetime, combining fuel/moisture
potential with a lifecycle term that decays as the cell approaches burnout.

### Burn lifetime

When a cell ignites, it receives a land-cover-dependent burn lifetime. After
that many fire updates, it becomes burned out and cannot reignite. Burnout
depends on land cover:

| Class | Burnout updates |
|---|---:|
| Road | 5-20 |
| Open | 5-20 |
| Brush | 20-60 |
| Forest | 60-200 |
| Rock | non-burning |
| Water | non-burning |

This makes forest fires persist longer than grass/open fires and keeps rock and
water as fire barriers.

### Target-area regulation

The simulator includes a bounded target-area regulation term to keep training
episodes useful. Each episode samples a target burned fraction, and the spread
rate is gently boosted when the fire is below that target. This prevents the
fire from immediately dying out or consuming the entire map in most training
runs. It is an engineering device for scenario diversity, not a physical
wildfire process.

The model is conceptually related to classical wildfire spread ideas, especially
the role of fuel, wind, slope, and moisture in the
[Rothermel surface fire spread model](https://research.fs.usda.gov/treesearch/55928).
However, OmniSearch does not implement Rothermel, FARSITE, flame length,
fireline intensity, crown fire, suppression, or atmospheric feedback. The fire
layer is designed to create plausible spatial hazards and smoke fields for
learning and evaluation, not to forecast real wildfire boundaries.

## 5. Smoke Model

Smoke is a dimensionless scalar field on the same grid as fire. It is updated
each environment step through emission, decay, diffusion, smoldering, and wind
advection. Conceptually, the smoke field is a compact visibility layer: it marks
where fire activity has recently degraded the aerial camera view.

The update has five stages:

1. **Active-fire emission:** burning cells add smoke proportional to fire
   intensity and local fuel.
2. **Smolder emission:** cells late in their burn lifetime continue producing
   lower-intensity residual smoke.
3. **Decay:** smoke fades over time, representing dilution and dissipation.
4. **Diffusion:** smoke spreads to neighboring cells by local smoothing.
5. **Advection:** wind shifts smoke downwind.

| Parameter | Value |
|---|---:|
| Active fire smoke emission | 0.18 |
| Smoke decay | 0.985 |
| Smoke diffusion | 0.16 |
| Wind advection strength | 0.30 |
| Smolder smoke emission | 0.04 |
| Smolder decay | 0.995 |
| Smolder start fraction | 0.65 of burn lifetime |

Active fire emits smoke according to:

```text
smoke_added = fire_intensity * smoke_emission * fuel_factor
```

After emission, the field is decayed and smoothed. Diffusion is implemented as
a four-neighbor averaging step, so smoke gradually fills nearby cells rather than
remaining exactly on the fire front. Advection then blends the smoke field in
the wind direction; the default advection strength is stronger than the fire
wind strength because smoke is allowed to drift faster than the fire front.

The smoke field enters perception through the UAV line-of-sight model. The
camera model samples smoke along the straight path from UAV to target and forms
a weighted smoke load:

```text
smoke_load = 0.65 * mean_path_smoke + 0.35 * target_smoke
```

RGB detection quality then decreases as `(1 - smoke_load)^1.24`. Thus smoke can
reduce detection even when the survivor is not directly inside a burning cell,
because smoke along the viewing path still lowers contrast.

The smoke model is not calibrated to PM2.5, optical depth, plume rise,
atmospheric stability, or wind shear. It does not distinguish black smoke,
white smoke, smoldering plume chemistry, or vertical plume height. It is a
raster-level visibility proxy for SAR policy learning.

## 6. UAV Perception

UAV survivor detection is probabilistic. A survivor or grid cell must lie inside
the circular camera footprint:

```text
footprint radius = altitude_AGL * tan(FOV / 2)
```

With the default 90 deg field of view, the footprint radius is approximately
equal to altitude. A 35 m AGL UAV therefore observes an approximate 35 m radius.

For a target location x, the instantaneous detection probability is the product
of four factors:

```text
p_detect =
    altitude_quality
  * footprint_range_quality
  * environment_quality
  * fire_smoke_quality
```

The implemented factor values are documented in detail in
[docs/perception_model.md](perception_model.md). In summary:

| Factor | Implemented model | Main calibration source |
|---|---|---|
| Altitude quality | shifted-Weibull curve, quality = 1 up to 30 m and 0.900 at 50 m | [Sambolek & Ivasic-Kos 2021](https://doi.org/10.1109/ACCESS.2021.3063681) |
| Footprint range | quadratic falloff to 0.70 at footprint edge | [Zheng et al. 2024](https://doi.org/10.1109/TPAMI.2024.3409416) |
| Environment | land-cover multiplier: road/open 1.00, brush 0.71, forest 0.56, rock 0.86, water 0.78 | UAV SAR and detection datasets listed in `perception_model.md` |
| Smoke | `(1 - smoke_load)^1.24` | [Liu et al. 2020](https://doi.org/10.3390/s20020349) |
| Fire glare | `1 - 0.35 * local_fire_load` | simulator-level camera saturation proxy |
| Heat distortion | `1 - 0.20 * mean_fire_along_path` | simulator-level heat shimmer proxy |

The default `rgb` mode represents an abstract electro-optical camera. The
optional `rgb_thermal` mode adds a conservative thermal benefit:

```text
environment_rgb_thermal = min(1, 1.30 * environment_rgb)
smoke_rgb_thermal       = 0.6 + 0.4 * smoke_rgb
```

The probability is used both for stochastic survivor detection and for the
inspection-confidence map, which accumulates the probability that a survivor at
each cell would already have been detected by the UAV team.

## 7. Survivor Scouting and Confirmation

The mission separates aerial scouting from ground confirmation:

1. UAVs scout survivors through the probabilistic perception model.
2. UGVs confirm survivors by reaching the physical confirmation radius.

This two-stage design reflects the operational workflow: aerial vehicles rapidly
reduce uncertainty over the search area, while ground robots verify survivor
locations and provide a closer contact point.

Key values:

| Quantity | Value |
|---|---:|
| Survivor count | 5 by default |
| Survivor radius | 0.35 m |
| Ground confirmation range | 10 m |
| Default survivor knowledge | hidden until scouted |
| Optional reveal schedule | stratified between steps 10 and 180 |

Confirmation is distance-based and deterministic once a UGV is close enough.
The simulator does not model survivor health state, medical triage, auditory
cues, or uncertainty in UGV close-range identification.

## 8. UGV Traversal and Planning

UGV mobility is physically scaled but still abstract. The controller produces
continuous motion, while the environment clips unsafe movement and exposes
terrain-aware route information in planner-enabled runs.

UGV traversability excludes:

- rock,
- water,
- tree/building cells,
- steep non-road terrain above 0.70 grade.

A* route costs can include:

| Term | Default value |
|---|---:|
| Land-cover traversal cost | class-specific, see Section 3 |
| Slope cost weight | 2.0 |
| Fire cost | 25.0 |
| Burned-cell cost | 2.0 |
| Smoke cost | 5.0 |
| Fire replan interval | 15 steps |

This is not a wheel-soil interaction model. There is no tire slip, rollover,
vehicle damage, battery thermal model, or detailed obstacle geometry. The
purpose is to make route choices reflect terrain and hazard structure at a
mission-planning level.

## 9. What OmniSearch Adds Beyond Standard 2D MARL Simulators

Standard 2D MARL simulators such as VMAS provide fast vectorized dynamics,
agents, sensors, collisions, and custom scenarios. OmniSearch keeps that
efficiency but adds domain structure needed for wildfire SAR:

- real-terrain raster layers from GIS data,
- physically scaled meters and seconds,
- UAV altitude and camera footprints,
- probabilistic perception tied to terrain, smoke, and fire,
- fire and smoke fields that evolve over time,
- heterogeneous UAV/UGV roles,
- terrain-aware ground traversal and A* route structure,
- mission-level scout-confirm dynamics.

The main improvement is not higher-fidelity low-level physics. It is the
addition of **mission-relevant structure**: policies must reason about search
uncertainty, terrain, hazards, aerial sensing, and ground confirmation in one
environment.

## 10. Main Assumptions

| Domain | Simulator abstraction | Reality gap |
|---|---|---|
| Physics | 2D holonomic VMAS motion plus UAV altitude scalar | no 6-DOF flight, wind-on-vehicle dynamics, wheel slip, or battery model |
| Terrain | six land-cover classes and gridded slope | real fuel and traversability vary continuously below grid scale |
| Fire | stochastic cellular automaton | no Rothermel/FARSITE implementation, crown fire, suppression, or plume feedback |
| Smoke | 2D scalar diffusion/advection field | no 3D buoyant plume or calibrated optical depth |
| Perception | factorized detection probability | no full image formation, target pose, detector thresholds, or correlated frame errors |
| Confirmation | deterministic distance threshold | no close-range sensor uncertainty or survivor condition model |
| Planning | A* over raster traversal costs | no full kinodynamic planning or recovery behavior |

OmniSearch is therefore best interpreted as a **conceptually grounded MARL test
environment** for wildfire SAR coordination, not as a deployment-ready
operational simulator.

## References

- [VMAS: Vectorized Multi-Agent Simulator](https://github.com/proroklab/vectorizedmultiagentsimulator)
- [USGS 3DEP products and services](https://www.usgs.gov/3d-elevation-program/about-3dep-products-services)
- [LANDFIRE fuel products](https://www.landfire.gov/fuel)
- [Rothermel surface fire spread model overview](https://research.fs.usda.gov/treesearch/55928)
- [Sambolek and Ivasic-Kos 2021, UAV person detection](https://doi.org/10.1109/ACCESS.2021.3063681)
- [Liu et al. 2020, fog and machine-vision detection](https://doi.org/10.3390/s20020349)
- [Zheng et al. 2024, spatial bias in object detection](https://doi.org/10.1109/TPAMI.2024.3409416)
- [Detailed OmniSearch perception documentation](perception_model.md)
