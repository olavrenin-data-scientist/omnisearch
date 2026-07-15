# OmniSearch Simulation — Conceptual Overview

This document describes the physical world represented by the OmniSearch MARL training environment, the assumptions behind each model component, and where those assumptions diverge from physical reality. The goal is to let a reader judge how well a policy trained in this simulator might transfer to a real wildfire scenario.

---

## 1. World Representation

The simulation world is a **2D continuous plane** with a **discrete grid overlay**. All agent motion (position, velocity, forces) lives in continuous 2D space. Fire, smoke, terrain type, and elevation are encoded on a coarse discrete grid — by default 16×16 cells — that is draped over the same area.

**What this means physically:** Agents move smoothly through space, but fire and terrain features are resolved only to the grid cell level. At the default 16×16 resolution over a roughly 2 km² area (Malibu Creek State Park), each cell represents roughly 125×125 m. Spatial detail finer than one cell is invisible to the fire and smoke models.

**Key simplification:** The world is strictly 2D from a motion standpoint. Drones do not actually fly in 3D — their horizontal position is simulated in 2D and altitude is tracked as a separate scalar (see Section 5). There is no 3D collision geometry.

---

## 2. Agents

### 2.1 Drones (aerial searchers)

Three drones operate in the environment. Their physical parameters are:

| Parameter | Value | Physical interpretation |
|---|---|---|
| Max speed | 0.5 sim units/step | ~2.5× faster than ground robots |
| Action force | 0.6 multiplier | Continuous 2D thrust |
| Drag | 0.25 | Velocity damping per step |
| Collision radius | 0.04 sim units | ~5 m sphere |
| Flight altitudes | 30 m / 60 m / 90 m AGL | Three discrete operating levels |

Drones collide with each other but **not with survivors** — their detection is handled by a camera model (Section 4), not physical contact. They do not collide with ground robots.

**Simplification:** The drag-force-velocity dynamics are a simple linear damping model (VMAS default), not aerodynamic flight. There is no wind effect on drone motion, no battery or endurance limit, and no minimum turn radius.

### 2.2 Ground Robots (UGVs)

Two ground robots operate at the terrain surface. Their physical parameters are:

| Parameter | Value | Physical interpretation |
|---|---|---|
| Max speed | 0.2 sim units/step | ~2.5× slower than drones |
| Action force | 0.3 multiplier | Lower thrust than drones |
| Collision radius | 0.04 sim units | ~5 m sphere |
| Lidar range | 0.20 sim units | Short-range obstacle sensor |

Ground robots physically collide with survivor landmarks — their confirmation is range-based contact (within 0.10 sim units ≈ ~12 m).

**Simplification:** There is no vehicle dynamics model. Ground robots are point masses with drag. No wheel slip, no tip-over risk on slopes, no articulation constraints.

---

## 3. Terrain

### 3.1 Data Sources

The terrain is loaded from a pre-built cache derived from two real-world datasets:

- **USGS 3DEP** (10 m resolution digital elevation model): provides bare-earth elevation for every grid cell, from which slope is derived.
- **OpenStreetMap**: provides road networks, water bodies, and building footprints.

An optional third source, **LANDFIRE**, provides vegetation fuel density maps. Without it, fuel density is derived heuristically from land cover type.

The default area is **Malibu Creek State Park, California** — a real wildfire-prone chaparral landscape with a mix of open terrain, dense brush, forested ridgelines, and creek drainages.

### 3.2 Land Cover Classification

Each grid cell is assigned one of six land cover classes, which drive both fire behavior and robot traversal:

| Class | Fire fuel factor | Ground robot cost | Ground robot speed | Physical meaning |
|---|---|---|---|---|
| Road | 0.05 | 0.65 | 1.0× | Paved or dirt track |
| Open | 0.40 | 1.0 | 0.9× | Grassland, cleared area |
| Brush | 1.10 | 1.5 | 0.65× | Chaparral, shrubs |
| Forest | 1.35 | 2.2 | 0.45× | Wooded area |
| Rock | 0.0 | 4.0 | 0.0× | Impassable terrain |
| Water | 0.0 | 8.0 | 0.0× | Rivers, lakes |

Objects (trees, buildings) add additional fuel on top of the land cover base.

**Assumption:** Land cover is static throughout the episode — it does not change as vegetation burns. In reality, a burned cell transitions to open terrain with no fuel.

**Assumption:** The six-class schema is a significant reduction of real land cover complexity. Malibu chaparral includes chamise, manzanita, laurel sumac, and many other fuel types with very different burn characteristics; all are collapsed into "brush."

### 3.3 Slope

Slope is computed from the USGS elevation grid as the elevation difference between neighboring cells. It affects:

- **Ground robot traversability:** cells steeper than 70% grade (≈35°) are impassable, except on roads.
- **Ground robot speed:** speed is divided by (1 + 1.5 × slope), so a 30° slope roughly halves robot speed.
- **Ground robot cost:** cost is multiplied by (1 + 2.0 × slope).
- **Fire spread:** steeper uphill cells receive exponentially higher spread probability (Section 4.1).

**Assumption:** Slope is a single scalar per cell derived from the 16×16 grid, not the raw 10 m DEM. Subgrid slope variation — gullies, cliff edges — is not represented.

### 3.4 Moisture

A per-cell moisture field is derived from elevation and land cover (higher, wetter areas; water-adjacent cells are wetter). It modulates both fire spread probability and fire intensity. Moisture is static and does not respond to fire-induced drying or simulated rainfall.

**Physical reality gap:** Real wildfire moisture conditions are driven by relative humidity, time of day, recent precipitation, and fuel moisture content — none of which are modeled dynamically. The moisture field is a fixed proxy.

---

## 4. Fire Model

### 4.1 Spread Mechanism — Cellular Automaton

Fire spreads via a **stochastic cellular automaton** on the discrete grid. At each fire update step (every 3 environment steps), each unburned cell receives a spread attempt from each of its 8 neighbors.

The **ignition probability** for a target cell is:

```
p_ignite = 1 − (1 − base_prob)^rate
```

where the effective rate combines:

| Factor | Formula | Physical interpretation |
|---|---|---|
| Fire exposure | Σ (neighbor intensity × wind_factor × slope_factor) | Radiant heat and ember flux from adjacent burning cells |
| Fuel | land_cover_fuel × (0.65 + 0.55 × fuel_density) | Available combustible material |
| Moisture | exp(−1.15 × moisture) | Suppression from fuel wetness |
| Wind alignment | exp(1.25 × wind_strength × cos(θ)) | Wind blows fire downwind |
| Uphill slope | exp(1.65 × elevation_rise) | Fire climbs faster uphill |
| Stochastic variability | lognormal factor, clipped to [0.35, 2.25] | Run-to-run variability |
| Target boost | 0.35 + 1.05 × (remaining fraction / target) | Keeps fire near target area |

The base ignition probability is 0.065 per neighbor per update.

**Wind:** Wind is a constant vector (default: eastward, strength 0.06). Wind alignment with each of the 8 spread directions is computed as a dot product; directions aligned with wind receive an exponential boost, opposing directions receive a penalty.

**Spotting:** A very low probability (8×10⁻⁵ per update per cell) allows fire to jump over unburned cells via ember transport, but only to cells already adjacent to smoke. This represents long-range spotting in a simplified way.

**Physical reality gap:**
- The cellular automaton is a coarse approximation of fire front propagation. Real models (Rothermel, FARSITE) use reaction intensity, flame residence time, wind correction factors for slope, and fuel moisture of extinction as continuous physics-based equations.
- Fire does not propagate through partially burned or low-fuel cells differently — ignition is binary.
- There is no crown fire vs. surface fire distinction.
- Spotting distance is limited to smoke-adjacent cells; in reality embers can travel kilometers.
- Wind is spatially uniform and constant; topographic channeling and fire-induced convective columns are not modeled.

### 4.2 Burn Lifetime and Burnout

Each ignited cell is assigned a random burn lifetime between 5 and 14 fire update steps. Once the cell has burned for its lifetime, it is extinguished. The burned state is permanent — cells that have burned cannot reignite. This approximates fuel exhaustion.

**Physical reality gap:** Real burnout time depends on fuel load, density, and moisture. The random uniform distribution is a placeholder.

### 4.3 Fire Intensity

Each burning cell carries an intensity value in [0, 1] that evolves each update step. Intensity is driven by:

- **Fuel and moisture potential:** `0.15 + 0.85 × (fuel/moisture factor × slope factor)`
- **Burn lifecycle:** intensity decays linearly to 25% of its peak as the cell approaches burnout
- **Temporal smoothing:** 58% of prior intensity + 42% of target potential — a simple exponential moving average toward the physics-based potential

Intensity affects how strongly a burning cell contributes to neighboring spread exposure, how much smoke it emits, and how severely it degrades drone camera visibility.

**Physical reality gap:** Real fire intensity (heat release rate, flame height) is a function of fireline intensity with complex wind and slope dependencies. The simplified scalar is conceptually aligned but not quantitatively calibrated.

### 4.4 Target Area Regulation

Each episode samples a target burned fraction (default: 20%–40% of the grid). A feedback term boosts spread probability when the fire is below target and caps new ignitions when the target is reached. This keeps training scenarios varied without the fire either dying out or consuming everything.

**This mechanism has no physical counterpart** — it is an engineering choice to maintain scenario diversity during training. In a real wildfire the area burned is determined entirely by physics.

---

## 5. Smoke Model

Smoke is a scalar field on the same 16×16 grid, updated every environment step. The update has three parts:

1. **Emission:** Each burning cell emits smoke proportional to its fire intensity and local fuel density: `smoke += intensity × 0.18 × fuel`.
2. **Decay:** All smoke decays by a factor of 0.96 each step (approximately exponential decay with a time constant of ~25 steps).
3. **Diffusion:** A 4-neighbor average smooths the field: `smoke += 0.16 × (mean_of_4_neighbors − smoke)`.
4. **Wind advection:** The smoke field is shifted in the wind direction, blended at a weight equal to wind strength (0.06 by default).

Smoke affects drone camera detection probability (Section 6.2). It also enables fire spotting — cells with any smoke load (> 0.08) are candidates for spotting ignition.

**Physical reality gap:**
- There is no plume dynamics model. Real smoke plume behavior is governed by buoyancy, atmospheric stability, and wind shear — none of which are represented.
- Smoke density is dimensionless and not calibrated to any physical quantity (e.g., PM2.5 concentration or optical depth).
- There is no separation between smoke from active flaming and residual smoldering.

---

## 6. Drone Perception Model

### 6.1 Camera Footprint

Each drone's visible ground area is determined by its altitude and a fixed camera field-of-view angle (default: 65°). The ground footprint radius is:

```
footprint_radius = altitude_AGL × tan(FOV / 2)
```

At 30 m AGL, this gives a radius of about 32 m; at 90 m, about 96 m. A survivor is only a detection candidate if they fall within this footprint.

**Assumption:** The camera footprint is a circle, not the rectangular footprint of a real camera sensor. There is no lens distortion, no image resolution degradation toward the edges (beyond the distance factor below).

### 6.2 Detection Probability

The abstract UAV perception mode defaults to `rgb`. A second mode,
`rgb_thermal`, is available for experiments with an RGB+thermal sensor stack.
In the current calibration step, `rgb_thermal` intentionally uses the same
equations and values as `rgb`; thermal-specific contrast, smoke penetration,
and heat-crossover terms are left for a later calibration.

If a survivor is within the footprint, detection is **stochastic**: a random draw against a probability that is the product of four independent factors:

| Factor | Formula | Physical meaning |
|---|---|---|
| Distance factor | `1 − 0.30 × (dist/footprint)²` | Detection is highest at nadir and falls quadratically to a 70% edge floor |
| Environment factor | Per-class value | Terrain, clutter, concealment, and water-background effects |
| Smoke/fire factor | Product of smoke, glare, and heat terms | Atmospheric degradation of the camera image |
| Altitude quality | Interpolated from flight level | Proxy for image resolution and integration time |

**Smoke attenuation:** `(1 − smoke_load)^1.24` — smoke intensity is interpreted as contrast loss from an atmospheric-transmission conversion. The exponent is fitted to the clear-normalized Faster R-CNN recalls reported by Liu et al. (2020), *Analysis of the Influence of Foggy Weather Environment on the Detection Effect of Machine Vision Obstacles*. Detection quality approaches zero as smoke becomes opaque.

**Footprint-edge quality:** the 70% floor is motivated by *Zone Evaluation: Revealing Spatial Bias in Object Detection* (TPAMI, 2024), which reports that outer image regions often retain roughly 70–80% of center performance depending on detector and dataset. The paper reports region-wise AP rather than a radial probability function, so the quadratic radial form remains a simulator assumption.

**Environment quality:** terrain factors are ordered as `road, open, brush, forest, rock, water` and currently use `(1.00, 1.00, 0.71, 0.56, 0.86, 0.78)`. Road and open terrain are treated as unobstructed. Brush and forest follow the medium/high vegetation classes from SAVIOUR 2024 as an empirical terrain-quality proxy. Rock is treated as low-to-medium clutter. Water uses the SeaDronesSee swimmer AP50 reference as a compact proxy for glare, wave clutter, partial submersion, and maritime background ambiguity.

**Fire glare:** `1 − 0.35 × max(local_fire_intensity, fire_density_near_survivor)` — fire near the survivor degrades the image as though saturating the camera sensor.

**Heat shimmer:** `1 − 0.20 × mean_fire_intensity_along_path` — heat-induced refractive distortion modeled as a linear penalty.

These three factors are sampled along 8 interpolated points on the straight-line path from drone to survivor, representing the integrated optical path through the atmosphere.

**Altitude quality:** Lower altitudes give better detection quality (fewer pixels per meter, better feature resolution) but a smaller footprint. The quality interpolates linearly between 0.95 at the lowest flight level (30 m) and 0.55 at the highest (90 m).

**Physical reality gap:**
- The multiplicative independence of factors is a significant simplification. In reality smoke, glare, and viewing angle interact nonlinearly.
- The exponential smoke attenuation assumes uniform smoke density along the vertical column, which ignores the actual plume structure.
- There is no false positive detection. The model only tracks whether a known survivor is detected, not whether an agent misidentifies a non-survivor.
- The model does not represent actual image processing. It is an abstract probability that stands in for what a real YOLOv8 pipeline (see `detection/`) would compute.

### 6.3 Drone Flight Altitude

Drones operate in "2.5D" — horizontal motion is 2D continuous, but each drone maintains a continuously tracked AGL altitude. The altitude controller works as follows:

- The required minimum clearance over each cell is: `obstacle_height + 15 m safety margin`.
- The drone's target altitude is the maximum required clearance along its path, clamped to the operating range [30 m, 90 m AGL].
- Actual altitude moves toward the target at a limited rate: 10 m/step climb, 8 m/step descent.
- A hysteresis margin of 10 m prevents rapid oscillation when crossing ridge lines.

The drone's MSL altitude is: `terrain_elevation + AGL_altitude`.

**Simplification:** Drones do not choose their altitude as part of their MARL action — altitude is determined automatically by the terrain. This removes one degree of freedom from the learned policy. A real drone operator would explicitly trade off altitude (footprint size vs. image resolution) as a decision.

---

## 7. Ground Robot Traversal

### 7.1 Traversability

Ground robots cannot cross cells that are:
- Water or bare rock (impassable by classification)
- Occupied by a building or tree object
- Steeper than 70% grade (except on roads)

If a robot's commanded action would move it across an impassable cell boundary, the move is rejected and the closest safe partial move is applied instead, using a 10-candidate sliding/shortening fallback.

### 7.2 Speed and Cost

A robot's effective speed is scaled at each step by the terrain speed multiplier at its current cell. The multiplier ranges from 1.0 on roads to 0 on water and rock (impassable). On forest terrain a robot moves at 45% of its maximum speed.

Travel cost (used in the reward) is the Euclidean distance moved multiplied by the average mobility cost over the path — effectively terrain-weighted path length.

**Simplification:** Speed and cost are scalars per cell — there is no directional dependence (e.g., traversing a slope diagonally vs. directly uphill). Real vehicle traction and energy consumption are direction-dependent.

**Simplification:** Ground robots are not affected by fire in terms of traversal (they can enter burning cells; the penalty is purely in the reward signal, not in physical movement capability). In reality a ground robot in active fire would be destroyed or would trigger an emergency stop.

---

## 8. Communication

Agents observe the relative positions of all other agents via shared state. Communication dropout is implemented as a Bernoulli mask applied to each agent's observation of each neighbor's position:

```
observed_delta = actual_delta × Bernoulli(1 − dropout_rate)
```

When a message is dropped, the agent sees zero for that neighbor's relative position. The dropout is applied independently each step, with no temporal correlation (lost packets are not followed by more lost packets at a higher rate).

**Simplification:** Real radio communication in a wildfire environment experiences correlated losses due to terrain shadowing, smoke absorption, and relay availability. The i.i.d. Bernoulli model is the simplest possible dropout model. There is no bandwidth limit, no latency, and no message corruption beyond total loss.

---

## 9. Episode Structure and Reward

### 9.1 Episode Configuration

Each episode samples:
- A target burned area fraction (20%–40% of the grid)
- An initial fire ignition patch (≈2.5% of the grid, seeded in high-fuel terrain)
- Random initial positions for all agents and survivors in traversable terrain

The episode ends when either all 5 survivors are confirmed or 200 steps elapse (a step is one environment tick, roughly interpreted as 1–2 seconds of real time depending on scale).

### 9.2 Scout–Confirm Protocol

Survivor discovery requires **two distinct events**:
1. **Scout:** A drone detects a survivor via its camera model (stochastic, distance- and environment-dependent).
2. **Confirm:** A ground robot comes within 0.10 sim units (~12 m) of the survivor.

This two-step design reflects the operational concept: drones rapidly survey the area and mark candidate locations; ground robots navigate to each candidate to verify and (in the real system) provide assistance.

**Simplification:** In reality, confirmation would involve physical assessment of the survivor's condition by the ground robot. Here it is a pure distance threshold with no noise.

### 9.3 Reward Structure

All agents share a team reward, plus individual credit terms:

| Component | Agent | Value |
|---|---|---|
| New survivor confirmed | All (team) | +1.0 |
| Time step penalty | All (team) | −0.001 |
| Drone scouts a new survivor | Drone (individual) | +0.3 |
| Ground robot confirms a new survivor | Ground robot (individual) | +0.5 |
| Ground robot in burning cell | Ground robot (individual) | −1.0 per step |
| Ground robot terrain travel cost | Ground robot (individual) | −0.05 × weighted distance |
| Drone altitude change | Drone (individual) | −0.02 × meters climbed |

**Design note:** The individual credit terms are incentives to produce the desired role specialization (drones scout, ground robots confirm). Without them, agents could receive team reward without contributing to the scouting/confirming sub-tasks.

---

## 10. Summary of Key Simplifications

| Domain | What the simulation does | What reality looks like |
|---|---|---|
| **Spatial resolution** | 16×16 grid over ~2 km² (125 m/cell) | Meter-level or finer variation in fuel, slope, and fire |
| **Fire physics** | Stochastic CA with empirical factors | Rothermel/FARSITE reaction intensity, fuel moisture of extinction, flame geometry |
| **Fire area** | Regulated to a sampled target fraction | Determined entirely by fuel, weather, and suppression |
| **Smoke** | Scalar diffusion–advection field | 3D buoyant plume, Gaussian dispersion, atmospheric stability |
| **Drone motion** | 2D + automatic altitude, linear drag | Full 6-DOF flight dynamics, battery, wind effects on trajectory |
| **Drone detection** | Probability product of independent factors | Real image pipeline (YOLOv8) with occlusion, contrast, and resolution effects |
| **Altitude control** | Automatic terrain-following | Pilot decision or flight plan |
| **Ground robot dynamics** | Point mass with drag and speed scaling | Wheel-terrain interaction, tip-over, power consumption |
| **Ground robots in fire** | Move normally, incur reward penalty | Would be destroyed; emergency stop |
| **Terrain moisture** | Static per-cell proxy | Dynamic function of humidity, time, rain, fuel moisture content |
| **Communication** | i.i.d. Bernoulli dropout per step | Terrain-shadowing, relay topology, latency, bandwidth limits |
| **Survivor detection** | Distance threshold at ground robot | Physical contact, sensor reading, condition assessment |
| **False positives** | Not modeled | Fire debris, mannequins, animals trigger false alarms in real YOLO |

The simulation is well-suited for comparing coordination strategies and testing policy robustness to comms dropout. The fire physics, drone perception, and terrain traversal are conceptually grounded, but none are calibrated quantitatively to field data. A policy trained here should be expected to need significant fine-tuning before deployment on a real platform.
