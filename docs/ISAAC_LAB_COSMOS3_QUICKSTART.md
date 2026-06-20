# Isaac Lab + Cosmos 3 Quickstart (OmniSearch)

Practical setup to add a first high-fidelity Isaac demo with video output while keeping OmniSearch's existing mission-metrics workflow.

---

## Why This Path

- Use current VMAS flow for fast iteration.
- Use Isaac Lab for high-fidelity validation (robot dynamics + sensors).
- Use Cosmos 3 to diversify scenarios and improve robustness.

Keep your evaluation contract unchanged: `evaluation/mission_metrics.py` remains the final scorecard.

---

## 1) Preflight Check

Run from repo root:

```bash
source .venv/bin/activate
python scripts/isaac_preflight.py
```

This writes a report to `results/isaac_demo/preflight_report.json` with:
- Python status
- GPU driver visibility (`nvidia-smi`)
- `ffmpeg` availability
- Isaac-related Python modules import checks

If your local machine is not Isaac-capable, use the remote setup guide:
- `docs/ISAAC_REMOTE_GPU_SETUP.md`

---

## 2) First Video Demo Goal

Target deliverable for day 1:
- one Isaac scene with a drone + ground robot + survivors,
- one camera stream (RGB),
- one saved video clip (`.mp4` or encoded from saved frames),
- one JSON episode metadata blob compatible with future mission-metrics mapping.

---

## 3) Suggested Isaac Workflow

1. Create an Isaac task that mirrors OmniSearch entities:
   - 3 drones, 2 UGVs, survivor targets, hazard field.
2. Attach one viewport and one onboard camera sensor.
3. Run a short scripted policy (random or waypoint sweep).
4. Save frame sequence to `results/isaac_demo/frames/`.
5. Encode frames to video via `ffmpeg`.

Example frame encoding:

```bash
ffmpeg -framerate 20 -i results/isaac_demo/frames/frame_%06d.png -c:v libx264 -pix_fmt yuv420p results/isaac_demo/first_demo.mp4
```

If your Isaac build supports direct recording, you can save MP4 directly from the recorder extension and skip frame encoding.

---

## 4) Cosmos 3 Use in This Project

Use Cosmos 3 for scenario generation, not for replacing RL training:

- Generate diverse scenario specs (JSON):
  - terrain profile class
  - smoke opacity/wind profile
  - ignition seeds
  - survivor priors and occlusion level
- Feed those specs into Isaac reset parameters.
- Train/evaluate with domain randomization across generated specs.

Store generated specs under:

```text
data/cosmos_specs/
```

Recommended schema fields:
- `seed`
- `terrain_type`
- `wind_vector`
- `smoke_density`
- `ignition_points`
- `survivor_count`
- `survivor_distribution`
- `visibility_noise`

---

## 5) Bridge Back to OmniSearch Metrics

Do not change the metric contract. Instead map Isaac rollout logs to:

- `survivor_recall`
- `time_to_verification`
- `false_positive_trips`
- `hazard_exposure`
- `ugv_travel_cost`

Then use the existing DRR logic:
- `evaluation/mission_metrics.py::degradation_resilience_ratio`

This keeps VMAS-vs-Isaac comparisons honest.

---

## 6) Repo Integration Plan (Minimal)

Add these files incrementally:

- `agents/isaac_harl_env.py` - HARL-compatible adapter for Isaac single env
- `agents/isaac_harl_vec_env.py` - vectorized adapter
- `scripts/train_happo_isaac.py` - Isaac-targeted HAPPO launcher
- `scripts/export_isaac_episode.py` - episode export for metrics + viewer

Keep existing scripts untouched for baseline reproducibility.

---

## 7) First Milestone Checklist

- [ ] `python scripts/isaac_preflight.py` passes all major checks
- [ ] Isaac scene runs with at least one drone and one UGV
- [ ] Camera feed is visible in viewport
- [ ] Video file is saved to `results/isaac_demo/`
- [ ] Episode JSON is exported
- [ ] Mission-metrics mapping script reads that JSON and produces metrics

---

## 8) Practical Notes

- Isaac runs are slower than VMAS; keep VMAS for hyperparameter exploration.
- Validate policies in Isaac after narrowing candidate configs.
- Keep episode length long (`>=1000`) for parity with your latest tuning setup.
- Keep `drone_min_footprint=0.0` and `ground_confirm_min=0.0` if you want strict parity with your current request.
