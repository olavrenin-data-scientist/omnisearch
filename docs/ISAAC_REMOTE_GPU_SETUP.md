# Remote Isaac Setup (Linux GPU) for OmniSearch

Use this when your local machine cannot run Isaac Lab/Sim (for example macOS without NVIDIA GPU).

This guide gives a copy-paste path:
- bootstrap remote host,
- run first Isaac demo + record video,
- sync artifacts back into this repo.

---

## 1) Prepare a Remote Linux GPU Host

Target machine:
- Ubuntu 22.04 or 24.04
- NVIDIA GPU + up-to-date driver
- SSH access

On the remote host:

```bash
git clone <YOUR_REPO_URL> omnisearch
cd omnisearch
bash scripts/isaac_remote_bootstrap.sh --install
```

If you do not want package installs from script:

```bash
bash scripts/isaac_remote_bootstrap.sh --check
```

---

## 2) Verify OmniSearch Preflight on Remote

```bash
cd ~/omnisearch
python3 scripts/isaac_preflight.py --output-dir results/isaac_demo_remote
```

You want:
- `gpu_driver` -> OK
- Isaac modules importable after Isaac install
- `ffmpeg` -> OK

---

## 3) Install Isaac Lab

Install steps change by Isaac release; use the official docs:

- [Isaac Lab docs](https://isaac-sim.github.io/IsaacLab/)

Typical flow:

```bash
git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab
./isaaclab.sh --install
```

Then run a basic Isaac Lab example to confirm rendering and sensors.

---

## 4) First Video Demo (Remote)

Goal: produce one short camera/viewport recording.

Recommended workflow:
1. Run an Isaac scene with at least one camera.
2. Save frame sequence to `~/omnisearch/results/isaac_demo_remote/frames/`.
3. Encode to MP4:

```bash
cd ~/omnisearch
ffmpeg -framerate 20 -i results/isaac_demo_remote/frames/frame_%06d.png \
  -c:v libx264 -pix_fmt yuv420p results/isaac_demo_remote/first_demo.mp4
```

If Isaac recorder outputs MP4 directly, keep that file under `results/isaac_demo_remote/`.

---

## 5) Keep Long Jobs Alive

Use `tmux`:

```bash
tmux new -s isaac
# run long commands
# detach: Ctrl-b then d
tmux attach -t isaac
```

---

## 6) Sync Results Back to Local Repo

Run from your local machine (inside local repo root):

```bash
rsync -avz --progress \
  <user>@<remote-host>:~/omnisearch/results/isaac_demo_remote/ \
  results/isaac_demo_remote/
```

Optional: pull training artifacts too:

```bash
rsync -avz --progress \
  <user>@<remote-host>:~/omnisearch/results/harl_runs/ \
  results/harl_runs_remote/
```

---

## 7) Integrate with OmniSearch Metrics

After pulling results:

1. Keep Isaac logs/video under `results/isaac_demo_remote/`.
2. Convert episode logs to the same fields used by:
   - `evaluation/mission_metrics.py`
3. Compare VMAS vs Isaac using the same metric definitions.

This preserves scientific comparability while adding realism.

---

## 8) Suggested Next Implementation Step

Add an Isaac adapter skeleton in this repo:
- `agents/isaac_harl_env.py`
- `agents/isaac_harl_vec_env.py`

That lets HAPPO training reuse your current runner pattern with minimal code churn.
