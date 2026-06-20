#!/usr/bin/env bash
set -euo pipefail

# Bootstrap helper for Ubuntu + NVIDIA remote machines.
# This script installs common dependencies (optional) and validates that the
# host is ready for Isaac-style workflows.
#
# Usage:
#   bash scripts/isaac_remote_bootstrap.sh --check
#   bash scripts/isaac_remote_bootstrap.sh --install
#
# Notes:
# - --install uses apt and requires sudo privileges.
# - It does NOT install Isaac Lab itself (versions evolve quickly). It verifies
#   system prerequisites and prints next-step commands.

MODE="check"
if [[ "${1:-}" == "--install" ]]; then
  MODE="install"
elif [[ "${1:-}" == "--check" || "${1:-}" == "" ]]; then
  MODE="check"
else
  echo "Usage: $0 [--check|--install]"
  exit 1
fi

have() { command -v "$1" >/dev/null 2>&1; }

echo "===================================================================="
echo " OmniSearch remote bootstrap (${MODE})"
echo "===================================================================="

if [[ "$MODE" == "install" ]]; then
  echo "[1/4] Installing base packages (apt)..."
  sudo apt-get update
  sudo apt-get install -y \
    build-essential \
    curl \
    ffmpeg \
    git \
    git-lfs \
    jq \
    rsync \
    tmux \
    unzip \
    wget
fi

echo "[2/4] System checks"
if have nvidia-smi; then
  echo "  [OK ] nvidia-smi available"
  nvidia-smi --query-gpu=name,driver_version --format=csv,noheader || true
else
  echo "  [ERR] nvidia-smi missing (install NVIDIA GPU driver first)"
fi

if have docker; then
  echo "  [OK ] docker available"
else
  echo "  [WARN] docker missing (recommended for Isaac container workflows)"
fi

if have ffmpeg; then
  echo "  [OK ] ffmpeg available"
else
  echo "  [WARN] ffmpeg missing"
fi

echo "[3/4] Python environment checks"
if have python3; then
  python3 --version
else
  echo "  [ERR] python3 missing"
fi

if have conda; then
  echo "  [OK ] conda available"
else
  echo "  [WARN] conda missing (common for Isaac Lab native installs)"
fi

echo "[4/4] Next-step commands"
cat <<'EOF'

# Clone/update OmniSearch
git clone <YOUR_REPO_URL> omnisearch || true
cd omnisearch
git pull --ff-only || true

# Run OmniSearch preflight on this machine
python3 scripts/isaac_preflight.py --output-dir results/isaac_demo_remote

# Install Isaac Lab (follow official docs for your chosen version):
# https://isaac-sim.github.io/IsaacLab/
#
# Typical pattern:
#   git clone https://github.com/isaac-sim/IsaacLab.git
#   cd IsaacLab
#   ./isaaclab.sh --install
#
# Then run a minimal Isaac Lab example to verify rendering/camera.

# Keep long runs alive:
#   tmux new -s isaac
#   <run commands>
#   Ctrl-b then d   (detach)

EOF

echo "===================================================================="
echo " Done."
echo "===================================================================="
