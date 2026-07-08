#!/bin/bash
# Gen @ 260708
# : Install Isaac Lab 2.3.2 + Isaac Sim 5.1.0 (pip) in a uv Python 3.11 venv
# Ref: https://isaac-sim.github.io/IsaacLab/v2.3.2/source/setup/installation/pip_installation.html

set -e
set -x

# Prevents apt/debconf from ever blocking on a live prompt (e.g. tzdata's
# "Geographic area:" question) when a dependency pulls one in.
echo 'debconf debconf/frontend select Noninteractive' | sudo debconf-set-selections
export DEBIAN_FRONTEND=noninteractive

# Auto-accept the NVIDIA Omniverse EULA so the first Isaac Sim launch doesn't hang
# waiting for an interactive "Do you accept the EULA? (Yes/No)" prompt.
export OMNI_KIT_ACCEPT_EULA=YES

UV_VENV_DIR="$HOME/env_isaaclab"
PYTHON_VERSION="3.11"
ISAACSIM_VERSION="5.1.0"
ISAACLAB_TAG="v2.3.2"
ISAACLAB_DIR="$HOME/IsaacLab"

ARCH=$(uname -m)

echo "[Checking GLIBC version]"
# Isaac Sim's pip install requires GLIBC 2.35+ (Ubuntu 22.04/24.04 are fine;
# Ubuntu 20.04's GLIBC 2.31 is not).
ldd --version | head -n1

echo "[Installing system packages]"
sudo apt-get update -y
# cmake/build-essential are needed by robomimic, which Isaac Lab's --install pulls in.
sudo apt-get install -y ca-certificates curl git cmake build-essential

echo "[Installing uv]"
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
else
  echo "uv already installed, skipping"
fi

echo "[Creating uv venv: $UV_VENV_DIR (python $PYTHON_VERSION)]"
if [ ! -d "$UV_VENV_DIR" ]; then
  # --seed installs pip/setuptools/wheel into the venv; uv venvs don't include
  # pip by default, and Isaac Lab's install script needs it.
  uv venv --python "$PYTHON_VERSION" --seed "$UV_VENV_DIR"
else
  echo "uv venv $UV_VENV_DIR already exists, skipping creation"
fi

source "$UV_VENV_DIR/bin/activate"

echo "[Installing Isaac Sim ${ISAACSIM_VERSION} via uv pip]"
uv pip install "isaacsim[all,extscache]==${ISAACSIM_VERSION}" --extra-index-url https://pypi.nvidia.com

echo "[Installing PyTorch matching Isaac Sim's CUDA build]"
if [ "$ARCH" = "aarch64" ]; then
  uv pip install -U torch==2.9.0 torchvision==0.24.0 --index-url https://download.pytorch.org/whl/cu130
else
  uv pip install -U torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
fi

echo "[Cloning Isaac Lab ${ISAACLAB_TAG}]"
if [ ! -d "$ISAACLAB_DIR" ]; then
  git clone --branch "$ISAACLAB_TAG" https://github.com/isaac-sim/IsaacLab.git "$ISAACLAB_DIR"
else
  echo "IsaacLab already cloned at $ISAACLAB_DIR, skipping clone"
fi

cd "$ISAACLAB_DIR"

echo "[Installing Isaac Lab extensions]"
./isaaclab.sh --install

echo "[Verifying installation]"
if [ -n "$DISPLAY" ]; then
  ./isaaclab.sh -p scripts/tutorials/00_sim/create_empty.py
else
  echo "No \$DISPLAY detected (headless session) - skipping the GUI verification launch."
  echo "Once you have a display, verify manually with:"
  echo "  source $UV_VENV_DIR/bin/activate"
  echo "  cd $ISAACLAB_DIR && ./isaaclab.sh -p scripts/tutorials/00_sim/create_empty.py"
fi

echo "[Complete!!!]"
