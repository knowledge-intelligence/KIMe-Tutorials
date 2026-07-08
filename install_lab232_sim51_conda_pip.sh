#!/bin/bash
# Gen @ 260708
# : Install Isaac Lab 2.3.2 + Isaac Sim 5.1.0 (pip) in a Miniconda Python 3.11 env
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

CONDA_ENV_NAME="env_isaaclab"
PYTHON_VERSION="3.11"
ISAACSIM_VERSION="5.1.0"
ISAACLAB_TAG="v2.3.2"
MINICONDA_DIR="$HOME/miniconda3"
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

echo "[Installing Miniconda]"
if [ ! -d "$MINICONDA_DIR" ]; then
  if [ "$ARCH" = "aarch64" ]; then
    MINICONDA_INSTALLER="Miniconda3-latest-Linux-aarch64.sh"
  else
    MINICONDA_INSTALLER="Miniconda3-latest-Linux-x86_64.sh"
  fi
  curl -fsSL "https://repo.anaconda.com/miniconda/${MINICONDA_INSTALLER}" -o /tmp/miniconda.sh
  bash /tmp/miniconda.sh -b -p "$MINICONDA_DIR"
  rm -f /tmp/miniconda.sh
else
  echo "Miniconda already installed at $MINICONDA_DIR, skipping"
fi

source "$MINICONDA_DIR/etc/profile.d/conda.sh"
"$MINICONDA_DIR/bin/conda" init bash

echo "[Creating conda environment: $CONDA_ENV_NAME (python $PYTHON_VERSION)]"
if ! conda env list | grep -q "^${CONDA_ENV_NAME} "; then
  conda create -y -n "$CONDA_ENV_NAME" python="$PYTHON_VERSION"
else
  echo "Conda environment $CONDA_ENV_NAME already exists, skipping creation"
fi

conda activate "$CONDA_ENV_NAME"

echo "[Upgrading pip]"
pip install --upgrade pip

echo "[Installing Isaac Sim ${ISAACSIM_VERSION} via pip]"
pip install "isaacsim[all,extscache]==${ISAACSIM_VERSION}" --extra-index-url https://pypi.nvidia.com

echo "[Installing PyTorch matching Isaac Sim's CUDA build]"
if [ "$ARCH" = "aarch64" ]; then
  pip install -U torch==2.9.0 torchvision==0.24.0 --index-url https://download.pytorch.org/whl/cu130
else
  pip install -U torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
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
  echo "  conda activate $CONDA_ENV_NAME"
  echo "  cd $ISAACLAB_DIR && ./isaaclab.sh -p scripts/tutorials/00_sim/create_empty.py"
fi

echo "[Complete!!!]"
