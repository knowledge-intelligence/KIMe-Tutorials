#!/bin/bash
# Gen @ 260708
# : Install Isaac Lab 2.3.2 + Isaac Sim 5.1.0 (binary workstation install) with uv
# Ref (Isaac Lab):  https://isaac-sim.github.io/IsaacLab/v2.3.2/source/setup/installation/binaries_installation.html
# Ref (Isaac Sim):  https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/install_workstation.html
# NOTE: Ubuntu 22.04/24.04, NVIDIA driver >= 580.65.06, RTX GPU (16GB+ VRAM) required.

set -e
set -x

# Prevents apt/debconf from ever blocking on a live prompt (e.g. tzdata's
# "Geographic area:" question) when a dependency pulls one in.
echo 'debconf debconf/frontend select Noninteractive' | sudo debconf-set-selections
export DEBIAN_FRONTEND=noninteractive

# Auto-accept the NVIDIA Omniverse EULA so the first Isaac Sim launch doesn't hang
# waiting for an interactive "Do you accept the EULA? (Yes/No)" prompt.
export OMNI_KIT_ACCEPT_EULA=YES

UV_ENV_NAME="env_isaaclab"
ISAACSIM_VERSION="5.1.0"
ISAACLAB_TAG="v2.3.2"
ISAACSIM_PATH="$HOME/isaacsim"
ISAACLAB_DIR="$HOME/IsaacLab"

ARCH=$(uname -m)
if [ "$ARCH" = "aarch64" ]; then
  ISAACSIM_ZIP="isaac-sim-standalone-${ISAACSIM_VERSION}-linux-aarch64.zip"
else
  ISAACSIM_ZIP="isaac-sim-standalone-${ISAACSIM_VERSION}-linux-x86_64.zip"
fi

echo "[Checking GLIBC version]"
# Isaac Sim requires GLIBC 2.35+ (Ubuntu 22.04/24.04 are fine; Ubuntu 20.04's
# GLIBC 2.31 is not).
ldd --version | head -n1

echo "[Installing system packages]"
sudo apt-get update -y
# cmake/build-essential are needed by robomimic, which Isaac Lab's --install pulls in.
sudo apt-get install -y ca-certificates curl git unzip cmake build-essential

echo "[Downloading Isaac Sim ${ISAACSIM_VERSION} binaries]"
if [ ! -x "$ISAACSIM_PATH/isaac-sim.sh" ]; then
  mkdir -p "$ISAACSIM_PATH"
  curl -fsSL "https://downloads.isaacsim.nvidia.com/${ISAACSIM_ZIP}" -o "/tmp/${ISAACSIM_ZIP}"
  unzip -q "/tmp/${ISAACSIM_ZIP}" -d "$ISAACSIM_PATH"
  rm -f "/tmp/${ISAACSIM_ZIP}"
  (cd "$ISAACSIM_PATH" && ./post_install.sh)
else
  echo "Isaac Sim already installed at $ISAACSIM_PATH, skipping download"
fi

export ISAACSIM_PATH="$ISAACSIM_PATH"
export ISAACSIM_PYTHON_EXE="${ISAACSIM_PATH}/python.sh"

echo "[Cloning Isaac Lab ${ISAACLAB_TAG}]"
if [ ! -d "$ISAACLAB_DIR" ]; then
  git clone --branch "$ISAACLAB_TAG" https://github.com/isaac-sim/IsaacLab.git "$ISAACLAB_DIR"
else
  echo "IsaacLab already cloned at $ISAACLAB_DIR, skipping clone"
fi

cd "$ISAACLAB_DIR"

echo "[Linking Isaac Sim binaries into Isaac Lab]"
if [ ! -e _isaac_sim ]; then
  ln -s "$ISAACSIM_PATH" _isaac_sim
fi

echo "[Installing uv]"
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
else
  echo "uv already installed, skipping"
fi

echo "[Creating uv venv: $UV_ENV_NAME via isaaclab.sh (experimental on Linux)]"
if [ ! -d "$ISAACLAB_DIR/$UV_ENV_NAME" ]; then
  ./isaaclab.sh --uv "$UV_ENV_NAME"
else
  echo "uv venv $UV_ENV_NAME already exists, skipping creation"
fi

source "$ISAACLAB_DIR/$UV_ENV_NAME/bin/activate"

echo "[Installing Isaac Lab extensions]"
./isaaclab.sh --install

echo "[Verifying installation]"
if [ -n "$DISPLAY" ]; then
  ./isaaclab.sh -p scripts/tutorials/00_sim/create_empty.py
else
  echo "No \$DISPLAY detected (headless session) - skipping the GUI verification launch."
  echo "Once you have a display, verify manually with:"
  echo "  source $ISAACLAB_DIR/$UV_ENV_NAME/bin/activate"
  echo "  cd $ISAACLAB_DIR && ./isaaclab.sh -p scripts/tutorials/00_sim/create_empty.py"
fi

echo "[Complete!!!]"
