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

# isaaclab.sh runs `tabs 4` at startup, which aborts the script when TERM is unset
# or "dumb" (as in headless Docker/CI shells): "'ansi+tabs': unknown terminal
# type." / "terminal type 'dumb' cannot reset tabs". Normalize those to xterm (its
# terminfo ships in ncurses-base) while leaving a real interactive terminal
# (xterm-256color, etc.) untouched.
case "${TERM:-}" in ""|dumb) export TERM=xterm ;; esac

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

# Isaac Sim's RTX renderer needs system OpenGL/Vulkan/X11 runtime libraries even in
# --headless mode. A minimal Ubuntu (e.g. a clean container) lacks them, so the app
# aborts with "libGL.so.1: cannot open shared object file" and Vulkan
# "ERROR_INCOMPATIBLE_DRIVER". A workstation with the NVIDIA driver usually already
# has these; installing them makes the run work on a headless/minimal host too.
sudo apt-get install -y \
  libgl1 libglu1-mesa libegl1 libgomp1 libatomic1 \
  libsm6 libice6 libxt6 libxi6 libxrandr2 libxrender1 libxext6 libx11-6 \
  libxfixes3 libxcursor1 libxinerama1 libfontconfig1 libfreetype6 libvulkan1

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

echo "[Creating uv venv: $UV_ENV_NAME (python 3.11) via isaaclab.sh (experimental on Linux)]"
# isaaclab.sh --uv creates the venv without pinning a Python version, so uv picks
# the newest interpreter it can find (e.g. 3.14) — for which Isaac Sim and torch
# have no wheels (they're cp311), making `--install` unsatisfiable. Pre-create the
# venv at Python 3.11 (with --seed so `python -m pip` probes inside isaaclab.sh
# still work); isaaclab.sh --uv then reuses it (it only skips the `uv venv` step
# when the env dir exists) and still installs its Isaac Sim activation hooks.
if [ ! -d "$ISAACLAB_DIR/$UV_ENV_NAME" ]; then
  uv venv --python 3.11 --seed "$ISAACLAB_DIR/$UV_ENV_NAME"
fi
./isaaclab.sh --uv "$UV_ENV_NAME"

source "$ISAACLAB_DIR/$UV_ENV_NAME/bin/activate"

echo "[Pinning build-time setuptools<81 for legacy sdists (e.g. flatdict==4.0.1)]"
# Isaac Lab pins flatdict==4.0.1, whose setup.py does `import pkg_resources`.
# setuptools >= 81 removed pkg_resources, so flatdict's PEP517 isolated build fails
# with "ModuleNotFoundError: No module named 'pkg_resources'", which breaks the
# isaaclab core install. pip applies PIP_CONSTRAINT to isolated build envs, so pin
# the build-time setuptools to a version that still ships pkg_resources.
echo "setuptools<81" > "$HOME/isaaclab-build-constraints.txt"
export PIP_CONSTRAINT="$HOME/isaaclab-build-constraints.txt"
# conda paths use `python -m pip` (PIP_CONSTRAINT); uv paths use `uv pip`, which
# reads UV_BUILD_CONSTRAINT for its isolated build environment instead.
export UV_BUILD_CONSTRAINT="$HOME/isaaclab-build-constraints.txt"

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
