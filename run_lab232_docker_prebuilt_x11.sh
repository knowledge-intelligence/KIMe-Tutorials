#!/bin/bash
# Gen @ 260708
# : Run Isaac Lab 2.3.2 via the official *pre-built* NGC Docker container,
#   with X11 GUI forwarding onto the host display. NO ROS2.
#
# Ref (Isaac Lab Docker):    https://isaac-sim.github.io/IsaacLab/v2.3.2/source/deployment/docker.html
# Ref (Run docker example):  https://isaac-sim.github.io/IsaacLab/v2.3.2/source/deployment/run_docker_example.html
#
# Image: nvcr.io/nvidia/isaac-lab:2.3.2  (Isaac Sim + Isaac Lab bundled -> self-contained;
#        you do NOT need a separate Isaac Sim install to run Isaac Lab.)
#
# WHY DOCKER NAMED VOLUMES (and not host bind-mounts)?
#   Under this machine's *rootless* Docker, the container's "root" (uid 0) is mapped to the
#   host user, NOT to real host root. If you bind-mount a host directory whose ownership does
#   not line up with that mapping (e.g. a dir left owned by real root:root from a previous run),
#   the container process cannot write into it. Isaac Lab then fails while creating its
#   extension-registry cache:
#       PermissionError: [Errno 13] Permission denied: '/root/.local/share/ov/data/exts'
#     -> "Syncing with extension registry unavailable"
#     -> dependency solver failure (e.g. isaacsim.asset.importer.urdf '=2.4.31' can't be satisfied)
#     -> ModuleNotFoundError: No module named 'omni.kit.usd'  -> app exits.
#   Docker *named volumes* are created writable for the container user (and seed themselves from
#   the image content), so this class of permission failure simply cannot happen. That is what
#   the official Isaac Lab docker-compose uses, and what we use below.
#
# NOTE: Ubuntu 22.04/24.04, NVIDIA driver + NVIDIA Container Toolkit, RTX GPU (16GB+ VRAM).
#       This is the Isaac Lab companion to  run_sim51_docker_x11.sh  (standalone Isaac Sim 5.1).
#       Both scripts share the "isaac-omni-assets" volume + host network so they interoperate.

set -e

# ===========================================================================
# Configuration  (every value can be overridden from the environment, e.g.
#   HOST_DISPLAY=:0 ./run_lab232_docker_prebuilt_x11.sh example logtime )
# ===========================================================================
LAB_IMAGE="${LAB_IMAGE:-nvcr.io/nvidia/isaac-lab:2.3.2}"   # NGC pre-built Isaac Lab image
LAB_CONTAINER="${LAB_CONTAINER:-isaac-lab}"                # name of the running container
LAB_WORKDIR="/workspace/isaaclab"                          # Isaac Lab repo path inside the image

# Host X display that GUI windows are drawn onto. On this machine the *local* Xorg session is :1.
# An SSH shell's $DISPLAY (e.g. localhost:10.0) is a TCP X11 forward and cannot show
# GPU-accelerated (RTX) windows, so we default to the local display :1.
HOST_DISPLAY="${HOST_DISPLAY:-:1}"

# Small helper to print consistently-prefixed status lines.
log() { echo -e "[isaac-lab-docker] $*"; }

# ===========================================================================
# Usage / help text
# ===========================================================================
usage() {
  cat <<EOF
Usage: $(basename "$0") [command] [args]

Isaac Lab 2.3.2  (image: $LAB_IMAGE  = Isaac Sim + Isaac Lab, self-contained)

Commands:
  pull                 Pull the pre-built Isaac Lab image from NGC and exit.
  shell                (default) Open an interactive bash shell in the container
                       with GPU + X11 GUI enabled. Inside, try:
                         ./isaaclab.sh -p scripts/tutorials/00_sim/create_empty.py
  example <key>        Run one built-in example and exit. Keys:
                         empty     -> tutorials/00_sim/create_empty.py   (headless smoke test)
                         logtime   -> tutorials/00_sim/log_time.py       (headless, official docker example)
                         spawn     -> tutorials/00_sim/spawn_prims.py    (GUI window)
                         quadruped -> demos/quadrupeds.py                (GUI window)
                         train     -> RL training, Ant, 50 iterations    (headless)
  run <cmd...>         Run an arbitrary command inside the container and exit, e.g.
                         $(basename "$0") run ./isaaclab.sh -p scripts/tutorials/00_sim/log_time.py --headless
  stop                 Force-remove the running isaac-lab container.
  clean-volumes        Remove this script's named volumes (frees cache; next run recompiles shaders).

Environment overrides: LAB_IMAGE, LAB_CONTAINER, HOST_DISPLAY (default :1)

Named volumes (see 'docker volume ls'):
  isaac-lab-kit-cache, isaac-lab-pip-cache, isaac-lab-gl-cache, isaac-lab-compute-cache,
  isaac-lab-logs, isaac-lab-data, isaac-lab-docs
  isaac-omni-assets   <- SHARED with run_sim51_docker_x11.sh (Omniverse asset cache)
EOF
}

# ===========================================================================
# Pre-flight checks
# ===========================================================================
check_docker() {
  command -v docker >/dev/null 2>&1 || { log "ERROR: docker not installed. Run ./install_docker.sh first."; exit 1; }
  docker info      >/dev/null 2>&1 || { log "ERROR: cannot talk to the Docker daemon (is it running / do you have access?)."; exit 1; }
}

# Pull the image if it is not already present. Guides the user through NGC login on failure.
pull_image() {
  if docker image inspect "$LAB_IMAGE" >/dev/null 2>&1; then
    log "Image already present locally: $LAB_IMAGE (skipping pull)"
    return 0
  fi
  log "Pulling pre-built Isaac Lab image from NGC: $LAB_IMAGE (~17.6 GB, first time only)"
  if ! docker pull "$LAB_IMAGE"; then
    cat <<EOF

[isaac-lab-docker] ERROR: 'docker pull $LAB_IMAGE' failed.
This image lives on NVIDIA NGC (nvcr.io) and usually requires authentication.
Log in once with your NGC API key, then re-run this script:

    docker login nvcr.io
      Username: \$oauthtoken
      Password: <your NGC API key from https://ngc.nvidia.com/setup/api-key>

EOF
    exit 1
  fi
}

# ===========================================================================
# X11 forwarding: allow the container (a local client) to open windows on the
# host X server at $HOST_DISPLAY. We combine an xhost grant with mounting the
# X socket and (if present) the Xauthority cookie in build_run_args().
# ===========================================================================
setup_x11() {
  if ! command -v xhost >/dev/null 2>&1; then
    log "WARN: xhost not found; GUI windows may fail. Install with: sudo apt-get install x11-xserver-utils"
    return 0
  fi
  log "Granting local X access on DISPLAY=$HOST_DISPLAY (xhost +local:)"
  DISPLAY="$HOST_DISPLAY" xhost +local:root >/dev/null 2>&1 || true
  DISPLAY="$HOST_DISPLAY" xhost +local:      >/dev/null 2>&1 || true
}

# ===========================================================================
# Assemble the 'docker run' argument list.
#   - GPU passthrough + NVIDIA capabilities (compute AND graphics for the RTX renderer)
#   - Omniverse EULA / privacy consent accepted non-interactively
#   - X11 GUI forwarding
#   - persistent NAMED VOLUMES for every cache/log/data dir (see header for why)
# The Isaac Lab image runs as root (uid 0) with HOME=/root, so root-owned named
# volumes are writable with no chown needed.
# ===========================================================================
build_run_args() {
  RUN_ARGS=(
    --rm                                   # auto-remove the container when it exits
    --name "$LAB_CONTAINER"
    --network=host                         # share host network -> reach a livestream/Nucleus on localhost
    # ---- GPU (verified on the RTX A6000) ----
    --gpus all
    -e NVIDIA_VISIBLE_DEVICES=all
    -e NVIDIA_DRIVER_CAPABILITIES=all      # 'all' includes graphics/display so the RTX renderer works
    # ---- Omniverse EULA / privacy (non-interactive) ----
    -e ACCEPT_EULA=Y
    -e PRIVACY_CONSENT=Y
    -e OMNI_KIT_ACCEPT_EULA=YES
    # ---- X11 GUI forwarding onto the host display ----
    -e DISPLAY="$HOST_DISPLAY"
    -e XAUTHORITY=/root/.Xauthority
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw    # X socket bind-mount (single path, no perm issue)
    # ---- Persistent cache/log/data as NAMED VOLUMES (writable by the container user) ----
    -v isaac-lab-kit-cache:/isaac-sim/kit/cache          # RTX/MDL shader + kit cache (slow to build once)
    -v isaac-omni-assets:/root/.cache/ov                 # SHARED Omniverse asset cache (also used by Sim script)
    -v isaac-lab-pip-cache:/root/.cache/pip              # pip wheels
    -v isaac-lab-gl-cache:/root/.cache/nvidia/GLCache    # GL shader cache
    -v isaac-lab-compute-cache:/root/.nv/ComputeCache    # CUDA compute cache
    -v isaac-lab-logs:/root/.nvidia-omniverse/logs       # Omniverse/Kit logs
    -v isaac-lab-data:/root/.local/share/ov/data         # extension registry cache lives here (was the PermissionError path)
    -v isaac-lab-docs:/root/Documents                    # user documents/output
  )
  # Mount the host X cookie read-only if it exists (belt-and-suspenders alongside the xhost grant).
  [ -f "$HOME/.Xauthority" ] && RUN_ARGS+=(-v "$HOME/.Xauthority:/root/.Xauthority:ro")

  # The RTX renderer wants the GPU DRI render node. On this host /dev/dri/renderD128 is
  # owned root:render, so pass the device through and add the host's 'render' GID.
  if [ -e /dev/dri ]; then
    RUN_ARGS+=(--device /dev/dri)
    local rgid; rgid="$(getent group render | cut -d: -f3)"
    [ -n "$rgid" ] && RUN_ARGS+=(--group-add "$rgid")
  fi
}

# Allocate a TTY (-t) only when stdin is a real terminal; a bare -t under a pipe aborts
# with "the input device is not a TTY".
tty_flag() { [ -t 0 ] && echo "-t"; }

# ===========================================================================
# Launchers
# ===========================================================================

# Interactive bash shell inside the Isaac Lab container.
run_interactive_shell() {
  setup_x11
  build_run_args
  log "Starting interactive shell in '$LAB_CONTAINER' (Isaac Lab repo at $LAB_WORKDIR)."
  log "Try:  ./isaaclab.sh -p scripts/tutorials/00_sim/create_empty.py"
  log "NOTE: the FIRST GUI/sim launch compiles RTX/MDL shaders (10-30+ min here, CPU-bound); cached afterwards."
  exec docker run "${RUN_ARGS[@]}" -it --entrypoint bash -w "$LAB_WORKDIR" "$LAB_IMAGE"
}

# Run one arbitrary command inside the container, then exit.
run_in_container() {
  setup_x11
  build_run_args
  log "Running inside '$LAB_CONTAINER': $*"
  exec docker run "${RUN_ARGS[@]}" -i $(tty_flag) --entrypoint bash -w "$LAB_WORKDIR" "$LAB_IMAGE" -lc "$*"
}

# Map a short example key to its Isaac Lab command (from the official run_docker_example page).
run_example() {
  local key="${1:-logtime}" cmd
  case "$key" in
    empty)     cmd="./isaaclab.sh -p scripts/tutorials/00_sim/create_empty.py --headless" ;;
    logtime)   cmd="./isaaclab.sh -p scripts/tutorials/00_sim/log_time.py --headless" ;;
    spawn)     cmd="./isaaclab.sh -p scripts/tutorials/00_sim/spawn_prims.py" ;;
    quadruped) cmd="./isaaclab.sh -p scripts/demos/quadrupeds.py" ;;
    train)     cmd="./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task=Isaac-Ant-v0 --headless --max_iterations 50" ;;
    *)         log "Unknown example key: '$key'"; usage; exit 1 ;;
  esac
  run_in_container "$cmd"
}

# ===========================================================================
# Maintenance
# ===========================================================================
stop_container() {
  docker rm -f "$LAB_CONTAINER" 2>/dev/null || true
  log "Removed container '$LAB_CONTAINER' (if it was running)."
}

# Remove this script's named volumes. NOTE: 'isaac-omni-assets' is SHARED with the Isaac Sim
# script; we leave it in place by default so the Sim container keeps its cache. Remove it
# manually with 'docker volume rm isaac-omni-assets' if you really want to.
clean_volumes() {
  docker volume rm -f \
    isaac-lab-kit-cache isaac-lab-pip-cache isaac-lab-gl-cache isaac-lab-compute-cache \
    isaac-lab-logs isaac-lab-data isaac-lab-docs 2>/dev/null || true
  log "Removed Isaac Lab named volumes (kept shared 'isaac-omni-assets'). Next launch recompiles shaders."
}

# ===========================================================================
# Main dispatch
# ===========================================================================
check_docker

case "${1:-shell}" in
  -h|--help|help) usage ;;
  pull)           pull_image ;;
  shell)          pull_image; run_interactive_shell ;;
  example)        shift; pull_image; run_example "$@" ;;
  run)            shift; pull_image; run_in_container "$@" ;;
  stop)           stop_container ;;
  clean-volumes)  clean_volumes ;;
  *)              pull_image; run_in_container "$@" ;;   # treat anything else as a raw command
esac
