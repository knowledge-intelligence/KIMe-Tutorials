#!/bin/bash
# Gen @ 260708
# : Run the standalone Isaac Sim 5.1.0 app via the official NGC Docker container,
#   with X11 GUI forwarding onto the host display (and a headless/livestream option). NO ROS2.
#
# Ref (Isaac Sim container):  https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/install_container.html
#
# Image: nvcr.io/nvidia/isaac-sim:5.1.0  (the Isaac Sim application by itself, without Isaac Lab)
#
# WHY DOCKER NAMED VOLUMES (and why we run as root, not the doc's uid 1234)?
#   The official doc mounts host directories (bind-mounts) and does 'sudo chown -R 1234:1234 ...'.
#   Under this machine's *rootless* Docker that ownership dance is fragile: a bind-mount dir whose
#   host owner does not match the container-user mapping is unwritable, which breaks Isaac Sim's
#   extension-registry / shader caches (PermissionError -> registry sync fails -> app aborts).
#   Instead we use Docker *named volumes* (created writable, seeded from the image) and run the
#   container as root with HOME=/isaac-sim -- exactly how the Isaac Lab image runs its bundled Sim.
#   Then every cache/data volume is writable with no chown, and the shared asset volume is
#   cleanly read/write for both (root) containers.
#
# NOTE: Ubuntu 22.04/24.04, NVIDIA driver + NVIDIA Container Toolkit, RTX GPU (16GB+ VRAM).
#       This is the Isaac Sim companion to  run_lab232_docker_prebuilt_x11.sh  (pre-built Isaac Lab).
#       Both scripts share the "isaac-omni-assets" volume + host network so they interoperate.

set -e

# ===========================================================================
# Configuration  (override from the environment, e.g. HOST_DISPLAY=:0 ./run_sim51_docker_x11.sh sim)
# ===========================================================================
SIM_IMAGE="${SIM_IMAGE:-nvcr.io/nvidia/isaac-sim:5.1.0}"   # NGC standalone Isaac Sim image
SIM_CONTAINER="${SIM_CONTAINER:-isaac-sim}"                # name of the running container
SIM_WORKDIR="/isaac-sim"                                   # Isaac Sim install path inside the image

# Host X display for GUI windows. Local Xorg session on this machine is :1 (an SSH $DISPLAY
# like localhost:10.0 is a TCP forward and cannot show RTX windows).
HOST_DISPLAY="${HOST_DISPLAY:-:1}"

log() { echo -e "[isaac-sim-docker] $*"; }

# ===========================================================================
# Usage / help text
# ===========================================================================
usage() {
  cat <<EOF
Usage: $(basename "$0") [command] [args]

Isaac Sim 5.1.0  (image: $SIM_IMAGE  = standalone Sim app, no Isaac Lab)

Commands:
  pull                 Pull the Isaac Sim image from NGC and exit.
  shell                (default) Open an interactive bash shell in the container
                       with GPU + X11 GUI enabled. Inside:
                         ./isaac-sim.sh       # GUI (a window opens on the host display)
                         ./runheadless.sh     # headless + WebRTC livestream (no local window)
  run <cmd...>         Run an arbitrary command inside the container and exit, e.g.
                         $(basename "$0") run ./runheadless.sh
  stop                 Force-remove the running isaac-sim container.
  clean-volumes        Remove this script's named volumes (frees cache; next run recompiles shaders).

Environment overrides: SIM_IMAGE, SIM_CONTAINER, HOST_DISPLAY (default :1)

Named volumes (see 'docker volume ls'):
  isaac-sim-cache, isaac-sim-computecache, isaac-sim-logs,
  isaac-sim-config, isaac-sim-data, isaac-sim-pkg
  isaac-omni-assets   <- SHARED with run_lab232_docker_prebuilt_x11.sh (Omniverse asset cache)

Headless / livestream: after './runheadless.sh -v' loads, connect from the WebRTC Streaming
Client to this host's IP. See the Isaac Sim container docs for the client download.
EOF
}

# ===========================================================================
# Pre-flight checks
# ===========================================================================
check_docker() {
  command -v docker >/dev/null 2>&1 || { log "ERROR: docker not installed. Run ./install_docker.sh first."; exit 1; }
  docker info      >/dev/null 2>&1 || { log "ERROR: cannot talk to the Docker daemon (is it running / do you have access?)."; exit 1; }
}

pull_image() {
  if docker image inspect "$SIM_IMAGE" >/dev/null 2>&1; then
    log "Image already present locally: $SIM_IMAGE (skipping pull)"
    return 0
  fi
  log "Pulling Isaac Sim image from NGC: $SIM_IMAGE (~15.1 GB, first time only)"
  if ! docker pull "$SIM_IMAGE"; then
    cat <<EOF

[isaac-sim-docker] ERROR: 'docker pull $SIM_IMAGE' failed.
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
# X11 forwarding: allow the container to open windows on the host X server.
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
# Assemble the 'docker run' argument list for Isaac Sim.
#   Runs as root (-u 0:0, HOME=/isaac-sim) so named volumes are writable with no chown.
#   Cache/data paths differ from Isaac Lab because Isaac Sim's HOME is /isaac-sim (not /root):
#     /isaac-sim/.cache, /isaac-sim/.nv/ComputeCache, /isaac-sim/.local/share/ov/{data,pkg}, ...
# ===========================================================================
build_run_args() {
  RUN_ARGS=(
    --rm
    --name "$SIM_CONTAINER"
    --network=host                         # share host network -> WebRTC livestream reachable on localhost/host IP
    # ---- GPU ----
    --gpus all
    -e NVIDIA_VISIBLE_DEVICES=all
    -e NVIDIA_DRIVER_CAPABILITIES=all
    # ---- Omniverse EULA / privacy (non-interactive) ----
    -e ACCEPT_EULA=Y
    -e PRIVACY_CONSENT=Y
    -e OMNI_KIT_ACCEPT_EULA=YES
    # ---- Run as root with Sim's HOME so named volumes are writable (see header) ----
    -u 0:0
    -e HOME=/isaac-sim
    # ---- X11 GUI forwarding onto the host display ----
    -e DISPLAY="$HOST_DISPLAY"
    -e XAUTHORITY=/isaac-sim/.Xauthority
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw    # X socket bind-mount (single path, no perm issue)
    # ---- Persistent cache/log/data as NAMED VOLUMES ----
    -v isaac-sim-cache:/isaac-sim/.cache                 # main Kit/OV cache
    -v isaac-omni-assets:/isaac-sim/.cache/ov            # SHARED Omniverse asset cache (nested under .cache)
    -v isaac-sim-computecache:/isaac-sim/.nv/ComputeCache
    -v isaac-sim-logs:/isaac-sim/.nvidia-omniverse/logs
    -v isaac-sim-config:/isaac-sim/.nvidia-omniverse/config
    -v isaac-sim-data:/isaac-sim/.local/share/ov/data    # extension registry cache lives here
    -v isaac-sim-pkg:/isaac-sim/.local/share/ov/pkg      # downloaded packages
  )
  [ -f "$HOME/.Xauthority" ] && RUN_ARGS+=(-v "$HOME/.Xauthority:/isaac-sim/.Xauthority:ro")

  # Pass the GPU DRI render node through for the RTX renderer (root:render on this host).
  if [ -e /dev/dri ]; then
    RUN_ARGS+=(--device /dev/dri)
    local rgid; rgid="$(getent group render | cut -d: -f3)"
    [ -n "$rgid" ] && RUN_ARGS+=(--group-add "$rgid")
  fi
}

# Allocate a TTY only when stdin is a real terminal.
tty_flag() { [ -t 0 ] && echo "-t"; }

# ===========================================================================
# Launchers
# ===========================================================================

# Interactive bash shell inside the Isaac Sim container.
run_interactive_shell() {
  setup_x11
  build_run_args
  log "Starting interactive shell in '$SIM_CONTAINER' (Isaac Sim at $SIM_WORKDIR)."
  log "GUI:                ./isaac-sim.sh"
  log "Headless/livestream: ./runheadless.sh -v   (then connect a WebRTC client to this host)"
  log "NOTE: the FIRST launch compiles RTX/MDL shaders (10-30+ min here, CPU-bound); cached afterwards."
  exec docker run "${RUN_ARGS[@]}" -it --entrypoint bash -w "$SIM_WORKDIR" "$SIM_IMAGE"
}

# Run one arbitrary command inside the container, then exit.
run_in_container() {
  setup_x11
  build_run_args
  log "Running inside '$SIM_CONTAINER': $*"
  exec docker run "${RUN_ARGS[@]}" -i $(tty_flag) --entrypoint bash -w "$SIM_WORKDIR" "$SIM_IMAGE" -lc "$*"
}

# ===========================================================================
# Maintenance
# ===========================================================================
stop_container() {
  docker rm -f "$SIM_CONTAINER" 2>/dev/null || true
  log "Removed container '$SIM_CONTAINER' (if it was running)."
}

# Remove this script's named volumes. 'isaac-omni-assets' is SHARED with the Isaac Lab script,
# so we keep it by default; remove it manually with 'docker volume rm isaac-omni-assets' if wanted.
clean_volumes() {
  docker volume rm -f \
    isaac-sim-cache isaac-sim-computecache isaac-sim-logs \
    isaac-sim-config isaac-sim-data isaac-sim-pkg 2>/dev/null || true
  log "Removed Isaac Sim named volumes (kept shared 'isaac-omni-assets'). Next launch recompiles shaders."
}

# ===========================================================================
# Main dispatch
# ===========================================================================
check_docker

case "${1:-shell}" in
  -h|--help|help) usage ;;
  pull)           pull_image ;;
  shell|sim)      pull_image; run_interactive_shell ;;
  run)            shift; pull_image; run_in_container "$@" ;;
  stop)           stop_container ;;
  clean-volumes)  clean_volumes ;;
  *)              pull_image; run_in_container "$@" ;;   # treat anything else as a raw command
esac
