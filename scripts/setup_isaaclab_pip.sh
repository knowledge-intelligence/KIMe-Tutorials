#!/usr/bin/env bash
#
# Isaac Lab 2.3.2 + Isaac Sim 5.1 (pip) 자동 설치 스크립트
#
# 저장소 clone → conda 환경 생성 → Isaac Sim/Isaac Lab 설치까지 수행합니다.
# 이름을 하나만 받아 폴더는 <이름>, conda 환경은 env_<이름>으로 만듭니다.
#
# 사용법:
#   ./setup_isaaclab_pip.sh [옵션] <이름>
#
# 예시:
#   ./setup_isaaclab_pip.sh islab            # ./islab 폴더 + env_islab 환경
#   ./setup_isaaclab_pip.sh --uninstall islab  # 위에서 만든 폴더와 환경을 삭제
#
# 옵션:
#   --uninstall     설치하지 않고 <이름> 폴더와 env_<이름> 환경을 삭제
#   --verify-only   설치를 건너뛰고 import/CUDA 검증만 수행
#   --skip-verify   설치만 하고 검증은 건너뜀
#   --force         폴더/환경이 이미 있으면 재사용 (기본은 중단)
#   -y, --yes       삭제 시 확인 프롬프트를 건너뜀
#   -h, --help      도움말 출력
#
# 참고: https://isaac-sim.github.io/IsaacLab/v2.3.2/source/setup/installation/pip_installation.html

set -euo pipefail

ISAACLAB_REPO="https://github.com/isaac-sim/IsaacLab.git"
ISAACLAB_TAG="v2.3.2"
PYTHON_VERSION="3.11"
ISAACSIM_VERSION="5.1.0"
TORCH_VERSION="2.7.0"
TORCHVISION_VERSION="0.22.0"
TORCHAUDIO_VERSION="2.7.0"
TORCH_INDEX="https://download.pytorch.org/whl/cu128"
NVIDIA_INDEX="https://pypi.nvidia.com"
ENV_PREFIX="env_"

DO_INSTALL=1
DO_VERIFY=1
DO_UNINSTALL=0
FORCE=0
ASSUME_YES=0
POSITIONAL=()

usage() { sed -n '2,24p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; }

while [ $# -gt 0 ]; do
    case "$1" in
        --uninstall)   DO_UNINSTALL=1 ;;
        --verify-only) DO_INSTALL=0 ;;
        --skip-verify) DO_VERIFY=0 ;;
        --force)       FORCE=1 ;;
        -y|--yes)      ASSUME_YES=1 ;;
        -h|--help)     usage; exit 0 ;;
        -*)            echo "알 수 없는 옵션: $1" >&2; usage; exit 1 ;;
        *)             POSITIONAL+=("$1") ;;
    esac
    shift
done

if [ "${#POSITIONAL[@]}" -ne 1 ]; then
    echo "오류: 이름을 하나만 지정해야 합니다." >&2
    echo >&2
    usage >&2
    exit 1
fi

NAME_ARG="${POSITIONAL[0]}"

# 폴더는 <이름>, conda 환경은 env_<이름>. 경로를 줘도 되도록 환경 이름은 basename으로 만든다.
case "$NAME_ARG" in
    /*) LAB_DIR="$NAME_ARG" ;;
    *)  LAB_DIR="$(pwd)/${NAME_ARG}" ;;
esac
CONDA_ENV="${ENV_PREFIX}$(basename "$NAME_ARG")"

log()  { echo -e "\n\033[1;34m[$(date +%H:%M:%S)] $*\033[0m"; }
ok()   { echo -e "\033[1;32m  ✓ $*\033[0m"; }
warn() { echo -e "\033[1;33m  ! $*\033[0m"; }
die()  { echo -e "\033[1;31m  ✗ $*\033[0m" >&2; exit 1; }

command -v conda >/dev/null || die "conda를 찾을 수 없습니다. Miniconda/Anaconda를 먼저 설치하세요."

# conda activate/env remove를 비대화형 셸에서 쓰기 위한 훅
eval "$(conda shell.bash hook)"

env_exists() { conda env list | awk '{print $1}' | grep -qx "$1"; }

# ---------------------------------------------------------------------------
# 삭제 모드
# ---------------------------------------------------------------------------
if [ "$DO_UNINSTALL" -eq 1 ]; then
    log "삭제 대상 확인"

    FOUND=0

    if [ -d "$LAB_DIR" ]; then
        # 엉뚱한 폴더를 지우지 않도록 Isaac Lab 저장소가 맞는지 확인한다.
        if [ ! -f "${LAB_DIR}/isaaclab.sh" ]; then
            die "${LAB_DIR}는 Isaac Lab 저장소가 아닙니다 (isaaclab.sh 없음). 안전을 위해 중단합니다."
        fi
        echo "  폴더:      ${LAB_DIR}  ($(du -sh "$LAB_DIR" 2>/dev/null | cut -f1))"
        # 커밋되지 않은 변경이 있으면 알린다.
        if git -C "$LAB_DIR" rev-parse --git-dir >/dev/null 2>&1; then
            CHANGES="$(git -C "$LAB_DIR" status --porcelain 2>/dev/null | wc -l)"
            [ "$CHANGES" -gt 0 ] && warn "커밋되지 않은 변경 ${CHANGES}건이 있습니다. 삭제하면 사라집니다."
        fi
        FOUND=1
    else
        echo "  폴더:      ${LAB_DIR}  (없음)"
    fi

    if env_exists "$CONDA_ENV"; then
        ENV_PATH="$(conda env list | awk -v e="$CONDA_ENV" '$1==e {print $NF}')"
        echo "  conda 환경: ${CONDA_ENV}  ($(du -sh "$ENV_PATH" 2>/dev/null | cut -f1))"
        FOUND=1
    else
        echo "  conda 환경: ${CONDA_ENV}  (없음)"
    fi

    [ "$FOUND" -eq 1 ] || die "삭제할 대상이 없습니다."

    # 현재 활성화된 환경은 삭제할 수 없다.
    if [ "${CONDA_DEFAULT_ENV:-}" = "$CONDA_ENV" ]; then
        die "'${CONDA_ENV}'가 현재 활성화되어 있습니다. 'conda deactivate' 후 다시 실행하세요."
    fi

    if [ "$ASSUME_YES" -eq 0 ]; then
        echo
        read -r -p "  위 항목을 삭제합니다. 계속하시겠습니까? [y/N] " REPLY
        case "$REPLY" in
            [yY]|[yY][eE][sS]) ;;
            *) echo "  취소했습니다."; exit 0 ;;
        esac
    fi

    log "삭제"
    if [ -d "$LAB_DIR" ]; then
        rm -rf "$LAB_DIR"
        ok "폴더 삭제: ${LAB_DIR}"
    fi
    if env_exists "$CONDA_ENV"; then
        conda env remove -n "$CONDA_ENV" -y >/dev/null 2>&1
        ok "conda 환경 삭제: ${CONDA_ENV}"
    fi

    log "삭제 완료"
    echo -e "\n  pip 캐시는 남아 있습니다. 함께 지우려면:  rm -rf ~/.cache/pip\n"
    exit 0
fi

# ---------------------------------------------------------------------------
# 0. 사전 점검
# ---------------------------------------------------------------------------
log "사전 점검"
ok "conda $(conda --version | awk '{print $2}')"
echo "  폴더: ${LAB_DIR}"
echo "  conda 환경: ${CONDA_ENV}"

command -v git >/dev/null || die "git을 찾을 수 없습니다."

if command -v nvidia-smi >/dev/null; then
    ok "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
else
    warn "nvidia-smi 없음 — GPU 없이는 시뮬레이션이 동작하지 않습니다."
fi

# Isaac Sim은 40GB 이상을 사용한다 (pip 캐시 포함).
AVAIL_GB="$(df -BG --output=avail "$(dirname "$LAB_DIR")" 2>/dev/null | tail -1 | tr -dc '0-9' || echo 0)"
if [ "${AVAIL_GB:-0}" -lt 60 ] && [ "$DO_INSTALL" -eq 1 ]; then
    warn "여유 공간 ${AVAIL_GB}GB — Isaac Sim 설치에는 60GB 이상을 권장합니다."
else
    ok "여유 공간 ${AVAIL_GB}GB"
fi

# Isaac Sim 첫 실행 시 NVIDIA EULA 프롬프트가 뜨는데, 비대화형 셸에서는
# "Unable to bootstrap inner kit kernel: EOF when reading a line"로 실패한다.
export OMNI_KIT_ACCEPT_EULA=YES

# ---------------------------------------------------------------------------
# 1. 저장소 clone
# ---------------------------------------------------------------------------
if [ "$DO_INSTALL" -eq 1 ]; then
    log "Isaac Lab ${ISAACLAB_TAG} clone → ${LAB_DIR}"

    if [ -d "$LAB_DIR" ]; then
        if [ "$FORCE" -eq 1 ]; then
            [ -f "${LAB_DIR}/isaaclab.sh" ] || die "${LAB_DIR}가 Isaac Lab 저장소가 아닙니다."
            ok "기존 폴더 재사용 (--force)"
        else
            die "${LAB_DIR}가 이미 존재합니다. 다른 이름을 쓰거나 --force로 재사용하세요."
        fi
    else
        git clone --depth 1 --branch "$ISAACLAB_TAG" "$ISAACLAB_REPO" "$LAB_DIR"
        ok "clone 완료 (v$(cat "${LAB_DIR}/VERSION"))"
    fi
fi

[ -f "${LAB_DIR}/isaaclab.sh" ] || die "isaaclab.sh를 찾을 수 없습니다: ${LAB_DIR}"

# ---------------------------------------------------------------------------
# 2. conda 환경 생성
# ---------------------------------------------------------------------------
ENV_EXISTS=0
env_exists "$CONDA_ENV" && ENV_EXISTS=1

if [ "$DO_INSTALL" -eq 1 ]; then
    log "conda 환경 준비: ${CONDA_ENV}"

    if [ "$ENV_EXISTS" -eq 1 ]; then
        if [ "$FORCE" -eq 1 ]; then
            ok "기존 환경 재사용 (--force)"
        else
            die "conda 환경 '${CONDA_ENV}'가 이미 존재합니다. 다른 이름을 쓰거나 --force로 재사용하세요."
        fi
    else
        conda create -n "$CONDA_ENV" "python=${PYTHON_VERSION}" -y
        ok "환경 생성 완료"
    fi
else
    [ "$ENV_EXISTS" -eq 1 ] || die "conda 환경 '${CONDA_ENV}'가 없습니다."
fi

# conda activate는 내부적으로 0이 아닌 값을 반환하는 명령을 포함해 set -e에 걸린다.
# 따라서 종료 코드가 아니라 실제 활성화 결과로 성공 여부를 판정한다.
set +e
conda activate "$CONDA_ENV"
set -e
[ "${CONDA_DEFAULT_ENV:-}" = "$CONDA_ENV" ] || die "conda 환경 '${CONDA_ENV}' 활성화에 실패했습니다."
ok "활성화: ${CONDA_DEFAULT_ENV} (${CONDA_PREFIX})"

PY_VER="$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
[ "$PY_VER" = "$PYTHON_VERSION" ] || die "Isaac Sim 5.x는 Python ${PYTHON_VERSION}이 필요합니다 (현재: ${PY_VER})."
ok "Python ${PY_VER}"

# ---------------------------------------------------------------------------
# 3. 설치
# ---------------------------------------------------------------------------
if [ "$DO_INSTALL" -eq 1 ]; then
    log "pip 업그레이드"
    pip install --upgrade pip

    # 중단된 pip uninstall이 남긴 ~orch, ~unctorch 같은 파편을 정리한다.
    # 남아 있으면 매 pip 호출마다 "Ignoring invalid distribution" 경고가 나고
    # import를 방해할 수 있다.
    SITE_PACKAGES="$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
    LEFTOVERS="$(find "$SITE_PACKAGES" -maxdepth 1 -name '~*' 2>/dev/null | wc -l)"
    if [ "$LEFTOVERS" -gt 0 ]; then
        find "$SITE_PACKAGES" -maxdepth 1 -name '~*' -exec rm -rf {} + 2>/dev/null || true
        ok "중단된 설치 파편 ${LEFTOVERS}개 정리"
    fi

    # pip 제약 파일. 두 가지 문제를 한 번에 막는다.
    #
    # 1) setuptools<81 — flatdict==4.0.1은 sdist만 제공되고 setup.py가 pkg_resources를
    #    import하는데, setuptools 81+는 pkg_resources를 제거했다. 그대로 두면 빌드 격리
    #    환경이 최신 setuptools를 받아 "No module named 'pkg_resources'"로 실패한다.
    #    (제약은 빌드 격리 환경에만 걸리므로 실제 설치되는 패키지 버전은 바뀌지 않는다.)
    #
    # 2) torch 고정 — isaaclab_rl이 stable-baselines3>=2.6을 요구하는데 최신 sb3는
    #    torch>=2.8을 요구한다. 풀어두면 isaaclab.sh -i가 torch를 2.13(+cu13 스택
    #    수 GB)으로 올렸다가 되돌리면서 torchaudio를 잃는다. 고정해두면 resolver가
    #    torch 2.7과 맞는 sb3를 고르므로 이 왕복 자체가 없어진다.
    PIP_CONSTRAINTS="${CONDA_PREFIX}/share/isaaclab-constraints.txt"
    mkdir -p "$(dirname "$PIP_CONSTRAINTS")"
    cat > "$PIP_CONSTRAINTS" <<EOF
setuptools<81
torch==${TORCH_VERSION}
torchvision==${TORCHVISION_VERSION}
torchaudio==${TORCHAUDIO_VERSION}
EOF
    export PIP_CONSTRAINT="$PIP_CONSTRAINTS"
    ok "pip 제약 설정 (setuptools<81, torch ${TORCH_VERSION} 고정)"

    log "Isaac Sim ${ISAACSIM_VERSION} 설치 (수십 GB — 시간이 오래 걸립니다)"
    pip install "isaacsim[all,extscache]==${ISAACSIM_VERSION}" --extra-index-url "${NVIDIA_INDEX}"
    ok "Isaac Sim 패키지 $(pip list 2>/dev/null | grep -ci '^isaacsim')개 설치됨"

    # isaacsim 설치가 torch를 CPU 빌드로 덮어쓸 수 있으므로 이후에 CUDA 빌드를 고정한다.
    # torchaudio는 공식 문서에 없지만 isaacsim-core가 요구하므로 함께 설치한다.
    log "PyTorch ${TORCH_VERSION} (cu128) 설치"
    pip install -U "torch==${TORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}" \
                   "torchaudio==${TORCHAUDIO_VERSION}" --index-url "${TORCH_INDEX}"

    log "Isaac Lab 의존성 설치 (isaaclab.sh -i)"
    # isaacsim이 starlette 등을 다운그레이드하므로 Isaac Lab 요구 버전으로 되돌린다.
    "${LAB_DIR}/isaaclab.sh" -i

    # isaaclab.sh -i는 모듈을 하나씩 pip install하면서 개별 실패를 무시하고 계속 진행해
    # 0으로 끝난다. 즉 종료 코드만 믿으면 isaaclab 코어가 빠진 설치를 "성공"으로 오인한다.
    # 실제로 설치됐는지 직접 확인한다 (--skip-verify와 무관하게 항상).
    MISSING=""
    for m in isaaclab isaaclab_assets isaaclab_mimic isaaclab_rl isaaclab_tasks; do
        pip show "$m" >/dev/null 2>&1 || MISSING="${MISSING} ${m}"
    done
    [ -z "$MISSING" ] || die "isaaclab.sh -i가 0을 반환했지만 다음 모듈이 설치되지 않았습니다:${MISSING}
     위쪽 로그에서 해당 모듈의 pip 실패 원인을 확인하세요."
    ok "Isaac Lab 모듈 5개 설치 확인"

    # isaaclab.sh -i가 torch 계열을 다시 건드렸을 수 있으므로 cu128 빌드를 다시 못박는다.
    # (제약을 걸어도 -i가 torchaudio를 지우고 가는 경우가 있어 마지막에 재확인한다.)
    log "PyTorch cu128 재확인"
    pip install "torch==${TORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}" \
                "torchaudio==${TORCHAUDIO_VERSION}" --index-url "${TORCH_INDEX}"

    # -----------------------------------------------------------------------
    # 4. conda 활성화 스크립트 설정
    # -----------------------------------------------------------------------
    log "conda 활성화 스크립트 설정"
    # setenv.sh는 `isaaclab.sh --conda`(환경 생성)가 만드는 파일이라
    # conda create로 만든 환경에는 존재하지 않는다. 없으면 직접 만든다.
    SETENV="${CONDA_PREFIX}/etc/conda/activate.d/setenv.sh"
    mkdir -p "$(dirname "$SETENV")"

    if [ ! -f "$SETENV" ]; then
        cat > "$SETENV" <<EOF
#!/usr/bin/env bash

# for Isaac Lab
export ISAACLAB_PATH=${LAB_DIR}
alias isaaclab=${LAB_DIR}/isaaclab.sh

# show icon if not running headless
export RESOURCE_NAME="IsaacSim"
EOF
        ok "setenv.sh 생성 (ISAACLAB_PATH, isaaclab alias)"
    else
        # isaaclab.sh --conda는 생성 시점에 _isaac_sim/ 이 있으면 무조건 source 하는 줄을 넣는다.
        # 바이너리 설치를 지우고 pip으로 전환하면 이 줄이 매 activate마다 에러를 낸다.
        if grep -q '^source .*_isaac_sim/setup_conda_env.sh' "$SETENV"; then
            cp "$SETENV" "${SETENV}.bak"
            sed -i 's|^source \(.*_isaac_sim/setup_conda_env.sh\)|if [ -f "\1" ]; then source "\1"; fi|' "$SETENV"
            ok "_isaac_sim 참조를 조건부로 변경 (백업: ${SETENV}.bak)"
        fi
    fi

    if ! grep -q 'OMNI_KIT_ACCEPT_EULA' "$SETENV"; then
        printf '\n# accept the NVIDIA Omniverse EULA non-interactively\nexport OMNI_KIT_ACCEPT_EULA=YES\n' >> "$SETENV"
        ok "OMNI_KIT_ACCEPT_EULA=YES 추가"
    fi
fi

# ---------------------------------------------------------------------------
# 5. 검증
# ---------------------------------------------------------------------------
if [ "$DO_VERIFY" -eq 1 ]; then
    log "설치 검증"

    python -c "import isaacsim" 2>/dev/null && ok "isaacsim import" || die "isaacsim을 import할 수 없습니다."
    python -c "import isaaclab"  2>/dev/null && ok "isaaclab import"  || die "isaaclab을 import할 수 없습니다."

    # isaacsim-core가 torchaudio==2.7.0을 요구한다. 설치 과정의 torch 교체 와중에
    # 조용히 사라지는 일이 있어 명시적으로 확인한다.
    python -c "import torchaudio" 2>/dev/null && ok "torchaudio import" \
        || die "torchaudio가 없습니다 (isaacsim-core 요구사항). --force로 재실행하세요."

    python - <<'PY' || die "torch가 CUDA를 인식하지 못합니다."
import sys, torch
print(f"  ✓ torch {torch.__version__} / CUDA {torch.cuda.is_available()}", end="")
print(f" / {torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else "")
if "+cu" not in torch.__version__:
    print(f"  ! torch {torch.__version__}는 CPU 빌드입니다 — cu128 빌드가 아닙니다.")
    sys.exit(1)
sys.exit(0 if torch.cuda.is_available() else 1)
PY
fi

log "설치 완료"
cat <<EOF

  사용을 시작하려면:

    conda activate ${CONDA_ENV}
    cd ${LAB_DIR}

  동작 확인 (빈 씬 생성 — Ctrl+C로 종료):

    ./isaaclab.sh -p scripts/tutorials/00_sim/create_empty.py --headless

  데모 실행:

    # 로컬 디스플레이가 있는 경우
    ./isaaclab.sh -p scripts/demos/h1_locomotion.py

    # 원격(SSH) 접속인 경우 — WebRTC 스트리밍 클라이언트로 <서버IP>:49100 접속
    ./isaaclab.sh -p scripts/demos/h1_locomotion.py --livestream 2

  삭제하려면:

    $(basename "${BASH_SOURCE[0]}") --uninstall $(basename "$NAME_ARG")

EOF
