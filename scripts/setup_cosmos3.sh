#!/usr/bin/env bash
#
# setup_cosmos3.sh — Cosmos3 Reasoner (vLLM) / Generator (vLLM-Omni) 설치 및 실행 스크립트
#
# 참고: https://github.com/NVIDIA/cosmos/tree/main/cookbooks/cosmos3
#   - Reasoner  : vLLM       (#vllm)
#   - Generator : vLLM-Omni  (#vllm-omni)
#
# 사용법은 `./setup_cosmos3.sh help` 또는 COSMOS3_SETUP.md 참고.

set -Eeuo pipefail

# ---------------------------------------------------------------------------
# 기본 설정 (환경변수로 덮어쓸 수 있음)
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

COSMOS3_ROOT="${COSMOS3_ROOT:-${SCRIPT_DIR}/cosmos3}"
REASONER_VENV="${COSMOS3_ROOT}/venv-reasoner"
GENERATOR_VENV="${COSMOS3_ROOT}/venv-generator"
COSMOS3_MEDIA_ROOT="${COSMOS3_MEDIA_ROOT:-${COSMOS3_ROOT}/media}"
COSMOS3_OUTPUT_DIR="${COSMOS3_OUTPUT_DIR:-${COSMOS3_ROOT}/outputs}"
COSMOS3_WORKDIR="${COSMOS3_WORKDIR:-${COSMOS3_ROOT}}"

HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"

PYTHON_VERSION="${COSMOS3_PYTHON_VERSION:-3.13}"
VLLM_SPEC="${COSMOS3_VLLM_SPEC:-vllm>=0.23.0}"
VLLM_OMNI_SPEC="${COSMOS3_VLLM_OMNI_SPEC:-vllm-omni @ git+https://github.com/vllm-project/vllm-omni.git@refs/pull/3454/head}"
VLLM_OMNI_IMAGE="${COSMOS3_VLLM_OMNI_IMAGE:-vllm/vllm-omni:cosmos3}"

# 실행 인자 기본값
COMPONENT=""
ACTION=""
MODEL="nano"
PORT=""
GENERATOR_BACKEND="docker"   # docker | venv
TORCH_BACKEND=""             # 비어있으면 드라이버에서 자동 감지
DRY_RUN=0

# ---------------------------------------------------------------------------
# 출력 헬퍼
# ---------------------------------------------------------------------------
if [[ -t 1 ]]; then
  C_RESET=$'\033[0m'; C_BOLD=$'\033[1m'; C_RED=$'\033[31m'
  C_GREEN=$'\033[32m'; C_YELLOW=$'\033[33m'; C_BLUE=$'\033[34m'
else
  C_RESET=""; C_BOLD=""; C_RED=""; C_GREEN=""; C_YELLOW=""; C_BLUE=""
fi

info()  { printf '%s[info]%s  %s\n'  "${C_BLUE}"   "${C_RESET}" "$*"; }
ok()    { printf '%s[ok]%s    %s\n'  "${C_GREEN}"  "${C_RESET}" "$*"; }
warn()  { printf '%s[warn]%s  %s\n'  "${C_YELLOW}" "${C_RESET}" "$*" >&2; }
err()   { printf '%s[error]%s %s\n'  "${C_RED}"    "${C_RESET}" "$*" >&2; }
die()   { err "$*"; exit 1; }
head1() { printf '\n%s=== %s ===%s\n' "${C_BOLD}" "$*" "${C_RESET}"; }

# DRY_RUN=1 이면 실행하지 않고 명령만 출력
run() {
  if (( DRY_RUN )); then
    printf '%s[dry-run]%s %s\n' "${C_YELLOW}" "${C_RESET}" "$*"
  else
    printf '%s[run]%s %s\n' "${C_BLUE}" "${C_RESET}" "$*"
    "$@"
  fi
}

# ---------------------------------------------------------------------------
# 사용법
# ---------------------------------------------------------------------------
usage() {
  cat <<'EOF'
setup_cosmos3.sh — Cosmos3 Reasoner(vLLM) / Generator(vLLM-Omni) 환경 구축

사용법:
  ./setup_cosmos3.sh <action> <component> [options]

Action:
  check                사전 요구사항(GPU/드라이버/uv/docker/HF_TOKEN)만 점검
  install              선택한 컴포넌트 설치
  serve                선택한 컴포넌트의 서버 실행 (포그라운드)
  verify               실행 중인 서버에 헬스체크 요청
  example              실행 중인 서버로 샘플 추론 요청 전송
  clean                생성한 venv / 작업 디렉터리 삭제
  help                 이 도움말 출력

Component:
  reasoner             Cosmos3 Reasoner — vLLM 백엔드
  generator            Cosmos3 Generator — vLLM-Omni 백엔드
  all                  둘 다 (install / clean / check 에서만 사용 가능)

Options:
  --model <nano|super>          모델 선택 (기본: nano)
                                nano  = nvidia/Cosmos3-Nano  (GPU 1장)
                                super = nvidia/Cosmos3-Super (GPU 4장, TP=4)
  --port <N>                    서버 포트 (기본: reasoner=8000, generator=8001)
  --generator-backend <docker|venv>
                                Generator 설치/실행 방식 (기본: docker)
                                  docker = 공식 이미지, 전체 모달리티 지원 (권장)
                                  venv   = 네이티브 설치, t2i/t2v/i2v 만 지원
  --torch-backend <cu130|cu128> torch 휠 백엔드 강제 지정 (기본: 드라이버에서 자동 감지)
  --dry-run                     명령을 실행하지 않고 출력만 함
  -h, --help                    도움말

예시:
  ./setup_cosmos3.sh check all
  ./setup_cosmos3.sh install all
  ./setup_cosmos3.sh install reasoner
  ./setup_cosmos3.sh install generator --generator-backend venv
  ./setup_cosmos3.sh serve reasoner --model nano --port 8000
  ./setup_cosmos3.sh serve generator --model nano --port 8001
  ./setup_cosmos3.sh verify reasoner --port 8000
  ./setup_cosmos3.sh example generator --port 8001
  ./setup_cosmos3.sh install all --dry-run

환경변수:
  COSMOS3_ROOT         작업 루트 (기본: <script_dir>/cosmos3)
  COSMOS3_MEDIA_ROOT   로컬 미디어 루트 (기본: $COSMOS3_ROOT/media)
  HF_HOME              HuggingFace 캐시 (기본: ~/.cache/huggingface)
  HF_TOKEN             게이트된 모델(nvidia/Cosmos-1.0-Guardrail 등) 접근용 토큰
EOF
}

# ---------------------------------------------------------------------------
# 인자 파싱
# ---------------------------------------------------------------------------
parse_args() {
  [[ $# -eq 0 ]] && { usage; exit 0; }

  case "$1" in
    check|install|serve|verify|example|clean) ACTION="$1"; shift ;;
    help|-h|--help) usage; exit 0 ;;
    *) die "알 수 없는 action: '$1' (./setup_cosmos3.sh help 참고)" ;;
  esac

  # check/clean 은 component 생략 시 all 로 간주
  if [[ $# -gt 0 && "$1" != -* ]]; then
    case "$1" in
      reasoner|generator|all) COMPONENT="$1"; shift ;;
      *) die "알 수 없는 component: '$1' (reasoner|generator|all)" ;;
    esac
  else
    case "${ACTION}" in
      check|clean) COMPONENT="all" ;;
      *) die "'${ACTION}' 에는 component 가 필요합니다 (reasoner|generator|all)" ;;
    esac
  fi

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --model)              MODEL="${2:-}"; shift 2 ;;
      --port)               PORT="${2:-}"; shift 2 ;;
      --generator-backend)  GENERATOR_BACKEND="${2:-}"; shift 2 ;;
      --torch-backend)      TORCH_BACKEND="${2:-}"; shift 2 ;;
      --dry-run)            DRY_RUN=1; shift ;;
      -h|--help)            usage; exit 0 ;;
      *) die "알 수 없는 옵션: '$1'" ;;
    esac
  done

  [[ "${MODEL}" =~ ^(nano|super)$ ]] || die "--model 은 nano 또는 super 여야 합니다 (입력: '${MODEL}')"
  [[ "${GENERATOR_BACKEND}" =~ ^(docker|venv)$ ]] || \
    die "--generator-backend 는 docker 또는 venv 여야 합니다 (입력: '${GENERATOR_BACKEND}')"
  [[ -z "${TORCH_BACKEND}" || "${TORCH_BACKEND}" =~ ^(cu130|cu128)$ ]] || \
    die "--torch-backend 는 cu130 또는 cu128 이어야 합니다 (입력: '${TORCH_BACKEND}')"

  if [[ "${COMPONENT}" == "all" && ! "${ACTION}" =~ ^(install|clean|check)$ ]]; then
    die "'all' 은 install/clean/check 에서만 사용할 수 있습니다. '${ACTION}' 에는 reasoner 또는 generator 를 지정하세요."
  fi
}

# ---------------------------------------------------------------------------
# 조회 헬퍼
# ---------------------------------------------------------------------------
model_repo() {
  case "$1" in
    nano)  echo "nvidia/Cosmos3-Nano" ;;
    super) echo "nvidia/Cosmos3-Super" ;;
  esac
}

# nano=1장, super=4장
model_tp_size() {
  case "$1" in
    nano)  echo 1 ;;
    super) echo 4 ;;
  esac
}

default_port() {
  case "$1" in
    reasoner)  echo 8000 ;;
    generator) echo 8001 ;;
  esac
}

# nvidia-smi 의 "CUDA Version: X.Y" 로 torch 휠 백엔드 결정
detect_torch_backend() {
  [[ -n "${TORCH_BACKEND}" ]] && { echo "${TORCH_BACKEND}"; return; }

  local cuda_ver major
  cuda_ver="$(nvidia-smi 2>/dev/null | sed -n 's/.*CUDA Version: \([0-9]\+\.[0-9]\+\).*/\1/p' | head -1)"
  if [[ -z "${cuda_ver}" ]]; then
    warn "드라이버의 CUDA 버전을 감지하지 못했습니다. cu128 로 폴백합니다 (--torch-backend 로 강제 지정 가능)."
    echo "cu128"; return
  fi

  major="${cuda_ver%%.*}"
  if (( major >= 13 )); then echo "cu130"; else echo "cu128"; fi
}

gpu_count() { nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l; }

# super 모델을 GPU 4장 미만에서 실행하려 할 때 경고
warn_if_insufficient_gpus() {
  [[ "${MODEL}" != "super" ]] && return 0
  local ngpu; ngpu="$(gpu_count)"
  (( ngpu >= 4 )) && return 0
  warn "Cosmos3-Super 는 GPU 4장(TP=4) 기준인데 현재 ${ngpu}장입니다 — OOM 가능성이 높습니다."
  warn "  단일 GPU 환경에서는 --model nano 를 사용하세요."
}

# ---------------------------------------------------------------------------
# 사전 점검
# ---------------------------------------------------------------------------
check_prereqs() {
  local component="$1"
  local failed=0

  head1 "사전 요구사항 점검 (${component})"

  # --- GPU / 드라이버 ---
  if command -v nvidia-smi >/dev/null 2>&1; then
    local cuda_ver ngpu
    cuda_ver="$(nvidia-smi 2>/dev/null | sed -n 's/.*CUDA Version: \([0-9]\+\.[0-9]\+\).*/\1/p' | head -1)"
    ngpu="$(gpu_count)"
    ok "nvidia-smi 확인됨 — GPU ${ngpu}장, 드라이버 CUDA ${cuda_ver:-unknown}"
    nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader | sed 's/^/         /'
    info "선택된 torch 백엔드: $(detect_torch_backend)"
  else
    err "nvidia-smi 를 찾을 수 없습니다. NVIDIA 드라이버가 필요합니다."
    failed=1
  fi

  # --- uv (reasoner 항상, generator 는 venv 백엔드일 때) ---
  if [[ "${component}" == "reasoner" || "${component}" == "all" || "${GENERATOR_BACKEND}" == "venv" ]]; then
    if command -v uv >/dev/null 2>&1; then
      ok "uv 확인됨 — $(uv --version)"
    else
      err "uv 를 찾을 수 없습니다. 설치: curl -LsSf https://astral.sh/uv/install.sh | sh"
      failed=1
    fi
  fi

  # --- docker (generator + docker 백엔드) ---
  if [[ "${component}" == "generator" || "${component}" == "all" ]] && [[ "${GENERATOR_BACKEND}" == "docker" ]]; then
    if command -v docker >/dev/null 2>&1; then
      ok "docker 확인됨 — $(docker --version)"
      if docker info --format '{{json .Runtimes}}' 2>/dev/null | grep -q nvidia; then
        ok "docker nvidia 런타임 확인됨"
      else
        warn "docker 에 nvidia 런타임이 보이지 않습니다. nvidia-container-toolkit 설치가 필요할 수 있습니다."
      fi
    else
      err "docker 를 찾을 수 없습니다 (--generator-backend venv 로 우회 가능)."
      failed=1
    fi
  fi

  # --- HF 토큰 ---
  if [[ -n "${HF_TOKEN:-}" ]]; then
    ok "HF_TOKEN 설정됨"
  elif [[ -f "${HF_HOME}/token" ]]; then
    ok "HuggingFace 토큰 파일 확인됨 — ${HF_HOME}/token"
  else
    warn "HF_TOKEN 이 없습니다. 게이트된 리포(nvidia/Cosmos-1.0-Guardrail 등) 다운로드가 실패합니다."
    warn "  export HF_TOKEN=<token>   또는   uvx hf@latest auth login"
  fi

  # --- VRAM 관련 안내 ---
  if [[ "${MODEL}" == "super" ]]; then
    local ngpu; ngpu="$(gpu_count)"
    if (( ngpu < 4 )); then
      warn "Cosmos3-Super 는 GPU 4장(TP=4) 기준입니다. 현재 ${ngpu}장 — 실행 시 OOM 가능성이 높습니다."
    fi
  fi

  head1 "점검 결과"
  if (( failed )); then
    err "필수 요구사항이 충족되지 않았습니다."
    return 1
  fi
  ok "사전 요구사항 충족."
}

ensure_dirs() {
  run mkdir -p "${COSMOS3_ROOT}" "${COSMOS3_MEDIA_ROOT}" "${COSMOS3_OUTPUT_DIR}" "${HF_HOME}"
}

# ---------------------------------------------------------------------------
# 설치 — Reasoner (vLLM)
# ---------------------------------------------------------------------------
install_reasoner() {
  head1 "Reasoner 설치 (vLLM)"
  local backend; backend="$(detect_torch_backend)"

  ensure_dirs
  info "venv: ${REASONER_VENV} (python ${PYTHON_VERSION}, torch 백엔드 ${backend})"

  run uv venv --python "${PYTHON_VERSION}" --seed --managed-python "${REASONER_VENV}"
  run env VIRTUAL_ENV="${REASONER_VENV}" \
      uv pip install --python "${REASONER_VENV}/bin/python" \
        --torch-backend="${backend}" "${VLLM_SPEC}"

  ok "Reasoner 설치 완료 — ${REASONER_VENV}"
  info "실행: ./setup_cosmos3.sh serve reasoner --model ${MODEL}"
}

# ---------------------------------------------------------------------------
# 설치 — Generator (vLLM-Omni)
# ---------------------------------------------------------------------------
install_generator() {
  head1 "Generator 설치 (vLLM-Omni / ${GENERATOR_BACKEND})"
  ensure_dirs

  if [[ "${GENERATOR_BACKEND}" == "docker" ]]; then
    info "공식 이미지 pull: ${VLLM_OMNI_IMAGE} (전체 모달리티 지원: t2i / t2v / i2v / audio / action)"
    run docker pull "${VLLM_OMNI_IMAGE}"
    ok "Generator(docker) 준비 완료 — 이미지 ${VLLM_OMNI_IMAGE}"
  else
    local backend; backend="$(detect_torch_backend)"
    warn "네이티브 venv 설치는 text-to-image / text-to-video / image-to-video 만 지원합니다 (audio/action 미지원)."
    info "venv: ${GENERATOR_VENV} (python ${PYTHON_VERSION}, torch 백엔드 ${backend})"

    run uv venv --python "${PYTHON_VERSION}" --seed --managed-python "${GENERATOR_VENV}"
    run env VIRTUAL_ENV="${GENERATOR_VENV}" \
        uv pip install --python "${GENERATOR_VENV}/bin/python" \
          --torch-backend="${backend}" "${VLLM_OMNI_SPEC}"

    ok "Generator(venv) 설치 완료 — ${GENERATOR_VENV}"
  fi

  info "실행: ./setup_cosmos3.sh serve generator --model ${MODEL} --generator-backend ${GENERATOR_BACKEND}"
}

# ---------------------------------------------------------------------------
# 실행 — Reasoner (vLLM)
# ---------------------------------------------------------------------------
serve_reasoner() {
  local repo tp port devices
  repo="$(model_repo "${MODEL}")"
  tp="$(model_tp_size "${MODEL}")"
  port="${PORT:-$(default_port reasoner)}"
  devices="$(seq -s, 0 $((tp - 1)))"

  head1 "Reasoner 실행 — ${repo} (TP=${tp}, port=${port})"
  warn_if_insufficient_gpus

  [[ -x "${REASONER_VENV}/bin/vllm" ]] || (( DRY_RUN )) || \
    die "vLLM 이 설치되지 않았습니다. 먼저: ./setup_cosmos3.sh install reasoner"

  info "OpenAI 호환 엔드포인트: http://localhost:${port}/v1"
  info "미디어 루트: ${COSMOS3_MEDIA_ROOT}"

  run env \
    CUDA_VISIBLE_DEVICES="${devices}" \
    HF_HOME="${HF_HOME}" \
    ${HF_TOKEN:+HF_TOKEN="${HF_TOKEN}"} \
    "${REASONER_VENV}/bin/vllm" serve "${repo}" \
      --tensor-parallel-size "${tp}" \
      --mm-encoder-tp-mode data \
      --async-scheduling \
      --allowed-local-media-path "${COSMOS3_MEDIA_ROOT}" \
      --media-io-kwargs '{"video": {"num_frames": -1}}' \
      --port "${port}"
}

# ---------------------------------------------------------------------------
# 실행 — Generator (vLLM-Omni)
# ---------------------------------------------------------------------------
serve_generator() {
  local repo tp port
  repo="$(model_repo "${MODEL}")"
  tp="$(model_tp_size "${MODEL}")"
  port="${PORT:-$(default_port generator)}"

  head1 "Generator 실행 — ${repo} (${GENERATOR_BACKEND}, TP=${tp}, port=${port})"
  warn_if_insufficient_gpus

  if [[ "${GENERATOR_BACKEND}" == "docker" ]]; then
    local gpu_arg=( --gpus all )
    local extra=()
    if [[ "${MODEL}" == "nano" ]]; then
      # exec 로 직접 전달하므로 README 의 셸 인용부호('"device=0"')는 불필요
      gpu_arg=( --gpus device=0 )
    else
      # Super: TP=4 + layerwise offload 로 VRAM 사용량 완화
      extra=( --tensor-parallel-size "${tp}" --enable-layerwise-offload )
    fi

    run docker run --rm -it --runtime nvidia "${gpu_arg[@]}" \
      -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
      ${HF_TOKEN:+-e HF_TOKEN="${HF_TOKEN}"} \
      -v "${HF_HOME}:/root/.cache/huggingface" \
      -v "${COSMOS3_WORKDIR}:/workspace" \
      -p "${port}:8000" --ipc=host \
      "${VLLM_OMNI_IMAGE}" \
      vllm serve "${repo}" \
        --omni \
        --model-class-name Cosmos3OmniDiffusersPipeline \
        --allowed-local-media-path / \
        "${extra[@]}" \
        --port 8000 \
        --init-timeout 1800
  else
    [[ -x "${GENERATOR_VENV}/bin/vllm" ]] || (( DRY_RUN )) || \
      die "vLLM-Omni 가 설치되지 않았습니다. 먼저: ./setup_cosmos3.sh install generator --generator-backend venv"

    local extra=()
    [[ "${MODEL}" == "super" ]] && extra=( --tensor-parallel-size "${tp}" --enable-layerwise-offload )

    run env \
      CUDA_VISIBLE_DEVICES="$(seq -s, 0 $((tp - 1)))" \
      HF_HOME="${HF_HOME}" \
      ${HF_TOKEN:+HF_TOKEN="${HF_TOKEN}"} \
      "${GENERATOR_VENV}/bin/vllm" serve "${repo}" \
        --omni \
        --model-class-name Cosmos3OmniDiffusersPipeline \
        --allowed-local-media-path "${COSMOS3_MEDIA_ROOT}" \
        "${extra[@]}" \
        --port "${port}" \
        --init-timeout 1800
  fi
}

# ---------------------------------------------------------------------------
# 헬스체크
# ---------------------------------------------------------------------------
verify_server() {
  local component="$1"
  local port; port="${PORT:-$(default_port "${component}")}"

  head1 "헬스체크 — ${component} @ localhost:${port}"

  if ! curl -sS --max-time 5 "http://localhost:${port}/v1/models" -o /dev/null 2>/dev/null; then
    err "http://localhost:${port}/v1/models 응답 없음 — 서버가 실행 중인지 확인하세요."
    return 1
  fi

  ok "서버 응답 확인됨. 로드된 모델:"
  curl -sS "http://localhost:${port}/v1/models" | { jq . 2>/dev/null || cat; }
}

# ---------------------------------------------------------------------------
# 샘플 추론 요청
# ---------------------------------------------------------------------------
example_reasoner() {
  local port repo
  port="${PORT:-$(default_port reasoner)}"
  repo="$(model_repo "${MODEL}")"

  head1 "Reasoner 샘플 요청 (OpenAI 호환 chat completions)"

  run curl -sS -X POST "http://localhost:${port}/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "{
      \"model\": \"${repo}\",
      \"messages\": [
        {\"role\": \"user\", \"content\": \"Describe what a warehouse robot should check before lifting a pallet.\"}
      ],
      \"max_tokens\": 256,
      \"temperature\": 0.2
    }"
  echo
}

example_generator() {
  local port out
  port="${PORT:-$(default_port generator)}"
  out="${COSMOS3_OUTPUT_DIR}/cosmos3_generator_t2v.mp4"

  head1 "Generator 샘플 요청 (text-to-video)"
  info "출력 파일: ${out}"

  if (( DRY_RUN )); then
    printf '%s[dry-run]%s curl -sS -X POST http://localhost:%s/v1/infer ... > %s\n' \
      "${C_YELLOW}" "${C_RESET}" "${port}" "${out}"
    return 0
  fi

  mkdir -p "${COSMOS3_OUTPUT_DIR}"
  curl -sS -X POST "http://localhost:${port}/v1/infer" \
    -H 'Accept: application/json' \
    -H 'Content-Type: application/json' \
    -d '{
      "prompt": "A humanoid robot walks through a futuristic warehouse, inspecting shelves of mechanical components.",
      "seed": 42,
      "guidance_scale": 6.0,
      "steps": 35,
      "resolution": "256",
      "num_output_frames": 25,
      "fps": 24.0
    }' | jq -r '.b64_video' | base64 -d > "${out}"

  ok "생성 완료 — ${out}"
}

# ---------------------------------------------------------------------------
# 정리
# ---------------------------------------------------------------------------
clean_component() {
  local component="$1"
  head1 "정리 — ${component}"
  case "${component}" in
    reasoner)  run rm -rf "${REASONER_VENV}" ;;
    generator) run rm -rf "${GENERATOR_VENV}" ;;
    all)       run rm -rf "${REASONER_VENV}" "${GENERATOR_VENV}" ;;
  esac
  ok "삭제 완료 (HF 캐시 ${HF_HOME} 및 docker 이미지는 유지됩니다)"
}

# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
main() {
  parse_args "$@"

  case "${ACTION}" in
    check)
      check_prereqs "${COMPONENT}"
      ;;
    install)
      check_prereqs "${COMPONENT}" || die "사전 요구사항 미충족으로 설치를 중단합니다."
      case "${COMPONENT}" in
        reasoner)  install_reasoner ;;
        generator) install_generator ;;
        all)       install_reasoner; install_generator ;;
      esac
      head1 "설치 완료"
      info "다음 단계는 COSMOS3_SETUP.md 참고"
      ;;
    serve)
      case "${COMPONENT}" in
        reasoner)  serve_reasoner ;;
        generator) serve_generator ;;
      esac
      ;;
    verify)
      verify_server "${COMPONENT}"
      ;;
    example)
      case "${COMPONENT}" in
        reasoner)  example_reasoner ;;
        generator) example_generator ;;
      esac
      ;;
    clean)
      clean_component "${COMPONENT}"
      ;;
  esac
}

main "$@"
