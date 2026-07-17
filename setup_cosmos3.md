# Cosmos3 — Reasoner (vLLM) / Generator (vLLM-Omni) 설치 및 실행 가이드

NVIDIA Cosmos3 의 **Reasoner** 를 vLLM 으로, **Generator** 를 vLLM-Omni 로 각각 띄우기 위한
설치/실행 스크립트(`setup_cosmos3.sh`) 사용 설명서입니다.

원문 참고: <https://github.com/NVIDIA/cosmos/tree/main/cookbooks/cosmos3>
([#vllm](https://github.com/NVIDIA/cosmos/tree/main/cookbooks/cosmos3#vllm) /
[#vllm-omni](https://github.com/NVIDIA/cosmos/tree/main/cookbooks/cosmos3#vllm-omni))

> **주의:** 이 스크립트는 설치와 서버 기동까지를 담당합니다. 이 문서를 작성한 환경(RTX A6000 48GB × 1)에서는
> VRAM 이 부족하여 **전체 모델 로드/추론 테스트는 수행하지 않았습니다.** 아래 "검증 범위" 절을 확인하세요.

```bash
curl -LsSf https://raw.githubusercontent.com/knowledge-intelligence/KIMe-Tutorials/main/setup_cosmos3.sh | bash -s install all
```

---

## 1. 두 컴포넌트 개요

|       | **Reasoner**                                  | **Generator**           |
| ----- | --------------------------------------------- | ----------------------- |
| 백엔드   | vLLM (`vllm>=0.23.0`)                         | vLLM-Omni (`vllm-omni`) |
| 역할    | 멀티모달 이해/추론 (VLM)                              | 이미지·비디오 생성 (Diffusion)  |
| API   | OpenAI 호환 `/v1/chat/completions`              | `/v1/infer`             |
| 기본 포트 | `8000`                                        | `8001`                  |
| 설치 방식 | uv venv (네이티브)                                | Docker(권장) 또는 uv venv   |
| 모델    | `nvidia/Cosmos3-Nano`, `nvidia/Cosmos3-Super` | 동일                      |

두 백엔드는 **의존성이 충돌**하므로 스크립트가 각각 별도의 venv 를 만듭니다
(`cosmos3/venv-reasoner`, `cosmos3/venv-generator`).

### Nano vs Super

| 모델    | HF 리포                  | GPU | 텐서 병렬 |
| ----- | ---------------------- | --- | ----- |
| Nano  | `nvidia/Cosmos3-Nano`  | 1장  | TP=1  |
| Super | `nvidia/Cosmos3-Super` | 4장  | TP=4  |

단일 GPU 환경에서는 **Nano** 를 사용하세요. Super 를 4장 미만에서 실행하면 스크립트가 경고하고,
실제 실행 시 OOM 이 발생할 가능성이 높습니다.

---

## 2. 사전 요구사항

- NVIDIA 드라이버 (CUDA 13 또는 12.x)
- Python 3.13 — `uv --managed-python` 이 자동으로 받아오므로 시스템에 없어도 됩니다
- [`uv`](https://docs.astral.sh/uv/) — 없으면: `curl -LsSf https://astral.sh/uv/install.sh | sh`
- Docker + nvidia-container-toolkit (Generator 를 docker 백엔드로 쓸 때)
- HuggingFace 토큰 — 게이트된 `nvidia/Cosmos-1.0-Guardrail` 접근에 필요

```bash
export HF_TOKEN=<your_token>
# 또는
uvx hf@latest auth login
```

한 번에 점검:

```bash
./setup_cosmos3.sh check all
```

이 명령은 GPU 개수/드라이버 CUDA 버전/`uv`/`docker` nvidia 런타임/HF 토큰을 확인하고,
드라이버 CUDA 버전에 맞춰 torch 휠 백엔드(`cu130` 또는 `cu128`)를 자동으로 선택해 보여줍니다.

---

## 3. 스크립트 사용법

```
./setup_cosmos3.sh <action> <component> [options]
```

### Action

| Action    | 설명                           |
| --------- | ---------------------------- |
| `check`   | 사전 요구사항만 점검                  |
| `install` | 컴포넌트 설치                      |
| `serve`   | 서버 실행 (포그라운드)                |
| `verify`  | 실행 중인 서버 헬스체크 (`/v1/models`) |
| `example` | 샘플 추론 요청 전송                  |
| `clean`   | 생성한 venv 삭제                  |
| `help`    | 도움말                          |

### Component

`reasoner` / `generator` / `all` — `all` 은 `install`, `clean`, `check` 에서만 사용 가능합니다.

### Options

| 옵션                                   | 기본값                           | 설명                 |
| ------------------------------------ | ----------------------------- | ------------------ |
| `--model <nano\|super>`              | `nano`                        | 모델 선택              |
| `--port <N>`                         | reasoner=8000, generator=8001 | 서버 포트              |
| `--generator-backend <docker\|venv>` | `docker`                      | Generator 설치/실행 방식 |
| `--torch-backend <cu130\|cu128>`     | 드라이버에서 자동 감지                  | torch 휠 백엔드 강제 지정  |
| `--dry-run`                          | —                             | 실행하지 않고 명령만 출력     |

### 환경변수

| 변수                   | 기본값                     | 설명                                    |
| -------------------- | ----------------------- | ------------------------------------- |
| `COSMOS3_ROOT`       | `<script_dir>/cosmos3`  | 작업 루트 (venv/미디어/출력)                   |
| `COSMOS3_MEDIA_ROOT` | `$COSMOS3_ROOT/media`   | 로컬 미디어 루트 — `file://` 참조가 이 아래에 있어야 함 |
| `COSMOS3_OUTPUT_DIR` | `$COSMOS3_ROOT/outputs` | 생성 결과 저장 위치                           |
| `HF_HOME`            | `~/.cache/huggingface`  | HuggingFace 캐시                        |
| `HF_TOKEN`           | —                       | 게이트 모델 접근 토큰                          |

---

## 4. Generator: Docker vs venv

|         | **Docker** (기본, 권장)                 | **venv**                |
| ------- | ----------------------------------- | ----------------------- |
| 지원 모달리티 | t2i, t2v, i2v, **audio, action** 전부 | t2i, t2v, i2v **만**     |
| 설치      | `vllm/vllm-omni:cosmos3` 이미지 pull   | upstream PR 브랜치에서 소스 빌드 |
| 의존성     | 이미지에 고정됨                            | `--torch-backend` 수동 관리 |

audio/action 모달리티가 필요하면 반드시 Docker 백엔드를 쓰세요.

---

## 5. 빠른 시작

### 5-1. 전체 설치

```bash
./setup_cosmos3.sh install all
```

Reasoner venv + Generator docker 이미지를 모두 준비합니다.
실제로 무엇이 실행되는지 먼저 보고 싶다면:

```bash
./setup_cosmos3.sh install all --dry-run
```

### 5-2. 개별 설치

```bash
# Reasoner 만 (vLLM)
./setup_cosmos3.sh install reasoner

# Generator 만 (vLLM-Omni, Docker)
./setup_cosmos3.sh install generator

# Generator 를 네이티브 venv 로
./setup_cosmos3.sh install generator --generator-backend venv
```

### 5-3. 서버 실행

`serve` 는 포그라운드로 동작하므로 **터미널을 따로 띄우거나** 백그라운드로 돌리세요.

```bash
# 터미널 A — Reasoner (port 8000)
./setup_cosmos3.sh serve reasoner --model nano --port 8000

# 터미널 B — Generator (port 8001)
./setup_cosmos3.sh serve generator --model nano --port 8001
```

로그 파일로 백그라운드 실행:

```bash
nohup ./setup_cosmos3.sh serve reasoner --model nano > reasoner.log 2>&1 &
tail -f reasoner.log
```

> 첫 실행은 HuggingFace 에서 가중치를 내려받으므로 수십 분이 걸릴 수 있습니다.
> Generator 는 `--init-timeout 1800` (30분)으로 기동 타임아웃을 넉넉히 잡아둡니다.

### 5-4. 헬스체크 및 샘플 요청

```bash
./setup_cosmos3.sh verify reasoner  --port 8000
./setup_cosmos3.sh verify generator --port 8001

./setup_cosmos3.sh example reasoner  --port 8000
./setup_cosmos3.sh example generator --port 8001   # -> cosmos3/outputs/cosmos3_generator_t2v.mp4
```

### 5-5. 정리

```bash
./setup_cosmos3.sh clean all     # venv 삭제 (HF 캐시/docker 이미지는 유지)
```

---

## 6. 스크립트가 실제로 실행하는 명령

`--dry-run` 으로 확인한 실제 명령입니다. 쿡북 원문과 동일합니다.

### Reasoner 설치

```bash
uv venv --python 3.13 --seed --managed-python cosmos3/venv-reasoner
uv pip install --torch-backend=cu130 "vllm>=0.23.0"
```

### Reasoner 실행 (Nano, 1 GPU)

```bash
CUDA_VISIBLE_DEVICES=0 \
vllm serve nvidia/Cosmos3-Nano \
  --tensor-parallel-size 1 \
  --mm-encoder-tp-mode data \
  --async-scheduling \
  --allowed-local-media-path "$COSMOS3_MEDIA_ROOT" \
  --media-io-kwargs '{"video": {"num_frames": -1}}' \
  --port 8000
```

Super 는 `CUDA_VISIBLE_DEVICES=0,1,2,3` + `--tensor-parallel-size 4` 로 바뀝니다.

### Generator 설치

```bash
# docker
docker pull vllm/vllm-omni:cosmos3

# venv
uv venv --python 3.13 --seed --managed-python cosmos3/venv-generator
uv pip install --torch-backend=cu130 \
  "vllm-omni @ git+https://github.com/vllm-project/vllm-omni.git@refs/pull/3454/head"
```

### Generator 실행 (Nano, Docker)

```bash
docker run --rm -it --runtime nvidia --gpus device=0 \
  -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
  -v "${HF_HOME}:/root/.cache/huggingface" \
  -v "${COSMOS3_WORKDIR}:/workspace" \
  -p 8001:8000 --ipc=host \
  vllm/vllm-omni:cosmos3 \
  vllm serve nvidia/Cosmos3-Nano \
    --omni \
    --model-class-name Cosmos3OmniDiffusersPipeline \
    --allowed-local-media-path / \
    --port 8000 \
    --init-timeout 1800
```

Super 는 `--gpus all` + `--tensor-parallel-size 4 --enable-layerwise-offload` 가 추가됩니다.

---

## 7. API 사용 예시

### Reasoner — OpenAI 호환

```bash
curl -sS -X POST http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "nvidia/Cosmos3-Nano",
    "messages": [{"role": "user", "content": "Describe what a warehouse robot should check before lifting a pallet."}],
    "max_tokens": 256
  }'
```

Python (`openai` 클라이언트):

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")
resp = client.chat.completions.create(
    model="nvidia/Cosmos3-Nano",
    messages=[{"role": "user", "content": "Describe this scene."}],
)
print(resp.choices[0].message.content)
```

로컬 비디오/이미지를 입력하려면 파일이 `--allowed-local-media-path` 아래(기본 `cosmos3/media/`)에
있어야 하고, `file://` 절대경로로 참조합니다.

### Generator — text-to-video

```bash
curl -sS -X POST http://localhost:8001/v1/infer \
  -H 'Accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
    "prompt": "A humanoid robot walks through a futuristic warehouse.",
    "seed": 42,
    "guidance_scale": 6.0,
    "steps": 35,
    "resolution": "256",
    "num_output_frames": 25,
    "fps": 24.0
  }' | jq -r '.b64_video' | base64 -d > out.mp4
```

---

## 8. 검증 범위 (중요)

이 스크립트에서 **실제로 확인한 것**:

- `check` — GPU/드라이버/`uv`/`docker` nvidia 런타임/HF 토큰 탐지, CUDA 13.0 → `cu130` 자동 선택
- 모든 `install`/`serve`/`example` 경로의 `--dry-run` 명령 생성 결과가 쿡북 원문과 일치
- 인자 검증 및 에러 처리 (잘못된 action/component/model/backend/옵션)
- 미설치 상태에서 `serve` 시 안내 메시지 출력
- GPU 4장 미만에서 `--model super` 지정 시 경고

**확인하지 않은 것** — VRAM 제약으로 미실행:

- 실제 패키지 설치 (`uv pip install`, `docker pull`)
- 모델 가중치 다운로드 및 서버 기동
- 실제 추론 요청/응답

즉 **명령 조립과 환경 점검까지는 검증되었고, 모델 로드 이후는 미검증**입니다.
GPU 가 충분한 환경에서 처음 실행할 때는 `check` → `install --dry-run` → `install` 순서로 진행하세요.

---

## 9. 트러블슈팅

| 증상                                | 원인/해결                                                              |
| --------------------------------- | ------------------------------------------------------------------ |
| `uv 를 찾을 수 없습니다`                  | `curl -LsSf https://astral.sh/uv/install.sh \| sh` 후 셸 재시작         |
| `docker nvidia 런타임이 보이지 않습니다`     | nvidia-container-toolkit 설치 후 `sudo systemctl restart docker`      |
| 게이트 리포 401/403                    | `export HF_TOKEN=...` 및 HF 웹에서 `nvidia/Cosmos-1.0-Guardrail` 접근 승인 |
| CUDA/torch 버전 불일치                 | `--torch-backend cu128` 로 강제 지정                                    |
| OOM                               | `--model nano` 사용, Super 는 GPU 4장 필요                               |
| 포트 충돌                             | Reasoner/Generator 는 서로 다른 포트를 쓰도록 `--port` 지정                     |
| `file://` 미디어 접근 거부               | 파일을 `$COSMOS3_MEDIA_ROOT` 아래로 옮기거나 해당 환경변수를 조정                     |
| Generator venv 에서 audio/action 실패 | 해당 모달리티는 Docker 백엔드에서만 지원                                          |
