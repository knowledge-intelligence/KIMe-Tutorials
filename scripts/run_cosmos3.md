# Cosmos3 실행 가이드 (Run)

설치가 끝난 뒤 **서버를 띄우고 추론 요청을 보내는 것**만 다루는 문서입니다.
설치 자체는 [`setup_cosmos3.md`](./setup_cosmos3.md) 를 참고하세요.

원문: <https://github.com/NVIDIA/cosmos>
([Generator with vLLM-Omni](https://github.com/NVIDIA/cosmos#generator-with-vllm-omni) /
[Reasoner with vLLM](https://github.com/NVIDIA/cosmos#reasoner-with-vllm))

---

## 0. 실행 전 환경변수

`/Temp` 를 도커/HF 캐시 저장소로 쓰는 구성입니다. 매 셸에서 먼저 로드하세요.

```bash
source /Temp/cosmos3.env
```

`/Temp/cosmos3.env` 내용:

```bash
export HF_HOME=/Temp/hf-cache
export HF_TOKEN=$(cat /Temp/hf-cache/token 2>/dev/null)
export COSMOS3_ROOT=/Temp/cosmos3
export COSMOS3_MEDIA_ROOT=/Temp/cosmos3/media
export COSMOS3_OUTPUT_DIR=/Temp/cosmos3/outputs
export COSMOS3_WORKDIR=/Temp/cosmos3
export UV_CACHE_DIR=/Temp/uv-cache
```

Generator 는 `cosmos-guardrail` 이 포함된 파생 이미지를 써야 합니다 (아래 3절 참고):

```bash
export COSMOS3_VLLM_OMNI_IMAGE=vllm-omni:cosmos3-guardrail
```

---

## 1. ⚠️ GPU 1장 = 동시 실행 불가

| 컴포넌트      | 검증 환경에서의 VRAM 점유 |
| --------- | ---------------- |
| Reasoner  | 약 95 GB          |
| Generator | 약 37 GB          |

RTX PRO 6000 (96GB) **1장** 기준으로 두 서버를 동시에 띄울 수 없습니다.
vLLM 은 기본적으로 `--gpu-memory-utilization 0.9` 로 VRAM 을 선점하므로,
**한쪽을 내리고 다른 쪽을 올리는 순차 방식**으로 사용하세요.

동시 실행이 꼭 필요하면 각각 `--gpu-memory-utilization` 을 낮춰야 하지만(예: Reasoner 0.5),
Generator 는 diffusion 파이프라인 특성상 여유가 없으면 실패하기 쉬우므로 권장하지 않습니다.

---

## 2. Reasoner (vLLM) — 포트 8000

### 2-1. 서버 실행

```bash
source /Temp/cosmos3.env
cd /Temp

# 포그라운드
./setup_cosmos3.sh serve reasoner --model nano --port 8000

# 백그라운드 (로그 파일)
nohup ./setup_cosmos3.sh serve reasoner --model nano --port 8000 > /Temp/reasoner.log 2>&1 &
tail -f /Temp/reasoner.log
```

스크립트가 실제로 실행하는 명령:

```bash
CUDA_VISIBLE_DEVICES=0 \
PATH="/Temp/cosmos3/venv-reasoner/bin:$PATH" \
HF_HOME=/Temp/hf-cache \
/Temp/cosmos3/venv-reasoner/bin/vllm serve nvidia/Cosmos3-Nano \
  --tensor-parallel-size 1 \
  --mm-encoder-tp-mode data \
  --async-scheduling \
  --allowed-local-media-path /Temp/cosmos3/media \
  --media-io-kwargs '{"video": {"num_frames": -1}}' \
  --port 8000
```

> `PATH` 에 venv 의 `bin` 이 반드시 들어가야 합니다. flashinfer 가 JIT 컴파일 시
> `ninja` 를 subprocess 로 호출하는데, 없으면 모델 로드가 끝난 직후
> `FileNotFoundError: 'ninja'` 로 엔진 초기화가 실패합니다.

### 2-2. 기동 대기 / 헬스체크

첫 실행은 가중치 다운로드 때문에 오래 걸립니다. 폴링으로 기다리세요.

```bash
until curl -sf http://localhost:8000/v1/models >/dev/null 2>&1; do sleep 15; done; echo READY

./setup_cosmos3.sh verify reasoner --port 8000
```

### 2-3. 텍스트 추론

```bash
curl -sS -X POST http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "nvidia/Cosmos3-Nano",
    "messages": [{"role": "user", "content": "Describe what a warehouse robot should check before lifting a pallet."}],
    "max_tokens": 256,
    "temperature": 0.2
  }'
```

또는 스크립트 내장 예제:

```bash
./setup_cosmos3.sh example reasoner --port 8000
```

### 2-4. 이미지 추론 (멀티모달)

로컬 파일은 `--allowed-local-media-path` 아래(기본 `/Temp/cosmos3/media`)에 있어야 하고
`file://` **절대경로**로 참조합니다.

```bash
curl -sS -X POST http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "nvidia/Cosmos3-Nano",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "file:///Temp/cosmos3/media/test.jpg"}},
        {"type": "text", "text": "Caption this image in detail."}
      ]}
    ],
    "max_tokens": 200
  }'
```

Python (`openai` 클라이언트):

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="not-used")
resp = client.chat.completions.create(
    model="nvidia/Cosmos3-Nano",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "file:///Temp/cosmos3/media/test.jpg"}},
            {"type": "text", "text": "Caption this image in detail."},
        ]},
    ],
    max_tokens=512,
)
print(resp.choices[0].message.content)
```

### 2-5. 종료

```bash
pkill -f "venv[-]reasoner/bin/vllm"
```

> `venv[-]reasoner` 처럼 대괄호를 넣는 이유: 그냥 `venv-reasoner` 로 하면
> `pkill` 이 **자기 자신의 명령줄까지 매칭**해서 셸이 같이 죽습니다.

---

## 3. Generator (vLLM-Omni, Docker) — 포트 8001

### 3-1. guardrail 이미지 준비 (최초 1회, 필수)

공식 이미지 `vllm/vllm-omni:cosmos3` 에는 `cosmos-guardrail` 이 **빠져 있어** 기동이 거부됩니다:

```
RuntimeError: Orchestrator initialization failed: You have disabled the safety checker
for CosmosSafetyChecker. This is in violation of the NVIDIA Open Model License Agreement.
Please install cosmos-guardrail package to enable safety checks.
```

패키지를 추가한 파생 이미지를 만듭니다 (torch/vllm 은 이미 충족되어 재설치되지 않음):

```bash
mkdir -p /Temp/build-guardrail && cd /Temp/build-guardrail
cat > Dockerfile <<'EOF'
FROM vllm/vllm-omni:cosmos3
RUN pip install --no-cache-dir cosmos-guardrail==0.3.1
EOF
docker build -t vllm-omni:cosmos3-guardrail .
```

### 3-2. 서버 실행

```bash
source /Temp/cosmos3.env
export COSMOS3_VLLM_OMNI_IMAGE=vllm-omni:cosmos3-guardrail
cd /Temp

nohup ./setup_cosmos3.sh serve generator --model nano --port 8001 > /Temp/generator.log 2>&1 &
tail -f /Temp/generator.log
```

스크립트가 실제로 실행하는 명령:

```bash
docker run --rm -i --runtime nvidia --gpus device=0 \
  -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
  -e HF_TOKEN="$HF_TOKEN" \
  -v /Temp/hf-cache:/root/.cache/huggingface \
  -v /Temp/cosmos3:/workspace \
  -p 8001:8000 --ipc=host \
  vllm-omni:cosmos3-guardrail \
  vllm serve nvidia/Cosmos3-Nano \
    --omni \
    --model-class-name Cosmos3OmniDiffusersPipeline \
    --allowed-local-media-path / \
    --port 8000 \
    --init-timeout 1800
```

> `-t` 는 TTY 가 있을 때만 붙습니다. `nohup`/CI 처럼 TTY 가 없는 환경에서 `-it` 를 쓰면
> `the input device is not a TTY` 로 실패합니다. 스크립트가 `[[ -t 0 ]]` 로 자동 판별합니다.

### 3-3. 기동 대기 / 헬스체크

```bash
until curl -sf http://localhost:8001/v1/models >/dev/null 2>&1; do sleep 20; done; echo READY

./setup_cosmos3.sh verify generator --port 8001
```

### 3-4. text-to-video 생성

**엔드포인트는 `/v1/videos/sync` 이고 multipart/form-data 로 보내며 mp4 바이너리를 그대로 반환합니다.**

```bash
curl -sS -X POST http://localhost:8001/v1/videos/sync \
  --form-string "prompt=A small warehouse robot moves a blue box across a clean floor." \
  --form-string "negative_prompt=blurry, distorted, low quality" \
  --form-string "size=1280x720" \
  --form-string "num_frames=189" \
  --form-string "fps=24" \
  --form-string "num_inference_steps=35" \
  --form-string "guidance_scale=6.0" \
  --form-string "flow_shift=10.0" \
  --form-string "seed=0" \
  --form-string 'extra_params={"use_resolution_template":false,"use_duration_template":false,"guardrails":true}' \
  -o cosmos3_t2v_output.mp4
```

스크립트 내장 예제 (위 curl 과 동일한 요청):

```bash
# README 원본 파라미터 (189프레임/35스텝) — 약 6분
./setup_cosmos3.sh example generator --port 8001

# 빠른 확인 — 약 20초
./setup_cosmos3.sh example generator --port 8001 --num-frames 29 --steps 20
```

결과는 `$COSMOS3_OUTPUT_DIR/cosmos3_generator_t2v.mp4` 에 저장되고, `ffprobe` 가 있으면
코덱/해상도/프레임 수까지 자동 출력합니다.

독립 실행 스크립트도 있습니다:

```bash
/Temp/cosmos3-experiments/inputs/generator_t2v_request.sh 8001 out.mp4
```

#### 주요 파라미터

| 파라미터                  | 예시            | 설명                                    |
| --------------------- | ------------- | ------------------------------------- |
| `prompt`              | —             | 생성 프롬프트                               |
| `negative_prompt`     | —             | 배제할 특성                                |
| `size`                | `1280x720`    | 해상도                                   |
| `num_frames`          | `189`         | 프레임 수 — 생성 시간에 가장 크게 영향               |
| `fps`                 | `24`          | 출력 fps (189프레임 @ 24fps ≈ 7.9초)        |
| `num_inference_steps` | `35`          | 디노이징 스텝                               |
| `guidance_scale`      | `6.0`         | 프롬프트 반영 강도                            |
| `flow_shift`          | `10.0`        | flow matching shift                   |
| `seed`                | `0`           | 재현성                                   |
| `extra_params`        | JSON 문자열      | `guardrails` 는 `true` 유지 (라이선스 준수)    |

빠른 확인용으로는 `num_frames=29`, `num_inference_steps=20` 으로 줄이면 20초 정도에 끝납니다.

### 3-5. 결과 검증

```bash
ffprobe -v error \
  -show_entries stream=codec_name,width,height,nb_frames,r_frame_rate \
  -show_entries format=duration \
  -of default=noprint_wrappers=1 cosmos3_t2v_output.mp4

# 프레임 추출해서 눈으로 확인
ffmpeg -y -i cosmos3_t2v_output.mp4 -vf "select=eq(n\,120)" -vframes 1 frame.png
```

### 3-6. 종료

```bash
docker ps -q --filter ancestor=vllm-omni:cosmos3-guardrail | xargs -r docker stop
```

---

## 4. 순차 전환 (Reasoner ↔ Generator)

GPU 1장이므로 전환 시 **VRAM 이 비었는지 반드시 확인**하고 다음 서버를 올리세요.

```bash
# Reasoner -> Generator
pkill -f "venv[-]reasoner/bin/vllm"
sleep 8
nvidia-smi --query-gpu=memory.used --format=csv,noheader   # 유휴값(~2.8GB)까지 떨어졌는지 확인
export COSMOS3_VLLM_OMNI_IMAGE=vllm-omni:cosmos3-guardrail
nohup ./setup_cosmos3.sh serve generator --model nano --port 8001 > /Temp/generator.log 2>&1 &

# Generator -> Reasoner
docker ps -q --filter ancestor=vllm-omni:cosmos3-guardrail | xargs -r docker stop
sleep 10
nvidia-smi --query-gpu=memory.used --format=csv,noheader
nohup ./setup_cosmos3.sh serve reasoner --model nano --port 8000 > /Temp/reasoner.log 2>&1 &
```

---

## 5. 알려진 차이점 / 주의사항

| 항목                                   | 내용                                                                                                       |
| ------------------------------------ | -------------------------------------------------------------------------------------------------------- |
| Generator API 는 `/v1/videos/sync`     | 초기 스크립트는 `/v1/infer` + `.b64_video` JSON 을 가정했으나 실제 API 는 multipart + mp4 바이너리입니다. **`setup_cosmos3.sh` 에 반영 완료** — `example generator` 로 바로 생성됩니다. |
| Generator 에 `cosmos-guardrail` 필수    | 공식 이미지에 미포함 — 3-1 절로 파생 이미지를 만들어야 기동됩니다.                                                                 |
| Reasoner 는 PATH 에 venv/bin 필요        | 없으면 `ninja` 를 못 찾아 엔진 초기화 실패.                                                                             |
| 백그라운드 docker 는 `-t` 금지               | TTY 없으면 `-it` 가 실패.                                                                                       |
| 동시 실행 불가 (GPU 1장)                    | 1절 참고.                                                                                                    |
| 첫 기동은 수십 분                           | 가중치 다운로드 (`/Temp/hf-cache`, 약 30GB+).                                                                     |

---

## 6. 로그 / 산출물 위치

| 대상             | 경로                                    |
| -------------- | ------------------------------------- |
| Reasoner 로그    | `/Temp/reasoner.log`                  |
| Generator 로그   | `/Temp/generator.log`                 |
| 생성 결과 기본 경로    | `/Temp/cosmos3/outputs/`              |
| 입력 미디어 루트      | `/Temp/cosmos3/media/`                |
| HF 캐시          | `/Temp/hf-cache/`                     |
| Docker 이미지 실체  | `/Temp/containerd/`                   |
| 실험 기록          | `/Temp/cosmos3-experiments/`          |
