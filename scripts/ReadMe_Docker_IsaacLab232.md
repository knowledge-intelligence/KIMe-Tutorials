# Isaac Lab 2.3.2 — 공식 Pre-Built Docker 컨테이너 + X11 GUI 실행 가이드

NVIDIA 공식 **Pre-Built Isaac Lab v2.3.2 Docker 컨테이너**(`nvcr.io/nvidia/isaac-lab:2.3.2`)를
직접 빌드하지 않고 NGC에서 받아 실행하고, **X11 포워딩으로 Isaac Sim GUI 창을 호스트 화면에 띄우는**
방법을 설명합니다. **ROS2는 사용하지 않습니다.**

- 실행 스크립트: [`run_lab232_docker_prebuilt_x11.sh`](./run_lab232_docker_prebuilt_x11.sh)
- 짝이 되는 Isaac Sim 5.1 문서/스크립트: [`ReadMe_Docker_IsaacSim51.md`](./ReadMe_Docker_IsaacSim51.md) / [`run_sim51_docker_x11.sh`](./run_sim51_docker_x11.sh)
- 이 이미지는 **Isaac Sim + Isaac Lab이 한 이미지에 모두 들어있어 자체 완결(self-contained)** 입니다. Isaac Lab을 돌리는 데 별도 Isaac Sim 설치가 필요 없습니다.

공식 문서(함께 참고):
- Docker Guide: https://isaac-sim.github.io/IsaacLab/v2.3.2/source/deployment/docker.html
- Running an example with Docker: https://isaac-sim.github.io/IsaacLab/v2.3.2/source/deployment/run_docker_example.html

---

## 0. 요약 (TL;DR)

```bash
# 1) (최초 1회) NGC 로그인 - 이미 되어 있으면 생략
docker login nvcr.io          # Username: $oauthtoken  /  Password: <NGC API Key>

# 2) 이미지 받고 대화형 셸로 진입 (GPU + X11 GUI)
./run_lab232_docker_prebuilt_x11.sh            # = ./run_lab232_docker_prebuilt_x11.sh shell

# 컨테이너 내부에서:
./isaaclab.sh -p scripts/tutorials/00_sim/create_empty.py         # GUI 빈 씬
./isaaclab.sh -p scripts/tutorials/00_sim/log_time.py --headless  # 공식 headless 예제

# 또는 스크립트로 예제 한 번에:
./run_lab232_docker_prebuilt_x11.sh example logtime     # headless
./run_lab232_docker_prebuilt_x11.sh example quadruped   # GUI 데모
```

> **첫 실행은 RTX/MDL 셰이더 컴파일 때문에 10~30분 이상** 걸릴 수 있습니다(멈춘 것 아님, CPU 집약적). 캐시를 named volume에 저장하므로 두 번째 실행부터는 훨씬 빠릅니다.

---

## 1. 사전 준비물 (Prerequisites)

| 항목 | 이 머신의 확인된 상태 |
|------|----------------------|
| Docker Engine | rootless Docker 29.5.3, `sudo` 없이 동작 (overlay2) |
| NVIDIA 드라이버 | 580.159.04 (CUDA 13) |
| GPU | NVIDIA RTX A6000 48GB |
| NVIDIA Container Toolkit | `--gpus all` 로 컨테이너에서 GPU/Vulkan 정상 인식 |
| 디스크 여유 | `/` 에 넉넉히 (이미지 약 17.6GB + 셰이더 캐시) |
| 호스트 X 디스플레이 | 로컬 Xorg 세션이 **`:1`** (소켓 `/tmp/.X11-unix/X1`) |
| NGC 로그인 | `docker login nvcr.io` 완료 상태 |

없다면:
- Docker 설치: 저장소의 [`install_docker.sh`](./install_docker.sh)
- NVIDIA Container Toolkit: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html
- NGC API Key 발급: https://ngc.nvidia.com/setup/api-key

### NGC 로그인
Pre-Built 이미지는 `nvcr.io`(NGC)에 있어 인증이 필요할 수 있습니다.
```bash
docker login nvcr.io
#  Username: $oauthtoken
#  Password: <NGC API Key>
```

---

## 2. 스크립트 사용법 (`run_lab232_docker_prebuilt_x11.sh`)

```
사용법: ./run_lab232_docker_prebuilt_x11.sh [command] [args]

  pull                 Isaac Lab 이미지를 NGC에서 받고 종료
  shell                (기본) GPU + X11 GUI가 켜진 대화형 bash 셸로 진입
  example <key>        내장 예제 하나 실행 후 종료. key:
                         empty     -> tutorials/00_sim/create_empty.py   (headless 스모크 테스트)
                         logtime   -> tutorials/00_sim/log_time.py       (headless, 공식 docker 예제)
                         spawn     -> tutorials/00_sim/spawn_prims.py    (GUI 창)
                         quadruped -> demos/quadrupeds.py                (GUI 창)
                         train     -> RL 학습, Ant, 50 iterations        (headless)
  run <cmd...>         컨테이너 안에서 임의 명령 실행 후 종료
  stop                 실행 중인 isaac-lab 컨테이너 강제 제거
  clean-volumes        이 스크립트가 만든 named volume 제거(캐시 삭제 → 다음 실행 느려짐)

환경변수 오버라이드: LAB_IMAGE, LAB_CONTAINER, HOST_DISPLAY (기본 :1)
```

### 대표 예시
```bash
# 대화형 셸 (내부에서 원하는 스크립트 실행)
./run_lab232_docker_prebuilt_x11.sh shell

# 공식 headless 예제 바로 실행
./run_lab232_docker_prebuilt_x11.sh example logtime

# GUI 데모(사족보행 로봇) - X11 창이 호스트 :1 화면에 뜸
./run_lab232_docker_prebuilt_x11.sh example quadruped

# 임의 명령
./run_lab232_docker_prebuilt_x11.sh run ./isaaclab.sh -p scripts/tutorials/00_sim/create_empty.py --headless

# 다른 디스플레이로 GUI 보내기
HOST_DISPLAY=:0 ./run_lab232_docker_prebuilt_x11.sh example quadruped
```

---

## 3. 설치·실행 절차 (단계별 설명)

1. **이미지 Pull** — `pull` 또는 다른 명령 실행 시 자동으로 없으면 받습니다(약 17.6GB, 최초 1회).
2. **X11 권한 부여** — 스크립트가 `xhost +local:` 로 컨테이너(로컬 클라이언트)의 X 접근을 허용합니다.
3. **docker run 조립** — 스크립트가 아래를 자동 설정합니다:
   - `--gpus all` + `NVIDIA_DRIVER_CAPABILITIES=all` (RTX 렌더러용 그래픽 능력 포함)
   - EULA/PRIVACY 비대화식 수락
   - `DISPLAY=:1` + `/tmp/.X11-unix` 소켓 마운트 + `.Xauthority`
   - `/dev/dri` 렌더 노드 + `render` 그룹(EGL/Vulkan용)
   - **모든 캐시/로그/데이터를 named volume으로** 마운트 (4장 참고)
4. **컨테이너 진입/실행** — `shell`이면 bash로 진입, `example`/`run`이면 명령 실행 후 종료.
5. **첫 실행 셰이더 컴파일** — 처음엔 RTX/MDL 셰이더를 컴파일하느라 10~30분 이상 걸립니다. 이후 캐시(named volume)로 빨라집니다.

---

## 4. 왜 named volume인가 (중요 — 과거 PermissionError 원인/해결)

이 스크립트는 캐시/데이터를 **호스트 bind-mount가 아니라 Docker named volume**으로 마운트합니다. 이유:

- 이 머신은 **rootless Docker**라 컨테이너의 `root`(uid 0)가 **호스트의 진짜 root가 아니라 호스트 사용자**로 매핑됩니다.
- 만약 호스트 디렉터리를 bind-mount 했는데 그 디렉터리가 (이전 실행 잔재로) **`root:root` 소유**라면, 컨테이너 프로세스는 그 안에 쓸 수 없습니다. 그러면 Isaac Lab이 확장 레지스트리 캐시를 만들다가 실패합니다:
  ```
  PermissionError: [Errno 13] Permission denied: '/root/.local/share/ov/data/exts'
    -> Syncing with extension registry unavailable
    -> dependency: 'isaacsim.asset.importer.urdf' ='2.4.31' can't be satisfied (available 2.4.30)
    -> ModuleNotFoundError: No module named 'omni.kit.usd'  -> 앱 종료
  ```
- **named volume**은 도커가 컨테이너 사용자가 쓸 수 있도록 생성/초기화하므로 이 문제가 원천적으로 발생하지 않습니다. (공식 Isaac Lab docker-compose도 named volume 사용)

사용되는 볼륨(`docker volume ls`):
```
isaac-lab-kit-cache      # RTX/MDL 셰이더 + kit 캐시 (첫 컴파일 결과 저장)
isaac-lab-pip-cache      # pip wheel
isaac-lab-gl-cache       # GL 셰이더 캐시
isaac-lab-compute-cache  # CUDA 컴퓨트 캐시
isaac-lab-logs           # Omniverse/Kit 로그
isaac-lab-data           # 확장 레지스트리 캐시(과거 PermissionError가 나던 경로)
isaac-lab-docs           # 사용자 문서/출력
isaac-omni-assets        # ★ Isaac Sim 스크립트와 공유하는 에셋 캐시
```

---

## 5. Isaac Sim 5.1 컨테이너와의 관계 / 연결

- **Isaac Lab 컨테이너는 Isaac Sim을 자체 번들**하므로, Isaac Lab만 쓸 거라면 이 스크립트 하나로 충분합니다.
- 독립 **Isaac Sim 5.1** 앱(스트리밍/Sim 워크플로우 등)이 필요하면 [`run_sim51_docker_x11.sh`](./run_sim51_docker_x11.sh)를 함께 사용합니다.
- 두 스크립트는 **연결**되어 있습니다:
  - 둘 다 `--network=host` → 같은 호스트 네트워크(localhost) 공유 (예: Isaac Sim 라이브스트림 서버에 상호 접근 가능)
  - 둘 다 **`isaac-omni-assets`** 볼륨을 마운트 → **Omniverse 에셋 캐시를 공유**(다운로드한 에셋 재사용)

---

## 6. 트러블슈팅

| 증상 | 원인 / 해결 |
|------|-------------|
| `PermissionError ... /root/.local/share/ov/data/exts` | 과거 bind-mount 방식의 문제. 현재 스크립트는 **named volume**을 써서 해결됨. 예전 호스트 캐시(`~/docker/isaac-sim/*`)가 남아 있으면 무시(이 스크립트는 사용 안 함). |
| GUI 창이 안 뜸 / `cannot open display` | `HOST_DISPLAY`가 로컬 세션과 맞는지 확인(`echo $DISPLAY`가 `:1`인 셸에서 실행 권장). SSH의 `localhost:10.0`은 RTX 창을 못 띄움. `xhost +local:` 적용됐는지 확인. |
| 첫 실행이 10~30분+ 멈춘 듯함 | 정상. RTX/MDL 셰이더 **첫 컴파일**(CPU 집약적). named volume 캐시 덕에 다음 실행은 빠름. |
| `vkCreateInstance failed` / `libGL.so.1` | Pre-Built 컨테이너엔 그래픽 런타임이 포함돼 정상. 호스트에서 `--gpus all`/드라이버/Container Toolkit 확인. |
| 렌더 노드 접근 문제 | 스크립트가 `--device /dev/dri` + `--group-add <render gid>`를 자동 추가. |

---

## 7. 정리 (Cleanup)

```bash
# 실행 중 컨테이너 제거
./run_lab232_docker_prebuilt_x11.sh stop

# Isaac Lab 캐시 볼륨 제거 (공유 isaac-omni-assets는 유지 → 다음 실행 시 셰이더 재컴파일)
./run_lab232_docker_prebuilt_x11.sh clean-volumes

# 공유 에셋 캐시까지 완전 제거하려면 수동으로:
docker volume rm isaac-omni-assets
```

---

## 8. ROS2 관련

이 구성은 **ROS2를 설치하거나 활성화하지 않습니다.** ROS2 브리지가 필요하면 별도 설정이 필요하며, 본 가이드 범위 밖입니다.
