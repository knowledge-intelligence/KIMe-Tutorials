# Isaac Sim 5.1.0 — 공식 Docker 컨테이너 + X11 GUI / Headless 실행 가이드

NVIDIA 공식 **Isaac Sim 5.1.0 Docker 컨테이너**(`nvcr.io/nvidia/isaac-sim:5.1.0`)를 NGC에서 받아
실행하고, **X11 포워딩으로 GUI 창을 호스트 화면에 띄우거나**(또는 headless + WebRTC 라이브스트림으로)
사용하는 방법을 설명합니다. **ROS2는 사용하지 않습니다.**

- 실행 스크립트: [`run_sim51_docker_x11.sh`](./run_sim51_docker_x11.sh)
- 짝이 되는 Isaac Lab 2.3.2 문서/스크립트: [`ReadMe_Docker_IsaacLab232.md`](./ReadMe_Docker_IsaacLab232.md) / [`run_lab232_docker_prebuilt_x11.sh`](./run_lab232_docker_prebuilt_x11.sh)
- 이 이미지는 **Isaac Sim 앱 그 자체**입니다(Isaac Lab 미포함). Isaac Lab 튜토리얼/RL을 돌리려면 Isaac Lab 스크립트를 쓰세요.

공식 문서(함께 참고):
- Isaac Sim Container Installation: https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/install_container.html

---

## 0. 요약 (TL;DR)

```bash
# 1) (최초 1회) NGC 로그인 - 이미 되어 있으면 생략
docker login nvcr.io          # Username: $oauthtoken  /  Password: <NGC API Key>

# 2) 이미지 받고 대화형 셸로 진입 (GPU + X11 GUI)
./run_sim51_docker_x11.sh            # = ./run_sim51_docker_x11.sh shell

# 컨테이너 내부에서:
./isaac-sim.sh        # GUI (창이 호스트 :1 화면에 뜸)
./runheadless.sh -v   # headless + WebRTC 라이브스트림 (로컬 창 없음)
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
| 디스크 여유 | `/` 에 넉넉히 (이미지 약 15.1GB + 셰이더 캐시) |
| 호스트 X 디스플레이 | 로컬 Xorg 세션이 **`:1`** (소켓 `/tmp/.X11-unix/X1`) |
| NGC 로그인 | `docker login nvcr.io` 완료 상태 |

- NVIDIA Container Toolkit: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html
- NGC API Key 발급: https://ngc.nvidia.com/setup/api-key

### NGC 로그인
```bash
docker login nvcr.io
#  Username: $oauthtoken
#  Password: <NGC API Key>
```

---

## 2. 스크립트 사용법 (`run_sim51_docker_x11.sh`)

```
사용법: ./run_sim51_docker_x11.sh [command] [args]

  pull                 Isaac Sim 이미지를 NGC에서 받고 종료
  shell (또는 sim)     (기본) GPU + X11 GUI가 켜진 대화형 bash 셸로 진입
                       내부에서:
                         ./isaac-sim.sh       # GUI (호스트 화면에 창)
                         ./runheadless.sh     # headless + WebRTC 라이브스트림
  run <cmd...>         컨테이너 안에서 임의 명령 실행 후 종료
  stop                 실행 중인 isaac-sim 컨테이너 강제 제거
  clean-volumes        이 스크립트가 만든 named volume 제거(캐시 삭제 → 다음 실행 느려짐)

환경변수 오버라이드: SIM_IMAGE, SIM_CONTAINER, HOST_DISPLAY (기본 :1)
```

### 대표 예시
```bash
# 대화형 셸 진입 후 GUI 실행
./run_sim51_docker_x11.sh shell
#   컨테이너 내부:  ./isaac-sim.sh

# headless + 라이브스트림을 바로 실행
./run_sim51_docker_x11.sh run ./runheadless.sh

# 다른 디스플레이로 GUI 보내기
HOST_DISPLAY=:0 ./run_sim51_docker_x11.sh shell
```

---

## 3. 설치·실행 절차 (단계별 설명)

1. **이미지 Pull** — 없으면 자동으로 받습니다(약 15.1GB, 최초 1회).
2. **X11 권한 부여** — `xhost +local:` 로 컨테이너의 X 접근 허용.
3. **docker run 조립** — 스크립트가 자동 설정:
   - `--gpus all` + `NVIDIA_DRIVER_CAPABILITIES=all`
   - EULA/PRIVACY 비대화식 수락
   - **`-u 0:0` + `HOME=/isaac-sim`** (root로 실행 → named volume 쓰기 가능, 4장 참고)
   - `DISPLAY=:1` + `/tmp/.X11-unix` + `.Xauthority`
   - `/dev/dri` 렌더 노드 + `render` 그룹
   - **모든 캐시/로그/데이터를 named volume으로** 마운트
4. **실행 방식 선택**
   - **GUI**: 컨테이너 내부에서 `./isaac-sim.sh` → 호스트 화면(`:1`)에 창이 뜸
   - **Headless/라이브스트림**: `./runheadless.sh -v` → 로그에 "Streaming server started." 가 뜨면 준비 완료. **WebRTC Streaming Client**로 이 호스트 IP에 접속(클라이언트는 공식 문서에서 다운로드).
5. **첫 실행 셰이더 컴파일** — 처음엔 10~30분 이상 걸립니다. 이후 캐시로 빨라집니다.

---

## 4. 왜 named volume + root 실행인가 (중요)

공식 문서는 호스트 디렉터리를 **bind-mount**하고 `sudo chown -R 1234:1234 ~/docker/isaac-sim` 후 **`-u 1234:1234`**로 실행합니다. 그러나 **rootless Docker**에서는 이 소유권 방식이 취약합니다:
- 호스트 디렉터리의 소유권이 컨테이너 사용자 매핑과 어긋나면 컨테이너가 캐시에 쓰지 못해, 확장 레지스트리/셰이더 캐시 생성이 `PermissionError`로 실패하고 앱이 죽습니다(Isaac Lab에서 실제로 겪은 문제와 동일한 부류).

그래서 이 스크립트는:
- **Docker named volume** 사용 (도커가 쓰기 가능하게 생성/초기화 → chown 불필요)
- **root(`-u 0:0`) + `HOME=/isaac-sim`**로 실행 (Isaac Lab 이미지가 번들 Sim을 돌리는 방식과 동일). root 소유 named volume은 그대로 쓰기 가능하고, 공유 에셋 볼륨도 두 컨테이너 모두 root라 깔끔하게 공유됩니다.

사용되는 볼륨(`docker volume ls`):
```
isaac-sim-cache          # 메인 Kit/OV 캐시
isaac-sim-computecache   # CUDA 컴퓨트 캐시
isaac-sim-logs           # Omniverse/Kit 로그
isaac-sim-config         # Omniverse 설정
isaac-sim-data           # 확장 레지스트리 캐시
isaac-sim-pkg            # 다운로드 패키지
isaac-omni-assets        # ★ Isaac Lab 스크립트와 공유하는 에셋 캐시
```

---

## 5. Isaac Lab 2.3.2 컨테이너와의 관계 / 연결

- **Isaac Sim 5.1(이 이미지)** = Sim 앱 그 자체. Sim 워크플로우·스트리밍·확장 개발 등에 사용.
- **Isaac Lab 2.3.2** = Isaac Sim + Isaac Lab. RL/Isaac Lab 튜토리얼에 사용 → [`run_lab232_docker_prebuilt_x11.sh`](./run_lab232_docker_prebuilt_x11.sh).
- 두 스크립트는 **연결**되어 있습니다:
  - 둘 다 `--network=host` → 같은 호스트 네트워크 공유 (예: 이 Sim 컨테이너가 띄운 라이브스트림 서버에 다른 컨테이너/호스트가 접근 가능)
  - 둘 다 **`isaac-omni-assets`** 볼륨 마운트 → Omniverse 에셋 캐시 공유

> 참고: Isaac Lab만 필요하면 Isaac Sim 컨테이너는 굳이 띄우지 않아도 됩니다(Isaac Lab이 Sim을 자체 번들). 이 스크립트는 "독립 Isaac Sim 앱"이 필요할 때 사용합니다.

---

## 6. 트러블슈팅

| 증상 | 원인 / 해결 |
|------|-------------|
| `PermissionError ... /isaac-sim/.local/share/ov/...` | bind-mount + uid 1234 방식의 문제. 이 스크립트는 **named volume + root 실행**으로 회피. 옛 호스트 캐시(`~/docker/isaac-sim/*`)는 무시됨. |
| GUI 창이 안 뜸 / `cannot open display` | `HOST_DISPLAY`(기본 `:1`)가 로컬 세션과 일치하는지 확인. SSH `localhost:10.0`은 RTX 창 불가. 대안: `./runheadless.sh` + WebRTC 클라이언트. |
| 첫 실행이 10~30분+ 멈춘 듯함 | 정상. RTX/MDL 셰이더 첫 컴파일. named volume 캐시로 다음 실행은 빠름. |
| 라이브스트림 접속 안 됨 | `./runheadless.sh -v` 로그에 "Streaming server started." 확인 후, WebRTC Streaming Client로 **호스트 IP** 접속. `--network=host`라 포트는 호스트에 노출됨. |
| `vkCreateInstance failed` / GPU 미인식 | 호스트에서 `--gpus all`/드라이버/Container Toolkit 확인. `nvidia-smi`가 컨테이너에서 GPU를 보는지 점검. |

---

## 7. 정리 (Cleanup)

```bash
# 실행 중 컨테이너 제거
./run_sim51_docker_x11.sh stop

# Isaac Sim 캐시 볼륨 제거 (공유 isaac-omni-assets는 유지)
./run_sim51_docker_x11.sh clean-volumes

# 공유 에셋 캐시까지 완전 제거하려면 수동으로:
docker volume rm isaac-omni-assets
```

---

## 8. ROS2 관련

이 구성은 **ROS2를 설치하거나 활성화하지 않습니다.**
