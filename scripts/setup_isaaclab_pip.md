# setup_isaaclab_pip.sh 사용 설명서

Isaac Lab 2.3.2 + Isaac Sim 5.1을 **pip 방식**으로 자동 설치하는 스크립트입니다.
저장소 clone부터 conda 환경 생성, 패키지 설치까지 한 번에 수행하고, 삭제도 지원합니다.

기반 문서: [Isaac Lab 2.3.2 Pip Installation](https://isaac-sim.github.io/IsaacLab/v2.3.2/source/setup/installation/pip_installation.html)

```bash
curl -LsSf https://raw.githubusercontent.com/knowledge-intelligence/KIMe-Tutorials/main/scripts/setup_isaaclab_pip.sh | bash -s isaaclab
```

---

## 1. 사용법

```bash
./setup_isaaclab_pip.sh [옵션] <이름>
```

**이름 하나만** 지정하면 됩니다. 폴더와 conda 환경 이름이 여기서 자동으로 정해집니다.

| 입력한 이름 | 만들어지는 폴더 | 만들어지는 conda 환경 |
|---|---|---|
| `islab` | `./islab` | `env_islab` |
| `~/work/mylab` | `~/work/mylab` | `env_mylab` |

경로를 주면 폴더는 그 경로에, 환경 이름은 마지막 이름(basename)에 `env_`를 붙여 만듭니다.

### 기본 예시

```bash
cd ~
./setup_isaaclab_pip.sh islab      # ~/islab 폴더 + env_islab 환경

conda activate env_islab
cd ~/islab
```

### 옵션

| 옵션 | 설명 |
|---|---|
| `--uninstall` | 설치하지 않고 `<이름>` 폴더와 `env_<이름>` 환경을 삭제 |
| `--verify-only` | 설치를 건너뛰고 import/CUDA 검증만 수행 |
| `--skip-verify` | 설치만 하고 검증은 건너뜀 |
| `--force` | 폴더/환경이 이미 있으면 재사용 (기본은 중단) |
| `-y`, `--yes` | 삭제 시 확인 프롬프트를 건너뜀 |
| `-h`, `--help` | 도움말 출력 |

```bash
# 이미 설치된 환경이 정상인지만 확인 (몇 초)
./setup_isaaclab_pip.sh --verify-only islab

# 중간에 실패한 설치를 이어서 재시도
./setup_isaaclab_pip.sh --force islab
```

### 삭제

```bash
./setup_isaaclab_pip.sh --uninstall islab
```

삭제 대상(폴더 경로, 용량, conda 환경)을 먼저 보여주고 확인을 받습니다. 자동화 시에는 `-y`를 붙이세요.

안전장치가 걸려 있습니다:

- 폴더에 `isaaclab.sh`가 없으면 **Isaac Lab 저장소가 아니라고 판단해 중단**합니다 (엉뚱한 폴더 삭제 방지)
- 커밋되지 않은 변경이 있으면 경고합니다
- 삭제하려는 환경이 **현재 활성화되어 있으면 중단**합니다 (`conda deactivate` 후 재실행)

pip 캐시(약 40GB)는 남습니다. 함께 지우려면 `rm -rf ~/.cache/pip`.

> **주의:** 이름 규칙이 `env_<폴더명>`으로 고정이라, 이 규칙에 맞지 않는 기존 설치는 이 스크립트로 관리할 수 없습니다.
> 예를 들어 폴더 `IsaacLab2.3` + 환경 `env_isaaclab` 조합은 `env_IsaacLab2.3`를 찾으므로 실패합니다.

---

## 2. 설치되는 구성

| 항목 | 버전 |
|---|---|
| Isaac Lab | v2.3.2 (git tag) |
| Isaac Sim | 5.1.0 (`isaacsim[all,extscache]`) |
| Python | 3.11 (Isaac Sim 5.x 요구사항) |
| PyTorch | 2.7.0 + cu128 |
| torchvision / torchaudio | 0.22.0 / 2.7.0 |

버전을 바꾸려면 스크립트 상단의 변수를 수정하세요. 환경 이름 prefix도 `ENV_PREFIX` 변수로 바꿀 수 있습니다.

---

## 3. 실행 단계

1. **사전 점검** — conda, git, GPU, 디스크 여유 공간(60GB 권장) 확인
2. **clone** — `https://github.com/isaac-sim/IsaacLab.git` 의 `v2.3.2` 태그를 `--depth 1`로 clone
3. **conda 환경 생성** — Python 3.11로 생성 후 활성화
4. **Isaac Sim 설치** — `pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com`
5. **PyTorch 설치** — cu128 빌드를 Isaac Sim **이후에** 고정 설치
6. **Isaac Lab 의존성** — `./isaaclab.sh -i`
7. **활성화 스크립트 설정** — `setenv.sh` 생성/보정
8. **검증** — `isaacsim`/`isaaclab` import 확인, torch CUDA 인식 확인

소요 시간은 네트워크에 따라 **30분~1시간**입니다 (Isaac Sim 다운로드가 수십 GB).
디스크는 conda 환경 약 20GB + pip 캐시 약 40GB를 사용합니다.

검증은 import와 CUDA 인식까지만 확인합니다. 실제 시뮬레이션 동작은 아래 명령으로 직접 확인하세요.

---

## 4. 설치 후 동작 확인 및 데모

```bash
conda activate env_islab
cd ~/islab

# 빈 씬 생성 — "[INFO]: Setup complete..."가 뜨면 정상 (Ctrl+C로 종료)
./isaaclab.sh -p scripts/tutorials/00_sim/create_empty.py --headless

# 데모 (로컬 디스플레이가 있는 경우)
./isaaclab.sh -p scripts/demos/h1_locomotion.py

# 데모 (원격 SSH 접속인 경우)
./isaaclab.sh -p scripts/demos/h1_locomotion.py --livestream 2
```

첫 실행은 셰이더 컴파일 때문에 **3분 이상** 걸립니다. 이후에는 캐시되어 빨라집니다.
`create_empty.py`는 `while simulation_app.is_running()` 무한 루프라 **스스로 종료하지 않습니다**. Ctrl+C로 끝내세요.

### 원격 접속 시 주의

SSH의 X11 포워딩으로는 Isaac Sim GUI가 **뜨지 않습니다**. RTX/Vulkan 로컬 렌더링이 필요하기 때문입니다.
원격에서는 `--livestream 2`를 쓰고, [Isaac Sim WebRTC Streaming Client](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/download.html)로 접속하세요.

- 접속 주소: `<서버IP>:49100`
- Isaac Lab 2.3.2는 `omni.services.livestream.nvcf` 익스텐션을 쓰며 포트는 **49100**입니다.
  (구형 Isaac Sim 4.x의 8211 포트가 아닙니다.)

라이브스트림 모드에서 아래 경고가 반복되는 것은 **정상**입니다 — 로컬 창을 열지 않으므로 GLFW 초기화가 실패하는 게 맞습니다.

```
[Warning] [carb.windowing-glfw.plugin] GLFW initialization failed.
[Warning] [omni.platforminfo.plugin] failed to open the default display.
```

### H1 데모 조작법

로봇을 마우스로 클릭해 선택한 뒤:

| 키 | 동작 |
|---|---|
| ↑ | 전진 |
| ← / → | 좌회전 / 우회전 |
| ↓ | 정지 |
| C | 3인칭 ↔ 원근 시점 전환 |
| ESC | 3인칭 시점 해제 |

---

## 5. 스크립트가 처리하는 함정들

공식 문서대로만 하면 걸리는 문제들을 스크립트가 자동으로 처리합니다.

### EULA 프롬프트로 인한 설치 실패

Isaac Sim 첫 실행 시 NVIDIA 라이선스 동의를 `input()`으로 묻는데, 비대화형 셸에서는 다음과 같이 실패합니다.

```
Unable to bootstrap inner kit kernel: EOF when reading a line
```

→ `OMNI_KIT_ACCEPT_EULA=YES`를 설정하고 `setenv.sh`에도 남깁니다.

### `_isaac_sim` 참조로 인한 activate 에러

`isaaclab.sh`는 환경 생성 시점에 `_isaac_sim/` 폴더가 있으면 이를 무조건 `source`하는 줄을 `setenv.sh`에 넣습니다.
바이너리 설치를 지우고 pip으로 전환하면 `conda activate` 때마다 아래 에러가 납니다.

```
setenv.sh: line 11: .../_isaac_sim/setup_conda_env.sh: No such file or directory
```

→ 해당 줄을 조건부(`if [ -f ... ]`)로 바꾸고 원본은 `.bak`으로 백업합니다.

### `setenv.sh` 미생성

`setenv.sh`는 `isaaclab.sh --conda`(환경 생성 명령)가 만드는 파일이고 `-i`(설치 명령)는 만들지 않습니다.
`conda create`로 만든 환경에는 없으므로 스크립트가 직접 생성합니다 (`ISAACLAB_PATH`, `isaaclab` alias, `RESOURCE_NAME`).

### PyTorch 설치 순서

`isaacsim` 설치가 torch를 덮어쓸 수 있어 **Isaac Sim 다음에** cu128 빌드를 고정 설치합니다.
`torchaudio`는 공식 문서에 없지만 `isaacsim-core`가 요구하므로 함께 설치합니다.

### 의존성 충돌 메시지

설치 중 아래 메시지가 나오지만 무해합니다. `isaacsim`이 `starlette` 등을 다운그레이드하면 이어지는 `isaaclab.sh -i`가 Isaac Lab 요구 버전으로 되돌립니다.

```
ERROR: pip's dependency resolver does not currently take into account all the packages...
```

---

## 6. 문제 해결

### `conda 환경 '...'가 이미 존재합니다`

의도한 재사용이면 `--force`를, 새로 만들려면 `--uninstall` 후 재실행하세요.

### `isaacsim을 import할 수 없습니다`

설치가 중간에 끊겼을 가능성이 큽니다. `--force`로 재실행하면 이미 받은 패키지는 건너뛰고 이어서 진행합니다.

### `torch가 CUDA를 인식하지 못합니다`

- `nvidia-smi`로 드라이버 확인 (Isaac Sim 5.1은 드라이버 535 이상 필요)
- CPU 빌드가 설치된 경우: `pip list | grep torch`로 확인. `2.7.0+cu128`이 아니면 `--force`로 재실행

### 시뮬레이션이 뜨지 않음

- **다른 Isaac Sim이 실행 중** — GPU 메모리 경합. `nvidia-smi`로 확인 후 정리
- **첫 실행이 느림** — 셰이더 컴파일로 3분 이상 걸리는 것이 정상

### 완전히 되돌리려면

```bash
./setup_isaaclab_pip.sh --uninstall <이름>
rm -rf ~/.cache/pip          # pip 캐시까지 정리 (약 40GB)
```
