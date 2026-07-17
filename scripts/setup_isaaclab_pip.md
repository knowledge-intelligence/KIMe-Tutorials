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
4. **파편 정리 + pip 제약 설정** — 중단된 설치 잔재(`~orch` 등) 삭제, `setuptools<81`과 torch 버전을 고정하는 제약 파일 생성
5. **Isaac Sim 설치** — `pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com`
6. **PyTorch 설치** — cu128 빌드를 Isaac Sim **이후에** 고정 설치
7. **Isaac Lab 의존성** — `./isaaclab.sh -i` 실행 후 모듈 5개가 실제로 설치됐는지 확인
8. **PyTorch cu128 재확인** — `-i`가 torch를 건드렸을 수 있으므로 다시 못박음
9. **활성화 스크립트 설정** — `setenv.sh` 생성/보정
10. **검증** — `isaacsim`/`isaaclab`/`torchaudio` import 확인, torch가 cu128 빌드이며 CUDA를 인식하는지 확인

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

첫 실행은 셰이더 컴파일 때문에 **3분 이상** 걸립니다. 이후에는 캐시되어 빨라집니다(캐시가 데워진 뒤에는 10초 내외).
`create_empty.py`는 `while simulation_app.is_running()` 무한 루프라 **스스로 종료하지 않습니다**. Ctrl+C로 끝내세요.

### 스크립트로 자동 확인할 때 (터미널에서 직접 볼 때는 해당 없음)

출력을 파일이나 파이프로 넘기면 두 가지 함정이 있습니다. 둘 다 **정상 동작을 실패로 오인하게** 만듭니다.

1. **stdout 블록 버퍼링** — 터미널이 아니면 python이 출력을 버퍼에 모읍니다. `[INFO]: Setup complete...`가
   이미 나왔는데도 로그에 안 보여서 멈춘 것처럼 보입니다. `PYTHONUNBUFFERED=1`을 주세요.
2. **`timeout`이 손자 프로세스를 못 죽임** — `timeout`은 자식인 `isaaclab.sh`만 종료시키고, 그 아래
   python은 **고아로 살아남아 GPU를 계속 점유**합니다. `setsid`로 프로세스 그룹을 묶어야 합니다.

```bash
PYTHONUNBUFFERED=1 setsid timeout -s KILL 420 \
    ./isaaclab.sh -p scripts/tutorials/00_sim/create_empty.py --headless > run.log 2>&1
grep -a "Setup complete" run.log && echo "정상"
```

"멈춘 것 같다" 싶으면 죽었다고 단정하기 전에 CPU 시간이 증가하는지 확인하세요. 늘고 있으면 작업 중입니다.

```bash
ps -o pid,etime,time,rss -C python | grep -a create_empty   # TIME이 증가하면 정상 동작 중
```

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

### `flatdict` 빌드 실패 (setuptools 81+)

`isaaclab` 코어가 요구하는 `flatdict==4.0.1`은 PyPI에 sdist만 있고, 그 `setup.py`가 `pkg_resources`를 import합니다.
setuptools는 **81부터 `pkg_resources`를 제거**했기 때문에, 빌드 격리 환경이 최신 setuptools를 받아오면 이렇게 깨집니다.

```
ModuleNotFoundError: No module named 'pkg_resources'
ERROR: Failed to build 'flatdict' when getting requirements to build wheel
```

→ `PIP_CONSTRAINT` 제약 파일에 `setuptools<81`을 넣습니다. 제약은 **빌드 격리 환경에만** 적용되므로
환경에 실제로 설치되는 setuptools 버전은 그대로입니다.

### `isaaclab.sh -i`가 실패를 삼킴 — 가장 위험한 함정

`-i`는 Isaac Lab 모듈을 하나씩 pip install하는데, **개별 모듈이 실패해도 멈추지 않고 계속 진행한 뒤 0을 반환**합니다.
위의 `flatdict` 실패로 `isaaclab` 코어가 통째로 빠졌는데도 `-i`는 나머지 모듈을 설치하고 정상 종료했습니다.
즉 **종료 코드만 믿으면 깨진 설치를 "성공"으로 오인**합니다.

→ `-i` 직후 `isaaclab`, `isaaclab_assets`, `isaaclab_mimic`, `isaaclab_rl`, `isaaclab_tasks` 5개가 실제로
설치됐는지 `pip show`로 확인하고, 하나라도 빠지면 중단합니다. `--skip-verify`와 무관하게 항상 검사합니다.

### PyTorch 설치 순서 — 그리고 `-i`의 torch 갈아엎기

`isaacsim` 설치가 torch를 덮어쓸 수 있어 **Isaac Sim 다음에** cu128 빌드를 고정 설치합니다.
`torchaudio`는 공식 문서에 없지만 `isaacsim-core`가 요구하므로 함께 설치합니다.

문제는 그 **다음에 오는 `isaaclab.sh -i`가 torch를 다시 헤집는다**는 점입니다.
`isaaclab_rl`이 `stable-baselines3>=2.6`을 요구 → pip이 최신 sb3(2.9.0)를 선택 → 이게 `torch>=2.8`을 요구 →
**pip이 torch를 2.13.0으로 업그레이드**하며 cu13 계열 NVIDIA 스택 수 GB를 통째로 받아옵니다.
이후 다른 모듈이 torch를 2.7.0+cu128로 되돌리지만, 그 왕복 과정에서 `torchaudio`가 조용히 사라집니다.

```
torchaudio 2.7.0+cu128 requires torch==2.7.0, but you have torch 2.13.0 which is incompatible.
isaacsim-core 5.1.0.0 requires torchaudio==2.7.0, which is not installed.
```

→ 제약 파일에 `torch`/`torchvision`/`torchaudio` 버전을 고정합니다. 그러면 resolver가 torch 2.7과 호환되는
sb3(2.8.0)를 고르므로 업그레이드 연쇄 자체가 생기지 않고, 불필요한 500MB+ 왕복 다운로드도 없어집니다.
`-i` 이후 cu128 빌드를 한 번 더 못박고, 검증에서 `torchaudio` import와 `+cu` 빌드 여부를 확인합니다.

### 중단된 설치가 남긴 파편

설치가 중간에 끊기면 site-packages에 `~orch`, `~orchgen`, `~unctorch` 같은 디렉터리가 남습니다
(pip이 uninstall 중 이름을 바꿔둔 잔재). 매 pip 호출마다 경고를 뿜고 import를 방해할 수 있습니다.

```
WARNING: Ignoring invalid distribution ~orch (.../site-packages)
```

→ 설치 시작 전에 site-packages의 `~*` 디렉터리를 정리합니다.

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

### `isaaclab.sh -i가 0을 반환했지만 다음 모듈이 설치되지 않았습니다`

`-i`가 개별 모듈 pip install에 실패하고도 계속 진행했다는 뜻입니다. 스크립트가 이를 잡아 중단한 것이므로,
**위쪽 pip 로그에서 해당 모듈의 실제 실패 원인**을 찾으세요. 로그가 길면 이렇게 거르면 됩니다.

```bash
./setup_isaaclab_pip.sh --force islab 2>&1 | tee install.log
grep -nE "ERROR:|error:|Traceback|Failed to build" install.log
```

### `torchaudio가 없습니다`

`isaacsim-core`가 `torchaudio==2.7.0`을 요구하는데 설치 과정의 torch 교체 와중에 사라진 경우입니다.
`--force`로 재실행하면 제약 파일과 cu128 재확인 단계가 복구합니다.

### `torch가 CUDA를 인식하지 못합니다`

- `nvidia-smi`로 드라이버 확인 (Isaac Sim 5.1은 드라이버 535 이상 필요)
- CPU 빌드가 설치된 경우: `pip list | grep torch`로 확인. `2.7.0+cu128`이 아니면 `--force`로 재실행
  (검증 단계가 `+cu` 접미사가 없으면 CPU 빌드로 판단해 실패시킵니다)

### `Ignoring invalid distribution ~orch` 경고가 계속 나옴

이전 설치가 중단되며 남은 파편입니다. `--force`로 재실행하면 스크립트가 정리합니다.
수동으로 지우려면: `rm -rf $CONDA_PREFIX/lib/python3.11/site-packages/~*`

### 시뮬레이션이 뜨지 않음

- **다른 Isaac Sim이 실행 중** — GPU 메모리 경합. `nvidia-smi`로 확인 후 정리
- **첫 실행이 느림** — 셰이더 컴파일로 3분 이상 걸리는 것이 정상

### 완전히 되돌리려면

```bash
./setup_isaaclab_pip.sh --uninstall <이름>
rm -rf ~/.cache/pip          # pip 캐시까지 정리 (약 40GB)
```
