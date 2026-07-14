#!/usr/bin/env bash
#
# docker_relocate.sh
# Ubuntu 24.04 - Docker 기존 이미지/캐시 삭제 + 저장 위치(data-root) 변경
#
# 사용법:
#   chmod +x docker_relocate.sh
#   sudo ./docker_relocate.sh /new/docker/path
#
# 예시:
#   sudo ./docker_relocate.sh /Temp/data/docker
#
set -euo pipefail

# ---------- 0. 사전 체크 ----------
if [[ $EUID -ne 0 ]]; then
  echo "root 권한으로 실행하세요: sudo $0 <new_path>"
  exit 1
fi

if [[ $# -lt 1 ]]; then
  echo "사용법: sudo $0 <새 저장 경로>"
  echo "예시:   sudo $0 /mnt/data/docker"
  exit 1
fi

NEW_ROOT="$1"
DAEMON_JSON="/etc/docker/daemon.json"
OLD_ROOT="$(docker info --format '{{.DockerRootDir}}' 2>/dev/null || echo /var/lib/docker)"

echo "===================================================="
echo " 현재 Docker Root Dir : ${OLD_ROOT}"
echo " 변경할 새 경로        : ${NEW_ROOT}"
echo "===================================================="
read -rp "계속 진행하시겠습니까? 기존 이미지/컨테이너/캐시가 모두 삭제됩니다. (y/N): " CONFIRM
if [[ "${CONFIRM}" != "y" && "${CONFIRM}" != "Y" ]]; then
  echo "취소되었습니다."
  exit 0
fi

# ---------- 1. 기존 컨테이너/이미지/캐시 정리 ----------
echo ">>> [1/5] 실행 중인 컨테이너 중지"
docker ps -q | xargs -r docker stop

echo ">>> [2/5] 컨테이너, 이미지, 네트워크, 볼륨, 빌드 캐시 전체 삭제"
docker system prune -a --volumes -f
docker builder prune -a -f

# ---------- 2. Docker 서비스 중지 ----------
echo ">>> [3/5] Docker 서비스 중지"
systemctl stop docker.socket docker.service || true

# ---------- 3. 새 경로 준비 및 daemon.json 설정 ----------
echo ">>> [4/5] 새 저장 경로 생성 및 daemon.json 설정"
mkdir -p "${NEW_ROOT}"

mkdir -p /etc/docker
if [[ -f "${DAEMON_JSON}" ]]; then
  cp "${DAEMON_JSON}" "${DAEMON_JSON}.bak.$(date +%s)"
  echo "    기존 daemon.json 백업 완료 -> ${DAEMON_JSON}.bak.*"
fi

# data-root 키만 갱신 (jq 있으면 병합, 없으면 새로 작성)
if command -v jq >/dev/null 2>&1 && [[ -f "${DAEMON_JSON}" ]]; then
  jq --arg root "${NEW_ROOT}" '. + {"data-root": $root}' "${DAEMON_JSON}" > "${DAEMON_JSON}.tmp"
  mv "${DAEMON_JSON}.tmp" "${DAEMON_JSON}"
else
  cat > "${DAEMON_JSON}" <<EOF
{
  "data-root": "${NEW_ROOT}"
}
EOF
fi

echo "    daemon.json 내용:"
cat "${DAEMON_JSON}"

# ---------- 4. 기존 데이터 이전 (prune 후 남은 게 있을 경우 대비) ----------
if [[ -d "${OLD_ROOT}" && "${OLD_ROOT}" != "${NEW_ROOT}" ]]; then
  echo ">>> 기존 데이터 이전 중 (${OLD_ROOT} -> ${NEW_ROOT})"
  rsync -aP "${OLD_ROOT}/" "${NEW_ROOT}/"
  mv "${OLD_ROOT}" "${OLD_ROOT}.old.$(date +%s)"
fi

# ---------- 5. Docker 재시작 및 확인 ----------
echo ">>> [5/5] Docker 서비스 재시작"
systemctl daemon-reload
systemctl start docker

sleep 2
echo "===================================================="
echo " 변경 결과 확인"
echo "===================================================="
docker info --format '  DockerRootDir: {{.DockerRootDir}}'
docker system df

echo ""
echo "완료되었습니다."
echo "이전 데이터 백업이 필요 없다면 아래 명령으로 삭제하세요:"
echo "  sudo rm -rf ${OLD_ROOT}.old.*"