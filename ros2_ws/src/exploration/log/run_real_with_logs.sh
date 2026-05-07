#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE_DIR="$(cd "${ROOT_DIR}/../.." && pwd)"
LOG_ROOT="${EXPLORATION_LOG_ROOT:-${ROOT_DIR}/log}"
LOG_DIR="${LOG_ROOT}/$(date +%Y%m%d_%H%M%S)"
mkdir -p "${LOG_DIR}"

source /opt/ros/humble/setup.bash
source "${WORKSPACE_DIR}/install/setup.bash"

cleanup() {
  jobs -p | xargs -r kill >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

echo "logs: ${LOG_DIR}"

stdbuf -oL -eL ros2 launch exploration exploration_real_robot.launch.py rviz:=false "log_root:=${LOG_ROOT}" "$@" \
  2>&1 | tee "${LOG_DIR}/launch.log" &

sleep 5

stdbuf -oL -eL ros2 topic echo /cmd_vel > "${LOG_DIR}/cmd_vel.log" 2>&1 &
stdbuf -oL -eL ros2 topic echo /controller/cmd_vel > "${LOG_DIR}/controller_cmd_vel.log" 2>&1 &
stdbuf -oL -eL ros2 topic echo /odom > "${LOG_DIR}/odom.log" 2>&1 &
stdbuf -oL -eL ros2 topic echo /exploration/status > "${LOG_DIR}/exploration_status.log" 2>&1 &
stdbuf -oL -eL ros2 topic echo /navigation/status > "${LOG_DIR}/navigation_status.log" 2>&1 &

wait
