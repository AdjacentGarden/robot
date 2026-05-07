#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE_DIR="$(cd "${ROOT_DIR}/../.." && pwd)"
LOG_ROOT="${EXPLORATION_LOG_ROOT:-${ROOT_DIR}/log}"
RUN_DIR="${LOG_ROOT}/nav_with_map_$(date +%Y%m%d_%H%M%S)"
DIAG_DIR="${RUN_DIR}/diagnostics"
BAG_DIR="${RUN_DIR}/bag"

mkdir -p "${RUN_DIR}" "${DIAG_DIR}" "${BAG_DIR}"

source /opt/ros/humble/setup.bash
source "${WORKSPACE_DIR}/install/setup.bash"

cleanup() {
  jobs -p | xargs -r kill >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

cat > "${RUN_DIR}/README.txt" <<EOF
launch_log: ${RUN_DIR}/launch.log
bag_dir: ${BAG_DIR}
topic_logs: ${RUN_DIR}/*.log
diagnostic_recorder_root: ${DIAG_DIR}
EOF

echo "run dir: ${RUN_DIR}"

stdbuf -oL -eL ros2 launch exploration navigation_with_map.launch.py \
  rviz:=false \
  record_diagnostics:=true \
  "log_root:=${DIAG_DIR}" \
  "$@" 2>&1 | tee "${RUN_DIR}/launch.log" &

sleep 6

stdbuf -oL -eL ros2 bag record \
  -o "${BAG_DIR}" \
  /scan \
  /scan_raw \
  /tf \
  /tf_static \
  /odom \
  /map \
  /amcl_pose \
  /particle_cloud \
  /cmd_vel \
  /controller/cmd_vel \
  /initialpose \
  /goal_pose > "${RUN_DIR}/rosbag.log" 2>&1 &

stdbuf -oL -eL ros2 topic echo /scan > "${RUN_DIR}/scan.log" 2>&1 &
stdbuf -oL -eL ros2 topic echo /odom > "${RUN_DIR}/odom.log" 2>&1 &
stdbuf -oL -eL ros2 topic echo /amcl_pose > "${RUN_DIR}/amcl_pose.log" 2>&1 &
stdbuf -oL -eL ros2 topic echo /tf > "${RUN_DIR}/tf.log" 2>&1 &

wait
