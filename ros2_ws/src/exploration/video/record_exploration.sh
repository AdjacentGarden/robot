#!/bin/bash
# Record exploration in headless mode.
# RViz renders in a virtual framebuffer (Xvfb) — nothing appears on your desktop.
# Auto-stops when all cycles complete.
#
# Usage:
#   bash video/record_exploration.sh [output_name] [seed]
#
# Examples:
#   bash video/record_exploration.sh                    # random seed
#   bash video/record_exploration.sh exploration_demo 42
#
# Dependencies: Xvfb, ffmpeg

OUTPUT="${1:-exploration_run}"
SEED="${2:--1}"
RESOLUTION="1920x1080"
FPS=15
VDISPLAY=":99"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PKG_DIR="$(dirname "$SCRIPT_DIR")"
WS_DIR="$(cd "$PKG_DIR/../.." && pwd)"
OUT_PATH="${SCRIPT_DIR}/${OUTPUT}.mp4"
LOG_FILE="/tmp/exploration_record_$$.log"

# Source ROS workspace
source /opt/ros/humble/setup.bash
source "${WS_DIR}/install/setup.bash"

cleanup() {
    echo ""
    echo "Stopping..."
    kill "$FFMPEG_PID" 2>/dev/null; wait "$FFMPEG_PID" 2>/dev/null
    kill "$LAUNCH_PID" 2>/dev/null; wait "$LAUNCH_PID" 2>/dev/null
    # Kill all child ROS nodes
    killall -9 sim_environment custom_navigator frontier_explorer rviz2 2>/dev/null
    kill "$XVFB_PID" 2>/dev/null
    echo "Video saved: ${OUT_PATH}"
    ls -lh "${OUT_PATH}" 2>/dev/null
}
trap cleanup EXIT

# ── Start virtual framebuffer ──
# Kill any leftover Xvfb on this display
kill "$(cat /tmp/.X99-lock 2>/dev/null)" 2>/dev/null; sleep 0.5
Xvfb ${VDISPLAY} -screen 0 ${RESOLUTION}x24 +extension GLX &
XVFB_PID=$!
sleep 1

echo "Virtual display ${VDISPLAY} started (PID: ${XVFB_PID})"
echo "Recording to: ${OUT_PATH}"
echo "Seed: ${SEED}"
echo ""

# ── Launch exploration on virtual display ──
DISPLAY=${VDISPLAY} LIBGL_ALWAYS_SOFTWARE=1 \
    ros2 launch exploration exploration_test_rviz.launch.py seed:="${SEED}" \
    > "${LOG_FILE}" 2>&1 &
LAUNCH_PID=$!
sleep 12  # Wait for all nodes (sim 0s, nav 3s, explorer 8s, rviz startup)

echo "Exploration launched (PID: ${LAUNCH_PID})"

# ── Start recording the virtual display ──
ffmpeg -y \
    -video_size ${RESOLUTION} \
    -framerate ${FPS} \
    -f x11grab \
    -draw_mouse 0 \
    -i "${VDISPLAY}" \
    -c:v libx264 \
    -preset fast \
    -crf 23 \
    "${OUT_PATH}" \
    > /dev/null 2>&1 &
FFMPEG_PID=$!
sleep 1

echo "Recording started (PID: ${FFMPEG_PID})"
echo "Waiting for all cycles to complete..."
echo ""

# ── Wait for completion ──
ELAPSED=0
while true; do
    sleep 10
    ELAPSED=$((ELAPSED + 10))

    # Print status
    STATUS=$(grep -oP '(?<=\[frontier_explorer\]: ).*' "${LOG_FILE}" | tail -1)
    echo "[${ELAPSED}s] ${STATUS}"

    # Check if all cycles finished
    if grep -q "ALL CYCLES COMPLETE" "${LOG_FILE}" 2>/dev/null; then
        echo ""
        echo "*** All cycles complete! Recording 10 more seconds... ***"
        sleep 10
        break
    fi

    # Check if launch process died
    if ! kill -0 "$LAUNCH_PID" 2>/dev/null; then
        echo "Launch process exited unexpectedly"
        break
    fi

    # Safety timeout: 45 minutes
    if [ "$ELAPSED" -ge 2700 ]; then
        echo "Safety timeout (45 min) reached"
        break
    fi
done

echo ""
echo "Done!"
