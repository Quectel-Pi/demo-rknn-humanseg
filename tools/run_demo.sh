#!/bin/bash
# 板端常驻启动 humanseg_cam（虚拟背景 demo）
# 用法: ./run_demo.sh [bg_mode] [disp WxH]
BG=${1:-0}
DISP=${2:-1280x800}
LOG=/data/hs_test/cam_live.log

pkill -f humanseg_cam 2>/dev/null
sleep 1

cd /data/hs_test || exit 1

echo "=== start $(date) bg=$BG disp=$DISP ===" >> "$LOG"

sudo -u qq env \
    XDG_RUNTIME_DIR=/run/user/1000 \
    WAYLAND_DISPLAY=wayland-0 \
    LD_LIBRARY_PATH=/data/hs_test:/usr/lib \
    /data/hs_test/humanseg_cam \
        /data/hs_test/humanseg_v1_lite_398x224_static.rknn \
        --device /dev/video73 \
        --cam 1280x720 \
        --disp "$DISP" \
        --bg "$BG" \
        --fps >> "$LOG" 2>&1

echo "=== exit $? $(date) ===" >> "$LOG"
