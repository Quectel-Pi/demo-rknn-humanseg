#!/bin/bash
# 板端常驻运行：yolov8n-seg 人像分割 + 虚拟背景
#
# 用法: ./run_y8.sh [bg] [disp] [cam]
#   bg   0=深青绿 1=原图暗化 2=灰 3=绿幕
BG=${1:-0}
DISP=${2:-1280x800}
CAM=${3:-1280x720}
LOG=/data/hs_test/y8_run.log

cd /data/hs_test || exit 1

# 用精确进程名匹配，避免 pkill -f 匹配到本脚本自身
for p in $(ps -eo pid,comm | awk '$2=="hy8"{print $1}'); do
    kill -9 "$p" 2>/dev/null
done
sleep 1

echo "=== start $(date) bg=$BG disp=$DISP cam=$CAM ===" >> "$LOG"

sudo -u qq env \
    XDG_RUNTIME_DIR=/run/user/1000 \
    WAYLAND_DISPLAY=wayland-0 \
    LD_LIBRARY_PATH=/data/hs_test:/usr/lib \
    /data/hs_test/hy8 \
        /data/hs_test/y8s_rkv.rknn \
        --device /dev/video73 \
        --cam "$CAM" \
        --disp "$DISP" \
        --bg "$BG" \
        --fps >> "$LOG" 2>&1

echo "=== exit $? $(date) ===" >> "$LOG"
