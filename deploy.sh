#!/bin/bash
# ============================================================================
# 部署到板子：adb push 二进制 + 模型 + 库 + 显示端脚本，然后启动。
#
# 用法：
#   ./deploy.sh              部署并启动
#   ./deploy.sh --no-start   只部署不启动
#   ADB="adb -s <serial>" ./deploy.sh   多设备时指定
# ============================================================================
set -eu
HERE=$(cd "$(dirname "$0")" && pwd)
ADB=${ADB:-adb}
DIR=${DIR:-/data/hs_test}
START=1
[ "${1:-}" = "--no-start" ] && START=0

"$ADB" get-state >/dev/null 2>&1 || {
    echo "ERROR: 没连上板子。先确认 adb devices 能看到设备。" >&2
    exit 1
}

echo "=== 部署到 $DIR ==="
"$ADB" shell "mkdir -p '$DIR'"

push() {
    printf '  %-32s' "$(basename "$1")"
    "$ADB" push "$1" "$DIR/$(basename "$1")" >/dev/null 2>&1 \
        && echo "OK" || { echo "FAIL"; exit 1; }
}
push "$HERE/build/hy8"
push "$HERE/model/y8s_rkv.rknn"
push "$HERE/3rdparty/rknn_api/lib/librknnrt.so"
push "$HERE/scripts/humanseg_view.py"
push "$HERE/scripts/humanseg_settings.py"
push "$HERE/scripts/run_humanseg.sh"

"$ADB" shell "chmod +x '$DIR/hy8' '$DIR/run_humanseg.sh'"

[ "$START" -eq 0 ] && { echo "已部署（未启动）"; exit 0; }

echo
echo "=== 启动 ==="
"$ADB" shell "sh '$DIR/run_humanseg.sh' --restart"
