#!/bin/bash
# ============================================================================
# 部署到板子：adb push 二进制 + 模型 + 库 + 显示端脚本，然后启动。
#
# 不需要编译：仓库自带 install/hy8（预编译）和 3rdparty/ 里全部 .so 与头文件，
# 也没有 SDK 也能用。只有你想自己改代码时，才需要 ./build.sh 重新生成。
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

# 优先用预编译产物；没有就退回 build/ 并提示先编译
BIN=$HERE/install/hy8
if [ ! -f "$BIN" ]; then
    BIN=$HERE/build/hy8
    [ -f "$BIN" ] || {
        echo "ERROR: 找不到 install/hy8 或 build/hy8。先跑 ./build.sh" >&2
        exit 1
    }
    echo "提示: 用的是 build/hy8（本机编译产物），仓库预编译版本应在 install/hy8" >&2
fi

echo "=== 部署到 $DIR ==="
"$ADB" shell "mkdir -p '$DIR'"

push() {
    printf '  %-32s' "$(basename "$1")"
    "$ADB" push "$1" "$DIR/$(basename "$1")" >/dev/null 2>&1 \
        && echo "OK" || { echo "FAIL"; exit 1; }
}
push "$BIN"
push "$HERE/model/y8s_rkv.rknn"
push "$HERE/3rdparty/rknn_api/lib/librknnrt.so"
push "$HERE/scripts/humanseg_view.py"
push "$HERE/scripts/humanseg_settings.py"
push "$HERE/scripts/run_humanseg.sh"
push "$HERE/scripts/check_e2e.py"
push "$HERE/scripts/check_shm.py"
push "$HERE/scripts/check_gear.py"
push "$HERE/scripts/check_smooth.py"

"$ADB" shell "chmod +x '$DIR/hy8' '$DIR/run_humanseg.sh'"

# 控制文件：不存在才写默认值，避免覆盖用户现场调好的参数
"$ADB" shell "[ -f '$DIR/ctl.conf' ] || printf 'seg=1\nbg=0\nthr=35\nup=1\nmargin=0\norphan=20\n' > '$DIR/ctl.conf'"

[ "$START" -eq 0 ] && { echo "已部署（未启动）"; exit 0; }

echo
echo "=== 启动 ==="
"$ADB" shell "sh '$DIR/run_humanseg.sh' --restart"
