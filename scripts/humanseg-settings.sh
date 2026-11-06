#!/bin/bash
# 打开「人像分割」设置窗口（桌面图标 / 应用菜单都调用这个）
# 已经开着就把它提到前面，不重复启动。
CTL_DIR=/data/hs_test
SELF="$(readlink -f "$0")"

export XDG_RUNTIME_DIR=/run/user/1000
export WAYLAND_DISPLAY=wayland-0

# 已在运行 -> 只提示（GTK 单窗口，避免开出一堆）
if pgrep -f "humanseg_settings.py" >/dev/null 2>&1; then
    exit 0
fi

exec python3 "$CTL_DIR/humanseg_settings.py"
