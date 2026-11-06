#!/bin/bash
# ============================================================================
# 交叉编译 humanseg_y8seg.cc -> build/hy8（aarch64）
#
# 为什么不用 pkg-config：target 的 .pc 文件指向 SDK staging 的绝对路径，
# 换一台机器 / 换个 SDK 路径就失效。头文件和链接库都已随仓库提供，
# 这里显式给出 -I / -L，拿到仓库即可编译，不依赖本机 SDK 安装。
#
# 只需要交叉工具链（aarch64-none-linux-gnu-g++）。装法见 README。
# ============================================================================
set -eu

HERE=$(cd "$(dirname "$0")" && pwd)
OUT=${1:-$HERE/build/hy8}
CXX=${CXX:-aarch64-none-linux-gnu-g++}

# 找工具链：优先 $CXX，其次 SDK prebuilts，最后 PATH
if ! command -v "$CXX" >/dev/null 2>&1; then
    for c in \
        "$HOME"/share/zenith_code/RK3576_Linux6.1_BL01_v1.2.0/prebuilts/gcc/linux-x86/aarch64/*/bin/aarch64-none-linux-gnu-g++ \
        /opt/*/bin/aarch64-none-linux-gnu-g++ \
        /usr/bin/aarch64-linux-gnu-g++
    do
        [ -x "$c" ] && { CXX=$c; break; }
    done
fi
command -v "$CXX" >/dev/null 2>&1 || {
    echo "ERROR: 找不到 aarch64 交叉编译器。" >&2
    echo "       装好后用 CXX=/path/to/aarch64-none-linux-gnu-g++ $0" >&2
    echo "       Debian/Ubuntu: sudo apt install g++-aarch64-linux-gnu" >&2
    exit 1
}

mkdir -p "$(dirname "$OUT")"

# 头文件搜索路径：gst + glib + glibconfig + rknn
INC=(
    -I"$HERE/3rdparty/gst/include/gstreamer-1.0"
    -I"$HERE/3rdparty/gst/include/glib-2.0"
    -I"$HERE/3rdparty/gst/include/glib-2.0-internal"
    -I"$HERE/3rdparty/rknn_api/include"
)
LIB=(
    -L"$HERE/3rdparty/gst/lib"
    -L"$HERE/3rdparty/rknn_api/lib"
)

echo "编译器: $CXX"
"$CXX" --version | head -1

"$CXX" -O2 -std=c++11 -Wall \
    "$HERE/src/humanseg_y8seg.cc" \
    "${INC[@]}" \
    -o "$OUT" \
    "${LIB[@]}" \
    -lrknnrt -lgstreamer-1.0 -lgstapp-1.0 -lgobject-2.0 -lglib-2.0 \
    -Wl,-rpath-link,"$HERE/3rdparty/gst/lib" \
    -Wl,--allow-shlib-undefined \
    -lpthread -lrt

echo
echo "built: $OUT  ($(stat -c %s "$OUT") 字节)"
file "$OUT" 2>/dev/null | sed 's/^/  /' || true
