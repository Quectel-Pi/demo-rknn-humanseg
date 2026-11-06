#!/bin/sh
# ============================================================================
# 一键启动：人像分割实时 demo（后端 hy8 + GTK4 显示端）
#
# 设计要点：
#   * 桌面用户**不硬编码**。板子上跑桌面的人可能是 qq、pi、rock 或任何
#     名字，写死必然在别人板子上失败。这里从 Wayland socket 的属主反查：
#     /run/user/<uid>/wayland-<n> 的 owner 就是真正持有桌面的用户。
#   * XDG_RUNTIME_DIR / WAYLAND_DISPLAY 同样从该 socket 反推，不猜。
#   * 幂等：已在跑就不重复启动，可反复执行。
#   * 不假设自己在哪个目录，全部路径可覆盖。
#
# 用法：
#   ./run_humanseg.sh                 启动（后端 + 显示端）
#   ./run_humanseg.sh --restart       先停再起
#   ./run_humanseg.sh --stop          停止
#   ./run_humanseg.sh --status        只看状态
#   ./run_humanseg.sh --no-view       只启后端（调试后端时用）
#   ./run_humanseg.sh --user pi       强制指定桌面用户
#   ./run_humanseg.sh --cam /dev/video0 --cam-res 640x480
# ============================================================================
set -u

# ── 默认值（全部可覆盖）────────────────────────────────────────────────────
DATA_DIR=/data/hs_test
MODEL=y8s_rkv.rknn
BACKEND=hy8
VIEWER=humanseg_view.py
CAM=/dev/video73
CAM_RES=1280x720
DISP_RES=1280x800
SHM=/dev/shm/humanseg.frames
CTL=$DATA_DIR/ctl.conf
FORCE_USER=
DO_STOP=0
DO_STATUS=0
DO_RESTART=0
NO_VIEW=0

log()  { printf '%s\n' "$*"; }
info() { printf '  %s\n' "$*"; }
err()  { printf 'ERROR: %s\n' "$*" >&2; }
die()  { err "$*"; exit 1; }

# ── 参数 ───────────────────────────────────────────────────────────────────
while [ $# -gt 0 ]; do
    case "$1" in
        --dir)      DATA_DIR=$2; shift 2 ;;
        --model)    MODEL=$2; shift 2 ;;
        --backend)  BACKEND=$2; shift 2 ;;
        --viewer)   VIEWER=$2; shift 2 ;;
        --cam)      CAM=$2; shift 2 ;;
        --cam-res)  CAM_RES=$2; shift 2 ;;
        --disp)     DISP_RES=$2; shift 2 ;;
        --shm)      SHM=$2; shift 2 ;;
        --ctl)      CTL=$2; shift 2 ;;
        --user)     FORCE_USER=$2; shift 2 ;;
        --stop)     DO_STOP=1; shift ;;
        --status)   DO_STATUS=1; shift ;;
        --restart)  DO_RESTART=1; shift ;;
        --no-view)  NO_VIEW=1; shift ;;
        -h|--help)  sed -n '3,20p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *)          die "未知参数: $1（--help 看用法）" ;;
    esac
done

# ============================================================================
# 探测桌面会话：从 Wayland socket 反查 uid / display / 用户名
#
# 为什么不查 gnome-shell 进程：进程名随桌面环境变化（weston / kwin / sway），
# 而 Wayland socket 是协议层约定，任何合成器都必须建它，最稳。
#
# 多用户共存时（比如 root 手工 --user 建过一个、或残留 socket），
# 单纯取第一个会选错。真正在用的 socket 一定被合成器进程持有 fd，
# 所以用 /proc/*/fd 反查「有进程打开着它」来区分真假。
# ============================================================================
socket_held_by() {
    # 判断「这个 Wayland socket 是否真的有人在用」。
    #
    # 踩过的坑：不能拿 /proc/<pid>/fd 的 readlink 结果去 stat 文件系统上的 socket
    # —— readlink 给的是 "socket:[N]"，N 是 sockfs 的独立编号，与 socket 文件的
    # 文件系统 inode 毫无关系，比对永远不相等（实测「持有者=0」）。
    #
    # 权威来源是 /proc/net/unix：它把「路径」和「inode」放在同一行，
    # 于是能拿到该路径对应的 inode，再去 /proc/*/fd 里找 socket:[inode]。
    target=$1
    inos=$(awk -v p="$target" '$NF == p { print $7 }' /proc/net/unix 2>/dev/null | sort -u)
    [ -n "$inos" ] || return 1
    for fd in /proc/[0-9]*/fd/*; do
        l=$(readlink "$fd" 2>/dev/null) || continue
        case "$l" in "socket:["*) ;; *) continue ;; esac
        want=${l#socket:[}; want=${want%]}
        for i in $inos; do
            [ "$want" = "$i" ] && return 0
        done
    done
    return 1
}

detect_wayland() {
    picked=
    any=
    for s in /run/user/*/wayland-[0-9]*; do
        # 只要真正的 socket（wayland-0 是 socket，wayland-0.lock 是普通文件）
        [ -S "$s" ] || continue
        rundir=$(dirname "$s")
        uid=$(basename "$rundir")
        case "$uid" in ''|*[!0-9]*) continue ;; esac
        disp=$(basename "$s")
        owner=$(stat -c %U "$s" 2>/dev/null)
        [ -n "$owner" ] || continue

        cand_uid=$uid
        cand_disp=$disp
        cand_user=$owner
        cand_xdg=$rundir

        if [ -z "$any" ]; then
            any=1
            WUID=$cand_uid; WDISP=$cand_disp; WUSER=$cand_user; WXDG=$cand_xdg
        fi
        # 优先选被真实进程持有的（= 合成器正在跑）
        if socket_held_by "$s"; then
            picked=1
            WUID=$cand_uid; WDISP=$cand_disp; WUSER=$cand_user; WXDG=$cand_xdg
            break
        fi
    done
    [ -n "$any" ] || return 1
    if [ -z "$picked" ]; then
        err "找到 Wayland socket 但没有任何进程持有它（桌面可能没在运行）"
        err "将使用 $WUSER ($WXDG/$WDISP)，如有误请加 --user <用户名>"
    fi
    return 0
}

# 指定了 --user：反查它的 uid 与 socket
resolve_user() {
    WUSER=$FORCE_USER
    WUID=$(id -u "$WUSER" 2>/dev/null) || die "用户不存在: $WUSER"
    WXDG=/run/user/$WUID
    # 该用户可能有 wayland-0 / wayland-1 ...，取第一个存在的 socket
    for s in "$WXDG"/wayland-[0-9]*; do
        [ -S "$s" ] || continue
        WDISP=$(basename "$s")
        return 0
    done
    err "用户 $WUSER 有 uid=$WUID，但 $WXDG 下没有 Wayland socket"
    err "（= 这个用户当前没登录图形桌面，或桌面没起来）"
    return 1
}

# ============================================================================
# 进程定位：只认「可执行文件 / 命令行首个参数」完全匹配的进程
#
# pgrep -f 会把发起查询的 shell 自己也匹配进来（命令行里含同样的字符串），
# 之前多次误杀 adb shell 自身，所以这里读 /proc/<pid>/cmdline 逐字比对。
# ============================================================================
backend_pids() {
    for p in /proc/[0-9]*; do
        pid=${p#/proc/}
        [ -r "$p/comm" ] || continue
        comm=$(cat "$p/comm" 2>/dev/null) || continue
        [ "$comm" = "$(basename "$BACKEND")" ] && printf '%s\n' "$pid"
    done
}

viewer_pids() {
    want="python3 $DATA_DIR/$VIEWER"
    for p in /proc/[0-9]*; do
        pid=${p#/proc/}
        # 遍历期间进程可能已退出，读不到就跳过（否则 sh 会把重定向错误
        # 直接打到 stderr，混进 --status 输出里）
        [ -r "$p/cmdline" ] || continue
        cmd=$(tr '\0' ' ' < "$p/cmdline" 2>/dev/null) || continue
        # 去掉尾部空格后精确比较，避免匹配到 shell / 探测进程
        cmd=${cmd% }
        [ "$cmd" = "$want" ] && printf '%s\n' "$pid"
    done
}

# ============================================================================
# 停止
# ============================================================================
stop_all() {
    log "=== 停止 ==="
    n=0
    for pid in $(viewer_pids); do
        kill "$pid" 2>/dev/null && { info "显示端 pid=$pid 已停止"; n=$((n+1)); }
    done
    for pid in $(backend_pids); do
        kill "$pid" 2>/dev/null && { info "后端   pid=$pid 已停止"; n=$((n+1)); }
    done
    [ "$n" -eq 0 ] && info "没有在运行的进程"
    sleep 1
    # 还在就强杀
    for pid in $(viewer_pids) $(backend_pids); do
        kill -9 "$pid" 2>/dev/null && info "pid=$pid 强杀"
    done
    return 0
}

# ============================================================================
# 状态
# ============================================================================
show_status() {
    log "=== 状态 ==="
    bp=$(backend_pids | tr '\n' ' ')
    vp=$(viewer_pids | tr '\n' ' ')
    info "后端  : ${bp:-未运行}"
    info "显示端: ${vp:-未运行}"

    if [ -e "$SHM" ]; then
        info "共享内存: $(stat -c %s "$SHM" 2>/dev/null) 字节"
        seq1=$(od -An -tu4 -N4 "$SHM" 2>/dev/null | tr -d ' ')
        sleep 0.5
        seq2=$(od -An -tu4 -N4 "$SHM" 2>/dev/null | tr -d ' ')
        if [ -n "$seq1" ] && [ "$seq1" != "$seq2" ]; then
            info "当前 seq: $seq2（持续增长 = 正在出帧）"
        else
            info "当前 seq: ${seq2:-?}（**不再增长 = 已停更**）"
        fi
    else
        info "共享内存: 不存在"
    fi

    # 进程已停时日志是历史值，必须标出来，否则会误读成"画面在跑"
    if [ -n "$bp" ]; then tag=""
    else tag="  ← 历史记录，进程已停"; fi
    [ -f "$DATA_DIR/live.log" ] && \
        info "后端: $(grep 'shown=' "$DATA_DIR/live.log" 2>/dev/null | tail -1)$tag"

    if [ -n "$vp" ]; then tag=""
    else tag="  ← 历史记录，进程已停"; fi
    [ -f "$DATA_DIR/view.log" ] && \
        info "显示: $(tail -1 "$DATA_DIR/view.log" 2>/dev/null)$tag"

    [ -f "$CTL" ] && info "配置: $(tr '\n' ' ' < "$CTL")"
    return 0
}

# ============================================================================
# 主流程
# ============================================================================
[ "$DO_STOP" -eq 1 ]   && { stop_all; exit 0; }
[ "$DO_STATUS" -eq 1 ] && { show_status; exit 0; }
[ "$DO_RESTART" -eq 1 ] && stop_all

log "=== 人像分割 demo 启动 ==="

# ── 1. 依赖检查（先报清楚缺什么，别让用户对着空画面猜）──────────────
log "[1/5] 检查依赖"
miss=0
for f in "$DATA_DIR/$BACKEND" "$DATA_DIR/$MODEL" "$DATA_DIR/librknnrt.so"; do
    if [ -f "$f" ]; then info "OK   $f"; else err "缺失 $f"; miss=1; fi
done
[ "$NO_VIEW" -eq 0 ] && {
    if [ -f "$DATA_DIR/$VIEWER" ]; then info "OK   $DATA_DIR/$VIEWER"
    else err "缺失 $DATA_DIR/$VIEWER"; miss=1; fi
}
if [ -c "$CAM" ]; then info "OK   摄像头 $CAM"
else err "摄像头不存在: $CAM（ls /dev/video* 看实际节点）"; miss=1; fi
[ "$miss" -eq 1 ] && die "依赖不全，已中止（未启动任何进程）"

# ── 2. 桌面会话 ────────────────────────────────────────────────────────────
log "[2/5] 探测桌面会话"
if [ -n "$FORCE_USER" ]; then
    resolve_user || die "指定的桌面用户不可用"
else
    detect_wayland || die "没找到任何 Wayland socket（/run/user/*/wayland-*）。
       板子是否已登录图形桌面？若是纯串口/headless 环境，用 --no-view 只启后端；
       或 --user <用户名> 手工指定。"
fi
info "桌面用户 : $WUSER (uid=$WUID)"
info "会话环境 : XDG_RUNTIME_DIR=$WXDG  WAYLAND_DISPLAY=$WDISP"

# 以 root 运行时，显示端必须降到桌面用户；否则拿不到该用户的 Wayland 会话
if [ "$(id -u)" -eq 0 ]; then
    if ! id "$WUSER" >/dev/null 2>&1; then
        die "要以 $WUSER 启动显示端，但该用户不存在"
    fi
fi

# ── 3. 后端 ────────────────────────────────────────────────────────────────
log "[3/5] 后端 $BACKEND"
if [ -n "$(backend_pids)" ]; then
    info "已在运行 (pid $(backend_pids | tr '\n' ' '))，跳过"
else
    [ -d "$DATA_DIR" ] || die "数据目录不存在: $DATA_DIR"
    rm -f "$SHM"
    cd "$DATA_DIR" || die "无法进入 $DATA_DIR"
    # 后端以当前身份跑（root 时就是 root）—— 摄像头与 NPU 都要访问设备节点
    setsid env \
        XDG_RUNTIME_DIR="$WXDG" \
        WAYLAND_DISPLAY="$WDISP" \
        LD_LIBRARY_PATH="$DATA_DIR" \
        "./$BACKEND" "./$MODEL" \
            --device "$CAM" --cam "$CAM_RES" --disp "$DISP_RES" \
            --fps --prof --ctl "$CTL" \
            --shm "$SHM" > "$DATA_DIR/live.log" 2>&1 < /dev/null &
    info "启动中 -> $DATA_DIR/live.log"
fi

# ── 4. 等后端出第一帧 ──────────────────────────────────────────────────────
log "[4/5] 等待后端出帧"
i=0
while [ $i -lt 40 ]; do
    if [ -e "$SHM" ]; then
        seq=$(od -An -tu4 -N4 "$SHM" 2>/dev/null | tr -d ' ')
        [ -n "$seq" ] && [ "$seq" -gt 0 ] 2>/dev/null && { info "已出帧 seq=$seq"; break; }
    fi
    if [ -n "$(backend_pids)" ] || [ -e "$SHM" ]; then :; else
        tail -5 "$DATA_DIR/live.log" 2>/dev/null
        die "后端已退出，见 $DATA_DIR/live.log"
    fi
    sleep 0.2
    i=$((i+1))
done
[ $i -ge 40 ] && die "等待出帧超时（8s），检查 $DATA_DIR/live.log"

# ── 5. 显示端 ──────────────────────────────────────────────────────────────
if [ "$NO_VIEW" -eq 1 ]; then
    log "[5/5] 已按 --no-view 跳过显示端"
else
    log "[5/5] 显示端 $VIEWER"
    if [ -n "$(viewer_pids)" ]; then
        info "已在运行 (pid $(viewer_pids | tr '\n' ' '))，跳过"
    else
        if [ "$(id -un)" = "$WUSER" ]; then
            setsid env XDG_RUNTIME_DIR="$WXDG" WAYLAND_DISPLAY="$WDISP" \
                python3 "$DATA_DIR/$VIEWER" \
                > "$DATA_DIR/view.log" 2>&1 < /dev/null &
        else
            # 降权到桌面用户；-H 让它继承当前环境变量之外只保留我们显式给的
            setsid su -s /bin/sh "$WUSER" -c \
                "XDG_RUNTIME_DIR='$WXDG' WAYLAND_DISPLAY='$WDISP' \
                 python3 '$DATA_DIR/$VIEWER' \
                 > '$DATA_DIR/view.log' 2>&1 < /dev/null &" \
                2>/dev/null || \
            setsid sudo -u "$WUSER" env XDG_RUNTIME_DIR="$WXDG" \
                WAYLAND_DISPLAY="$WDISP" \
                python3 "$DATA_DIR/$VIEWER" \
                > "$DATA_DIR/view.log" 2>&1 < /dev/null &
        fi
        info "启动中 -> $DATA_DIR/view.log"
    fi
    sleep 3
fi

# ── 汇报 ───────────────────────────────────────────────────────────────────
log ""
log "=== 启动完成 ==="
bp=$(backend_pids | tr '\n' ' ')
vp=$(viewer_pids | tr '\n' ' ')
info "后端   : ${bp:-未运行}"
if [ "$NO_VIEW" -eq 0 ]; then info "显示端 : ${vp:-未运行}"; fi
[ -f "$DATA_DIR/live.log" ] && \
    info "推理   : $(grep 'shown=' "$DATA_DIR/live.log" 2>/dev/null | tail -1)"
if [ "$NO_VIEW" -eq 0 ] && [ -f "$DATA_DIR/view.log" ]; then
    last=$(tail -1 "$DATA_DIR/view.log" 2>/dev/null)
    info "显示   : $last"
    case "$last" in
        *painted*) : ;;
        *) err "显示端还没画出帧，检查 $DATA_DIR/view.log" ;;
    esac
fi
info "配置   : $( [ -f "$CTL" ] && tr '\n' ' ' < "$CTL" )"
log ""
log "看状态: $0 --status    停止: $0 --stop    重来: $0 --restart"
