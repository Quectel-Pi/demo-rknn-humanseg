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
BP_STARTED=0

log()  { printf '%s\n' "$*"; }
info() { printf '  %s\n' "$*"; }
err()  { printf 'ERROR: %s\n' "$*" >&2; }
die()  { err "$*"; exit 1; }

# 分段计时：默认关闭，HS_PROF=1 时把每个阶段耗时打到 stderr。
# 启动慢的时候不用再猜是哪一段，直接 HS_PROF=1 ./run_humanseg.sh 看数字。
_T_START=$(date +%s%N 2>/dev/null || echo 0)
_T_LAST=$_T_START
mark() {
    [ "${HS_PROF:-0}" = 1 ] || return 0
    _t=$(date +%s%N 2>/dev/null) || return 0
    printf '[prof] %-22s %6d ms  (累计 %6d ms)\n' \
        "$1" $(( (_t - _T_LAST) / 1000000 )) $(( (_t - _T_START) / 1000000 )) >&2
    _T_LAST=$_t
    return 0
}

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
    # 旧实现遍历 /proc/<pid>/fd/*，把所有进程的所有 fd 都 readlink 一遍，
    # 再和 /proc/net/unix 里的 inode 比对。逻辑正确但在板上极慢：约 400 个
    # 进程、上万个 fd，实测这一段独占 2.8 秒（启动总耗时的一半以上）。
    #
    # 换个等价而廉价得多的判据：Wayland socket 要被真正使用，前提是那一端
    # 有合成器在跑。合成器进程存在 <=> socket 活着，不必逐个 fd 去查。
    # 这里只问「有没有 gnome-shell / kwin / weston / sway 之类在跑」，
    # 用 pgrep -x（约 75ms）。查不到也不致命：调用方会退化成"用第一个
    # socket + 打警告"，而 socket 属主仍然来自 stat，选人依旧准确。
    target=$1
    for c in gnome-shell kwin_wayland kwin_x11 weston sway wayfire labwc mutter; do
        pgrep -x "$c" >/dev/null 2>&1 && return 0
    done
    # 兜底：socket 存在且在最近被触碰过，也认为桌面在跑
    [ -S "$target" ] || return 1
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
# ── 进程探测 ───────────────────────────────────────────────────────────────
#
# 不要手写 `for p in /proc/[0-9]*` 去读 comm/cmdline：板上实测这样扫一遍要
# 1.6 秒（约 400 个进程，逐个 open+read 会卡在 D 状态进程上）。脚本原先要
# 调用十几次，启动时间几乎全耗在这里（实测端到端 16 秒）。
# pgrep 由内核一次遍历完成，同一个查询只要 ~75ms，快 20 倍。
backend_pids() {
    pgrep -x "$(basename "$BACKEND")" 2>/dev/null
}

viewer_pids() {
    # 用 -f 匹配完整命令行，^ 锚定开头，末尾**不锚定**：
    # /proc/<pid>/cmdline 里各 argv 之间是 \0，pgrep 内部按 NUL 处理，
    # 而单测时用 tr 看会多出一个尾空格，加 $ 反而匹配不到（实测踩坑）。
    # 前缀已足够精确：只有真正的 `python3 <路径>` 才会命中；
    # 包含该路径的 shell/探测命令都以 /bin/... 开头，不会被 ^ 放过。
    pgrep -f "^python3 $DATA_DIR/$VIEWER" 2>/dev/null
}

# 兼容旧调用点：现在每次都是实时查询，无需缓存/刷新
rescan() { :; }
rescan_all() { :; }

# ============================================================================
# 停止
# ============================================================================
stop_all() {
    log "=== 停止 ==="
    n=0
    PIDS=""
    for pid in $(viewer_pids); do
        PIDS="$PIDS $pid"
        kill "$pid" 2>/dev/null && { info "显示端 pid=$pid 已停止"; n=$((n+1)); }
    done
    for pid in $(backend_pids); do
        PIDS="$PIDS $pid"
        kill "$pid" 2>/dev/null && { info "后端   pid=$pid 已停止"; n=$((n+1)); }
    done
    [ "$n" -eq 0 ] && info "没有在运行的进程"
    if [ "$n" -eq 0 ]; then return 0; fi

    # 等它们真的退出（最多 2s），而不是直接 sleep 1 再无条件 -9。
    # 后端要释放摄像头/NPU 并反初始化 GStreamer，实测常需 0.3~2s；
    # 过早 -9 会留下没释放的 shm 和设备句柄。
    k=0
    while [ $k -lt 40 ]; do
        alive=0
        for pid in $PIDS; do
            kill -0 "$pid" 2>/dev/null && alive=1
        done
        [ "$alive" -eq 0 ] && break
        sleep 0.05
        k=$((k+1))
    done
    for pid in $PIDS; do
        if kill -0 "$pid" 2>/dev/null; then
            kill -9 "$pid" 2>/dev/null && info "pid=$pid 未响应，强杀"
        fi
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
mark "start"
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
mark "deps"
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
mark "session"
log "[3/5] 后端 $BACKEND"
_bp_now=$(backend_pids)
if [ -n "$_bp_now" ]; then
    info "已在运行 (pid $(echo $_bp_now))，跳过"
    BP_STARTED=0
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
    BP_STARTED=$!
    info "启动中 -> $DATA_DIR/live.log (pid $BP_STARTED)"
fi

# ── 4. 等后端出第一帧 ──────────────────────────────────────────────────────
mark "backend-spawn"
log "[4/5] 等待后端出帧"
# 轮询只看两件事：shm 里 seq 是否 > 0，以及我们启动的那个 pid 是否还活着。
# 存活判断用 kill -0 打单个 pid（O(1)），不再每次遍历 /proc——那要 1.6 秒。
i=0
while [ $i -lt 100 ]; do
    if [ -e "$SHM" ]; then
        seq=$(od -An -tu4 -N4 "$SHM" 2>/dev/null | tr -d ' ')
        if [ -n "$seq" ] && [ "$seq" -gt 0 ] 2>/dev/null; then
            info "已出帧 seq=$seq ($((i * 50)) ms)"
            break
        fi
    fi
    # 自己启动的进程若已退出就立即报错，不要干等到超时
    if [ "$BP_STARTED" != 0 ] && ! kill -0 "$BP_STARTED" 2>/dev/null; then
        tail -5 "$DATA_DIR/live.log" 2>/dev/null
        die "后端已退出，见 $DATA_DIR/live.log"
    fi
    sleep 0.05
    i=$((i+1))
done
[ $i -ge 100 ] && die "等待出帧超时（5s），检查 $DATA_DIR/live.log"

# ── 5. 显示端 ──────────────────────────────────────────────────────────────
if [ "$NO_VIEW" -eq 1 ]; then
    log "[5/5] 已按 --no-view 跳过显示端"
else
    mark "backend-first-frame"
    log "[5/5] 显示端 $VIEWER"
    if [ -n "$(viewer_pids)" ]; then
        info "已在运行 (pid $(viewer_pids | tr '\n' ' '))，跳过"
    else
        # 只在真要起新进程时清空日志。上一轮留下的 "first frame" 会让下面的
        # 等待循环立刻 break，把"其实是旧日志"误报成"已上屏"；反过来，新进程
        # 第一行还没落盘时读旧文件又会误报"没画出帧"。
        # 幂等分支（已在运行）绝不能清，否则抹掉正在跑那套进程的历史。
        : > "$DATA_DIR/view.log"
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


    # 等显示端真的画出第一帧，而不是盲等固定 3 秒。
    # 实测首帧约 0.4~0.9 秒（导入 GTK 占大头），盲等 3 秒纯属浪费；
    # 而机器慢时 3 秒又可能不够，所以按日志里的实际标记判定。
    j=0
    while [ $j -lt 120 ]; do
        grep -q "first frame" "$DATA_DIR/view.log" 2>/dev/null && break
        sleep 0.05
        j=$((j+1))
    done
    if [ $j -ge 120 ]; then
        err "显示端 6s 内没画出首帧，检查 $DATA_DIR/view.log"
    else
        info "已上屏：$(grep 'first frame' "$DATA_DIR/view.log" | tail -1)"
    fi
fi

mark "viewer-ready"
# ── 汇报 ───────────────────────────────────────────────────────────────────
log ""
log "=== 启动完成 ==="

# 显示端是经由 su/sudo 再 setsid 起来的，父进程拿不到它的 pid，只能靠
# pgrep 反查。进程刚 exec 完时 pgrep 可能还没把它算进 /proc 的可见集合，
# 之前这里就因此误报"显示端未运行"（而日志里明明已有 first frame）。
# 判据以**日志里的首帧标记**为准，pid 只是附加信息，取不到不算失败。
vpid=""
k=0
while [ $k -lt 40 ]; do
    vpid=$(viewer_pids | tr '\n' ' ')
    grep -q "first frame" "$DATA_DIR/view.log" 2>/dev/null && break
    sleep 0.05
    k=$((k+1))
done
bp=$(backend_pids | tr '\n' ' ')
info "后端   : ${bp:-未运行}"
if [ "$NO_VIEW" -eq 0 ]; then info "显示端 : ${vpid:-（日志已确认在绘帧）}"; fi
[ -f "$DATA_DIR/live.log" ] && \
    info "推理   : $(grep 'shown=' "$DATA_DIR/live.log" 2>/dev/null | tail -1)"
if [ "$NO_VIEW" -eq 0 ] && [ -f "$DATA_DIR/view.log" ]; then
    # 取 painted/首帧那一行，不能取 tail -1：日志里还混着 Gtk-CRITICAL、
    # MESA/ZINK 之类 stderr 噪声，最后一行常常是警告而不是状态，会误报。
    last=$(grep -E "painted [0-9]+ frames|first frame at" "$DATA_DIR/view.log" \
           2>/dev/null | tail -1)
    if [ -n "$last" ]; then
        info "显示   : $last"
    else
        err "显示端还没画出帧，检查 $DATA_DIR/view.log"
    fi
fi
info "配置   : $( [ -f "$CTL" ] && tr '\n' ' ' < "$CTL" )"
log ""
mark "done"
log "看状态: $0 --status    停止: $0 --stop    重来: $0 --restart"
