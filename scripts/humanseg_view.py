#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""人像分割输出界面 —— 画面 + 右上角齿轮设置按钮（GTK4 全屏）。

为什么是这个结构：
  Wayland 下应用不能自己决定窗口位置，板上也没有 layer-shell，
  所以**没法**把按钮浮在 waylandsink 的视频上面。唯一能让按钮
  真正压在画面右上角的办法，就是让画面本身画在 GTK4 窗口里。
  后端 C++ 通过 --shm 把每帧像素丢过来，本进程只负责上屏。

帧协议（与 humanseg_y8seg.cc 的 shm_publish 对应）：
  offset 0   uint32 seq   递增序号（变了 = 有新帧）
  offset 4   uint32 w
  offset 8   uint32 h
  offset 12  uint32 pad
  offset 16  BGR 像素 w*h*3
"""
import mmap
import os
import struct
import sys
import time

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Gdk", "4.0")
from gi.repository import Gdk, GLib, Gtk  # noqa: E402

SHM = "/dev/shm/humanseg.frames"
CTL = "/data/hs_test/ctl.conf"
HDR = 16
BG_NAMES = ["Teal (default)", "Dimmed camera", "Solid gray", "Green screen"]

# 样式表。必须在这里定义、由 View.__init__ 安装，不能只在 main() 里装：
# 任何其它入口（测试脚本、别的进程 import 本模块）如果绕过 main()，
# 就会拿到一套**完全没样式**的控件 —— 齿轮和面板都是主题默认的白底，
# 而文字又继承了浅色，结果就是白底白字看不清。
# background-image: none 是必须的：Adwaita 的主题底色就是用
# background-image 画的，只设 background-color 会被它盖住。
_CSS = b"""
.humanseg-gear {
    background-image: none;
    background-color: rgba(0, 0, 0, 0.55);
    color: #ffffff;
    min-width: 40px;
    min-height: 40px;
    box-shadow: none;
    text-shadow: none;
}
.humanseg-gear:hover {
    background-image: none;
    background-color: rgba(0, 0, 0, 0.80);
    color: #ffffff;
}
.humanseg-gear image {
    color: #ffffff;
}
.humanseg-panel {
    background-image: none;
    background-color: rgba(24, 24, 28, 0.96);
    border-radius: 12px;
    padding: 6px;
    box-shadow: 0 6px 24px rgba(0, 0, 0, 0.55);
}
/* All text inside the panel: force light, per-class so the theme
   cannot win on specificity. */
.humanseg-panel label,
.humanseg-panel .label,
.humanseg-panel .title-4,
.humanseg-panel .dim-label,
.humanseg-panel button {
    color: #f2f2f4;
}
.humanseg-panel .dim-label {
    color: rgba(242, 242, 244, 0.70);
}
.humanseg-panel button {
    background-image: none;
    background-color: rgba(255, 255, 255, 0.14);
    box-shadow: none;
}
.humanseg-panel button:hover {
    background-color: rgba(255, 255, 255, 0.24);
}
/* DropDown search entry: without this it inherits the theme's dark text
   and disappears against our dark backdrop. */
.humanseg-panel entry,
.humanseg-panel entry text {
    color: #f2f2f4;
    background-image: none;
    background-color: rgba(255, 255, 255, 0.10);
    caret-color: #f2f2f4;
}
.humanseg-panel entry > text > placeholder {
    color: rgba(242, 242, 244, 0.55);
}
/* The dropdown list is a separate popover surface and the theme renders it
   light by default; our light item text would vanish on it. Force it dark. */
.humanseg-panel popover,
.humanseg-panel popover > contents,
.humanseg-panel popover listview,
.humanseg-panel popover scrolledwindow {
    background-image: none;
    background-color: #1a1a1e;
    color: #f2f2f4;
}
.humanseg-panel popover row,
.humanseg-panel popover row label {
    color: #f2f2f4;
}
.humanseg-panel popover row:selected,
.humanseg-panel popover row:selected label {
    background-color: rgba(255, 255, 255, 0.18);
    color: #ffffff;
}
"""

_css_installed = False


def install_css():
    """幂等地把 _CSS 挂到 display 上。"""
    global _css_installed
    if _css_installed:
        return
    disp = Gdk.Display.get_default()
    if disp is None:
        return
    prov = Gtk.CssProvider()
    prov.load_from_data(_CSS)
    Gtk.StyleContext.add_provider_for_display(
        disp, prov, Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION)
    _css_installed = True


# ── 控制文件 ───────────────────────────────────────────────────────
def read_ctl():
    seg, bg = 1, 0
    try:
        with open(CTL) as f:
            for line in f:
                line = line.replace(" ", "")
                if line.startswith("seg="):
                    seg = 0 if line[4:].strip() in ("0", "") else 1
                elif line.startswith("bg="):
                    try:
                        v = int(line[3:].strip())
                        if 0 <= v <= 3:
                            bg = v
                    except ValueError:
                        pass
    except FileNotFoundError:
        pass
    return seg, bg


def write_ctl(seg, bg):
    """只改 seg/bg，**保留 ctl.conf 里的其它键**（thr/up/margin/orphan/dump）。

    旧实现只写 "seg=%d\\nbg=%d\\n" 两行，用户在面板上点一下开关，
    现场调好的 thr=35、up=1 等参数就被整段抹掉，画面悄悄退回默认值
    —— 表现为「点一下开关，轮廓又变差了」。这是本函数的唯一职责边界：
    它没有资格丢弃自己不认识的键。

    用唯一临时名：固定 .tmp 在多个写入方（界面 / 验证脚本）并发时
    会互相 rename 掉，实测报 FileNotFoundError 导致设置写不进去。
    """
    keep = []
    try:
        with open(CTL) as f:
            for line in f:
                s = line.strip()
                if not s or s.startswith("#"):
                    continue
                k = s.split("=", 1)[0].strip()
                # 自己要写的键由新值接管，其余原样保留
                if k in ("seg", "bg"):
                    continue
                keep.append(s)
    except FileNotFoundError:
        pass

    tmp = "%s.%d.tmp" % (CTL, os.getpid())
    with open(tmp, "w") as f:
        f.write("seg=%d\nbg=%d\n" % (seg, bg))
        for s in keep:
            f.write(s + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, CTL)


def backend_pids():
    out = []
    for name in os.listdir("/proc"):
        if not name.isdigit():
            continue
        try:
            with open("/proc/%s/comm" % name) as f:
                if f.read().strip() == "hy8":
                    out.append(int(name))
        except OSError:
            pass
    return out


# ── 共享内存取帧 ───────────────────────────────────────────────────
class Frames:
    """映射后端发布的帧缓冲。后端重启会重建文件，需重新映射。"""

    def __init__(self):
        self.mm = None
        self.fd = None
        self.seq = 0
        self.ino = -1
        self.w = self.h = 0
        self.buf = None

    def attach(self):
        try:
            self.fd = os.open(SHM, os.O_RDONLY)
        except OSError:
            return False
        size = os.fstat(self.fd).st_size
        if size <= HDR:
            self.close()
            return False
        self.mm = mmap.mmap(self.fd, size, mmap.MAP_SHARED, mmap.PROT_READ)
        # 头部先读一次拿尺寸
        seq, w, h, _ = struct.unpack_from("<IIII", self.mm, 0)
        if w == 0 or h == 0 or size < HDR + w * h * 3:
            self.close()
            return False
        self.w, self.h = w, h
        self.seq = seq
        self.ino = os.fstat(self.fd).st_ino
        self.buf = bytearray(w * h * 3)
        return True

    def _stale(self):
        """当前 SHM 路径是否已指向另一个文件（后端重启会 rm 后重建）。

        后端重启时 up.sh 会 `rm -f /dev/shm/humanseg.frames` 再新建，
        本进程仍 mmap 着**已删除的旧 inode** —— 读得到数据、seq 却永远
        停住，表现就是画面完全冻结而进程还活着（2026-11-06 线上事故）。
        """
        try:
            return os.stat(SHM).st_ino != self.ino
        except OSError:
            return True          # 文件暂时不存在也算失效，下一帧重试

    def close(self):
        if self.mm is not None:
            self.mm.close()
            self.mm = None
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None

    def latest(self):
        """拿最新一帧。返回 True 表示有更新。

        注意这里刻意不复用自建 bytearray：原来每帧 bytes(self.buf) 会额外
        分配 3MB（1280*800*3），28fps 就是 84MB/s 的分配量，Python GC
        压力大，点按钮时容易撞上停顿。现在只做一次切片拷贝。
        """
        if self.mm is None and not self.attach():
            return False
        # 后端重启会 rm+重建 shm，本进程还挂在旧 inode 上 —— 此时必须重挂，
        # 否则画面永久冻结（进程活着但 seq 再也不动）。
        if self._stale():
            self.close()
            if not self.attach():
                return False
        # 双检：两次序号相同才算读完整（后端一边写我们一边拷会撕裂）
        n = self.w * self.h * 3
        for _ in range(3):
            s0 = struct.unpack_from("<I", self.mm, 0)[0]
            if s0 == self.seq:
                return False                     # 没有新帧
            # mmap 没有 readinto，用切片拿到 bytes（一次拷贝）
            self.buf = self.mm[HDR:HDR + n]
            s1 = struct.unpack_from("<I", self.mm, 0)[0]
            if s0 == s1:
                self.seq = s0
                return True
        return False                             # 一直撕裂，下次再试

    def texture(self):
        # BGR -> 内存纹理（GDK 认 B8G8R8A8，我们只喂 3 字节紧凑 BGR，
        # 用 B8G8R8 对应的 strided 形式即可，GTK 会自己补齐）
        b = GLib.Bytes.new(self.buf)
        return Gdk.MemoryTexture.new(
            self.w, self.h, Gdk.MemoryFormat.B8G8R8, b, self.w * 3)


# ── 右上角齿轮展开的设置面板 ───────────────────────────────────────
class SettingsPanel(Gtk.Box):
    """点齿轮才出现。里面只做两件事：开/关分割、选背景。

    基类是普通 Gtk.Box 而不是 Gtk.Popover：popover 在 Wayland 下是独立
    xdg_popup 表面 + 隐式 grab，关闭时那次点击会穿透到下面，导致
    "关了又被齿轮点开"。内嵌 Box 没有独立表面，可见性完全由我们控制。
    """

    def __init__(self, on_change):
        super().__init__(orientation=Gtk.Orientation.VERTICAL, spacing=12)
        self._on_change = on_change
        for side in ("top", "bottom", "start", "end"):
            getattr(self, "set_margin_" + side)(16)

        title = Gtk.Label(label="Portrait Segmentation", xalign=0)
        title.add_css_class("title-4")
        self.append(title)

        row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=12)
        lbl = Gtk.Label(label="Enable segmentation", xalign=0, hexpand=True)
        row.append(lbl)
        self.sw = Gtk.Switch(valign=Gtk.Align.CENTER)
        self.sw.connect("state-set", self._seg_changed)
        row.append(self.sw)
        self.append(row)

        self.append(Gtk.Separator())

        self.append(Gtk.Label(label="Virtual background", xalign=0))
        self.dd = Gtk.DropDown.new_from_strings(BG_NAMES)
        self.dd.connect("notify::selected", self._bg_changed)
        self.append(self.dd)

        self.status = Gtk.Label(xalign=0)
        self.status.add_css_class("dim-label")
        self.append(self.status)

        # 后端没起来时给个启动按钮
        self.startbtn = Gtk.Button(label="Start backend")
        self.startbtn.connect("clicked", self._start)
        self.append(self.startbtn)

        self.reload()

    def reload(self):
        seg, bg = read_ctl()
        self.sw.set_active(bool(seg))
        self.dd.set_selected(bg)
        alive = bool(backend_pids())
        self.startbtn.set_visible(not alive)
        self.status.set_label(
            ("Backend running · background: %s" % BG_NAMES[bg]) if alive
            else "Backend not running")

    def _seg_changed(self, sw, state):
        _seg, bg = read_ctl()
        write_ctl(1 if state else 0, bg)
        self.reload()
        self._on_change()
        return False

    def _bg_changed(self, dd, _p):
        seg, _bg = read_ctl()
        write_ctl(seg, dd.get_selected())
        self.reload()
        self._on_change()

    def _start(self, _b):
        # 幂等：已经在跑就绝不再起一个。重复启动会抢摄像头，表现就是画面卡住不动。
        if backend_pids():
            self.reload()
            return
        # 用 spawn_async 而不是 subprocess：subprocess 会 fork+exec 并可能
        # 在 fork 瞬间复制整个 3MB 级缓冲，GUI 线程上做这个会明显卡一下。
        try:
            GLib.spawn_async(
                argv=["/bin/sh", "-c", LAUNCH],
                flags=GLib.SpawnFlags.SEARCH_PATH
                | GLib.SpawnFlags.DO_NOT_REAP_CHILD,
                child_setup=None, user_data=None)
        except GLib.Error as e:
            print("spawn failed:", e, flush=True)
        GLib.timeout_add(2500, lambda: (self.reload(), False)[1])


LAUNCH = (
    "cd /data/hs_test && setsid env "
    "XDG_RUNTIME_DIR=/run/user/1000 WAYLAND_DISPLAY=wayland-0 "
    "LD_LIBRARY_PATH=/data/hs_test ./hy8 ./y8s_rkv.rknn "
    "--device /dev/video73 --cam 1280x720 --disp 1280x800 "
    "--fps --ctl /data/hs_test/ctl.conf --shm /dev/shm/humanseg.frames "
    "> /data/hs_test/live.log 2>&1 < /dev/null &"
)


# ── 主窗口：画面 + 右上角齿轮 ──────────────────────────────────────
class View(Gtk.Window):
    def __init__(self):
        super().__init__(title="Portrait Segmentation")
        # 在这里装样式：任何入口构造 View 都要有正确配色，
        # 不能只在 main() 里装（否则测试脚本/其它进程看到的是白底白字）。
        install_css()
        # 无边框全屏：画面铺满，不出现标题栏/关闭按钮（Wayland 由合成器定位置，
        # 这里只负责窗口内容；要真正全屏需窗口管理器配合）。
        self.set_decorated(False)
        self.fullscreen()

        self.frames = Frames()

        over = Gtk.Overlay()
        self.set_child(over)

        # 画面层
        self.pic = Gtk.Picture(hexpand=True, vexpand=True,
                               content_fit=Gtk.ContentFit.CONTAIN)
        over.set_child(self.pic)

        # 右上角齿轮 + 它弹的设置面板
        # 用普通 Button + 窗口内嵌面板，**刻意不用 Gtk.MenuButton/Gtk.Popover**：
        # Wayland 下 popover 是独立的 xdg_popup 表面并带隐式 grab，
        # 关闭它的那次点击会穿透到下面（齿轮），于是"关了又立刻开"，
        # 表现就是弹窗永远收不回去。补 popdown()/手势都治不了根。
        # 内嵌成 Overlay 的一个普通子控件就没有独立表面、没有 grab，
        # 齿轮的点击是唯一入口，点一次开、再点一次关，行为完全确定。
        self.gear = Gtk.Button()
        self.gear.set_icon_name("emblem-system-symbolic")
        self.gear.add_css_class("circular")
        self.gear.add_css_class("humanseg-gear")
        self.gear.set_halign(Gtk.Align.END)
        self.gear.set_valign(Gtk.Align.START)
        self.gear.set_margin_top(16)
        self.gear.set_margin_end(16)
        self.gear.set_tooltip_text("Settings")
        self.gear.connect("clicked", self._toggle_panel)
        over.add_overlay(self.gear)

        self.panel = SettingsPanel(self._on_setting_changed)
        self.panel.set_halign(Gtk.Align.END)
        self.panel.set_valign(Gtk.Align.START)
        self.panel.set_margin_top(72)      # 让开齿轮
        self.panel.set_margin_end(16)
        self.panel.add_css_class("humanseg-panel")
        self.panel.set_visible(False)
        over.add_overlay(self.panel)

        # 点画面关面板 / Esc 关面板
        click = Gtk.GestureClick()
        click.connect("pressed", self._close_panel)
        self.pic.add_controller(click)

        key = Gtk.EventControllerKey()
        key.connect("key-pressed", self._on_key)
        self.add_controller(key)

        # 每帧检查共享内存有没有新画面；没有就保持上一帧，不刷新纹理
        self._n = 0
        self._t0 = 0.0
        self._tick = GLib.timeout_add(16, self._pull)   # ~60Hz 轮询
        GLib.timeout_add(1500, self._refresh_panel)

    def _toggle_panel(self, *_a):
        if self.panel.get_visible():
            self.panel.set_visible(False)
        else:
            self.panel.reload()
            self.panel.set_visible(True)

    def _close_panel(self, *_a):
        if self.panel.get_visible():
            self.panel.set_visible(False)

    def _on_key(self, _c, keyval, _code, _state):
        if keyval == Gdk.KEY_Escape and self.panel.get_visible():
            self._close_panel()
            return True
        return False

    def _pull(self):
        if self.frames.latest():
            self.pic.set_paintable(self.frames.texture())
            # 打点：证明画面真的在更新（而不是停在白屏上）
            if self._n == 0:
                self._t0 = time.time()
            self._n += 1
            if self._n % 120 == 0:
                dt = time.time() - self._t0
                print("painted %d frames (%.1f fps) seq=%d"
                      % (self._n, self._n / dt if dt else 0, self.frames.seq),
                      flush=True)
        return True

    def _refresh_panel(self):
        # 后端活着/死掉要能反映出来，但没必要每 1.5s 扫一遍 /proc——
        # 只在面板打开、或状态行还显示"未运行"时检查（那时才需要发现它起来）。
        if self.panel.get_visible() or not backend_pids():
            self.panel.reload()
        return True

    def _on_setting_changed(self):
        pass        # 后端自己轮询 ctl.conf，这里无需通知


def main():
    # 不用 Gtk.Application：它会尝试在 session bus 注册唯一名，
    # `sudo -u qq` 下实测拿不到可用 DBUS，报 NoReply 后窗口不出现。
    loop = GLib.MainLoop()
    win = View()
    win.connect("close-request", lambda *a: (loop.quit(), False)[1])

    win.present()
    loop.run()


if __name__ == "__main__":
    main()
