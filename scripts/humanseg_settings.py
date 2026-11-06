#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""人像分割 / 虚拟背景 —— 桌面设置开关（GTK4）

跟后端进程的分工：
  * 本程序只做一件事：把 seg / bg 写进一个控制文件（原子替换）。
  * hy8 显示线程每帧检查该文件 mtime，变了就重读并立即生效，
    **不需要重启进程**，所以开关是即时的。

控制文件格式（后端 ctl_poll() 解析，宽松：顺序任意、缺项保留原值）：
    seg=1
    bg=0
"""
import os
import subprocess
import sys

import gi

gi.require_version("Gtk", "4.0")
from gi.repository import GLib, Gtk  # noqa: E402

# ── 路径 ────────────────────────────────────────────────────────────
STATE_DIR = "/data/hs_test"
CTL = os.path.join(STATE_DIR, "ctl.conf")
LOG = os.path.join(STATE_DIR, "live.log")

# 后端启动命令（与常驻脚本保持一致）
LAUNCH = (
    "cd /data/hs_test && exec env "
    "XDG_RUNTIME_DIR=/run/user/1000 WAYLAND_DISPLAY=wayland-0 "
    "LD_LIBRARY_PATH=/data/hs_test "
    "./hy8 ./y8s_rkv.rknn --device /dev/video73 "
    "--cam 1280x720 --disp 1280x800 --fps --prof --ctl /data/hs_test/ctl.conf"
)

BG_CHOICES = [
    ("Teal (default)", 0),
    ("Dimmed camera", 1),
    ("Solid gray", 2),
    ("Green screen", 3),
]


def running():
    """后端是否在跑。"""
    try:
        out = subprocess.run(["pgrep", "-x", "hy8"], capture_output=True, text=True)
        return bool(out.stdout.strip())
    except Exception:
        return False


def read_ctl():
    seg, bg = 1, 0
    try:
        with open(CTL) as f:
            for line in f:
                line = line.replace(" ", "")
                if line.startswith("seg="):
                    seg = 1 if line[4:].strip() not in ("0", "") else 0
                elif line.startswith("bg="):
                    try:
                        bg = int(line[3:].strip())
                    except ValueError:
                        pass
    except FileNotFoundError:
        pass
    return seg, bg


def write_ctl(seg, bg):
    """原子写：先写临时文件再 rename，避免后端读到半截内容。

    只改 seg/bg，**保留其它键**（thr/up/margin/orphan/dump）。
    早期版本只写这两行，结果用户在面板上点一下开关就把
    thr=35、up=1 等现场调好的参数一起抹掉，画面悄悄退回默认值
    —— 表现为「点了下开关，轮廓又变差了」。
    """
    os.makedirs(STATE_DIR, exist_ok=True)
    keep = []
    try:
        with open(CTL) as f:
            for line in f:
                s = line.strip()
                if not s or s.startswith("#"):
                    continue
                k = s.split("=", 1)[0].strip()
                if k in ("seg", "bg"):
                    continue
                keep.append(s)
    except FileNotFoundError:
        pass
    tmp = CTL + ".tmp"
    with open(tmp, "w") as f:
        f.write("seg=%d\nbg=%d\n" % (seg, bg))
        for s in keep:
            f.write(s + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, CTL)


class Win(Gtk.Window):
    def __init__(self):
        super().__init__(title="Portrait Segmentation")
        self.set_default_size(440, 330)

        seg, bg = read_ctl()

        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=14)
        box.set_margin_top(22)
        box.set_margin_bottom(22)
        box.set_margin_start(22)
        box.set_margin_end(22)
        self.set_child(box)

        title = Gtk.Label(label="Portrait Segmentation · Virtual Background")
        title.add_css_class("title-2")
        title.set_halign(Gtk.Align.START)
        box.add_css_class("box")
        box.append(title)

        # ── 主开关 ──
        row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=12)
        lbl = Gtk.Label(label="Enable segmentation")
        lbl.set_xalign(0)
        lbl.set_hexpand(True)
        lbl.add_css_class("title-4")
        row.append(lbl)

        self.sw = Gtk.Switch()
        self.sw.set_active(bool(seg))
        self.sw.set_valign(Gtk.Align.CENTER)
        self.sw.connect("state-set", self.on_seg)
        row.append(self.sw)
        box.append(row)

        sub = Gtk.Label(label="When off, the raw camera image is shown (no keying)")
        sub.set_xalign(0)
        sub.add_css_class("dim-label")
        box.append(sub)

        box.append(Gtk.Separator())

        # ── 背景样式 ──
        blbl = Gtk.Label(label="Virtual background style")
        blbl.set_xalign(0)
        blbl.add_css_class("title-4")
        box.append(blbl)

        self.dd = Gtk.DropDown.new_from_strings([c[0] for c in BG_CHOICES])
        for i, (_, v) in enumerate(BG_CHOICES):
            if v == bg:
                self.dd.set_selected(i)
                break
        self.dd.connect("notify::selected", self.on_bg)
        box.append(self.dd)

        box.append(Gtk.Separator())

        # ── 状态行 ──
        self.status = Gtk.Label(label="")
        self.status.set_xalign(0)
        self.status.add_css_class("dim-label")
        box.append(self.status)

        # 后端没起来时给个按钮，否则开关点了也没画面
        self.startbtn = Gtk.Button(label="Start segmentation view")
        self.startbtn.connect("clicked", self.on_start)
        box.append(self.startbtn)

        self.refresh_status()
        GLib.timeout_add(1500, self.refresh_status)

    # ── 回调 ──
    def on_start(self, _btn):
        subprocess.Popen(
            ["bash", "-lc", LAUNCH],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        GLib.timeout_add(2000, self.refresh_status)

    def on_seg(self, sw, state):
        seg = 1 if state else 0
        bg = BG_CHOICES[self.dd.get_selected()][1]
        write_ctl(seg, bg)
        self.refresh_status()
        return False

    def on_bg(self, dd, _pspec):
        seg = 1 if self.sw.get_active() else 0
        bg = BG_CHOICES[dd.get_selected()][1]
        write_ctl(seg, bg)
        self.refresh_status()

    def refresh_status(self):
        alive = running()
        self.startbtn.set_visible(not alive)
        if alive:
            seg, bg = read_ctl()
            txt = "Backend running · segmentation %s · background: %s" % (
                "ON" if seg else "OFF",
                dict((v, n) for n, v in BG_CHOICES)[bg],
            )
        else:
            txt = "Backend not running — press the button below to start"
        self.status.set_label(txt)
        return True


def main():
    # 故意不用 Gtk.Application：它会尝试在 session bus 上注册唯一名，
    # 而 `sudo -u qq` 启动时实测拿不到可用的 DBUS 会话，报
    # org.freedesktop.DBus.Error.NoReply 后窗口根本不出现。
    # 这里只需要一个能显示的窗口，不需要单实例/DBus 激活。
    loop = GLib.MainLoop()
    win = Win()
    win.connect("close-request", lambda *a: (loop.quit(), False)[1])
    win.present()
    loop.run()


if __name__ == "__main__":
    main()
