#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""端到端：动设置面板的控件 -> 看共享内存里的画面真的变了。

这是唯一能证明"按钮不是摆设"的检查。不看截图，直接比对像素。
背景色（BGR）：Teal [170,178,32]，Gray [90,90,90]。
"""
import mmap
import os
import struct
import sys
import time

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Gdk", "4.0")
from gi.repository import GLib, Gtk  # noqa: E402

sys.path.insert(0, "/data/hs_test")
import humanseg_view as hv  # noqa: E402

SHM = "/dev/shm/humanseg.frames"
HDR = 16
CTL = "/data/hs_test/ctl.conf"
TEAL = (170, 178, 32)      # BGR，与 humanseg_y8seg.cc: bb=170,bgc=178,br=32 一致
fails = []


def grab(settle=1.2):
    """等一会儿让后端处理新设置，再读一帧。

    必须确认读到的是**新**帧：只 sleep 固定时间，在高负载下可能仍读到
    改配置之前的那一帧，导致「改了没生效」被误判成「生效了」或反之。
    这里比对两次 seq，确保帧序号在前进。
    """
    deadline = time.time() + settle
    last = None
    while time.time() < deadline:
        fd = os.open(SHM, os.O_RDONLY)
        try:
            size = os.fstat(fd).st_size
            mm = mmap.mmap(fd, size, mmap.MAP_SHARED, mmap.PROT_READ)
            try:
                seq = struct.unpack_from("<I", mm, 0)[0]
            finally:
                mm.close()
        finally:
            os.close(fd)
        if last is not None and seq != last:
            break
        last = seq
        time.sleep(0.15)

    fd = os.open(SHM, os.O_RDONLY)
    try:
        size = os.fstat(fd).st_size
        mm = mmap.mmap(fd, size, mmap.MAP_SHARED, mmap.PROT_READ)
        _seq, w, h, _ = struct.unpack_from("<IIII", mm, 0)
        n = w * h * 3
        px = mm[HDR:HDR + n]
        mm.close()
    finally:
        os.close(fd)
    return w, h, px


def teal_ratio(px):
    """青色像素占比（抠像生效时背景应该是青色）。"""
    tot = 0
    hit = 0
    step = 3 * 37                      # 采样步长，够快够准
    for i in range(0, len(px) - 3, step):
        tot += 1
        b, g, r = px[i], px[i + 1], px[i + 2]
        if abs(b - TEAL[0]) < 30 and abs(g - TEAL[1]) < 30 and abs(r - TEAL[2]) < 30:
            hit += 1
    return hit / tot if tot else 0.0


def ctl():
    return hv.read_ctl()


def check(label, cond, detail):
    print("  %-44s %s   %s" % (label, "OK" if cond else "NG", detail))
    if not cond:
        fails.append("%s: %s" % (label, detail))


def run_scenarios():
    try:
        # 基准：seg=1 bg=0 -> 应有大量青色
        hv.write_ctl(1, 0)
        w, h, px = grab()
        r = teal_ratio(px)
        check("seg=1 bg=0 背景是青色", r > 0.25, "青色占比=%.1f%%" % (100 * r))

        # 关分割 -> 纯相机画面，青色应几乎归零
        hv.write_ctl(0, 0)
        w, h, px = grab()
        r0 = teal_ratio(px)
        check("seg=0 青色消失(纯相机)", r0 < 0.02, "青色占比=%.2f%%" % (100 * r0))

        # 换背景 gray -> 青色应为 0
        hv.write_ctl(1, 2)
        w, h, px = grab()
        r2 = teal_ratio(px)
        check("seg=1 bg=2 青色归零", r2 < 0.02, "青色占比=%.2f%%" % (100 * r2))

        # 恢复
        hv.write_ctl(1, 0)
        w, h, px = grab()
        r3 = teal_ratio(px)
        check("恢复 seg=1 bg=0", r3 > 0.25, "青色占比=%.1f%%" % (100 * r3))
    except Exception as e:
        print("EXC:", e)
        fails.append(str(e))


def gui_path(win, loop):
    """走界面控件（不是直接写文件），确认 UI 真的接通了。"""
    try:
        p = win.panel
        p.reload()

        # 1) 关掉开关
        p.sw.set_active(False)
        time.sleep(1.4)
        seg, _bg = ctl()
        check("UI 开关关 -> ctl seg=0", seg == 0, "ctl seg=%d" % seg)
        _w, _h, px = grab(0.4)
        check("UI 关 -> 画面无青色", teal_ratio(px) < 0.02,
              "青色=%.2f%%" % (100 * teal_ratio(px)))

        # 2) 打开开关 + 选绿色背景
        p.sw.set_active(True)
        p.dd.set_selected(3)
        time.sleep(1.4)
        seg, bg = ctl()
        check("UI 开关开 + 选 Green", seg == 1 and bg == 3,
              "ctl seg=%d bg=%d" % (seg, bg))

        # 3) 回到青色
        p.dd.set_selected(0)
        time.sleep(1.4)
        seg, bg = ctl()
        check("UI 选回 Teal", seg == 1 and bg == 0, "ctl seg=%d bg=%d" % (seg, bg))
        _w, _h, px = grab(0.4)
        check("UI 改完 -> 画面回到青色", teal_ratio(px) > 0.25,
              "青色=%.1f%%" % (100 * teal_ratio(px)))
    except Exception as e:
        print("EXC:", e)
        fails.append(str(e))

    print()
    print("RESULT:", "PASS" if not fails else "FAIL")
    for f in fails:
        print("  -", f)
    loop.quit()


def main():
    global loop
    loop = GLib.MainLoop()
    print("== 直接写控制文件（后端链路）==")
    run_scenarios()
    hv.write_ctl(1, 0)

    print()
    print("== 走界面控件（UI 链路）==")
    win = hv.View()          # 复用真实窗口，自带 panel
    GLib.timeout_add(1500, lambda: (gui_path(win, loop), False)[1])
    GLib.timeout_add(45000, lambda: (print("timeout"), loop.quit(), False)[2])
    loop.run()
    return 1 if fails else 0


# 只在作为脚本运行时才跑。
# 旧的顶层执行让本文件无法被 import —— 一 import 就抢真实 SHM/ctl 跑全流程，
# 验证脚本想复用 teal_ratio()/grab() 只能复制代码，等于测副本。
if __name__ == "__main__":
    sys.exit(main())
