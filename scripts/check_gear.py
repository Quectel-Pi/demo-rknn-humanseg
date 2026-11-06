#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""验证右上角齿轮按钮的真实位置和可点击性。

不看截图（Wayland 拒绝），直接用 GTK 的坐标系统问：
  1. 齿轮是不是在窗口右上角（坐标、对齐、边距）
  2. 齿轮是否真的在窗口之上（overlay 关系）
  3. 用 pick() 在齿轮坐标上做命中测试，看命中的是不是它
  4. 点一下看 popover 是否弹出，里面有 Switch / DropDown
"""
import sys

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Gdk", "4.0")
from gi.repository import GLib, Gtk  # noqa: E402

sys.path.insert(0, "/data/hs_test")
import humanseg_view as hv  # noqa: E402

fails = []


def check(win, loop):
    try:
        gear = win.gear
        w = win.get_width()
        h = win.get_height()
        print("窗口尺寸: %dx%d" % (w, h))

        # 齿轮相对窗口的坐标
        ok, b = gear.compute_bounds(win)
        print("齿轮 bounds: ok=%s x=%.1f y=%.1f w=%.1f h=%.1f"
              % (ok, b.get_x(), b.get_y(), b.get_width(), b.get_height()))

        if not ok:
            fails.append("拿不到齿轮坐标")
        else:
            gx, gy = b.get_x(), b.get_y()
            gw, gh = b.get_width(), b.get_height()
            # 右上角判据：x 靠右、y 靠上
            if gx < w * 0.5:
                fails.append("齿轮不在右半区 (x=%.0f, w=%d)" % (gx, w))
            if gy > h * 0.25:
                fails.append("齿轮不在上方 (y=%.0f, h=%d)" % (gy, h))
            # 离右边/上边各留了 16px 边距
            right_gap = w - (gx + gw)
            print("右边距=%.0f  上边距=%.0f" % (right_gap, gy))
            if not (5 <= right_gap <= 40):
                fails.append("右边距异常 %.0f" % right_gap)
            if not (5 <= gy <= 40):
                fails.append("上边距异常 %.0f" % gy)

        # 命中测试：在齿轮中心 pick，必须命中齿轮（或齿轮内部的图标）
        if ok:
            cx, cy = int(gx + gw / 2), int(gy + gh / 2)
            target = win.pick(cx, cy, Gtk.PickFlags.DEFAULT)
            # MenuButton 内部包着 Image（图标），命中它等于命中按钮本身，
            # 所以沿父链往上找，看齿轮是否在祖先链里。
            hit_gear = False
            node = target
            while node is not None:
                if node is gear:
                    hit_gear = True
                    break
                node = node.get_parent()
            print("pick(%d,%d) -> %s  属于齿轮=%s"
                  % (cx, cy, type(target).__name__ if target else None, hit_gear))
            if not hit_gear:
                fails.append("齿轮位置被画面遮挡/命中不到 (命中 %s)"
                             % (type(target).__name__ if target else "None"))

        # 点开 popover。注意：GtkMenuButton.set_active(True) 在无事件循环
        # 交互时不一定触发弹出，直接调 popover.popup() 更接近真实点击结果。
        gear.set_active(True)
        p = gear.get_popover()
        p.popup()
        GLib.timeout_add(900, lambda: after_open(win, loop))
    except Exception as e:
        print("EXC:", e)
        fails.append(str(e))
        loop.quit()


def after_open(win, loop):
    try:
        p = win.gear.get_popover()
        vis = p.get_visible()
        print("popover 可见=%s" % vis)
        if not vis:
            fails.append("点齿轮没弹出设置面板")

        # 面板里的控件
        def walk(w, out):
            out.append(w)
            c = w.get_first_child()
            while c is not None:
                walk(c, out)
                c = c.get_next_sibling()

        nodes = []
        walk(p, nodes)
        kinds = {}
        for n in nodes:
            kinds[type(n).__name__] = kinds.get(type(n).__name__, 0) + 1
        print("面板控件:", kinds)

        if "Switch" not in kinds:
            fails.append("面板里没有分割开关")
        if "DropDown" not in kinds:
            fails.append("面板里没有背景下拉")

        texts = [n.get_label() for n in nodes
                 if isinstance(n, Gtk.Label) and n.get_label()]
        print("面板文字:", texts)
        if any("\u4e00" <= ch <= "\u9fff" for t in texts for ch in t):
            fails.append("面板里有中文")
    except Exception as e:
        print("EXC:", e)
        fails.append(str(e))

    print()
    print("RESULT:", "PASS" if not fails else "FAIL")
    for f in fails:
        print("  -", f)
    loop.quit()


loop = GLib.MainLoop()
win = hv.View()
win.present()
GLib.timeout_add(2000, lambda: (check(win, loop), False)[1])
GLib.timeout_add(20000, lambda: (print("timeout"), loop.quit(), False)[2])
loop.run()
