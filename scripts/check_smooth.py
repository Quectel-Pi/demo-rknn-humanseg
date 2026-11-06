#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""测两件事，都要求有数字：
  1. 帧是否连续（不再出现 seq 停滞 = 画面卡住）
  2. 打开面板/切开关时，GUI 主循环最大停顿多少毫秒
停顿用 GLib 的 tick 间隔衡量：正常 16ms 一档，超过 200ms 就算明显卡。
"""
import sys
import time

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Gdk", "4.0")
from gi.repository import GLib, Gtk  # noqa: E402

sys.path.insert(0, "/data/hs_test")
import humanseg_view as hv  # noqa: E402

gaps = []
last = [None]
fails = []


def sampler():
    now = time.time()
    if last[0] is not None:
        gaps.append((now - last[0]) * 1000.0)
    last[0] = now
    return True


def run(win, loop):
    try:
        st = {"n0": win.frames.seq, "t0": time.time()}

        # 阶段 1：静止 2s，看帧有没有断
        time.sleep(2.0)
        f1 = win.frames.seq
        dt = time.time() - st["t0"]
        fps = (f1 - st["n0"]) / dt
        print("静止 2s: seq %d -> %d  (%.1f fps)" % (st["n0"], f1, fps))
        if fps < 20:
            fails.append("帧率过低 %.1f fps（画面会卡）" % fps)

        # 阶段 2：反复开关面板 + 切背景，同时测主循环停顿
        gaps.clear()
        t = GLib.timeout_add(16, sampler)
        for _ in range(6):
            win.gear.set_active(True)
            win.gear.get_popover().popup()
            time.sleep(0.25)
            win.gear.set_active(False)
            time.sleep(0.15)
            win.panel.dd.set_selected(1)
            time.sleep(0.2)
            win.panel.dd.set_selected(0)
            time.sleep(0.15)
        GLib.source_remove(t)
        print("面板操作 6 轮完成")

        if gaps:
            gaps.sort()
            n = len(gaps)
            p50 = gaps[n // 2]
            p99 = gaps[min(n - 1, int(n * 0.99))]
            mx = gaps[-1]
            print("主循环间隔: p50=%.0fms p99=%.0fms max=%.0fms (样本 %d)"
                  % (p50, p99, mx, n))
            if mx > 200:
                fails.append("主循环停顿 %.0fms，点按钮会明显卡" % mx)
            elif mx > 120:
                print("  注意: 最大停顿 %.0fms，属于偶发轻微卡顿" % mx)

        # 阶段 3：操作期间帧还在动吗
        a = win.frames.seq
        time.sleep(1.5)
        b = win.frames.seq
        print("操作后 1.5s 帧前进 %d" % (b - a))
        if b - a < 20:
            fails.append("操作后画面停住了（只前进 %d 帧）" % (b - a))
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
GLib.timeout_add(2500, lambda: (run(win, loop), False)[1])
GLib.timeout_add(60000, lambda: (print("timeout"), loop.quit(), False)[2])
loop.run()
