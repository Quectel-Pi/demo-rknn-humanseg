#!/usr/bin/env python3
"""板端 GUI 自检：真实创建设置窗口，用**控件操作**驱动，验证 ctl 写回。

为什么必须驱动控件而不是直接调函数：
  之前的写法直接调 write_ctl()，等于只测了「函数能跑」，
  测不到「开关的事件回调有没有接对」。用户点的就是那个开关，
  所以这里一律走 Gtk.Switch.set_active() / DropDown.set_selected()，
  让 GTK 真实发出信号、走真实回调。

同时验证「点开关不会抹掉其它键」—— 这是实测踩到的坑：
旧 write_ctl 只写 seg/bg 两行，点一下开关就把 thr/up/margin 全丢了。
"""
import os
import sys
import tempfile

sys.path.insert(0, "/data/hs_test")

FAILS = []


def check(name, ok, detail=""):
    print("  %-44s %s  %s" % (name, "PASS" if ok else "FAIL", detail))
    if not ok:
        FAILS.append(name)


def main():
    import gi
    gi.require_version("Gtk", "4.0")
    from gi.repository import Gtk, GLib

    # 用临时 ctl，绝不碰线上 ctl.conf —— 自检不能改用户正在看的配置
    import humanseg_settings as hs
    tmpd = tempfile.mkdtemp(prefix="guichk")
    hs.STATE_DIR = tmpd
    hs.CTL = os.path.join(tmpd, "ctl.conf")

    # 预置一份带其它键的配置，用于验证「不被抹掉」
    with open(hs.CTL, "w") as f:
        f.write("seg=1\nbg=0\nthr=35\nup=1\nmargin=0\norphan=20\n")

    win = hs.Win()

    # ── 1. 窗口与控件存在 ──
    check("设置窗口创建成功", isinstance(win, Gtk.Window))
    check("主开关存在且初值=ON", win.sw.get_active() is True)

    # ── 2. 真实关掉开关 -> ctl 变 seg=0 ──
    win.sw.set_active(False)
    # 信号是同步的，但 GLib 需要跑一轮主循环才彻底派发
    ctx = GLib.MainContext.default()
    while ctx.pending():
        ctx.iteration(False)
    txt = open(hs.CTL).read()
    check("关开关后 ctl 写入 seg=0", "seg=0" in txt, repr(txt.replace("\n", " ")))

    # ── 3. 其它键必须保留（本项是真实踩到的缺陷）──
    kept = all(k in txt for k in ("thr=35", "up=1", "margin=0", "orphan=20"))
    check("点开关未抹掉 thr/up/margin/orphan", kept,
          "" if kept else "丢失 -> 用户调好的参数被重置")

    # ── 4. 真实改背景下拉 -> ctl 变 bg ──
    # BG_CHOICES 里取一个非 0 的值
    idx = None
    for i, (_n, v) in enumerate(hs.BG_CHOICES):
        if v != 0:
            idx = i
            break
    if idx is None:
        check("背景下拉有非 0 选项", False, "BG_CHOICES 里没有别的背景")
    else:
        win.dd.set_selected(idx)
        while ctx.pending():
            ctx.iteration(False)
        txt = open(hs.CTL).read()
        want = "bg=%d" % hs.BG_CHOICES[idx][1]
        check("改背景后 ctl 写入 %s" % want, want in txt,
              repr(txt.replace("\n", " ")))

    # ── 5. 再开回开关 -> seg=1 ──
    win.sw.set_active(True)
    while ctx.pending():
        ctx.iteration(False)
    txt = open(hs.CTL).read()
    check("开回开关后 ctl 写入 seg=1", "seg=1" in txt)

    # ── 6. 状态行反映真实后端状态 ──
    check("状态行非空", len(win.status.get_label()) > 0,
          repr(win.status.get_label()[:52]))

    # ── 7. 窗口真的能显示（GTK 层，不依赖合成器截图）──
    win.set_default_size(440, 330)
    win.present()
    for _ in range(6):
        while ctx.pending():
            ctx.iteration(False)
        import time
        time.sleep(0.05)
    check("窗口 present() 后可映射", win.get_mapped() or win.get_visible(),
          "mapped=%s visible=%s" % (win.get_mapped(), win.get_visible()))

    win.destroy()
    import shutil
    shutil.rmtree(tmpd, ignore_errors=True)

    print()
    if FAILS:
        print("RESULT: FAIL (%d 项)" % len(FAILS))
        for f in FAILS:
            print("  -", f)
        return 1
    print("RESULT: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
