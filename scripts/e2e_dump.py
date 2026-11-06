#!/usr/bin/env python3
"""板端 e2e 抓帧：从共享内存取**真正送去显示的像素**，按三组配置各存一盘。

为什么用 shm 而不是 --out 落盘：
  * --out 要重启后端（会打断用户正在看的画面），shm 是现成的
  * shm 里的像素就是 display_thread 交给 GTK 的那一份，等价于屏幕内容

布局（humanseg_y8seg.cc: shm_open_out）：
    16 字节头 (seq, w, h, pad) + w*h*3 字节 BGR

早期用 shell `dd` 逐帧读，每帧一次进程启动，36 帧就超时 —— 改 Python
一次 mmap 全程复用。

用法：python3 /data/hs_test/e2e_dump.py
产出：/data/hs_test/eOn, eOff, eGray  各 FRAMES 张 o_*.bgr
"""
import mmap
import os
import shutil
import struct
import sys
import time

BD = "/data/hs_test"
SHM = "/dev/shm/humanseg.frames"
CTL = os.path.join(BD, "ctl.conf")
FRAMES = 12
TIMEOUT = 8.0          # 每组最多等这么久


def read_ctl_lines():
    try:
        with open(CTL) as f:
            return f.read().splitlines()
    except FileNotFoundError:
        return []


def set_ctl(seg, bg):
    """只改 seg/bg，保留 thr/up/margin 等 —— 与 GUI 的 write_ctl 行为一致。"""
    keep = []
    for ln in read_ctl_lines():
        s = ln.strip()
        if not s or s.startswith("#"):
            continue
        k = s.split("=", 1)[0].strip()
        if k in ("seg", "bg"):
            continue
        keep.append(s)
    tmp = CTL + ".tmp"
    with open(tmp, "w") as f:
        f.write("seg=%d\nbg=%d\n" % (seg, bg))
        for s in keep:
            f.write(s + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, CTL)
    time.sleep(3)      # 后端每帧轮询 ctl，再等 mask 追上


def grab(fd, mm, outdir):
    if os.path.isdir(outdir):
        shutil.rmtree(outdir)
    os.makedirs(outdir)
    seq_off = 0
    last = None
    n = 0
    t0 = time.time()
    while n < FRAMES and time.time() - t0 < TIMEOUT:
        mm.seek(seq_off)
        seq = struct.unpack("<I", mm.read(4))[0]
        if seq == last:
            time.sleep(0.004)
            continue
        w = struct.unpack("<I", mm.read(4))[0]
        h = struct.unpack("<I", mm.read(4))[0]
        if w * h * 3 + 16 > len(mm):
            print("FAIL: shm 尺寸与头不符 w=%d h=%d" % (w, h))
            return 0
        mm.seek(16)
        px = mm.read(w * h * 3)
        fn = os.path.join(outdir, "o_%05d_%d_-1.bgr" % (n, seq))
        with open(fn, "wb") as f:
            f.write(px)
        last = seq
        n += 1
    return n


def main():
    if not os.path.exists(SHM):
        print("FAIL: 没有 %s（后端没跑？）" % SHM)
        return 1
    fd = os.open(SHM, os.O_RDONLY)
    mm = mmap.mmap(fd, 0, prot=mmap.PROT_READ)
    try:
        hdr = struct.unpack("<4I", mm[:16])
        print("shm: seq=%d %dx%d" % (hdr[0], hdr[1], hdr[2]))
        if hdr[1] == 0 or hdr[2] == 0:
            print("FAIL: shm 头无效")
            return 1

        backup = CTL + ".e2e.bak"
        try:
            shutil.copy(CTL, backup)
        except FileNotFoundError:
            backup = None

        for label, seg, bg, out in (
            ("seg=1 bg=0 (teal)", 1, 0, "eOn"),
            ("seg=0 bypass", 0, 0, "eOff"),
            ("seg=1 bg=2 (gray)", 1, 2, "eGray"),
        ):
            set_ctl(seg, bg)
            n = grab(fd, mm, os.path.join(BD, out))
            print("%-20s -> %s/%s  %d 帧" % (label, BD, out, n))
            if n < FRAMES:
                print("FAIL: %s 只抓到 %d/%d 帧" % (out, n, FRAMES))
                return 1
        return 0
    finally:
        if backup and os.path.exists(backup):
            shutil.copy(backup, CTL)
            os.remove(backup)
            print("ctl 已还原:", " ".join(read_ctl_lines()))
        mm.close()
        os.close(fd)


if __name__ == "__main__":
    sys.exit(main())
