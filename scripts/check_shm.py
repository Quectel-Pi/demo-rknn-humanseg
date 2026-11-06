#!/usr/bin/env python3
"""验证显示进程真的在画：独立读共享内存，确认有真实画面内容。

不看窗口截图（Wayland 拒绝），直接查数据源：
  1. 共享内存里的帧序号在增长
  2. 像素不是全 0 / 全同色（说明真有图像）
  3. 有背景色/肤色这类真实分布
"""
import mmap
import os
import struct
import sys
import time

SHM = "/dev/shm/humanseg.frames"
HDR = 16


def read_once():
    fd = os.open(SHM, os.O_RDONLY)
    size = os.fstat(fd).st_size
    mm = mmap.mmap(fd, size, mmap.MAP_SHARED, mmap.PROT_READ)
    seq, w, h, _ = struct.unpack_from("<IIII", mm, 0)
    n = w * h * 3
    px = mm[HDR:HDR + n]
    mm.close()
    os.close(fd)
    return seq, w, h, px


try:
    seq1, w, h, px1 = read_once()
except Exception as e:
    print("FAIL 打不开共享内存:", e)
    sys.exit(1)

print("尺寸: %dx%d   帧序号=%d" % (w, h, seq1))
time.sleep(1.0)
seq2, _w, _h, px2 = read_once()
print("1 秒后帧序号=%d  (+%d)" % (seq2, seq2 - seq1))

fails = []
if seq2 <= seq1:
    fails.append("帧序号没有增长，后端没在发布")
if w != 1280 or h != 800:
    fails.append("尺寸异常 %dx%d" % (w, h))

# 内容检查
uniq = len(set(px1[::997]))          # 稀疏采样估色数
zeros = px1.count(0)
nonblack = sum(1 for i in range(0, len(px1), 3009) if px1[i] > 20)
total = len(range(0, len(px1), 3009))
print("稀疏唯一值=%d  零字节=%d  非黑占比=%.1f%%"
      % (uniq, zeros, 100.0 * nonblack / total))

if uniq < 10:
    fails.append("颜色种类太少(%d)，可能是纯色/空白" % uniq)
if nonblack / total < 0.3:
    fails.append("非黑像素太少，画面可能是黑的")

# 与刚读到的第二帧比，确认在动
diff = sum(1 for i in range(0, len(px1), 3009) if px1[i] != px2[i])
print("两帧间变化采样点=%d/%d" % (diff, total))
if seq2 != seq1 and diff == 0:
    fails.append("帧序号变了但像素没变")

print()
if fails:
    print("RESULT: FAIL")
    for f in fails:
        print("  -", f)
    sys.exit(1)
print("RESULT: PASS  (像素真实、在刷新)")
