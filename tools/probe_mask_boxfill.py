#!/usr/bin/env python3
"""判定掩码是「内容驱动（框只做裁剪）」还是「随框拉伸（等于填框）」。

方法：固定图像与候选的 mcoef 不变，只改采样窗口(box)大小，
看 mask 在【图像坐标】下的内容边界如何变化。

  内容驱动：缩小窗口只是裁剪，mask 内容边界不动
  随框拉伸：mask 跟着窗口一起缩放，撑满窗口

用法: python3 scripts/probe_mask_boxfill.py /tmp/cap/cap/y8_00000.rgb
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import y8seg_ref as Y


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/cap/cap/y8_00000.rgb"
    rgb = Y.load_frame(path)
    outs, protos = Y.infer(Y.load_sess(), rgb)
    keep = Y.nms(Y.candidates(outs))
    if not keep:
        print("no person candidate")
        return 1

    c = keep[0]
    print("输入: %s" % os.path.basename(path))
    print("NMS 首框: x %.0f-%.0f  y %.0f-%.0f  score=%.2f  grid=%d"
          % (c.x1, c.x2, c.y1, c.y2, c.s, c.grid))

    cx, cy = (c.x1 + c.x2) / 2, (c.y1 + c.y2) / 2
    hw, hh = (c.x2 - c.x1) / 2, (c.y2 - c.y1) / 2

    print()
    print("窗口缩放  窗口范围(图坐标)        框内填充  mask 内容边界(图坐标)")
    print("-" * 78)
    for sc in (0.70, 0.80, 0.90, 1.00, 1.10, 1.25):
        win = Y.Cand(max(0, cx - hw * sc), max(0, cy - hh * sc),
                     min(Y.IMG - 1, cx + hw * sc), min(Y.IMG - 1, cy + hh * sc),
                     c.s, c.coef, c.grid)
        m, (x1, y1, x2, y2) = Y.mask_of(win, protos)
        if m.size == 0 or (m > 0.5).sum() < 50:
            print("  %.2f    x%3d-%3d y%3d-%3d      <空>"
                  % (sc, x1, x2, y1, y2))
            continue
        fg = m > 0.5
        ys, xs = np.where(fg)
        print("  %.2f    x%3d-%3d y%3d-%3d      %.2f     x%3d-%3d y%3d-%3d"
              % (sc, x1, x2, y1, y2, fg.mean(),
                 x1 + xs.min(), x1 + xs.max(), y1 + ys.min(), y1 + ys.max()))

    print()
    print("判读：mask 边界随窗口缩放 -> 等于填框；不随窗口变 -> 框只做裁剪。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
