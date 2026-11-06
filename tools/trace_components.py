#!/usr/bin/env python3
"""逐帧归因：把每个连通块对应到它的检测框，用于判断「孤立块过滤」该不该删。

背景（2026-11-04）：
  y8s 的 person 检测在有旁人/背景物时会给出额外的小框。人体掩码由框内
  protos 解码而来，所以「小块」天然有归属框。原来的 mask_drop_orphans 只看
  「面积 < 主体 20%」，既不看到底有没有自己的框，也不看时间连续性，
  于是框边缘和主体偶尔粘连的帧被保留、脱开的帧被删掉 —— 同一目标忽隐忽现。

本脚本给出每帧每个连通块的：
  area / bbox / fill          —— 形状
  归属框（像素多数归属）/ 该框 score / 该框面积
  comp/box                    —— 块占自己框的比例（真人接近框内填充率，
                                 背景泄漏只有百分之几甚至更低）
  keep_score                  —— 若块是「框内唯一内容」则高

用法：
    python3 scripts/trace_components.py --dir /tmp/fa/flick_a --out /tmp/trace
"""
import argparse
import glob
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import y8seg_ref as R                                        # noqa: E402

IMG = R.IMG


def frame_trace(sess, rgb, conf=0.25):
    """返回 (prob640, [(comp_area, bbox, owner_idx, owner_score, owner_box,
                        comp/box, box_fill)])"""
    outs, protos = R.infer(sess, rgb)
    keep = R.nms(R.candidates(outs, conf))

    owner = np.full((IMG, IMG), -1, np.int8)
    best = np.zeros((IMG, IMG), np.float32)
    boxes = []
    for bi, c in enumerate(keep):
        p, (x1, y1, x2, y2) = R.mask_of(c, protos)
        if not p.size:
            boxes.append(None)
            continue
        boxes.append((c.s, c.grid, (x1, y1, x2, y2), p))
        sub = best[y1:y2 + 1, x1:x2 + 1]
        upd = p > sub
        sub[upd] = p[upd]
        ow = owner[y1:y2 + 1, x1:x2 + 1]
        ow[upd] = bi

    m = (best > 0.5).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, 8)
    out = []
    for i in range(1, n):
        x, y, w, h, a = (int(v) for v in st[i][:5])
        ow = owner[lab == i]
        ow = ow[ow >= 0]
        if len(ow) == 0:
            ob, osc, obox, ofill = -1, 0.0, (0, 0, 0, 0), 0.0
        else:
            u, c = np.unique(ow, return_counts=True)
            ob = int(u[c.argmax()])
            b = boxes[ob]
            if b is None:
                osc, obox, ofill = 0.0, (0, 0, 0, 0), 0.0
            else:
                osc, _g, obox, p = b
                ofill = float((p > 0.5).mean())
        ba = max(1, (obox[2] - obox[0] + 1) * (obox[3] - obox[1] + 1))
        out.append(dict(area=a, box=(x, y, w, h), owner=ob, oscore=osc,
                        obox=obox, box_area=ba, comp_over_box=a / ba,
                        box_fill=ofill))
    return best, boxes, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', required=True)
    ap.add_argument('--skip', type=int, default=1,
                    help='每 N 帧取一帧（推理较慢时用）')
    args = ap.parse_args()

    sess = R.load_sess()
    fs = [f for f in sorted(glob.glob(os.path.join(args.dir, 'y8_*.f32')))
          if '_pre' not in f][::args.skip]

    print('帧       块面积  bbox             归属框score 框面积   comp/box 框fill')
    seq = {}
    for f in fs:
        rgbp = f.replace('.f32', '.rgb')
        if not os.path.exists(rgbp):
            continue
        prob, boxes, comps = frame_trace(sess, R.load_frame(rgbp))
        name = os.path.basename(f)[:8]
        comps.sort(key=lambda d: -d['area'])
        main = comps[0]['area'] if comps else 0
        for d in comps:
            print('%-9s %6d  (%3d,%3d,%3d,%3d)  %5.2f %7d   %6.3f   %.2f  %s'
                  % (name, d['area'], *d['box'], d['oscore'], d['box_area'],
                     d['comp_over_box'], d['box_fill'],
                     '' if d['area'] == main else
                     '%.1f%%主体' % (100.0 * d['area'] / max(1, main))))
        # 记录「非主体块」数量，用于看时间稳定性
        seq[name] = len([d for d in comps if d['area'] != main])

    print('\n非主体块数量逐帧:', list(seq.values()))


if __name__ == '__main__':
    main()
