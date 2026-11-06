#!/usr/bin/env python3
"""对比不同的候选筛选策略，找出能压掉「靠背」残留的解码参数。

板端现象：yolov8s-seg 的 person 检测框横跨画面 95-98% 宽，
mask 底部带状区域宽达整框，人坐下时椅子靠背被并进来。

本脚本在真实 dump 帧上测试：
  A. 现状（全网格、conf=0.25）
  B. 只用细网格（stride 8 / 16，丢掉 stride 32）
  C. 提高 conf
  D. 只保留面积/宽高比合理的框
并报告每个策略下的：框宽、mask 占框比例、底部带宽、是否仍横跨全画面。

用法: python3 probe_decode_variants.py
"""
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import y8seg_ref as Y

IMG = 640
NCOEF = 32
PH = PW = 160


def collect(sess, rgb, conf=0.25):
    x = (rgb.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]
    o = sess.run(None, {sess.get_inputs()[0].name: x})
    proto = o[12][0]
    cand = []
    for sc, g in enumerate((80, 40, 20)):
        box, cls, mcf = o[sc * 4 + 0][0], o[sc * 4 + 1][0], o[sc * 4 + 3][0]
        stride = IMG // g
        score = cls[Y.PERSON]
        for i, j in zip(*np.where(score >= conf)):
            d = Y.dfl_dec(box[:, i, j].reshape(4, Y.DFL_LEN))
            cand.append(dict(
                x1=(-d[0] + j + .5) * stride, y1=(-d[1] + i + .5) * stride,
                x2=(d[2] + j + .5) * stride, y2=(d[3] + i + .5) * stride,
                s=float(score[i, j]), grid=g, sc=sc,
                coef=mcf[:, i, j].astype(np.float32).copy()))
    cand.sort(key=lambda c: -c['s'])
    keep = []
    for c in cand:
        if all(Y.iou((c['x1'], c['y1'], c['x2'], c['y2']),
                     (k['x1'], k['y1'], k['x2'], k['y2'])) <= Y.NMS_THR
               for k in keep):
            keep.append(c)
    return keep, proto


def mask_of(c, proto):
    x1 = max(0, int(c['x1'])); y1 = max(0, int(c['y1']))
    x2 = min(IMG - 1, int(c['x2'])); y2 = min(IMG - 1, int(c['y2']))
    yy = np.arange(y1, y2 + 1) * PH // IMG
    xx = np.arange(x1, x2 + 1) * PW // IMG
    acc = c['coef'] @ proto[:, yy][:, :, xx].reshape(NCOEF, -1)
    m = (1.0 / (1.0 + np.exp(-acc))).reshape(len(yy), len(xx)) > 0.5
    return m, (x1, y1, x2, y2)


def describe(tag, keep, proto):
    if not keep:
        print('  %-26s <无框>' % tag)
        return None
    m, (x1, y1, x2, y2) = mask_of(keep[0], proto)
    bw, bh = x2 - x1, y2 - y1
    if m.size == 0:
        return None
    # 底部 20% 带的最大宽度
    h = m.shape[0]
    bot = m[int(h * 0.8):]
    bwidth = max((np.where(r)[0].ptp() if r.any() else 0) for r in bot) if bot.any() else 0
    fill = m.mean()
    print('  %-26s grid=%2d 框(%3d,%3d,%3d,%3d) w=%3d(%.0f%%画面)  '
          '框内填充=%.2f 底部带宽/框宽=%.2f'
          % (tag, keep[0]['grid'], x1, y1, x2, y2, bw, 100 * bw / IMG,
             fill, bwidth / max(bw, 1)))
    return dict(bw=bw, fill=fill, bwratio=bwidth / max(bw, 1))


def main():
    sess = Y.load_sess()
    files = sorted(glob.glob('/tmp/cap/cap/y8_*.rgb'))[::10]
    if not files:
        print('需要 /tmp/cap/cap/y8_*.rgb')
        return 1

    strategies = [
        ('A 现状 conf=.25 全网格', dict(conf=0.25), None),
        ('B 只用 stride8/16', dict(conf=0.25), lambda c: c['grid'] in (80, 40)),
        ('C conf=.50 全网格', dict(conf=0.50), None),
        ('D conf=.50 + stride8/16', dict(conf=0.50),
         lambda c: c['grid'] in (80, 40)),
        ('E conf=.25 + 宽<60%画面', dict(conf=0.25),
         lambda c: (c['x2'] - c['x1']) < 0.6 * IMG),
    ]

    for name, kw, filt in strategies:
        print('\n=== %s ===' % name)
        for f in files:
            rgb = np.fromfile(f, dtype=np.uint8).reshape(IMG, IMG, 3)
            keep, proto = collect(sess, rgb, **kw)
            if filt:
                keep = [c for c in keep if filt(c)]
            describe(' ' + os.path.basename(f)[3:-4], keep, proto)
    return 0


if __name__ == '__main__':
    sys.exit(main())
