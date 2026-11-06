#!/usr/bin/env python3
"""实测 humanseg v1（无检测框）能否抠掉椅子靠背。

关键点：摄像头原生 1280x720 = 16:9，humanseg 输入 398x224 = 16:9，
两者宽高比完全一致 —— 直接缩放，不需要 letterbox 填充，
也不会像 y8s 那样因为灰边导致构图失真。

对比对象：板端 y8s_rkv.rknn 的实际输出（dump 的 .f32）。

用法: python3 scripts/probe_humanseg_live.py [dump目录]
"""
import glob
import os
import sys

import numpy as np

IMG = 640
PAD_T, PAD_B = 140, 500        # 1280x720 -> 640 letterbox 的有效画面行
HW, HH = 398, 224
MEAN = np.array([123.675, 116.28, 103.53], np.float32)
STD = np.array([58.395, 57.12, 57.375], np.float32)

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ONNX = os.path.join(_HERE, "model/humanseg/hs_v1.onnx")


def to_hw2(a):
    """RKNN/ONNX 尾部 Transpose 让 H/W 也可能交换，按「长度=2 的轴即通道」判断。"""
    a = a[0] if a.ndim == 4 else a
    ch = next(ax for ax in range(a.ndim) if a.shape[ax] == 2)
    a = np.moveaxis(a, ch, -1)
    return a.transpose(1, 0, 2) if a.shape[0] != HH else a


def run(sess, rgb16x9):
    """rgb16x9: (360,640,3) uint8 -> (360,640) 前景概率"""
    x = (rgb16x9.astype(np.float32) - MEAN) / STD
    x = np.transpose(x, (2, 0, 1))[None]
    out = sess.run(None, {sess.get_inputs()[0].name: x})[0]
    lg = to_hw2(out)
    m = lg.argmax(-1).astype(np.float32)
    import cv2
    return cv2.resize(m, (IMG, PAD_B - PAD_T), interpolation=cv2.INTER_LINEAR)


def stats(fg, sat, tag):
    """报告底部带的宽度与中性灰占比 —— 靠背残留的核心指标。"""
    ys, xs = np.where(fg)
    if not len(ys):
        print('  %-10s <空>' % tag)
        return None
    # 视野底部 20% 带（y 288-360）
    bot = fg[int(360 * 0.8):]
    bw = max((np.where(r)[0].ptp() + 1 if r.any() else 0) for r in bot)
    neu = (fg & (sat <= 0.08)).sum()
    print('  %-10s fg=%.3f  bbox x%d-%d y%d-%d  底部带宽=%3d/640  '
          '中性灰占fg=%.0f%%'
          % (tag, fg.mean(), xs.min(), xs.max(), ys.min(), ys.max(),
             bw, 100 * neu / max(fg.sum(), 1)))
    return dict(fg=fg.mean(), bw=bw)


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else '/tmp/cap2/cap2'
    import onnxruntime as ort
    import cv2
    so = ort.SessionOptions()
    so.intra_op_num_threads = 4
    sess = ort.InferenceSession(ONNX, so, providers=["CPUExecutionProvider"])

    files = sorted(glob.glob(os.path.join(d, 'y8_*.rgb')))[::2]
    if not files:
        print('没有 %s/y8_*.rgb' % d)
        return 1

    print('humanseg v1 (398x224, 无检测框)  vs  板端 y8s (有检测框)\n')
    for f in files:
        rgb = np.fromfile(f, dtype=np.uint8).reshape(IMG, IMG, 3)
        y8 = np.fromfile(f.replace('.rgb', '.f32'),
                         dtype=np.float32).reshape(IMG, IMG)

        view = rgb[PAD_T:PAD_B]                    # 640x360 原生 16:9
        sat = ((view.astype(np.float32).max(2) - view.astype(np.float32).min(2))
               / (view.astype(np.float32).max(2) + 1e-6))

        hs = run(sess, view) > 0.5
        y8b = (y8[PAD_T:PAD_B] > 0.5)

        print('%s' % os.path.basename(f)[3:-4])
        stats(y8b, sat, 'y8s(现状)')
        stats(hs, sat, 'humanseg')
        inter = (hs & y8b).sum()
        union = (hs | y8b).sum()
        print('  IoU(y8s, humanseg) = %.3f\n' % (inter / max(union, 1)))

        # 落盘对比图（原图 | y8s | humanseg）
        tiles = []
        for m in (None, y8b, hs):
            img = view.copy()
            if m is not None:
                img[~m] = (32, 178, 170)
            tiles.append(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        cv2.imwrite('/tmp/cap2/cmp_%s.jpg' % os.path.basename(f)[3:-4],
                    np.hstack(tiles), [cv2.IMWRITE_JPEG_QUALITY, 88])
    return 0


if __name__ == '__main__':
    sys.exit(main())
