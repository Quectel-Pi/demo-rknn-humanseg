#!/usr/bin/env python3
"""双模型交集：用 humanseg 的「人体先验」削掉 y8s 的靠背过度分割。

思路（利用两个模型的互补误差）：
  y8s      —— 有检测框，会沿框横向蔓延把靠背吞进来（过分割）
  humanseg —— 无检测框，全图分类，不会吃靠背，但会漏掉身体局部（欠分割）

  humanseg 的 fg 在竖直方向只覆盖到 y≈219，下方整片判为背景。
  用它当"人体位置先验"去约束 y8s，理论上能砍掉底部靠背，
  同时保留 y8s 的精细边缘。

做法：把 humanseg 掩码做适度膨胀（补偿它的欠分割），与 y8s 掩码取交集。

用法: python3 scripts/probe_dual_model.py [dump目录]
"""
import glob
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import y8seg_ref as Y
import probe_humanseg_live as H

PAD_T, PAD_B = 140, 500
VIEW_H = PAD_B - PAD_T


def humanseg_mask(sess, view):
    """返回视野坐标下的 humanseg 前景概率 (360,640) float。"""
    return H.run(sess, view)


def stats(m, sat, ref, tag):
    if m.sum() < 100:
        return dict(ret=0, bw=0, neu=1)
    ret = (m & ref).sum() / max(ref.sum(), 1)
    bot = m[int(VIEW_H * 0.8):]
    bw = max((np.where(r)[0].ptp() + 1 if r.any() else 0)
             for r in bot) if bot.any() else 0
    neu = (m & (sat <= 0.08)).sum() / m.sum()
    return dict(ret=ret, bw=bw, neu=neu)


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else '/tmp/cap3/cap3'
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = 4
    hs = ort.InferenceSession(H.ONNX, so, providers=["CPUExecutionProvider"])

    from ultralytics import YOLO
    um = YOLO('/tmp/yolov8s-seg.pt')

    files = sorted(glob.glob(os.path.join(d, 'y8_*.rgb')))
    if not files:
        print('没有 %s/y8_*.rgb' % d)
        return 1

    variants = [
        ('y8s 原状', lambda y8, hp: y8),
        ('humanseg 单独', lambda y8, hp: hp > 0.5),
        ('交集 膨胀3', lambda y8, hp: y8 & cv2.dilate(
            (hp > 0.5).astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)),
        ('交集 膨胀15', lambda y8, hp: y8 & cv2.dilate(
            (hp > 0.5).astype(np.uint8), np.ones((15, 15), np.uint8)).astype(bool)),
        ('交集 膨胀31', lambda y8, hp: y8 & cv2.dilate(
            (hp > 0.5).astype(np.uint8), np.ones((31, 31), np.uint8)).astype(bool)),
    ]
    acc = {n: [] for n, _ in variants}

    for f in files:
        rgb = np.fromfile(f, dtype=np.uint8).reshape(Y.IMG, Y.IMG, 3)
        p = np.fromfile(f.replace('.rgb', '.f32'),
                        dtype=np.float32).reshape(Y.IMG, Y.IMG)
        view = rgb[PAD_T:PAD_B]
        fv = view.astype(np.float32)
        sat = (fv.max(2) - fv.min(2)) / (fv.max(2) + 1e-6)

        y8 = p[PAD_T:PAD_B] > 0.5
        hp = humanseg_mask(hs, view)

        # 参考：ultralytics 人体掩码（PyTorch，未量化）
        r = um.predict(cv2.cvtColor(view, cv2.COLOR_RGB2BGR),
                       imgsz=640, conf=0.25, verbose=False)[0]
        ref = np.zeros((VIEW_H, Y.IMG), bool)
        mr = getattr(r, 'masks', None)
        if mr is not None and r.boxes is not None:
            for bx, mk in zip(r.boxes, mr.data):
                if um.names[int(bx.cls)] != 'person':
                    continue
                a = mk.cpu().numpy()
                if a.shape != (Y.IMG, Y.IMG):
                    a = cv2.resize(a, (Y.IMG, Y.IMG),
                                   interpolation=cv2.INTER_LINEAR)
                ref |= (a[PAD_T:PAD_B] > 0.5)
        if ref.sum() < 500:
            continue

        for n, fn in variants:
            acc[n].append(stats(fn(y8, hp), sat, ref, n))

    print('帧数: %d\n' % len(acc[variants[0][0]]))
    print('%-16s %9s %10s %9s' % ('方案', '人保留率', '底部带宽', '中性灰率'))
    print('-' * 50)
    for n, _ in variants:
        r = acc[n]
        if not r:
            print('%-16s <无数据>' % n)
            continue
        print('%-16s %9.3f %10.0f %8.0f%%'
              % (n, np.mean([x['ret'] for x in r]),
                 np.mean([x['bw'] for x in r]),
                 100 * np.mean([x['neu'] for x in r])))
    print()
    print('人保留率 = 与 ultralytics 人体掩码的交集比（>=0.85 可接受）')
    print('底部带宽 = 视野底部 20%% 带最大前景宽度（越小越好，去掉靠背的关键）')
    print('中性灰率 = 前景中低饱和像素占比（靠背就是中性暗色）')
    return 0


if __name__ == '__main__':
    sys.exit(main())
