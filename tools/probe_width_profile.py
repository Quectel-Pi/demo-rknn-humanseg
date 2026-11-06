#!/usr/bin/env python3
"""最后一种针对性修复：用「人体宽度剖面」约束掩码横向蔓延。

原理（针对根因，不是调阈值）：
  人的腿被画面下边缘截断时，分割模型无法闭合轮廓，掩码会沿画面底部
  横向蔓开，把地板/椅子/阴影一起吞进来。
  而人体宽度剖面有一个硬约束：**肩膀/臀部以下不会比最宽处更宽**。

做法：
  1. 逐行取前景宽度，找到「上半身高饱和区」的最大宽度作为基准 Wmax
  2. 以下所有行的掩码宽度不允许超过 Wmax * k（k 默认 1.15）
  3. 超出部分从该行前景的外侧裁掉（保留最长连续段，围绕躯干中心）

和之前「锥形限幅」的区别：基准取【最大值】而非中位数，
且只在【超过阈值的那一行】裁，不会整体收缩，所以不切肩膀。

用法: python3 scripts/probe_width_profile.py [dump目录]
"""
import glob
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import y8seg_ref as Y

PAD_T, PAD_B = 140, 500
VIEW_H = PAD_B - PAD_T
SAT_PERSON = 0.15


def longest_run(mask_row):
    """返回该行最长 True 连续段的 (start, end)，无则 None。

    并列时取【第一个】，与 C++ mask_width_profile 里的
    `if (len > best) {...}`（严格大于）保持一致。
    """
    if not mask_row.any():
        return None
    idx = np.where(mask_row)[0]
    runs = np.split(idx, np.where(np.diff(idx) > 1)[0] + 1)
    best = runs[0]
    for r in runs[1:]:
        if len(r) > len(best):          # 严格大于 -> 并列保留先出现的
            best = r
    return int(best[0]), int(best[-1])


def width_profile_fix(p, sat, k=1.15, anchor='saturated', top_frac=0.5):
    """按人体宽度剖面收紧掩码。**必须与 C++ mask_width_profile 逐位一致。**

    基准宽度只取【人体上半部分】(前 top_frac 的竖直跨度)，
    因为横向蔓延发生在下半部 —— 若把整段都算进去，
    基准会被蔓延区本身污染，cap 永远大于蔓延宽度，等于没裁。

    返回 float 概率图（不是布尔）：cap 以内的行原样保留，
    只有超宽行才把 [lo,hi] 之外清零，与 C++ 的 continue 语义对齐。
    """
    prob = np.array(p, dtype=np.float32, copy=True)
    m = prob > 0.5
    if not m.any():
        return prob

    rows = np.where(m.any(1))[0]
    y0, y1 = rows[0], rows[-1]
    cut = int(y0 + (y1 - y0) * top_frac)
    wmax = max(m[y].sum() for y in range(y0, cut + 1))
    if wmax <= 0:
        return prob

    # 基准本身已接近全宽，说明框就是错的，退回用高饱和像素的宽度
    if anchor == 'saturated' and wmax > 0.75 * m.shape[1]:
        sat_max = max((m[y] & (sat[y] > SAT_PERSON)).sum()
                      for y in range(y0, y1 + 1))
        if sat_max > 30:
            wmax = sat_max

    # 与 C++ 的 lround 对齐：float/double 下 wmax*k 截断方向不同
    # （200*1.15f=230.0 但 200*1.15=229.9999...），round 消除差异
    cap = max(1, int(round(wmax * k)))
    for y in range(y0, y1 + 1):
        run = longest_run(m[y])
        if run is None:
            continue
        lo, hi = run
        if hi - lo + 1 <= cap:
            continue                       # 与 C++ 一致：不动这一行
        c = (lo + hi) // 2
        a = max(0, c - cap // 2)
        b = min(m.shape[1] - 1, a + cap)
        prob[y, :a] = 0.0
        prob[y, b + 1:] = 0.0
    return prob


def measure(m, sat, person_ref):
    """人保留率用「真实人体参考掩码」评估，而不是所有彩色像素。"""
    if m.sum() < 100:
        return dict(ret=0.0, bw=0, neu=1.0)
    ret = (m & person_ref).sum() / max(person_ref.sum(), 1)
    bot = m[int(VIEW_H * 0.8):]
    bw = max((np.where(r)[0].ptp() + 1 if r.any() else 0)
             for r in bot) if bot.any() else 0
    neu = (m & (sat <= 0.08)).sum() / m.sum()
    return dict(ret=ret, bw=bw, neu=neu)


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else '/tmp/cap3/cap3'
    files = sorted(glob.glob(os.path.join(d, 'y8_*.rgb')))
    if not files:
        print('没有 %s/y8_*.rgb' % d)
        return 1

    # 人体参考：用 ultralytics 原生 yolov8s-seg 在 PyTorch 上的人体掩码（未量化）
    from ultralytics import YOLO
    um = YOLO('/tmp/yolov8s-seg.pt')

    print('帧数 %d\n' % len(files))
    print('%-16s %9s %9s %9s' % ('策略', '人保留率', '底部带宽', '中性灰率'))
    print('-' * 48)
    acc = {}
    for f in files:
        rgb = np.fromfile(f, dtype=np.uint8).reshape(Y.IMG, Y.IMG, 3)
        p = np.fromfile(f.replace('.rgb', '.f32'),
                        dtype=np.float32).reshape(Y.IMG, Y.IMG)
        view = rgb[PAD_T:PAD_B]
        pv = p[PAD_T:PAD_B]
        fv = view.astype(np.float32)
        sat = (fv.max(2) - fv.min(2)) / (fv.max(2) + 1e-6)

        # 参考: ultralytics 的人体掩码
        r = um.predict(cv2.cvtColor(view, cv2.COLOR_RGB2BGR),
                       imgsz=640, conf=0.25, verbose=False)[0]
        ref = np.zeros((VIEW_H, Y.IMG), bool)
        mr = getattr(r, 'masks', None)
        if mr is not None and r.boxes is not None:
            for bx, mk in zip(r.boxes, mr.data):
                if um.names[int(bx.cls)] != 'person':
                    continue
                # ultralytics 的 mask 尺寸不一定是 640x640，先对齐到视野尺寸
                a = mk.cpu().numpy()
                if a.shape != (Y.IMG, Y.IMG):
                    a = cv2.resize(a, (Y.IMG, Y.IMG),
                                   interpolation=cv2.INTER_LINEAR)
                ref |= (a[PAD_T:PAD_B] > 0.5)
        if ref.sum() < 500:
            continue

        for name, fn in [
            ('原状', lambda a, b: a > 0.5),
            ('宽度剖面 k=1.00', lambda a, b: width_profile_fix(a, b, 1.00)),
            ('宽度剖面 k=1.15', lambda a, b: width_profile_fix(a, b, 1.15)),
            ('宽度剖面 k=1.30', lambda a, b: width_profile_fix(a, b, 1.30)),
        ]:
            acc.setdefault(name, []).append(measure(fn(pv, sat), sat, ref))

    for name in ['原状', '宽度剖面 k=1.00', '宽度剖面 k=1.15', '宽度剖面 k=1.30']:
        r = acc.get(name) or []
        if not r:
            print('%-16s <无数据>' % name)
            continue
        print('%-16s %9.3f %9.0f %8.0f%%'
              % (name, np.mean([x['ret'] for x in r]),
                 np.mean([x['bw'] for x in r]),
                 100 * np.mean([x['neu'] for x in r])))
    print()
    print('人保留率 = 与 ultralytics 人体掩码的交集/人体掩码面积（>=0.90 才不算切到人）')
    return 0


if __name__ == '__main__':
    sys.exit(main())
