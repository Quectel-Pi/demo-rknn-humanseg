#!/usr/bin/env python3
"""对照实测：哪些后处理能真正去掉椅子靠背，同时不伤到人。

评价指标（都用真实 dump 帧，不用猜测）：
  人保留率  = 高饱和前景像素中被保留的比例（越高越好，<0.90 就是切到人了）
  底部带宽  = 视野底部 20% 带的最大前景宽度（越小越好，原状约等于框宽）
  中性灰率  = 残留在前景里的中性灰占比（越低越好，靠背就是中性灰）

用法: python3 scripts/probe_postproc.py [dump目录]
"""
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import y8seg_ref as Y

PAD_T, PAD_B = 140, 500
VIEW_H = PAD_B - PAD_T
SAT_PERSON, SAT_NEUTRAL = 0.15, 0.08


def sat_of(view):
    f = view.astype(np.float32)
    return (f.max(2) - f.min(2)) / (f.max(2) + 1e-6)


# ── 后处理策略 ────────────────────────────────────────────────
def pp_none(p, sat):
    return p > 0.5


def pp_thr(p, sat, thr=0.9):
    return p > thr


def pp_largest(p, sat):
    import cv2
    m = (p > 0.5).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, 8)
    if n <= 1:
        return m.astype(bool)
    return lab == (1 + np.argmax(st[1:, cv2.CC_STAT_AREA]))


def pp_rowcolor(p, sat, margin=18, minw=40):
    """逐行把前景限制在「彩色像素跨度 + margin」内，并保留最宽的连续段。

    靠背是中性灰且横跨全宽；人体有皮肤/衣服的颜色。
    逐行做，避免整块误删。
    """
    m = p > 0.5
    out = np.zeros_like(m)
    for y in range(m.shape[0]):
        row = m[y]
        if not row.any():
            continue
        col = np.where((sat[y] > SAT_PERSON) & row)[0]
        if len(col) >= minw:
            lo = max(0, col.min() - margin)
            hi = min(m.shape[1] - 1, col.max() + margin)
            keep = np.zeros_like(row)
            keep[lo:hi + 1] = True
            out[y] = row & keep
        else:
            out[y] = row
    return out


def pp_rowcolor_cc(p, sat, margin=18, minw=40):
    """pp_rowcolor 之后再取最大连通域，去掉被切散的碎块。"""
    import cv2
    m = pp_rowcolor(p, sat, margin, minw).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, 8)
    if n <= 1:
        return m.astype(bool)
    return lab == (1 + np.argmax(st[1:, cv2.CC_STAT_AREA]))


def pp_taper(p, sat, max_ratio=0.55):
    """限制每行前景宽度不超过「人体基准宽度」的 max_ratio~1 倍。

    自上而下取每行前景宽度，用上面 60% 行的宽度中位数作为基准人体宽，
    下方行的宽度不允许超过基准 * 1.35（人体下半身通常不比肩宽很多）。
    """
    m = p > 0.5
    heights = [np.where(m[y])[0].ptp() + 1 for y in range(m.shape[0]) if m[y].any()]
    if not heights:
        return m
    heights = np.array(heights)
    base = np.median(heights[:max(3, len(heights) * 6 // 10)])
    cap = base * 1.35
    out = np.zeros_like(m)
    for y in range(m.shape[0]):
        row = m[y]
        if not row.any():
            continue
        idx = np.where(row)[0]
        if len(idx) <= cap:
            out[y] = row
            continue
        # 以该行前景中点为轴，截取宽度 cap
        c = int(np.median(idx))
        lo = max(0, c - int(cap // 2))
        hi = min(m.shape[1] - 1, lo + int(cap))
        out[y, lo:hi + 1] = row[lo:hi + 1]
    return out


STRATS = [
    ('原状 (无后处理)', pp_none),
    ('提阈值 0.90', pp_thr),
    ('最大连通域', pp_largest),
    ('逐行彩色限幅 m18', pp_rowcolor),
    ('逐行彩色限幅+CC', pp_rowcolor_cc),
    ('逐行宽度锥形限幅', pp_taper),
]


def measure(m, sat):
    if m.sum() < 100:
        return dict(ret=0.0, bw=0, neu=1.0)
    person_ref = (sat > SAT_PERSON) & (m | (sat > SAT_PERSON))
    ref = sat > SAT_PERSON                     # 参考：全部有色像素
    ret = (m & ref).sum() / max(ref.sum(), 1)
    bot = m[int(VIEW_H * 0.8):]
    bw = max((np.where(r)[0].ptp() + 1 if r.any() else 0) for r in bot) if bot.any() else 0
    neu = (m & (sat <= SAT_NEUTRAL)).sum() / max(m.sum(), 1)
    return dict(ret=ret, bw=bw, neu=neu)


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else '/tmp/cap2/cap2'
    files = sorted(glob.glob(os.path.join(d, 'y8_*.rgb')))
    if not files:
        print('没有 %s/y8_*.rgb' % d)
        return 1

    acc = {name: [] for name, _ in STRATS}
    for f in files:
        rgb = np.fromfile(f, dtype=np.uint8).reshape(Y.IMG, Y.IMG, 3)
        p = np.fromfile(f.replace('.rgb', '.f32'),
                        dtype=np.float32).reshape(Y.IMG, Y.IMG)
        view = rgb[PAD_T:PAD_B]
        pv = p[PAD_T:PAD_B]
        sat = sat_of(view)
        for name, fn in STRATS:
            acc[name].append(measure(fn(pv, sat), sat))

    print('帧数: %d   视野 640x%d\n' % (len(files), VIEW_H))
    print('%-22s %8s %10s %10s' % ('策略', '人保留率', '底部带宽', '中性灰率'))
    print('-' * 56)
    for name, _ in STRATS:
        r = acc[name]
        print('%-22s %8.3f %10.0f %9.0f%%'
              % (name,
                 np.mean([x['ret'] for x in r]),
                 np.mean([x['bw'] for x in r]),
                 100 * np.mean([x['neu'] for x in r])))
    print()
    print('判读：人保留率要 >=0.90（否则切到人）；底部带宽、中性灰率越低越好。')
    return 0


if __name__ == '__main__':
    sys.exit(main())
