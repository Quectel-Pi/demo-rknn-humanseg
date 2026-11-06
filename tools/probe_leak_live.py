#!/usr/bin/env python3
"""固定机位连续帧：判定「靠背残留」随人动，还是钉在画面上不动。

只读板端 dump 的 y8_XXXXX.rgb + .f32（真实 NPU 输出，非复刻）。
把 fg 拆成两类：
  人    饱和度 > 0.15（皮肤 / 有色衣服）
  可疑  饱和度 <= 0.08（墙 / 椅背 / 白衬衫高光）

  人质心在动 且 可疑质心几乎不动 -> 可疑区钉死 = 背景泄漏
  可疑质心跟着人动                -> 框内过度分割

用法: python3 scripts/probe_leak_live.py [dump目录]
"""
import glob
import os
import sys

import numpy as np

IMG = 640
PAD_T, PAD_B = 140, 500       # 1280x720 -> 640 letterbox 后的有效画面行
FG_THR = 0.5
SAT_PERSON, SAT_NEUTRAL = 0.15, 0.08
MIN_PX = 300


def load(f):
    rgb = np.fromfile(f, dtype=np.uint8).reshape(IMG, IMG, 3).astype(np.float32)
    p = np.fromfile(f.replace('.rgb', '.f32'), dtype=np.float32)
    sat = (rgb.max(2) - rgb.min(2)) / (rgb.max(2) + 1e-6)
    return p.reshape(IMG, IMG), sat


def measure(f):
    p, sat = load(f)
    fg = p > FG_THR
    fg[:PAD_T] = False
    fg[PAD_B:] = False
    person = fg & (sat > SAT_PERSON)
    susp = fg & (sat <= SAT_NEUTRAL)
    if person.sum() < MIN_PX or susp.sum() < MIN_PX:
        return None
    py, px = np.where(person)
    sy, sx = np.where(susp)
    return dict(person=(px.mean(), py.mean()), susp=(sx.mean(), sy.mean()),
                n=int(susp.sum()), p=float(p[susp].mean()))


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else '/tmp/cap/cap'
    files = sorted(glob.glob(os.path.join(d, 'y8_*.rgb')))
    rows = [(os.path.basename(f)[3:-4], measure(f)) for f in files]
    rows = [(n, r) for n, r in rows if r]
    if not rows:
        print('没有可用帧（%s）' % d)
        return 1

    print('dump: %s  (%d 帧可用)\n' % (d, len(rows)))
    print(' 帧        人质心      可疑质心    可疑像素  可疑均值p')
    print('-' * 62)
    for n, r in rows:
        print(' %-8s (%5.0f,%5.0f)  (%5.0f,%5.0f)  %7d    %.2f'
              % (n, r['person'][0], r['person'][1],
                 r['susp'][0], r['susp'][1], r['n'], r['p']))

    pc = np.array([r['person'] for _, r in rows])
    sc = np.array([r['susp'] for _, r in rows])
    rel = sc - pc
    print()
    print('人质心 std      = (%.1f, %.1f) px   范围 %.0f x %.0f'
          % (pc[:, 0].std(), pc[:, 1].std(), pc[:, 0].ptp(), pc[:, 1].ptp()))
    print('可疑质心 std    = (%.1f, %.1f) px   范围 %.0f x %.0f'
          % (sc[:, 0].std(), sc[:, 1].std(), sc[:, 0].ptp(), sc[:, 1].ptp()))
    print('可疑-人 相对 std = (%.1f, %.1f) px' % (rel[:, 0].std(), rel[:, 1].std()))
    print()
    if sc[:, 0].std() < pc[:, 0].std() * 0.6:
        print('>> 人在动、可疑区几乎不动 => 可疑区钉在画面上 =【背景泄漏】')
        print('   后处理（阈值/连通域/开运算/颜色）无效，须改模型或解码策略。')
    elif rel[:, 0].std() < sc[:, 0].std() * 0.7:
        print('>> 可疑区跟着人动 => 【框内过度分割】')
    else:
        print('>> 两者都在动，样本不足（建议人基本不动、只轻微左右移动）')
    return 0


if __name__ == '__main__':
    sys.exit(main())
