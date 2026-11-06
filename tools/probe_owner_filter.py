#!/usr/bin/env python3
"""验证「按归属框」替换「按面积比例」的孤立块过滤（修忽隐忽现）。

问题（2026-11-04 板端实测）：
    mask_drop_orphans 只比面积：块 < 主体 20% 就删。后面那个人个子小，
    块只有主体的 3%，只要这一帧没和主体粘连，就被当成背景杂物删掉；
    下一帧粘连了又留下 —— 90 帧里状态切换 16 次，10fps 下约 0.5s 一闪。

    但模型侧那个人是稳定的：90/90 帧都有独立 person 框，score 中位数 0.77。
    所以闪烁不是模型的锅，是我们自己的过滤器。

新规则：
    每个连通块都能溯源到产生它的检测框（掩码是「框内 protos 解码」，
    天然带归属）。只有归属于**主体框自己**的小块才删 —— 那是同一根框
    产生的碎屑；来自**另一个检测框**的块一律保留，因为那是一个被判成
    person 的独立目标，大小不能作为否定它的理由。

度量口径（关键）：
    不能数「独立连通块个数」—— 小块和主体粘连时它也是可见的，只是并进了
    主体的连通域，数块数会把这种帧误判成「消失」。正确口径是
    **第二人检测框内的前景覆盖率**：
        coverage = fg∩box / box面积
    覆盖率掉到接近 0 = 视觉上消失；保持在框内填充率水平 = 可见。

用法：
    python3 scripts/probe_owner_filter.py --dir /tmp/fa/flick_a
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
FAILS = []

SECOND_BOX = (40, 330, 170, 450)      # x_lo, y_lo, x_hi, y_hi：第二人所在区域


def check(name, cond, detail=''):
    print('%-58s %s %s' % (name, 'PASS' if cond else 'FAIL', detail))
    if not cond:
        FAILS.append(name)


def seg_with_owner(sess, rgb, conf=0.25):
    """复刻 C++ seg_run，额外返回 owner 图（每像素归属的 keep 索引）。

    C++ 用 `if (p > prob[idx])` 取最大，归属跟随最大值一起更新。
    """
    outs, protos = R.infer(sess, rgb)
    keep = R.nms(R.candidates(outs, conf))
    prob = np.zeros((IMG, IMG), np.float32)
    owner = np.full((IMG, IMG), -1, np.int8)
    for bi, c in enumerate(keep):
        p, (x1, y1, x2, y2) = R.mask_of(c, protos)
        if not p.size:
            continue
        sub = prob[y1:y2 + 1, x1:x2 + 1]
        ow = owner[y1:y2 + 1, x1:x2 + 1]
        upd = p > sub
        sub[upd] = p[upd]
        ow[upd] = bi
    return prob, owner, keep


def filter_by_owner(prob, owner, frac):
    """新规则：只删「归属主体框」的小块。返回 (新prob, 统计)。"""
    m = (prob > 0.5).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, 8)
    if n <= 1:
        return prob, dict(ncomp=0, dropped=0, kept_others=0)

    order = sorted(range(1, n), key=lambda i: -st[i, cv2.CC_STAT_AREA])
    main = order[0]
    main_area = int(st[main, cv2.CC_STAT_AREA])
    ow = owner[lab == main]
    ow = ow[ow >= 0]
    main_owner = int(np.bincount(ow).argmax()) if len(ow) else -1

    out = prob.copy()
    thr = max(1, int(main_area * frac))
    dropped = kept = 0
    for i in order[1:]:
        a = int(st[i, cv2.CC_STAT_AREA])
        owc = owner[lab == i]
        owc = owc[owc >= 0]
        cowner = int(np.bincount(owc).argmax()) if len(owc) else -1
        # 来自别的检测框 => 独立目标，不看大小，一律保留
        if cowner >= 0 and cowner != main_owner:
            kept += 1
            continue
        # 主体框自己的碎屑 => 面积够小才删
        if a < thr:
            out[lab == i] = 0.0
            dropped += 1
    return out, dict(ncomp=n - 1, dropped=dropped, kept_others=kept)


def filter_old(prob, frac):
    """旧规则复刻：纯面积比，块 < 主体*frac 就删。"""
    out = prob.copy()
    m = (prob > 0.5).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, 8)
    if n <= 1:
        return out
    order = sorted(range(1, n), key=lambda i: -st[i, cv2.CC_STAT_AREA])
    thr = max(1, int(st[order[0], cv2.CC_STAT_AREA] * frac))
    for i in order[1:]:
        if int(st[i, cv2.CC_STAT_AREA]) < thr:
            out[lab == i] = 0.0
    return out


def coverage(p, box):
    """前景在 box 内的覆盖率（0~1）。box=(x_lo,y_lo,x_hi,y_hi)。"""
    x0, y0, x1, y1 = box
    sub = (p[y0:y1, x0:x1] > 0.5)
    return float(sub.mean())


def flips(seq):
    return int(sum(1 for i in range(1, len(seq)) if seq[i] != seq[i - 1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', default='/tmp/fa/flick_a')
    ap.add_argument('--frac', type=float, default=0.20)
    ap.add_argument('--chunk', type=int, default=2,
                    help='每 N 帧推理一次（板端 dump 是逐帧，抽样即可看趋势）')
    args = ap.parse_args()

    sess = R.load_sess()
    fs = [f for f in sorted(glob.glob(os.path.join(args.dir, 'y8_*.f32')))
          if '_pre' not in f][::args.chunk]

    print('第二人区域 x%d-%d y%d-%d，度量=该框内前景覆盖率'
          % (SECOND_BOX[0], SECOND_BOX[2], SECOND_BOX[1], SECOND_BOX[3]))
    print()
    print('帧        检测框数  旧覆盖率  新覆盖率  旧删px  新删px  新保留外来块')
    cov_old, cov_new, cov_raw = [], [], []
    junk_cleared = 0
    main_same_all = True
    for f in fs:
        rgbp = f.replace('.f32', '.rgb')
        if not os.path.exists(rgbp):
            continue
        prob, owner, keep = seg_with_owner(sess, R.load_frame(rgbp))
        if not keep:
            continue

        newp, stt = filter_by_owner(prob, owner, args.frac)
        oldp = filter_old(prob, args.frac)

        m0 = (prob > 0.5)
        m1 = (newp > 0.5)
        m2 = (oldp > 0.5)
        co, cn = coverage(oldp, SECOND_BOX), coverage(newp, SECOND_BOX)
        cov_old.append(co)
        cov_new.append(cn)
        cov_raw.append(coverage(prob, SECOND_BOX))

        # 主体逐位不变
        def big_comp(mask):
            n, l, s, _ = cv2.connectedComponentsWithStats(
                mask.astype(np.uint8), 8)
            if n <= 1:
                return None
            b = max(range(1, n), key=lambda i: s[i, cv2.CC_STAT_AREA])
            return l == b

        b0, b1 = big_comp(m0), big_comp(m1)
        if b0 is not None and b1 is not None and not np.array_equal(b0, b1):
            main_same_all = False

        if stt['dropped'] > 0:
            junk_cleared += stt['dropped']
        print('%-9s %8d   %7.3f  %7.3f  %6d  %6d  %7d'
              % (os.path.basename(f)[:8], len(keep), co, cn,
                 int((m0 & ~m2).sum()), int((m0 & ~m1).sum()),
                 stt['kept_others']))

    cov_old = np.array(cov_old)
    cov_new = np.array(cov_new)
    cov_raw = np.array(cov_raw)
    # 「消失」= 覆盖率 < 该目标可见时典型值的一半
    vis = np.median(cov_new[cov_new > 0.01]) if (cov_new > 0.01).any() else 0.1
    vold = (cov_old > vis * 0.5).astype(int)
    vnew = (cov_new > vis * 0.5).astype(int)

    print()
    print('== 时间稳定性（覆盖率 > %.3f 认定为可见）==' % (vis * 0.5))
    print('原始输出: 覆盖率 mean=%.3f std=%.4f  (不过滤，作为抖动基准)'
          % (cov_raw.mean(), cov_raw.std()))
    print('旧规则: 可见 %d/%d 帧, 覆盖率 min=%.3f max=%.3f, 状态切换 %d 次'
          % (vold.sum(), len(vold), cov_old.min(), cov_old.max(), flips(vold)))
    print('新规则: 可见 %d/%d 帧, 覆盖率 min=%.3f max=%.3f, 状态切换 %d 次'
          % (vnew.sum(), len(vnew), cov_new.min(), cov_new.max(), flips(vnew)))
    print('std: 原始 %.4f / 旧 %.4f / 新 %.4f'
          % (cov_raw.std(), cov_old.std(), cov_new.std()))
    print()

    print('== 断言 ==')
    check('旧规则确实让第二人忽隐忽现（这是要修的问题）', flips(vold) > 0,
          '%d 次切换' % flips(vold))
    check('新规则: 第二人逐帧稳定可见（切换=0）', flips(vnew) == 0,
          '%d 次切换' % flips(vnew))
    check('新规则不再引入额外抖动（std ≈ 原始输出）',
          abs(cov_new.std() - cov_raw.std()) < 0.005,
          '新 %.4f vs 原始 %.4f' % (cov_new.std(), cov_raw.std()))
    check('新规则覆盖率等于原始输出（完全不删这块）',
          cov_new.mean() >= cov_raw.mean() - 0.002,
          '新 %.3f vs 原始 %.3f' % (cov_new.mean(), cov_raw.mean()))
    check('新规则覆盖率显著高于旧规则',
          cov_new.mean() > cov_old.mean() + 0.005,
          '新 %.3f vs 旧 %.3f' % (cov_new.mean(), cov_old.mean()))
    check('主体块逐位不变（只减不增、不碰人）', main_same_all)
    check('主体自己的碎屑仍被清理（过滤未被架空）', junk_cleared >= 1,
          '%d 个碎屑块' % junk_cleared)

    print()
    if FAILS:
        print('FAILED (%d): %s' % (len(FAILS), ', '.join(FAILS)))
        sys.exit(1)
    print('ALL CHECKS PASSED')


if __name__ == '__main__':
    main()
