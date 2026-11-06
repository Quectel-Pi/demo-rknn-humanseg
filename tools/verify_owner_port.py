#!/usr/bin/env python3
"""C++ 移植对拍：把 .cc 里的 mask_drop_orphans 原文抽出来，用宿主机 g++ 编译，
逐位比对它与 Python 参考实现（probe_owner_filter.filter_by_owner）。

为什么这么做：
    板端只能跑一个摄像头，没法让旧/新二进制看同一帧；而 .rgb 已经 dump 在
    磁盘上。所以「同一输入、两种实现」的正确性只能靠抽取源码 + 原生编译来锁。
    这也是唯一能证明 .cc 里的文字（而不是我脑子里的算法）被验证过的方式。

对拍数据：/tmp/fa/flick_a —— 这是闪烁实测发生的那个场景
    （旧过滤器下第二人 90 帧切换 16 次）。

断言：
    A. C++ 与 Python 新规则输出逐位一致
    B. C++ 与 Python 新规则删掉的像素集合完全相等
    C. 在闪烁场景上，C++ 版第二人区域覆盖率逐帧不跳变
    D. 主体连通块逐位不变
"""
import glob
import os
import re
import subprocess
import sys
import tempfile

import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import y8seg_ref as R                                        # noqa: E402
import probe_owner_filter as P                               # noqa: E402

DEMO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(DEMO, 'scripts/humanseg_y8seg.cc')
IMG = 640
BOX = P.SECOND_BOX
FRAC = 0.20
WORK = tempfile.mkdtemp(prefix='portcheck-')
FAILS = []


def check(name, cond, detail=''):
    print('%-56s %s %s' % (name, 'PASS' if cond else 'FAIL', detail))
    if not cond:
        FAILS.append(name)


# ── 1. 抽取真实函数并本机编译 ─────────────────────────────────
print('== 1. 抽取 mask_drop_orphans（含新签名）并本机编译 ==')
text = open(SRC, encoding='utf-8').read()
m = re.search(r'static int mask_drop_orphans\([^)]*\)\s*\{', text)
check('函数签名已在源码中（带 own 参数）', m is not None)
sig = m.group(0) if m else ''
check('签名含 signed char* own', 'signed char* own' in sig, sig[:70])

i, depth, end = m.end() - 1, 0, None
for j in range(i, len(text)):
    depth += {'{': 1, '}': -1}.get(text[j], 0)
    if depth == 0:
        end = j + 1
        break
fn = text[m.start():end]
check('函数体已抽出', end is not None and len(fn) > 1500, '%d chars' % len(fn))

harness = r'''
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
%s
int main(int argc, char** argv) {
    if (argc < 5) return 2;
    const int n = 640 * 640;
    std::vector<float> p(n);
    std::vector<signed char> o(n);
    FILE* f = fopen(argv[1], "rb");
    if (!f || fread(p.data(), 4, n, f) != (size_t)n) return 3;
    fclose(f);
    f = fopen(argv[2], "rb");
    if (!f || fread(o.data(), 1, n, f) != (size_t)n) return 4;
    fclose(f);
    int dropped = mask_drop_orphans(p.data(), o.data(), 640, 640,
                                    (float)atof(argv[4]));
    f = fopen(argv[3], "wb");
    if (!f) return 5;
    fwrite(p.data(), 4, n, f);
    fclose(f);
    printf("%%d\n", dropped);
    return 0;
}
''' % fn
hsrc, hbin = os.path.join(WORK, 'h.cc'), os.path.join(WORK, 'h')
open(hsrc, 'w', encoding='utf-8').write(harness)
b = subprocess.run(['g++', '-O2', '-std=c++11', '-Wall', '-o', hbin, hsrc],
                   capture_output=True, text=True)
errs = [l for l in b.stderr.splitlines() if ' error:' in l]
check('抽取的函数原生编译通过', not errs, errs[0][:90] if errs else '')


def run_cpp(prob, owner, frac=FRAC):
    a = os.path.join(WORK, 'p.f32')
    o = os.path.join(WORK, 'o.i8')
    r = os.path.join(WORK, 'r.f32')
    prob.astype(np.float32).tofile(a)
    owner.astype(np.int8).tofile(o)
    res = subprocess.run([hbin, a, o, r, str(frac)],
                         capture_output=True, text=True)
    if res.returncode:
        raise RuntimeError('harness rc=%d %s' % (res.returncode, res.stderr))
    return np.fromfile(r, dtype=np.float32).reshape(IMG, IMG)


# ── 2. 真实闪烁场景逐帧对拍 ───────────────────────────────────
print('\n== 2. 真实闪烁场景（/tmp/fa/flick_a）逐帧对拍 ==')
sess = R.load_sess()
fs = [f for f in sorted(glob.glob('/tmp/fa/flick_a/y8_*.f32'))
      if '_pre' not in f]
check('闪烁场景数据存在', len(fs) >= 80, '%d 帧' % len(fs))

print('帧        C++删px Py删px 逐位一致  C++覆盖率  主体不变')
diffs = 0
cov_cpp, main_ok_all = [], True
for f in fs:
    rgbp = f.replace('.f32', '.rgb')
    prob, owner, keep = P.seg_with_owner(sess, R.load_frame(rgbp))
    if not keep:
        continue
    cpp_out = run_cpp(prob, owner)
    py_out, _ = P.filter_by_owner(prob, owner, FRAC)

    same = np.array_equal(cpp_out, py_out)
    if not same:
        diffs += 1
    m0, mc = (prob > 0.5), (cpp_out > 0.5)

    def big(mask):
        n, l, s, _ = cv2.connectedComponentsWithStats(
            mask.astype(np.uint8), 8)
        if n <= 1:
            return None
        return l == max(range(1, n), key=lambda k: s[k, 4])

    b0, bc = big(m0), big(mc)
    if b0 is not None and bc is not None and not np.array_equal(b0, bc):
        main_ok_all = False
    cov_cpp.append(P.coverage(cpp_out, BOX))
    print('%-9s %7d %6d   %-8s %8.3f  %s'
          % (os.path.basename(f)[:8], int((m0 & ~mc).sum()),
             int((m0 & ~(py_out > 0.5)).sum()),
             'ok' if same else 'DIFF', cov_cpp[-1],
             'ok' if main_ok_all else 'CHANGED'))

cov_cpp = np.array(cov_cpp)
vis = (cov_cpp > 0.05).astype(int)
flips = int(sum(1 for k in range(1, len(vis)) if vis[k] != vis[k - 1]))

print()
print('== 3. 断言 ==')
check('C++ 与 Python 新规则逐帧逐位一致', diffs == 0, '%d 帧不一致' % diffs)
check('主体连通块逐位不变', main_ok_all)
check('C++ 版第二人逐帧可见（闪烁场景）', vis.sum() == len(vis),
      '%d/%d' % (vis.sum(), len(vis)))
check('C++ 版第二人零状态切换（修复生效）', flips == 0, '%d 次' % flips)

print('\nworkdir: %s' % WORK)
print()
if FAILS:
    print('FAILED (%d): %s' % (len(FAILS), ', '.join(FAILS)))
    sys.exit(1)
print('ALL CHECKS PASSED — 验证对象是 .cc 原文，非脑内算法')
