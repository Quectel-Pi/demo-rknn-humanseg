#!/usr/bin/env python3
"""宿主侧 y8seg 流水线复刻（与 scripts/humanseg_y8seg.cc 逐行对齐）。

目的：在不反复编译/部署的前提下快速迭代分割后处理。
输入 640x640 RGB letterbox 帧（板端 dump 的 .rgb），输出 640x640 前景概率。
与板端 NPU 输出的 mask IoU 实测 0.986，可放心当作离线基准。

ONNX 输出布局（13 个，来自 RK 官方预优化 ONNX）：
    idx = sc*4 + 0   box  (1,64,g,g)   sc=0,1,2 -> g=80,40,20
    idx = sc*4 + 1   cls  (1,80,g,g)
    idx = sc*4 + 2   sum  (1,1,g,g)     未用
    idx = sc*4 + 3   mcoef(1,32,g,g)    mask 系数
    idx = 12         proto(1,32,160,160)

用法:
    python3 scripts/y8seg_ref.py --rgb /tmp/cap/cap/y8_00000.rgb --out /tmp/y8
"""
import argparse
import os
from dataclasses import dataclass, field

import cv2
import numpy as np

IMG = 640
PERSON = 0
NCOEF = 32
PH = PW = 160
DFL_LEN = 16
CONF = 0.25
NMS_THR = 0.45
GRIDS = (80, 40, 20)          # stride 8 / 16 / 32

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ONNX = os.path.join(_HERE, "model/humanseg/yolov8n_seg_rk_official.onnx")


@dataclass
class Cand:
    """一个 person 候选。coef 是 32 维 protos 权重。"""
    x1: float
    y1: float
    x2: float
    y2: float
    s: float
    coef: np.ndarray = field(repr=False)
    grid: int = 0

    @property
    def box(self) -> tuple:
        return (self.x1, self.y1, self.x2, self.y2)


def dfl_dec(t):
    """(4,16) -> (4,) softmax 期望值。"""
    e = np.exp(t - t.max(axis=1, keepdims=True))
    e /= e.sum(axis=1, keepdims=True)
    return (e * np.arange(DFL_LEN)).sum(axis=1)


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    ua = ((a[2] - a[0]) * (a[3] - a[1]) +
          (b[2] - b[0]) * (b[3] - b[1]) - inter)
    return inter / ua if ua > 0 else 0.0


def infer(sess, rgb640):
    """单次推理，返回 (raw_outs, protos)。protos 为 (32,160,160)。"""
    x = (rgb640.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]
    outs = sess.run(None, {sess.get_inputs()[0].name: x})
    return outs, outs[12][0]


def candidates(outs, conf=CONF):
    """收集 person 候选并按 score 降序返回（未做 NMS）。"""
    cand = []
    for sc, g in enumerate(GRIDS):
        box, cls, mcf = outs[sc * 4], outs[sc * 4 + 1], outs[sc * 4 + 3]
        box, cls, mcf = box[0], cls[0], mcf[0]
        stride = IMG // g
        score = cls[PERSON]
        for i, j in zip(*np.where(score >= conf)):
            d = dfl_dec(box[:, i, j].reshape(4, DFL_LEN))
            cand.append(Cand(
                (-d[0] + j + .5) * stride, (-d[1] + i + .5) * stride,
                (d[2] + j + .5) * stride, (d[3] + i + .5) * stride,
                float(score[i, j]),
                mcf[:, i, j].astype(np.float32).copy(), g))
    cand.sort(key=lambda c: -c.s)
    return cand


def nms(cand, thr=NMS_THR):
    keep = []
    for c in cand:
        if all(iou(c.box, k.box) <= thr for k in keep):
            keep.append(c)
    return keep


def mask_of(c, protos):
    """把候选的 coef 与 protos 相乘，返回框内的前景概率 (h,w) 及裁剪后的框。"""
    x1, y1 = max(0, int(c.x1)), max(0, int(c.y1))
    x2, y2 = min(IMG - 1, int(c.x2)), min(IMG - 1, int(c.y2))
    if x2 < x1 or y2 < y1:
        return np.zeros((0, 0), np.float32), (x1, y1, x1, y1)
    yy = np.arange(y1, y2 + 1) * PH // IMG
    xx = np.arange(x1, x2 + 1) * PW // IMG
    acc = c.coef @ protos[:, yy][:, :, xx].reshape(NCOEF, -1)
    p = (1.0 / (1.0 + np.exp(-acc))).reshape(len(yy), len(xx))
    return p, (x1, y1, x2, y2)


def seg_run(sess, rgb640, conf=CONF, verbose=True):
    """返回 (prob640, boxes)。boxes 为 [(x1,y1,x2,y2,score)]。"""
    outs, protos = infer(sess, rgb640)
    keep = nms(candidates(outs, conf))

    prob = np.zeros((IMG, IMG), np.float32)
    for c in keep:
        p, (x1, y1, x2, y2) = mask_of(c, protos)
        if p.size:
            sub = prob[y1:y2 + 1, x1:x2 + 1]
            np.maximum(sub, p, out=sub)

    if verbose:
        for c in keep:
            print("  box %.0f,%.0f - %.0f,%.0f  s=%.2f  grid=%d"
                  % (c.x1, c.y1, c.x2, c.y2, c.s, c.grid))
    return prob, [c.box + (c.s,) for c in keep]


def load_sess(onnx=ONNX):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = 4
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(onnx, so, providers=["CPUExecutionProvider"])


def load_frame(path):
    """读板端 dump 的 640x640 RGB888 裸帧。"""
    return np.fromfile(path, dtype=np.uint8).reshape(IMG, IMG, 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rgb", default="/tmp/cap/cap/y8_00000.rgb")
    ap.add_argument("--out", default="/tmp/y8")
    ap.add_argument("--onnx", default=ONNX)
    ap.add_argument("--conf", type=float, default=CONF)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    rgb = load_frame(args.rgb)
    prob, boxes = seg_run(load_sess(args.onnx), rgb, args.conf)

    base = os.path.splitext(os.path.basename(args.rgb))[0]
    np.save(os.path.join(args.out, base + "_prob.npy"), prob)
    print("%s: fg(p>0.5)=%.4f  boxes=%d"
          % (base, (prob > 0.5).mean(), len(boxes)))

    cv2.imwrite(os.path.join(args.out, base + "_prob.png"),
                np.dstack([(prob * 255).astype(np.uint8)] * 3))
    comp = rgb.copy()
    comp[prob <= 0.5] = (170, 178, 32)
    cv2.imwrite(os.path.join(args.out, base + "_comp.png"), comp)


if __name__ == "__main__":
    main()
