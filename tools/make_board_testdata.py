#!/usr/bin/env python3
"""生成板端验证用的测试数据，并在宿主算好 ONNX 参考 mask。

产出（默认 /tmp/hs_board）：
  img_NNN.rgb     398x224x3 RGB888 裸数据（给板端 C++ 程序）
  img_NNN_onnx.mask   ONNX softmax 前 logits 的 argmax（0/1，398x224）
  img_NNN_gt.mask     GT 掩码（0/1）
  manifest.txt    每行: <rgb 文件> <onnx mask> <gt mask>

用法::

    python3 scripts/make_board_testdata.py --samples 10 --out /tmp/hs_board
"""

import argparse
import glob
import os
import sys

import cv2
import numpy as np

H, W = 224, 398
MEAN = np.array([123.675, 116.28, 103.53], np.float32)
STD = np.array([58.395, 57.12, 57.375], np.float32)
PRE_SOFTMAX = "Transpose.13"

DEFAULT_IMAGES = "/tmp/mini_sup/mini_supervisely/Images"
DEFAULT_ANNS = "/tmp/mini_sup/mini_supervisely/Annotations"


def to_hw2(arr):
    """与 verify_humanseg.py:to_hw2() 相同的布局归一化。"""
    a = np.asarray(arr)
    if a.ndim == 4:
        a = a[0]
    ch = next((ax for ax in range(a.ndim) if a.shape[ax] == 2), None)
    if ch is None:
        raise ValueError("no channel axis of size 2 in shape %s" % (a.shape,))
    a = np.moveaxis(a, ch, -1)
    if a.shape[0] != H:
        a = a.transpose(1, 0, 2)
    return a


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--onnx", default="model/humanseg/hs_v1.onnx")
    ap.add_argument("--images", default=DEFAULT_IMAGES)
    ap.add_argument("--anns", default=DEFAULT_ANNS)
    ap.add_argument("--samples", type=int, default=10)
    ap.add_argument("--out", default="/tmp/hs_board")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    anns = {os.path.splitext(os.path.basename(p))[0]: p
            for p in glob.glob(os.path.join(args.anns, "*.png"))}
    picks = []
    for ip in sorted(glob.glob(os.path.join(args.images, "*.jpg"))):
        b = os.path.splitext(os.path.basename(ip))[0]
        if b in anns:
            picks.append((ip, anns[b]))
        if len(picks) >= args.samples:
            break
    if not picks:
        print("ERROR: no image/annotation pairs under %s / %s"
              % (args.images, args.anns))
        return 1

    import onnx
    import onnxruntime as ort
    from onnx import helper

    m = onnx.load(args.onnx)
    if not any(o.name == PRE_SOFTMAX for o in m.graph.output):
        m.graph.output.append(
            helper.make_tensor_value_info(PRE_SOFTMAX, onnx.TensorProto.FLOAT, None))
        m.ir_version = 8
        tmp = os.path.join(args.out, "_logits_ref.onnx")
        onnx.save(m, tmp)
        path = tmp
    else:
        path = args.onnx
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])

    lines = []
    for idx, (ip, ap) in enumerate(picks):
        bgr = cv2.imread(ip)
        if bgr is None:
            continue
        rgb = cv2.cvtColor(cv2.resize(bgr, (W, H), interpolation=cv2.INTER_LINEAR),
                           cv2.COLOR_BGR2RGB)
        base = "img_%03d" % idx
        rgb_path = os.path.join(args.out, base + ".rgb")
        # 关键：板端喂的就是这个 RGB888 裸数据，训练/推理用同一份像素
        rgb.tofile(rgb_path)

        x = ((rgb.astype(np.float32) - MEAN) / STD).transpose(2, 0, 1)[None]
        logits = to_hw2(sess.run([PRE_SOFTMAX], {"x": x})[0])
        onnx_mask = (logits.argmax(-1) > 0).astype(np.uint8)
        om_path = os.path.join(args.out, base + "_onnx.mask")
        onnx_mask.tofile(om_path)

        gt = cv2.imread(ap, cv2.IMREAD_GRAYSCALE)
        if gt is None:
            continue
        gt_mask = (cv2.resize(gt, (W, H), interpolation=cv2.INTER_NEAREST) > 0).astype(np.uint8)
        gt_path = os.path.join(args.out, base + "_gt.mask")
        gt_mask.tofile(gt_path)

        lines.append("%s %s %s" % (rgb_path, om_path, gt_path))
        print("%s  onnx_fg=%.3f gt_fg=%.3f  (%s)"
              % (base, onnx_mask.mean(), gt_mask.mean(), os.path.basename(ip)))

    with open(os.path.join(args.out, "manifest.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n%d samples -> %s/manifest.txt" % (len(lines), args.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
