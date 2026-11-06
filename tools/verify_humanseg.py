#!/usr/bin/env python3
"""PP-HumanSegV1-Lite 精度验证：ONNX(FP32) vs RKNN(simulator)。

覆盖两件事：
1. 确定 ONNX 正确的输入预处理（NOTES §7，GT 二值化用 ``> 0``）
2. 用同一套预处理对比 ONNXRuntime 与 RKNN 的 logits/argmax，验证
   ``humanseg_v1_lite_398x224.rknn`` 精度是否与 ONNX 一致

RKNN 模型的输出是 **Softmax 之前的 2 通道 logits**（转换时用
``load_onnx(outputs=['Transpose.13'])`` 截断），不是概率。所以这里
只做 argmax，不检查概率和 —— 见 NOTES §6 与 convert 脚本顶部注释。

用法::

    python3 scripts/verify_humanseg.py --onnx model/humanseg/hs_v1.onnx
    python3 scripts/verify_humanseg.py --rknn model/humanseg/humanseg_v1_lite_398x224.rknn
    python3 scripts/verify_humanseg.py --onnx ... --rknn ... --samples 40 --save-vis
"""

import argparse
import glob
import os
import sys

import cv2
import numpy as np

H, W = 224, 398
MEAN = np.array([123.675, 116.28, 103.53], np.float32)   # ImageNet mean * 255
STD = np.array([58.395, 57.12, 57.375], np.float32)

DEFAULT_IMAGES = "/tmp/mini_sup/mini_supervisely/Images"
DEFAULT_ANNS = "/tmp/mini_sup/mini_supervisely/Annotations"

# ONNX 图里 Softmax 的输入张量（NHWC 型布局的 2 通道 logits）
PRE_SOFTMAX = "Transpose.13"

MODES = ["raw", "imagenet", "scale01", "scale01_imagenet"]


# ── 标注与形状工具 ────────────────────────────────────────────────────

def load_gt(ann_path):
    """读标注并按 0/1 语义二值化（NOTES §7）。

    该数据集的 Annotations/*.png 像素值是 0/1 而非 0/255，用 ``> 127``
    会把 GT 全部判成背景，导致 IoU 恒为 0。
    """
    gt = cv2.imread(ann_path, cv2.IMREAD_GRAYSCALE)
    if gt is None:
        return None
    return cv2.resize(gt, (W, H), interpolation=cv2.INTER_NEAREST) > 0


def to_hw2(arr):
    """把各种布局统一成 (H, W, 2)。

    ONNX 尾部 Transpose 的 perm 是 [0,3,2,1]，作用在 NCHW 上得到
    (N, W, H, C)，H/W 是交换的、也不是标准 NHWC。RKNN 侧导出的
    布局同样不保证。所以按"长度 2 的轴就是通道轴"来判定，再把剩下
    两个轴按 H/W 摆正。
    """
    a = np.asarray(arr)
    if a.ndim == 4:
        a = a[0]
    ch = next((ax for ax in range(a.ndim) if a.shape[ax] == 2), None)
    if ch is None:
        raise ValueError("no channel axis of size 2 in shape %s" % (a.shape,))
    a = np.moveaxis(a, ch, -1)
    if a.shape[0] != H:                      # (W,H,2) -> (H,W,2)
        a = a.transpose(1, 0, 2)
    return a


def iou(pred, gt):
    u = np.logical_or(pred, gt).sum()
    return float(np.logical_and(pred, gt).sum() / u) if u else 0.0


def collect_pairs(images_dir, anns_dir, limit):
    anns = {os.path.splitext(os.path.basename(p))[0]: p
            for p in glob.glob(os.path.join(anns_dir, "*.png"))}
    pairs = []
    for ip in sorted(glob.glob(os.path.join(images_dir, "*.jpg"))):
        b = os.path.splitext(os.path.basename(ip))[0]
        if b in anns:
            pairs.append((ip, anns[b]))
        if len(pairs) >= limit:
            break
    return pairs


def load_rgb(ip):
    bgr = cv2.imread(ip)
    if bgr is None:
        return None
    return cv2.cvtColor(cv2.resize(bgr, (W, H), interpolation=cv2.INTER_LINEAR),
                        cv2.COLOR_BGR2RGB)


# ── ONNX ────────────────────────────────────────────────────────────

def preprocess(rgb, mode):
    x = rgb.astype(np.float32)
    if mode == "raw":
        pass
    elif mode == "imagenet":
        x = (x - MEAN) / STD
    elif mode == "scale01":
        x = x / 255.0
    elif mode == "scale01_imagenet":        # 0-1 域再做 ImageNet 归一化
        x = (x / 255.0 - MEAN / 255.0) / (STD / 255.0)
    else:
        raise ValueError(mode)
    return x.transpose(2, 0, 1)[None]


def onnx_logits_session(onnx_path, tmp_path):
    """返回一个能直接产出 softmax 前 logits 的 ORT session。

    原 ONNX 的 graph output 只有 softmax_0.tmp_0，中间张量不在 outputs
    里，ORT 不能直接 run；这里复制一份把 PRE_SOFTMAX 提升为 graph output。
    """
    import onnx
    import onnxruntime as ort
    from onnx import helper

    m = onnx.load(onnx_path)
    if not any(o.name == PRE_SOFTMAX for o in m.graph.output):
        m.graph.output.append(
            helper.make_tensor_value_info(PRE_SOFTMAX, onnx.TensorProto.FLOAT, None))
        m.ir_version = 8
        onnx.save(m, tmp_path)
        path = tmp_path
    else:
        path = onnx_path
    return ort.InferenceSession(path, providers=["CPUExecutionProvider"])


def eval_onnx(onnx_path, pairs, modes, tmp_path):
    sess = onnx_logits_session(onnx_path, tmp_path)
    in_name = sess.get_inputs()[0].name
    print("\n=== ONNXRuntime (softmax 前 logits) ===")
    print("%-20s %8s %10s %10s" % ("preprocess", "IoU", "pred_fg", "gt_fg"))
    best = None
    for mode in modes:
        ious, fgs, gfs = [], [], []
        for ip, ap in pairs:
            rgb, gt = load_rgb(ip), load_gt(ap)
            if rgb is None or gt is None:
                continue
            o = to_hw2(sess.run([PRE_SOFTMAX], {in_name: preprocess(rgb, mode)})[0])
            pred = o.argmax(-1) > 0
            ious.append(iou(pred, gt))
            fgs.append(pred.mean())
            gfs.append(gt.mean())
        if not ious:
            print("%-20s  (no samples)" % mode)
            continue
        m_iou = float(np.mean(ious))
        print("%-20s %8.4f %10.3f %10.3f"
              % (mode, m_iou, np.mean(fgs), np.mean(gfs)))
        if best is None or m_iou > best[1]:
            best = (mode, m_iou)
    return best


# ── RKNN ────────────────────────────────────────────────────────────

def eval_rknn(onnx_path, rknn_path, pairs, quantize, dataset,
              save_vis_dir=None):
    """在 simulator 上跑 rknn 模型并算 IoU。

    注意：simulator 不支持 ``load_rknn()`` + ``inference()`` 的组合
    （报 "not support inference on the simulator"），必须从 ONNX
    重新 build。所以这里验证的是"同一份 ONNX + 同一套 config 再 build"
    的结果，与 ``export_rknn`` 导出的 .rknn 是同一次编译的等价物。
    真要验证已导出的 .rknn 文件，得在板端用 rknnlite 跑。
    """
    from rknn.api import RKNN

    r = RKNN(verbose=False)
    if r.config(mean_values=[MEAN.tolist()], std_values=[STD.tolist()],
                target_platform="rk3576",
                dynamic_input=[[[1, 3, H, W]]]) != 0:
        print("ERROR: rknn.config failed")
        return None
    if r.load_onnx(model=onnx_path, outputs=[PRE_SOFTMAX]) != 0:
        print("ERROR: rknn.load_onnx failed")
        return None
    if r.build(do_quantization=quantize,
               dataset=dataset if quantize else None) != 0:
        print("ERROR: rknn.build failed")
        return None
    if r.init_runtime() != 0:
        print("ERROR: rknn.init_runtime failed")
        return None

    tag = "RKNN-INT8" if quantize else "RKNN-FP"
    print("\n=== %s (simulator, softmax 前 logits) ===" % tag)
    ious, fgs, gfs = [], [], []
    try:
        for idx, (ip, ap) in enumerate(pairs):
            rgb, gt = load_rgb(ip), load_gt(ap)
            if rgb is None or gt is None:
                continue
            outs = r.inference(inputs=[rgb[None].astype(np.uint8)],
                               data_format="nhwc")
            o = to_hw2(np.asarray(outs[0]))
            pred = o.argmax(-1) > 0
            ious.append(iou(pred, gt))
            fgs.append(pred.mean())
            gfs.append(gt.mean())
            if save_vis_dir and idx < 5:
                os.makedirs(save_vis_dir, exist_ok=True)
                base = os.path.splitext(os.path.basename(ip))[0]
                comp = rgb.copy()
                comp[~pred] = (40, 90, 200)
                cv2.imwrite(os.path.join(save_vis_dir, "%s_comp.png" % base),
                            cv2.cvtColor(comp, cv2.COLOR_RGB2BGR))
                cv2.imwrite(os.path.join(save_vis_dir, "%s_mask.png" % base),
                            (pred * 255).astype(np.uint8))
    finally:
        r.release()

    if not ious:
        print("  (no samples)")
        return None
    print("  IoU = %.4f   pred_fg = %.3f   gt_fg = %.3f   samples = %d"
          % (np.mean(ious), np.mean(fgs), np.mean(gfs), len(ious)))
    return float(np.mean(ious))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--onnx", default=None)
    ap.add_argument("--rknn", default=None,
                    help="给出则在 simulator 上重建并评估（用于对比 ONNX）")
    ap.add_argument("--dataset", default="scripts/dataset.txt")
    ap.add_argument("--images", default=DEFAULT_IMAGES)
    ap.add_argument("--anns", default=DEFAULT_ANNS)
    ap.add_argument("--samples", type=int, default=20)
    ap.add_argument("--modes", default=",".join(MODES),
                    help="要试的预处理列表，逗号分隔（默认全部）")
    ap.add_argument("--quant", action="store_true",
                    help="RKNN 侧做 INT8 量化（默认 FP，更快）")
    ap.add_argument("--save-vis", metavar="DIR",
                    help="把前 5 张的合成图/掩码存到该目录")
    args = ap.parse_args()

    if not args.onnx and not args.rknn:
        ap.error("至少给一个 --onnx 或 --rknn")

    pairs = collect_pairs(args.images, args.anns, args.samples)
    if not pairs:
        print("ERROR: no image/annotation pairs under %s / %s"
              % (args.images, args.anns))
        print("  下载: curl -O https://paddleseg.bj.bcebos.com/humanseg/data/"
              "mini_supervisely.zip")
        return 1
    print("samples: %d" % len(pairs))

    onnx_path = args.onnx
    if onnx_path and not os.path.isfile(onnx_path):
        print("ERROR: onnx not found: %s" % onnx_path)
        return 1

    if onnx_path:
        best = eval_onnx(onnx_path, pairs, args.modes.split(","),
                         "/tmp/hs_logits_ref.onnx")
        if best:
            print("\n>>> ONNX 最佳预处理: %s (IoU %.4f)" % best)

    if args.rknn:
        print("\n注意: RKNN 侧输入固定为 uint8 RGB，归一化由 rknn.config 的 "
              "mean/std 完成（与 ONNX 的 imagenet 预处理等价）。")
        eval_rknn(onnx_path, args.rknn, pairs, args.quant, args.dataset,
                  args.save_vis)
    return 0


if __name__ == "__main__":
    sys.exit(main())
