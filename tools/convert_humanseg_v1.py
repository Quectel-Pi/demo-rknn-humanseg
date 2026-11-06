#!/usr/bin/env python3
"""PP-HumanSegV1-Lite (398x224) Paddle -> ONNX -> RKNN 转换脚本。

详见 docs/humanseg-virtual-bg-NOTES.md §5。

依赖（宿主机，python3.10）::

    pip install rknn-toolkit2==2.3.2
    pip install paddlepaddle==2.6.2 paddle2onnx==1.3.1
    pip install protobuf==4.25.4        # rknn-toolkit2 要求 >=4.21.6,<=4.25.4

注意事项：

* paddle2onnx 必须是 1.3.x。2.1.0 强制要求 paddle>=3.0 夜间版，import 即抛异常。
* 本模型 ONNX 的 input 是动态 shape，rknn.config 必须给 dynamic_input，
  否则 load_onnx 报 "input shape ... is not support"。
* PP-HumanSegV2-Lite (256x144) 无法转换 —— ONNX 含 Paddle 自适应池化产生的
  15 组动态 Shape/Resize 链，rknn-toolkit2 2.3.2 在 load_onnx / fold_constant
  阶段除零崩溃（SIGFPE）。详见 NOTES §5.2。

用法::

    # 1) 下载 Paddle 推理模型（带 softmax，不要 argmax 版）
    curl -O https://paddleseg.bj.bcebos.com/dygraph/pp_humanseg_v2/portrait_pp_humansegv1_lite_398x224_inference_model_with_softmax.zip
    unzip portrait_pp_humansegv1_lite_398x224_inference_model_with_softmax.zip

    # 2) Paddle -> ONNX
    cd portrait_pp_humansegv1_lite_398x224_inference_model_with_softmax
    paddle2onnx --model_dir . --model_filename model.pdmodel \\
        --params_filename model.pdiparams \\
        --save_file ../../model/humanseg/hs_v1.onnx \\
        --opset_version 11 --deploy_backend rknn --enable_onnx_checker True

    # 3) 准备量化校准集（146 张人像图）
    curl -O https://paddleseg.bj.bcebos.com/humanseg/data/mini_supervisely.zip
    unzip mini_supervisely.zip
    python3 convert_humanseg_v1.py --make-dataset ./mini_supervisely/Images

    # 4) ONNX -> RKNN
    python3 convert_humanseg_v1.py
"""

import argparse
import glob
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
DEFAULT_ONNX = os.path.join(REPO, "model", "humanseg", "hs_v1.onnx")
DEFAULT_RKNN = os.path.join(REPO, "model", "humanseg", "humanseg_v1_lite_398x224.rknn")
DEFAULT_DATASET = os.path.join(HERE, "dataset.txt")

H, W = 224, 398
MEAN = [123.675, 116.28, 103.53]   # ImageNet mean * 255
STD = [58.395, 57.12, 57.375]

# 必须在 Softmax 之前截断输出。
#
# NOTES §6 记录的"输出概率塌缩"（逐像素通道和恒为 1e-5，max 却比 sum 大
# 40 倍）真因不是量化、不是 mean/std、也不是 dynamic_input —— 是 rknn-toolkit2
# 2.3.2 编译 Softmax 算子出错。
#
# ONNX 尾部的结构是：
#   ... -> Transpose(perm=[0,3,2,1]) -> Softmax(axis=-1) -> Transpose(perm=[0,3,2,1])
# 第一个 Transpose 的 perm 把 H/W 也交换了（输出是 N,W,H,C 而不是标准 NHWC），
# 紧跟一个 axis=-1 的 Softmax，rknn 编译后概率归一化完全失效。
#
# 语义分割只需要 argmax(logits)，不需要概率和，所以在 Softmax 的输入张量处
# 截断模型，CPU 侧直接对 2 通道 logits 取 argmax。实测（20 张 mini_supervisely）：
#   ONNX logits 基线   IoU 0.6852
#   RKNN logits-FP     IoU 0.6853
#   RKNN logits-INT8   IoU 0.6944
# 三者一致，模型精度完好。
PRE_SOFTMAX_OUTPUT = "Transpose.13"


def make_dataset(images_dir, out_path, count=60):
    """从图片目录随机抽 count 张写成校准集清单（每行一个绝对路径）。"""
    imgs = sorted(glob.glob(os.path.join(images_dir, "*.jpg")))
    if not imgs:
        print("ERROR: no *.jpg under %s" % images_dir)
        return 1
    random.seed(0)
    sel = random.sample(imgs, min(count, len(imgs)))
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(os.path.abspath(p) for p in sel) + "\n")
    print("dataset: %d images -> %s" % (len(sel), out_path))
    return 0


def convert(onnx_path, rknn_path, dataset_path, quantize, pre_softmax):
    from rknn.api import RKNN

    if not os.path.isfile(onnx_path):
        print("ERROR: onnx not found: %s" % onnx_path)
        return 1
    if quantize and not os.path.isfile(dataset_path):
        print("ERROR: dataset not found: %s (run --make-dataset first)" % dataset_path)
        return 1

    rknn = RKNN(verbose=True)

    print("--> config")
    ret = rknn.config(
        mean_values=[MEAN],
        std_values=[STD],
        target_platform="rk3576",
        # ONNX input 是动态 shape，必须显式给出，否则 load_onnx 拒绝
        dynamic_input=[[[1, 3, H, W]]],
    )
    print("config ret", ret)
    if ret:
        return ret

    print("--> load_onnx (outputs=%r)" % (pre_softmax,))
    # outputs 参数把图在 Softmax 之前截断。注意 rknn 会警告
    # "If you don't need to crop the model, don't set 'outputs'!" ——
    # 这里是刻意裁剪，警告可忽略。
    ret = rknn.load_onnx(model=onnx_path, outputs=[pre_softmax])
    print("load ret", ret)
    if ret:
        return ret

    print("--> build (quantize=%s)" % quantize)
    ret = rknn.build(do_quantization=quantize, dataset=dataset_path if quantize else None)
    print("build ret", ret)
    if ret:
        return ret

    print("--> export %s" % rknn_path)
    ret = rknn.export_rknn(rknn_path)
    print("export ret", ret)
    rknn.release()

    if ret == 0 and os.path.isfile(rknn_path):
        print("DONE: %s (%.2f MB)" % (rknn_path, os.path.getsize(rknn_path) / 1024 / 1024))
    return ret


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--onnx", default=DEFAULT_ONNX)
    ap.add_argument("--out", default=DEFAULT_RKNN)
    ap.add_argument("--dataset", default=DEFAULT_DATASET)
    ap.add_argument("--make-dataset", metavar="IMAGES_DIR",
                    help="从该目录抽 60 张 jpg 生成校准集清单后退出")
    ap.add_argument("--no-quant", action="store_true", help="fp 模型，不做 INT8 量化")
    ap.add_argument("--pre-softmax-output", default=PRE_SOFTMAX_OUTPUT,
                    help="在 Softmax 之前截断模型所用的张量名（默认 %(default)s）；"
                         "传空字符串则保留完整模型（输出会因 rknn 的 Softmax "
                         "编译问题而塌缩，仅供对照）")
    args = ap.parse_args()

    if args.make_dataset:
        return make_dataset(args.make_dataset, args.dataset)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    return convert(args.onnx, args.out, args.dataset, not args.no_quant,
                   args.pre_softmax_output or None)


if __name__ == "__main__":
    sys.exit(main())
