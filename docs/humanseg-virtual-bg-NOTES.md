# 人像分割 + 虚拟背景（视频会议）Demo — 技术笔记

> 状态：**模型已定稿并验证通过**（RKNN logits 版 IoU 与 ONNX 一致，可进入 C++ demo 阶段）
> 日期：2026-11-03
> 参考：[PaddleSeg](https://github.com/PaddlePaddle/PaddleSeg)、[rknn_model_zoo/examples/ppseg](https://github.com/airockchip/rknn_model_zoo/tree/main/examples/ppseg)

---

## 1. 目标

USB 摄像头实时画面 → NPU 人像分割 → 抠像 → 替换虚拟背景 → 屏幕显示（后续加 v4l2loopback 输出成虚拟摄像头给会议软件）。

硬件/软件基线：RK3576 + librknnrt 2.3.2 + Debian 13(trixie) / kernel 6.1。

---

## 2. 采集与显示链路

本 demo 的完整链路（实现在 `src/humanseg_y8seg.cc`）：

| 阶段 | 实现 | 说明 |
|------|------|------|
| 采集 | `v4l2src io-mode=4` + `mppjpegdec` + `appsink` | MJPG 1280×720@30fps，MPP 硬件 JPEG 解码，直接出 BGR，无 CPU 软解瓶颈 |
| 预处理 | letterbox 到 640×640 | pad = 114 灰 |
| 推理 | `rknn_run` | 约 16 ms/帧，三线程流水线中独立运行，可跳帧 |
| 掩码后处理 | 双线性上采样 + 羽化 + 时序 EMA | 640 域 4 px 一级的台阶在放大到 1280 后会是 8 px，必须双线性 |
| 合成 | alpha 混合虚拟背景 | 4 档背景样式，运行时切换 |
| 显示 | 共享内存 → GTK4 全屏 | 显示端零丢帧；不用 `waylandsink`，避免视频层盖住设置面板 |

三线程以有损 ring（drop-oldest）解耦：实时抠像只关心最新帧，积压的旧帧没有价值，
丢弃优于让延迟无界增长。单线程串行实测只有 9.3 fps，拆分后稳定 28 fps。

采集与显示参数：

```bash
./hy8 y8s_rkv.rknn --device /dev/video73 --cam 1280x720 --disp 1280x800
```

---

## 3. 模型选型（PP-HumanSeg 系列）

### 3.1 候选与实测下载地址

| 模型 | 输入 | mIoU(人像) | 体积 | 下载 |
|------|------|-----------|------|------|
| **PP-HumanSegV1-Lite** | 398×224 | 93.60 | 2.3MB | `https://paddleseg.bj.bcebos.com/dygraph/pp_humanseg_v2/portrait_pp_humansegv1_lite_398x224_inference_model_with_softmax.zip` (580KB) |
| PP-HumanSegV2-Lite | 256×144 | 96.63 | 5.4MB | `https://paddleseg.bj.bcebos.com/dygraph/pp_humanseg_v2/portrait_pp_humansegv2_lite_256x144_smaller/portrait_pp_humansegv2_lite_256x144_inference_model_with_softmax.zip` (3.5MB) |
| PP-HumanSegV2-Mobile | 192×192 | 93.13 | 2.67MB | `https://paddleseg.bj.bcebos.com/dygraph/pp_humanseg_v2/human_pp_humansegv2_mobile_192x192_inference_model_with_softmax.zip` (28MB) |

> ⚠️ 必须用 `_inference_model_with_softmax` 版本。`_argmax` 版本输出是 0/1 索引图，丢失置信度，无法做边缘羽化。

### 3.2 结论

**采用 PP-HumanSegV1-Lite (398×224)**，理由见 §5.2：V2-Lite 的 ONNX 含 Paddle 自适应池化，rknn-toolkit2 无法处理（必崩）。V1-Lite 算子干净。

V2-Mobile 还未尝试（体积 28MB 但模型本身只 2.67MB，zip 里含 Paddle 训练态）。

---

## 4. 宿主机工具链环境（已配好）

```bash
# 已安装版本（~/.local，python3.10）
rknn-toolkit2   2.3.2
paddlepaddle    2.6.2      # 必须 2.6.x，2.1.0 的 paddle2onnx 要求 paddle>=3.0.dev
paddle2onnx     1.3.1      # 必须 1.3.x；2.1.0 强制要求 paddle 3.0 夜间版
onnx            1.17.0
onnxruntime     1.23.2
onnxsim         已装
protobuf        4.25.4     # rknn-toolkit2 要求 >=4.21.6,<=4.25.4
```

**踩过的坑：**

1. `pip install paddle2onnx` 默认装 2.1.0 → `import paddle2onnx` 直接抛
   `ValueError: The paddlepaddle version should not be less than 3.0.0.dev20250426`。
   **必须 pin `paddle2onnx==1.3.1`**。
2. 系统原 protobuf 3.20.3（被 tensorflow 2.10 钉住），与 rknn-toolkit2 要求
   `>=4.21.6` 冲突。升级到 4.25.4 后 tensorflow 2.10 会坏（本任务不用，可接受）。
3. `rknn-toolkit2` 的 `load_onnx` 崩溃**跟 protobuf 版本无关**，升到 4.25.4 仍崩（已实测），
   真因是模型算子（见 §5.2）。

---

## 5. Paddle → ONNX → RKNN 转换

### 5.1 Paddle → ONNX

```bash
cd <解压后的 *_inference_model_with_softmax 目录>
paddle2onnx \
  --model_dir . \
  --model_filename model.pdmodel \
  --params_filename model.pdiparams \
  --save_file out.onnx \
  --opset_version 11 \
  --deploy_backend rknn \
  --enable_onnx_checker True
```

> ⚠️ `paddle2onnx` **没有** `--input_shape_dict` 参数（网上教程里的写法在 1.3.1 上会报
> `unrecognized arguments`）。输入 shape 只能事后改 ONNX，或用 rknn 的 `dynamic_input`。

**产物 input 是动态 shape**，必须显式处理：

```
IN  x                  [DynamicDimension.0, 3, DynamicDimension.1, DynamicDimension.2]
OUT softmax_0.tmp_0    [DynamicDimension.3, 2, DynamicDimension.4, DynamicDimension.5]
```

### 5.2 ⚠️ V2-Lite 转 RKNN 必崩（无解，已放弃）

```
$ python3 convert.py
--> config ret 0
--> load_onnx
timeout: the monitored command dumped core    # Floating point exception (SIGFPE)
```

试过的所有绕法，**全部崩在 `load_onnx`**：

| 尝试 | 结果 |
|------|------|
| 原始动态 shape + `inputs=`/`input_size_list=` 裁剪 | FPE 崩 |
| 手工改写 ONNX input dim 为 [1,3,144,256] + ir_version=8 | FPE 崩 |
| `onnx.shape_inference.infer_shapes` | FPE 崩 |
| `onnxsim.simplify` | FPE 崩 |
| `--deploy_backend rknn` 导出 | FPE 崩 |
| `dynamic_input=[[[1,3,144,256]]]` | 通过 load，**崩在 build 的 fold_constant** |
| `optimization_level=0` | 同样崩 |

**根因**：V2-Lite 的 ONNX 含 15 组 `Shape → Cast → Slice → Concat → Resize` 节点链，
来自 Paddle 的 adaptive pooling（`Adaptive only support static input shape` 警告）。
rknn-toolkit2 2.3.2 在这些动态 Resize 的常量折叠阶段除零。

对比 V1-Lite ONNX 算子统计（干净）：

```
Constant 414, Cast 124, Conv 48, BatchNorm 48, Concat 39, Relu 31,
Slice 28, Reshape 24, Shape 16, Transpose 14, Split 10, Resize 2, MaxPool 1, Softmax 1
```
（Resize 只 2 个且在静态图内可解，V2-Lite 是 9 个且依赖运行时 Shape）

### 5.3 V1-Lite 转 RKNN（成功，已在 Softmax 前截断）

```python
from rknn.api import RKNN
r = RKNN(verbose=False)
r.config(
    mean_values=[[123.675, 116.28, 103.53]],   # ImageNet mean
    std_values=[[58.395, 57.12, 57.375]],
    target_platform='rk3576',
    dynamic_input=[[[1, 3, 224, 398]]],        # ← 关键：ONNX 是动态 shape，必须给
)
r.load_onnx(model='hs_v1.onnx', outputs=['Transpose.13'])   # ← 关键：Softmax 前截断
r.build(do_quantization=True, dataset='dataset.txt')
r.export_rknn('humanseg_v1_lite_398x224.rknn')
```

`outputs=['Transpose.13']` 是**必须**的，原因见 §6。产物：**1.54 MiB**
（1,609,682 字节），INT8 量化。**输出是 Softmax 之前的 2 通道 logits，不是概率。**

构建日志里的这些**是无害警告**，可忽略：

```
E RKNN: REGTASK: The bit width of field value exceeds the limit, target: f2, offset: 0x402c ...
E RKNN: Unkown op target: 0
W build: The default input dtype of 'x' is changed from 'float32' to 'int8'
W load_onnx: If you don't need to crop the model, don't set 'inputs'/'input_size_list'/'outputs'!
```

> 最后一条警告是因为我们**刻意**裁剪模型，属于预期行为。
> `Unkown op target: 0` 与精度无关 —— 截断 Softmax 后精度已与 ONNX 对齐（见 §6.4）。

### 5.4 量化校准集

```bash
# 官方 PP-HumanSeg 小数据集（331MB，146 jpg + 350 标注 png）
curl -O https://paddleseg.bj.bcebos.com/humanseg/data/mini_supervisely.zip
unzip mini_supervisely.zip -d mini_sup
# 生成 dataset.txt：从 Images/ 随机抽 60 张，每行一个绝对路径
```

---

## 6. ✅ 已解决：RKNN 输出概率塌缩

**结论：真因是 rknn-toolkit2 2.3.2 编译 `Softmax` 算子出错。模型和量化都没有问题。**

解法：`load_onnx(model=..., outputs=['Transpose.13'])` 在 Softmax 之前截断，
RKNN 只输出 2 通道 logits，CPU 侧自己 argmax。语义分割不需要概率和。

### 6.1 现象

同一张图，RKNN(simulator) vs ONNXRuntime：

| 引擎 | 输出 | 通道和 | 前景占比 |
|------|------|--------|----------|
| ONNXRuntime | min 0.000000 / max 1.000000 | **恒为 1.0000** ✅ | 0.539 合理 |
| RKNN 量化 (INT8) | max 1.55e-4 | ~4.8e-7 ❌ | 0.737 全判前景 |
| RKNN 非量化 (FP) | max 1.48e-4 | ~6.0e-7 ❌ | 0.707 全判前景 |

`softmax` 输出**两个类别概率之和应该恒为 1**，RKNN 侧却是 1e-7 量级。

关键线索：**`max` 比 `prob_sum` 大 40 倍** —— 2 通道 softmax 数学上不可能。
所以 RKNN 给出的根本不是 softmax 结果，而是算子被编译坏了。

### 6.2 已排除（全部实测）

- ❌ 不是整数量化精度问题：`do_quantization=False` 同样塌缩
- ❌ 不是 protobuf 版本：4.25.4 下复现
- ❌ 不是模拟器差异：同一次 build 的 `inference()` 直接取
- ❌ 不是输入 layout：`data_format='nhwc'` + `[1,224,398,3] uint8`
- ❌ **不是 mean/std 重复归一化**：`mean=0/std=1` 且手工喂归一化好的 float，
  输出与 `mean=ImageNet` 喂 uint8 **逐位完全相同** → 说明归一化路径无关
- ❌ **不是 dynamic_input**：把 ONNX 固化成静态 `[1,3,224,398]`
  （`dim_param` 改 `dim_value`, `ir_version=8`）后结果完全一样
- ❌ 不是输入被忽略：两张不同图输出 L1 差异 0.000008（有效但极小）

### 6.3 根因

ONNX 尾部结构：

```
relu_30 -> Shape/Slice/Concat -> Resize(linear, half_pixel)
        -> Transpose(perm=[0,3,2,1])      # ← 注意：H/W 也被交换了
        -> Softmax(axis=-1)
        -> Transpose(perm=[0,3,2,1])
```

第一个 Transpose 的 `perm=[0,3,2,1]` 作用在 NCHW 上得到 **(N, W, H, C)** ——
既不是 NCHW 也不是标准 NHWC（H/W 反了）。紧跟的 `Softmax(axis=-1)` 依赖这个
非标准布局，rknn-toolkit2 2.3.2 在编译该组合时出错，归一化完全失效。

用 `outputs=['Transpose.13']`（Softmax 的输入张量）截断即绕开。

### 6.4 验证结果

20~30 张 mini_supervisely，GT 用 `> 0` 二值化（§7）：

| 引擎 | 输出 | IoU |
|------|------|-----|
| Paddle 原生推理 | softmax 概率 | 0.6852 |
| ONNXRuntime | softmax 前 logits | 0.6852 |
| RKNN FP | softmax 前 logits | 0.6853 |
| RKNN INT8 | softmax 前 logits | 0.6944 |

**三者一致，INT8 量化后甚至还略好。** Paddle 原生与 ONNX 逐位对齐，
证明 paddle2onnx 转换无损。

> 注：0.68 远低于官方声称的 mIoU 93.6，但 Paddle 原生推理也是 0.6852 ——
> 说明这是 PP-HumanSegV1-Lite（最轻量版本）在该数据集上的真实水平，
> 不是转换损失。若追求更高精度需改用 V2-Mobile（见 §3.1，尚未尝试）。

### 6.5 复现脚本

| 脚本 | 用途 |
|------|------|
| `scripts/convert_humanseg_v1.py` | 转换（已含 `outputs=['Transpose.13']` 截断） |
| `scripts/verify_humanseg.py` | 精度验证：预处理判定 + ONNX/RKNN 对比 + 抠图可视化 |

```bash
python3 scripts/verify_humanseg.py \
    --onnx model/humanseg/hs_v1.onnx \
    --rknn model/humanseg/humanseg_v1_lite_398x224.rknn \
    --samples 30 --quant --save-vis /tmp/hs_vis
```

### 6.6 踩坑备忘

- `rknn.load_rknn()` 出来的对象**不能**在 simulator 上 `inference()`
  （报 `not support inference on the simulator, please set 'target' first`）。
  验证必须在宿主机走 `load_onnx` + `build` + `init_runtime`，
  或到板端用 `rknnlite` 跑真正的 `.rknn` 文件。
- `rknn.config(mean_values=[[...]])` 的元素必须是 Python 原生 `float`，
  传 `np.float32` 会报 `invalid format`。
- 输出布局不能按固定 NHWC/NCHW 解析，要按"长度为 2 的轴即通道轴"判断。
  见 `verify_humanseg.py:to_hw2()`。
- `onnxruntime` 不能直接 run `Transpose.13`，因为中间张量不在
  `graph.output` 里。需复制一份 ONNX 并 append 该 `value_info`。

---

## 7. ⚠️ 测试集标注格式坑（已确认）

`mini_supervisely/Annotations/*.png` 的像素值是 **0 和 1**，不是 0/255：

```python
>>> np.unique(cv2.imread(ann, cv2.IMREAD_UNCHANGED))
array([0, 1], dtype=uint8)
```

用 `> 127` 做二值化会把 GT 全部误判为背景 → 之前跑的 IoU 全 0（0.0000）**是测试代码 bug，
不是模型问题**。正确写法：

```python
gt_mask = cv2.imread(ann_path, cv2.IMREAD_GRAYSCALE) > 0
```

`ort_variants.py` 测的三种输入预处理（raw0-255 / imagenet归一化 / scale0-1）IoU 全 0
同样是这个 bug，**结论作废**。用修好的脚本重测后（`scripts/verify_humanseg.py`，
20 张样本）确定 **正确预处理是 ImageNet 归一化**：

| 预处理 | IoU | pred_fg |
|--------|-----|---------|
| raw (0-255) | 0.5170 | 0.712 |
| **imagenet** | **0.6852** | 0.392 |
| scale01 (0-1) | 0.6017 | 0.316 |

Paddle 原生推理跑同样三种预处理给出**完全相同**的数字（0.5170 / 0.6852 / 0.6017），
交叉验证了结论。

---

## 8. 下一步 TODO

- [x] 修 GT 二值化（`> 0`），确定 ONNX 正确的输入预处理 → **ImageNet 归一化**（§7）
- [x] 定位 RKNN 概率塌缩真因 → **Softmax 算子编译错误**，截断输出绕开（§6）
- [x] 模型定稿后新建独立仓库 `demo-rknn-humanseg`，复用人像分割链路
- [ ] 抠像后处理：mask 边缘羽化（guided filter / 形态学 / 双线性上采样），
      否则人像边缘锯齿明显
- [ ] **⚠️ 靠背残留（背景泄漏）**：y8s 在近距离构图下输出
      横跨 99% 画面宽的 person 框，掩码贴满框边缘，把椅子靠背并进前景。
      提高阈值/连通域/颜色切除/开运算**全部无效**（靠背置信度 0.96，
      且与人体连成一块）。
      **已修复**：`mask_width_profile`（人体宽度剖面收紧，默认 k=1.15）
      把底部带宽从 506 压到 224（-56%），裁掉的像素 77% 是靠背、仅 4% 是人体。
      **详见 `靠背残留-根因定位.md`**
- [x] **✅ 后面的人忽隐忽现**：不是模型问题 —— 模型侧那人 90/90 帧都有
      独立 person 框（score 中位数 0.77），是我们自己的 `mask_drop_orphans`
      只看面积（块仅主体 3% < 20% 阈值），脱开主体就被删、粘连就留下，
      90 帧切换 16 次。**已修复**：改为按「归属框」判定 —— 来自别的检测框
      的块一律保留，只删主体框自己的碎屑。旧规则 10/30 帧可见、10 次切换
      → 新规则 30/30 帧可见、0 次切换；C++ 与 Python 复刻逐位一致。
      **详见 `后面的人忽隐忽现-根因与修复.md`**
- [x] **✅ 丢帧优化**：单线程 107ms/帧 → 9.3fps，源 30fps 丢 2/3。
      `--prof` 实测瓶颈**不是 NPU**（仅 16ms）：protos 解码 46.6ms +
      显示合成 32.9ms。已改为**采集/推理/显示三线程 + 两级有损 ring buffer**
      （drop-oldest），25.7fps 且**零丢帧**；拆线程后 decode 还从 34.7ms
      降到 11.1ms（工作集稳定、缓存命中率回来）。
      **详见 `丢帧优化-三线程与ringbuffer.md`**
- [ ] 显示线程成新瓶颈（29.5ms/帧）：`mask_to_disp` 双线性 + 显示缩放。
      `--disp` 与 `--cam` 同尺寸时 `scaled` 那段可整段跳过，是最省事的下一步
- [ ] 虚拟背景：纯色 / 图片 / 模糊原图 三种模式，可命令行切换
- [ ] 板端实测 FPS（stb_image 软解 30-40ms 仍是瓶颈，人像分割本身
      V1-Lite 398×224 在 RK3576 上预计 <10ms）
- [ ] （可选）v4l2loopback 输出虚拟摄像头，供会议软件使用
      参考 `docs/skills/ros2-integration-SKILL.md:415` §13b，
      注意 M2 上 video4/5 已被 rkcif 占用，需动态发现节点

---

## 9. 中间产物位置

模型与转换脚本**已入库**（不随 /tmp 重启丢失）：

| 路径 | 说明 |
|------|------|
| `model/humanseg/humanseg_v1_lite_398x224.rknn` | **可用产物**，INT8，1.54MiB（Softmax 前截断） |
| `model/humanseg/hs_v1.onnx` | V1-Lite ONNX（622KB），中间产物，便于排查 |
| `scripts/convert_humanseg_v1.py` | 可复现转换脚本，含 `--make-dataset` / `--no-quant` ⭐ |
| `scripts/verify_humanseg.py` | 精度验证 + 预处理判定 + 抠图可视化 ⭐ |
| `docs/humanseg-virtual-bg-NOTES.md` | 本文件 |

只在 /tmp 里、重启即失（**都能从上面三项 + §3.1 下载地址重建**）：

| 路径 | 说明 |
|------|------|
| `/tmp/mini_sup/mini_supervisely/` | 校准 + 评测数据集（331MB，需重新下载） |
| `/tmp/hs/humanseg_v2_lite.onnx` | V2-Lite ONNX，转 rknn 必崩，仅留作反面样本 |
| `/tmp/zoo2/examples/ppseg/` | 官方参考实现（C++ 后处理 `cpp/rknpu2/ppseg.cc`） |

`scripts/dataset.txt`（量化校准集清单）已 gitignore —— 内容是本机绝对路径，
重启 /tmp 后失效。重建方式：

```bash
python3 scripts/convert_humanseg_v1.py --make-dataset <mini_supervisely/Images 目录>
```

