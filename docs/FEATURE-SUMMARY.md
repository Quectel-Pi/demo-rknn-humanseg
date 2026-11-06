基于 Quectel PI M2（RK3576）的摄像头实时人像分割 + 虚拟背景
    平台：Quectel PI M2（RK3576 SoC，dtb model "Quectel PI M2 RK3576 Board"），
    使用 RKNN Runtime 2.3.2（librknnrt.so，随仓库/板端提供）跑 yolov8n-seg 单类
    （person）分割模型 y8s_rkv.rknn（NPU INT8 量化，7.0MB，输入 640x640 letterbox，
    13 个输出：1 组 box/cls + 32 通道 protos 160x160），不依赖云端、不依赖外部
    服务，开机即可跑。纯 C++ 后端 + GTK4 显示端，无 Python 推理依赖。
    功能：
    1. USB 摄像头实时采集 → NPU 推理 yolov8n-seg 人像分割（COCO person 单类）
       * GStreamer v4l2src(MJPG 1280x720) -> mppjpegdec 硬件解码 -> appsink(BGR)
       * letterbox 到 640x640 送 NPU，DFL 解框 + NMS + protos 解码出逐像素前景概率
    2. 掩码后处理，让"人身上不露背景、背景里不露人"
       * 四线程流水线（采集 / 推理 / 显示 + 有损 ring buffer drop-oldest），
         后处理和显示解耦：单线程实测只有 9.3fps，拆开后稳定 28fps
       * proto 160x160 -> 640 用双线性放大（不是 4x4 块复制），消除 8px 锯齿
       * RLE + 并查集做孤立块过滤（orphan），按检测框归属决定"碎屑"还是"另一个目标"
       * 概率域羽化 + 时序 EMA 平滑，静止时轮廓不抖、晃动时 mask 不滞后
    3. 屏幕实时全屏显示 + 运行时可调设置面板
       * 后端把合好的 1280x800 BGR 帧写共享内存，显示端 GTK4 全屏读帧上屏，
         显示端零丢帧（推理可跳帧，显示不跳），实测 show-drop=0.00%
       * 4 种虚拟背景：深青绿（默认）/ 原图暗化 / 纯灰 / 绿幕，可运行时切换
       * 右上角齿轮打开设置面板：分割开关、背景样式；改完立即生效
       * 控制参数全部走文本控制文件（seg/bg/thr/up/margin/orphan），
         后端按文件内容变化热加载，改参数不用重启进程
    实测指标（板端 1280x800 全屏，摄像头 1280x720 MJPG）：
       * 推理/显示 28.0 fps，show-drop=0.00%，masklag 1.0~1.4 帧
       * 一键启动 run_humanseg.sh 冷启动 2.1s（含停旧进程、起后端、等首帧、起显示端）
       * yolov8n-seg 相比 PP-HumanSegV1-Lite：IoU 0.6995->0.8820，
         recall 0.7944->0.9195，precision 0.814->0.9543
