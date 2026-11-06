# demo-rknn-humanseg

RK3576 上的人像分割 + 虚拟背景实时 demo。摄像头实时抠人，替换背景，
桌面全屏显示，带设置面板可运行时切换。

- **后端** `hy8`：GStreamer 取流 → RKNN NPU 跑 `yolov8n-seg` → 掩码解码/后处理 → 合成 → 共享内存
- **显示端** `humanseg_view.py`：GTK4 全屏读共享内存上屏，右上角齿轮打开设置面板
- 目标帧率 **28 fps**，显示端**零丢帧**（推理可跳帧，显示不跳）

## 硬件/系统要求

| 项 | 要求 |
|---|---|
| SoC | Rockchip RK3576（NPU + RKNN Runtime 2.x） |
| 系统 | Linux 6.1 aarch64，带 GStreamer 1.0 |
| 桌面 | Wayland（GNOME 已验证）+ GTK4 + Python3 GI |
| 摄像头 | UVC，默认 `/dev/video73`、`1280x720 MJPG` |

> 摄像头节点每块板子可能不同，用 `ls /dev/video*` 查，启动时用 `--cam` 指定。

## 快速开始

### 0. 最省事：不编译，直接用预编译版本

仓库自带 `install/hy8`（aarch64 预编译，已与板端验证版本 MD5 一致），
配合 `3rdparty/` 里的 `librknnrt.so`，**既不需要 RK3576 SDK，也不需要交叉编译器**：

```bash
./deploy.sh
```

它会推送二进制、模型、`librknnrt.so` 和全部脚本，然后自动启动。

### 1. 从源码编译（想改代码时）

只需要一个 aarch64 交叉编译器，**头文件和库都已随仓库提供**（`3rdparty/`，25MB），
不用装 SDK、不用配 pkg-config、不用设 `PKG_CONFIG_PATH`。

```bash
./build.sh
# 产物: build/hy8
```

没有工具链的话：

```bash
# Debian/Ubuntu
sudo apt install g++-aarch64-linux-gnu
CXX=aarch64-linux-gnu-g++ ./build.sh
```

或指定 RK3576 SDK 里的（可选，仓库不依赖 SDK）：

```bash
CXX=$SDK/prebuilts/gcc/linux-x86/aarch64/gcc-arm-10.3-2021.07-x86_64-aarch64-none-linux-gnu/bin/aarch64-none-linux-gnu-g++ \
./build.sh
```

> 已实测：在清空 `PKG_CONFIG_PATH` / `PKG_CONFIG_SYSROOT_DIR`、只用交叉编译器的环境里，
> `./build.sh` 能编译出与板端 `fb7268ebd5a4fb80957f0ad5669dfaa4` **完全相同**的 `hy8`。

### 2. 部署 + 启动

```bash
./deploy.sh
```

它会把二进制、模型、`librknnrt.so`、显示端脚本推到板子 `/data/hs_test`，
然后自动启动。多设备时：`ADB="adb -s <serial>" ./deploy.sh`。

### 3. 板端控制

```bash
sh /data/hs_test/run_humanseg.sh            # 启动
sh /data/hs_test/run_humanseg.sh --status   # 看状态
sh /data/hs_test/run_humanseg.sh --stop     # 停止
sh /data/hs_test/run_humanseg.sh --restart  # 重来
```

常用参数：

```bash
--cam /dev/video0          换摄像头节点
--cam-res 640x480          摄像头分辨率
--disp 1280x800            显示分辨率
--no-view                  只启后端（调后端时用）
--user pi                  强制指定桌面用户（默认自动探测）
```

**桌面用户是自动探测的**，不写死。脚本从 `/run/user/*/wayland-*` 反查
socket 属主和持有它的合成器进程，所以换个用户名、换台板子也能直接跑。

## 运行时可调参数

设置面板（右上角齿轮）能改分割开关和背景，写进 `/data/hs_test/ctl.conf`，
后端每帧检查，**改了立即生效，不用重启**。

也可以直接编辑 `ctl.conf`：

| 键 | 取值 | 说明 |
|---|---|---|
| `seg` | `0`/`1` | 分割开关。`0` = 纯相机旁路 |
| `bg` | `0`~`3` | `0` 深青绿 · `1` 原图暗化 · `2` 灰 · `3` 绿幕 |
| `thr` | `5`~`90` | 前景概率阈值。**调低 = 轮廓更宽**（补边缘留白），默认 `35` 对应 0.35 |
| `up` | `0`/`1` | proto 上采样。`1` 双线性（消锯齿，默认）· `0` 4×4 块复制（旧行为，有 8px 台阶） |
| `orphan` | `0`~`100` | 孤立块面积占比阈值（%），过滤背景碎片 |
| `margin` | `0`~`160` | 解码框外扩像素。**默认 0**：实测加到 40 会让 decode 涨 2.4ms、masklag 恶化，不划算 |
| `dump` | `0`~`N` | 落 N 帧调试数据到 `/data/hs_test/dump_live`，`0` 关闭 |

推荐现场配置：

```
seg=1
bg=0
thr=35
up=1
margin=0
orphan=20
dump=0
```

> `up=1` 是默认。它把邻接台阶从 95.1% 降到 1.0%，而且比块复制更快
> （decode 9099us vs 9551us），没有性能代价。

## 性能基线

RK3576 / 1280×720 输入 / 1280×800 显示：

```
~27.8 fps   show-drop=0.00%   masklag≈1.0 帧
infer     ~16.5ms
decode     ~9.1ms
letterbox  ~1.5ms
orphan     ~1.8ms
display   ~17.4ms
```

## 目录结构

```
build.sh              交叉编译（自带头文件/库，只要工具链）
deploy.sh             adb 部署 + 启动（优先用 install/hy8，不需要先编译）
install/hy8           预编译 aarch64 二进制，入库，push 即跑
src/humanseg_y8seg.cc 后端全部逻辑（取流/推理/解码/后处理/合成/共享内存）
scripts/
  run_humanseg.sh     板端一键启动（自动探测桌面用户）
  humanseg_view.py    GTK4 显示端 + 设置面板 + 共享内存重挂
  humanseg_settings.py 独立设置窗口（桌面图标用）
  humanseg-settings.sh / .desktop  桌面启动器
  gnome-ext/          GNOME Shell 扩展
  check_*.py          板端自检（e2e / 齿轮 / 共享内存 / 流畅度）
model/y8s_rkv.rknn    已量化 RKNN 模型（与板端验证版本一致）
3rdparty/
  rknn_api/           rknn_api.h + librknnrt.so
  gst/                GStreamer/glib 头文件 + 板端 .so
docs/                 设计与问题定位记录
```

## 常见问题

**画面卡住不动**
后端重启会 `rm` 再重建 `/dev/shm/humanseg.frames`。显示端已按 inode 自动重挂，
正常不需要手工干预。如果显示端是旧版本，杀掉重开即可。

**没有画面 / 显示端没进程**
```bash
adb shell 'tail -20 /data/hs_test/view.log'
```
多半是 Wayland 会话探测错了用户，用 `--user <用户名>` 指定。

**`Gtk-CRITICAL: Unable to register the application`**
旧实例的 DBus 名字还没释放，等几秒重开；不影响画面。

**画面全是背景色，抠不出人**
相机前没人时 `seg=1` 就是满屏背景色，这是正常行为，不是 bug。

**轮廓有 8 像素一级的台阶**
确认 `ctl.conf` 里 `up=1`。若被 GUI 或手动改成 `0`，会退回块复制。

## 目录结构

```
src/humanseg_y8seg.cc    后端全部逻辑（取流/推理/掩码后处理/合成/共享内存）
scripts/humanseg_view.py GTK4 全屏显示 + 设置面板
scripts/humanseg_settings.py  设置面板控件
scripts/run_humanseg.sh  一键启动（自动探测桌面用户，不写死用户名）
scripts/check_*.py       板端自检：e2e / 齿轮按钮 / 共享内存 / 帧连续性
scripts/gnome-ext/       GNOME 顶栏扩展
tools/                   开发期排查脚本（probe_*/verify_*），历史存档
model/y8s_rkv.rknn       当前使用模型
model/legacy/            早期试过的 humanseg_v1 系列模型
3rdparty/                GStreamer + RKNN 头文件与库
```

## License

见 [LICENSE](LICENSE)。
