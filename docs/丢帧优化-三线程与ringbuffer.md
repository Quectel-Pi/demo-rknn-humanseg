# 丢帧优化：三线程流水线 + ring buffer

> 日期：2026-11-04
> 场景：yolov8s-seg（`y8s_rkv.rknn`）人像抠像，RK3576 板端实时
> 相关：`scripts/humanseg_y8seg.cc`（`--prof` 打开分段计时）

---

## 0. 结果

| 版本 | 帧率 | 显示端丢帧 |
|---|---|---|
| 单线程 | **9.3 fps** | 源 30fps，丢 2/3 |
| **三线程 + ring buffer** | **14–27 fps（随桌面负载）** | **0** |

单帧推理链路耗时从 107ms 降到 32–66ms。

> ⚠️ **帧率不稳定，不要拿单一数字当基线。**
> 同一二进制在同一块板子上实测跨度 **11～27fps**，取决于 `gnome-shell`
> 是否在渲染 DSI 屏。27.6fps 那次是桌面临时空闲；14fps 那次 `gnome-shell`
> 占 23% CPU，`decode` 从 11ms 涨到 41ms（同样代码、同样二进制）。
> **判据请看分段耗时，不要看总 fps** —— 见 §4。

---

## 1. 先量，再改

优化前先加 `--prof` 打点，得到单线程的耗时构成（30fps 预算 = 33333us）：

| 阶段 | 耗时 | 占 loop |
|---|---|---|
| **protos 解码**（取输出 + DFL/NMS + mask 解码） | **46566 us** | **43%** |
| 显示合成（mask_to_disp 双线性 + 缩放 + composite） | 32860 us | 31% |
| rknn 推理（inputs_set + run） | 16090 us | 15% |
| 孤立块过滤 | 6459 us | 6% |
| letterbox | 1411 us | 1% |
| **loop 合计** | **107260 us** | 100% |

**关键发现：NPU 只占 16ms，后处理才是大头。**
「只给推理加缓存」是错的 —— 推理压根不是瓶颈。

而且三块里 **解码** 和 **显示合成** 互不依赖（显示只需要 mask + 原帧），
串行跑纯属浪费。这是拆线程的依据。

---

## 2. 结构

```
[采集线程]  v4l2src(MJPG) -> mppjpegdec -> appsink(BGR)
                │  raw ring  (cap=8, drop-oldest)
                ▼
[推理线程]  letterbox 640 -> RKNN -> DFL/NMS -> protos 解码 -> orphan -> dump
                │  mask ring (cap=4, drop-oldest)
                ▼
[显示线程]  mask 上采样 -> alpha 合成 -> appsrc -> waylandsink
```

`RingBuf<T>` 是**有损**队列：满时丢最旧一帧。

> 为什么有损是对的：实时抠像只关心最新画面。让慢阶段决定源帧率，
> 只会让延迟无界增长、画面越来越旧。drop-oldest 保证处理的永远是
> 最接近现在的那一帧 —— 这比「一帧不丢但延迟 500ms」好得多。

**两个队列的丢帧情况不一样，要分开看**：

| 队列 | 实测 | 含义 |
|---|---|---|
| `mask`（推理→显示） | `mask_q=0 drop mask=0` | 显示线程跟得上，**这是改造的直接目标，达成了** |
| `raw`（采集→推理） | `raw_q=8 drop raw=N` 持续增长 | 源 30fps、推理 ~15fps，队列必然满 |

`raw` 丢帧**不是 bug**，是 ring buffer 在正常工作：它把「摄像头 30fps」
解耦成「处理 15fps」，丢掉的是最旧的帧，处理的永远是最新画面。

> 早期有一次跑到 `drop raw=0`，那是因为当次桌面临时空闲、推理恰好跟上了
> 30fps。**不要把「零丢帧」当成验收标准** —— 它取决于桌面负载，
> 而不是代码正确性。正确的验收标准是「显示端不丢 + 采集端丢帧被计数」。

---

## 3. 踩过的三个坑（都有实测证据）

### 3.1 `push()` 忘了加锁 + signal → 推理线程一帧都取不到

第一版 `push()` 直接裸操作 `std::deque` 而没有 `pthread_cond_signal`，
消费者 `pop()` 睡在 `cond_wait` 上永远不醒。

板端现象：采集线程 30fps 正常入队（`[cap] got=292 raw_q=8`），
但推理线程一行日志都没有，`shown=0`。

修法：`push()` 全程持锁并 `signal`。已有独立对拍验证（见 §5 `wake_on_push`）。

### 3.2 `std::move(prob640)` 把缓冲搬空 → `data()` 返回 NULL → 段错误

为了省一次 1.6MB 拷贝，我用了「双缓冲 + move」。但 `move` 之后那块
vector 就空了，第 3 帧轮回到它时 `data()` 是 NULL，直接段错误。

板端现象：没有任何输出（stdout 全缓冲，连 `config:` 都没刷出来），
进程秒退。

修法：老老实实拷贝（`mf.mask640 = prob640`）。1.6MB memcpy 约 200us，
占 loop 的 0.6%，不值得为它冒险。

### 3.3 `--frames N` 语义错位 → `captured=150 inferred=8 shown=4`

`--frames` 原本限制**采集**线程，采集很快跑满 150 就退出并 `close()` 队列，
此时推理/显示还排在后面，队列里绝大多数帧直接被丢弃。

修法：`--frames` 改为限制**已显示帧数**，采集线程见到 `shown >= maxf` 才停。

> `shown` 跨线程读，用 `__atomic_load_n(..., __ATOMIC_RELAXED)` 避免数据竞争
> （只是退出条件，不需要内存序）。

---

## 4. 关于 decode 耗时（数据不稳定，结论要打折）

首次实测拆线程后 decode 从 34.7ms 降到 11.1ms，当时判断是「工作集稳定、
缓存命中率回来」。但**后续复测无法稳定复现该提升**：

| 条件 | decode | infer-loop |
|---|---|---|
| 单线程（首次测） | 34852 | 59457 |
| 三线程（首次测，桌面临时空闲） | **11060** | 31969 |
| 三线程（复测，`gnome-shell` 占 23% CPU） | **41031** | 66226 |

同样的二进制、同样的代码路径，decode 差了 3.7 倍。变量的差别是
**桌面是否在渲染 DSI 屏**。

**结论降级**：拆线程能确定的收益是
1. 显示线程不再被推理阻塞（`mask_q=0 mask=0`，显示端零丢帧）
2. 慢阶段不再拖住摄像头采集（`raw` 队列按设计丢弃最旧帧）

而「decode 变快 3 倍」**不能作为已验证的结论** —— 它可能只是当次
测量环境的产物。要下定论得在关掉桌面的纯净环境里做 A/B，这个还没做。

---

## 5. 验证（临时脚本对拍，非项目测试套件）

本仓库没有正式的测试目标（`scripts/*.cc` 是板端参考源码，手动交叉编译），
所以用临时脚本抽 `.cc` **原文**编译验证：

**A. 后处理回归**（改造不能破坏之前的修复）

```
r1 byte-identical to Python owner rule (all frames)   PASS
r2 main component never touched                       PASS
r3 second person still visible in every frame         PASS 90/90
r4 still zero state flips                             PASS 0
```

**B. RingBuf 语义**（抽 `RingBuf<T>` 模板原文，真线程跑）

```
fifo_no_drop  PASS drops=0
drop_oldest   PASS depth=4 kept=4 first=16 last=19 drops=16
wake_on_push  PASS got=1 seq=4242          <- 正是 §3.1 那个 bug
concurrent    PASS got=1001 reorder=0 pushed=2000 dropped=999
```

`concurrent` 丢掉 999 是**设计行为**（生产远快于消费），断言只验证
「不重排、不阻塞、收到的都是真推过的」。

**C. 板端端到端**

```
total captured=202 inferred=202 shown=202  ring drops: raw=0 mask=0
shown=180  25.7 fps  (raw_q=0 mask_q=0  drop raw=0 mask=0)
```

---

## 6. 当前瓶颈与下一步

拆完线程后 **显示线程 29.5ms/帧** 成了新瓶颈（占 1080p 输出链路）：

| 可优化点 | 预估 | 说明 |
|---|---|---|
| `mask_to_disp` 双线性 1280×800 | ~20ms | 可在 640 域先合成再一次性放大；或改用 NEON 手写 |
| `scaled` 最近邻缩放 1280→1280 | ~8ms | cw==dw 时是恒等映射，可以整段跳过 |
| `decode` 的 protos 查表 | ~11ms | 框内逐像素 32 次乘加，可用 im2col + NEON |

> 注意 `--disp 1280x800` 与 `--cam 1280x720` 宽高比不同，`mask_to_disp`
> 内部做的是一次重采样。若显示尺寸等于相机尺寸，`scaled` 那段可以
> 直接 memcpy（甚至省掉），这是最省事的下一步。

---

## 7. 参数与用法

```bash
# 正常跑
./humanseg_y8seg y8s_rkv.rknn --device /dev/video73 --cam 1280x720 \
    --disp 1280x800 --bg 0 --fps

# 看分段耗时
./humanseg_y8seg y8s_rkv.rknn --frames 200 --fps --prof

# 关闭孤立块过滤（回到纯面积规则前先别关，见另一份文档）
./humanseg_y8seg ... --keep-orphans
```

| 参数 | 作用 |
|---|---|
| `--prof` | 分段计时（us）：infer / decode / letterbox / orphan / disp |
| `--frames N` | **限制已显示帧数**（不是采集数，见 §3.3） |
| `--fps` | 每 30 帧打印帧率 + 队列深度 + 丢帧数 |

队列容量写死在 `Pipe`：

```cpp
struct Pipe {
    RingBuf<RawFrame>  raw{8};    // 采集 -> 推理（约 0.27s 缓冲）
    RingBuf<MaskFrame> mask{4};   // 推理 -> 显示（约 0.16s 缓冲）
};
```

实测两个队列都跑在 0 深度，buffer 只是兜底。若想更低延迟可以缩到 2~4；
若想在卡顿时保住更长的连续性可以加大 —— 代价是画面更"旧"。
