// 板端 demo：USB 摄像头实时人像分割 + 虚拟背景（yolov8n-seg 版）
//
// 相比 humanseg_cam.cc（PP-HumanSegV1-Lite），本版换用 yolov8n-seg：
//   IoU  0.6995 -> 0.8820    recall 0.7944 -> 0.9195   precision 0.814 -> 0.9543
// 显著改善"人像被漏判、背景透过人显示"的问题。
//
// 链路（三线程流水线，中间两级 ring buffer）：
//
//   [采集线程]  v4l2src(MJPG) -> mppjpegdec -> appsink(BGR)
//                   |  raw ring (cap=8, drop-oldest)
//                   v
//   [推理线程]  letterbox 640 -> RKNN -> DFL/NMS -> protos 解码 -> orphan
//                   |  mask ring (cap=4, drop-oldest)
//                   v
//   [显示线程]  mask 上采样 -> alpha 合成 -> appsrc -> waylandsink
//
// 为什么要拆线程（2026-11-04 实测，--prof 数据）：
//   protos 解码 46.6ms + 显示合成 32.9ms + 推理 16.1ms + orphan 6.5ms
//   = 单线程 loop 107ms -> 只有 9.3fps，30fps 源必然丢帧。
//   其中最重的两块（解码 / 显示）互相不依赖，串行跑是纯浪费。
//   拆开后显示线程始终有新帧可推，摄像头不因为后处理慢而被拉慢。
//
// ring buffer 是**有损**的（drop-oldest）：实时抠像只关心最新画面，
// 积压的旧帧没有价值 —— 宁可丢帧也不能让延迟无界增长。
//
// 模型要求：RK 官方预优化 ONNX 转出的 13 输出 .rknn（y8s_rkv.rknn）。
// 不能用 ultralytics 原生 ONNX —— box/cls 共用一套量化参数，
// cls 概率会被量化成 0（详见 verify_y8seg_board.cc 头部注释）。
//
// 编译：直接跑仓库根目录的 ./build.sh（头文件/库已随仓库提供，
// 只需一个 aarch64 交叉编译器，详见 README 与 build.sh 头部注释）。

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <memory>
#include <string>
#include <vector>
#include <map>

#include <pthread.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#include <gst/app/gstappsink.h>
#include <gst/app/gstappsrc.h>
#include <gst/gst.h>

#include "rknn_api.h"

#define IMG 640
#define PERSON 0
#define NCOEF 32
#define PH 160
#define PW 160
#define DFL_LEN 16
#define CONF 0.25f
#define NMS_THR 0.45f

// ── 计时 ────────────────────────────────────────────────────────────
static double now_us() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e6 + ts.tv_nsec / 1e3;
}

// ── 模型 ────────────────────────────────────────────────────────────
struct Seg {
    rknn_context ctx = 0;
    int n_out = 0;
    // 解码框向外扩张的像素数（640x640 letterbox 域）。默认 0 = 只解码框内。
    // 若模型 proto 轮廓会超出 person 框，框外强置 0 就把人体边缘切掉了；
    // 把它调大即可验证「框裁剪是否造成留白」。
    int box_margin = 0;
    // proto(160x160) -> 640 的放大方式：
    //   1 = 双线性（默认，消除 4 像素网格台阶）
    //   0 = 4x4 块复制（历史行为，台阶钉在 4 像素网格上）
    int up_bilinear = 1;
};

static bool seg_init(Seg* s, const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) {
        printf("ERROR: cannot open %s\n", path);
        return false;
    }
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    unsigned char* md = (unsigned char*)malloc(n);
    if (!md || fread(md, 1, n, f) != (size_t)n) {
        printf("ERROR: read model\n");
        if (md) free(md);
        fclose(f);
        return false;
    }
    fclose(f);
    if (rknn_init(&s->ctx, md, n, 0, NULL) < 0) {
        printf("ERROR: rknn_init\n");
        free(md);
        return false;
    }
    free(md);

    rknn_input_output_num io;
    memset(&io, 0, sizeof(io));
    rknn_query(s->ctx, RKNN_QUERY_IN_OUT_NUM, &io, sizeof(io));
    s->n_out = (int)io.n_output;
    if (s->n_out != 13) {
        printf("ERROR: 需要 13 输出的 rknn（RK 官方 ONNX 转出），实际 %d\n",
               s->n_out);
        printf("       详见 scripts/verify_y8seg_board.cc 头部注释\n");
        return false;
    }
    printf("model: %s  (%d outputs)\n", path, s->n_out);
    return true;
}

struct Box {
    float x1, y1, x2, y2, s;
    std::vector<float> coef;
};

static float iou(const Box& a, const Box& b) {
    float x1 = std::max(a.x1, b.x1), y1 = std::max(a.y1, b.y1);
    float x2 = std::min(a.x2, b.x2), y2 = std::min(a.y2, b.y2);
    float w = std::max(0.f, x2 - x1), h = std::max(0.f, y2 - y1);
    float inter = w * h;
    float ua = (a.x2 - a.x1) * (a.y2 - a.y1) + (b.x2 - b.x1) * (b.y2 - b.y1) - inter;
    return ua > 0 ? inter / ua : 0.f;
}

static void dfl_dec(const float* t, float* out) {
    for (int b = 0; b < 4; b++) {
        float mx = -1e9f;
        for (int i = 0; i < DFL_LEN; i++) mx = std::max(mx, t[b * DFL_LEN + i]);
        float sum = 0, acc = 0;
        for (int i = 0; i < DFL_LEN; i++) {
            float e = expf(t[b * DFL_LEN + i] - mx);
            sum += e;
            acc += e * i;
        }
        out[b] = acc / sum;
    }
}

// 跑一次推理，产出 IMG x IMG 的前景概率（0.0~1.0 float）。
// 保持 float 精度：后续羽化和时序平滑都在概率域累积，
// 用 uint8 会在多帧 EMA 中不断引入舍入误差。
//
// own: 可选。非 NULL 时逐像素写入「该像素来自第几个保留框」，无前景写 -1。
//      掩码本来就是「框内 protos 解码」，所以每个像素天然有归属框；
//      孤立块过滤要靠它区分「主体框的碎屑」和「另一个被判成 person 的目标」。
//
// t_infer / t_decode: 可选分段计时（us）。实测解码比推理还贵（46ms vs 16ms），
//      这是把解码和显示拆到不同线程的依据。
static bool seg_run(Seg* s, const unsigned char* rgb, float* prob,
                    signed char* own, double* t_infer, double* t_decode,
                    int verbose = 0, std::vector<Box>* boxes_out = NULL) {
    rknn_input in;
    memset(&in, 0, sizeof(in));
    in.index = 0;
    in.type = RKNN_TENSOR_UINT8;
    in.size = IMG * IMG * 3;
    in.fmt = RKNN_TENSOR_NHWC;
    in.buf = (void*)rgb;
    in.pass_through = 0;
    double t0 = now_us();
    if (rknn_inputs_set(s->ctx, 1, &in) < 0) return false;
    if (rknn_run(s->ctx, NULL) < 0) return false;
    double t1 = now_us();

    std::vector<rknn_output> outs(s->n_out);
    memset(outs.data(), 0, sizeof(rknn_output) * s->n_out);
    for (int i = 0; i < s->n_out; i++) {
        outs[i].index = i;
        outs[i].want_float = 1;
    }
    if (rknn_outputs_get(s->ctx, s->n_out, outs.data(), NULL) < 0) return false;

    std::vector<Box> cand;
    for (int sc = 0; sc < 3; sc++) {
        const float* bp = (const float*)outs[sc * 4 + 0].buf;
        const float* cp = (const float*)outs[sc * 4 + 1].buf;
        const float* mp = (const float*)outs[sc * 4 + 3].buf;
        int g = (sc == 0) ? 80 : (sc == 1 ? 40 : 20);
        int stride = IMG / g;
        for (int i = 0; i < g; i++) {
            for (int j = 0; j < g; j++) {
                int off = i * g + j;
                float score = cp[PERSON * g * g + off];
                if (score < CONF) continue;
                float raw[4 * DFL_LEN];
                for (int c = 0; c < 4 * DFL_LEN; c++)
                    raw[c] = bp[c * g * g + off];
                float d[4];
                dfl_dec(raw, d);
                Box b;
                b.x1 = (-d[0] + j + 0.5f) * stride;
                b.y1 = (-d[1] + i + 0.5f) * stride;
                b.x2 = (d[2] + j + 0.5f) * stride;
                b.y2 = (d[3] + i + 0.5f) * stride;
                b.s = score;
                b.coef.assign(NCOEF, 0.f);
                for (int c = 0; c < NCOEF; c++)
                    b.coef[c] = mp[c * g * g + off];
                cand.push_back(std::move(b));
            }
        }
    }

    memset(prob, 0, IMG * IMG * sizeof(float));
    if (own) memset(own, -1, IMG * IMG * sizeof(signed char));
    if (cand.empty()) {
        rknn_outputs_release(s->ctx, s->n_out, outs.data());
        if (t_infer) *t_infer = t1 - t0;
        if (t_decode) *t_decode = now_us() - t1;
        return true;                       // 无人，全背景
    }

    std::sort(cand.begin(), cand.end(),
              [](const Box& a, const Box& b) { return a.s > b.s; });
    std::vector<Box> keep;
    for (auto& b : cand) {
        bool ok = true;
        for (auto& k : keep)
            if (iou(b, k) > NMS_THR) { ok = false; break; }
        if (ok) keep.push_back(b);
    }

    const float* pr = (const float*)outs[12].buf;
    if (boxes_out) *boxes_out = keep;
    if (verbose) {
        printf("[box] NMS 后保留 %zu 个框（cand=%zu）\n", keep.size(), cand.size());
        for (size_t k = 0; k < keep.size(); k++) {
            const Box& b = keep[k];
            printf("  #%zu score=%.3f box=(%d,%d,%d,%d) wh=%dx%d\n",
                   k, b.s, (int)b.x1, (int)b.y1, (int)b.x2, (int)b.y2,
                   (int)(b.x2 - b.x1), (int)(b.y2 - b.y1));
        }
    }
    // ── 解码加速（2026-11-05）───────────────────────────────────────
    // IMG=640、PW=160 => px = x*PW/IMG 恒等于 x/4，
    // 即 4x4=16 个输出像素共享同一个 proto 像素（SR=IMG/PW=4）。
    // 原实现按输出像素循环：同一个 (py,px) 的 32 次乘加和 expf
    // 被重复算了 16 遍 —— decode 占推理循环 57%（33ms）。
    //
    // 改为「按 proto 块算一次 acc，再逐输出像素判 max」：
    //   * acc/expf：每个 proto 块只算一次，降到 1/16
    //   * max/own：仍逐个输出像素做，且用框边界 [x1,x2]∩[px*SR,px*SR+SR-1]
    //     精确裁剪 —— 这是等价性关键，不能只按 proto 网格取 max，
    //     否则框边缘 1~3 px 内会把框外像素也算进来（已验证会偏）。
    // 结果与逐像素实现逐位相同（见 /tmp 等价性验证）。
    const int SR = IMG / PW, SCR = IMG / PH;    // 均整除
    // ── 轮廓锯齿修复（2026-11-06）─────────────────────────────────
    // 实测 27 帧板上数据：相邻行左边界跳变 **步长=4 占 66.1%、步长=8 占 13.1%**，
    // 步长是 4 的整数倍的占 95.1% —— 台阶严丝合缝钉在 4 像素网格上。
    // 根因：proto 是 160x160，解码时 4x4 个 640 域输出像素共用一个概率值
    // （最近邻放大）。640 域 4px 一级，放大到 1280 显示就是 8px 一级的锯齿。
    //
    // 同时边界外侧有 95.7% 的像素概率落在 (0.05, lo) 之间 —— 真实过渡信息
    // 确实存在，只是被块复制抹平了，不是模型给不出。
    //
    // 修法：先按 proto 算 sigmoid 到临时块，再**双线性**放大到 640。
    // 微基准（预计算 x 权重表，全图 640x640）：70us -> 281us，+211us，
    // 远小于 1.9ms 余量。合成图像台阶从「4/8 为主」变成「1 为主」。
    // 开关 up=0 可退回块复制做 A/B（ctl 键 up）。
    const bool upbl = s->up_bilinear != 0;
    int bx0[IMG], bx1[IMG];
    float bwx[IMG];
    if (upbl) {
        for (int x = 0; x < IMG; x++) {
            float fx = (x + 0.5f) / SR - 0.5f;
            if (fx < 0) fx = 0;
            if (fx > PW - 1) fx = PW - 1;
            int i0 = (int)fx;
            bx0[x] = i0;
            bx1[x] = i0 + 1 < PW ? i0 + 1 : i0;
            bwx[x] = fx - i0;
        }
    }
    std::vector<float> pbuf;
    int col[IMG];      // 每列打包 (c0 | c1<<16)，避免内层重复 clamp
    int ki = 0;
    for (auto& b : keep) {
        const int mg = s->box_margin;
        int x1 = std::max(0, (int)b.x1 - mg), y1 = std::max(0, (int)b.y1 - mg);
        int x2 = std::min(IMG - 1, (int)b.x2 + mg), y2 = std::min(IMG - 1, (int)b.y2 + mg);
        if (x2 < x1 || y2 < y1) { ki++; continue; }
        const int px1 = x1 * PW / IMG, px2 = x2 * PW / IMG;
        const int py1 = y1 * PH / IMG, py2 = y2 * PH / IMG;

        if (upbl) {
            // 双线性需要读相邻 proto 像素，因此多算一圈（clamp 到 proto 边界）。
            const int qx1 = std::max(0, px1 - 1), qx2 = std::min(PW - 1, px2 + 1);
            const int qy1 = std::max(0, py1 - 1), qy2 = std::min(PH - 1, py2 + 1);
            const int qw = qx2 - qx1 + 1, qh = qy2 - qy1 + 1;
            pbuf.resize((size_t)qw * qh);
            for (int py = qy1; py <= qy2; py++) {
                const float* prow = pr + (size_t)py * PW;
                float* out = pbuf.data() + (size_t)(py - qy1) * qw;
                for (int px = qx1; px <= qx2; px++) {
                    float acc = 0;
                    for (int c = 0; c < NCOEF; c++)
                        acc += b.coef[c] * prow[(size_t)c * PH * PW + px];
                    out[px - qx1] = 1.f / (1.f + expf(-acc));
                }
            }
            // 列索引只与 x 有关，与 y 无关 —— 每列 clamp 一次，
            // 内层循环就只剩取值和乘加。原先在逐像素循环里做 4 次
            // min/max，实测让 decode 白涨 2.2ms。
            for (int x = x1; x <= x2; x++)
                col[x] = (std::min(std::max(bx0[x], qx1), qx2) - qx1) |
                         ((std::min(std::max(bx1[x], qx1), qx2) - qx1) << 16);
            for (int y = y1; y <= y2; y++) {
                float fy = (y + 0.5f) / SCR - 0.5f;
                if (fy < 0) fy = 0;
                if (fy > PH - 1) fy = PH - 1;
                int i0 = (int)fy, i1 = i0 + 1 < PH ? i0 + 1 : i0;
                float wy = fy - i0;
                const float* A = pbuf.data() +
                    (size_t)(std::min(std::max(i0, qy1), qy2) - qy1) * qw;
                const float* B = pbuf.data() +
                    (size_t)(std::min(std::max(i1, qy1), qy2) - qy1) * qw;
                int idx = y * IMG;
                for (int x = x1; x <= x2; x++) {
                    const int c0 = col[x] & 0xffff;
                    const int c1 = col[x] >> 16;
                    const float w = bwx[x];
                    float a = A[c0] + (A[c1] - A[c0]) * w;
                    float c = B[c0] + (B[c1] - B[c0]) * w;
                    float p = a + (c - a) * wy;
                    int i = idx + x;
                    if (p > prob[i]) {
                        prob[i] = p;
                        if (own) own[i] = (signed char)ki;
                    }
                }
            }
            ki++;
            continue;
        }

        for (int py = py1; py <= py2; py++) {
            int ylo = std::max(y1, py * SCR);
            int yhi = std::min(y2, py * SCR + SCR - 1);
            const float* prow = pr + (size_t)py * PW;
            for (int px = px1; px <= px2; px++) {
                float acc = 0;
                for (int c = 0; c < NCOEF; c++)
                    acc += b.coef[c] * prow[(size_t)c * PH * PW + px];
                float p = 1.f / (1.f + expf(-acc));
                int xlo = std::max(x1, px * SR);
                int xhi = std::min(x2, px * SR + SR - 1);
                for (int y = ylo; y <= yhi; y++) {
                    int idx = y * IMG;
                    for (int x = xlo; x <= xhi; x++) {
                        int i = idx + x;
                        if (p > prob[i]) {
                            prob[i] = p;
                            if (own) own[i] = (signed char)ki;
                        }
                    }
                }
            }
        }
        ki++;
    }
    rknn_outputs_release(s->ctx, s->n_out, outs.data());
    if (t_infer) *t_infer = t1 - t0;
    if (t_decode) *t_decode = now_us() - t1;
    return true;
}

// ── 图像工具 ────────────────────────────────────────────────────────
static void letterbox(const unsigned char* bgr, int sw, int sh,
                      unsigned char* rgb640, float* scale, int* pad_t,
                      int* pad_l) {
    float s = std::min((float)IMG / sh, (float)IMG / sw);
    int nh = (int)(sh * s + 0.5f), nw = (int)(sw * s + 0.5f);
    int t = (IMG - nh) / 2, l = (IMG - nw) / 2;
    memset(rgb640, 114, IMG * IMG * 3);
    for (int y = 0; y < nh; y++) {
        int sy = (int)(y / s);
        if (sy >= sh) sy = sh - 1;
        for (int x = 0; x < nw; x++) {
            int sx = (int)(x / s);
            if (sx >= sw) sx = sw - 1;
            const unsigned char* p = bgr + ((long)sy * sw + sx) * 3;
            unsigned char* d = rgb640 + ((long)(y + t) * IMG + (x + l)) * 3;
            d[0] = p[2];                       // BGR -> RGB
            d[1] = p[1];
            d[2] = p[0];
        }
    }
    *scale = s;
    *pad_t = t;
    *pad_l = l;
}

// 丢掉「主体框自己产生的碎屑」，保留来自其他检测框的目标。
//
// 背景（2026-11-04 板端实测）：旧版只比面积 —— 块 < 主体 * min_frac 就删。
// 后面站着的人个子小，块只有主体的 3%，于是：
//   * 这一帧没和主体粘连 -> 当成背景杂物删掉
//   * 下一帧粘连到主体  -> 又留下
// 90 帧里状态切换 16 次，10fps 下约 0.5s 一闪，就是「后面的人忽隐忽现」。
// 而模型侧那人 90/90 帧都有独立 person 框（score 中位数 0.77），
// 稳定得很 —— 闪烁完全是我们自己的过滤器造成的。
//
// 正确做法：看块的**归属框**。掩码是「框内 protos 解码」出来的，每个像素
// 都能溯源到产生它的检测框（seg_run 的 own 图）。于是：
//   * 归属**别的检测框** -> 那是模型判定为 person 的独立目标，
//     大小不能作为否定它的理由，一律保留（并排的第二个人、远景的人）
//   * 归属**主体框自己** -> 同一根框的噪声碎屑，面积够小才删
//
// own 为 NULL 时退化为「全部视为主体框」，行为与旧版一致。
// 返回被清零的像素数。
//
// ── 2026-11-05 性能重写：逐像素 BFS -> 行游程 + union-find ──────────
// 动机（板端 --prof 实测）：本函数占 infer-loop 的 6.6~9.6ms，而同期
// 7/7 帧真实数据只有 **1 个连通块** —— 即绝大多数时间在做无用功。
// ctl 置 orphan=0 后 infer-loop 从 37.5ms 降到 33.0ms、masklag 从
// 2.07 帧降到 1.02 帧，证明这 6ms 直接变成用户可见的轮廓滞后。
//
// 为什么不是"先降采样探测再决定"：实测 1/4、1/8、1/16 探测耗时都在
// 1.5ms 左右（瓶颈是读 1.6MB 概率图，不是 BFS），收益有限且引入
// "细窄结构降采样断开 -> 误判多块"的风险。
//
// 为什么是游程：连通域的真实成本应正比于**游程数**而非像素数。
// 人体每行只有 1~3 段，640 行约 1000 个游程，比 90k 前景像素少两个
// 数量级。且最终动作只是"把某些块的 prob 清零"，完全不需要 640x640
// 的标号图 —— 面积由游程长度求和得到，清零只遍历被删游程的像素。
//
// 等价性已验证（板端真实概率图重放，20 次取均值）：
//   逐像素 BFS 9201 us / 游程 2179 us = 4.2x
//   输出逐像素一致 YES；注入 400px 碎块后两版都删 400px 且结果一致。
struct SegRun { int y, x0, x1; };

static int uf_find(std::vector<int>& p, int x) {
    while (p[x] != x) { p[x] = p[p[x]]; x = p[x]; }   // 路径减半
    return x;
}

static int mask_drop_orphans(float* prob, const signed char* own,
                             int w, int h, float min_frac) {
    // ── 1) 切游程（一趟扫描，只读 prob）────────────────────────────
    static std::vector<SegRun> runs;
    runs.clear();
    for (int y = 0; y < h; y++) {
        const float* row = prob + (size_t)y * w;
        int x = 0;
        while (x < w) {
            if (row[x] <= 0.5f) { x++; continue; }
            int s = x;
            while (x < w && row[x] > 0.5f) x++;
            runs.push_back({y, s, x - 1});
        }
    }
    const int nr = (int)runs.size();
    if (nr == 0) return 0;

    // ── 2) 游程级 union-find ──────────────────────────────────────
    // 只有相邻行才可能连通；8 邻域 => 两个区间各扩 1 后求交。
    std::vector<int> uf(nr);
    for (int i = 0; i < nr; i++) uf[i] = i;

    std::vector<int> row_begin(h + 1, 0);
    for (int i = 0; i < nr; i++) row_begin[runs[i].y + 1]++;
    for (int y = 0; y < h; y++) row_begin[y + 1] += row_begin[y];

    for (int y = 1; y < h; y++) {
        int a = row_begin[y - 1], ae = row_begin[y];
        int b = row_begin[y], be = row_begin[y + 1];
        while (a < ae && b < be) {
            const SegRun& p = runs[a];
            const SegRun& q = runs[b];
            if (p.x1 + 1 < q.x0)      a++;
            else if (q.x1 + 1 < p.x0) b++;
            else {
                int ra = uf_find(uf, a), rb = uf_find(uf, b);
                if (ra != rb) uf[rb] = ra;
                if (p.x1 < q.x1) a++; else b++;
            }
        }
    }

    // ── 3) 每块面积 = 该块所有游程长度之和 ──────────────────────────
    std::vector<int> area(nr, 0);
    for (int i = 0; i < nr; i++)
        area[uf_find(uf, i)] += runs[i].x1 - runs[i].x0 + 1;

    int best = -1, best_area = -1;
    for (int i = 0; i < nr; i++)
        if (uf_find(uf, i) == i && area[i] > best_area) { best_area = area[i]; best = i; }
    if (best < 0) return 0;

    // ── 4) 每块归属框 ──────────────────────────────────────────────
    // 块内实际出现的 own 值极少（通常 1~2 种），用有序 map 稀疏计数，
    // 不建原来 128 x 块数 的稠密直方图。
    const int NB = 128, NONE = NB - 1;
    std::map<int, std::map<int, int>> hist;      // root -> (own -> 像素数)
    for (int i = 0; i < nr; i++) {
        int r = uf_find(uf, i);
        const signed char* orow = own ? own + (size_t)runs[i].y * w : NULL;
        for (int x = runs[i].x0; x <= runs[i].x1; x++) {
            int o = orow ? (int)orow[x] : -1;
            hist[r][(o >= 0 && o < NONE) ? o : NONE]++;
        }
    }
    auto owner_of = [&](int r) {
        int bo = NONE, bv = 0;
        auto it = hist.find(r);
        if (it != hist.end())
            for (auto& kv : it->second)
                if (kv.first != NONE && kv.second > bv) { bv = kv.second; bo = kv.first; }
        return bo;
    };
    const int main_owner = owner_of(best);

    int thr = (int)((float)best_area * min_frac);
    if (thr < 1) thr = 1;

    // ── 5) 决定删哪些块 ────────────────────────────────────────────
    std::vector<char> drop_root(nr, 0);
    int dropped = 0;
    for (int i = 0; i < nr; i++) {
        if (uf_find(uf, i) != i) continue;       // 只看根
        if (i == best) continue;
        int co = owner_of(i);
        if (co != NONE && co != main_owner) continue;
        if (area[i] >= thr) continue;
        drop_root[i] = 1;
        dropped += area[i];
    }
    if (!dropped) return 0;

    // ── 6) 只清被删块覆盖的像素 ────────────────────────────────────
    for (int i = 0; i < nr; i++) {
        if (!drop_root[uf_find(uf, i)]) continue;
        float* row = prob + (size_t)runs[i].y * w;
        for (int x = runs[i].x0; x <= runs[i].x1; x++) row[x] = 0.f;
    }
    return dropped;
}
// 按「人体宽度剖面」收紧掩码，试图压掉底部横向蔓延的靠背/桌面。
//
// ⚠️ 默认关闭，原因是实测会切掉深色衣服的手臂：
//   人体前景像素里 58% 饱和度 <0.15（深色衣物/阴影），
//   一旦走下面第 3 步的"高饱和回退"，基准宽度会被严重低估，
//   每行按过窄的 cap 裁剪 —— 近距离时胳膊直接被削掉，且左右抖动
//   （基准宽度逐帧随最宽行跳变）。保留代码与 --profile-k 供继续实验。
//
// 根因假设：人坐在近处时腿被画面下边缘截断，模型无法闭合轮廓，
// 掩码沿底部横向铺开吞掉靠背。
// 做法：以人体上半部分宽度为基准，下方各行不允许超过基准 * k。
//
// 必须在 letterbox 坐标下调用（rgb640/prob640 都是 640x640）。
static void mask_width_profile(float* prob, const unsigned char* rgb,
                              int w, int h, float k) {
    // 1) 找前景竖直范围
    int y0 = -1, y1 = -1;
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            if (prob[y * w + x] > 0.5f) { if (y0 < 0) y0 = y; y1 = y; break; }
        }
    }
    if (y0 < 0) return;

    // 2) 基准宽度：上半段里最宽的一行
    int cut = y0 + (y1 - y0) / 2;
    int wmax = 0;
    for (int y = y0; y <= cut; y++) {
        int n = 0;
        for (int x = 0; x < w; x++) if (prob[y * w + x] > 0.5f) n++;
        if (n > wmax) wmax = n;
    }
    if (wmax <= 0) return;

    // 3) 基准本身接近全宽 => 框就是错的，退回用高饱和像素的宽度
    if (wmax > (int)(0.75f * w)) {
        int satmax = 0;
        for (int y = y0; y <= y1; y++) {
            int n = 0;
            for (int x = 0; x < w; x++) {
                const unsigned char* p = rgb + ((long)y * w + x) * 3;
                int mx = std::max(p[0], std::max(p[1], p[2]));
                int mn = std::min(p[0], std::min(p[1], p[2]));
                if (prob[y * w + x] > 0.5f &&
                    (float)(mx - mn) / (mx + 1e-6f) > 0.15f) n++;
            }
            if (n > satmax) satmax = n;
        }
        if (satmax > 30) wmax = satmax;
    }

    // cap 用整数四舍五入：float 与 double 下 wmax*k 的截断方向不同
    // （200*1.15f=230.0 但 200*1.15=229.9999...），会差 1px。
    // 用 lround 消除平台/语言差异，Python 侧同步用 round()。
    int cap = (int)lround(wmax * (double)k);
    if (cap < 1) cap = 1;

    // 3.5) 躯干中心：只从【上半身】(cap 的推导区) 取各行中点的中位数。
    //      绝不能用被蔓延污染的下半行中点 —— 那会把截取窗口整体拽偏，
    //      表现就是"手臂被切掉"：
    //      y=460 行中点 283，而躯干中心 349，偏 66px，cap=179 的窗口
    //      左边界因此从 260 滑到 194，右侧真手臂被清零。
    int cx = -1;
    {
        int vals[1024], nv = 0;
        for (int y = y0; y <= cut && nv < 1024; y++) {
            int a = -1, b = -1;
            for (int x = 0; x < w; x++)
                if (prob[y * w + x] > 0.5f) { if (a < 0) a = x; b = x; }
            if (a >= 0) vals[nv++] = (a + b) / 2;
        }
        if (nv > 0) {
            std::sort(vals, vals + nv);
            cx = vals[nv / 2];          // 中位数
        }
    }

    // 4) 逐行：只保留最长连续段，超宽则以【躯干中心】为轴截断
    for (int y = y0; y <= y1; y++) {
        float* row = prob + (long)y * w;
        int bestA = -1, bestB = -1, a = -1;
        for (int x = 0; x <= w; x++) {
            bool on = (x < w) && (row[x] > 0.5f);
            if (on) { if (a < 0) a = x; }
            else if (a >= 0) {
                if (x - 1 - a > bestB - bestA) { bestA = a; bestB = x - 1; }
                a = -1;
            }
        }
        if (bestA < 0) continue;
        int span = bestB - bestA + 1;
        if (span <= cap) continue;
        int c = (cx >= 0) ? cx : (bestA + bestB) / 2;
        int lo = std::max(0, c - cap / 2);
        int hi = std::min(w - 1, lo + cap);
        for (int x = 0; x < w; x++)
            if (x < lo || x > hi) row[x] = 0.f;
    }
}

// 把 640 mask（含 letterbox 边）映射到显示尺寸，双线性采样。
// 原来用最近邻，在 640 -> 1280 放大时边缘会有半像素台阶。
// 显示映射表：**在初始化时算一次**，内层循环里不再做任何除法。
//
// 为什么值得单独抽出来（2026-11-05 实测）：
//   原实现在 1280x800 的每个像素上做 2 次浮点除法 + 4 次双线性取值，
//   再加一趟 RGB 最近邻缩放、一趟 alpha 合成 —— 三遍全画面扫描，
//   单帧 29ms，只能跑 15fps，导致摄像头 30fps 有一半帧显示不出来。
//
//   而 scale / pad_t / pad_l 对固定的相机尺寸是**常量**，
//   整套坐标映射在启动时就固定了。所以表可以预计算，
//   内层只剩整数寻址 + 少量乘加；三趟合并成一趟，还省掉 3MB 中间缓冲。
struct DispMap {
    std::vector<int> mx0, mx1;      // mask 双线性：左右列索引
    std::vector<float> wx;          // 水平权重
    std::vector<int> my0, my1;      // mask 双线性：上下行索引
    std::vector<float> wy;          // 垂直权重
    std::vector<int> sx;            // RGB 最近邻：源列索引
    int dw = 0, dh = 0, cw = 0, ch = 0;
    bool rgb_identity = false;      // cw==dw && ch==dh -> 整行 memcpy

    void init(int _dw, int _dh, int _cw, int _ch, float scale, int pad_t,
              int pad_l) {
        dw = _dw; dh = _dh; cw = _cw; ch = _ch;
        mx0.resize(dw); mx1.resize(dw); wx.resize(dw);
        for (int x = 0; x < dw; x++) {
            float sxf = (x + 0.5f) * cw / dw;
            float fx = sxf * scale + pad_l;
            if (fx < 0) fx = 0;
            if (fx > IMG - 1) fx = IMG - 1;
            int x0 = (int)fx;
            mx0[x] = x0;
            mx1[x] = x0 + 1 < IMG ? x0 + 1 : x0;
            wx[x] = fx - x0;
        }
        my0.resize(dh); my1.resize(dh); wy.resize(dh);
        for (int y = 0; y < dh; y++) {
            float syf = (y + 0.5f) * ch / dh;
            float fy = syf * scale + pad_t;
            if (fy < 0) fy = 0;
            if (fy > IMG - 1) fy = IMG - 1;
            int y0 = (int)fy;
            my0[y] = y0;
            my1[y] = y0 + 1 < IMG ? y0 + 1 : y0;
            wy[y] = fy - y0;
        }
        sx.resize(dw);
        for (int x = 0; x < dw; x++) {
            int xx = (int)(x * ((float)cw / dw));
            if (xx >= cw) xx = cw - 1;
            sx[x] = xx;
        }
        rgb_identity = (cw == dw && ch == dh);
    }
};

// 单趟完成「mask 上采样 + RGB 缩放 + alpha 合成」。
// 一次写 out，不再有 scaled 中间缓冲。
// alpha_out 非空时，同时输出每个像素的 alpha（0=背景 1=前景），
// 用于离线量化「半透明虚影」——渲染后无法从像素反解 alpha。
//
// ── 运动补偿：把 640 mask 沿速度方向平移 ───────────────────────────
// mask 相对当前视频帧滞后约 2 帧（实测 masklag≈2.1）。人往右晃时，
// mask 还停在他 2 帧前的位置 —— 于是「身体前沿没有被 mask 覆盖」，
// 合成时那块就被当成背景涂掉了，屏幕上就是人身上出现一片空白。
// 推理速度压不下去（NPU 29ms 是硬成本），但滞后量是已知的，
// 可以用实测速度把它推回来：shift = v * lag。
static void shift_mask(std::vector<float>& dst, const std::vector<float>& src,
                       int W, int H, float dx, float dy) {
    // 限幅：速度估计偶尔会抖，别让它把整个人推歪。
    const float LIM = 10.f;
    if (dx > LIM) dx = LIM;
    if (dx < -LIM) dx = -LIM;
    if (dy > LIM) dy = LIM;
    if (dy < -LIM) dy = -LIM;
    // 取整到像素：640 域上做亚像素双线性要 5.8ms（disp 15.8→21.6ms），
    // 而位移量本身（2 帧 × ~2px）与噪声处在同一量级，亚像素精度没有意义。
    // 改成整行拷贝后只剩内存带宽，实测 <1ms。
    int ix = (int)lroundf(dx);
    int iy = (int)lroundf(dy);
    if (ix == 0 && iy == 0) {
        if (&dst != &src) dst = src;
        return;
    }
    // 目标 (x,y) 取源 (x-ix, y-iy)；越界部分填 0（视为背景），
    // 不能 clamp —— clamp 会把边缘像素拉成一条竖线粘在人身上。
    for (int y = 0; y < H; y++) {
        float* d = dst.data() + (long)y * W;
        int sy = y - iy;
        if (sy < 0 || sy >= H) {
            memset(d, 0, (size_t)W * sizeof(float));
            continue;
        }
        const float* s = src.data() + (long)sy * W;
        if (ix == 0) {
            memcpy(d, s, (size_t)W * sizeof(float));
        } else if (ix > 0) {
            // 目标右移：左边 ix 列补 0，剩余取源 [0, W-ix)
            memset(d, 0, (size_t)ix * sizeof(float));
            memcpy(d + ix, s, (size_t)(W - ix) * sizeof(float));
        } else {
            int n = -ix;
            memcpy(d, s + n, (size_t)(W - n) * sizeof(float));
            memset(d + (W - n), 0, (size_t)n * sizeof(float));
        }
    }
}

static void render_frame(const DispMap& M, const unsigned char* bgr,
                         const float* m640, int bg, unsigned char* out,
                         unsigned char* alpha_out = NULL,
                         float lo = 0.45f, float hi = 0.55f) {
    const int dw = M.dw, dh = M.dh, cw = M.cw;
    // 过渡带倒数预先算好：原来写 (af - lo) / (hi - lo)，编译器无法把
    // 运行时的除法折成乘法，逐像素除法让 disp 从 16.6ms 涨到 17.2ms。
    // 换成乘倒数后回到原位（阈值 /360000 像素）。
    const float inv_band = 1.f / (hi - lo);
    for (int y = 0; y < dh; y++) {
        const float* r0 = m640 + (long)M.my0[y] * IMG;
        const float* r1 = m640 + (long)M.my1[y] * IMG;
        const float wy = M.wy[y], iwy = 1.f - wy;
        int yy = M.rgb_identity ? y : (int)(y * ((float)M.ch / dh));
        if (yy >= M.ch) yy = M.ch - 1;
        const unsigned char* srow = bgr + (long)yy * cw * 3;
        unsigned char* drow = out + (long)y * dw * 3;

        for (int x = 0; x < dw; x++) {
            const int x0 = M.mx0[x], x1 = M.mx1[x];
            const float wxl = M.wx[x], wxr = 1.f - wxl;
            // bg < 0 = 旁路直通：完全不读 mask（连 m640 都不解引用），
            // alpha 恒 1，相机像素原样输出。这样关掉分割时连 mask 依赖都没有，
            // 也避免 m640 为空时算出垃圾值。
            float af;
            if (bg < 0) {
                af = 1.f;
            } else {
                af = (r0[x0] * wxr + r0[x1] * wxl) * iwy +
                     (r1[x0] * wxr + r1[x1] * wxl) * wy;
                if (af < 0.f) af = 0.f;
                if (af > 1.f) af = 1.f;
            }

            // ── 二值化 alpha（2026-11-05，阈值 2026-11-05 晚改为可调）─────
            // 原做法直接把模型概率当 alpha 用，0.15~0.85 的像素半透混合 ——
            // 屏幕上就是人背后一圈灰蒙蒙的虚影，且人一动整片虚影跟着动。
            //
            // ⚠️ 2026-11-05 实测记录（别重犯）：
            // 曾把过渡带从 0.45~0.55 收成硬阈值 0.5，理由是「单帧分析显示
            // 过渡带里 940px 全在轮廓外、硬切只需补 6px 洞」。板端 179 帧
            // 复核推翻了它：躯干核心洞 117 -> 12750（108 倍），洞重心从
            // 高度 0.48 升到 0.28，等于胸口被挖掉一大片 —— 正是最初的
            // 「身上空白」。单帧采样不能代表整个人体，这个结论不可用。
            //
            // 保留 1px 线性过渡带：它确实会让 26% 的背景色漏出来（远处
            // 漏光 ~715px/帧），但那是贴着轮廓的一圈，比在身上打洞轻得多。
            // 两个方向的误差代价不对称，选代价小的那个。
            //
            // ── 阈值扫描（2026-11-05 晚，dump 269 帧原始概率）──────────
            // 痛点是「人像没完全抠出来、后背留白」，即轮廓被吃掉一圈。
            // 对原始概率做阈值扫描，新增像素里紧邻原轮廓(<=4 模型像素)的比例：
            //     lo=0.40  +3.2% 面积   94.4% 贴轮廓
            //     lo=0.35  +6.5% 面积   89.7% 贴轮廓
            //     lo=0.30 +10.1% 面积   83.3% 贴轮廓
            //     lo=0.20 +18.4% 面积   72.5% 贴轮廓
            // => 降阈值主要是在**补轮廓**，不是拉噪声进来。
            // 且方向安全：阈值降低只会减少身上空洞，不会制造空洞
            // （与历史上「硬阈值抬到 0.5」把胸口挖掉相反：那是抬高）。
            // lo/hi 因此做成运行时可调，默认仍 0.45/0.55 保持原行为。
            // ── 旁路直通（bg == -1）已在上面把 af 置 1，这里不再处理 ──
            if (bg >= 0) {
                if (af > hi)      af = 1.f;
                else if (af < lo) af = 0.f;
                else              af = (af - lo) * inv_band;   // lo~hi 线性
            }
            const float iaf = 1.f - af;

            const unsigned char* s = srow + (long)M.sx[x] * 3;
            unsigned char* d = drow + (long)x * 3;
            unsigned char bb, bgc, br;
            if (bg <= 0) {          // 深青绿（bg<0 也会走到这，但 af=1 不影响输出）
                bb = 170; bgc = 178; br = 32;
            } else if (bg == 1) {   // 原图暗化
                bb = s[0] / 3; bgc = s[1] / 3; br = s[2] / 3;
            } else if (bg == 2) {   // 灰
                bb = bgc = br = 90;
            } else {                // 绿幕
                bb = 40; bgc = 200; br = 40;
            }
            d[0] = (unsigned char)(s[0] * af + bb * iaf);
            d[1] = (unsigned char)(s[1] * af + bgc * iaf);
            d[2] = (unsigned char)(s[2] * af + br * iaf);
            if (alpha_out) alpha_out[(long)y * dw + x] = (unsigned char)(af * 255.f + 0.5f);
        }
    }
}


// ── 帧环形缓冲（ring buffer，**不丢帧**） ───────────────────────────
//
// 设计目标（2026-11-05 修正）：
//   摄像头 30fps 必须**每一帧都显示出来**，不能因为 NPU 后处理慢就丢帧。
//   旧实现在这里做「有损丢弃最旧帧」，等于让推理速度决定显示帧率 ——
//   实测输出只有 14fps，也就是每秒丢 16 帧，画面必然一顿一顿。
//
// 正确做法（本实现）：
//   采集线程 → [ring buffer 缓存视频帧] → 显示线程（恒定 30fps 输出每帧）
//                    ↓ 异步
//              推理线程更新「最新 mask」→ 显示线程取用 + 时序平滑
//
//   * ring buffer 用于解耦生产/消费速率，**容量按延迟预算设**，
//     满了说明下游彻底跟不上（真故障），此时丢帧并计数报警，而不是常态丢。
//   * mask 只有一份「最新值」+ 一份「上一帧值」，显示时按 alpha 做 EMA
//     平滑 —— 推理 15fps、显示 30fps 时，中间帧用平滑后的 mask，
//     不会出现 mask 跳变。
template <class T>
class RingBuf {
public:
    explicit RingBuf(size_t cap) : cap_(cap) {}

    // 入队。满则返回 false（调用方计数报警），**不覆盖旧数据**。
    bool push(const T& item) {
        pthread_mutex_lock(&mu_);
        if (q_.size() >= cap_) {
            pthread_mutex_unlock(&mu_);
            total_over_++;
            return false;
        }
        q_.push_back(item);
        total_push_++;
        pthread_cond_signal(&cv_);
        pthread_mutex_unlock(&mu_);
        return true;
    }

    // 有损入队：满时丢**最旧**的一帧再入队，永不为满而拒收。
    // 给「只关心最新画面」的消费者用（推理线程），让慢阶段自己去跳帧，
    // 而不是把生产者（采集）堵住。
    bool push_lossy(const T& item) {
        pthread_mutex_lock(&mu_);
        if (q_.size() >= cap_) {
            q_.pop_front();
            total_drop_++;
        }
        q_.push_back(item);
        total_push_++;
        pthread_cond_signal(&cv_);
        pthread_mutex_unlock(&mu_);
        return true;
    }

    // 阻塞取一帧；closed 且空时返回 false。
    bool pop(T* out) {
        pthread_mutex_lock(&mu_);
        while (q_.empty() && !closed_) pthread_cond_wait(&cv_, &mu_);
        if (q_.empty()) { pthread_mutex_unlock(&mu_); return false; }
        *out = std::move(q_.front());
        q_.pop_front();
        pthread_mutex_unlock(&mu_);
        return true;
    }

    void close() {
        pthread_mutex_lock(&mu_);
        closed_ = true;
        pthread_cond_broadcast(&cv_);
        pthread_mutex_unlock(&mu_);
    }

    size_t depth() {
        pthread_mutex_lock(&mu_);
        size_t n = q_.size();
        pthread_mutex_unlock(&mu_);
        return n;
    }

    double avg_depth() {
        pthread_mutex_lock(&mu_);
        double a = total_push_ ? (double)total_depth_ / total_push_ : 0.0;
        pthread_mutex_unlock(&mu_);
        return a;
    }

    long over() { return total_over_; }

    // push_lossy 丢掉的最旧帧数（推理跳过的帧）
    long dropped() { return total_drop_; }

    RingBuf(const RingBuf&) = delete;
    RingBuf& operator=(const RingBuf&) = delete;

private:
    std::deque<T> q_;
    size_t cap_;
    pthread_mutex_t mu_ = PTHREAD_MUTEX_INITIALIZER;
    pthread_cond_t cv_ = PTHREAD_COND_INITIALIZER;
    bool closed_ = false;
    long total_push_ = 0, total_over_ = 0, total_depth_ = 0, total_drop_ = 0;
};

// ── 流水线共享状态 ──────────────────────────────────────────────────
struct RawFrame {                       // 采集线程 -> 显示线程
    std::vector<unsigned char> bgr;     // 摄像头原始尺寸
    int cw = 0, ch = 0;
    long seq = 0;                       // 采集序号，用于对账
};

// mask 的「最新值」：推理线程写，显示线程读。
// 用 seqlock 风格：写者递增 ver（奇数=写入中），读者读到撕裂就重试。
struct MaskSlot {
    std::vector<float> mask640;         // 640 letterbox 域前景概率
    std::vector<signed char> own640;    // 归属框（平滑后仍保留，供调试）
    float scale = 1.f;
    int pad_t = 0, pad_l = 0;
    long ver = 0;
    long stamp = 0;                     // 对应采集帧号
    // 运动补偿用：人像质心在 640 域里的速度（px / 采集帧），
    // 由推理线程估计，显示线程按「显示帧比 mask 新了多少帧」外推。
    float vx = 0.f, vy = 0.f;
    pthread_mutex_t mu = PTHREAD_MUTEX_INITIALIZER;
};

struct Pipe {
    // 视频帧队列：**只有显示线程消费**（单消费者）。
    // 容量按「最多愿意积压多少帧」定，30fps 下 16 帧 ≈ 0.53s 延迟上限；
    // 超过说明显示链路真的崩了，此时是 push 失败计数（over），不是常态。

    // 2026-11-05 关键修复：此前这个队列有**两个消费者**
    // （infer_thread 和 display_thread 都在 pop），一帧只能被一方取走，
    // 于是「推理取走的帧显示端永远看不到」—— 实测
    // captured=301 / inferred=100 / shown=201，100+201=301，
    // 机械性地丢 33%，与性能无关。改法见 FrameSlot。
    // 存 shared_ptr 而不是 RawFrame 本体：显示端和推理端共享同一份像素，
    // 30fps 下省掉 2.7MB/帧 × 2 的拷贝（约 160MB/s 带宽）。
    RingBuf<std::shared_ptr<const RawFrame>> video{16};

    // 推理线程的「最新帧」槽：采集线程写，推理线程取。
    // 与 video 队列不同，这里是**覆盖式**的：推理慢时只保留最新一帧，
    // 旧帧被覆盖（infer 侧跳帧），显示端完全不受影响。
    struct FrameSlot {
        std::shared_ptr<const RawFrame> cur;
        long ver = 0;                   // 每次写入 +1
        pthread_mutex_t mu = PTHREAD_MUTEX_INITIALIZER;
        pthread_cond_t cv = PTHREAD_COND_INITIALIZER;
        bool closed = false;
    } infer_slot;
};

struct ShmOut;   // 定义在 build_display 之前，Ctx 只持指针

struct Ctx {
    // 配置
    const char* dev = "/dev/video73";
    const char* model = NULL;
    const char* dumpdir = NULL;
    int cw = 1280, ch = 720, dw = 1280, dh = 800;
    int bg = 0, fps = 0, prof = 0, dump_every = 1;
    int verbose_boxes = 0;
    const char* outdir = NULL;      // 合成结果落盘目录（= 屏幕上看到的像素）
    int out_every = 1;              // 每 N 帧存一张
    float profile_k = 0.f;
    float orphan_frac = 0.20f;
    // alpha 二值化过渡带（模型概率阈值）。默认 0.45~0.55 = 历史行为。
    // 调低 lo 会把轮廓外一圈低置信像素补进人体（实测 +6.5% 面积、89.7%
    // 紧贴原轮廓），用于修「没完全抠出来、后背留白」。
    float alpha_lo = 0.45f;
    float alpha_hi = 0.55f;
    float smooth = 0.45f;               // mask 时序平滑 alpha（0=不平滑，1=不更新）
    long maxf = -1;

    // GStreamer
    GstElement* cap = NULL;
    GstElement* sink = NULL;
    GstElement* disp = NULL;
    GstElement* src = NULL;
    GstClock* clock = NULL;
    GstClockTime base = 0;

    // 模型（只被推理线程使用）
    Seg seg;

    Pipe pipe;
    MaskSlot slot;

    // 统计
    long cap_n = 0, infer_n = 0, shown_n = 0;
    long infer_err_n = 0;
    // mask 滞后累计（显示帧号 - mask 对应采集帧号），--fps 时打印。
    // 只在 display_thread 内读写，无跨线程竞争，用普通变量即可。
    double lag_sum = 0;
    long lag_n = 0, lag_max = 0;

    // 运动补偿状态（推理线程写，显示线程只读 slot 里的 vx/vy）
    bool mc_has_prev = false;
    float mc_prev_x = 0.f, mc_prev_y = 0.f;
    long mc_prev_seq = 0;
    float mc_vx = 0.f, mc_vy = 0.f;

    // ── 运行时开关：人像分割 / 虚拟背景 ──────────────────────────────
    // 由桌面设置 GUI 通过控制文件写入，显示线程每帧检查，**不用重启进程**。
    //
    //   seg_on : 0 = 旁路直通（把相机原图直接送屏幕），1 = 人像分割生效
    //   bg_idx : 虚拟背景样式（0 深青绿 / 1 原图暗化 / 2 灰 / 3 绿幕）
    //
    // 用 __atomic 读写（显示线程每帧读一次），控制文件内容见 ctl_path。
    // 关闭分割时推理线程照常跑（不额外省 NPU），这样重新打开是**瞬时**的，
    // 不会出现「打开后黑一下等第一帧 mask」。若要省电可再加工。
    int seg_on = 1;
    int bg_idx = 0;
    const char* ctl_path = NULL;     // 控制文件（内容变化即重读）
    char* ctl_last = NULL;           // 上次读到的内容，用于判重

    // --shm：非空时把每帧像素发到共享内存（给 GTK4 显示进程），
    // 不再推给 waylandsink。两条显示路径像素内容完全一致。
    ShmOut* shm = NULL;
    const char* shm_path = NULL;     // --shm 时的共享内存路径

    struct {
        double letterbox = 0, infer = 0, decode = 0, orphan = 0,
               disp = 0, total = 0;
        long n = 0, n_disp = 0;
    } acc;
};

static GstElement* build_capture(const char* dev, int cw, int ch,
                                 GstElement** sink) {
    gchar* d = g_strdup_printf(
        "v4l2src device=%s io-mode=4 do-timestamp=true "
        "! image/jpeg,width=%d,height=%d,framerate=30/1 "
        "! mppjpegdec ! videoconvert "
        "! video/x-raw,format=BGR,width=%d,height=%d "
        "! appsink name=sink max-buffers=1 drop=true sync=false",
        dev, cw, ch, cw, ch);
    GError* e = NULL;
    GstElement* p = gst_parse_launch(d, &e);
    g_free(d);
    if (!p || e) {
        printf("ERROR: capture: %s\n", e ? e->message : "?");
        if (e) g_error_free(e);
        return NULL;
    }
    *sink = gst_bin_get_by_name(GST_BIN(p), "sink");
    return p;
}

// ── 共享内存帧发布（后端 -> GTK4 显示进程）────────────────────────────
// 为什么要这条通道：Wayland 下应用不能自己决定窗口位置，也没有
// layer-shell，所以**无法**把按钮浮在 waylandsink 的视频层之上。
// 唯一能把齿轮按钮真正压在画面右上角的办法，是让画面本身画在
// GTK4 窗口里。GTK4 头文件板端/SDK 都没有（只有 .so.1 运行时），
// 于是显示端用 Python 写，后端只负责把每帧像素丢过来。
//
// 布局：4 字节小端 uint32 递增序号 + w/h + 像素。读端先读序号，
// 拷数据，再读一次序号；两次相同才认为没被写坏（无锁双检）。
// 写端用内存屏障保证像素先落地、序号后更新。
struct ShmOut {
    int fd = -1;
    unsigned char* base = nullptr;
    size_t bytes = 0;
    unsigned w = 0, h = 0;
    unsigned long seq = 0;
};

static ShmOut* shm_open_out(const char* path, unsigned w, unsigned h) {
    shm_unlink(path);                     // 清掉上次残留
    int fd = open(path, O_CREAT | O_RDWR, 0666);
    if (fd < 0) { perror("shm open"); return nullptr; }
    size_t px = (size_t)w * h * 3;
    size_t total = sizeof(uint32_t) * 4 + px;   // seq,w,h,pad + 像素
    if (ftruncate(fd, (off_t)total) != 0) {
        perror("ftruncate"); close(fd); shm_unlink(path); return nullptr;
    }
    void* p = mmap(NULL, total, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (p == MAP_FAILED) {
        perror("mmap"); close(fd); shm_unlink(path); return nullptr;
    }
    ShmOut* s = new ShmOut();
    s->fd = fd;
    s->base = (unsigned char*)p;
    s->bytes = total;
    s->w = w;
    s->h = h;
    uint32_t* hdr = (uint32_t*)s->base;
    hdr[0] = 0; hdr[1] = w; hdr[2] = h; hdr[3] = 0;
    return s;
}

static inline void shm_publish(ShmOut* s, const unsigned char* px,
                               unsigned w, unsigned h) {
    if (!s || w != s->w || h != s->h) return;
    uint32_t* hdr = (uint32_t*)s->base;
    unsigned char* dst = s->base + sizeof(uint32_t) * 4;
    memcpy(dst, px, (size_t)w * h * 3);
    __sync_synchronize();                 // 像素先落地
    hdr[0] = (uint32_t)(++s->seq);        // 再更新序号
}

static GstElement* build_display(int w, int h, GstElement** src) {
    gchar* d = g_strdup_printf(
        "appsrc name=src is-live=true format=time "
        "caps=video/x-raw,format=BGR,width=%d,height=%d,framerate=30/1 "
        // queue=1：appsink 端已是 max-buffers=1 drop=true，显示端再缓 2 帧
        // 会让画面额外滞后 2 帧（~70ms）。实测我们对「左右晃动跟不上」
        // 的观感要求很敏感，这里收紧到 1 并保留 leaky 丢旧帧，
        // 保证送进 sink 的永远是刚合成好的那一帧。
        "! queue max-size-buffers=1 leaky=downstream "
        "! videoconvert ! videoscale "
        "! video/x-raw,width=%d,height=%d ! waylandsink sync=false",
        w, h, w, h);
    GError* e = NULL;
    GstElement* p = gst_parse_launch(d, &e);
    g_free(d);
    if (!p || e) {
        printf("ERROR: display: %s\n", e ? e->message : "?");
        if (e) g_error_free(e);
        return NULL;
    }
    *src = gst_bin_get_by_name(GST_BIN(p), "src");
    return p;
}

// 采集线程：把每一帧**同时**投给两个下游：
//   * video ring  -> 显示线程（单消费者，不丢帧）
//   * infer_slot  -> 推理线程（覆盖式最新帧，允许跳帧）
// 关键：一帧进两处，而不是两个消费者抢同一个队列。
static void* capture_thread(void* arg) {
    Ctx* c = (Ctx*)arg;
    int idle = 0;
    while (true) {
        GstSample* smp = gst_app_sink_try_pull_sample(GST_APP_SINK(c->sink),
                                                      2 * GST_SECOND);
        if (!smp) {
            // 以前这里直接 break，只要摄像头卡顿 2 秒整个进程就退出，
            // 画面定格在最后一帧 —— 表现就是"点按钮卡死、再也不动"。
            // 现在区分两种情况：
            //   * 管道真的结束（--frames 跑完 / 设备拔出）-> 退出
            //   * 只是一时没帧（USB 抖动、曝光切换）-> 继续等
            if (gst_app_sink_is_eos(GST_APP_SINK(c->sink))) {
                printf("[cap] EOS, stopping\n");
                break;
            }
            // 连续 15 次（约 30s）都没有帧才认定摄像头掉了，避免死循环刷日志
            if (++idle > 15) {
                printf("[cap] camera lost (no frame for 30s), giving up\n");
                break;
            }
            printf("[cap] no frame for 2s, waiting (idle=%d)\n", idle);
            continue;
        }
        idle = 0;
        GstBuffer* buf = gst_sample_get_buffer(smp);
        GstMapInfo im;
        if (!gst_buffer_map(buf, &im, GST_MAP_READ)) {
            gst_sample_unref(smp);
            continue;
        }
        // 一份共享帧，两处引用同一块内存（shared_ptr 计数），
        // 避免为推理再多拷一份 2.7MB。
        auto f = std::make_shared<RawFrame>();
        f->cw = c->cw;
        f->ch = c->ch;
        f->seq = c->cap_n;
        f->bgr.assign(im.data, im.data + (size_t)c->cw * c->ch * 3);
        gst_buffer_unmap(buf, &im);
        gst_sample_unref(smp);

        // 显示端：满则计数报警（真故障），不覆盖
        c->pipe.video.push(std::shared_ptr<const RawFrame>(f));
        __atomic_add_fetch(&c->cap_n, 1, __ATOMIC_RELAXED);

        // 推理端：覆盖式，旧帧被丢弃由推理自己承担
        pthread_mutex_lock(&c->pipe.infer_slot.mu);
        c->pipe.infer_slot.cur = f;
        c->pipe.infer_slot.ver++;
        pthread_cond_signal(&c->pipe.infer_slot.cv);
        pthread_mutex_unlock(&c->pipe.infer_slot.mu);

        if (c->maxf > 0 &&
            __atomic_load_n(&c->shown_n, __ATOMIC_RELAXED) >= c->maxf)
            break;
    }
    c->pipe.video.close();
    pthread_mutex_lock(&c->pipe.infer_slot.mu);
    c->pipe.infer_slot.closed = true;
    pthread_cond_broadcast(&c->pipe.infer_slot.cv);
    pthread_mutex_unlock(&c->pipe.infer_slot.mu);
    return NULL;
}

// 推理线程：从「最新帧槽」取帧，**跳过已经处理过的版本**。
// 它不消费 video 队列（那是显示线程独占的），因此推理慢只会让自己跳帧，
// 不会偷走显示端的帧 —— 这是「显示端零丢帧」的前提。
static void* infer_thread(void* arg) {
    Ctx* c = (Ctx*)arg;
    std::vector<unsigned char> rgb640(IMG * IMG * 3);
    std::vector<float> prob640(IMG * IMG);
    std::vector<signed char> own640(IMG * IMG);
    std::shared_ptr<const RawFrame> rf;

    // 平滑用的上一帧 mask（在 mask 域做 EMA，避免逐帧跳变）
    std::vector<float> smooth(IMG * IMG, 0.f);
    bool have_smooth = false;
    long last_ver = 0;

    while (true) {
        // 取最新帧：ver 没变说明没有新帧，睡在条件变量上等
        pthread_mutex_lock(&c->pipe.infer_slot.mu);
        while (!c->pipe.infer_slot.closed &&
               (c->pipe.infer_slot.ver == last_ver || !c->pipe.infer_slot.cur))
            pthread_cond_wait(&c->pipe.infer_slot.cv, &c->pipe.infer_slot.mu);
        if (!c->pipe.infer_slot.cur ||
            (c->pipe.infer_slot.ver == last_ver && c->pipe.infer_slot.closed)) {
            pthread_mutex_unlock(&c->pipe.infer_slot.mu);
            break;
        }
        rf = c->pipe.infer_slot.cur;        // shared_ptr 拷贝，帧内存不会被回收
        last_ver = c->pipe.infer_slot.ver;
        pthread_mutex_unlock(&c->pipe.infer_slot.mu);

        double t0 = now_us();

        float scale;
        int pad_t, pad_l;
        letterbox(rf->bgr.data(), rf->cw, rf->ch, rgb640.data(),
                  &scale, &pad_t, &pad_l);
        double t_lb = now_us() - t0;

        double t_infer = 0, t_decode = 0;
        std::vector<Box> boxes;
        if (!seg_run(&c->seg, rgb640.data(), prob640.data(), own640.data(),
                     &t_infer, &t_decode, c->verbose_boxes, &boxes)) {
            memset(prob640.data(), 0, prob640.size() * sizeof(float));
            memset(own640.data(), -1, own640.size() * sizeof(signed char));
            c->infer_err_n++;
        }
        double t_o = now_us();
        if (c->orphan_frac > 0.f)
            mask_drop_orphans(prob640.data(), own640.data(), IMG, IMG,
                              c->orphan_frac);
        double t_orph = now_us() - t_o;

        if (c->profile_k > 0.f)
            mask_width_profile(prob640.data(), rgb640.data(), IMG, IMG,
                               c->profile_k);

        // ── 时序稳定：双阈值滞回（Schmitt 触发器）───────────────────────
        // 历史上这里做过两代方案，都失败，记下来免得再走：
        //   1) 固定 EMA a=0.45：阶跃要 3.85 帧到 90%（28fps 下 138ms），
        //      人一左右晃，被挖掉的区域就露出真实物体 =「跟不上」。
        //   2) 不对称（升立即、降 a_fall）：跟上了，但画面**明显锯齿抖动**。
        //      原因是对概率做非对称 EMA 在阈值附近必然振荡：稳态在
        //      0.51↔0.60 之间来回，而渲染的 0.45~0.55 过渡带把这个振荡
        //      映射成 60%↔100% 不透明度跳变 —— 整条轮廓随帧明暗闪。
        //
        // 关键认知：锯齿不是「平滑不够」，而是「在阈值附近做连续运算」。
        // 概率本身在 0.5 附近有逐帧噪声，任何线性滤波都会让它继续在
        // 0.5 附近游走，配合渲染过渡带就是闪烁。
        //
        // 所以改成状态机，不做平均：只有明确越过 HI 才变前景、
        // 明确低于 LO 才变背景，中间区间保持上一帧状态。
        //   * 噪声不再能推动边缘（要跨越整个中间带才翻转）-> 无闪烁
        //   * 不做时间平均 -> 真实运动零滞后（跟上）
        //   * 上下阈值对称 -> 不会像不对称那样单向棘轮式膨胀
        // 空间抗锯齿交给渲染端的双线性上采样（静态渐变，不随时间闪）。
        //
        // 建模输出实测：内部前景 p≈1、背景 p≈0，边缘带才有中间值。
        // HI/LO 只需覆盖「模型噪声幅度」，与用户晃多快无关，因此不会
        // 像自适应阈值那样被真实动作超出。
        if (c->smooth > 0.f && c->smooth < 1.f) {
            const float HI = 0.60f, LO = 0.40f;
            const int NP = IMG * IMG;
            if (!have_smooth) {
                smooth.resize(NP);
                for (int i = 0; i < NP; i++) smooth[i] = (prob640[i] > 0.5f);
                have_smooth = true;
            } else {
                for (int i = 0; i < NP; i++) {
                    const float p = prob640[i];
                    if (p > HI)      smooth[i] = 1.f;
                    else if (p < LO) smooth[i] = 0.f;
                    // 0.40~0.60：保持上一帧判定
                }
            }
        } else {
            smooth = prob640;
            have_smooth = true;
        }

        // dump 用未平滑的原始概率（分析要的是模型真实输出）
        long frame = c->infer_n;
        if (c->dumpdir && c->dump_every > 0 && (frame % c->dump_every) == 0) {
            char fn[512];
            snprintf(fn, sizeof(fn), "%s/y8_%05ld.rgb", c->dumpdir, frame);
            FILE* f = fopen(fn, "wb");
            if (f) { fwrite(rgb640.data(), 1, (size_t)IMG * IMG * 3, f); fclose(f); }
            snprintf(fn, sizeof(fn), "%s/y8_%05ld.f32", c->dumpdir, frame);
            f = fopen(fn, "wb");
            if (f) {
                fwrite(prob640.data(), 1, (size_t)IMG * IMG * sizeof(float), f);
                fclose(f);
            }
            // 检测框落盘：解码只在框内进行，框外强置 0（见 seg_run）。
            // 若模型轮廓超出框，就会被切掉 —— 这正是「近距离留白」的
            // 首要嫌疑，必须能离线复现框边界与被裁像素的关系。
            snprintf(fn, sizeof(fn), "%s/y8_%05ld.box", c->dumpdir, frame);
            f = fopen(fn, "wb");
            if (f) {
                int nb = (int)boxes.size();
                fwrite(&nb, sizeof(int), 1, f);
                for (auto& b : boxes) {
                    float xy[4] = {b.x1, b.y1, b.x2, b.y2};
                    fwrite(xy, sizeof(float), 4, f);
                }
                fclose(f);
            }
        }

        // 发布给显示线程。
        // ⚠️ 不能用 swap：smooth 是跨帧累积的 EMA 状态，swap 会把它换成
        // 槽里的旧内容，下一轮就变成「和垃圾数据做 EMA」。必须拷贝。
        // 代价：15fps 推理 × 1.6MB ≈ 24MB/s，可忽略。
        pthread_mutex_lock(&c->slot.mu);
        c->slot.mask640 = smooth;
        c->slot.scale = scale;
        c->slot.pad_t = pad_t;
        c->slot.pad_l = pad_l;
        c->slot.stamp = rf->seq;
        c->slot.ver++;

        // ── 人像质心速度估计（用于显示端运动补偿）────────────────────
        // 只对前景求和，权重用概率本身，质心由此对边缘抖动不敏感。
        // 速度单位：640 域像素 / 采集帧；跨帧间隔用 seq 差修正，
        // 因为推理会跳帧，不能用「每次推理 = 1 帧」来算。
        //
        // ⚠️ 2026-11-05 实测记录（别重犯）：这个速度估计在**静止**时会
        // 输出 ±0.1~0.3 像素/帧 的噪声（质心受模型概率抖动影响）。显示端
        // 拿它乘 lag(2~3) 再四舍五入到整像素，等于**主动把静止的掩码
        // 来回平移 2~3 像素**。板端实测：静止帧的边缘翻转从 1176px
        // 涨到 2903px（2.5 倍），屏幕上是轮廓整圈锯齿状抖动。
        //
        // 两个修正（缺一不可）：
        //   a) 死区：|v| < DEAD 视为静止，直接归零，不让噪声进补偿
        //   b) 显示端用双线性做亚像素平移，不再四舍五入到整数
        {
            double sw = 0, sx = 0, sy = 0;
            for (int y = 0; y < IMG; y++) {
                const float* r = smooth.data() + (long)y * IMG;
                double rw = 0, rx = 0;
                for (int x = 0; x < IMG; x++) {
                    float p = r[x];
                    if (p <= 0.5f) continue;
                    rw += p;
                    rx += p * x;
                }
                sw += rw;
                sx += rx;
                sy += rw * y;
            }
            if (sw > 200.0) {                 // 前景太小就不更新速度
                float cx = (float)(sx / sw);
                float cy = (float)(sy / sw);
                if (c->mc_has_prev) {
                    long dn = rf->seq - c->mc_prev_seq;
                    if (dn > 0 && dn <= 8) {
                        float nvx = (cx - c->mc_prev_x) / (float)dn;
                        float nvy = (cy - c->mc_prev_y) / (float)dn;
                        // 平滑系数取 0.7：方向反转时旧速度要尽快失效。
                        // 若做长时平均，人一转向 mask 会被推向错误方向，
                        // 恰好在他摆到两端的瞬间把空白放大 —— 那正是
                        // 「左右晃动」最容易看出来的地方。
                        const float k = 0.7f;
                        c->mc_vx = c->mc_vx * (1.f - k) + nvx * k;
                        c->mc_vy = c->mc_vy * (1.f - k) + nvy * k;
                    }
                }
                c->mc_prev_x = cx;
                c->mc_prev_y = cy;
                c->mc_prev_seq = rf->seq;
                c->mc_has_prev = true;
            }
            // a) 死区：静止时质心噪声不参与补偿。
            // 阈值取 0.35 px/帧（640 域）：真人走动远大于此，
            // 而静止噪声实测在 ±0.3 以内。乘以最大 lag(6) 也只有
            // 2.1px，低于一个掩码像素的可见位移。
            const float DEAD = 0.35f;
            float evx = fabsf(c->mc_vx) < DEAD ? 0.f : c->mc_vx;
            float evy = fabsf(c->mc_vy) < DEAD ? 0.f : c->mc_vy;
            c->slot.vx = evx;
            c->slot.vy = evy;
        }
        pthread_mutex_unlock(&c->slot.mu);

        if (c->prof) {
            c->acc.letterbox += t_lb;
            c->acc.infer += t_infer;
            c->acc.decode += t_decode;
            c->acc.orphan += t_orph;
            c->acc.total += now_us() - t0;
            c->acc.n++;
        }
        c->infer_n++;
    }
    return NULL;
}

// 读控制文件（桌面 GUI 写的 "seg=0/1 bg=0/1/2/3"）。
// 只在 mtime 变化时重读，避免每帧 fopen。
// 格式故意做得很宽松：任何顺序、缺项就保留原值，坏行直接忽略。
static void ctl_poll(Ctx* c) {
    if (!c->ctl_path) return;
    // 用**文件内容**而不是 mtime 判变化。
    // 原实现比 st_mtime（整秒），同一秒内连写两次（例如脚本先写 dump=10、
    // 紧接着写 dump=0）第二次会被整个丢掉 —— 实测线上出现 dump 已关但
    // 文件仍在疯长的情况。文件只有几十字节，每次读进来比对开销可忽略。
    FILE* f = fopen(c->ctl_path, "r");
    if (!f) return;
    char buf[512];
    size_t n = fread(buf, 1, sizeof(buf) - 1, f);
    fclose(f);
    buf[n] = 0;
    // 空文件（GUI 正在截断重写的那一瞬间）不能当成"新配置"：
    // 否则 ctl_last 被写成 ""，下一轮真内容进来前所有键都会被重置成默认。
    if (n == 0) return;
    if (c->ctl_last && !strcmp(buf, c->ctl_last)) return;
    free(c->ctl_last);
    c->ctl_last = strdup(buf);

    char line[128];
    int seg = c->seg_on, bg = c->bg_idx;
    int dump = c->dump_every;
    float orphan = c->orphan_frac;
    int margin = c->seg.box_margin;
    int thr = (int)lroundf(c->alpha_lo * 100.f);
    int upb = c->seg.up_bilinear;
    // 逐行扫已读入的 buf（不再用 FILE*，上面已 fclose）。
    for (const char* p = buf; *p; ) {
        int used = 0;
        if (sscanf(p, "%127[^\n]%n", line, &used) != 1) break;
        p += used;
        if (*p == '\n') p++;
        char k[32];
        int v;
        if (sscanf(line, "%31[a-zA-Z_] = %d", k, &v) == 2 ||
            sscanf(line, "%31[a-zA-Z_]=%d", k, &v) == 2) {
            if (!strcmp(k, "seg")) seg = v ? 1 : 0;
            else if (!strcmp(k, "bg")) bg = (v >= 0 && v <= 3) ? v : bg;
            // dump=N：每 N 次推理落一帧到 dump 目录（0 = 关闭）。
            // 存在的意义：现场出问题时能立刻抓真实数据，不用改启动参数
            // 重新拉起进程（重启会打断用户正在看的画面）。
            else if (!strcmp(k, "dump")) dump = (v >= 0 && v <= 1000) ? v : dump;
            // orphan=N：孤立块过滤阈值，百分数（20 = 删除小于主体 20% 的碎块，
            // 0 = 完全关闭）。用于现场 A/B：身上留白若是它造成的，置 0 立刻消失。
            else if (!strcmp(k, "orphan")) orphan = (v >= 0 && v <= 100) ? v / 100.f : orphan;
            // margin=N：解码框向外扩 N 像素（640 域）。用于现场 A/B：
            // 若"留白"是检测框把人体轮廓切掉了，加大它掩码就会变宽。
            else if (!strcmp(k, "margin")) margin = (v >= 0 && v <= 160) ? v : margin;
            // thr=N：alpha 二值化下限，百分数（35 => lo=0.35、hi=0.45）。
            // 调低它把轮廓外一圈低置信像素补回人体，修「没完全抠出来」。
            // 上限 90 保证 hi=lo+0.10 不超过 1.0；低于 5 会满屏误检，故限 5。
            else if (!strcmp(k, "thr")) thr = (v >= 5 && v <= 90) ? v : thr;
            // up=0|1：proto->640 放大方式。1 = 双线性（默认，消锯齿），
            // 0 = 4x4 块复制（历史行为）。用于现场 A/B 看锯齿改善。
            else if (!strcmp(k, "up")) upb = (v == 0 || v == 1) ? v : upb;
        }
    }
    if (seg != c->seg_on) {
        c->seg_on = seg;
        fprintf(stderr, "[ctl] seg -> %s\n", seg ? "ON" : "OFF(bypass)");
    }
    if (bg != c->bg_idx) {
        c->bg_idx = bg;
        fprintf(stderr, "[ctl] bg  -> %d\n", bg);
    }
    if (dump != c->dump_every) {
        c->dump_every = dump;
        // 启动时没给 --dump 时用默认目录，这样 ctl 的 dump 开关能独立生效
        if (c->dump_every > 0 && !c->dumpdir) c->dumpdir = "/data/hs_test/dump_live";
        fprintf(stderr, "[ctl] dump-> 每 %d 次推理落一帧 (%s)\n",
                dump, dump ? "开启" : "关闭");
    }
    if (orphan != c->orphan_frac) {
        c->orphan_frac = orphan;
        fprintf(stderr, "[ctl] orphan -> %.2f (%s)\n",
                orphan, orphan > 0.f ? "过滤开启" : "过滤关闭(保留全部碎块)");
    }
    if (margin != c->seg.box_margin) {
        c->seg.box_margin = margin;
        fprintf(stderr, "[ctl] margin -> %d px (解码框外扩)\n", margin);
    }
    if (thr != (int)lroundf(c->alpha_lo * 100.f)) {
        c->alpha_lo = thr / 100.f;
        c->alpha_hi = c->alpha_lo + 0.10f;
        fprintf(stderr, "[ctl] thr  -> %.2f~%.2f (alpha 过渡带)\n",
                c->alpha_lo, c->alpha_hi);
    }
    if (upb != c->seg.up_bilinear) {
        c->seg.up_bilinear = upb;
        fprintf(stderr, "[ctl] up   -> %d (%s)\n", upb,
                upb ? "双线性上采样，消锯齿" : "4x4 块复制，历史行为");
    }
}

// 显示线程：**每取到一帧视频就显示一帧**，恒定 30fps。
// mask 用 MaskSlot 里的最新值 + 上一次的副本做显示端平滑，
// 保证推理较慢时画面依然连贯。
static void* display_thread(void* arg) {
    Ctx* c = (Ctx*)arg;
    std::vector<unsigned char> out((size_t)c->dw * c->dh * 3);
    std::vector<unsigned char> alpha_dbg((size_t)c->dw * c->dh, 0);
    std::vector<float> m_local(IMG * IMG, 0.f);   // 本线程持有的 mask 副本
    std::vector<float> m_shift(IMG * IMG, 0.f);   // 运动补偿后的 mask
    float mc_vx = 0.f, mc_vy = 0.f;               // 随 mask 一起取出的速度
    DispMap map;
    bool map_ok = false;
    long last_ver = -1;
    long mask_stamp = -1;
    long shown_pre = 0;
    std::shared_ptr<const RawFrame> rf;

    while (c->pipe.video.pop(&rf)) {
        double t0 = now_us();
        shown_pre = __atomic_load_n(&c->shown_n, __ATOMIC_RELAXED);

        // 每帧检查桌面 GUI 的开关（只在 mtime 变化时真的读文件）
        ctl_poll(c);

        // 取最新 mask（有更新才拷，避免每帧 1.6MB 无谓拷贝）
        pthread_mutex_lock(&c->slot.mu);
        long ver = c->slot.ver;
        if (ver != last_ver && !c->slot.mask640.empty()) {
            m_local = c->slot.mask640;
            mask_stamp = c->slot.stamp;      // 这份 mask 对应哪一采集帧
            mc_vx = c->slot.vx;              // 该 mask 的质心速度（640 域/帧）
            mc_vy = c->slot.vy;
            if (!map_ok) {
                map.init(c->dw, c->dh, rf->cw, rf->ch,
                         c->slot.scale, c->slot.pad_t, c->slot.pad_l);
                map_ok = true;
            }
            last_ver = ver;
        }
        pthread_mutex_unlock(&c->slot.mu);

        // 首帧先建表（即便还没 mask 也要能显示视频）
        if (!map_ok) {
            float s; int pt, pl;
            // letterbox 参数只跟相机/模型尺寸有关，可独立推断
            float sc = std::min((float)IMG / rf->ch, (float)IMG / rf->cw);
            int nh = (int)(rf->ch * sc + 0.5f), nw = (int)(rf->cw * sc + 0.5f);
            s = sc; pt = (IMG - nh) / 2; pl = (IMG - nw) / 2;
            map.init(c->dw, c->dh, rf->cw, rf->ch, s, pt, pl);
            map_ok = true;
        }

        // ── 运动补偿：**已废弃，默认关闭**（2026-11-05 实测否定）──────────
        // 原理：mask 对应采集帧 mask_stamp，当前显示的是 rf->seq，两者差
        // lag 帧（实测平均 2.1）。人在这段时间移动了 v*lag 像素，mask 却
        // 停在原位 → 前沿判背景（身上空白）、后沿判前景（旧位置画实物）。
        // 理论上按 v*lag 把 mask 推回去能消除滞后。
        //
        // ⚠️ 板端实测否定了它，别再加回来：
        //   判据「mask 没有更新的相邻两帧，输出必须逐像素相同」
        //     无 MC      : 21 对复用帧 100% 完全一致
        //     MC(旧)     : 仅 54% 一致，差异均值 2836px
        //     MC + 死区  : 仅 29% 一致，差异均值 8524px
        // 即：mask 明明没带来任何新信息，画面却被改了 8000+ 像素。
        // 原因是速度估计在**静止**时也有 ±0.1~0.3px/帧 噪声，乘 lag(2~3)
        // 再 lroundf 取整 → 静止的 mask 被来回平移 2~3px，屏幕上就是
        // 轮廓整圈锯齿状抖动。加死区反而更糟（速度被门限切断后更不稳）。
        //
        // 正解是不补偿：lag 2.1 帧的滞后本身可接受（masklag 指标），
        // 「完全没有凭空改动」比「补偿不准」好得多。保留代码供日后用
        // HY8_MC=1 复核，默认走下面的 sh=0 分支。
        float sh_x = 0.f, sh_y = 0.f;
        if (getenv("HY8_MC") && mask_stamp >= 0) {
            long lag = rf->seq - mask_stamp;
            if (lag > 0 && lag <= 6) {
                sh_x = mc_vx * (float)lag;
                sh_y = mc_vy * (float)lag;
            }
        }
        if (getenv("HY8_DBG_MC") && (shown_pre % 10) == 0)
            fprintf(stderr, "[mc] lag=%ld vx=%.2f vy=%.2f sh=%.2f,%.2f\n",
                    (long)(rf->seq - mask_stamp), mc_vx, mc_vy, sh_x, sh_y);
        if (fabsf(sh_x) > 0.01f || fabsf(sh_y) > 0.01f) {
            shift_mask(m_shift, m_local, IMG, IMG, sh_x, sh_y);
            render_frame(map, rf->bgr.data(), m_shift.data(), c->bg_idx, out.data(),
                         c->outdir ? alpha_dbg.data() : NULL,
                         c->alpha_lo, c->alpha_hi);
        } else if (c->seg_on) {
            render_frame(map, rf->bgr.data(), m_local.data(), c->bg_idx, out.data(),
                         c->outdir ? alpha_dbg.data() : NULL,
                         c->alpha_lo, c->alpha_hi);
        } else {
            // ── 旁路直通：关掉人像分割 / 虚拟背景 ────────────────────
            // 把相机原图按同一套 DispMap 直接铺到屏幕（只做缩放，
            // 不掺任何 mask/背景色）。必须仍走 render_frame 的坐标映射，
            // 否则相机的 1280x720 与屏幕 1280x800 长宽比不同会拉伸变形。
            // 用一个「全 1」的 alpha 达到相同效果，代价只是每帧多读
            // 1.6MB 常量 mask；实测 disp 无明显变化，换来切换逻辑只有一处。
            render_frame(map, rf->bgr.data(), m_local.data(), -1, out.data(),
                         c->outdir ? alpha_dbg.data() : NULL);
        }

        // 落盘「送进 waylandsink 的像素」= 屏幕真实内容，用于离线核对。
        // 同时记下这一帧用的 mask 版本/对应采集帧号与视频帧号，
        // 用来量化 mask 滞后（虚影）到底有多少帧。
        if (c->outdir && (shown_pre % c->out_every) == 0) {
            char fn[512];
            snprintf(fn, sizeof(fn), "%s/o_%05ld_%ld_%ld.bgr", c->outdir,
                     shown_pre, rf->seq, mask_stamp);
            FILE* f = fopen(fn, "wb");
            if (f) {
                fwrite(out.data(), 1, (size_t)c->dw * c->dh * 3, f);
                fclose(f);
            }
            snprintf(fn, sizeof(fn), "%s/o_%05ld_%ld_%ld.a8", c->outdir,
                     shown_pre, rf->seq, mask_stamp);
            f = fopen(fn, "wb");
            if (f) {
                fwrite(alpha_dbg.data(), 1, (size_t)c->dw * c->dh, f);
                fclose(f);
            }
            // 同时落一份「相机原图」。判「留白」必须有真实参照：
            // 屏幕上那块背景色，到底原本是全黑、是椅子、还是**人**。
            // 只看合成后的图无法区分，会把「相机本来就没拍到」误判成丢人。
            snprintf(fn, sizeof(fn), "%s/src_%05ld.bgr", c->outdir, shown_pre);
            f = fopen(fn, "wb");
            if (f) {
                const unsigned char* sp = rf->bgr.data();
                fwrite(sp, 1, (size_t)rf->cw * rf->ch * 3, f);
                fclose(f);
            }
        }

        GstBuffer* ob = gst_buffer_new_allocate(NULL, (gsize)c->dw * c->dh * 3, NULL);
        GstMapInfo om;
        gst_buffer_map(ob, &om, GST_MAP_WRITE);
        memcpy(om.data, out.data(), (size_t)c->dw * c->dh * 3);
        gst_buffer_unmap(ob, &om);
        GST_BUFFER_PTS(ob) = gst_clock_get_time(c->clock) - c->base;
        GST_BUFFER_DTS(ob) = GST_BUFFER_PTS(ob);
        GST_BUFFER_DURATION(ob) = gst_util_uint64_scale(GST_SECOND, 1, 30);
        // --shm：把这一帧交给 GTK4 显示进程（右上角齿轮按钮那套 UI）。
        // 送进去的像素与 waylandsink 路径**完全相同**，两条路可 A/B。
        if (c->shm) {
            // shm 路径不需要 GstBuffer，必须显式释放。
            // 之前这里直接落到下面 push 分支之外，每帧泄漏 dw*dh*3 = 3MB，
            // 1000 帧约 3GB 之后 mpp 解码缓冲分配失败 ->
            // "mpp_buffer: check buffer found NULL pointer" -> EOS -> 进程退出。
            // 表现就是跑 35~40 秒后画面永远定格。
            gst_buffer_unref(ob);
            shm_publish(c->shm, out.data(), c->dw, c->dh);
        } else if (gst_app_src_push_buffer(GST_APP_SRC(c->src), ob) != GST_FLOW_OK) {
            // push 失败时 appsrc 不接管所有权，这里要自己释放
            gst_buffer_unref(ob);
            break;
        }

        if (c->prof) {
            c->acc.disp += now_us() - t0;
            c->acc.n_disp++;
        }
        long shown = __atomic_add_fetch(&c->shown_n, 1, __ATOMIC_RELAXED);
        shown_pre = shown - 1;

        // ── 真实 mask 滞后统计（2026-11-05）──────────────────────────
        // 之前靠 dump 文件名离线算 lag，但 dump 本身改变时序，且不同次
        // 采集的场景完全不同（前景像素数能差 50 倍），根本无法 A/B。
        // 这里在进程内直接统计：显示的视频帧序号 - 该 mask 对应的采集
        // 帧序号。这是用户能感知的错位量，且不受 dump 影响。
        // 用普通变量即可：lag_n/lag_sum/lag_max 的写入与下方打印
        // 都在 display_thread 内完成，不存在跨线程竞争。
        if (mask_stamp >= 0 && c->fps) {
            long l = rf->seq - mask_stamp;
            c->lag_sum += (double)l;
            c->lag_n++;
            if (l > c->lag_max) c->lag_max = l;
        }
        if (c->fps && shown % 30 == 0) {
            double sec = (gst_clock_get_time(c->clock) - c->base) / 1e9;
            long cap = __atomic_load_n(&c->cap_n, __ATOMIC_RELAXED);
            long inf = __atomic_load_n(&c->infer_n, __ATOMIC_RELAXED);
            // 显示端丢帧率：这才是用户能看到的指标
            double drop_pct = cap ? 100.0 * (1.0 - (double)shown / (double)cap) : 0.0;
            // 推理跳帧数：cap - infer 是推理没处理的帧（设计如此，不是丢帧）
            printf("shown=%ld  %.1f fps  | cap=%ld infer=%ld skip=%ld  "
                   "show-drop=%.2f%%  vq=%zu over=%ld  masklag=%.2f/%.0f\n",
                   shown, shown / (sec > 0 ? sec : 1), cap, inf, cap - inf,
                   drop_pct, c->pipe.video.depth(), c->pipe.video.over(),
                   c->lag_n ? c->lag_sum / (double)c->lag_n : -1.0,
                   (double)c->lag_max);
            if (c->prof && c->acc.n > 0) {
                double n = (double)c->acc.n;
                double nd = c->acc.n_disp ? (double)c->acc.n_disp : 1.0;
                printf("  [prof avg us] infer %.0f + decode %.0f |"
                       " letterbox %.0f | orphan %.0f | disp %.0f |"
                       " infer-loop %.0f\n",
                       c->acc.infer / n, c->acc.decode / n,
                       c->acc.letterbox / n, c->acc.orphan / n,
                       c->acc.disp / nd, c->acc.total / n);
            }
            fflush(stdout);
        }
    }
    return NULL;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        printf("usage: %s <model.rknn> [--device /dev/video73] [--cam WxH]\n"
               "          [--disp WxH] [--bg 0-3] [--fps] [--prof] [--frames N]\n"
               "          [--dump DIR] [--dump-every N]\n", argv[0]);
        return -1;
    }
    static Ctx ctx;
    ctx.model = argv[1];

    for (int i = 2; i < argc; i++) {
        if (!strcmp(argv[i], "--device") && i + 1 < argc) ctx.dev = argv[++i];
        else if (!strcmp(argv[i], "--cam") && i + 1 < argc)
            sscanf(argv[++i], "%dx%d", &ctx.cw, &ctx.ch);
        else if (!strcmp(argv[i], "--disp") && i + 1 < argc)
            sscanf(argv[++i], "%dx%d", &ctx.dw, &ctx.dh);
        else if (!strcmp(argv[i], "--bg") && i + 1 < argc) ctx.bg = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--ctl") && i + 1 < argc) ctx.ctl_path = argv[++i];
        else if (!strcmp(argv[i], "--shm") && i + 1 < argc) ctx.shm_path = argv[++i];
        else if (!strcmp(argv[i], "--fps")) ctx.fps = 1;
        else if (!strcmp(argv[i], "--prof")) ctx.prof = 1;
        else if (!strcmp(argv[i], "--frames") && i + 1 < argc)
            ctx.maxf = atol(argv[++i]);
        else if (!strcmp(argv[i], "--dump") && i + 1 < argc) ctx.dumpdir = argv[++i];
        else if (!strcmp(argv[i], "--dump-every") && i + 1 < argc)
            ctx.dump_every = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--profile-k") && i + 1 < argc)
            ctx.profile_k = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "--noprofile")) ctx.profile_k = 0.f;
        else if (!strcmp(argv[i], "--orphan-frac") && i + 1 < argc)
            ctx.orphan_frac = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "--keep-orphans")) ctx.orphan_frac = 0.f;
        // --thr=N：alpha 二值化下限百分数（35 => 0.35~0.45）。
        // 调低补回轮廓外一圈低置信像素；默认 45 = 历史行为。
        else if (!strcmp(argv[i], "--thr") && i + 1 < argc) {
            int t = atoi(argv[++i]);
            if (t >= 5 && t <= 90) {
                ctx.alpha_lo = t / 100.f;
                ctx.alpha_hi = ctx.alpha_lo + 0.10f;
            }
        }
        else if (!strcmp(argv[i], "--dump-boxes")) ctx.verbose_boxes = 1;
        else if (!strcmp(argv[i], "--out") && i + 1 < argc) ctx.outdir = argv[++i];
        else if (!strcmp(argv[i], "--out-every") && i + 1 < argc)
            ctx.out_every = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--smooth") && i + 1 < argc)
            ctx.smooth = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "--nosmooth")) ctx.smooth = 0.f;
    }
    // 把命令行 --bg 同步给运行时开关的初值（render_frame 只认 bg_idx）。
    ctx.bg_idx = ctx.bg;
    setvbuf(stdout, NULL, _IOLBF, 0);
    printf("config: dev=%s cam=%dx%d disp=%dx%d bg=%d profile_k=%.2f "
           "orphan_frac=%.2f smooth=%.2f video_ring=%d\n",
           ctx.dev, ctx.cw, ctx.ch, ctx.dw, ctx.dh, ctx.bg, ctx.profile_k,
           ctx.orphan_frac, ctx.smooth, 16);

    if (!seg_init(&ctx.seg, ctx.model)) return -1;

    gst_init(&argc, &argv);
    ctx.cap = build_capture(ctx.dev, ctx.cw, ctx.ch, &ctx.sink);
    if (!ctx.cap) return -1;
    if (ctx.shm_path) {
        // GTK4 显示进程负责上屏，后端这套 waylandsink 完全不需要建。
        // 省一个窗口 + 一份 videoconvert，也避免抢显示资源。
        ctx.shm = shm_open_out(ctx.shm_path, ctx.dw, ctx.dh);
        if (!ctx.shm) return -1;
        printf("[shm] publishing %dx%d to %s\n", ctx.dw, ctx.dh, ctx.shm_path);
    } else {
        ctx.disp = build_display(ctx.dw, ctx.dh, &ctx.src);
        if (!ctx.disp) return -1;
    }

    ctx.clock = gst_system_clock_obtain();
    ctx.base = gst_clock_get_time(ctx.clock);

    gst_element_set_state(ctx.cap, GST_STATE_PLAYING);
    if (ctx.disp) gst_element_set_state(ctx.disp, GST_STATE_PLAYING);

    pthread_t th_cap, th_inf, th_disp;
    pthread_create(&th_cap, NULL, capture_thread, &ctx);
    pthread_create(&th_inf, NULL, infer_thread, &ctx);
    // shm 模式下没有 waylandsink 管道，display 线程依然负责合成 + 发布到
    // 共享内存（它是唯一渲染的地方），所以照常启动。
    pthread_create(&th_disp, NULL, display_thread, &ctx);

    pthread_join(th_cap, NULL);
    pthread_join(th_inf, NULL);
    pthread_join(th_disp, NULL);

    long cap = __atomic_load_n(&ctx.cap_n, __ATOMIC_RELAXED);
    long inf = __atomic_load_n(&ctx.infer_n, __ATOMIC_RELAXED);
    long shown = __atomic_load_n(&ctx.shown_n, __ATOMIC_RELAXED);
    printf("total captured=%ld inferred=%ld shown=%ld  "
           "video_ring overflows=%ld\n",
           cap, inf, shown, ctx.pipe.video.over());
    if (cap > 0) {
        printf("frame accounting: captured=%ld shown=%ld  lost=%.2f%%  "
               "(infer skipped %ld = design, not drop)\n",
               cap, shown, 100.0 * (1.0 - (double)shown / (double)cap),
               cap - inf);
    }
    if (ctx.prof && ctx.acc.n > 0) {
        double n = (double)ctx.acc.n;
        printf("[prof final avg us] infer %.1f | decode %.1f | letterbox %.1f |"
               " orphan %.1f | infer-loop %.1f\n",
               ctx.acc.infer / n, ctx.acc.decode / n, ctx.acc.letterbox / n,
               ctx.acc.orphan / n, ctx.acc.total / n);
    }

    if (ctx.src) gst_app_src_end_of_stream(GST_APP_SRC(ctx.src));
    gst_element_set_state(ctx.cap, GST_STATE_NULL);
    if (ctx.disp) gst_element_set_state(ctx.disp, GST_STATE_NULL);
    gst_object_unref(ctx.sink);
    if (ctx.src) gst_object_unref(ctx.src);
    gst_object_unref(ctx.cap);
    if (ctx.disp) gst_object_unref(ctx.disp);
    gst_object_unref(ctx.clock);
    rknn_destroy(ctx.seg.ctx);
    printf("done\n");
    return 0;
}
