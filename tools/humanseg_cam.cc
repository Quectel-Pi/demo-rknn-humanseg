// 板端 demo：USB 摄像头实时人像分割 + 虚拟背景，显示到桌面
//
// 链路：
//   v4l2src(MJPG) -> mppjpegdec(硬解) -> videoconvert(BGR) -> appsink
//     -> 缩放到 398x224 -> RKNN -> 2ch logits -> argmax
//     -> mask 双线性上采样 -> 与原始 BGR 帧 alpha 混合 -> 虚拟背景合成
//     -> appsrc -> videoconvert -> waylandsink
//
// 模型输出：hotseg rknn 是 **Softmax 之前截断**的 2 通道 logits，
// 布局为 (1, W, H, 2)（ONNX 尾部 Transpose perm=[0,3,2,1] 把 H/W 也换了）。
//
// 编译（本仓库根目录）：
//   export GCC_COMPILER=<SDK>/prebuilts/gcc/linux-x86/aarch64/gcc-arm-10.3-2021.07-x86_64-aarch64-none-linux-gnu/bin/aarch64-none-linux-gnu
//   $GCC_COMPILER-g++ -O2 -std=c++11 tools/humanseg_cam.cc \
//     -I3rdparty/gst/include/gstreamer-1.0 \
//     -I3rdparty/gst/include/glib-2.0 \
//     -I3rdparty/gst/include/glib-2.0-internal \
//     -I3rdparty/rknn_api/include -o /tmp/humanseg_cam \
//     -L3rdparty/rknn_api/lib -lrknnrt \
//     -L3rdparty/gst/lib -lgstreamer-1.0 -lgstapp-1.0 -lgstvideo-1.0 \
//     -lgobject-2.0 -lglib-2.0 \
//     -Wl,-rpath-link,3rdparty/gst/lib -Wl,--allow-shlib-undefined -lpthread
//
// 运行（板端，桌面用户身份）：
//   sudo -u qq env XDG_RUNTIME_DIR=/run/user/1000 WAYLAND_DISPLAY=wayland-0 \
//     LD_LIBRARY_PATH=/data/hs_test:/usr/lib ./humanseg_cam <model.rknn> [选项]
//
// 选项：
//   --device /dev/video73   摄像头节点（默认自动探测 UGREEN/Camera）
//   --cam 1280x720          采集分辨率
//   --disp 1280x800         显示窗口尺寸
//   --bg 0|1|2|3            背景模式：0 纯色 1 原图暗化 2 灰 3 绿幕
//   --fps                   打印统计信息
//   --frames N              跑 N 帧后退出（默认无限）

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <gst/app/gstappsink.h>
#include <gst/app/gstappsrc.h>
#include <gst/gst.h>
#include <gst/video/video.h>

#include "rknn_api.h"

#define MODEL_W 398
#define MODEL_H 224
#define MODEL_C 3
#define IN_SIZE (MODEL_W * MODEL_H * MODEL_C)

// ── 模型 ────────────────────────────────────────────────────────────
struct Seg {
    rknn_context ctx = 0;
    int out_w = 0, out_h = 0;
};

static bool seg_init(Seg* s, const char* model_path) {
    FILE* fp = fopen(model_path, "rb");
    if (!fp) {
        printf("ERROR: cannot open model %s\n", model_path);
        return false;
    }
    fseek(fp, 0, SEEK_END);
    long sz = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    unsigned char* md = (unsigned char*)malloc(sz);
    if (!md || fread(md, 1, sz, fp) != (size_t)sz) {
        printf("ERROR: read model failed\n");
        if (md) free(md);
        fclose(fp);
        return false;
    }
    fclose(fp);

    if (rknn_init(&s->ctx, md, sz, 0, NULL) < 0) {
        printf("ERROR: rknn_init failed\n");
        free(md);
        return false;
    }
    free(md);

    rknn_tensor_attr oa;
    memset(&oa, 0, sizeof(oa));
    oa.index = 0;
    if (rknn_query(s->ctx, RKNN_QUERY_OUTPUT_ATTR, &oa, sizeof(oa)) < 0) {
        printf("ERROR: query output failed\n");
        return false;
    }
    if (oa.n_dims != 4 || oa.dims[3] != 2) {
        printf("ERROR: unexpected output dims [%d,%d,%d,%d]，"
               "期望 (1,W,H,2)\n", oa.dims[0], oa.dims[1], oa.dims[2],
               oa.dims[3]);
        return false;
    }
    s->out_w = oa.dims[1];      // (1, W, H, 2)
    s->out_h = oa.dims[2];
    printf("model: %s\n  logits %dx%d x2  (%d floats)\n", model_path,
           s->out_w, s->out_h, oa.n_elems);
    return true;
}

// RGB888 (MODEL_W x MODEL_H) -> 前景概率图 mask (MODEL_H*MODEL_W, 0..255)
static bool seg_run(Seg* s, const unsigned char* rgb, unsigned char* mask) {
    rknn_input in;
    memset(&in, 0, sizeof(in));
    in.index = 0;
    in.type = RKNN_TENSOR_UINT8;
    in.size = IN_SIZE;
    in.fmt = RKNN_TENSOR_NHWC;
    in.buf = (void*)rgb;
    in.pass_through = 0;
    if (rknn_inputs_set(s->ctx, 1, &in) < 0) return false;
    if (rknn_run(s->ctx, NULL) < 0) return false;

    rknn_output out;
    memset(&out, 0, sizeof(out));
    out.index = 0;
    out.want_float = 1;
    out.is_prealloc = 0;
    if (rknn_outputs_get(s->ctx, 1, &out, NULL) < 0) return false;

    const float* f = (const float*)out.buf;
    const int W_ = s->out_w, H_ = s->out_h;
    // 输出布局是 (N, W, H, 2)，不是标准 NHWC —— ONNX 尾部 Transpose 的
    // perm=[0,3,2,1] 作用在 NCHW 上会把 H/W 一起换掉。
    // 行主序下索引 = ((w * H) + h) * 2 + c，w 轴步长是 H*2，不是 W*2。
    for (int h = 0; h < H_; h++) {
        for (int w = 0; w < W_; w++) {
            long base = ((long)w * H_ + h) * 2;
            float d = f[base + 1] - f[base];      // fg - bg
            float p = 1.0f / (1.0f + expf(-d));   // logistic，边缘柔和
            mask[(long)h * W_ + w] = (unsigned char)(p * 255.0f + 0.5f);
        }
    }
    rknn_outputs_release(s->ctx, 1, &out);
    return true;
}

// ── 图像工具 ────────────────────────────────────────────────────────
// BGR src (sw x sh) -> RGB dst (MODEL_W x MODEL_H)，双线性
static void bgr_to_model_rgb(const unsigned char* src, int sw, int sh,
                             unsigned char* dst) {
    float sx = (float)sw / MODEL_W;
    float sy = (float)sh / MODEL_H;
    for (int y = 0; y < MODEL_H; y++) {
        float fy = (y + 0.5f) * sy - 0.5f;
        if (fy < 0) fy = 0;
        if (fy > sh - 1) fy = sh - 1;
        int y0 = (int)fy, y1 = y0 + 1 < sh ? y0 + 1 : y0;
        float wy = fy - y0;
        for (int x = 0; x < MODEL_W; x++) {
            float fx = (x + 0.5f) * sx - 0.5f;
            if (fx < 0) fx = 0;
            if (fx > sw - 1) fx = sw - 1;
            int x0 = (int)fx, x1 = x0 + 1 < sw ? x0 + 1 : x0;
            float wx = fx - x0;
            const unsigned char* p00 = src + ((long)y0 * sw + x0) * 3;
            const unsigned char* p01 = src + ((long)y0 * sw + x1) * 3;
            const unsigned char* p10 = src + ((long)y1 * sw + x0) * 3;
            const unsigned char* p11 = src + ((long)y1 * sw + x1) * 3;
            unsigned char* d = dst + ((long)y * MODEL_W + x) * 3;
            for (int c = 0; c < 3; c++) {
                // src 是 BGR，dst 要 RGB -> 反转通道
                int sc = 2 - c;
                float v = (p00[sc] * (1 - wx) + p01[sc] * wx) * (1 - wy) +
                          (p10[sc] * (1 - wx) + p11[sc] * wx) * wy;
                d[c] = (unsigned char)(v + 0.5f);
            }
        }
    }
}

// mask (MODEL_H x MODEL_W) 双线性放大到 (dw x dh)，存到 out
static void upscale_mask(const unsigned char* mask, int dw, int dh,
                         unsigned char* out) {
    float sx = (float)MODEL_W / dw;
    float sy = (float)MODEL_H / dh;
    for (int y = 0; y < dh; y++) {
        float fy = (y + 0.5f) * sy - 0.5f;
        if (fy < 0) fy = 0;
        if (fy > MODEL_H - 1) fy = MODEL_H - 1;
        int y0 = (int)fy, y1 = y0 + 1 < MODEL_H ? y0 + 1 : y0;
        float wy = fy - y0;
        for (int x = 0; x < dw; x++) {
            float fx = (x + 0.5f) * sx - 0.5f;
            if (fx < 0) fx = 0;
            if (fx > MODEL_W - 1) fx = MODEL_W - 1;
            int x0 = (int)fx, x1 = x0 + 1 < MODEL_W ? x0 + 1 : x0;
            float wx = fx - x0;
            float v = (mask[(long)y0 * MODEL_W + x0] * (1 - wx) +
                       mask[(long)y0 * MODEL_W + x1] * wx) * (1 - wy) +
                      (mask[(long)y1 * MODEL_W + x0] * (1 - wx) +
                       mask[(long)y1 * MODEL_W + x1] * wx) * wy;
            out[(long)y * dw + x] = (unsigned char)(v + 0.5f);
        }
    }
}

// 合成：src BGR (w x h) + mask -> dst BGR
static void composite(const unsigned char* src, const unsigned char* mask,
                      int w, int h, int bg_mode, unsigned char* dst) {
    static std::vector<unsigned char> big;
    if ((int)big.size() < w * h) big.resize(w * h);
    upscale_mask(mask, w, h, big.data());

    for (int i = 0; i < w * h; i++) {
        int a = big[i];
        int ia = 255 - a;
        const unsigned char* s = src + (long)i * 3;
        unsigned char* d = dst + (long)i * 3;
        unsigned char bb, bg, br;      // BGR
        if (bg_mode == 0) {            // 纯色：深青绿
            bb = 170; bg = 178; br = 32;
        } else if (bg_mode == 1) {     // 原图暗化（近似"模糊背景"）
            bb = s[0] / 3; bg = s[1] / 3; br = s[2] / 3;
        } else if (bg_mode == 2) {     // 灰
            bb = bg = br = 90;
        } else {                       // 绿幕
            bb = 40; bg = 200; br = 40;
        }
        d[0] = (unsigned char)((s[0] * a + bb * ia) / 255);
        d[1] = (unsigned char)((s[1] * a + bg * ia) / 255);
        d[2] = (unsigned char)((s[2] * a + br * ia) / 255);
    }
}

// ── GStreamer ───────────────────────────────────────────────────────
static GstElement* build_capture(const char* device, int cam_w, int cam_h,
                                 GstElement** appsink_out) {
    gchar* desc = g_strdup_printf(
        "v4l2src device=%s io-mode=4 do-timestamp=true "
        "! image/jpeg,width=%d,height=%d,framerate=30/1 "
        "! mppjpegdec "
        "! videoconvert "
        "! video/x-raw,format=BGR,width=%d,height=%d "
        "! appsink name=sink max-buffers=1 drop=true sync=false",
        device, cam_w, cam_h, cam_w, cam_h);
    GError* err = NULL;
    GstElement* p = gst_parse_launch(desc, &err);
    g_free(desc);
    if (!p || err) {
        printf("ERROR: capture pipeline: %s\n", err ? err->message : "unknown");
        if (err) g_error_free(err);
        return NULL;
    }
    *appsink_out = gst_bin_get_by_name(GST_BIN(p), "sink");
    return p;
}

static GstElement* build_display(int w, int h, GstElement** appsrc_out) {
    gchar* desc = g_strdup_printf(
        "appsrc name=src is-live=true format=time "
        "caps=video/x-raw,format=BGR,width=%d,height=%d,framerate=30/1 "
        "! queue max-size-buffers=2 leaky=downstream "
        "! videoconvert ! videoscale "
        "! video/x-raw,width=%d,height=%d "
        "! waylandsink sync=false",
        w, h, w, h);
    GError* err = NULL;
    GstElement* p = gst_parse_launch(desc, &err);
    g_free(desc);
    if (!p || err) {
        printf("ERROR: display pipeline: %s\n", err ? err->message : "unknown");
        if (err) g_error_free(err);
        return NULL;
    }
    *appsrc_out = gst_bin_get_by_name(GST_BIN(p), "src");
    return p;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        printf("usage: %s <model.rknn> [--device /dev/videoN] [--cam WxH]\n"
               "          [--disp WxH] [--bg 0-3] [--fps] [--frames N]\n",
               argv[0]);
        return -1;
    }

    const char* model_path = argv[1];
    const char* device = "/dev/video73";
    int cam_w = 1280, cam_h = 720;
    int disp_w = 1280, disp_h = 800;
    int bg_mode = 0;
    int show_fps = 0;
    long max_frames = -1;

    for (int i = 2; i < argc; i++) {
        if (!strcmp(argv[i], "--device") && i + 1 < argc) {
            device = argv[++i];
        } else if (!strcmp(argv[i], "--cam") && i + 1 < argc) {
            sscanf(argv[++i], "%dx%d", &cam_w, &cam_h);
        } else if (!strcmp(argv[i], "--disp") && i + 1 < argc) {
            sscanf(argv[++i], "%dx%d", &disp_w, &disp_h);
        } else if (!strcmp(argv[i], "--bg") && i + 1 < argc) {
            bg_mode = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--fps")) {
            show_fps = 1;
        } else if (!strcmp(argv[i], "--frames") && i + 1 < argc) {
            max_frames = atol(argv[++i]);
        } else {
            printf("unknown option: %s\n", argv[i]);
            return -1;
        }
    }

    printf("config: device=%s cam=%dx%d disp=%dx%d bg=%d\n", device, cam_w,
           cam_h, disp_w, disp_h, bg_mode);

    Seg seg;
    if (!seg_init(&seg, model_path)) return -1;

    gst_init(&argc, &argv);

    GstElement* cap_sink = NULL;
    GstElement* cap = build_capture(device, cam_w, cam_h, &cap_sink);
    if (!cap) return -1;
    GstElement* disp_src = NULL;
    GstElement* disp = build_display(disp_w, disp_h, &disp_src);
    if (!disp) return -1;

    // 采集分辨率与显示分辨率不同：合成前把 BGR 帧缩放到显示尺寸
    std::vector<unsigned char> scaled((size_t)disp_w * disp_h * 3);
    std::vector<unsigned char> composited((size_t)disp_w * disp_h * 3);
    std::vector<unsigned char> model_rgb(IN_SIZE);
    std::vector<unsigned char> mask(MODEL_W * MODEL_H);

    gst_element_set_state(cap, GST_STATE_PLAYING);
    gst_element_set_state(disp, GST_STATE_PLAYING);

    GstClock* clock = gst_system_clock_obtain();
    GstClockTime base = gst_clock_get_time(clock);
    long frame = 0;
    long ok = 0, fail_seg = 0, fail_pull = 0;
    GstClockTime t_start = gst_clock_get_time(clock);

    while (max_frames < 0 || frame < max_frames) {
        GstSample* sample = gst_app_sink_try_pull_sample(
            GST_APP_SINK(cap_sink), 2 * GST_SECOND);
        if (!sample) {
            fail_pull++;
            printf("pull timeout/fail (frame %ld)\n", frame);
            if (fail_pull > 10) break;
            continue;
        }
        GstBuffer* inbuf = gst_sample_get_buffer(sample);
        GstMapInfo imap;
        if (!gst_buffer_map(inbuf, &imap, GST_MAP_READ)) {
            gst_sample_unref(sample);
            continue;
        }
        int sw = cam_w, sh = cam_h;
        // 摄像头帧 -> 模型输入（顺带缩放到显示尺寸）
        bgr_to_model_rgb(imap.data, sw, sh, model_rgb.data());

        // 缩放整帧到显示尺寸（最近邻，够快；mask 用双线性）
        {
            float rx = (float)sw / disp_w, ry = (float)sh / disp_h;
            for (int y = 0; y < disp_h; y++) {
                int yy = (int)(y * ry);
                if (yy >= sh) yy = sh - 1;
                for (int x = 0; x < disp_w; x++) {
                    int xx = (int)(x * rx);
                    if (xx >= sw) xx = sw - 1;
                    const unsigned char* s = imap.data + ((long)yy * sw + xx) * 3;
                    unsigned char* d = scaled.data() + ((long)y * disp_w + x) * 3;
                    d[0] = s[0];
                    d[1] = s[1];
                    d[2] = s[2];
                }
            }
        }
        gst_buffer_unmap(inbuf, &imap);
        gst_sample_unref(sample);

        if (!seg_run(&seg, model_rgb.data(), mask.data())) {
            fail_seg++;
            if (fail_seg > 5) {
                printf("seg_run failed %ld times, abort\n", fail_seg);
                break;
            }
            continue;
        }
        ok++;

        composite(scaled.data(), mask.data(), disp_w, disp_h, bg_mode,
                  composited.data());

        GstBuffer* outbuf = gst_buffer_new_allocate(
            NULL, (gsize)disp_w * disp_h * 3, NULL);
        GstMapInfo omap;
        gst_buffer_map(outbuf, &omap, GST_MAP_WRITE);
        memcpy(omap.data, composited.data(), (size_t)disp_w * disp_h * 3);
        gst_buffer_unmap(outbuf, &omap);
        GST_BUFFER_PTS(outbuf) = gst_clock_get_time(clock) - base;
        GST_BUFFER_DTS(outbuf) = GST_BUFFER_PTS(outbuf);
        GST_BUFFER_DURATION(outbuf) =
            gst_util_uint64_scale(GST_SECOND, 1, 30);

        GstFlowReturn fret =
            gst_app_src_push_buffer(GST_APP_SRC(disp_src), outbuf);
        if (fret != GST_FLOW_OK) {
            printf("push ret=%d, stop\n", fret);
            break;
        }
        frame++;

        if (show_fps && frame % 30 == 0) {
            GstClockTime now = gst_clock_get_time(clock);
            double sec = (now - t_start) / 1e9;
            printf("frames=%ld ok=%ld  %.1f fps\n", frame, ok,
                   frame / (sec > 0 ? sec : 1));
            fflush(stdout);
        }
    }

    printf("total frames=%ld ok=%ld fail_seg=%ld fail_pull=%ld\n", frame, ok,
           fail_seg, fail_pull);

    gst_app_src_end_of_stream(GST_APP_SRC(disp_src));
    gst_element_set_state(cap, GST_STATE_NULL);
    gst_element_set_state(disp, GST_STATE_NULL);
    gst_object_unref(cap_sink);
    gst_object_unref(disp_src);
    gst_object_unref(cap);
    gst_object_unref(disp);
    gst_object_unref(clock);
    rknn_destroy(seg.ctx);
    printf("done\n");
    return 0;
}
