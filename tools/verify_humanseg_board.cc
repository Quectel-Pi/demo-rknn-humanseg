// 板端验证：humanseg rknn (softmax 前 logits) 推理，输出前景 mask
//
// 目的：确认 model/humanseg/humanseg_v1_lite_398x224.rknn（Softmax 前截断版）
// 在真实 NPU 上输出正常 —— 即 NOTES §6 的"概率塌缩"在板端确实已消失。
//
// 编译（demo-rknn-object-detect 根目录）：
//   export GCC_COMPILER=<SDK>/prebuilts/gcc/linux-x86/aarch64/\
//gcc-arm-10.3-2021.07-x86_64-aarch64-none-linux-gnu/bin/aarch64-none-linux-gnu
//   $GCC_COMPILER-g++ -O2 -std=c++11 scripts/verify_humanseg_board.cc \
//       -I3rdparty/librknn_api/include -o /tmp/verify_humanseg_board \
//       -L3rdparty/librknn_api/aarch64 -lrknnrt -lpthread
//
// 用法（板端）：
//   ./verify_humanseg_board <model.rknn> <img1.rgb> [img2.rgb ...]
//   *.rgb 为 398x224x3 的 RGB888 裸数据（宿主 scripts/make_board_testdata.py 生成）
//   每个输入会额外产出 <img>.fgmask（398*224 字节，0/1 前景），供宿主算 IoU。

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "rknn_api.h"

#define MODEL_W 398
#define MODEL_H 224
#define MODEL_C 3
#define IN_SIZE (MODEL_W * MODEL_H * MODEL_C)

static void dump_tensor_attr(rknn_tensor_attr* attr) {
    printf("  idx=%d name=%s n_dims=%d dims=[", attr->index, attr->name,
           attr->n_dims);
    for (uint32_t i = 0; i < attr->n_dims; i++) {
        printf("%d%s", attr->dims[i], i + 1 < attr->n_dims ? ", " : "");
    }
    printf("] fmt=%d type=%d size=%d\n", attr->fmt, attr->type, attr->size);
}

static unsigned char* load_file(const char* path, int* size) {
    FILE* fp = fopen(path, "rb");
    if (!fp) {
        printf("ERROR: cannot open %s\n", path);
        return NULL;
    }
    fseek(fp, 0, SEEK_END);
    long sz = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    unsigned char* buf = (unsigned char*)malloc(sz);
    if (!buf || fread(buf, 1, sz, fp) != (size_t)sz) {
        printf("ERROR: read failed %s\n", path);
        if (buf) free(buf);
        fclose(fp);
        return NULL;
    }
    fclose(fp);
    *size = (int)sz;
    return buf;
}

// 把 rknn 输出的 float logits 按"长度为 2 的轴即通道轴"解析成 mask[0/1]。
// 逻辑与宿主 scripts/verify_humanseg.py:to_hw2() 一致 —— 因为 ONNX 尾部
// Transpose 的 perm=[0,3,2,1] 会让 H/W 也交换，不能假定标准 NHWC/NCHW。
static bool logits_to_mask(const float* o, const rknn_tensor_attr& attr,
                           const int spatial[2], unsigned char* mask,
                           float* vmin, float* vmax, float* fg_ratio) {
    int ch_axis = -1;
    int nspatial[3];
    int nsp = 0;
    for (uint32_t i = 0; i < attr.n_dims; i++) {
        if (attr.dims[i] == 2 && ch_axis < 0) {
            ch_axis = (int)i;
        } else {
            nspatial[nsp++] = (int)attr.dims[i];
        }
    }
    if (ch_axis < 0) {
        printf("  ERROR: no axis of size 2 in dims\n");
        return false;
    }
    // 去掉 batch(=1) 后应剩两个空间轴
    int a = 0, b = 0;
    for (int i = 0; i < nsp; i++) {
        if (nspatial[i] == 1) continue;
        if (a == 0) a = nspatial[i];
        else if (b == 0) b = nspatial[i];
    }
    if (a == 0 || b == 0) {
        printf("  ERROR: cannot infer spatial dims\n");
        return false;
    }
    // 判断哪个是 H：等于 MODEL_H 的那个
    int w = a, h = b;
    if (a == MODEL_H && b == MODEL_W) {
        h = a;
        w = b;
    } else if (a == MODEL_W && b == MODEL_H) {
        w = a;
        h = b;
    } else {
        printf("  WARN: spatial %dx%d not %dx%d, assume (W,H)\n", a, b, MODEL_W,
               MODEL_H);
        w = a;
        h = b;
    }

    int total = w * h;
    long fg = 0;
    float lo = 1e9f, hi = -1e9f;
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            long i0, i1;
            if (ch_axis == (int)attr.n_dims - 1) {          // ...HWC, C 在最后
                long base = ((long)y * w + x) * 2;
                i0 = base;
                i1 = base + 1;
            } else if (ch_axis == 1) {                       // NCHW
                long hw = (long)total;
                i0 = (long)y * w + x;
                i1 = hw + i0;
            } else {
                printf("  ERROR: unsupported ch_axis=%d n_dims=%d\n", ch_axis,
                       attr.n_dims);
                return false;
            }
            float c0 = o[i0], c1 = o[i1];
            if (c0 < lo) lo = c0;
            if (c1 < lo) lo = c1;
            if (c0 > hi) hi = c0;
            if (c1 > hi) hi = c1;
            // 只写 mask 区域（可能 h/w 与 MODEL 不同，按行截断保护）
            if (y < MODEL_H && x < MODEL_W && mask) {
                mask[y * MODEL_W + x] = (c1 > c0) ? 1 : 0;
            }
            if (c1 > c0) fg++;
        }
    }
    *vmin = lo;
    *vmax = hi;
    *fg_ratio = (float)fg / (float)total;
    return true;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        printf("usage: %s <model.rknn> <img1.rgb> [img2.rgb ...]\n", argv[0]);
        printf("  *.rgb = %dx%dx%d RGB888 raw (%d bytes)\n", MODEL_W, MODEL_H,
               MODEL_C, IN_SIZE);
        return -1;
    }

    int model_size = 0;
    unsigned char* model_data = load_file(argv[1], &model_size);
    if (!model_data) return -1;
    printf("model: %s (%d bytes)\n", argv[1], model_size);

    rknn_context ctx = 0;
    int ret = rknn_init(&ctx, model_data, model_size, 0, NULL);
    if (ret < 0) {
        printf("ERROR: rknn_init ret=%d\n", ret);
        return -1;
    }

    rknn_input_output_num io_num;
    memset(&io_num, 0, sizeof(io_num));
    ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret < 0) {
        printf("ERROR: query io num ret=%d\n", ret);
        return -1;
    }
    printf("inputs=%d outputs=%d\n", io_num.n_input, io_num.n_output);

    for (uint32_t i = 0; i < io_num.n_input; i++) {
        rknn_tensor_attr a;
        memset(&a, 0, sizeof(a));
        a.index = i;
        rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &a, sizeof(a));
        dump_tensor_attr(&a);
    }
    rknn_tensor_attr out_attr;
    memset(&out_attr, 0, sizeof(out_attr));
    out_attr.index = 0;
    rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &out_attr, sizeof(out_attr));
    dump_tensor_attr(&out_attr);

    for (int k = 2; k < argc; k++) {
        int img_size = 0;
        unsigned char* img = load_file(argv[k], &img_size);
        if (!img) continue;
        if (img_size != IN_SIZE) {
            printf("ERROR: %s size=%d expected %d\n", argv[k], img_size, IN_SIZE);
            free(img);
            continue;
        }

        rknn_input in;
        memset(&in, 0, sizeof(in));
        in.index = 0;
        in.type = RKNN_TENSOR_UINT8;
        in.size = IN_SIZE;
        in.fmt = RKNN_TENSOR_NHWC;
        in.buf = img;
        in.pass_through = 0;

        if ((ret = rknn_inputs_set(ctx, 1, &in)) < 0 ||
            (ret = rknn_run(ctx, NULL)) < 0) {
            printf("ERROR: run failed ret=%d\n", ret);
            free(img);
            continue;
        }

        rknn_output out;
        memset(&out, 0, sizeof(out));
        out.index = 0;
        out.want_float = 1;
        out.is_prealloc = 0;
        if ((ret = rknn_outputs_get(ctx, 1, &out, NULL)) < 0) {
            printf("ERROR: outputs_get ret=%d\n", ret);
            free(img);
            continue;
        }

        unsigned char* mask = (unsigned char*)calloc(MODEL_W * MODEL_H, 1);
        float vmin = 0, vmax = 0, fg = 0;
        int sp[2] = {MODEL_W, MODEL_H};
        bool ok = logits_to_mask((const float*)out.buf, out_attr, sp, mask,
                                 &vmin, &vmax, &fg);
        printf("\n[%s]\n", argv[k]);
        if (ok) {
            printf("  fg_ratio=%.4f  logits=[%.4f, %.4f]\n", fg, vmin, vmax);
            std::string mp = std::string(argv[k]) + ".fgmask";
            FILE* mf = fopen(mp.c_str(), "wb");
            if (mf) {
                fwrite(mask, 1, MODEL_W * MODEL_H, mf);
                fclose(mf);
                printf("  mask -> %s\n", mp.c_str());
            }
        }
        free(mask);
        rknn_outputs_release(ctx, 1, &out);
        free(img);
    }

    rknn_destroy(ctx);
    free(model_data);
    printf("\ndone\n");
    return 0;
}
