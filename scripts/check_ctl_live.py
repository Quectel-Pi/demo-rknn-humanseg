#!/usr/bin/env python3
"""用**真实后端二进制**验证 ctl 越界值被拒绝。

为什么不用 Python 重写一遍解析逻辑：
  旧脚本用 Python 复制了一份 C 的解析代码再测它自己 —— 那是在测副本。
  C 端改了而副本没改，测试照样 PASS，属于假通过。

这里直接改板上 ctl.conf，看 live.log 里有没有对应的 [ctl] 行：
  * 合法值  -> 必须出现 "xxx -> 新值"
  * 越界值  -> 必须**不**出现该行（保持原值）
日志是后端自己打的，改不了也骗不了。

用法：板端 python3 check_ctl_live.py（需后端正在运行）
"""
import os
import subprocess
import sys
import time

BD = "/data/hs_test"
CTL = os.path.join(BD, "ctl.conf")
LOG = os.path.join(BD, "live.log")

FAILS = []


def check(name, ok, detail=""):
    print("  %-46s %s  %s" % (name, "PASS" if ok else "FAIL", detail))
    if not ok:
        FAILS.append(name)


def log_size():
    try:
        return os.path.getsize(LOG)
    except OSError:
        return 0


def write_ctl(pairs, keep_other=True):
    """pairs: dict 覆盖写入；keep_other 保留其它现有键。"""
    keep = []
    if keep_other:
        try:
            with open(CTL) as f:
                for ln in f:
                    s = ln.strip()
                    if not s or s.startswith("#"):
                        continue
                    k = s.split("=", 1)[0].strip()
                    if k in pairs:
                        continue
                    keep.append(s)
        except FileNotFoundError:
            pass
    tmp = CTL + ".tmp"
    with open(tmp, "w") as f:
        for k, v in pairs.items():
            f.write("%s=%s\n" % (k, v))
        for s in keep:
            f.write(s + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, CTL)


def new_log(since):
    """取 since 之后新增的日志。"""
    try:
        with open(LOG, "rb") as f:
            f.seek(since)
            return f.read().decode("utf-8", "replace")
    except OSError:
        return ""


def probe(pairs, wait=1.6):
    n0 = log_size()
    write_ctl(pairs)
    time.sleep(wait)
    return new_log(n0)


def main():
    if not os.path.exists(LOG):
        print("FAIL: 没有 %s" % LOG)
        return 1
    if subprocess.run(["pgrep", "-x", "hy8"], capture_output=True).returncode != 0:
        print("FAIL: 后端没在跑")
        return 1

    # 备份现场配置
    with open(CTL) as f:
        backup = f.read()

    try:
        print("=== 合法值必须生效 ===")
        out = probe({"thr": 40})
        check("thr=40 被接受", "[ctl] thr" in out and "0.40" in out,
              out.strip().replace("\n", " | ")[:70])

        out = probe({"margin": 24})
        check("margin=24 被接受", "[ctl] margin" in out and "24 px" in out,
              out.strip().replace("\n", " | ")[:70])

        out = probe({"orphan": 30})
        check("orphan=30 被接受", "[ctl] orphan" in out and "0.30" in out,
              out.strip().replace("\n", " | ")[:70])

        out = probe({"up": 0})
        check("up=0 被接受", "[ctl] up" in out and "-> 0" in out,
              out.strip().replace("\n", " | ")[:70])

        print("\n=== 越界值必须被拒绝（关键）===")
        # thr 边界 5..90
        out = probe({"thr": 95})
        check("thr=95 被拒绝", "[ctl] thr" not in out,
              out.strip().replace("\n", " | ")[:70] or "(无日志=正确)")
        out = probe({"thr": 4})
        check("thr=4 被拒绝", "[ctl] thr" not in out,
              out.strip().replace("\n", " | ")[:70] or "(无日志=正确)")

        # margin 边界 0..160
        out = probe({"margin": 999})
        check("margin=999 被拒绝", "[ctl] margin" not in out,
              out.strip().replace("\n", " | ")[:70] or "(无日志=正确)")

        # orphan 边界 0..100
        out = probe({"orphan": 200})
        check("orphan=200 被拒绝", "[ctl] orphan" not in out,
              out.strip().replace("\n", " | ")[:70] or "(无日志=正确)")

        # up 只认 0/1
        out = probe({"up": 7})
        check("up=7 被拒绝", "[ctl] up" not in out,
              out.strip().replace("\n", " | ")[:70] or "(无日志=正确)")

        # bg 边界 0..3
        out = probe({"bg": 9})
        check("bg=9 被拒绝", "[ctl] bg" not in out,
              out.strip().replace("\n", " | ")[:70] or "(无日志=正确)")
        out = probe({"bg": 3})
        check("bg=3 被接受", "[ctl] bg" in out and "-> 3" in out,
              out.strip().replace("\n", " | ")[:70])

        print("\n=== 灵敏度自查：越界拒绝不能是「日志本来就不打」 ===")
        # 紧接上面被拒绝的 bg=9，写一个相邻合法值，必须立刻有日志。
        # 若这里也没日志，说明是探测机制失效，上面所有「被拒绝」都不算数。
        out = probe({"bg": 1})
        ok_sens = "[ctl] bg" in out and "-> 1" in out
        check("相邻合法值 bg=1 立即生效", ok_sens,
              out.strip().replace("\n", " | ")[:70])
        if not ok_sens:
            print("  ==> 探测机制失效，上面全部「被拒绝」结论作废")

        return 1 if FAILS else 0
    finally:
        # 严格还原现场
        tmp = CTL + ".tmp"
        with open(tmp, "w") as f:
            f.write(backup)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, CTL)
        print("\nctl 已还原:", " ".join(backup.split()))


if __name__ == "__main__":
    sys.exit(main())
