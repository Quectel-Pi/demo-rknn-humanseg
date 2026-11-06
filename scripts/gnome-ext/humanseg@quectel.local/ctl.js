'use strict';

// 控制文件读写 —— 与 humanseg_settings.py / 后端 ctl_poll() 同一份协议：
//   seg=0|1
//   bg=0..3
// 写入必须原子（先写 .tmp 再 rename），否则后端可能读到半截内容。
import Gio from 'gi://Gio';
import GLib from 'gi://GLib';

const CTL = '/data/hs_test/ctl.conf';

export const BG_NAMES = [
    'Teal (default)',
    'Dimmed camera',
    'Solid gray',
    'Green screen',
];

function toBytes(text) {
    return new TextEncoder().encode(text);
}

export function readCtl() {
    let seg = 1, bg = 0;
    try {
        const [ok, data] = GLib.file_get_contents(CTL);
        if (!ok)
            return {seg, bg};
        const text = new TextDecoder().decode(data);
        for (const raw of text.split('\n')) {
            const line = raw.replace(/ /g, '');
            if (line.startsWith('seg=')) {
                const v = line.slice(4).trim();
                seg = (v === '0' || v === '') ? 0 : 1;
            } else if (line.startsWith('bg=')) {
                const v = parseInt(line.slice(3).trim(), 10);
                if (!Number.isNaN(v) && v >= 0 && v <= 3)
                    bg = v;
            }
        }
    } catch (e) {
        console.warn(`humanseg: read ${CTL} failed: ${e}`);
    }
    return {seg, bg};
}

export function writeCtl(seg, bg) {
    const text = `seg=${seg ? 1 : 0}\nbg=${bg}\n`;
    const tmp = `${CTL}.tmp`;
    try {
        const f = Gio.File.new_for_path(tmp);
        f.replace_contents(toBytes(text), null, false,
            Gio.FileCreateFlags.REPLACE_DESTINATION, null);
        // 原子替换：后端按 mtime 变化触发重读
        Gio.File.new_for_path(tmp).move(
            Gio.File.new_for_path(CTL),
            Gio.FileCopyFlags.OVERWRITE, null, null);
    } catch (e) {
        console.warn(`humanseg: write ${CTL} failed: ${e}`);
    }
}

// 后端是否在跑（pgrep -x hy8 的等价物：扫 /proc）
export function backendRunning() {
    let ok = false;
    try {
        const dir = Gio.File.new_for_path('/proc');
        const en = dir.enumerate_children(
            'standard::name', Gio.FileQueryInfoFlags.NONE, null);
        let info;
        while ((info = en.next_file(null)) !== null) {
            const name = info.get_name();
            if (!/^\d+$/.test(name))
                continue;
            try {
                const [r, cmd] = GLib.file_get_contents(`/proc/${name}/comm`);
                if (r && new TextDecoder().decode(cmd).trim() === 'hy8') {
                    ok = true;
                    break;
                }
            } catch (_) {
                // 进程可能刚好退出
            }
        }
        en.close(null);
    } catch (e) {
        console.warn(`humanseg: scan /proc failed: ${e}`);
    }
    return ok;
}
