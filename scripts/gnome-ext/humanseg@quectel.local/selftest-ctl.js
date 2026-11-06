#!/usr/bin/env gjs
// ad-hoc 验证 ctl.js：真实调用它的 readCtl/writeCtl，不模拟
imports.gi.versions.Gio = '2.0';
const GLib = imports.gi.GLib;

// 导入被测模块（gjs 1.82 / ESM：动态 import；顶层 await 需包一层 async）
const EXT = '/home/qq/.local/share/gnome-shell/extensions/humanseg@quectel.local';
const CTL = '/data/hs_test/ctl.conf';

main();

async function main() {
let fails = 0;
function check(name, cond, extra) {
    print((cond ? 'PASS  ' : 'FAIL  ') + name + (extra ? '  ' + extra : ''));
    if (!cond) fails++;
}

// 备份原文件
const [okBak, bak] = GLib.file_get_contents(CTL);
print('原 ctl.conf: ' + (okBak ? new TextDecoder().decode(bak).replace(/\n/g, ' ') : '(无)'));

// 导入被测模块（gjs 1.82 / ESM：用动态 import，不是老的 importModule）
let mod;
try {
    mod = await import(`file://${EXT}/ctl.js`);
} catch (e) {
    print('FAIL  无法导入 ctl.js: ' + e);
    imports.system.exit(1);
}

check('导出 BG_NAMES 4 项', mod.BG_NAMES && mod.BG_NAMES.length === 4);

// 1) 写 -> 读 往返
mod.writeCtl(0, 2);
let r = mod.readCtl();
check('writeCtl(0,2) -> readCtl', r.seg === 0 && r.bg === 2, `got seg=${r.seg} bg=${r.bg}`);

mod.writeCtl(1, 3);
r = mod.readCtl();
check('writeCtl(1,3) -> readCtl', r.seg === 1 && r.bg === 3, `got seg=${r.seg} bg=${r.bg}`);

// 2) 写入必须是原子替换：不能留 .tmp
const tmpExists = GLib.file_test(CTL + '.tmp', GLib.FileTest.EXISTS);
check('无残留 .tmp（原子替换）', !tmpExists);

// 3) 文件内容格式与 Python/C 端一致
const [okR, data] = GLib.file_get_contents(CTL);
const text = new TextDecoder().decode(data);
check('文件格式 seg=1\\nbg=3', /seg=1\nbg=3\n/.test(text), JSON.stringify(text));

// 4) 越界 bg 要被忽略（不能崩、不能写坏）
mod.writeCtl(1, 99);
r = mod.readCtl();
check('越界 bg=99 被忽略(回落 0)', r.bg === 0, `got bg=${r.bg}`);

// 5) 与 Python 端写的文件互认
GLib.file_set_contents(CTL, 'bg=1\nseg=0\n');  // Python 可能换序
r = mod.readCtl();
check('解析换序文件 bg=1/seg=0', r.seg === 0 && r.bg === 1, `got seg=${r.seg} bg=${r.bg}`);

// 6) 后端存活检测
const alive = mod.backendRunning();
print('      backendRunning() -> ' + alive + '（hy8 是否在跑，非断言项）');

// 还原
if (okBak)
    GLib.file_set_contents(CTL, bak);
else
    mod.writeCtl(1, 0);

print('');
print(fails === 0 ? 'RESULT: PASS' : `RESULT: FAIL (${fails})`);
imports.system.exit(fails === 0 ? 0 : 1);
}
