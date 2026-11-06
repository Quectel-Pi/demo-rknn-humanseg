'use strict';

// 右上角 Quick Settings 里的「Portrait Segmentation」开关 + 背景选择。
//
// 位置：panel.js addExternalIndicator() 会把指示器插到 QuickSettings 的
// 图标行里（亮度/音量那一排），展开面板后出现的开关排在 Background Apps 之前，
// 也就是与你说的「虚拟背景」那排图标平齐。
//
// 与后端的分工：本扩展只写 ctl.conf，后端显示线程每帧查 mtime，
// 变了立刻重绘，所以开关是即时的，不重启进程。
import Clutter from 'gi://Clutter';
import GObject from 'gi://GObject';
import St from 'gi://St';

import * as Main from 'resource:///org/gnome/shell/ui/main.js';
import * as PopupMenu from 'resource:///org/gnome/shell/ui/popupMenu.js';
import * as QuickSettings from 'resource:///org/gnome/shell/ui/quickSettings.js';
import {Extension} from 'resource:///org/gnome/shell/extensions/extension.js';

import {BG_NAMES, backendRunning, readCtl, writeCtl} from './ctl.js';

const {QuickMenuToggle, SystemIndicator} = QuickSettings;

// 背景样式子菜单：每项前面放一个小色块，比纯文字好认
class BgItem extends PopupMenu.PopupBaseMenuItem {
    static {
        GObject.registerClass(this);
    }

    constructor(index, title, onActivate) {
        super({activate: onActivate});
        this._index = index;

        const box = new St.BoxLayout({style_class: 'humanseg-bg-row'});
        this.add_child(box);
        box.set_x_expand(true);

        // 预览色块 —— 用 CSS 画，不依赖图标主题
        const swatch = new St.Widget({
            style_class: 'humanseg-swatch',
            style: `background-color: ${SWATCH[index]};`,
            y_align: Clutter.ActorAlign.CENTER,
        });
        box.add_child(swatch);

        const label = new St.Label({
            text: title,
            y_align: Clutter.ActorAlign.CENTER,
            x_expand: true,
        });
        box.add_child(label);

        // 当前选中项的勾
        this._check = new St.Icon({
            icon_name: 'object-select-symbolic',
            style_class: 'popup-menu-icon',
            visible: false,
        });
        box.add_child(this._check);
    }

    setChecked(on) {
        this._check.visible = on;
    }
}

const SWATCH = ['#22b2aa', '#2b2b2b', '#5a5a5a', '#00b140'];

const Toggle = GObject.registerClass(
class Toggle extends QuickMenuToggle {
    _init() {
        super._init({
            title: 'Portrait Segmentation',
            icon_name: 'camera-web-symbolic',
            // GObject 属性名在 _init 里是下划线形式（继承 St.Button）
            toggle_mode: true,
        });

        this._items = [];
        this._bgItems = [];
        for (let i = 0; i < BG_NAMES.length; i++) {
            const item = new BgItem(i, BG_NAMES[i], () => {
                this._apply(this.checked, i);
                this.menu.close();
            });
            this._bgItems.push(item);
            this.menu.addMenuItem(item);
        }

        this.menu.addMenuItem(new PopupMenu.PopupSeparatorMenuItem());
        this._status = new PopupMenu.PopupMenuItem('', {reactive: false});
        this.menu.addMenuItem(this._status);

        this.connect('clicked', () => {
            this._apply(this.checked, this._bg);
        });

        this._sync();
    }

    // 把 seg + bg 一起写下去，避免两次写导致中间态
    _apply(seg, bg) {
        this._bg = bg;
        writeCtl(seg, bg);
        this._sync();
    }

    _sync() {
        const {seg, bg} = readCtl();
        this._bg = bg;
        this.checked = !!seg;

        this.subtitle = seg ? BG_NAMES[bg] : 'Off';
        this.title = 'Portrait Segmentation';

        for (let i = 0; i < this._bgItems.length; i++)
            this._bgItems[i].setChecked(i === bg);

        const alive = backendRunning();
        this._status.label.text = alive
            ? `Backend running · background: ${BG_NAMES[bg]}`
            : 'Backend not running';
        this._status.setSensitive(false);
        // 后端没起来时开关点了也没画面，明确告诉用户
        this.menuEnabled = true;
    }

    // 后端可能被别处改（GUI/命令行），面板打开时同步一次
    refresh() {
        this._sync();
    }
});

const Indicator = GObject.registerClass(
class Indicator extends SystemIndicator {
    _init() {
        super._init();

        this._toggle = new Toggle();
        this.quickSettingsItems.push(this._toggle);

        // 图标行上的小图标：分割开着才显示，跟系统其他指示器行为一致
        this._icon = this._addIndicator();
        this._icon.icon_name = 'camera-web-symbolic';
        this._icon.visible = false;

        this._toggle.connect('notify::checked', () => {
            this._icon.visible = this._toggle.checked;
        });
        this._icon.visible = this._toggle.checked;
    }

    refresh() {
        this._toggle.refresh();
        this._icon.visible = this._toggle.checked;
    }
});

export default class HumanSegExtension extends Extension {
    enable() {
        this._indicator = new Indicator();
        Main.panel.statusArea.quickSettings.addExternalIndicator(this._indicator);

        // 外部改动（桌面 GUI / 命令行）也要反映到面板上。
        // 2s 一次足够，代价是每个周期扫一遍 /proc。
        this._timer = setInterval(() => this._indicator.refresh(), 2000);
    }

    disable() {
        if (this._timer) {
            clearInterval(this._timer);
            this._timer = null;
        }
        if (this._indicator) {
            this._indicator.quickSettingsItems.forEach(item => item.destroy());
            this._indicator.destroy();
            this._indicator = null;
        }
    }
}
