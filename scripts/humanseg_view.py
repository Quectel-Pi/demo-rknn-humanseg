#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Human segmentation output window - frame plus a gear settings button in the
top-right corner (GTK4 fullscreen).

Why this structure:
  Under Wayland an application cannot position its own window, and the board has
  no layer-shell, so there is **no way** to float a button above a waylandsink
  video surface. The only way to get the button genuinely on top of the frame is
  to draw the frame inside a GTK4 window itself. The C++ backend publishes each
  frame's pixels through --shm and this process only puts them on screen.

Frame protocol (mirrors shm_publish in humanseg_y8seg.cc):
  offset 0   uint32 seq   increasing sequence number (changed = new frame)
  offset 4   uint32 w
  offset 8   uint32 h
  offset 12  uint32 pad
  offset 16  BGR pixels w*h*3
"""
import mmap
import os
import struct
import sys
import time

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Gdk", "4.0")
from gi.repository import Gdk, GLib, Gtk  # noqa: E402

SHM = "/dev/shm/humanseg.frames"
CTL = "/data/hs_test/ctl.conf"
HDR = 16

# Timestamp of interpreter start, used to report first-frame latency. Measured
# from process start so it includes import and GTK startup, not just our loop.
_START_T = time.time()
BG_NAMES = ["Teal (default)", "Dimmed camera", "Solid gray", "Green screen"]

# Stylesheet. It must be defined here and installed by View.__init__, not only
# from main(): any other entry point (test scripts, another process importing
# this module) that bypasses main() would get **completely unstyled** widgets -
# the gear and panel fall back to the theme's white background while the text
# inherits a light colour, which is white on white and unreadable.
# background-image: none is mandatory: Adwaita paints its base colour with
# background-image, so setting only background-color gets covered by it.
_CSS = b"""
.humanseg-gear {
    background-image: none;
    background-color: rgba(0, 0, 0, 0.55);
    color: #ffffff;
    min-width: 40px;
    min-height: 40px;
    box-shadow: none;
    text-shadow: none;
}
.humanseg-gear:hover {
    background-image: none;
    background-color: rgba(0, 0, 0, 0.80);
    color: #ffffff;
}
.humanseg-gear image {
    color: #ffffff;
}
.humanseg-panel {
    background-image: none;
    background-color: rgba(24, 24, 28, 0.96);
    border-radius: 12px;
    padding: 6px;
    box-shadow: 0 6px 24px rgba(0, 0, 0, 0.55);
}
/* All text inside the panel: force light, per-class so the theme
   cannot win on specificity. */
.humanseg-panel label,
.humanseg-panel .label,
.humanseg-panel .title-4,
.humanseg-panel .dim-label,
.humanseg-panel button {
    color: #f2f2f4;
}
.humanseg-panel .dim-label {
    color: rgba(242, 242, 244, 0.70);
}
.humanseg-panel button {
    background-image: none;
    background-color: rgba(255, 255, 255, 0.14);
    box-shadow: none;
}
.humanseg-panel button:hover {
    background-color: rgba(255, 255, 255, 0.24);
}
/* Background style radio row: highlight the active choice.
   Plain Button + "selected" class, not a ToggleButton :checked state. */
.humanseg-panel .humanseg-bgbtn {
    font-size: 11px;
    padding: 6px 4px;
    background-color: rgba(255, 255, 255, 0.08);
}
.humanseg-panel .humanseg-bgbtn.selected {
    background-color: rgba(120, 190, 255, 0.55);
    color: #ffffff;
    font-weight: bold;
}
/* DropDown search entry: without this it inherits the theme's dark text
   and disappears against our dark backdrop. */
.humanseg-panel entry,
.humanseg-panel entry text {
    color: #f2f2f4;
    background-image: none;
    background-color: rgba(255, 255, 255, 0.10);
    caret-color: #f2f2f4;
}
.humanseg-panel entry > text > placeholder {
    color: rgba(242, 242, 244, 0.55);
}
/* The dropdown list is a separate popover surface and the theme renders it
   light by default; our light item text would vanish on it. Force it dark. */
.humanseg-panel popover,
.humanseg-panel popover > contents,
.humanseg-panel popover listview,
.humanseg-panel popover scrolledwindow {
    background-image: none;
    background-color: #1a1a1e;
    color: #f2f2f4;
}
.humanseg-panel popover row,
.humanseg-panel popover row label {
    color: #f2f2f4;
}
.humanseg-panel popover row:selected,
.humanseg-panel popover row:selected label {
    background-color: rgba(255, 255, 255, 0.18);
    color: #ffffff;
}
"""

_css_installed = False


def install_css():
    """Idempotently attach _CSS to the display."""
    global _css_installed
    if _css_installed:
        return
    disp = Gdk.Display.get_default()
    if disp is None:
        return
    prov = Gtk.CssProvider()
    prov.load_from_data(_CSS)
    Gtk.StyleContext.add_provider_for_display(
        disp, prov, Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION)
    _css_installed = True


# ── Control file ────────────────────────────────────────────────
def read_ctl():
    seg, bg = 1, 0
    try:
        with open(CTL) as f:
            for line in f:
                line = line.replace(" ", "")
                if line.startswith("seg="):
                    seg = 0 if line[4:].strip() in ("0", "") else 1
                elif line.startswith("bg="):
                    try:
                        v = int(line[3:].strip())
                        if 0 <= v <= 3:
                            bg = v
                    except ValueError:
                        pass
    except FileNotFoundError:
        pass
    return seg, bg


def write_ctl(seg, bg):
    """Change only seg/bg, **preserving every other key** in ctl.conf
    (thr/up/margin/orphan/dump).

    The old implementation wrote just the two lines "seg=%d\\nbg=%d\\n", so a
    single click on a panel switch wiped the tuned thr=35 / up=1 values and the
    picture silently fell back to defaults - which looked like "the outline got
    worse again after I touched a switch". That is this function's only
    responsibility boundary: it has no right to drop keys it does not
    understand.

    Uses a unique temp name: a fixed .tmp collides when several writers (UI and
    verification scripts) run concurrently, and we measured a FileNotFoundError
    from the rename that made settings impossible to save.
    """
    keep = []
    try:
        with open(CTL) as f:
            for line in f:
                s = line.strip()
                if not s or s.startswith("#"):
                    continue
                k = s.split("=", 1)[0].strip()
                # Keys we write are taken over by the new value, the rest kept as-is
                if k in ("seg", "bg"):
                    continue
                keep.append(s)
    except FileNotFoundError:
        pass

    tmp = "%s.%d.tmp" % (CTL, os.getpid())
    with open(tmp, "w") as f:
        f.write("seg=%d\nbg=%d\n" % (seg, bg))
        for s in keep:
            f.write(s + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, CTL)


def backend_pids():
    out = []
    for name in os.listdir("/proc"):
        if not name.isdigit():
            continue
        try:
            with open("/proc/%s/comm" % name) as f:
                if f.read().strip() == "hy8":
                    out.append(int(name))
        except OSError:
            pass
    return out


# ── Shared-memory frame source ──────────────────────────────────
class Frames:
    """Maps the frame buffer published by the backend. A backend restart
    recreates the file, so the mapping has to be re-established."""

    def __init__(self):
        self.mm = None
        self.fd = None
        self.seq = 0
        self.ino = -1
        self.w = self.h = 0
        self.buf = None

    def attach(self):
        try:
            self.fd = os.open(SHM, os.O_RDONLY)
        except OSError:
            return False
        size = os.fstat(self.fd).st_size
        if size <= HDR:
            self.close()
            return False
        self.mm = mmap.mmap(self.fd, size, mmap.MAP_SHARED, mmap.PROT_READ)
        # Read the header once to get the dimensions
        seq, w, h, _ = struct.unpack_from("<IIII", self.mm, 0)
        if w == 0 or h == 0 or size < HDR + w * h * 3:
            self.close()
            return False
        self.w, self.h = w, h
        self.seq = seq
        self.ino = os.fstat(self.fd).st_ino
        self.buf = bytearray(w * h * 3)
        return True

    def _stale(self):
        """Whether the SHM path now points at a different file (a backend
        restart removes and recreates it).

        On restart up.sh runs `rm -f /dev/shm/humanseg.frames` and creates a new
        one, while this process still mmaps the **already-unlinked old inode**:
        reads still succeed but seq never advances, so the picture is completely
        frozen while the process looks alive (field incident 2026-11-06).
        """
        try:
            return os.stat(SHM).st_ino != self.ino
        except OSError:
            return True          # missing file counts as stale, retry next frame

    def close(self):
        if self.mm is not None:
            self.mm.close()
            self.mm = None
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None

    def latest(self):
        """Fetch the newest frame. Returns True when there is an update.

        Note that we deliberately avoid building an intermediate bytearray here:
        the previous per-frame bytes(self.buf) allocated an extra 3 MB
        (1280*800*3), which at 28 fps is 84 MB/s of allocation, and that GC
        pressure showed up as stalls when clicking buttons. Now it is a single
        slice copy.
        """
        if self.mm is None and not self.attach():
            return False
        # A backend restart rm+recreates the shm while this process is still
        # attached to the old inode - we must re-attach, otherwise the picture
        # freezes forever (process alive but seq never moves again).
        if self._stale():
            self.close()
            if not self.attach():
                return False
        # Double-check: only a matching seq twice means a complete read
        # (copying while the backend writes would tear the frame)
        n = self.w * self.h * 3
        for _ in range(3):
            s0 = struct.unpack_from("<I", self.mm, 0)[0]
            if s0 == self.seq:
                return False                     # no new frame
            # mmap has no readinto, so slice to get bytes (one copy)
            self.buf = self.mm[HDR:HDR + n]
            s1 = struct.unpack_from("<I", self.mm, 0)[0]
            if s0 == s1:
                self.seq = s0
                return True
        return False                             # kept tearing, try again later

    def texture(self):
        # BGR -> memory texture (GDK wants B8G8R8A8; we feed packed 3-byte
        # BGR and use the strided B8G8R8 form, GTK pads it itself)
        b = GLib.Bytes.new(self.buf)
        return Gdk.MemoryTexture.new(
            self.w, self.h, Gdk.MemoryFormat.B8G8R8, b, self.w * 3)


# ── Settings panel opened by the top-right gear ──────────────────
class SettingsPanel(Gtk.Box):
    """Only shown after clicking the gear. It does exactly two things: turn
    segmentation on/off, and pick the background style.

    The base class is a plain Gtk.Box rather than Gtk.Popover: under Wayland a
    popover is a separate xdg_popup surface with an implicit grab, so the click
    that dismisses it falls through to whatever is underneath, which produced
    "it closes and the gear immediately reopens it". An embedded Box has no
    separate surface; visibility is entirely under our control.
    """

    def __init__(self, on_change):
        super().__init__(orientation=Gtk.Orientation.VERTICAL, spacing=12)
        self._on_change = on_change
        for side in ("top", "bottom", "start", "end"):
            getattr(self, "set_margin_" + side)(16)

        title = Gtk.Label(label="Portrait Segmentation", xalign=0)
        title.add_css_class("title-4")
        self.append(title)

        row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=12)
        lbl = Gtk.Label(label="Enable segmentation", xalign=0, hexpand=True)
        row.append(lbl)
        self.sw = Gtk.Switch(valign=Gtk.Align.CENTER)
        self.sw.connect("state-set", self._seg_changed)
        row.append(self.sw)
        self.append(row)

        self.append(Gtk.Separator())

        self.append(Gtk.Label(label="Virtual background", xalign=0))
        # Deliberately not Gtk.DropDown: when opened under GTK4 it creates a
        # separate xdg_popup surface, which on Wayland shows up as a detached
        # box floating in the middle of the screen (the "square pops up in the
        # centre" the user reported, same root cause as the gear issue noted
        # further down). An embedded row of buttons has no separate surface and
        # no grab, so clicking a choice simply selects it.
        self.bgbtns = []
        bgrow = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        bgrow.set_homogeneous(True)
        for i, name in enumerate(BG_NAMES):
            # Plain Button plus our own "selected" CSS class gives the radio
            # behaviour. Not ToggleButton: clicking the active one clears its
            # state and needs extra debouncing. Not DropDown either: that is
            # exactly what produced the box in the middle of the screen.
            b = Gtk.Button(label=name)
            b.add_css_class("humanseg-bgbtn")
            b.connect("clicked", self._bg_clicked, i)
            bgrow.append(b)
            self.bgbtns.append(b)
        self.append(bgrow)

        self.status = Gtk.Label(xalign=0)
        self.status.add_css_class("dim-label")
        self.append(self.status)

        # Offer a start button while the backend is not running
        self.startbtn = Gtk.Button(label="Start backend")
        self.startbtn.connect("clicked", self._start)
        self.append(self.startbtn)

        self.reload()

    def reload(self):
        seg, bg = read_ctl()
        self.sw.set_active(bool(seg))
        # The button row is a radio group: only the current choice is lit, marked
        # with a CSS class rather than the widget's own toggle state (a plain
        # Button has none), so the callback does not write ctl a second time.
        for i, b in enumerate(self.bgbtns):
            if i == bg:
                b.add_css_class("selected")
            else:
                b.remove_css_class("selected")
        alive = bool(backend_pids())
        self.startbtn.set_visible(not alive)
        self.status.set_label(
            ("Backend running · background: %s" % BG_NAMES[bg]) if alive
            else "Backend not running")

    def _seg_changed(self, sw, state):
        _seg, bg = read_ctl()
        write_ctl(1 if state else 0, bg)
        self.reload()
        self._on_change()
        return False

    def _bg_clicked(self, _btn, idx):
        seg, _bg = read_ctl()
        write_ctl(seg, idx)
        self.reload()
        self._on_change()

    def _start(self, _b):
        # Idempotent: never start a second instance. A duplicate grabs the camera
        # and the visible symptom is a frozen picture.
        if backend_pids():
            self.reload()
            return
        # spawn_async rather than subprocess: subprocess does fork+exec and may
        # copy the whole multi-MB buffer at fork time, which visibly stalls the
        # GUI thread.
        try:
            GLib.spawn_async(
                argv=["/bin/sh", "-c", LAUNCH],
                flags=GLib.SpawnFlags.SEARCH_PATH
                | GLib.SpawnFlags.DO_NOT_REAP_CHILD,
                child_setup=None, user_data=None)
        except GLib.Error as e:
            print("spawn failed:", e, flush=True)
        GLib.timeout_add(2500, lambda: (self.reload(), False)[1])


LAUNCH = (
    "cd /data/hs_test && setsid env "
    "XDG_RUNTIME_DIR=/run/user/1000 WAYLAND_DISPLAY=wayland-0 "
    "LD_LIBRARY_PATH=/data/hs_test ./hy8 ./y8s_rkv.rknn "
    "--device /dev/video73 --cam 1280x720 --disp 1280x800 "
    "--fps --ctl /data/hs_test/ctl.conf --shm /dev/shm/humanseg.frames "
    "> /data/hs_test/live.log 2>&1 < /dev/null &"
)


# ── Main window: picture plus the top-right gear ────────────────
class View(Gtk.Window):
    def __init__(self):
        super().__init__(title="Portrait Segmentation")
        # Install the stylesheet here: any entry point constructing a View needs
        # the correct colours, not just main() (test scripts and other processes
        # would otherwise see white on white).
        install_css()
        # Undecorated fullscreen: the picture fills everything, no title bar or
        # close button (Wayland lets the compositor place the window; this code
        # only owns the content. True fullscreen needs window-manager support).
        self.set_decorated(False)
        self.fullscreen()

        self.frames = Frames()

        over = Gtk.Overlay()
        self.set_child(over)

        # Picture layer
        self.pic = Gtk.Picture(hexpand=True, vexpand=True,
                               content_fit=Gtk.ContentFit.CONTAIN)
        over.set_child(self.pic)

        # Top-right gear plus the settings panel it opens.
        # A plain Button with an in-window panel, **deliberately not
        # Gtk.MenuButton/Gtk.Popover**: under Wayland a popover is a separate
        # xdg_popup surface with an implicit grab, so the dismissing click falls
        # through to the gear underneath and "closing it immediately reopens
        # it" - the popup could never be dismissed. popdown() and gestures do
        # not fix the root cause. Embedded as an ordinary Overlay child it has no
        # separate surface and no grab, the gear click is the only entry point,
        # so one click opens and the next closes.
        self.gear = Gtk.Button()
        self.gear.set_icon_name("emblem-system-symbolic")
        self.gear.add_css_class("circular")
        self.gear.add_css_class("humanseg-gear")
        self.gear.set_halign(Gtk.Align.END)
        self.gear.set_valign(Gtk.Align.START)
        self.gear.set_margin_top(16)
        self.gear.set_margin_end(16)
        self.gear.set_tooltip_text("Settings")
        self.gear.connect("clicked", self._toggle_panel)
        over.add_overlay(self.gear)

        self.panel = SettingsPanel(self._on_setting_changed)
        self.panel.set_halign(Gtk.Align.END)
        self.panel.set_valign(Gtk.Align.START)
        self.panel.set_margin_top(72)      # clear the gear
        self.panel.set_margin_end(16)
        self.panel.add_css_class("humanseg-panel")
        self.panel.set_visible(False)
        over.add_overlay(self.panel)

        # Click the picture or press Esc to close the panel
        click = Gtk.GestureClick()
        click.connect("pressed", self._close_panel)
        self.pic.add_controller(click)

        key = Gtk.EventControllerKey()
        key.connect("key-pressed", self._on_key)
        self.add_controller(key)

        # Each tick, check the shared memory for a new picture; if there is
        # none, keep the previous frame and do not refresh the texture
        self._n = 0
        self._t0 = 0.0
        self._tick = GLib.timeout_add(16, self._pull)   # ~60 Hz poll
        GLib.timeout_add(1500, self._refresh_panel)

    def _toggle_panel(self, *_a):
        if self.panel.get_visible():
            self.panel.set_visible(False)
        else:
            self.panel.reload()
            self.panel.set_visible(True)

    def _close_panel(self, *_a):
        if self.panel.get_visible():
            self.panel.set_visible(False)

    def _on_key(self, _c, keyval, _code, _state):
        if keyval == Gdk.KEY_Escape and self.panel.get_visible():
            self._close_panel()
            return True
        return False

    def _pull(self):
        if self.frames.latest():
            self.pic.set_paintable(self.frames.texture())
            # Log a heartbeat proving the picture really updates (rather than
            # sitting on a blank screen)
            if self._n == 0:
                self._t0 = time.time()
                # First frame: log it immediately. Without this the earliest
                # visible line is the 120-frame report ~4.4 s later, which makes
                # start-up latency impossible to measure from the log.
                print("first frame at %.2fs" % (self._t0 - _START_T),
                      flush=True)
            self._n += 1
            if self._n % 120 == 0:
                dt = time.time() - self._t0
                print("painted %d frames (%.1f fps) seq=%d"
                      % (self._n, self._n / dt if dt else 0, self.frames.seq),
                      flush=True)
        return True

    def _refresh_panel(self):
        # Backend death/liveness must be reflected, but scanning /proc every
        # 1.5 s is unnecessary - only check while the panel is open or while the
        # status line still says "not running" (that is when we need to notice).
        if self.panel.get_visible() or not backend_pids():
            self.panel.reload()
        return True

    def _on_setting_changed(self):
        pass        # the backend polls ctl.conf itself, nothing to notify


def main():
    # Not using Gtk.Application: it tries to register a unique name on the
    # session bus, and under `sudo -u <user>` we measured no usable DBUS - it
    # fails with NoReply and the window never appears.
    loop = GLib.MainLoop()
    win = View()
    win.connect("close-request", lambda *a: (loop.quit(), False)[1])

    win.present()
    loop.run()


if __name__ == "__main__":
    main()
