"""Windows HiDPI support for the topology editor."""
from __future__ import annotations

import ctypes
import sys
import tkinter as tk
from tkinter import font as tkfont
from typing import Callable, Optional

_PMv2 = -4
_LOGPIXELSX = 88
_AWARENESS_NAMES = {
    0: "UNAWARE (масштабирование Windows, размытие)",
    1: "SYSTEM_DPI_AWARE",
    2: "PER_MONITOR_DPI_AWARE",
}
_mode = "not-called"


def enable_dpi_awareness() -> str:
    """Declare the process DPI-aware before creating the first Tk window."""
    global _mode
    if sys.platform != "win32":
        _mode = "n/a (не Windows)"
        return _mode
    try:
        function = ctypes.windll.user32.SetProcessDpiAwarenessContext
        function.argtypes = [ctypes.c_void_p]
        function.restype = ctypes.c_int
        if function(ctypes.c_void_p(_PMv2)):
            _mode = "per-monitor-v2"
            return _mode
    except (AttributeError, OSError):
        pass
    try:
        if ctypes.windll.shcore.SetProcessDpiAwareness(2) == 0:
            _mode = "per-monitor"
            return _mode
    except (AttributeError, OSError):
        pass
    try:
        if ctypes.windll.user32.SetProcessDPIAware():
            _mode = "system"
            return _mode
    except (AttributeError, OSError):
        pass
    _mode = "уже задано манифестом или недоступно"
    return _mode


def applied_mode() -> str:
    return _mode


def process_awareness() -> str:
    if sys.platform != "win32":
        return "n/a"
    try:
        value = ctypes.c_int()
        ctypes.windll.shcore.GetProcessDpiAwareness(None, ctypes.byref(value))
        return _AWARENESS_NAMES.get(value.value, f"?{value.value}")
    except (AttributeError, OSError):
        try:
            return "SYSTEM_DPI_AWARE" if ctypes.windll.user32.IsProcessDPIAware() else "UNAWARE"
        except (AttributeError, OSError):
            return "неизвестно"


def hwnd_of(widget: tk.Misc) -> int:
    try:
        return int(widget.winfo_toplevel().winfo_id())
    except (tk.TclError, ValueError):
        return 0


def get_dpi(widget: Optional[tk.Misc] = None) -> float:
    if sys.platform == "win32":
        if widget is not None:
            try:
                dpi = ctypes.windll.user32.GetDpiForWindow(hwnd_of(widget))
                if dpi:
                    return float(dpi)
            except (AttributeError, OSError):
                pass
        try:
            dpi = ctypes.windll.user32.GetDpiForSystem()
            if dpi:
                return float(dpi)
        except (AttributeError, OSError):
            pass
        try:
            dc = ctypes.windll.user32.GetDC(0)
            dpi = ctypes.windll.gdi32.GetDeviceCaps(dc, _LOGPIXELSX)
            ctypes.windll.user32.ReleaseDC(0, dc)
            if dpi:
                return float(dpi)
        except (AttributeError, OSError):
            pass
    if widget is not None:
        try:
            return float(widget.winfo_fpixels("1i"))
        except tk.TclError:
            pass
    return 96.0


def dpi_scale(widget: Optional[tk.Misc] = None) -> float:
    return max(0.5, min(6.0, get_dpi(widget) / 96.0))


def screen_size_px(widget: tk.Misc) -> tuple[int, int]:
    return widget.winfo_screenwidth(), widget.winfo_screenheight()


def default_geometry(widget: tk.Misc, frac: float = 0.85) -> str:
    width, height = screen_size_px(widget)
    return f"{int(width * frac)}x{int(height * frac)}"


def diagnostics(widget: tk.Misc, ui_scale: float = 1.0) -> str:
    width, height = screen_size_px(widget)
    dpi = get_dpi(widget)
    try:
        scaling = float(widget.tk.call("tk", "scaling"))
    except tk.TclError:
        scaling = -1.0
    font = tkfont.nametofont("TkDefaultFont", root=widget)
    return "\n".join([
        f"Платформа:            {sys.platform}",
        f"Python:               {sys.version.split()[0]}",
        f"Tk:                   {widget.tk.call('info', 'patchlevel')}",
        f"DPI-awareness:        {process_awareness()}",
        f"Применённый режим:    {applied_mode()}",
        f"DPI монитора:         {dpi:.0f}  (масштаб Windows {dpi / 96 * 100:.0f}%)",
        f"Экран (Tk видит):     {width} × {height} px",
        f"tk scaling:           {scaling:.3f} px/pt",
        f"TkDefaultFont:        size={font.cget('size')}, linespace={font.metrics('linespace')} px",
        f"Множитель UI:         ×{ui_scale:.2f}",
    ])


class DpiWatcher:
    def __init__(self, widget: tk.Misc, on_change: Callable[[float], None], period_ms: int = 1000):
        self.widget = widget
        self.on_change = on_change
        self.period = period_ms
        self.last = get_dpi(widget)
        self._tick()

    def _tick(self) -> None:
        try:
            current = get_dpi(self.widget)
        except tk.TclError:
            return
        if abs(current - self.last) > 0.5:
            self.last = current
            try:
                self.on_change(current)
            except tk.TclError:
                return
        try:
            self.widget.after(self.period, self._tick)
        except tk.TclError:
            pass
