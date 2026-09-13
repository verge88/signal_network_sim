"""Themes and persistent view settings for the topology editor."""
from __future__ import annotations

import json
import os
import tkinter as tk
from dataclasses import asdict, dataclass, fields
from tkinter import font as tkfont, ttk

CONFIG_DIR = os.path.join(os.path.expanduser("~"), ".config", "signal_network_sim")
CONFIG_PATH = os.path.join(CONFIG_DIR, "gui_view.json")

THEMES: dict[str, dict] = {
    "light": {
        "label": "Светлая", "ttk": "clam", "canvas_bg": "#f7f8fa",
        "grid": "#e8ebf1", "grid_major": "#d5dae4", "link": "#8d97a8",
        "link_hi": "#d04a4a", "node_text": "#ffffff", "node_label": "#2c313a",
        "outline": "#5a6273", "outline_master": "#1b1b1b",
        "outline_compromised": "#d04a4a", "sel_ring": "#d04a4a",
        "pending": "#2f8f3f", "node_shade": 1.0, "panel_bg": "#f0f1f4",
        "panel_fg": "#1d1f24", "field_bg": "#ffffff", "field_fg": "#1d1f24",
        "button_bg": "#e2e5ea", "tab_bg": "#dfe3ea", "accent": "#3f7fc4",
        "status_fg": "#4a5160", "hint_fg": "#5b6270",
        "zones": ["#4a7fb5", "#b5744a", "#5aa06a", "#8a5ab5", "#b5525a", "#4aa8a8", "#9a9a4a", "#7a7a8a"],
    },
    "dark": {
        "label": "Тёмная", "ttk": "clam", "canvas_bg": "#171b22",
        "grid": "#1f252e", "grid_major": "#2b333f", "link": "#5d6878",
        "link_hi": "#ff6b6b", "node_text": "#f2f5fa", "node_label": "#b6c0ce",
        "outline": "#7d8899", "outline_master": "#e8edf5",
        "outline_compromised": "#ff6b6b", "sel_ring": "#ffb347",
        "pending": "#4bd07a", "node_shade": 0.92, "panel_bg": "#1e232b",
        "panel_fg": "#e4e9f1", "field_bg": "#12161c", "field_fg": "#e4e9f1",
        "button_bg": "#2a313b", "tab_bg": "#232a33", "accent": "#5b9fe0",
        "status_fg": "#9aa5b5", "hint_fg": "#8a95a5",
        "zones": ["#5b9fe0", "#e09a5b", "#63c98a", "#b184e8", "#e8737f", "#4fc8c8", "#c9c46a", "#93a0b2"],
    },
    "midnight": {
        "label": "Midnight (макс. контраст / OLED)", "ttk": "clam", "canvas_bg": "#07090c",
        "grid": "#12161c", "grid_major": "#1c222b", "link": "#4f5b6b",
        "link_hi": "#ff5252", "node_text": "#ffffff", "node_label": "#c9d3e0",
        "outline": "#8b97a8", "outline_master": "#ffffff",
        "outline_compromised": "#ff5252", "sel_ring": "#ffd166",
        "pending": "#3ddc84", "node_shade": 1.06, "panel_bg": "#0e1116",
        "panel_fg": "#eef2f8", "field_bg": "#07090c", "field_fg": "#eef2f8",
        "button_bg": "#1a2028", "tab_bg": "#141a21", "accent": "#66b3ff",
        "status_fg": "#a8b4c4", "hint_fg": "#93a0b0",
        "zones": ["#66b3ff", "#ffab5e", "#5fe39a", "#c08cff", "#ff7a86", "#55dede", "#e0da72", "#a4b1c2"],
    },
}

WINDOW_PRESETS = [
    ("1280 × 800 (ноутбук)", "1280x800"), ("1440 × 900", "1440x900"),
    ("1600 × 1000", "1600x1000"), ("1920 × 1200 (Full HD+)", "1920x1200"),
    ("2560 × 1440 (QHD)", "2560x1440"), ("3200 × 1800", "3200x1800"),
]


def palette_of(theme: str) -> dict:
    return THEMES.get(theme, THEMES["light"])


def shade(hex_color: str, factor: float) -> str:
    if factor == 1.0 or not hex_color.startswith("#") or len(hex_color) != 7:
        return hex_color
    rgb = [int(hex_color[i:i + 2], 16) for i in (1, 3, 5)]
    rgb = [max(0, min(255, int(round(value * factor)))) for value in rgb]
    return "#%02x%02x%02x" % tuple(rgb)


def node_fill(base_color: str, pal: dict) -> str:
    return shade(base_color, float(pal.get("node_shade", 1.0)))


def zone_color(zone_id: int, pal: dict) -> str:
    zones = pal.get("zones") or THEMES["light"]["zones"]
    return zones[int(zone_id) % len(zones)]


_BASE_PT: dict[str, float] = {}
_CAPTURE_DPI = 96.0
_FONT_NAMES = ("TkDefaultFont", "TkTextFont", "TkMenuFont", "TkHeadingFont",
               "TkFixedFont", "TkSmallCaptionFont", "TkCaptionFont",
               "TkTooltipFont", "TkIconFont")


def capture_base_fonts(root: tk.Misc, dpi: float) -> None:
    global _CAPTURE_DPI
    if _BASE_PT:
        return
    _CAPTURE_DPI = max(48.0, float(dpi))
    for name in _FONT_NAMES:
        try:
            font = tkfont.nametofont(name, root=root)
        except tk.TclError:
            continue
        size = int(font.cget("size"))
        _BASE_PT[name] = float(size) if size > 0 else abs(size) * 72.0 / _CAPTURE_DPI if size < 0 else 9.0


def apply_ui_scale(root: tk.Misc, dpi: float = 96.0, ui_scale: float = 1.0,
                   font_scale: float = 1.0) -> float:
    dpi = max(48.0, float(dpi))
    ui_scale = max(0.5, min(4.0, float(ui_scale)))
    font_scale = max(0.5, min(4.0, float(font_scale)))
    capture_base_fonts(root, dpi)
    geometry_scale = dpi / 96.0 * ui_scale
    try:
        root.tk.call("tk", "scaling", (dpi / 72.0) * ui_scale)
    except tk.TclError:
        pass
    for name, points in _BASE_PT.items():
        try:
            font = tkfont.nametofont(name, root=root)
        except tk.TclError:
            continue
        pixels = points * dpi / 72.0 * ui_scale * font_scale
        font.configure(size=-max(7, int(round(pixels))))
    return geometry_scale


def apply_ttk_theme(root: tk.Misc, pal: dict) -> None:
    style = ttk.Style(root)
    try:
        style.theme_use(pal["ttk"])
    except tk.TclError:
        pass
    bg, fg, field, accent = pal["panel_bg"], pal["panel_fg"], pal["field_bg"], pal["accent"]
    style.configure(".", background=bg, foreground=fg, fieldbackground=field,
                    bordercolor=pal["grid_major"], lightcolor=bg, darkcolor=bg,
                    troughcolor=pal["tab_bg"], focuscolor=accent)
    style.configure("TFrame", background=bg)
    style.configure("TLabelframe", background=bg, foreground=fg)
    style.configure("TLabelframe.Label", background=bg, foreground=fg)
    style.configure("TLabel", background=bg, foreground=fg)
    style.configure("Hint.TLabel", background=bg, foreground=pal["hint_fg"])
    style.configure("Status.TLabel", background=bg, foreground=pal["status_fg"])
    style.configure("Title.TLabel", background=bg, foreground=accent)
    style.configure("TButton", background=pal["button_bg"], foreground=fg,
                    bordercolor=pal["grid_major"], focusthickness=1, padding=(8, 4))
    style.map("TButton", background=[("pressed", accent), ("active", shade(pal["button_bg"], 1.18))],
              foreground=[("pressed", "#ffffff")])
    for cls in ("TCheckbutton", "TRadiobutton"):
        style.configure(cls, background=bg, foreground=fg, indicatorcolor=field, focuscolor=bg)
        style.map(cls, background=[("active", bg)], indicatorcolor=[("selected", accent)])
    style.configure("TEntry", fieldbackground=field, foreground=fg, insertcolor=fg, bordercolor=pal["grid_major"])
    style.configure("TSpinbox", fieldbackground=field, foreground=fg, arrowcolor=fg, insertcolor=fg, bordercolor=pal["grid_major"])
    style.configure("TCombobox", fieldbackground=field, background=pal["button_bg"], foreground=fg, arrowcolor=fg, bordercolor=pal["grid_major"])
    style.map("TCombobox", fieldbackground=[("readonly", field), ("disabled", bg)], foreground=[("readonly", fg)], selectbackground=[("readonly", field)], selectforeground=[("readonly", fg)])
    style.configure("TNotebook", background=bg, borderwidth=0)
    style.configure("TNotebook.Tab", background=pal["tab_bg"], foreground=fg, padding=(10, 5), borderwidth=0)
    style.map("TNotebook.Tab", background=[("selected", bg), ("active", shade(pal["tab_bg"], 1.15))], foreground=[("selected", accent)])
    try:
        line = tkfont.nametofont("TkDefaultFont", root=root).metrics("linespace")
    except tk.TclError:
        line = 16
    style.configure("Treeview", background=field, fieldbackground=field, foreground=fg,
                    bordercolor=pal["grid_major"], rowheight=int(line * 1.45))
    style.configure("Treeview.Heading", background=pal["tab_bg"], foreground=fg, relief="flat")
    style.map("Treeview", background=[("selected", accent)], foreground=[("selected", "#ffffff")])
    style.configure("TSeparator", background=pal["grid_major"])
    style.configure("TScale", background=bg, troughcolor=pal["tab_bg"])
    style.configure("TScrollbar", background=pal["button_bg"], troughcolor=pal["tab_bg"], arrowcolor=fg)
    style.configure("TPanedwindow", background=bg)
    try:
        root.winfo_toplevel().configure(bg=bg)
    except tk.TclError:
        pass
    for opt, value in (("*TCombobox*Listbox.background", field), ("*TCombobox*Listbox.foreground", fg),
                       ("*TCombobox*Listbox.selectBackground", accent), ("*TCombobox*Listbox.selectForeground", "#ffffff")):
        root.option_add(opt, value)


def style_menu(menu: tk.Menu, pal: dict) -> None:
    try:
        menu.configure(bg=pal["panel_bg"], fg=pal["panel_fg"], activebackground=pal["accent"],
                       activeforeground="#ffffff", disabledforeground=pal["hint_fg"],
                       selectcolor=pal["accent"], relief="flat", borderwidth=0, tearoff=0)
    except tk.TclError:
        pass


def style_text(widget: tk.Text, pal: dict) -> None:
    try:
        widget.configure(bg=pal["field_bg"], fg=pal["field_fg"], insertbackground=pal["field_fg"],
                         selectbackground=pal["accent"], selectforeground="#ffffff", highlightthickness=1,
                         highlightbackground=pal["grid_major"], highlightcolor=pal["accent"], relief="flat")
    except tk.TclError:
        pass


@dataclass
class ViewSettings:
    theme: str = "light"
    ui_scale: float = 1.0
    font_scale: float = 1.0
    auto_dpi: bool = True
    window_geometry: str = ""
    panel_width: int = 340
    panel_visible: bool = True
    toolbar_visible: bool = True
    node_radius: float = 22.0
    id_pt: int = 9
    label_pt: int = 8
    link_width_scale: float = 1.0
    show_grid: bool = True
    grid_step: float = 40.0
    layout_width: float = 1600.0
    layout_height: float = 1000.0
    export_scale: float = 3.0

    def palette(self) -> dict:
        return palette_of(self.theme)

    def save(self, path: str = CONFIG_PATH) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as stream:
            json.dump(asdict(self), stream, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str = CONFIG_PATH) -> "ViewSettings":
        result = cls()
        try:
            with open(path, encoding="utf-8") as stream:
                data = json.load(stream)
        except (OSError, ValueError):
            return result
        for field in fields(cls):
            if field.name not in data:
                continue
            value = data[field.name]
            try:
                if isinstance(getattr(result, field.name), bool):
                    value = bool(value) if isinstance(value, bool) else str(value).lower() in {"1", "true", "yes", "on"}
                else:
                    value = type(getattr(result, field.name))(value)
                setattr(result, field.name, value)
            except (TypeError, ValueError):
                pass
        if result.theme not in THEMES:
            result.theme = "light"
        result.ui_scale = max(0.6, min(3.0, result.ui_scale))
        result.font_scale = max(0.6, min(3.0, result.font_scale))
        result.node_radius = max(8.0, min(80.0, result.node_radius))
        result.export_scale = max(1.0, min(8.0, result.export_scale))
        return result

    def copy(self) -> "ViewSettings":
        return ViewSettings(**asdict(self))
