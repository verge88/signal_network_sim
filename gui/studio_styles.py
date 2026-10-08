"""Additional ttk styling for the Signal Network editor.

Keeps the UI within the standard-library Tkinter dependency.
"""
from __future__ import annotations

from tkinter import ttk


def _mix(color: str, target: str, amount: float) -> str:
    """Blend two #RRGGBB colors for active/hover states."""
    a = [int(color[i:i + 2], 16) for i in (1, 3, 5)]
    b = [int(target[i:i + 2], 16) for i in (1, 3, 5)]
    rgb = [round(x * (1 - amount) + y * amount) for x, y in zip(a, b)]
    return "#{:02x}{:02x}{:02x}".format(*rgb)


def apply_studio_styles(root, palette: dict) -> None:
    """Apply editor-specific styles after the selected built-in theme."""
    style = ttk.Style(root)
    bg = palette["panel_bg"]
    fg = palette["panel_fg"]
    muted = palette["hint_fg"]
    border = palette["grid_major"]
    accent = palette["accent"]
    field = palette["field_bg"]
    button = palette["button_bg"]
    hover = _mix(button, accent, 0.14)

    style.configure("Studio.TFrame", background=bg)
    style.configure("StudioFooter.TFrame", background=bg)
    style.configure("StudioBrand.TLabel", background=bg, foreground=fg,
                    font=("TkDefaultFont", 15, "bold"))
    style.configure("StudioEyebrow.TLabel", background=bg, foreground=muted,
                    font=("TkDefaultFont", 9))
    style.configure("StudioStats.TLabel", background=bg, foreground=fg,
                    font=("TkDefaultFont", 9, "bold"))
    style.configure("StudioZoom.TLabel", background=bg, foreground=accent,
                    font=("TkDefaultFont", 10, "bold"))

    style.configure("StudioAccent.TButton", background=accent,
                    foreground="#ffffff", bordercolor=accent,
                    relief="flat", padding=(13, 8))
    style.map("StudioAccent.TButton",
              background=[("disabled", button),
                          ("pressed", _mix(accent, "#000000", 0.23)),
                          ("active", _mix(accent, "#000000", 0.13))],
              foreground=[("disabled", muted), ("active", "#ffffff"),
                          ("pressed", "#ffffff")])
    style.configure("StudioSecondary.TButton", background=button, foreground=fg,
                    bordercolor=border, relief="flat", padding=(11, 7))
    style.map("StudioSecondary.TButton",
              background=[("active", hover), ("pressed", accent)],
              foreground=[("pressed", "#ffffff")])
    style.configure("StudioSmall.TButton", background=button, foreground=fg,
                    bordercolor=border, relief="flat", padding=(8, 4))
    style.map("StudioSmall.TButton", background=[("active", hover)])
    style.configure("StudioMode.TRadiobutton", background=button,
                    foreground=fg, padding=(9, 6))
    style.map("StudioMode.TRadiobutton",
              background=[("selected", _mix(button, accent, 0.18)),
                          ("active", hover)],
              foreground=[("selected", accent)])
    style.configure("Studio.TNotebook.Tab", padding=(12, 9))
    style.configure("Treeview", rowheight=28, borderwidth=0)
    style.configure("Treeview.Heading", padding=(7, 7), relief="flat")
    style.configure("TEntry", padding=(7, 6))
    style.configure("TCombobox", padding=(7, 5))
    style.configure("TButton", padding=(10, 6))
    style.configure("TScrollbar", background=button, troughcolor=field)
