"""Scrollable content container for long inspector forms."""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk


class InspectorScroll(ttk.Frame):
    """Keep long topology property forms reachable in small windows."""

    def __init__(self, master, palette: dict):
        super().__init__(master)
        self.canvas = tk.Canvas(self, borderwidth=0, highlightthickness=0,
                                bg=palette["panel_bg"])
        scrollbar = ttk.Scrollbar(self, orient="vertical",
                                  command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=scrollbar.set)
        self.canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        self.content = ttk.Frame(self.canvas, padding=(10, 10))
        self._window_id = self.canvas.create_window(
            (0, 0), window=self.content, anchor="nw"
        )
        self.content.bind("<Configure>", self._content_changed)
        self.canvas.bind("<Configure>", self._canvas_changed)
        self.canvas.bind("<MouseWheel>", self._mousewheel)
        self.canvas.bind("<Button-4>", lambda _event: self.scroll_units(-1))
        self.canvas.bind("<Button-5>", lambda _event: self.scroll_units(1))

    def _content_changed(self, _event=None):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _canvas_changed(self, event):
        self.canvas.itemconfigure(self._window_id, width=event.width)

    def _mousewheel(self, event):
        if event.delta:
            self.scroll_units(-1 if event.delta > 0 else 1)
        return "break"

    def scroll_units(self, amount: int) -> None:
        self.canvas.yview_scroll(amount, "units")

    def set_palette(self, palette: dict) -> None:
        self.canvas.configure(bg=palette["panel_bg"])
