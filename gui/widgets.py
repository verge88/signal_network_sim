"""Reusable GUI widgets."""
from __future__ import annotations

import os
import tkinter as tk
from tkinter import filedialog, ttk

from .theme import style_menu, style_text

_ALLOWED = {"c", "C", "a", "A", "x", "X", "Insert", "Prior", "Next", "Home",
            "End", "Up", "Down", "Left", "Right", "Tab"}


class LogView(ttk.Frame):
    """Read-only text view with selection, copying, and saving."""

    def __init__(self, master, pal: dict, height: int = 18,
                 max_lines: int = 20000, file_hint: str = ""):
        super().__init__(master)
        self.max_lines = max_lines
        self.file_hint = file_hint
        self.text = tk.Text(self, wrap="none", height=height, font="TkFixedFont",
                            relief="flat", undo=False, exportselection=True)
        ys = ttk.Scrollbar(self, orient="vertical", command=self.text.yview)
        xs = ttk.Scrollbar(self, orient="horizontal", command=self.text.xview)
        self.text.configure(yscrollcommand=ys.set, xscrollcommand=xs.set)
        self.text.grid(row=0, column=0, sticky="nsew")
        ys.grid(row=0, column=1, sticky="ns")
        xs.grid(row=1, column=0, sticky="ew")
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)
        style_text(self.text, pal)

        bar = ttk.Frame(self)
        bar.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(4, 0))
        ttk.Button(bar, text="Копировать выделенное", command=self.copy_selection).pack(side="left")
        ttk.Button(bar, text="Копировать всё", command=self.copy_all).pack(side="left", padx=4)
        ttk.Button(bar, text="Сохранить…", command=self.save_as).pack(side="left")
        ttk.Button(bar, text="Очистить", command=self.clear).pack(side="left", padx=4)
        self.autoscroll = tk.BooleanVar(value=True)
        ttk.Checkbutton(bar, text="Автопрокрутка", variable=self.autoscroll).pack(side="right")

        self.menu = tk.Menu(self, tearoff=0)
        for label, command in (("Копировать", self.copy_selection),
                               ("Копировать всё", self.copy_all),
                               ("Выделить всё", self.select_all),
                               ("Сохранить в файл…", self.save_as),
                               ("Открыть log.txt", self.open_file),
                               ("Очистить", self.clear)):
            self.menu.add_command(label=label, command=command)
        style_menu(self.menu, pal)

        self.text.bind("<Key>", self._on_key)
        self.text.bind("<<Paste>>", lambda _event: "break")
        self.text.bind("<Button-3>", self._popup)
        self.text.bind("<Control-c>", lambda _event: (self.copy_selection(), "break")[1])
        self.text.bind("<Control-a>", lambda _event: (self.select_all(), "break")[1])
        self.text.bind("<Control-Insert>", lambda _event: (self.copy_selection(), "break")[1])

    @staticmethod
    def _on_key(event):
        if event.state & 0x0004 and event.keysym in _ALLOWED:
            return None
        if event.keysym in ("Up", "Down", "Left", "Right", "Prior", "Next",
                            "Home", "End"):
            return None
        return "break"

    def _popup(self, event) -> None:
        try:
            self.menu.tk_popup(event.x_root, event.y_root)
        finally:
            self.menu.grab_release()

    def append(self, message: str) -> None:
        at_bottom = self.text.yview()[1] > 0.999
        has_selection = bool(self.text.tag_ranges("sel"))
        self.text.insert("end", message if message.endswith("\n") else message + "\n")
        lines = int(self.text.index("end-1c").split(".")[0])
        if lines > self.max_lines and not has_selection:
            self.text.delete("1.0", f"{lines - self.max_lines + 500}.0")
        if self.autoscroll.get() and at_bottom and not has_selection:
            self.text.see("end")

    def clear(self) -> None:
        self.text.delete("1.0", "end")

    def copy_selection(self) -> None:
        try:
            data = self.text.get("sel.first", "sel.last")
        except tk.TclError:
            return self.copy_all()
        self._to_clipboard(data)

    def copy_all(self) -> None:
        self._to_clipboard(self.text.get("1.0", "end-1c"))

    def _to_clipboard(self, data: str) -> None:
        if data:
            self.clipboard_clear()
            self.clipboard_append(data)
            self.update_idletasks()

    def select_all(self) -> None:
        self.text.tag_add("sel", "1.0", "end-1c")
        self.text.focus_set()

    def save_as(self) -> None:
        path = filedialog.asksaveasfilename(parent=self, defaultextension=".txt",
                                            filetypes=[("Текст", "*.txt")])
        if path:
            with open(path, "w", encoding="utf-8") as stream:
                stream.write(self.text.get("1.0", "end-1c"))

    def open_file(self) -> None:
        if self.file_hint and os.path.exists(self.file_hint):
            try:
                os.startfile(self.file_hint)  # type: ignore[attr-defined]
            except AttributeError:
                os.system(f'xdg-open "{self.file_hint}"')
