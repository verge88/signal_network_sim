"""Dialog for live editor view settings."""
from __future__ import annotations

import tkinter as tk
from dataclasses import asdict
from tkinter import messagebox, ttk
from typing import Callable

from . import hidpi
from .theme import THEMES, WINDOW_PRESETS, ViewSettings


class ViewSettingsDialog(tk.Toplevel):
    def __init__(self, master, view: ViewSettings,
                 on_apply: Callable[[ViewSettings], None],
                 on_geometry: Callable[[str], None]):
        super().__init__(master)
        self.title("Настройки отображения")
        self.transient(master.winfo_toplevel())
        self.resizable(False, False)
        self.view = view
        self.on_apply = on_apply
        self.on_geometry = on_geometry
        self.vars: dict[str, tk.Variable] = {}
        self._label2key = {spec["label"]: key for key, spec in THEMES.items()}
        self.configure(bg=view.palette()["panel_bg"])

        notebook = ttk.Notebook(self)
        notebook.pack(fill="both", expand=True, padx=10, pady=10)
        notebook.add(self._tab_theme(notebook), text="Оформление")
        notebook.add(self._tab_resolution(notebook), text="Разрешение")
        notebook.add(self._tab_geometry(notebook), text="Схема")
        row = ttk.Frame(self, padding=(10, 0, 10, 10))
        row.pack(fill="x")
        ttk.Button(row, text="Применить", command=self.apply).pack(side="left")
        ttk.Button(row, text="Сохранить по умолчанию", command=self.save_default).pack(side="left", padx=6)
        ttk.Button(row, text="Сбросить", command=self.reset).pack(side="left")
        ttk.Button(row, text="Закрыть", command=self.destroy).pack(side="right")
        self.bind("<Return>", lambda _event: self.apply())
        self.bind("<Escape>", lambda _event: self.destroy())

    def _heading(self, parent, row: int, text: str) -> None:
        ttk.Label(parent, text=text, style="Title.TLabel").grid(row=row, column=0, columnspan=2, sticky="w", pady=(0, 6))

    def _tab_theme(self, notebook) -> ttk.Frame:
        frame = ttk.Frame(notebook, padding=12)
        self._heading(frame, 0, "Тема оформления")
        var = tk.StringVar(value=THEMES[self.view.theme]["label"])
        self.vars["theme"] = var
        box = ttk.Combobox(frame, textvariable=var, state="readonly", width=34,
                           values=[spec["label"] for spec in THEMES.values()])
        box.grid(row=1, column=0, columnspan=2, sticky="ew")
        box.bind("<<ComboboxSelected>>", lambda _event: self.apply())
        ttk.Label(frame, style="Hint.TLabel", wraplength=330, justify="left",
                  text="Тема применяется к схеме, панелям, меню, журналу и экспорту.").grid(row=2, column=0, columnspan=2, sticky="w", pady=8)
        self._check(frame, 3, "Показывать сетку", "show_grid")
        self._spin(frame, 4, "Шаг сетки, px", "grid_step", 10, 200, 5)
        return frame

    def _tab_resolution(self, notebook) -> ttk.Frame:
        frame = ttk.Frame(notebook, padding=12)
        self._heading(frame, 0, "Масштаб интерфейса (DPI)")
        self._spin(frame, 1, "Плотность пикселей ×", "ui_scale", 0.6, 3.0, 0.05)
        self._spin(frame, 2, "Размер шрифтов ×", "font_scale", 0.6, 3.0, 0.05)
        self._check(frame, 3, "Определять масштаб по DPI монитора", "auto_dpi")
        dpi = hidpi.get_dpi(self)
        ttk.Label(frame, style="Hint.TLabel", wraplength=330, justify="left",
                  text=f"Текущий DPI: {dpi:.0f} ({dpi / 96 * 100:.0f}% Windows). "
                       "Автоопределение применяется поверх системного масштаба.").grid(
            row=4, column=0, columnspan=2, sticky="w", pady=(4, 10))
        ttk.Separator(frame).grid(row=5, column=0, columnspan=2, sticky="ew", pady=8)
        self._heading(frame, 6, "Размер окна")
        geometry = tk.StringVar(value=self.view.window_geometry)
        self.vars["window_geometry"] = geometry
        ttk.Combobox(frame, textvariable=geometry, width=32, values=[g for _, g in WINDOW_PRESETS]).grid(row=7, column=0, columnspan=2, sticky="ew", pady=4)
        ttk.Button(frame, text="Применить размер окна", command=lambda: self.on_geometry(geometry.get())).grid(row=8, column=0, sticky="w")
        ttk.Button(frame, text="Развернуть на весь экран", command=lambda: self.on_geometry("maximize")).grid(row=8, column=1, sticky="e")
        ttk.Separator(frame).grid(row=9, column=0, columnspan=2, sticky="ew", pady=8)
        self._heading(frame, 10, "Область просмотра и раскладки")
        self._spin(frame, 11, "Логическая ширина поля, px", "layout_width", 400, 8000, 100)
        self._spin(frame, 12, "Логическая высота поля, px", "layout_height", 400, 8000, 100)
        self._spin(frame, 13, "Ширина боковой панели, px", "panel_width", 220, 700, 10)
        self._spin(frame, 14, "Кратность экспорта PNG ×", "export_scale", 1.0, 8.0, 0.5)
        return frame

    def _tab_geometry(self, notebook) -> ttk.Frame:
        frame = ttk.Frame(notebook, padding=12)
        self._heading(frame, 0, "Элементы схемы")
        self._spin(frame, 1, "Радиус узла, px", "node_radius", 8, 80, 1)
        self._spin(frame, 2, "Кегль номера узла", "id_pt", 6, 24, 1)
        self._spin(frame, 3, "Кегль подписи типа", "label_pt", 6, 24, 1)
        self._spin(frame, 4, "Толщина линков ×", "link_width_scale", 0.3, 4.0, 0.1)
        return frame

    def _spin(self, parent, row: int, label: str, key: str, lo: float, hi: float, step: float) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=2)
        value = tk.StringVar(value=str(getattr(self.view, key)))
        self.vars[key] = value
        spin = ttk.Spinbox(parent, from_=lo, to=hi, increment=step, textvariable=value, width=10, command=self.apply)
        spin.grid(row=row, column=1, sticky="e", pady=2)
        spin.bind("<Return>", lambda _event: self.apply())

    def _check(self, parent, row: int, label: str, key: str) -> None:
        value = tk.BooleanVar(value=bool(getattr(self.view, key)))
        self.vars[key] = value
        ttk.Checkbutton(parent, text=label, variable=value, command=self.apply).grid(row=row, column=0, columnspan=2, sticky="w", pady=2)

    def _collect(self) -> bool:
        try:
            for key, variable in self.vars.items():
                if key == "theme":
                    self.view.theme = self._label2key.get(variable.get(), self.view.theme)
                elif key == "window_geometry":
                    self.view.window_geometry = variable.get().strip()
                else:
                    current = getattr(self.view, key)
                    raw = variable.get()
                    if isinstance(current, bool):
                        value = bool(raw) if isinstance(raw, bool) else str(raw).lower() in {"1", "true", "yes", "on"}
                    else:
                        value = type(current)(raw)
                    setattr(self.view, key, value)
        except (ValueError, tk.TclError) as exc:
            messagebox.showerror("Некорректное значение", str(exc), parent=self)
            return False
        return True

    def apply(self) -> None:
        if self._collect():
            self.on_apply(self.view)
            self.configure(bg=self.view.palette()["panel_bg"])

    def save_default(self) -> None:
        if self._collect():
            self.on_apply(self.view)
            self.view.save()
            messagebox.showinfo("Настройки", "Сохранено как настройки по умолчанию.", parent=self)

    def reset(self) -> None:
        fresh = ViewSettings()
        for key, value in asdict(fresh).items():
            setattr(self.view, key, value)
        self.on_apply(self.view)
        for key, variable in self.vars.items():
            variable.set(THEMES[self.view.theme]["label"] if key == "theme" else getattr(self.view, key))
