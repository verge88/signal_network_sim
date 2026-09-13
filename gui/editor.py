"""
gui/editor.py — GUI создания, визуализации и редактирования сигнальной сети.

Зависимости: только стандартная библиотека (tkinter). networkx/matplotlib
используются опционально: первый — для раскладок, второй — для экспорта
картинки топологии.

Управление:
  ЛКМ по узлу            — выбрать / перетащить
  ЛКМ по линку           — выбрать линк
  режим "Линк" + 2 клика — соединить узлы
  режим "Узел" + клик    — создать узел выбранного типа
  ПКМ по узлу/линку      — контекстное меню
  Delete                 — удалить выбранное
  колесо мыши            — зум, средняя кнопка / Shift+ЛКМ — панорама
  M — сделать MASTER/SLAVE, C — пометить compromised, A — автораскладка
"""
from __future__ import annotations

import json
import os
import queue
import tempfile
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog

from . import hidpi
from .model import PROTOCOLS, Topology, Node, Link, Scenario, link_key, template
from .theme import (THEMES, WINDOW_PRESETS, ViewSettings, apply_ttk_theme,
                    apply_ui_scale, node_fill, palette_of, style_menu,
                    style_text, zone_color)
from .view_settings import ViewSettingsDialog


class TopologyEditor(tk.Frame):
    def __init__(self, master: tk.Tk):
        super().__init__(master)
        self.pack(fill="both", expand=True)
        self.view = ViewSettings.load()
        self.pal = self.view.palette()
        self._texts: list[tk.Text] = []
        self._menus: list[tk.Menu] = []
        self._fullscreen = False
        self.dpi = hidpi.get_dpi(master)
        self.gs = apply_ui_scale(master, self.dpi if self.view.auto_dpi else 96.0,
                     self.view.ui_scale, self.view.font_scale)
        apply_ttk_theme(master, self.pal)
        self.topo = template("diameter")
        self.path: str | None = None
        self.dirty = False

        self.scale = 1.0
        self.offset = [0.0, 0.0]
        self.mode = tk.StringVar(value="select")
        self.new_type = tk.StringVar(value=self.topo.node_type_names()[0])
        self.color_by_zone = tk.BooleanVar(value=False)
        self.show_labels = tk.BooleanVar(value=True)
        self.theme_var = tk.StringVar(value=self.view.theme)
        self.selection: tuple[str, object] | None = None
        self._pending_link: int | None = None
        self._drag: dict | None = None
        self._log_q: "queue.Queue[str]" = queue.Queue()

        self._build_menu()
        self._build_toolbar()
        self._build_body()
        self._bind_events()
        self.apply_view(self.view, initial=True)
        self.dpi_watcher = hidpi.DpiWatcher(self, self._on_dpi_change)
        self.after(200, self._drain_log)

    # ── интерфейс ──────────────────────────────────────────────────
    def _build_menu(self) -> None:
        m = tk.Menu(self.master)
        f = tk.Menu(m, tearoff=0)
        f.add_command(label="Новая сеть…", accelerator="Ctrl+N", command=self.new_network)
        f.add_command(label="Открыть JSON…", accelerator="Ctrl+O", command=self.open_file)
        f.add_command(label="Сохранить", accelerator="Ctrl+S", command=self.save_file)
        f.add_command(label="Сохранить как…", command=lambda: self.save_file(True))
        f.add_separator()
        f.add_command(label="Экспорт GraphML…", command=self.export_graphml)
        f.add_command(label="Экспорт PNG/PS…", command=self.export_image)
        f.add_command(label="Экспорт SVG…", command=self.export_svg)
        f.add_separator()
        f.add_command(label="Выход", command=self.master.destroy)
        m.add_cascade(label="Файл", menu=f)

        t = tk.Menu(m, tearoff=0)
        for proto, spec in PROTOCOLS.items():
            t.add_command(label=f"Шаблон: {spec['label']}",
                          command=lambda p=proto: self.load_template(p))
        m.add_cascade(label="Шаблоны", menu=t)

        e = tk.Menu(m, tearoff=0)
        for kind, label in (("spring", "Spring"), ("kamada", "Kamada-Kawai"),
                            ("circular", "Circular"), ("shell", "Shell (хабы в центре)")):
            e.add_command(label=f"Раскладка: {label}",
                          command=lambda k=kind: self.do_layout(k))
        e.add_separator()
        e.add_command(label="Пересчитать зоны мониторинга", command=self.do_zones)
        e.add_command(label="Проверить топологию", command=self.do_validate)
        m.add_cascade(label="Правка", menu=e)

        r = tk.Menu(m, tearoff=0)
        r.add_command(label="Запустить симулятор…", command=self.run_simulation)
        r.add_command(label="Показать сводку по сети", command=self.show_summary)
        m.add_cascade(label="Симуляция", menu=r)

        v = tk.Menu(m, tearoff=0)
        for key, spec in THEMES.items():
            v.add_radiobutton(label=spec["label"], value=key,
                              variable=self.theme_var,
                              command=lambda: self.set_theme(self.theme_var.get()))
        v.add_separator()
        v.add_command(label="Настройки отображения…", accelerator="Ctrl+,",
                      command=self.open_view_settings)
        v.add_separator()
        v.add_command(label="Увеличить масштаб интерфейса", accelerator="Ctrl++",
                      command=lambda: self.bump_ui_scale(+0.1))
        v.add_command(label="Уменьшить масштаб интерфейса", accelerator="Ctrl+-",
                      command=lambda: self.bump_ui_scale(-0.1))
        v.add_command(label="Сбросить масштаб интерфейса", accelerator="Ctrl+0",
                      command=lambda: self.bump_ui_scale(None))
        v.add_separator()
        size_menu = tk.Menu(v, tearoff=0)
        for label, geometry in WINDOW_PRESETS:
            size_menu.add_command(label=label, command=lambda g=geometry: self.set_geometry(g))
        size_menu.add_separator()
        size_menu.add_command(label="Под экран (85%)",
                      command=lambda: self.set_geometry(hidpi.default_geometry(self.master, 0.85)))
        size_menu.add_command(label="Под экран (100%)",
                      command=lambda: self.set_geometry(hidpi.default_geometry(self.master, 1.0)))
        size_menu.add_separator()
        size_menu.add_command(label="Развернуть на весь экран", command=lambda: self.set_geometry("maximize"))
        v.add_cascade(label="Размер окна", menu=size_menu)
        v.add_separator()
        v.add_command(label="Боковая панель вкл/выкл", accelerator="F9", command=self.toggle_panel)
        v.add_command(label="Панель инструментов вкл/выкл", accelerator="F8", command=self.toggle_toolbar)
        v.add_command(label="Полный экран", accelerator="F11", command=self.toggle_fullscreen)
        v.add_command(label="Только схема (презентация)", accelerator="F12", command=self.presentation_mode)
        v.add_separator()
        v.add_command(label="Диагностика HiDPI…", command=self.show_dpi_info)
        m.add_cascade(label="Вид", menu=v)
        self._menus.extend([m, f, t, e, r, v, size_menu])

        self.master.config(menu=m)
        self.master.bind("<Control-n>", lambda e: self.new_network())
        self.master.bind("<Control-o>", lambda e: self.open_file())
        self.master.bind("<Control-s>", lambda e: self.save_file())

    def _build_toolbar(self) -> None:
        bar = ttk.Frame(self, padding=(6, 4))
        self.bar = bar
        bar.pack(side="top", fill="x")
        ttk.Label(bar, text="Режим:").pack(side="left")
        for val, label in (("select", "Выбор"), ("node", "Узел"),
                           ("link", "Линк"), ("delete", "Удалить")):
            ttk.Radiobutton(bar, text=label, value=val, variable=self.mode,
                            command=self._reset_pending).pack(side="left", padx=2)
        ttk.Separator(bar, orient="vertical").pack(side="left", fill="y", padx=8)
        ttk.Label(bar, text="Тип узла:").pack(side="left")
        self.type_box = ttk.Combobox(bar, textvariable=self.new_type, width=14,
                                     state="readonly",
                                     values=self.topo.node_type_names())
        self.type_box.pack(side="left", padx=4)
        ttk.Separator(bar, orient="vertical").pack(side="left", fill="y", padx=8)
        ttk.Checkbutton(bar, text="Раскраска по зонам", variable=self.color_by_zone,
                        command=self.redraw).pack(side="left")
        ttk.Checkbutton(bar, text="Подписи", variable=self.show_labels,
                        command=self.redraw).pack(side="left", padx=6)
        ttk.Button(bar, text="Вписать в окно", command=self.fit_view).pack(side="right")

    def _build_body(self) -> None:
        body = ttk.Frame(self)
        body.pack(fill="both", expand=True)

        left = ttk.Frame(body)
        left.pack(side="left", fill="both", expand=True)
        self.canvas = tk.Canvas(left, bg=self.pal["canvas_bg"], highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.status = ttk.Label(left, text="", anchor="w", padding=(6, 3), style="Status.TLabel")
        self.status.pack(fill="x")

        right = ttk.Frame(body, width=self.view.panel_width)
        self.right = right
        right.pack(side="right", fill="y")
        right.pack_propagate(False)

        nb = ttk.Notebook(right)
        nb.pack(fill="both", expand=True)
        self.prop_tab = ttk.Frame(nb, padding=8)
        self.scen_tab = ttk.Frame(nb, padding=8)
        self.sim_tab = ttk.Frame(nb, padding=8)
        self.log_tab = ttk.Frame(nb, padding=4)
        nb.add(self.prop_tab, text="Свойства")
        nb.add(self.scen_tab, text="Сценарии")
        nb.add(self.sim_tab, text="Параметры")
        nb.add(self.log_tab, text="Журнал")

        self._build_scenarios_tab()
        self._build_sim_tab()
        self.log = tk.Text(self.log_tab, height=10, wrap="word", font="TkFixedFont", relief="flat")
        self.log.pack(fill="both", expand=True)
        self._texts.append(self.log)
        self._show_properties()

    def _build_scenarios_tab(self) -> None:
        cols = ("kind", "name", "targets", "start", "end", "intensity")
        self.scen_tree = ttk.Treeview(self.scen_tab, columns=cols, show="headings",
                                      height=14)
        for c, w in zip(cols, (52, 120, 80, 46, 46, 56)):
            self.scen_tree.heading(c, text=c)
            self.scen_tree.column(c, width=w, anchor="center")
        self.scen_tree.pack(fill="both", expand=True)
        btns = ttk.Frame(self.scen_tab)
        btns.pack(fill="x", pady=6)
        ttk.Button(btns, text="Добавить", command=self.add_scenario).pack(side="left")
        ttk.Button(btns, text="Удалить", command=self.del_scenario).pack(side="left", padx=4)
        ttk.Label(self.scen_tab, wraplength=310, style="Hint.TLabel",
                  text="Цели берутся из текущего выделения на схеме, если оно есть. "
                       "Интервалы — в единицах interval_s.").pack(fill="x")

    def _build_sim_tab(self) -> None:
        self.sim_vars: dict[str, tk.StringVar] = {}
        for i, (k, v) in enumerate(self.topo.sim_params.items()):
            ttk.Label(self.sim_tab, text=k).grid(row=i, column=0, sticky="w", pady=2)
            var = tk.StringVar(value=str(v))
            self.sim_vars[k] = var
            ttk.Entry(self.sim_tab, textvariable=var, width=16).grid(
                row=i, column=1, sticky="e")
        r = len(self.sim_vars)
        ttk.Button(self.sim_tab, text="Применить", command=self.apply_sim_params).grid(
            row=r, column=0, columnspan=2, sticky="ew", pady=8)
        self.metrics_lbl = ttk.Label(self.sim_tab, justify="left", style="Hint.TLabel")
        self.metrics_lbl.grid(row=r + 1, column=0, columnspan=2, sticky="w")

    def _bind_events(self) -> None:
        c = self.canvas
        c.bind("<Button-1>", self.on_click)
        c.bind("<B1-Motion>", self.on_drag)
        c.bind("<ButtonRelease-1>", self.on_release)
        c.bind("<Button-3>", self.on_right_click)
        c.bind("<Double-Button-1>", self.on_double)
        c.bind("<Button-2>", self.on_pan_start)
        c.bind("<B2-Motion>", self.on_pan_move)
        c.bind("<MouseWheel>", self.on_wheel)
        c.bind("<Button-4>", lambda e: self.on_wheel(e, 120))
        c.bind("<Button-5>", lambda e: self.on_wheel(e, -120))
        c.bind("<Configure>", lambda e: self.redraw())
        self.master.bind("<Delete>", lambda e: self.delete_selection())
        self.master.bind("m", lambda e: self.toggle_master())
        self.master.bind("c", lambda e: self.toggle_compromised())
        self.master.bind("a", lambda e: self.do_layout("spring"))
        self.master.bind("<F8>", lambda e: self.toggle_toolbar())
        self.master.bind("<F9>", lambda e: self.toggle_panel())
        self.master.bind("<F11>", lambda e: self.toggle_fullscreen())
        self.master.bind("<F12>", lambda e: self.presentation_mode())
        self.master.bind("<Control-comma>", lambda e: self.open_view_settings())
        self.master.bind("<Control-plus>", lambda e: self.bump_ui_scale(+0.1))
        self.master.bind("<Control-equal>", lambda e: self.bump_ui_scale(+0.1))
        self.master.bind("<Control-minus>", lambda e: self.bump_ui_scale(-0.1))
        self.master.bind("<Control-0>", lambda e: self.bump_ui_scale(None))
        self.master.bind("<Control-KP_Add>", lambda e: self.bump_ui_scale(+0.1))
        self.master.bind("<Control-KP_Subtract>", lambda e: self.bump_ui_scale(-0.1))

    # ── координаты ─────────────────────────────────────────────────
    def apply_view(self, view: ViewSettings, initial: bool = False) -> None:
        self.view = view
        self.pal = palette_of(view.theme)
        self.theme_var.set(view.theme)
        self.dpi = hidpi.get_dpi(self.master)
        base_dpi = self.dpi if view.auto_dpi else 96.0
        self.gs = apply_ui_scale(self.master, base_dpi, view.ui_scale, view.font_scale)
        apply_ttk_theme(self.master, self.pal)
        for menu in self._menus:
            style_menu(menu, self.pal)
        for text in self._texts:
            style_text(text, self.pal)
        self.canvas.configure(bg=self.pal["canvas_bg"])
        self.right.configure(width=max(200, int(view.panel_width * self.gs)))
        self.master.minsize(int(760 * self.gs), int(520 * self.gs))
        if view.panel_visible and not self.right.winfo_ismapped():
            self.right.pack(side="right", fill="y")
        elif not view.panel_visible and self.right.winfo_ismapped():
            self.right.pack_forget()
        if view.toolbar_visible and not self.bar.winfo_ismapped():
            self.bar.pack(side="top", fill="x")
        elif not view.toolbar_visible and self.bar.winfo_ismapped():
            self.bar.pack_forget()
        if initial:
            self.set_geometry(view.window_geometry or hidpi.default_geometry(self.master))
        self._show_properties()
        self.redraw()

    def set_theme(self, theme: str) -> None:
        self.view.theme = theme
        self.apply_view(self.view)
        self.logln(f"тема: {THEMES[theme]['label']}")

    def open_view_settings(self) -> None:
        ViewSettingsDialog(self, self.view, on_apply=self.apply_view,
                           on_geometry=self.set_geometry)

    def bump_ui_scale(self, delta: float | None) -> None:
        self.view.ui_scale = 1.0 if delta is None else max(0.6, min(3.0, self.view.ui_scale + delta))
        self.view.font_scale = self.view.ui_scale
        self.apply_view(self.view)

    def show_dpi_info(self) -> None:
        info = hidpi.diagnostics(self.master, self.view.ui_scale)
        messagebox.showinfo("Диагностика HiDPI", info)
        self.logln(info)

    def _on_dpi_change(self, dpi: float) -> None:
        self.logln(f"DPI изменился: {self.dpi:.0f} → {dpi:.0f}")
        self.apply_view(self.view)
        self.after(60, self.fit_view)

    def set_geometry(self, geometry: str) -> None:
        if geometry == "maximize":
            try:
                self.master.state("zoomed")
            except tk.TclError:
                self.master.attributes("-zoomed", True)
            return
        try:
            self.master.state("normal")
            self.master.geometry(geometry)
            self.view.window_geometry = geometry
        except tk.TclError:
            messagebox.showerror("Размер окна", f"Некорректная геометрия: {geometry}")
        self.after(60, self.fit_view)

    def toggle_panel(self) -> None:
        self.view.panel_visible = not self.view.panel_visible
        self.apply_view(self.view)

    def toggle_toolbar(self) -> None:
        self.view.toolbar_visible = not self.view.toolbar_visible
        self.apply_view(self.view)

    def toggle_fullscreen(self) -> None:
        self._fullscreen = not self._fullscreen
        self.master.attributes("-fullscreen", self._fullscreen)
        self.after(80, self.fit_view)

    def presentation_mode(self) -> None:
        hide = self.view.panel_visible or self.view.toolbar_visible
        self.view.panel_visible = not hide
        self.view.toolbar_visible = not hide
        if hide != self._fullscreen:
            self.toggle_fullscreen()
        self.apply_view(self.view)
        self.after(80, self.fit_view)

    def style_text(self, widget: tk.Text) -> None:
        if widget not in self._texts:
            self._texts.append(widget)
        style_text(widget, self.pal)

    def w2s(self, x: float, y: float) -> tuple[float, float]:
        return x * self.scale + self.offset[0], y * self.scale + self.offset[1]

    def s2w(self, x: float, y: float) -> tuple[float, float]:
        return (x - self.offset[0]) / self.scale, (y - self.offset[1]) / self.scale

    def node_at(self, sx: float, sy: float) -> Node | None:
        for n in self.topo.nodes.values():
            nx_, ny_ = self.w2s(n.x, n.y)
            if (nx_ - sx) ** 2 + (ny_ - sy) ** 2 <= (self.view.node_radius * self.gs * self.scale) ** 2:
                return n
        return None

    def link_at(self, sx: float, sy: float, tol: float = 6.0) -> Link | None:
        for l in self.topo.links.values():
            a, b = self.topo.nodes.get(l.src), self.topo.nodes.get(l.dst)
            if not a or not b:
                continue
            x1, y1 = self.w2s(a.x, a.y)
            x2, y2 = self.w2s(b.x, b.y)
            dx, dy = x2 - x1, y2 - y1
            den = dx * dx + dy * dy
            if den == 0:
                continue
            t = max(0.0, min(1.0, ((sx - x1) * dx + (sy - y1) * dy) / den))
            px, py = x1 + t * dx, y1 + t * dy
            if (px - sx) ** 2 + (py - sy) ** 2 <= tol * tol:
                return l
        return None

    # ── отрисовка ──────────────────────────────────────────────────
    def redraw(self) -> None:
        c = self.canvas
        pal, view = self.pal, self.view
        c.delete("all")
        w = c.winfo_width() or 900
        h = c.winfo_height() or 700
        step = max(6.0, view.grid_step) * self.gs * self.scale
        if view.show_grid and step > 10:
            ox, oy = self.offset[0] % step, self.offset[1] % step
            major = step * 5
            mx, my = self.offset[0] % major, self.offset[1] % major
            x = ox
            while x < w:
                c.create_line(x, 0, x, h, fill=pal["grid"])
                x += step
            y = oy
            while y < h:
                c.create_line(0, y, w, y, fill=pal["grid"])
                y += step
            x = mx
            while x < w:
                c.create_line(x, 0, x, h, fill=pal["grid_major"])
                x += major
            y = my
            while y < h:
                c.create_line(0, y, w, y, fill=pal["grid_major"])
                y += major

        sel_kind, sel_obj = self.selection if self.selection else (None, None)
        import math
        for l in self.topo.links.values():
            a, b = self.topo.nodes.get(l.src), self.topo.nodes.get(l.dst)
            if not a or not b:
                continue
            x1, y1 = self.w2s(a.x, a.y)
            x2, y2 = self.w2s(b.x, b.y)
            width = (1.0 + min(4.0, math.log10(max(l.capacity_mbps, 0.01) / 0.05 + 1))) * view.link_width_scale * self.gs
            sel = (sel_kind == "link" and sel_obj is l)
            c.create_line(x1, y1, x2, y2,
                          width=max(1.0, width * (2 if sel else 1)),
                          fill=pal["link_hi"] if sel else pal["link"], capstyle="round")

        for n in self.topo.nodes.values():
            x, y = self.w2s(n.x, n.y)
            r = view.node_radius * self.gs * self.scale
            base_color = (zone_color(n.zone_id, pal) if self.color_by_zone.get()
                          else self.topo.color_of(n))
            fill = node_fill(base_color, pal)
            sel = (sel_kind == "node" and sel_obj is n)
            outline = (pal["outline_compromised"] if n.is_compromised
                       else pal["outline_master"] if n.is_master else pal["outline"])
            if n.is_master:
                c.create_rectangle(x - r, y - r, x + r, y + r, fill=fill,
                                   outline=outline, width=4 if sel else 2.5)
            else:
                c.create_oval(x - r, y - r, x + r, y + r, fill=fill,
                              outline=outline, width=3 if sel else 1.4)
            if sel:
                c.create_oval(x - r - 5, y - r - 5, x + r + 5, y + r + 5,
                              outline=pal["sel_ring"], dash=(3, 2))
            if self._pending_link == n.node_id:
                c.create_oval(x - r - 9, y - r - 9, x + r + 9, y + r + 9,
                              outline=pal["pending"], width=2)
            if self.show_labels.get() and self.scale > 0.55:
                id_pt = max(6, int(view.id_pt * self.scale))
                label_pt = max(6, int(view.label_pt * self.scale))
                c.create_text(x, y, text=str(n.node_id), fill=pal["node_text"],
                              font=("TkDefaultFont", id_pt, "bold"))
                c.create_text(x, y + r + 11 * self.gs * self.scale,
                              text=f"{n.node_type}", fill=pal["node_label"],
                              font=("TkDefaultFont", label_pt))
        self._update_status()
        self._update_metrics()
        self._refresh_scenarios()

    def _update_status(self) -> None:
        m = self.topo.metrics()
        name = os.path.basename(self.path) if self.path else "без имени"
        self.status.config(
            text=f"{PROTOCOLS[self.topo.protocol]['label']} | {name}"
                 f"{'*' if self.dirty else ''} | узлов: {m['nodes']}, "
                 f"линков: {m['links']}, мастеров: {m['masters']}, "
                  f"компонент: {m['components']} | зум {self.scale:.2f} | "
                  f"UI ×{self.view.ui_scale:.2f} | {THEMES[self.view.theme]['label']}",
              style="Status.TLabel")

    def _update_metrics(self) -> None:
        m = self.topo.metrics()
        self.metrics_lbl.config(text="\n".join(f"{k}: {v}" for k, v in m.items()))

    # ── события мыши ───────────────────────────────────────────────
    def on_click(self, ev) -> None:
        mode = self.mode.get()
        node = self.node_at(ev.x, ev.y)
        if ev.state & 0x0001:                      # Shift → панорама
            self.on_pan_start(ev)
            return
        if mode == "node" and node is None:
            wx, wy = self.s2w(ev.x, ev.y)
            n = self.topo.add_node(self.new_type.get(), wx, wy)
            self.select(("node", n))
            self.mark_dirty()
            return
        if mode == "link":
            if node is None:
                self._reset_pending()
            elif self._pending_link is None:
                self._pending_link = node.node_id
            else:
                if self._pending_link != node.node_id:
                    l = self.topo.add_link(self._pending_link, node.node_id)
                    if l:
                        self.select(("link", l))
                        self.mark_dirty()
                self._pending_link = None
            self.redraw()
            return
        if mode == "delete":
            if node is not None:
                self.topo.remove_node(node.node_id)
            else:
                l = self.link_at(ev.x, ev.y)
                if l:
                    self.topo.remove_link(l.src, l.dst)
            self.selection = None
            self.mark_dirty()
            return
        if node is not None:
            self.select(("node", node))
            self._drag = {"node": node, "sx": ev.x, "sy": ev.y,
                          "x0": node.x, "y0": node.y}
        else:
            l = self.link_at(ev.x, ev.y)
            self.select(("link", l) if l else None)

    def on_drag(self, ev) -> None:
        if not self._drag:
            return
        d = self._drag
        d["node"].x = d["x0"] + (ev.x - d["sx"]) / self.scale
        d["node"].y = d["y0"] + (ev.y - d["sy"]) / self.scale
        self.redraw()

    def on_release(self, ev) -> None:
        if self._drag:
            self._drag = None
            self.mark_dirty()

    def on_double(self, ev) -> None:
        n = self.node_at(ev.x, ev.y)
        if n:
            self.toggle_master()

    def on_right_click(self, ev) -> None:
        n = self.node_at(ev.x, ev.y)
        l = None if n else self.link_at(ev.x, ev.y)
        if not n and not l:
            return
        self.select(("node", n) if n else ("link", l))
        menu = tk.Menu(self, tearoff=0)
        if n:
            menu.add_command(label="MASTER ↔ SLAVE", command=self.toggle_master)
            menu.add_command(label="Пометить/снять compromised",
                             command=self.toggle_compromised)
            menu.add_command(label="Соединить со всеми выбранного типа…",
                             command=self.connect_to_type)
            menu.add_separator()
            menu.add_command(label="Удалить узел", command=self.delete_selection)
        else:
            menu.add_command(label="Сделать магистральным (backbone)",
                             command=self.make_backbone)
            menu.add_command(label="Удалить линк", command=self.delete_selection)
        menu.tk_popup(ev.x_root, ev.y_root)

    def on_pan_start(self, ev) -> None:
        self._pan = (ev.x, ev.y, self.offset[0], self.offset[1])

    def on_pan_move(self, ev) -> None:
        if not hasattr(self, "_pan"):
            return
        sx, sy, ox, oy = self._pan
        self.offset = [ox + ev.x - sx, oy + ev.y - sy]
        self.redraw()

    def on_wheel(self, ev, delta: int | None = None) -> None:
        d = delta if delta is not None else ev.delta
        factor = 1.1 if d > 0 else 1 / 1.1
        wx, wy = self.s2w(ev.x, ev.y)
        self.scale = max(0.2, min(4.0, self.scale * factor))
        nx_, ny_ = self.w2s(wx, wy)
        self.offset[0] += ev.x - nx_
        self.offset[1] += ev.y - ny_
        self.redraw()

    # ── действия ───────────────────────────────────────────────────
    def _reset_pending(self) -> None:
        self._pending_link = None
        self.redraw()

    def select(self, sel) -> None:
        self.selection = sel
        self._show_properties()
        self.redraw()

    def mark_dirty(self) -> None:
        self.dirty = True
        self._show_properties()
        self.redraw()

    def toggle_master(self) -> None:
        if self.selection and self.selection[0] == "node":
            n: Node = self.selection[1]
            n.role = "SLAVE" if n.is_master else "MASTER"
            self.topo.assign_zones()
            self.mark_dirty()

    def toggle_compromised(self) -> None:
        if self.selection and self.selection[0] == "node":
            n: Node = self.selection[1]
            n.is_compromised = not n.is_compromised
            self.mark_dirty()

    def make_backbone(self) -> None:
        if self.selection and self.selection[0] == "link":
            l: Link = self.selection[1]
            d = self.topo.profile["backbone_link"]
            l.capacity_mbps = d["capacity_mbps"]
            l.propagation_delay_ms = d["propagation_delay_ms"]
            l.base_loss_prob = d["base_loss_prob"]
            self.mark_dirty()

    def connect_to_type(self) -> None:
        if not (self.selection and self.selection[0] == "node"):
            return
        node: Node = self.selection[1]
        t = simpledialog.askstring("Соединить", "Тип узлов-адресатов:",
                                   initialvalue=self.new_type.get(), parent=self)
        if not t:
            return
        for other in list(self.topo.nodes.values()):
            if other.node_type == t and other.node_id != node.node_id:
                self.topo.add_link(node.node_id, other.node_id)
        self.mark_dirty()

    def delete_selection(self) -> None:
        if not self.selection:
            return
        kind, obj = self.selection
        if kind == "node":
            self.topo.remove_node(obj.node_id)
        else:
            self.topo.remove_link(obj.src, obj.dst)
        self.selection = None
        self.mark_dirty()

    def do_layout(self, kind: str) -> None:
        self.topo.auto_layout(kind, width=float(self.view.layout_width),
                              height=float(self.view.layout_height),
                              seed=int(self.topo.sim_params.get("seed", 42)))
        self.fit_view()
        self.mark_dirty()

    def do_zones(self) -> None:
        self.topo.assign_zones()
        self.color_by_zone.set(True)
        self.mark_dirty()

    def do_validate(self) -> None:
        probs = self.topo.validate()
        if not probs:
            messagebox.showinfo("Проверка", "Топология корректна.")
        else:
            messagebox.showwarning("Проверка", "\n".join(probs[:40]))
        self.logln("── проверка топологии ──")
        for p in probs or ["ok"]:
            self.logln(p)

    def fit_view(self) -> None:
        if not self.topo.nodes:
            return
        xs = [n.x for n in self.topo.nodes.values()]
        ys = [n.y for n in self.topo.nodes.values()]
        w = (self.canvas.winfo_width() or 900) - 80
        h = (self.canvas.winfo_height() or 700) - 80
        dx = (max(xs) - min(xs)) or 1
        dy = (max(ys) - min(ys)) or 1
        self.scale = max(0.2, min(3.0, min(w / dx, h / dy)))
        self.offset = [40 - min(xs) * self.scale, 40 - min(ys) * self.scale]
        self.redraw()

    # ── панель свойств ─────────────────────────────────────────────
    def _show_properties(self) -> None:
        for w in self.prop_tab.winfo_children():
            w.destroy()
        if not self.selection:
            ttk.Label(self.prop_tab, wraplength=300, style="Hint.TLabel",
                      text="Ничего не выбрано.\n\nВыберите узел или линк на схеме, "
                           "либо добавьте новый в режиме «Узел» / «Линк».").pack()
            return
        kind, obj = self.selection
        self._fields: dict[str, tk.StringVar] = {}

        def row(i: int, label: str, key: str, value) -> None:
            ttk.Label(self.prop_tab, text=label).grid(row=i, column=0, sticky="w", pady=2)
            var = tk.StringVar(value=str(value))
            self._fields[key] = var
            ttk.Entry(self.prop_tab, textvariable=var, width=20).grid(
                row=i, column=1, sticky="e")

        if kind == "node":
            n: Node = obj
            ttk.Label(self.prop_tab, text=f"Узел #{n.node_id}",
                      font=("TkDefaultFont", 11, "bold")).grid(
                row=0, column=0, columnspan=2, sticky="w", pady=(0, 6))
            ttk.Label(self.prop_tab, text="node_type").grid(row=1, column=0, sticky="w")
            tvar = tk.StringVar(value=n.node_type)
            self._fields["node_type"] = tvar
            ttk.Combobox(self.prop_tab, textvariable=tvar, state="readonly", width=18,
                         values=self.topo.node_type_names()).grid(row=1, column=1, sticky="e")
            ttk.Label(self.prop_tab, text="role").grid(row=2, column=0, sticky="w")
            rvar = tk.StringVar(value=n.role)
            self._fields["role"] = rvar
            ttk.Combobox(self.prop_tab, textvariable=rvar, state="readonly", width=18,
                         values=["MASTER", "SLAVE"]).grid(row=2, column=1, sticky="e")
            row(3, "hostname", "hostname", n.hostname)
            row(4, "realm", "realm", n.realm)
            row(5, "base_rate (msg/interval)", "base_rate", n.base_rate)
            row(6, "zone_id", "zone_id", n.zone_id)
            row(7, "master_id", "master_id",
                "" if n.master_id is None else n.master_id)
            cvar = tk.BooleanVar(value=n.is_compromised)
            ttk.Checkbutton(self.prop_tab, text="is_compromised",
                            variable=cvar).grid(row=8, column=0, columnspan=2, sticky="w")
            self._compromised_var = cvar
            ttk.Label(self.prop_tab, text="interface_dist (JSON)").grid(
                row=9, column=0, columnspan=2, sticky="w", pady=(6, 0))
            txt = tk.Text(self.prop_tab, height=7, width=34, font="TkFixedFont")
            txt.insert("1.0", json.dumps(n.interface_dist, indent=1))
            txt.grid(row=10, column=0, columnspan=2, sticky="ew")
            self._iface_text = txt
            self.style_text(txt)
            deg = len(self.topo.neighbors(n.node_id))
            ttk.Label(self.prop_tab, text=f"степень узла: {deg}",
                      style="Hint.TLabel").grid(row=11, column=0, columnspan=2, sticky="w")
            ttk.Button(self.prop_tab, text="Применить",
                       command=self._apply_node).grid(row=12, column=0, columnspan=2,
                                                      sticky="ew", pady=8)
        else:
            l: Link = obj
            ttk.Label(self.prop_tab, text=f"Линк {l.src} ↔ {l.dst}",
                      font=("TkDefaultFont", 11, "bold")).grid(
                row=0, column=0, columnspan=2, sticky="w", pady=(0, 6))
            row(1, "capacity_mbps", "capacity_mbps", l.capacity_mbps)
            row(2, "propagation_delay_ms", "propagation_delay_ms", l.propagation_delay_ms)
            row(3, "base_loss_prob", "base_loss_prob", l.base_loss_prob)
            ttk.Button(self.prop_tab, text="Применить",
                       command=self._apply_link).grid(row=4, column=0, columnspan=2,
                                                      sticky="ew", pady=8)

    def _apply_node(self) -> None:
        n: Node = self.selection[1]
        try:
            n.node_type = self._fields["node_type"].get()
            n.role = self._fields["role"].get()
            n.hostname = self._fields["hostname"].get()
            n.realm = self._fields["realm"].get()
            n.base_rate = float(self._fields["base_rate"].get())
            n.zone_id = int(self._fields["zone_id"].get())
            mid = self._fields["master_id"].get().strip()
            n.master_id = int(mid) if mid else None
            n.is_compromised = bool(self._compromised_var.get())
            n.interface_dist = json.loads(self._iface_text.get("1.0", "end") or "{}")
        except (ValueError, json.JSONDecodeError) as exc:
            messagebox.showerror("Некорректное значение", str(exc))
            return
        self.mark_dirty()

    def _apply_link(self) -> None:
        l: Link = self.selection[1]
        try:
            l.capacity_mbps = float(self._fields["capacity_mbps"].get())
            l.propagation_delay_ms = float(self._fields["propagation_delay_ms"].get())
            l.base_loss_prob = float(self._fields["base_loss_prob"].get())
        except ValueError as exc:
            messagebox.showerror("Некорректное значение", str(exc))
            return
        self.mark_dirty()

    def apply_sim_params(self) -> None:
        for k, var in self.sim_vars.items():
            raw = var.get()
            try:
                self.topo.sim_params[k] = int(raw) if raw.isdigit() else float(raw)
            except ValueError:
                self.topo.sim_params[k] = raw
        self.mark_dirty()

    # ── сценарии ───────────────────────────────────────────────────
    def _refresh_scenarios(self) -> None:
        if not hasattr(self, "scen_tree"):
            return
        self.scen_tree.delete(*self.scen_tree.get_children())
        for i, s in enumerate(self.topo.scenarios):
            self.scen_tree.insert("", "end", iid=str(i),
                                  values=(s.kind, s.name,
                                          ",".join(map(str, s.target_nodes)),
                                          s.start_interval, s.end_interval,
                                          s.intensity))

    def add_scenario(self) -> None:
        prof = self.topo.profile
        dlg = tk.Toplevel(self)
        dlg.title("Новый сценарий")
        dlg.transient(self.master)
        dlg.grab_set()
        dlg.configure(bg=self.pal["panel_bg"])
        kind = tk.StringVar(value="attack")
        name = tk.StringVar(value=prof["attacks"][0])
        targets = tk.StringVar(
            value=str(self.selection[1].node_id)
            if self.selection and self.selection[0] == "node" else "")
        start = tk.StringVar(value="100")
        end = tk.StringVar(value="200")
        inten = tk.StringVar(value="1.0")

        name_box = ttk.Combobox(dlg, textvariable=name, state="readonly", width=26,
                                values=prof["attacks"])

        def on_kind(*_):
            vals = prof["attacks"] if kind.get() == "attack" else prof["events"]
            name_box.config(values=vals)
            name.set(vals[0])

        rows = [("Тип", ttk.Combobox(dlg, textvariable=kind, state="readonly", width=26,
                                     values=["attack", "normal"])),
                ("Сценарий", name_box),
                ("Целевые узлы (через запятую)", ttk.Entry(dlg, textvariable=targets, width=28)),
                ("start_interval", ttk.Entry(dlg, textvariable=start, width=28)),
                ("end_interval", ttk.Entry(dlg, textvariable=end, width=28)),
                ("intensity", ttk.Entry(dlg, textvariable=inten, width=28))]
        for i, (lbl, widget) in enumerate(rows):
            ttk.Label(dlg, text=lbl).grid(row=i, column=0, sticky="w", padx=8, pady=3)
            widget.grid(row=i, column=1, padx=8, pady=3)
        kind.trace_add("write", on_kind)

        def ok() -> None:
            try:
                tl = [int(t) for t in targets.get().replace(" ", "").split(",") if t]
                sc = Scenario(kind=kind.get(), name=name.get(), target_nodes=tl,
                              start_interval=int(start.get()),
                              end_interval=int(end.get()),
                              intensity=float(inten.get()))
            except ValueError as exc:
                messagebox.showerror("Ошибка", str(exc), parent=dlg)
                return
            self.topo.scenarios.append(sc)
            dlg.destroy()
            self.mark_dirty()

        ttk.Button(dlg, text="Добавить", command=ok).grid(
            row=len(rows), column=0, columnspan=2, sticky="ew", padx=8, pady=8)

    def del_scenario(self) -> None:
        for iid in self.scen_tree.selection():
            idx = int(iid)
            if 0 <= idx < len(self.topo.scenarios):
                self.topo.scenarios.pop(idx)
        self.mark_dirty()

    # ── файлы ──────────────────────────────────────────────────────
    def new_network(self) -> None:
        proto = simpledialog.askstring(
            "Новая сеть", "Протокол " + " / ".join(PROTOCOLS) + ":",
            initialvalue=self.topo.protocol, parent=self)
        if not proto or proto not in PROTOCOLS:
            return
        self.topo = Topology(proto)
        self._after_load(None)

    def load_template(self, proto: str) -> None:
        self.topo = template(proto, seed=42)
        self._after_load(None)

    def open_file(self) -> None:
        p = filedialog.askopenfilename(filetypes=[("Topology JSON", "*.json")])
        if not p:
            return
        try:
            self.topo = Topology.load(p)
        except Exception as exc:
            messagebox.showerror("Не удалось открыть", str(exc))
            return
        self._after_load(p)

    def _after_load(self, path: str | None) -> None:
        self.path = path
        self.dirty = False
        self.selection = None
        self.type_box.config(values=self.topo.node_type_names())
        self.new_type.set(self.topo.node_type_names()[0])
        for k, var in list(self.sim_vars.items()):
            if k in self.topo.sim_params:
                var.set(str(self.topo.sim_params[k]))
        self.fit_view()
        self._show_properties()

    def save_file(self, as_new: bool = False) -> None:
        p = self.path
        if as_new or not p:
            p = filedialog.asksaveasfilename(defaultextension=".json",
                                             filetypes=[("Topology JSON", "*.json")])
        if not p:
            return
        self.topo.save(p)
        self.path, self.dirty = p, False
        self.logln(f"сохранено: {p}")
        self.redraw()

    def export_graphml(self) -> None:
        p = filedialog.asksaveasfilename(defaultextension=".graphml")
        if not p:
            return
        try:
            import networkx as nx
            nx.write_graphml(self.topo.to_networkx(), p)
            self.logln(f"GraphML: {p}")
        except Exception as exc:
            messagebox.showerror("Ошибка экспорта", str(exc))

    def export_svg(self) -> None:
        p = filedialog.asksaveasfilename(
            defaultextension=".svg",
            filetypes=[("SVG", "*.svg")],
        )
        if not p:
            return
        if not self.topo.nodes:
            messagebox.showinfo("Экспорт SVG", "Схема пуста.")
            return

        pad = 24 * self.gs
        positions = {
            node.node_id: self.w2s(node.x, node.y)
            for node in self.topo.nodes.values()
        }
        radius = self.view.node_radius * self.gs * self.scale
        label_gap = 11 * self.gs * self.scale
        xs = [position[0] for position in positions.values()]
        ys = [position[1] for position in positions.values()]
        x0 = min(xs) - radius - pad
        y0 = min(ys) - radius - pad
        x1 = max(xs) + radius + pad
        y1 = max(ys) + radius + label_gap + pad
        width = max(1.0, x1 - x0)
        height = max(1.0, y1 - y0)

        def esc(value: object) -> str:
            from html import escape
            return escape(str(value), quote=True)

        def point(value: float) -> str:
            return f"{value:.2f}"

        svg: list[str] = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            '<svg xmlns="http://www.w3.org/2000/svg" version="1.1" '
            f'width="{point(width)}" height="{point(height)}" '
            f'viewBox="{point(x0)} {point(y0)} {point(width)} {point(height)}">',
            f'<rect x="{point(x0)}" y="{point(y0)}" width="{point(width)}" '
            f'height="{point(height)}" fill="{esc(self.pal["canvas_bg"])}"/>',
        ]

        if self.view.show_grid:
            step = max(6.0, self.view.grid_step) * self.gs * self.scale
            if step > 10:
                start_x = x0 - (x0 % step)
                start_y = y0 - (y0 % step)
                x = start_x
                while x <= x1:
                    svg.append(
                        f'<path d="M {point(x)} {point(y0)} V {point(y1)}" '
                        f'stroke="{esc(self.pal["grid"])}" stroke-width="1"/>'
                    )
                    x += step
                y = start_y
                while y <= y1:
                    svg.append(
                        f'<path d="M {point(x0)} {point(y)} H {point(x1)}" '
                        f'stroke="{esc(self.pal["grid"])}" stroke-width="1"/>'
                    )
                    y += step

        sel_kind, sel_obj = self.selection if self.selection else (None, None)
        import math
        for link in self.topo.links.values():
            if link.src not in positions or link.dst not in positions:
                continue
            sx, sy = positions[link.src]
            tx, ty = positions[link.dst]
            base = 1.0 + min(4.0, math.log10(max(link.capacity_mbps, 0.01) / 0.05 + 1))
            selected = sel_kind == "link" and sel_obj is link
            stroke_width = base * self.view.link_width_scale * self.gs * (2 if selected else 1)
            color = self.pal["link_hi"] if selected else self.pal["link"]
            svg.append(
                f'<line x1="{point(sx)}" y1="{point(sy)}" x2="{point(tx)}" y2="{point(ty)}" '
                f'stroke="{esc(color)}" stroke-width="{point(stroke_width)}" '
                'stroke-linecap="round"/>'
            )

        for node in self.topo.nodes.values():
            x, y = positions[node.node_id]
            fill = node_fill(
                zone_color(node.zone_id, self.pal) if self.color_by_zone.get()
                else self.topo.color_of(node), self.pal
            )
            selected = sel_kind == "node" and sel_obj is node
            outline = (self.pal["outline_compromised"] if node.is_compromised
                       else self.pal["outline_master"] if node.is_master
                       else self.pal["outline"])
            shape = "rect" if node.is_master else "circle"
            if shape == "rect":
                svg.append(
                    f'<rect x="{point(x - radius)}" y="{point(y - radius)}" '
                    f'width="{point(radius * 2)}" height="{point(radius * 2)}" '
                    f'fill="{esc(fill)}" stroke="{esc(outline)}" '
                    f'stroke-width="{point(4 if selected else 2.5)}"/>'
                )
            else:
                svg.append(
                    f'<circle cx="{point(x)}" cy="{point(y)}" r="{point(radius)}" '
                    f'fill="{esc(fill)}" stroke="{esc(outline)}" '
                    f'stroke-width="{point(3 if selected else 1.4)}"/>'
                )
            if selected:
                svg.append(
                    f'<circle cx="{point(x)}" cy="{point(y)}" r="{point(radius + 5)}" '
                    f'fill="none" stroke="{esc(self.pal["sel_ring"])}" '
                    'stroke-width="2" stroke-dasharray="3 2"/>'
                )
            if self._pending_link == node.node_id:
                svg.append(
                    f'<circle cx="{point(x)}" cy="{point(y)}" r="{point(radius + 9)}" '
                    f'fill="none" stroke="{esc(self.pal["pending"])}" stroke-width="2"/>'
                )
            if self.show_labels.get() and self.scale > 0.55:
                id_size = max(6, int(self.view.id_pt * self.scale))
                label_size = max(6, int(self.view.label_pt * self.scale))
                svg.append(
                    f'<text x="{point(x)}" y="{point(y + id_size * 0.35)}" '
                    f'fill="{esc(self.pal["node_text"])}" font-family="sans-serif" '
                    f'font-size="{id_size}" font-weight="bold" text-anchor="middle">'
                    f'{esc(node.node_id)}</text>'
                )
                svg.append(
                    f'<text x="{point(x)}" y="{point(y + radius + label_gap + label_size * 0.35)}" '
                    f'fill="{esc(self.pal["node_label"])}" font-family="sans-serif" '
                    f'font-size="{label_size}" text-anchor="middle">{esc(node.node_type)}</text>'
                )
        svg.append("</svg>")
        try:
            with open(p, "w", encoding="utf-8") as stream:
                stream.write("\n".join(svg))
        except OSError as exc:
            messagebox.showerror("Ошибка экспорта SVG", str(exc))
            return
        self.logln(f"SVG: {p} ({int(width)}×{int(height)} px, вектор)")

    def export_image(self) -> None:
        p = filedialog.asksaveasfilename(defaultextension=".png",
                                         filetypes=[("PNG", "*.png"),
                                                    ("PostScript", "*.ps")])
        if not p:
            return
        bbox = self.canvas.bbox("all")
        if not bbox:
            messagebox.showinfo("Экспорт", "Схема пуста.")
            return
        scale = max(1.0, float(self.view.export_scale))
        pad = 24
        x0, y0 = bbox[0] - pad, bbox[1] - pad
        width = max(1, bbox[2] - bbox[0] + 2 * pad)
        height = max(1, bbox[3] - bbox[1] + 2 * pad)
        background = self.canvas.create_rectangle(x0, y0, x0 + width, y0 + height,
                                                   fill=self.pal["canvas_bg"], outline="")
        self.canvas.tag_lower(background)
        try:
            ps = self.canvas.postscript(colormode="color", x=x0, y=y0,
                                        width=width, height=height,
                                        pagewidth=f"{int(width * scale)}p",
                                        pageheight=f"{int(height * scale)}p")
        finally:
            self.canvas.delete(background)
        if p.lower().endswith(".ps"):
            with open(p, "w", encoding="latin-1") as f:
                f.write(ps)
            self.logln(f"PS: {p} ({int(width * scale)}×{int(height * scale)} px)")
            return
        tmp = None
        try:
            from PIL import Image
            with tempfile.NamedTemporaryFile("w", suffix=".eps", delete=False,
                                             encoding="latin-1") as stream:
                stream.write(ps)
                tmp = stream.name
            image = Image.open(tmp)
            image.load()
            image.convert("RGB").save(p, dpi=(int(72 * scale), int(72 * scale)))
            self.logln(f"PNG: {p} ({image.width}×{image.height} px, ×{scale:g})")
        except Exception as exc:
            alt = os.path.splitext(p)[0] + ".ps"
            with open(alt, "w", encoding="latin-1") as f:
                f.write(ps)
            messagebox.showinfo("Экспорт", f"Pillow/ghostscript недоступны ({exc}).\n"
                                           f"Векторный вариант сохранён: {alt}")
        finally:
            if tmp and os.path.exists(tmp):
                os.unlink(tmp)

    # ── запуск симуляции ───────────────────────────────────────────
    def show_summary(self) -> None:
        m = self.topo.metrics()
        by_type: dict[str, int] = {}
        for n in self.topo.nodes.values():
            by_type[n.node_type] = by_type.get(n.node_type, 0) + 1
        lines = [f"{k}: {v}" for k, v in m.items()]
        lines.append("состав: " + ", ".join(f"{k}×{v}" for k, v in sorted(by_type.items())))
        lines.append("сценариев: %d" % len(self.topo.scenarios))
        messagebox.showinfo("Сводка", "\n".join(lines))

    def run_simulation(self) -> None:
        probs = [p for p in self.topo.validate() if p.startswith("ERROR")]
        if probs:
            messagebox.showerror("Топология некорректна", "\n".join(probs[:20]))
            return
        sim_dir = filedialog.askdirectory(
            title="Каталог с модулем симулятора (например .../diameter)")
        if not sim_dir:
            return
        from .adapters import run_with_topology
        self.logln(f"── запуск симулятора из {sim_dir} ──")

        def worker() -> None:
            try:
                out = run_with_topology(self.topo, sim_dir, log=self.logln)
                self.logln(f"готово: {out}")
            except Exception as exc:  # noqa: BLE001 — показываем пользователю
                self.logln(f"ОШИБКА: {type(exc).__name__}: {exc}")

        threading.Thread(target=worker, daemon=True).start()

    # ── журнал ─────────────────────────────────────────────────────
    def logln(self, msg: str) -> None:
        self._log_q.put(str(msg))

    def _drain_log(self) -> None:
        try:
            while True:
                line = self._log_q.get_nowait()
                self.log.insert("end", line + "\n")
                self.log.see("end")
        except queue.Empty:
            pass
        self.after(200, self._drain_log)


def main() -> None:
    hidpi.enable_dpi_awareness()
    root = tk.Tk()
    root.title("Signal Network Editor — signal_network_sim")
    editor = TopologyEditor(root)

    def on_close() -> None:
        if editor.dirty and not messagebox.askokcancel(
                "Выход", "Есть несохранённые изменения топологии. Выйти?"):
            return
        try:
            editor.view.window_geometry = root.geometry().split("+")[0]
            editor.view.save()
        except Exception:
            pass
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    editor.logln(f"DPI-awareness: {hidpi.process_awareness()} / "
                 f"DPI={hidpi.get_dpi(root):.0f}, "
                 f"экран {root.winfo_screenwidth()}×{root.winfo_screenheight()}")
    root.mainloop()


if __name__ == "__main__":
    main()
