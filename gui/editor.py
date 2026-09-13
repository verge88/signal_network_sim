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

import io
import json
import os
import queue
import sys
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog

from .model import PROTOCOLS, Topology, Node, Link, Scenario, link_key, template

NODE_R = 22.0
BG = "#f7f8fa"
GRID = "#e6e9ef"


class TopologyEditor(tk.Frame):
    def __init__(self, master: tk.Tk):
        super().__init__(master)
        self.pack(fill="both", expand=True)
        self.topo = template("diameter")
        self.path: str | None = None
        self.dirty = False

        self.scale = 1.0
        self.offset = [0.0, 0.0]
        self.mode = tk.StringVar(value="select")
        self.new_type = tk.StringVar(value=self.topo.node_type_names()[0])
        self.color_by_zone = tk.BooleanVar(value=False)
        self.show_labels = tk.BooleanVar(value=True)
        self.selection: tuple[str, object] | None = None
        self._pending_link: int | None = None
        self._drag: dict | None = None
        self._log_q: "queue.Queue[str]" = queue.Queue()

        self._build_menu()
        self._build_toolbar()
        self._build_body()
        self._bind_events()
        self.redraw()
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

        self.master.config(menu=m)
        self.master.bind("<Control-n>", lambda e: self.new_network())
        self.master.bind("<Control-o>", lambda e: self.open_file())
        self.master.bind("<Control-s>", lambda e: self.save_file())

    def _build_toolbar(self) -> None:
        bar = ttk.Frame(self, padding=(6, 4))
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
        self.canvas = tk.Canvas(left, bg=BG, highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.status = ttk.Label(left, text="", anchor="w", padding=(6, 3))
        self.status.pack(fill="x")

        right = ttk.Frame(body, width=340)
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
        self.log = tk.Text(self.log_tab, height=10, wrap="word", font=("TkFixedFont", 9))
        self.log.pack(fill="both", expand=True)
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
        ttk.Label(self.scen_tab, wraplength=310, foreground="#555",
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
        self.metrics_lbl = ttk.Label(self.sim_tab, justify="left", foreground="#333")
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

    # ── координаты ─────────────────────────────────────────────────
    def w2s(self, x: float, y: float) -> tuple[float, float]:
        return x * self.scale + self.offset[0], y * self.scale + self.offset[1]

    def s2w(self, x: float, y: float) -> tuple[float, float]:
        return (x - self.offset[0]) / self.scale, (y - self.offset[1]) / self.scale

    def node_at(self, sx: float, sy: float) -> Node | None:
        for n in self.topo.nodes.values():
            nx_, ny_ = self.w2s(n.x, n.y)
            if (nx_ - sx) ** 2 + (ny_ - sy) ** 2 <= (NODE_R * self.scale) ** 2:
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
        c.delete("all")
        w = c.winfo_width() or 900
        h = c.winfo_height() or 700
        step = 40 * self.scale
        if step > 12:
            ox, oy = self.offset[0] % step, self.offset[1] % step
            x = ox
            while x < w:
                c.create_line(x, 0, x, h, fill=GRID)
                x += step
            y = oy
            while y < h:
                c.create_line(0, y, w, y, fill=GRID)
                y += step

        sel_kind, sel_obj = self.selection if self.selection else (None, None)
        import math
        for l in self.topo.links.values():
            a, b = self.topo.nodes.get(l.src), self.topo.nodes.get(l.dst)
            if not a or not b:
                continue
            x1, y1 = self.w2s(a.x, a.y)
            x2, y2 = self.w2s(b.x, b.y)
            width = 1.0 + min(4.0, math.log10(max(l.capacity_mbps, 0.01) / 0.05 + 1))
            sel = (sel_kind == "link" and sel_obj is l)
            c.create_line(x1, y1, x2, y2, width=width * (2 if sel else 1),
                          fill="#d04a4a" if sel else "#8d97a8")

        for n in self.topo.nodes.values():
            x, y = self.w2s(n.x, n.y)
            r = NODE_R * self.scale
            fill = (self._zone_color(n.zone_id) if self.color_by_zone.get()
                    else self.topo.color_of(n))
            sel = (sel_kind == "node" and sel_obj is n)
            outline = "#d04a4a" if n.is_compromised else ("#1b1b1b" if n.is_master
                                                          else "#5a6273")
            if n.is_master:
                c.create_rectangle(x - r, y - r, x + r, y + r, fill=fill,
                                   outline=outline, width=4 if sel else 2.5)
            else:
                c.create_oval(x - r, y - r, x + r, y + r, fill=fill,
                              outline=outline, width=3 if sel else 1.4)
            if sel:
                c.create_oval(x - r - 5, y - r - 5, x + r + 5, y + r + 5,
                              outline="#d04a4a", dash=(3, 2))
            if self._pending_link == n.node_id:
                c.create_oval(x - r - 9, y - r - 9, x + r + 9, y + r + 9,
                              outline="#2f8f3f", width=2)
            if self.show_labels.get() and self.scale > 0.55:
                c.create_text(x, y, text=str(n.node_id),
                              fill="white", font=("TkDefaultFont", int(9 * self.scale)))
                c.create_text(x, y + r + 11 * self.scale,
                              text=f"{n.node_type}", fill="#333",
                              font=("TkDefaultFont", int(8 * self.scale)))
        self._update_status()
        self._update_metrics()
        self._refresh_scenarios()

    @staticmethod
    def _zone_color(zone: int) -> str:
        palette = ["#4a7fb5", "#b5744a", "#5aa06a", "#8a5ab5", "#b5525a",
                   "#4aa8a8", "#9a9a4a", "#7a7a8a"]
        return palette[zone % len(palette)]

    def _update_status(self) -> None:
        m = self.topo.metrics()
        name = os.path.basename(self.path) if self.path else "без имени"
        self.status.config(
            text=f"{PROTOCOLS[self.topo.protocol]['label']} | {name}"
                 f"{'*' if self.dirty else ''} | узлов: {m['nodes']}, "
                 f"линков: {m['links']}, мастеров: {m['masters']}, "
                 f"компонент: {m['components']} | зум {self.scale:.2f}")

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
        w = self.canvas.winfo_width() or 1000
        h = self.canvas.winfo_height() or 700
        self.topo.auto_layout(kind, width=w / self.scale, height=h / self.scale,
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
            ttk.Label(self.prop_tab, wraplength=300, foreground="#555",
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
            txt = tk.Text(self.prop_tab, height=7, width=36, font=("TkFixedFont", 9))
            txt.insert("1.0", json.dumps(n.interface_dist, indent=1))
            txt.grid(row=10, column=0, columnspan=2, sticky="ew")
            self._iface_text = txt
            deg = len(self.topo.neighbors(n.node_id))
            ttk.Label(self.prop_tab, text=f"степень узла: {deg}",
                      foreground="#555").grid(row=11, column=0, columnspan=2, sticky="w")
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

    def export_image(self) -> None:
        p = filedialog.asksaveasfilename(defaultextension=".png",
                                         filetypes=[("PNG", "*.png"),
                                                    ("PostScript", "*.ps")])
        if not p:
            return
        ps = self.canvas.postscript(colormode="color")
        if p.lower().endswith(".ps"):
            with open(p, "w", encoding="latin-1") as f:
                f.write(ps)
            self.logln(f"PS: {p}")
            return
        try:
            from PIL import Image  # опционально
            Image.open(io.BytesIO(ps.encode("latin-1"))).save(p)
            self.logln(f"PNG: {p}")
        except Exception:
            alt = os.path.splitext(p)[0] + ".ps"
            with open(alt, "w", encoding="latin-1") as f:
                f.write(ps)
            messagebox.showinfo("Экспорт", f"Pillow/ghostscript недоступны, "
                                           f"сохранено как {alt}")

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
    root = tk.Tk()
    root.title("Signal Network Editor — signal_network_sim")
    root.geometry("1400x860")
    try:
        ttk.Style().theme_use("clam")
    except tk.TclError:
        pass
    TopologyEditor(root)
    root.mainloop()


if __name__ == "__main__":
    main()
