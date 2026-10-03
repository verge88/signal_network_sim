from __future__ import annotations

import json
import os
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import pandas as pd

from .charts import AGGREGATIONS, CHART_TYPES, ERROR_TYPES, build_figure, suggested_spec
from .export import export_result_package, export_workbook, save_figure, save_table
from .loader import load_experiment
from .model import ChartSpec, ExperimentResult
from .tables import aggregate_table, categorical_columns, descriptive_table, display_value, numeric_columns

try:
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
    _MPL_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover
    FigureCanvasTkAgg = None
    NavigationToolbar2Tk = None
    _MPL_IMPORT_ERROR = exc


class ExperimentResultsWindow(tk.Toplevel):
    def __init__(self, master, initial_root: str | os.PathLike | None = None):
        super().__init__(master)
        self.title("Результаты экспериментов")
        self.geometry("1280x820")
        self.minsize(1000, 680)

        self.result: ExperimentResult | None = None
        self.current_figure = None
        self.chart_canvas = None
        self.chart_toolbar = None
        self._table_sort_state: dict[tuple[int, str], bool] = {}
        self._last_stats_df = pd.DataFrame()

        self.path_var = tk.StringVar()
        self.status_var = tk.StringVar(value="Откройте каталог эксперимента.")

        self.chart_table_var = tk.StringVar()
        self.chart_type_var = tk.StringVar(value="Столбчатая")
        self.x_var = tk.StringVar()
        self.y_var = tk.StringVar()
        self.group_var = tk.StringVar()
        self.agg_var = tk.StringVar(value="Среднее")
        self.error_var = tk.StringVar(value="95% CI")
        self.title_var = tk.StringVar()
        self.x_label_var = tk.StringVar()
        self.y_label_var = tk.StringVar()
        self.filter_col_var = tk.StringVar()
        self.filter_value_var = tk.StringVar()
        self.table_var = tk.StringVar()
        self.stats_table_var = tk.StringVar()
        self.json_var = tk.StringVar()
        self.table_info_var = tk.StringVar()

        self._build_ui()
        if initial_root:
            p = Path(initial_root)
            if p.exists():
                self.path_var.set(str(p))

    def _build_ui(self):
        top = ttk.Frame(self, padding=8); top.pack(fill="x")
        ttk.Label(top, text="Каталог:").pack(side="left")
        ttk.Entry(top, textvariable=self.path_var).pack(side="left", fill="x", expand=True, padx=(6, 6))
        ttk.Button(top, text="Открыть…", command=self.open_directory).pack(side="left")
        ttk.Button(top, text="Загрузить", command=self.load_path).pack(side="left", padx=(6, 0))
        ttk.Button(top, text="Экспорт пакета…", command=self.export_package).pack(side="left", padx=(12, 0))

        self.nb = ttk.Notebook(self); self.nb.pack(fill="both", expand=True, padx=8, pady=(0, 8))
        self.overview_tab = ttk.Frame(self.nb, padding=10)
        self.chart_tab = ttk.Frame(self.nb, padding=8)
        self.table_tab = ttk.Frame(self.nb, padding=8)
        self.stats_tab = ttk.Frame(self.nb, padding=8)
        self.manifest_tab = ttk.Frame(self.nb, padding=8)
        self.nb.add(self.overview_tab, text="Обзор")
        self.nb.add(self.chart_tab, text="Графики")
        self.nb.add(self.table_tab, text="Таблицы")
        self.nb.add(self.stats_tab, text="Статистика")
        self.nb.add(self.manifest_tab, text="Manifest / JSON")
        self._build_overview(); self._build_chart_tab(); self._build_table_tab(); self._build_stats_tab(); self._build_manifest_tab()
        ttk.Label(self, textvariable=self.status_var, anchor="w", padding=(8, 4)).pack(fill="x", side="bottom")

    def _build_overview(self):
        self.summary_tree = ttk.Treeview(self.overview_tab, columns=("key", "value"), show="headings", height=13)
        self.summary_tree.heading("key", text="Параметр"); self.summary_tree.heading("value", text="Значение")
        self.summary_tree.column("key", width=220, anchor="w"); self.summary_tree.column("value", width=850, anchor="w")
        self.summary_tree.pack(fill="both", expand=True)
        ttk.Label(self.overview_tab,
                  text="Viewer читает сохранённые артефакты и не запускает симуляцию повторно. Это позволяет воспроизводимо строить рисунки и таблицы из конкретного эксперимента.",
                  wraplength=1050).pack(fill="x", pady=(8, 0))

    def _build_chart_tab(self):
        controls = ttk.LabelFrame(self.chart_tab, text="Конструктор графика", padding=8)
        controls.pack(side="left", fill="y", padx=(0, 8)); controls.columnconfigure(1, weight=1)
        def row(label, widget, r):
            ttk.Label(controls, text=label).grid(row=r, column=0, sticky="w", pady=3)
            widget.grid(row=r, column=1, sticky="ew", pady=3)

        self.chart_table_box = ttk.Combobox(controls, textvariable=self.chart_table_var, state="readonly", width=31)
        self.chart_type_box = ttk.Combobox(controls, textvariable=self.chart_type_var, values=list(CHART_TYPES), state="readonly", width=20)
        self.x_box = ttk.Combobox(controls, textvariable=self.x_var, state="readonly")
        self.y_box = ttk.Combobox(controls, textvariable=self.y_var, state="readonly")
        self.group_box = ttk.Combobox(controls, textvariable=self.group_var, state="readonly")
        self.agg_box = ttk.Combobox(controls, textvariable=self.agg_var, values=list(AGGREGATIONS), state="readonly")
        self.error_box = ttk.Combobox(controls, textvariable=self.error_var, values=list(ERROR_TYPES), state="readonly")
        self.filter_col_box = ttk.Combobox(controls, textvariable=self.filter_col_var, state="readonly")
        self.filter_value_box = ttk.Combobox(controls, textvariable=self.filter_value_var, state="readonly")
        for r, (label, widget) in enumerate([
            ("Таблица:", self.chart_table_box), ("Тип:", self.chart_type_box), ("X:", self.x_box), ("Y:", self.y_box),
            ("Группа:", self.group_box), ("Агрегация:", self.agg_box), ("Ошибки:", self.error_box),
            ("Фильтр, столбец:", self.filter_col_box), ("Фильтр, значение:", self.filter_value_box),
            ("Заголовок:", ttk.Entry(controls, textvariable=self.title_var)),
            ("Подпись X:", ttk.Entry(controls, textvariable=self.x_label_var)),
            ("Подпись Y:", ttk.Entry(controls, textvariable=self.y_label_var)),
        ]): row(label, widget, r)
        buttons = ttk.Frame(controls); buttons.grid(row=12, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        ttk.Button(buttons, text="Построить", command=self.build_chart).pack(side="left", fill="x", expand=True)
        ttk.Button(buttons, text="Авто", command=self.auto_chart_spec).pack(side="left", padx=(6, 0))
        ttk.Button(controls, text="PNG/SVG/PDF…", command=self.save_chart).grid(row=13, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        self.chart_table_box.bind("<<ComboboxSelected>>", self._on_chart_table)
        self.filter_col_box.bind("<<ComboboxSelected>>", self._on_filter_column)
        self.chart_host = ttk.Frame(self.chart_tab); self.chart_host.pack(side="left", fill="both", expand=True)
        if _MPL_IMPORT_ERROR is not None:
            ttk.Label(self.chart_host, text=f"Matplotlib недоступен.\nУстановите: pip install matplotlib\n\n{_MPL_IMPORT_ERROR}", justify="center").pack(expand=True)

    def _make_tree_host(self, parent):
        host = ttk.Frame(parent); host.pack(fill="both", expand=True, pady=(8, 0))
        tree = ttk.Treeview(host, show="headings")
        vs = ttk.Scrollbar(host, orient="vertical", command=tree.yview)
        hs = ttk.Scrollbar(host, orient="horizontal", command=tree.xview)
        tree.configure(yscrollcommand=vs.set, xscrollcommand=hs.set)
        tree.grid(row=0, column=0, sticky="nsew"); vs.grid(row=0, column=1, sticky="ns"); hs.grid(row=1, column=0, sticky="ew")
        host.rowconfigure(0, weight=1); host.columnconfigure(0, weight=1)
        return tree

    def _build_table_tab(self):
        top = ttk.Frame(self.table_tab); top.pack(fill="x")
        ttk.Label(top, text="Таблица:").pack(side="left")
        self.table_box = ttk.Combobox(top, textvariable=self.table_var, state="readonly", width=50); self.table_box.pack(side="left", padx=6)
        self.table_box.bind("<<ComboboxSelected>>", lambda e: self.show_table())
        ttk.Button(top, text="CSV/XLSX/TEX/MD…", command=self.save_current_table).pack(side="left", padx=(8, 0))
        ttk.Button(top, text="Все таблицы → XLSX…", command=self.save_workbook).pack(side="left", padx=(6, 0))
        ttk.Label(top, textvariable=self.table_info_var).pack(side="right")
        self.table_tree = self._make_tree_host(self.table_tab)

    def _build_stats_tab(self):
        top = ttk.Frame(self.stats_tab); top.pack(fill="x")
        ttk.Label(top, text="Исходная таблица:").pack(side="left")
        self.stats_table_box = ttk.Combobox(top, textvariable=self.stats_table_var, state="readonly", width=50); self.stats_table_box.pack(side="left", padx=6)
        ttk.Button(top, text="Описательная статистика", command=self.show_descriptive).pack(side="left", padx=(8, 0))
        ttk.Button(top, text="Агрегировать…", command=self.open_aggregate_dialog).pack(side="left", padx=(6, 0))
        self.stats_tree = self._make_tree_host(self.stats_tab)
        ttk.Button(self.stats_tab, text="Сохранить текущую статистическую таблицу…", command=self.save_stats_table).pack(anchor="w", pady=(6, 0))

    def _build_manifest_tab(self):
        top = ttk.Frame(self.manifest_tab); top.pack(fill="x")
        self.json_box = ttk.Combobox(top, textvariable=self.json_var, state="readonly", width=70); self.json_box.pack(side="left", fill="x", expand=True)
        self.json_box.bind("<<ComboboxSelected>>", lambda e: self.show_json())
        self.json_text = tk.Text(self.manifest_tab, wrap="none", font="TkFixedFont"); self.json_text.pack(fill="both", expand=True, pady=(8, 0))

    def open_directory(self):
        initial = self.path_var.get() or os.getcwd()
        path = filedialog.askdirectory(parent=self, title="Каталог результатов эксперимента", initialdir=initial if os.path.isdir(initial) else os.getcwd())
        if path:
            self.path_var.set(path); self.load_path()

    def load_path(self):
        path = self.path_var.get().strip()
        if not path: return
        try: self.result = load_experiment(path)
        except Exception as exc:
            messagebox.showerror("Ошибка", str(exc), parent=self); return
        self.path_var.set(str(self.result.root)); self._refresh_result()
        self.status_var.set(f"Загружено: {self.result.experiment_id}; таблиц: {len(self.result.tables)}" + (f"; предупреждений: {len(self.result.warnings)}" if self.result.warnings else ""))

    def _refresh_result(self):
        assert self.result is not None
        self.summary_tree.delete(*self.summary_tree.get_children())
        for key, value in self.result.summary().items(): self.summary_tree.insert("", "end", values=(key, value))
        for warning in self.result.warnings: self.summary_tree.insert("", "end", values=("Предупреждение", warning))
        tables = self.result.table_names()
        for box in (self.chart_table_box, self.table_box, self.stats_table_box): box["values"] = tables
        if tables:
            self.chart_table_var.set(tables[0]); self.table_var.set(tables[0]); self.stats_table_var.set(tables[0]); self._on_chart_table(); self.show_table()
        json_names = sorted(self.result.json_documents); self.json_box["values"] = json_names
        if json_names:
            preferred = next((n for n in json_names if n.endswith("manifest.json")), json_names[0]); self.json_var.set(preferred); self.show_json()

    def _on_chart_table(self, event=None):
        if not self.result: return
        name = self.chart_table_var.get()
        if name not in self.result.tables: return
        df = self.result.tables[name]; all_cols = [str(c) for c in df.columns]
        self.x_box["values"] = [""] + all_cols; self.y_box["values"] = [""] + numeric_columns(df)
        self.group_box["values"] = [""] + categorical_columns(df); self.filter_col_box["values"] = [""] + all_cols
        spec = suggested_spec(df, name)
        self.x_var.set(spec.x); self.y_var.set(spec.y); self.group_var.set(spec.group)
        self.agg_var.set("Среднее"); self.error_var.set("95% CI"); self.filter_col_var.set(""); self.filter_value_var.set("")
        self.title_var.set(""); self.x_label_var.set(""); self.y_label_var.set("")

    def _on_filter_column(self, event=None):
        if not self.result: return
        name, col = self.chart_table_var.get(), self.filter_col_var.get()
        if name not in self.result.tables or col not in self.result.tables[name].columns:
            self.filter_value_box["values"] = [""]; self.filter_value_var.set(""); return
        values = self.result.tables[name][col].dropna().astype(str).drop_duplicates().head(500).tolist()
        self.filter_value_box["values"] = [""] + values; self.filter_value_var.set("")

    def auto_chart_spec(self): self._on_chart_table(); self.build_chart()

    def _current_chart_spec(self) -> ChartSpec:
        return ChartSpec(chart_type=CHART_TYPES.get(self.chart_type_var.get(), "bar"), table_name=self.chart_table_var.get(),
                         x=self.x_var.get(), y=self.y_var.get(), group=self.group_var.get(), aggregation=AGGREGATIONS.get(self.agg_var.get(), "mean"),
                         error=ERROR_TYPES.get(self.error_var.get(), "none"), title=self.title_var.get(), x_label=self.x_label_var.get(), y_label=self.y_label_var.get(),
                         filter_column=self.filter_col_var.get(), filter_value=self.filter_value_var.get())

    def build_chart(self):
        if _MPL_IMPORT_ERROR is not None:
            messagebox.showerror("Matplotlib", "Matplotlib не установлен.\n\npip install matplotlib", parent=self); return
        if not self.result: return
        name = self.chart_table_var.get()
        if name not in self.result.tables: return
        try: fig = build_figure(self.result.tables[name], self._current_chart_spec())
        except Exception as exc:
            messagebox.showerror("График", str(exc), parent=self); return
        for child in self.chart_host.winfo_children(): child.destroy()
        self.current_figure = fig; self.chart_canvas = FigureCanvasTkAgg(fig, master=self.chart_host); self.chart_canvas.draw(); self.chart_canvas.get_tk_widget().pack(fill="both", expand=True)
        self.chart_toolbar = NavigationToolbar2Tk(self.chart_canvas, self.chart_host, pack_toolbar=False); self.chart_toolbar.update(); self.chart_toolbar.pack(fill="x")
        self.status_var.set(f"График построен из таблицы: {name}")

    def save_chart(self):
        if self.current_figure is None:
            messagebox.showinfo("График", "Сначала постройте график.", parent=self); return
        path = filedialog.asksaveasfilename(parent=self, title="Сохранить график", defaultextension=".png",
                                            filetypes=[("PNG, 300 dpi", "*.png"), ("SVG", "*.svg"), ("PDF", "*.pdf")])
        if not path: return
        try: save_figure(self.current_figure, path)
        except Exception as exc: messagebox.showerror("Экспорт", str(exc), parent=self); return
        self.status_var.set(f"График сохранён: {path}")

    def _fill_tree(self, tree: ttk.Treeview, df: pd.DataFrame, max_rows: int = 5000):
        tree.delete(*tree.get_children()); cols = [str(c) for c in df.columns]; tree["columns"] = cols
        for col in cols:
            tree.heading(col, text=col, command=lambda c=col, t=tree, d=df: self._sort_tree(t, d, c))
            tree.column(col, width=min(240, max(85, len(col) * 9)), minwidth=60, anchor="center")
        for _, row in df.head(max_rows).iterrows(): tree.insert("", "end", values=[display_value(row[c]) for c in df.columns])

    def _sort_tree(self, tree: ttk.Treeview, df: pd.DataFrame, col: str):
        key = (id(tree), col); ascending = self._table_sort_state.get(key, True); self._table_sort_state[key] = not ascending
        try: sorted_df = df.sort_values(col, ascending=ascending, na_position="last")
        except Exception: sorted_df = df.assign(_sort=df[col].astype(str)).sort_values("_sort", ascending=ascending).drop(columns="_sort")
        self._fill_tree(tree, sorted_df)

    def show_table(self):
        if not self.result: return
        name = self.table_var.get()
        if name not in self.result.tables: return
        df = self.result.tables[name]; self._fill_tree(self.table_tree, df); self.table_info_var.set(f"{min(len(df), 5000)}/{len(df)} строк × {len(df.columns)} столбцов")

    def save_current_table(self):
        if not self.result: return
        name = self.table_var.get()
        if name not in self.result.tables: return
        path = filedialog.asksaveasfilename(parent=self, title="Сохранить таблицу", defaultextension=".csv",
                                            filetypes=[("CSV UTF-8", "*.csv"), ("Excel XLSX", "*.xlsx"), ("LaTeX", "*.tex"), ("Markdown", "*.md")])
        if not path: return
        try: save_table(self.result.tables[name], path)
        except Exception as exc: messagebox.showerror("Экспорт", str(exc), parent=self); return
        self.status_var.set(f"Таблица сохранена: {path}")

    def save_workbook(self):
        if not self.result: return
        path = filedialog.asksaveasfilename(parent=self, title="Сохранить все таблицы", defaultextension=".xlsx", filetypes=[("Excel XLSX", "*.xlsx")])
        if not path: return
        try: export_workbook(self.result, path)
        except Exception as exc: messagebox.showerror("Экспорт", str(exc), parent=self); return
        self.status_var.set(f"Workbook сохранён: {path}")

    def show_descriptive(self):
        if not self.result: return
        name = self.stats_table_var.get()
        if name not in self.result.tables: return
        self._last_stats_df = descriptive_table(self.result.tables[name]); self._fill_tree(self.stats_tree, self._last_stats_df); self.status_var.set(f"Описательная статистика: {name}")

    def open_aggregate_dialog(self):
        if not self.result: return
        name = self.stats_table_var.get()
        if name not in self.result.tables: return
        df = self.result.tables[name]
        dialog = tk.Toplevel(self); dialog.title("Агрегация"); dialog.transient(self); dialog.grab_set(); dialog.geometry("620x520")
        ttk.Label(dialog, text="Группировка (можно выбрать несколько):").pack(anchor="w", padx=10, pady=(10, 4))
        group_list = tk.Listbox(dialog, selectmode="extended", exportselection=False)
        for c in categorical_columns(df) + [c for c in numeric_columns(df) if c == "seed"]: group_list.insert("end", c)
        group_list.pack(fill="both", expand=True, padx=10)
        ttk.Label(dialog, text="Метрики (можно выбрать несколько):").pack(anchor="w", padx=10, pady=(10, 4))
        metric_list = tk.Listbox(dialog, selectmode="extended", exportselection=False)
        for c in numeric_columns(df): metric_list.insert("end", c)
        metric_list.pack(fill="both", expand=True, padx=10)
        buttons = ttk.Frame(dialog); buttons.pack(fill="x", padx=10, pady=10)
        def do_aggregate():
            groups = [group_list.get(i) for i in group_list.curselection()]; metrics = [metric_list.get(i) for i in metric_list.curselection()]
            if not metrics:
                messagebox.showwarning("Агрегация", "Выберите хотя бы одну метрику.", parent=dialog); return
            try: out = aggregate_table(df, groups, metrics)
            except Exception as exc: messagebox.showerror("Агрегация", str(exc), parent=dialog); return
            self._last_stats_df = out; self._fill_tree(self.stats_tree, out); dialog.destroy()
        ttk.Button(buttons, text="Рассчитать", command=do_aggregate).pack(side="right")
        ttk.Button(buttons, text="Отмена", command=dialog.destroy).pack(side="right", padx=(0, 6))

    def save_stats_table(self):
        if self._last_stats_df.empty:
            messagebox.showinfo("Статистика", "Сначала сформируйте статистическую таблицу.", parent=self); return
        path = filedialog.asksaveasfilename(parent=self, title="Сохранить статистическую таблицу", defaultextension=".csv",
                                            filetypes=[("CSV UTF-8", "*.csv"), ("Excel XLSX", "*.xlsx"), ("LaTeX", "*.tex"), ("Markdown", "*.md")])
        if not path: return
        try: save_table(self._last_stats_df, path)
        except Exception as exc: messagebox.showerror("Экспорт", str(exc), parent=self)

    def show_json(self):
        if not self.result: return
        name = self.json_var.get()
        if name not in self.result.json_documents: return
        self.json_text.delete("1.0", "end"); self.json_text.insert("1.0", json.dumps(self.result.json_documents[name], ensure_ascii=False, indent=2, default=str))

    def export_package(self):
        if not self.result:
            messagebox.showinfo("Экспорт", "Сначала загрузите эксперимент.", parent=self); return
        path = filedialog.askdirectory(parent=self, title="Каталог для экспортируемого пакета")
        if not path: return
        target = Path(path) / f"{self.result.experiment_id}_report"
        try: export_result_package(self.result, target)
        except Exception as exc: messagebox.showerror("Экспорт", str(exc), parent=self); return
        self.status_var.set(f"Пакет результатов сохранён: {target}")


def open_results_window(master, initial_root=None):
    return ExperimentResultsWindow(master, initial_root=initial_root)
