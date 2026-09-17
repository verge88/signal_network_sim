"""Tk panel for planning experiments and checking topology transfer."""
from __future__ import annotations

import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import pandas as pd

from ..theme import style_text
from ..widgets import LogView
from .protocol import JobSpec
from .registry import (aggregate_seeds, list_runs, metrics_frame, new_run_dir,
                       transfer_matrix, write_manifest)
from .runner import JobRunner, JobState


class MLLab(tk.Toplevel):
    def __init__(self, editor):
        super().__init__(editor.master)
        self.editor = editor
        self.pal = editor.pal
        self.title("Лаборатория моделей")
        self.geometry(f"{int(1150 * editor.gs)}x{int(720 * editor.gs)}")
        self.configure(bg=self.pal["panel_bg"])
        self.repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.runs_root = os.path.join(self.repo_root, "runs")
        self.runner = JobRunner(self.repo_root, max_parallel=1)
        self._rows: dict[int, JobState] = {}
        notebook = ttk.Notebook(self)
        notebook.pack(fill="both", expand=True, padx=8, pady=8)
        notebook.add(self._plan_tab(notebook), text="Эксперимент")
        notebook.add(self._queue_tab(notebook), text="Очередь и журнал")
        notebook.add(self._results_tab(notebook), text="Результаты")
        notebook.add(self._transfer_tab(notebook), text="Тест переноса")
        self.protocol("WM_DELETE_WINDOW", self._close)
        self.after(150, self._pump)

    def _plan_tab(self, notebook) -> ttk.Frame:
        frame = ttk.Frame(notebook, padding=12)
        self.vars = {
            "sim_dir": tk.StringVar(value=self._guess_sim_dir()),
            "tag": tk.StringVar(value="baseline"),
            "seeds": tk.StringVar(value="42,43,44,45,46"),
            "dataset": tk.StringVar(value=""),
            "make_dataset": tk.BooleanVar(value=True),
            "do_train": tk.BooleanVar(value=True),
            "capture": tk.BooleanVar(value=True),
            "threads": tk.StringVar(value="0"),
            "parallel": tk.StringVar(value="1"),
        }
        row = 0
        ttk.Label(frame, text=f"Протокол: {self.editor.topo.protocol}", style="Title.TLabel").grid(row=row, column=0, columnspan=3, sticky="w", pady=(0, 8))
        row += 1
        self._path_row(frame, row, "Каталог симулятора/пайплайна", "sim_dir", directory=True); row += 1
        self._entry_row(frame, row, "Метка эксперимента", "tag"); row += 1
        self._entry_row(frame, row, "Сиды (через запятую)", "seeds"); row += 1
        ttk.Checkbutton(frame, text="Генерировать датасет из текущей топологии", variable=self.vars["make_dataset"]).grid(row=row, column=0, columnspan=3, sticky="w"); row += 1
        self._path_row(frame, row, "Или готовый CSV", "dataset"); row += 1
        ttk.Checkbutton(frame, text="Обучать модели", variable=self.vars["do_train"]).grid(row=row, column=0, sticky="w")
        ttk.Checkbutton(frame, text="Сохранять обученные модели", variable=self.vars["capture"]).grid(row=row, column=1, sticky="w"); row += 1
        self._entry_row(frame, row, "Потоков на прогон (0 = все)", "threads"); row += 1
        self._entry_row(frame, row, "Прогонов параллельно", "parallel"); row += 1
        ttk.Label(frame, text="Переопределения TrainingConfig (ключ=значение, по одному в строке)", style="Hint.TLabel").grid(row=row, column=0, columnspan=3, sticky="w", pady=(10, 2)); row += 1
        self.over_text = tk.Text(frame, height=7, width=64, font="TkFixedFont")
        self.over_text.insert("1.0", "")
        self.over_text.grid(row=row, column=0, columnspan=3, sticky="ew"); style_text(self.over_text, self.pal); row += 1
        buttons = ttk.Frame(frame); buttons.grid(row=row, column=0, columnspan=3, sticky="ew", pady=10)
        ttk.Button(buttons, text="В очередь (все сиды)", command=self.enqueue).pack(side="left")
        ttk.Button(buttons, text="Только один сид", command=lambda: self.enqueue(single=True)).pack(side="left", padx=6)
        frame.columnconfigure(1, weight=1)
        return frame

    def _entry_row(self, parent, row, label, key):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=3)
        ttk.Entry(parent, textvariable=self.vars[key], width=52).grid(row=row, column=1, sticky="ew")

    def _path_row(self, parent, row, label, key, directory=False):
        self._entry_row(parent, row, label, key)
        command = lambda: self._choose(key, directory)
        ttk.Button(parent, text="…", width=3, command=command).grid(row=row, column=2)

    def _choose(self, key, directory):
        value = filedialog.askdirectory(parent=self) if directory else filedialog.askopenfilename(parent=self, filetypes=[("CSV", "*.csv")])
        if value:
            self.vars[key].set(value)

    def _queue_tab(self, notebook):
        frame = ttk.Frame(notebook, padding=8)
        columns = ("run", "seed", "status", "stage", "progress", "elapsed")
        self.qtree = ttk.Treeview(frame, columns=columns, show="headings", height=9)
        for column in columns:
            self.qtree.heading(column, text=column); self.qtree.column(column, width=130, anchor="center")
        self.qtree.pack(fill="x")
        buttons = ttk.Frame(frame); buttons.pack(fill="x", pady=6)
        ttk.Button(buttons, text="Отменить выбранный", command=self.cancel_selected).pack(side="left")
        ttk.Button(buttons, text="Отменить все", command=self.runner.cancel_all).pack(side="left", padx=6)
        self.qlog = LogView(frame, self.pal, height=18)
        self.qlog.pack(fill="both", expand=True)
        return frame

    def _results_tab(self, notebook):
        frame = ttk.Frame(notebook, padding=8)
        columns = ("run", "protocol", "seed", "tag", "status", "best_model", "best_roc_auc", "pr_auc", "commit")
        self.rtree = ttk.Treeview(frame, columns=columns, show="headings", height=14)
        for column in columns:
            self.rtree.heading(column, text=column); self.rtree.column(column, width=120, anchor="center")
        self.rtree.column("run", width=220); self.rtree.pack(fill="both", expand=True)
        buttons = ttk.Frame(frame); buttons.pack(fill="x", pady=6)
        ttk.Button(buttons, text="Обновить", command=self.refresh_runs).pack(side="left")
        ttk.Button(buttons, text="Метрики", command=self.show_metrics).pack(side="left", padx=6)
        ttk.Button(buttons, text="Свести по сидам", command=self.aggregate).pack(side="left")
        ttk.Button(buttons, text="Экспорт CSV", command=self.export_results).pack(side="left", padx=6)
        self.rinfo = LogView(frame, self.pal, height=8)
        self.rinfo.pack(fill="both", expand=True)
        self.refresh_runs()
        return frame

    def _transfer_tab(self, notebook):
        frame = ttk.Frame(notebook, padding=12)
        self.test_run, self.test_data = tk.StringVar(), tk.StringVar()
        ttk.Label(frame, text="Модель одной топологии на другой", style="Title.TLabel").grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 8))
        ttk.Label(frame, text="Каталог прогона").grid(row=1, column=0, sticky="w")
        ttk.Entry(frame, textvariable=self.test_run, width=56).grid(row=1, column=1, sticky="ew")
        ttk.Button(frame, text="…", command=lambda: self._choose_test("run")).grid(row=1, column=2)
        ttk.Label(frame, text="Тестовый CSV").grid(row=2, column=0, sticky="w", pady=4)
        ttk.Entry(frame, textvariable=self.test_data, width=56).grid(row=2, column=1, sticky="ew")
        ttk.Button(frame, text="…", command=lambda: self._choose_test("data")).grid(row=2, column=2)
        buttons = ttk.Frame(frame); buttons.grid(row=3, column=0, columnspan=3, sticky="w", pady=8)
        ttk.Button(buttons, text="Оценить", command=self.run_transfer).pack(side="left")
        ttk.Button(buttons, text="Матрица переноса", command=self.show_matrix).pack(side="left", padx=6)
        self.tinfo = LogView(frame, self.pal, height=20)
        self.tinfo.grid(row=4, column=0, columnspan=3, sticky="nsew")
        frame.columnconfigure(1, weight=1); frame.rowconfigure(4, weight=1)
        return frame

    def _choose_test(self, kind):
        value = filedialog.askdirectory(parent=self) if kind == "run" else filedialog.askopenfilename(parent=self, filetypes=[("CSV", "*.csv")])
        if value:
            (self.test_run if kind == "run" else self.test_data).set(value)

    def _guess_sim_dir(self):
        folder = {"diameter": "diameter", "ss7": "ss7", "sip": "sip", "5g_sba": "5g"}.get(self.editor.topo.protocol, "")
        return os.path.join(self.repo_root, folder)

    def _overrides(self):
        result = {}
        for line in self.over_text.get("1.0", "end").splitlines():
            line = line.split("#", 1)[0].strip()
            if not line or "=" not in line:
                continue
            key, raw = (part.strip() for part in line.split("=", 1))
            try: result[key] = int(raw)
            except ValueError:
                try: result[key] = float(raw)
                except ValueError: result[key] = {"true": True, "false": False}.get(raw.lower(), raw)
        return result

    def enqueue(self, single=False):
        if self.vars["make_dataset"].get() and any(item.startswith("ERROR") for item in self.editor.topo.validate()):
            messagebox.showerror("Топология некорректна", "Исправьте ошибки топологии перед запуском.", parent=self); return
        if not self.vars["make_dataset"].get() and not self.vars["dataset"].get():
            messagebox.showerror("Нет данных", "Укажите CSV или включите генерацию.", parent=self); return
        try: seeds = [int(item) for item in self.vars["seeds"].get().replace(" ", "").split(",") if item]
        except ValueError:
            messagebox.showerror("Сиды", "Список сидов должен состоять из целых чисел.", parent=self); return
        if single: seeds = seeds[:1]
        self.runner.max_parallel = max(1, int(self.vars["parallel"].get() or 1))
        protocol, tag, overrides = self.editor.topo.protocol, self.vars["tag"].get(), self._overrides()
        for seed in seeds:
            run_dir = new_run_dir(self.runs_root, protocol, f"{tag}_s{seed}")
            topology_path = os.path.join(run_dir, "topology.json")
            self.editor.topo.save(topology_path)
            write_manifest(run_dir, protocol=protocol, seed=seed, tag=tag, topology_path=topology_path, topology_snapshot=self.editor.topo.to_dict(), overrides=overrides, notes=self.editor.topo.notes)
            spec = JobSpec(job_id=os.path.basename(run_dir), protocol=protocol, sim_dir=self.vars["sim_dir"].get(), run_dir=run_dir, seed=seed, make_dataset=bool(self.vars["make_dataset"].get()), topology_path=topology_path, dataset_path=self.vars["dataset"].get(), do_train=bool(self.vars["do_train"].get()), train_overrides=overrides, capture_models=bool(self.vars["capture"].get()), threads=int(self.vars["threads"].get() or 0))
            state = self.runner.submit(spec)
            iid = self.qtree.insert("", "end", values=(spec.job_id, seed, "queued", "", "0%", "0s"))
            state.tree_iid = iid  # type: ignore[attr-defined]
            self._rows[id(state)] = state

    def cancel_selected(self):
        for iid in self.qtree.selection():
            for state in self._rows.values():
                if getattr(state, "tree_iid", None) == iid: self.runner.cancel(state)

    def refresh_runs(self):
        self.rtree.delete(*self.rtree.get_children())
        columns = self.rtree["columns"]
        for row in list_runs(self.runs_root):
            self.rtree.insert("", "end", iid=row["dir"], values=[row.get(column, "") for column in columns])

    def show_metrics(self):
        selected = self.rtree.selection()
        if not selected: return
        frame = metrics_frame(selected[0])
        self.rinfo.clear()
        self.rinfo.append(frame.to_string(index=False) if frame is not None else "таблица метрик не найдена")

    def aggregate(self):
        selected = list(self.rtree.selection())
        if len(selected) < 2: messagebox.showinfo("Сводка", "Выберите два прогона или больше.", parent=self); return
        frame = aggregate_seeds(selected)
        self.rinfo.clear()
        self.rinfo.append(frame.to_string(index=False) if not frame.empty else "нет сопоставимых метрик")

    def export_results(self):
        path = filedialog.asksaveasfilename(parent=self, defaultextension=".csv", filetypes=[("CSV", "*.csv")])
        if path: pd.DataFrame(list_runs(self.runs_root)).to_csv(path, index=False)

    def run_transfer(self):
        from .evaluate import evaluate_run
        if not self.test_run.get() or not self.test_data.get(): messagebox.showerror("Тест", "Укажите каталог прогона и CSV.", parent=self); return
        try: rows = evaluate_run(self.test_run.get(), self.test_data.get())
        except Exception as exc: messagebox.showerror("Тест не выполнен", f"{type(exc).__name__}: {exc}", parent=self); return
        frame = pd.DataFrame(rows)
        self.tinfo.clear()
        self.tinfo.append(frame.to_string(index=False))

    def show_matrix(self):
        frame = transfer_matrix(self.runs_root)
        self.tinfo.clear()
        self.tinfo.append(frame.to_string() if not frame.empty else "нет результатов переноса")

    def _pump(self):
        self.runner.drain(self._on_event)
        for state in self._rows.values():
            iid = getattr(state, "tree_iid", None)
            if iid and self.qtree.exists(iid): self.qtree.item(iid, values=(state.spec.job_id, state.spec.seed, state.status, state.stage, f"{state.progress * 100:.0f}%", f"{state.elapsed:.0f}s"))
        self.after(250, self._pump)

    def _on_event(self, kind, state, payload):
        if kind == "log": self._log(f"[{state.spec.seed}] {payload.get('line', '')}")
        elif kind == "stage": self._log(f"[{state.spec.seed}] этап {payload.get('name')}: {payload.get('status')}")
        elif kind == "warn": self._log(f"[{state.spec.seed}] ВНИМАНИЕ: {payload.get('msg')}")
        elif kind == "done":
            status = payload.get("status", state.status)
            self._log(f"[{state.spec.seed}] завершено: {status} ({state.elapsed:.0f} с)")
            if state.traceback:
                self._log(state.traceback)
            elif state.error:
                self._log(f"[{state.spec.seed}] {state.error}")
            self.refresh_runs()

    def _log(self, message):
        self.qlog.append(message)

    def _close(self):
        self.runner.shutdown(); self.destroy()
