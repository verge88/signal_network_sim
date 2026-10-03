from __future__ import annotations

import shutil
from pathlib import Path


def find_repo_root() -> Path:
    p = Path.cwd().resolve()
    for candidate in (p, *p.parents):
        if (candidate / "gui" / "editor.py").exists():
            return candidate
    raise RuntimeError("Не найден корень signal_network_sim.")


def main() -> int:
    root = find_repo_root()
    editor = root / "gui" / "editor.py"
    backup = root / ".results_viewer_backup" / "gui" / "editor.py"
    backup.parent.mkdir(parents=True, exist_ok=True)
    if not backup.exists():
        shutil.copy2(editor, backup)

    text = editor.read_text(encoding="utf-8")
    original = text

    menu_anchor = '        r.add_command(label="Показать сводку по сети", command=self.show_summary)\n'
    menu_line = '        r.add_command(label="Результаты экспериментов…", command=self.open_results_viewer)\n'
    if menu_line not in text:
        if menu_anchor not in text:
            raise RuntimeError("Не найден пункт 'Показать сводку по сети' в gui/editor.py.")
        text = text.replace(menu_anchor, menu_anchor + menu_line, 1)

    method_anchor = "    def open_ml_lab(self) -> None:\n"
    method = '''    def open_results_viewer(self) -> None:\n        from .results.window import ExperimentResultsWindow\n\n        if (\n            getattr(self, "_results_viewer", None)\n            and self._results_viewer.winfo_exists()\n        ):\n            self._results_viewer.lift()\n            self._results_viewer.focus_force()\n            return\n\n        repo_root = os.path.dirname(\n            os.path.dirname(os.path.abspath(__file__))\n        )\n        initial_root = os.path.join(repo_root, "runs")\n        self._results_viewer = ExperimentResultsWindow(\n            self.master,\n            initial_root=initial_root,\n        )\n\n'''
    if "    def open_results_viewer(self) -> None:\n" not in text:
        if method_anchor not in text:
            raise RuntimeError("Не найден метод open_ml_lab; автоматическая вставка невозможна.")
        text = text.replace(method_anchor, method + method_anchor, 1)

    if text != original:
        editor.write_text(text, encoding="utf-8")
        print("gui/editor.py обновлён.")
    else:
        print("gui/editor.py уже содержит Results Viewer.")
    print("Резервная копия:", backup)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
