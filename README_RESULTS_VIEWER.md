# GUI Results Viewer

Модуль добавляет в Tkinter GUI просмотр, построение и экспорт результатов экспериментов.

## Возможности

- загрузка каталогов `runs/...`;
- поиск `manifest.json`, `metrics.json`, `null_test.json`;
- поддержка `all_seed_metrics.csv`, `multiseed_raw.csv`, `multiseed_agg.csv`,
  `multiseed_tests.csv`, `aggregated.csv`, `significance.csv`, `models.csv`,
  абляций и `sophistication_sweep.csv`;
- таблицы через `ttk.Treeview`;
- сортировка по заголовкам;
- описательная статистика;
- произвольная агрегация mean/std/count/CI95;
- bar/line/scatter/boxplot/histogram;
- группировка и фильтрация;
- error bars: std / 95% CI;
- экспорт графиков PNG 300 dpi / SVG / PDF;
- экспорт таблиц CSV / XLSX / LaTeX / Markdown;
- экспорт всех таблиц в один XLSX;
- экспорт самодостаточного пакета результатов.

## Зависимости

```powershell
pip install pandas matplotlib openpyxl tabulate
```

## Установка

Скопируйте содержимое архива в корень `signal_network_sim`, затем:

```powershell
python .\tools\install_results_viewer.py
python -m unittest tests.test_results_viewer
python -m gui
```

В меню появится:

```text
Симуляция
  └─ Результаты экспериментов…
```

Открывать можно `runs/baseline/<run-id>/`, конкретный протокол или внутренний training-run.

## Форматы для диссертации

- SVG — основной векторный рисунок;
- PDF — публикации;
- PNG 300 dpi — Word;
- XLSX — проверка таблиц;
- LaTeX — публикации;
- CSV — воспроизводимость.
