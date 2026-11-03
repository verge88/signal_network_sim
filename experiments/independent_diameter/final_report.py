"""Validate completed Actions evidence and write the final Russian report."""
import argparse
import json
from pathlib import Path
import shutil

import pandas as pd


def unique(root, name):
    files = list(Path(root).rglob(name))
    if len(files) != 1:
        raise ValueError(f'expected one {name}, found {files}')
    return files[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--diameter', required=True)
    parser.add_argument('--ss7', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    d = pd.read_csv(unique(args.diameter, 'all_independent_metrics.csv'))
    audit = pd.read_csv(unique(args.diameter, 'replay_audit.csv'))
    # The resume job also downloaded the first consolidated summary, whose
    # audit repeats the leaf artifacts. Count original leaf evidence only.
    audit_source_rows = len(audit)
    if 'artifact' in audit:
        audit = audit[audit.artifact.eq('evidence')].copy()
    s = pd.read_csv(unique(args.ss7, 'independent_metrics.csv'))
    sa = pd.read_csv(unique(args.ss7, 'historical_metric_audit.csv'))
    sp = json.loads(unique(args.ss7, 'provenance.json').read_text())
    assert len(d) == 606, len(d)
    assert set(d.test_seed) == {1001, 1002, 1003}
    assert set(d.train_seed) == set(range(42, 52))
    assert (d.groupby(['suite','train_seed','model']).size() == 6).all()
    assert len(audit) == 101 and audit.historical_predictions_match.all()
    assert len(s) == 6 and set(s.test_seed) == {1001, 1002, 1003}
    assert not sp['test_labels_used_for_fit'] and not sp['test_labels_used_for_threshold']
    failures = d[d.status.fillna('ok').ne('ok')]
    hashes = d.groupby(['test_seed','scenario']).dataset_sha256.nunique()
    dh = int(d.dataset_sha256.nunique())
    lines = [
        '# Независимая проверка моделей в GitHub Actions', '',
        'Проверены восстановленные экземпляры ранее обученных моделей Diameter и SS7. В исходных артефактах веса не сохранялись. Для Diameter сверка всех 101 экземпляра с архивными прогнозами пройдена; для SS7 доступны только архивные метрики, их сверка приведена отдельно.', '',
        'Предобработка и пороги зафиксированы до применения новых тестов. Тестовые метки не использованы для обучения, настройки порогов или выбора модели. Сохранены веса для последующих проверок без обучения.', '',
        'Это новые синтетические данные того же семейства симуляторов. Результат не подтверждает перенос на реальный операторский трафик.', '',
        '## Diameter', '',
        'Обучающие seed: 42–51 для классических моделей; отдельный полный прогон с AE/VAE: seed 42. Тестовые seed: 1001–1003. Для каждого — исходное расписание и заранее заданные сдвиги событий с перестановкой целевых узлов в пределах их типа и роли.', '',
        'Средние по десяти обучающим seed и трём тестовым seed для сдвинутых сценариев. Полнота и FPR относятся к окнам; порог выбран по 1% FPR на валидационных отрицательных примерах.', '',
        '| Модель | ROC AUC | Полнота | FPR |', '|---|---:|---:|---:|']
    classical = d[(d.suite == 'classical') & (d.scenario == 'shifted')]
    means = classical.groupby('model')[['roc_auc','recall','fpr']].mean()
    for model in ['Gradient Boosting','Cascade','Random Forest','Isolation Forest','Meta-Ensemble','RF (no MS)','RF (no temporal)']:
        r = means.loc[model]
        lines.append(f'| {model} | {r.roc_auc:.3f} | {100*r.recall:.1f}% | {100*r.fpr:.2f}% |')
    lines += ['', 'Для Meta-Ensemble использован фиксированный порог 0,5: его FPR нельзя трактовать как калибровку на 1%.', '',
              'Двухоконное подтверждение для Gradient Boosting даёт среднюю полноту '
              f'{100*classical[classical.model.eq("Gradient Boosting")].debounce2_recall.mean():.1f}% и FPR '
              f'{100*classical[classical.model.eq("Gradient Boosting")].debounce2_fpr.mean():.2f}%. '
              'Это заранее заданное правило, а не настройка по тестам.', '',
              '## Нейросетевые модели', '']
    ae = d[(d.suite == 'neural') & d.model.eq('Autoencoder') & d.scenario.eq('shifted')]
    lines += [f'Автоэнкодер, seed 42: AUC {ae.roc_auc.mean():.3f}, полнота {100*ae.recall.mean():.1f}%, FPR {100*ae.fpr.mean():.2f}% на сдвинутых сценариях.', '',
              f'VAE: численные ошибки в {len(failures)} из 6 тестовых наборов. В каждом таком наборе одна неконечная оценка на нормальном окне. Полные AUC/полнота/FPR для этих наборов не приводятся; некорректные оценки не заменялись нулём или медианой. Метаансамбль сохраняет предусмотренную исходным кодом медианную замену неконечных входов; соответствующие случаи отмечены в CSV.', '',
              '## SS7', '',
              'Восстановлены Random Forest и метаансамбль из завершённого задания seed 42: обучение на шести днях, тесты на трёх новых двухсуточных реализациях. Использованы прежние валидационные пороги и исходное строгое сравнение score > threshold.', '',
              '| Модель | ROC AUC | Полнота окон | FPR окон | Полнота эпизодов | Ложные инциденты/сутки |',
              '|---|---:|---:|---:|---:|---:|']
    for model, group in s.groupby('model'):
        r = group.select_dtypes('number').mean()
        lines.append(f'| {model} | {r.roc_auc:.3f} | {100*r.window_recall:.1f}% | {100*r.window_fpr:.3f}% | {100*r.episode_recall:.1f}% | {r.false_incidents_per_day:.2f} |')
    rf = s[s.model.eq('random_forest')]
    lines += ['', f'Разброс SS7 между тестовыми реализациями: AUC {rf.roc_auc.min():.3f}–{rf.roc_auc.max():.3f}; '
              f'FPR {100*rf.window_fpr.min():.3f}–{100*rf.window_fpr.max():.3f}%; '
              f'ложные инциденты {rf.false_incidents_per_day.min():.1f}–{rf.false_incidents_per_day.max():.1f} в сутки.']
    lines += ['', f'Метаансамбль перешёл на предусмотренный исходным кодом Random Forest: {sp["meta_ensemble_fell_back_to_rf"]}. '
              'При этой замене это не независимый дополнительный алгоритм.', '',
              f'Совпали архивные метрики SS7: {int(sa.matches.sum())}/{len(sa)}. '
              'Совпадение метрик не доказывает равенство старых весов: они не были сохранены.', '',
              '## Ограничения воспроизводимости и интерпретации', '',
              f'Для шести настроек тестов Diameter зафиксировано {dh} различных SHA-256 CSV. '
              'Число вариантов для каждой пары seed/сценарий:', '', '```', hashes.to_string(), '```', '',
              'Следовательно, побайтово общая тестовая выборка для всех заданий не подтверждена. Причина различий не установлена. Средние описывают фактически полученные результаты; парные статистические выводы на предположении идентичных файлов и доверительные интервалы по коррелированным строкам не рассчитывались.', '',
              'Средний FPR на новых данных превышает валидационную цель 1% у ряда моделей. Высокий AUC сам по себе не подтверждает пригодность выбранного рабочего порога.', '',
              '## Запуски и материалы', '',
              '- Исходное восстановление Diameter: https://github.com/verge88/signal_network_sim/actions/runs/41654090862',
              '- Завершённая оценка Diameter из сохранённых весов: https://github.com/verge88/signal_network_sim/actions/runs/41656721943',
              '- SS7: https://github.com/verge88/signal_network_sim/actions/runs/41654690603', '',
              'CSV рядом с отчётом содержат все результаты, статусы численных ошибок, хеши и сверку с историческими прогнозами/метриками.']
    (out / 'report_ru.md').write_text('\n'.join(lines), encoding='utf-8')
    for source, name in [(unique(args.diameter,'all_independent_metrics.csv'),'diameter_metrics.csv'),
                         (unique(args.ss7,'independent_metrics.csv'),'ss7_metrics.csv'),
                         (unique(args.ss7,'historical_metric_audit.csv'),'ss7_metric_audit.csv'),
                         (unique(args.ss7,'provenance.json'),'ss7_provenance.json')]:
        shutil.copyfile(source, out / name)
    audit.to_csv(out / 'diameter_replay_audit.csv', index=False)
    write = dict(diameter_rows=len(d), diameter_model_replicas=len(audit),
                 diameter_replay_matches=bool(audit.historical_predictions_match.all()),
                 audit_summary_duplicate_rows_excluded=audit_source_rows-len(audit),
                 diameter_numerical_failures=len(failures), diameter_distinct_dataset_hashes=dh,
                 ss7_rows=len(s), ss7_archived_metric_matches=int(sa.matches.sum()),
                 external_real_traffic=False, historical_weights_available=False)
    (out / 'validation.json').write_text(json.dumps(write, indent=2), encoding='utf-8')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
