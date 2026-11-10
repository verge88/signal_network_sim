"""Generate the fixed-protocol report from all six verified outcomes."""
import json
from pathlib import Path
import numpy as np

root=Path(__file__).parent; s=json.loads((root/'results/summary.json').read_text())
d=json.loads((root/'results/diagnostics.json').read_text())
early=s['by_mode']['early']; late=s['by_mode']['late']
success=all(late['preregistered_outcomes_all_three'].values())
et,lt=early['timing'],late['timing']
lines=['# Получение контрольного отчёта непосредственно перед решением', '',
f"Предварительно заданные критерии для исправленного порядка {'выполнены во всех трёх прогонах' if success else 'выполнены не полностью; результаты приведены ниже'}. При прежнем порядке после возврата контрольного вызова устарели {et['normal_reports_aging_after_return']} из {et['normal_reports_fresh_at_return']} нормальных свежих отчётов ({100*et['post_return_expiry_fraction']:.4f}%). При запросе непосредственно перед решением — {lt['normal_reports_aging_after_return']} из {lt['normal_reports_fresh_at_return']} ({100*lt['post_return_expiry_fraction']:.4f}%). Предел свежести 250 мс сохранён.", '',
'[Запуск 42771575096](https://github.com/verge88/signal_network_sim/actions/runs/42771575096) · измеренный код `5e5767ea52a54743e1458eb270336754cf3dd2b6` · [предварительный протокол](PROTOCOL_RU.md). Исправленный аудит: [42779123046](https://github.com/verge88/signal_network_sim/actions/runs/42779123046), commit `4a1a46a407732953f8a42532d24d931968c8cf60`.', '',
'## Изменение и независимая проверка', '',
'Ранее контрольный отчёт запрашивался при обработке slot-4, затем ждал остальные NF и пакетный расчёт оценок. Теперь в режиме late он запрашивается после этой работы, непосредственно перед проверкой slot-4. В обоих режимах остаётся один запрос на окно. Время решения читается после получения и записи отчёта; подписанная метка не изменяется. Правила проверки подписи, nonce, сессии, последовательности и доступности сохранены.', '',
'Шесть новых часовых прогонов: три ранних early и три поздних late, каждый на 32 логических NF-позициях. Для каждой пары одинаковые новые seeds 54133, 54233, 54333 и одинаковая прежняя модель из серии 42577623990. Использованы прежние SHA-256 моделей; обучения и изменения порогов нет. Четыре фоновых режима и все 36 эпизодов на прогон сохранены. Всего 691200 NF-окон и 216 эпизодов; на режим 345600 NF-окон, 1410 атакующих окон и 344190 нормальных.', '',
'Все 45 исходных программных тестов прошли локально и в каждом из шести заданий измерений. Часовые измерения завершились, базовый независимый аудит воспроизвёл решения, но дополнительный аудит остановился с TypeError: pandas автоматически превратил decision_time из числа секунд в Timestamp. Поэтому исходный workflow имеет статус failure, и этот факт сохраняется. Чтение исправлено на сохранение исходных числовых значений JSON, добавлен регрессионный тест; хеш нагрузки канонизирован с LF для переноса между Linux и Windows. Данные, детекторы, пороги и измерения не изменены. Отдельный повторный аудит 42779123046 использовал те же шесть архивов, выполнил по 46 тестов и успешно завершился во всех заданиях. Все архивы проверены локально по SHA-256; полный локальный пересчёт совпал с исправленным аудитом CI.', '',
f"Совпадение реализованных счётчиков и семантических входов между режимами подтверждено в {sum(p['realized_workload_matches'] for p in s['pairs'])}/3 пар по хешам. Сравнение значений по ячейкам дало {sum(p['different_cells'] for p in d['paired_inputs'])} различий. Временные метки и задержки в этот хеш не входят. Стенды работают на разных облачных runner: одинаковая нагрузка не означает одинаковое оборудование.", '',
'## Первичный показатель по каждой модели', '',
'Отчёт относится к нормальным по метке нормального окна slot-4. Знаменатель ниже включает только отчёты, свежие при возврате контрольного вызова. Первичный критерий late — не более 0,1% последующих истечений срока в каждом прогоне. Это показатель свежести, отличный от FPR; достижение одного не означает достижение другого.', '',
'| Source seed | Early: устарели / свежие при возврате | Late: устарели / свежие при возврате | Early p95 возраста, мс | Late p95 возраста, мс |',
'|---|---:|---:|---:|---:|']
for p in s['pairs']:
    a,b=p['early_timing'],p['late_timing']
    lines.append(f"| {p['source_seed']} | {a['normal_reports_aging_after_return']}/{a['normal_reports_fresh_at_return']} | {b['normal_reports_aging_after_return']}/{b['normal_reports_fresh_at_return']} | {a['normal_report_age_p95_ms']:.3f} | {b['normal_report_age_p95_ms']:.3f} |")
lines += ['', 'Дополнительная диагностика после основного анализа проверяет возраст и к концу вычисления решения. Она не заменяет предварительный критерий:', '']
for mode in ['early','late']:
    selected=[r for r in d['runs'] if r['mode']==mode]
    expired=sum(r['normal_reports_aging_after_return_by_decision_end'] for r in selected)
    max_duration=max(r['normal_evaluation_duration_max_ms'] for r in selected)
    lines.append(f'- {mode}: {expired} нормальных отчётов, свежих при возврате, истекли к концу решения; максимальная длительность вычисления решения {max_duration:.3f} мс.')
lines += ['', 'p95 возраста рассчитан по всем нормальным окнам с существующим исходным отчётом, включая намеренные штатные задержки доставки. Медианы и p95 ожидания после возврата отдельно сохранены в summary.json.', '',
'| Нормальный контроль slot-4 | Early | Late |', '|---|---:|---:|']
for label,key in [('Нормальных окон','normal_control_windows'),('Исходный отчёт отсутствует','normal_missing_original'),('Отчёт уже старше 250 мс при возврате','normal_reports_already_late_at_return'),('Отчёт старше 250 мс в момент решения','normal_reports_late_at_evaluation')]:
    lines.append(f"| {label} | {et[key]} | {lt[key]} |")
lines += ['', 'Штатные задержки 350 мс и реальные остановки наблюдателя сохранены. Получение отчёта позже устраняет его ожидание в сборщике, но не делает задержанный или отсутствующий отчёт достоверным.', '',
'## Сохранение контроля и предупреждений', '', '| Сценарий | Early: безопасность / эпизодов | Late: безопасность / эпизодов | Early: недоступность | Late: недоступность |', '|---|---:|---:|---:|---:|']
for scenario in ['C','F','R','D','S1','S2','S3','S4','G']:
    a=early['episodes']['CARRY']['scenarios'][scenario]; b=late['episodes']['CARRY']['scenarios'][scenario]
    lines.append(f"| {scenario} | {a['security']}/{a['n']} | {b['security']}/{b['n']} | {a['availability']} | {b['availability']} |")
lines += ['', 'C — изменение применённой конфигурации; F — подделка контрольного отчёта; R — повтор старого отчёта; D — подавление доставки. S1–S4 — изменение профиля NSSF в NRF, запрещённый discovery, неверный маршрут исследовательского потребителя и серия ошибочных запросов. G — рост нагрузки.', '',
'Тревога внутри эпизода не доказывает причинную связь с воздействием. Сохраняется заранее определённая метрика наличия сигнала в интервале; уведомление о недоступности не считается подтверждением атаки.', '',
f"Уведомление о недоступности уже действовало на предыдущем окне перед {early['preexisting_availability_episodes']} эпизодами early и {late['preexisting_availability_episodes']} эпизодами late. Тревога безопасности уже действовала перед {early['preexisting_security_episodes']} и {late['preexisting_security_episodes']} эпизодами соответственно. Состояния всех границ и отдельные случаи C без тревоги и D с тревогой сохранены в diagnostics.json.", '',
'| Режим CARRY | Эпизоды с тревогой безопасности /108 | Только недоступность | Без сигнала | Атакующие окна с тревогой /1410 | Ложные окна /344190 |', '|---|---:|---:|---:|---:|---:|']
for mode,v in [('early',early),('late',late)]:
    e=v['episodes']['CARRY']; m=v['metrics']['all']['CARRY']
    lines.append(f"| {mode} | {e['security']} | {e['operational']-e['security']} | {e['n']-e['operational']} | {m['security_tp']} | {m['security_fp']} ({100*m['security_fpr']:.4f}%) |")
lines += ['', '| Режим CARRY | Recall по атакующим окнам | Precision тревог безопасности | FPR нормальных окон |', '|---|---:|---:|---:|']
for mode,v in [('early',early),('late',late)]:
    m=v['metrics']['all']['CARRY']
    lines.append(f"| {mode} | {100*m['security_recall']:.4f}% | {100*m['security_precision']:.4f}% | {100*m['security_fpr']:.4f}% |")
lines += ['', 'Precision относится к заданной доле атак в данном тесте и меняется при другой распространённости атак. Для фиксированного бинарного решения CARRY полноценная PR-кривая и PR-AUC здесь не заявляются. Контрольная цель FPR из исходных рекомендаций составляет 0,1%; она оценивается отдельно от исправления свежести.']
lines += ['', '| Нормальные окна slot-4 | Early /10320 | Late /10320 |', '|---|---:|---:|']
for label,key in [('Unknown','unknown_normal'),('Недоступность','availability_normal_windows'),('Тревога безопасности','security_fp')]:
    lines.append(f"| {label} | {early['metrics']['slot-4']['CARRY'][key]} | {late['metrics']['slot-4']['CARRY'][key]} |")
lines += ['', '## Задержки сигналов', '', '| Сценарий | Режим | Эпизодов с сигналом безопасности | Медиана, с | p95, с |', '|---|---|---:|---:|---:|']
for scenario in ['C','F','R','G']:
    for mode in ['early','late']:
        delays=[e['security_delay'] for e in s['episodes'] if e['method']=='CARRY' and e['mode']==mode and e['scenario']==scenario and e['security']]
        if delays: lines.append(f"| {scenario} | {mode} | {len(delays)} | {np.median(delays):.4f} | {np.percentile(delays,95):.4f} |")
lines += ['', 'Пропуски не заменяются нулевыми задержками. Перенос запроса позволяет проверять свежий отчёт после обработки нагрузки; он не сокращает весь цикл сбора и не проверяет обнаружение кратковременного изменения, исчезающего до запроса. Воздействия на контроль здесь длятся десять окон.', '',
'## Оставшиеся ограничения нагрузки', '', '| Фон | Нормальных окон на режим | Ложные окна early | Ложные окна late |', '|---|---:|---:|---:|']
for regime in ['reference','higher_load','cyclic','bursty_recovery']:
    a,b=early['regimes'][regime],late['regimes'][regime]
    lines.append(f"| {regime} | {a['normal_windows']} | {a['security_fp']} | {b['security_fp']} |")
lines += ['', '| Вариант G | Early: безопасность /3 | Late: безопасность /3 |', '|---|---:|---:|']
for variant in ['fast','slow','intermittent','inside_permit']:
    lines.append(f"| {variant} | {early['episodes']['CARRY']['growth'][variant]['security']} | {late['episodes']['CARRY']['growth'][variant]['security']} |")
lines += ['', 'Порог 37 запросов, временной признак последовательных приращений и правила разрешений сохранены. Исправление свежести не является исправлением медленного роста или переносимости калибровки на высокий законный фон. Последующая разработка этих компонентов требует новых нормальных обучающих и калибровочных данных, затем отдельного независимого теста. Текущая серия после анализа также становится просмотренным диагностическим набором.', '',
'## Расходы и пределы вывода', '', '| Режим | Source seed | Длительность, с | Интервал p95, с | CPU сборщика, с | RSS сборщика, MiB |', '|---|---:|---:|---:|---:|---:|']
for r in sorted(s['run_details'],key=lambda r:(r['source_seed'],r['mode'])):
    o=r['overhead']; lines.append(f"| {r['mode']} | {r['source_seed']} | {o['elapsed_seconds']:.2f} | {r['interval_p95_s']:.4f} | {o['collector_cpu_seconds']:.2f} | {o['collector_peak_rss_kib']/1024:.1f} |")
lines += ['', 'CPU и память относятся только к сборщику. Разные runner ограничивают причинное сравнение расходов и задержек. Окна зависимы, исходных моделей три, поэтому тысячи окон не трактуются как независимые повторы для статистической значимости.', '',
'Это стенд с реальным NRF, штатной NSSF Open5GS, исследовательской NF и программными клиентами. Контрольный канал использует доверенные локальные процессы и IPC. Радиодоступ, UE/PDU, аппаратная аттестация и компрометация доверенных ОС/ключей не проверены. Результат относится к данной организации контроля, а не к предельной производительности или сертификации операторской сети.', '',
'## Материалы', '', '![Сравнение](results/comparison.png)', '',
'- [Все сводные метрики и эпизоды](results/summary.json)',
'- [Диагностика причин и совпадения входных данных](results/diagnostics.json)',
'- [Контрольные суммы и совпадение локального аудита с CI](results/archive_integrity.json)',
'- [Исходные тесты и сохранённая ошибка аудита](results/ci_audit_logs.json)',
'- [46 тестов и исправленный аудит CI](results/corrected_ci_audit_logs.json)',
'- [График PDF](results/comparison.pdf) · [SVG](results/comparison.svg)', '',
'В results сохранены также шесть отдельных `*-audit.json`. Исходные ZIP сохранены локально и в артефактах указанного запуска с периодом хранения 30 дней. Компактные сводки публикуются отдельными артефактами. Для повторного полного аудита нужен основной ZIP:', '',
'```text', 'python -m experiments.freshness_5g.audit_archive PATH_TO_ARCHIVE.zip',
'python -m experiments.freshness_5g.summarize ARCHIVE_DIRECTORY experiments/freshness_5g/results',
'python -m experiments.freshness_5g.diagnose ARCHIVE_DIRECTORY experiments/freshness_5g/results',
'python -m experiments.freshness_5g.plot_results experiments/freshness_5g/results',
'python -m experiments.freshness_5g.report', '```', '']
(root/'README.md').write_text('\n'.join(lines),encoding='utf-8')
print('Freshness report generated')
