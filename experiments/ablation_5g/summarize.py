"""Aggregate all preregistered replicates without selecting successful outcomes."""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

def run(source,out):
    reports=[json.loads(p.read_text()) for p in source.rglob('report.json')]
    assert len(reports)==3 and {r['seed'] for r in reports}=={64133,64233,64333}
    assert all(r['rows_per_method']==115200 for r in reports)
    methods=reports[0]['primary_methods']; totals=[]
    for method in [m['method'] for m in reports[0]['metrics']]:
        rows=[next(m for m in r['metrics'] if m['method']==method) for r in reports]
        d={k:sum(r[k] for r in rows) for k in ['tp','fp','fn','tn','n']}
        episodes=[e for r in reports for e in r['episodes'] if e['method']==method]
        assert len(episodes)==108
        detected=[e for e in episodes if e['detected']]
        delays=[e['delay_seconds'] for e in detected]
        d.update(method=method,primary=method in methods,recall=d['tp']/(d['tp']+d['fn']),
            fpr=d['fp']/(d['fp']+d['tn']),precision=d['tp']/(d['tp']+d['fp']) if d['tp']+d['fp'] else None,
            detected_episodes=len(detected),episodes=len(episodes),
            median_delay=float(np.median(delays)) if delays else None,
            p95_delay=float(np.percentile(delays,95)) if delays else None,
            per_run=[dict(seed=r['seed'],**next(m for m in r['metrics'] if m['method']==method)) for r in reports],
            scenarios={s:dict(detected=sum(e['detected'] for e in episodes if e['scenario']==s),
                total=sum(e['scenario']==s for e in episodes),availability=sum(e['availability'] for e in episodes if e['scenario']==s)) for s in sorted({e['scenario'] for e in episodes})})
        totals.append(d)
    out.mkdir(exist_ok=True,parents=True)
    result=dict(runs=3,rows_per_method=345600,metrics=totals,reports=reports)
    (out/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    lines=['# Абляционное сравнение на новых наблюдениях 5G','',
        'Три часовых прогона, 32 логические NF-позиции, 345600 одинаковых наблюдений на метод, 108 эпизодов. Модели и пороги заморожены до сбора данных.','',
        '| Метод | TP / атакующих окон | FP / нормальных окон | Recall | FPR | Precision | Эпизоды |',
        '|---|---:|---:|---:|---:|---:|---:|']
    totals.sort(key=lambda r: (methods.index(r['method']) if r['method'] in methods else len(methods),r['method']))
    for r in totals:
        if r['primary']:
            precision='—' if r['precision'] is None else f"{r['precision']:.2%}"
            lines.append(f"| {r['method']} | {r['tp']}/{r['tp']+r['fn']} | {r['fp']}/{r['fp']+r['tn']} | {r['recall']:.2%} | {r['fpr']:.4%} | {precision} | {r['detected_episodes']}/108 |")
    lines+=['','## Эпизоды по сценариям','', '| Метод | '+ ' | '.join(['S1','S2','S3','S4','C','F','R','D','G'])+' |', '|---|'+'---:|'*9]
    for r in totals:
        if r['primary']:lines.append('| '+r['method']+' | '+' | '.join(f"{r['scenarios'][s]['detected']}/{r['scenarios'][s]['total']}" for s in ['S1','S2','S3','S4','C','F','R','D','G'])+' |')
    carry=next(r for r in totals if r['method']=='CARRY')
    lines+=['','Таблица учитывает наличие тревоги безопасности внутри размеченного эпизода, а не доказанную причинную связь каждой тревоги с воздействием. Уведомления о недоступности D не засчитываются как обнаружение компрометации.',
        f"Для D зарегистрированы тревоги безопасности в {carry['scenarios']['D']['detected']}/12 эпизодов, а уведомления о недоступности — в {carry['scenarios']['D']['availability']}/12. Проверка причин сопутствующих тревог сохранена отдельно в D_incidental_security.json; наличие тревоги в D нельзя трактовать как специфическое обнаружение подавления доставки.",'',
        '## Разброс и задержки','',
        '| Метод | FPR по трём прогонам, % | Медиана задержки, с | p95 задержки, с |',
        '|---|---|---:|---:|']
    for r in totals:
        if r['primary']:
            median='—' if r['median_delay'] is None else f"{r['median_delay']:.3f}"
            p95='—' if r['p95_delay'] is None else f"{r['p95_delay']:.3f}"
            lines.append(f"| {r['method']} | "+', '.join(f"{p['fpr']*100:.4f}" for p in sorted(r['per_run'],key=lambda p:p['seed']))+f" | {median} | {p95} |")
    lines+=['','Задержки рассчитаны только для эпизодов с тревогой; это не оценка задержки для пропущенных воздействий.','',
        '## Парные расхождения на одинаковых окнах','',
        '| Левый вариант → правый | Окна | Тревога только слева | Тревога только справа |',
        '|---|---|---:|---:|']
    for pair in reports[0]['paired_discordance']:
        rows=[p for r in reports for p in r['paired_discordance'] if (p['left'],p['right'],p['attack'])==(pair['left'],pair['right'],pair['attack'])]
        assert len(rows)==3
        lines.append(f"| {pair['left']} → {pair['right']} | {'Атака' if pair['attack'] else 'Норма'} | {sum(p['left_only'] for p in rows)} | {sum(p['right_only'] for p in rows)} |")
    lines+=['','Дополнительные тревоги справа на атакующих окнах означают прирост обнаружения, на нормальных — рост ложных тревог. Пара CARRY_without_witness → CARRY сохраняет остальные компоненты детектора и выделяет вклад живого контрольного канала.','',
        '## Ограничения интерпретации','',
        'Четыре основных варианта различаются не только признаками, но и алгоритмом. Их разность оценивает системы целиком. В summary.json сохранены вспомогательные варианты и парные расхождения для анализа вклада компонентов.',
        'Доверительные интервалы по независимым окнам не строятся: окна внутри прогона зависимы. Три прогона дают ограниченную оценку переносимости. Задержки условны по обнаруженным эпизодам; пропуски сохранены.',
        'AP непрерывных IF-оценок и бинарного CARRY имеют разные шкалы; сравнение качества ранжирования между ними не проводится.',
        'Это Open5GS и исследовательские NF, без полного RAN/UE/PDU и аппаратной аттестации. Успех CI означает воспроизводимость, а не положительный научный результат.']
    (out/'REPORT_RU.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps([r for r in totals if r['primary']],indent=2))

if __name__=='__main__':run(Path(sys.argv[1]),Path(sys.argv[2]))
