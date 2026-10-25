# Калибровка SS7-модели по Osmocom

Этот контур предназначен для независимой **software-in-the-loop** проверки
транспортной части SS7/SIGTRAN-модели без физического стенда.

## Что запускается в GitHub Actions

Workflow \`.github/workflows/ss7-osmocom-calibration.yml\`:

1. фиксирует конкретный commit \`osmocom/docker-playground\`;
2. запускает официальный набор \`nplab-m3ua-test\` против \`OsmoSTP\`;
3. сохраняет PCAP SCTP/M3UA обмена между независимыми программными
   реализациями;
4. нормализует трассу через \`tshark\`;
5. первые 60% трассы использует только для оценки параметров;
6. последние 40% оставляет как временной holdout;
7. запускает \`ss7/osmocom_calibration.py\`;
8. публикует PCAP, CSV, профиль калибровки, holdout-сравнение, логи и
   provenance как GitHub Actions artifact.

Workflow можно запустить вручную во вкладке **Actions → SS7 Osmocom
calibration → Run workflow**.

## Что именно калибруется

Из M3UA/SCTP-трассы измеряются частота кадров, размеры кадров, интервалы между
кадрами и доля управляющего обмена. Для интенсивности используется
\`M3UA DATA\`, если таких кадров достаточно; иначе используется явно
помеченный fallback по всем M3UA/SCTP кадрам.

Калибровщик оценивает:

- множитель интенсивности \`flow_scale\` относительно canonical SS7
  source-of-truth \`ss7/fix.py\`;
- рекомендованную пропускную способность **эталонного linkset** при заданном
  \`target_rho\`;
- согласие распределения интенсивности на отложенной части трассы
  (квантили, KS-статистика и относительная ошибка медианы).

Функция \`calibrated_template()\` возвращает \`Ss7Sim\` с применённым
\`flow_scale\`. Такой template можно передавать в существующий
\`build_dataset(..., sim_template=..., topo=..., size_capacities=False)\`.

## Что намеренно не калибруется

M3UA conformance-тесты не являются операторской нагрузкой. Поэтому по ним
**не оцениваются** распределения MAP/TCAP/ISUP операций, параметры атак,
суточный профиль, AR-динамика, шум отчётности и модель ключей/целостности.

Корректная формулировка для диссертации:

> Выполнена независимая software-in-the-loop калибровка транспортных
> характеристик модели SS7/SIGTRAN по трассам, сформированным сторонней
> реализацией Osmocom OsmoSTP/M3UA. Калибровочная и проверочная части трассы
> разделены во времени. Эксперимент не интерпретируется как валидация на
> операторском трафике.

Это сильнее полностью внутренней проверки генератора, но не заменяет
эксплуатационную валидацию на данных оператора.

## Артефакты

После прогона workflow формируются:

- \`artifacts/osmocom/osmocom-m3ua.pcap.gz\`;
- \`artifacts/osmocom/osmocom-m3ua.csv\`;
- \`artifacts/calibration/calibrated_profile.json\`;
- \`artifacts/calibration/calibration_trace.csv\`;
- \`artifacts/calibration/holdout_trace.csv\`;
- \`artifacts/calibration/holdout_comparison.csv\`;
- \`artifacts/calibration/summary.md\`;
- \`artifacts/provenance.txt\`.
