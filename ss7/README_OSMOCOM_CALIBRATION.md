# Калибровка SS7/SIGTRAN по Osmocom

Контур выполняет независимую **software-in-the-loop** калибровку транспортной
части canonical SS7-модели без физического стенда.

Главный принцип: conformance-трафик Osmocom не используется как суррогат
операторской нагрузки. Вместо этого поверх реального M3UA/SCTP стека
генерируется контролируемая серия M3UA DATA сообщений с известными размерами
payload. По ней оценивается физически интерпретируемый параметр:
**добавочный объём M3UA/SCTP/IPv4 на одно сигнальное сообщение**.

## GitHub Actions

Workflow .github/workflows/ss7-osmocom-calibration.yml:

1. фиксирует commit osmocom/docker-playground;
2. строит и запускает OsmoSTP и официальный nplab/m3ua-testtool;
3. выполняет официальный M3UA SGP suite как проверку работоспособности стека;
4. не останавливая testbed и packet capture, запускает ss7/osmocom_m3ua_load.scm;
5. генератор отправляет по 30 M3UA DATA сообщений для payload:
   24, 40, 56, 72, 92, 152, 192, 256 байт;
6. для калибровочных сообщений используется отдельный Service Indicator SI=15,
   поэтому они однозначно отделяются от conformance traffic;
7. tshark извлекает frame.len, ip.len, m3ua.message_length, OPC/DPC/SI и SCTP ports;
8. ss7/osmocom_calibration.py оценивает transport overhead;
9. часть размеров payload используется только как holdout;
10. PCAP, CSV, профиль, holdout-таблица, логи и provenance сохраняются как artifact.

## Fit / holdout

По умолчанию fit использует payload 24, 56, 92, 152 байт.
Holdout использует 40, 72, 192, 256 байт.

Разделение выполняется по размерам payload, а не случайным разбиением кадров.
Holdout проверяет переносимость найденного encapsulation overhead на размеры
сообщений, не использованные при оценке параметра.

Физически ограниченная модель:

IP_bytes = align4(payload_bytes) + transport_overhead_bytes.

Дополнительно отдельно оцениваются M3UA overhead, SCTP+IPv4 overhead, holdout MAE,
максимальная абсолютная ошибка и медианная относительная ошибка.

## Интеграция с canonical SS7-моделью

В ss7/fix.py добавлен параметр transport_overhead_bytes.
При значении 0 поведение модели остаётся прежним. После калибровки
calibrated_template() устанавливает измеренный overhead, и он учитывается
только в расчёте битовой нагрузки linkset и очередей.

Количество транзакций и частоты сообщений не масштабируются; attack priors и
распределение MAP/TCAP/ISUP операций не подгоняются.

Это принципиально отличается от раннего варианта flow_scale: произвольная
частота conformance suite больше не используется для калибровки нагрузки.

## Граница научного утверждения

> Транспортная составляющая имитационной модели SS7/SIGTRAN откалибрована по
> контролируемым M3UA DATA сообщениям, сформированным независимой программной
> реализацией Osmocom. Параметр encapsulation overhead оценён на одной группе
> размеров сообщений и проверен на отложенной группе размеров.

Нельзя утверждать, что этим экспериментом откалибрована операторская нагрузка
или распределения MAP/TCAP/ISUP.

## Артефакты

- osmocom/osmocom-m3ua.pcap.gz — исходный packet capture;
- osmocom/osmocom-m3ua.csv — полный M3UA/SCTP trace;
- osmocom/osmocom-m3ua-data.csv — контролируемые M3UA DATA;
- calibration/calibrated_profile.json;
- calibration/controlled_data_trace.csv;
- calibration/payload_size_summary.csv;
- calibration/holdout_validation.csv;
- calibration/summary.md;
- osmocom/logs/.../calibration-load.log;
- provenance.txt.
