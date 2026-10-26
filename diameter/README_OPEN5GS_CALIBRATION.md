# Калибровка Diameter S6a по Open5GS

Этот контур выполняет независимую **software-in-the-loop** калибровку
транспортной части Diameter-модели на реальной сторонней реализации EPC:
Open5GS MME ↔ HSS по S6a/freeDiameter.

## Что именно калибруется

В исходной модели нагрузка Diameter-ссылки рассчитывается через единый
placeholder `avg_message_size_bytes = 500`.

При этом S6a-команды в симуляторе представлены как транзакции AIR, ULR, CLR,
IDR, DSR, PUR, RSR и NOR. Поэтому калибровщик измеряет:

`request Diameter bytes + answer Diameter bytes`

на одну полную S6a-транзакцию.

Если для конкретной команды в fit-части достаточно наблюдений, сохраняется
отдельный средний размер транзакции по этой команде. Для редких команд
используется общий S6a fallback.

Не калибруются по тестовой нагрузке Open5GS:

- интенсивность трафика;
- time-of-day профиль;
- распределение S6a-команд;
- параметры атак;
- masking/compromise;
- топология и пропускная способность;
- Gx/Rx/S13 и прочий non-S6a traffic;
- AR-динамика.

## Почему не используется frame.len или ip.len как целевой параметр

Diameter работает поверх TCP/SCTP, и крупное сообщение может быть
реассемблировано Wireshark из нескольких транспортных сегментов. Поэтому
`ip.len` кадра, в котором отображается reassembled Diameter PDU, не является
полным wire-size сообщения.

Калибровка использует поле `diameter.length`, которое соответствует размеру
самого Diameter PDU. Длины IP-кадров сохраняются в исходной трассе как
диагностические данные, но не подставляются в модель.

## Fit / holdout

Open5GS запускает официальный EPC integration suite:

`./build/tests/attach/attach`

В него входят реальные MME/HSS Diameter обмены во время LTE attach/auth/etc.
Трафик снимается на loopback, после чего tshark оставляет только:

`diameter.applicationId == 16777251`

Request и answer связываются по:

- Command-Code;
- Hop-by-Hop Identifier;
- End-to-End Identifier.

Транзакции сортируются по времени. Первые 60% используются для fit, последние
40% — только для holdout-проверки.

Проверяются:

- общая средняя длина S6a-транзакции;
- средние размеры по отдельным командам;
- holdout MAE в байтах;
- медианная и p90 относительная ошибка;
- request-answer RTT как диагностическая величина.

RTT **не переносится** в `propagation_delay_ms`, потому что включает время
обработки MME/HSS и обращения к БД.

## Интеграция с моделью

В `SimulationConfig` добавлены:

- `s6a_transaction_size_bytes`;
- `s6a_command_transaction_size_bytes`.

Если они не заданы, симулятор сохраняет прежнее поведение с 500 байт на
транзакцию.

После применения профиля изменяется только расчёт объёма S6a-трафика на link.
Количество транзакций, command mix и сценарии атак остаются прежними.

## Граница научного утверждения

Корректная формулировка:

> Размерная составляющая S6a-трафика имитационной Diameter-модели
> откалибрована по полным request/answer транзакциям независимой реализации
> Open5GS MME/HSS. Оценка выполнена на хронологической fit-части трассы и
> проверена на отложенной holdout-части.

Некорректно называть этот эксперимент калибровкой на операторском трафике или
утверждать, что Open5GS-тесты подтверждают реальное распределение нагрузки
оператора.

## GitHub Actions artifact

Workflow сохраняет:

- `open5gs/open5gs-epc.pcap`;
- `open5gs/s6a-frames.csv`;
- `open5gs/attach-test.log`;
- `open5gs/tshark-protocol-hierarchy.txt`;
- `calibration/calibrated_profile.json`;
- `calibration/s6a_transactions.csv`;
- `calibration/fit_transactions.csv`;
- `calibration/holdout_validation.csv`;
- `calibration/command_fit_summary.csv`;
- `calibration/summary.md`;
- `provenance.txt`.
