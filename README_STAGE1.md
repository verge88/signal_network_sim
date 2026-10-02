# Этап 1 — научная корректность

Реализация предназначена для текущей структуры `signal_network_sim`.

## Что делает

- закрепляет `ss7/fix.py` как научный SS7 source-of-truth через `ss7/canonical.py`;
- исправляет ошибку `compromised_since == 0`;
- убирает прямые compromise-only RTT/integrity fingerprints из Diameter и SIP;
- заменяет Diameter low-variance/+0.8 masking на benign-history/counterfactual masking;
- сохраняет SIP независимый транзитный `200 OK / ACK` invariant;
- исправляет Stage-0 baseline на Windows: child Python и чтение stdout идут в UTF-8.

## Установка

Распакуйте архив в корень репозитория, объединив каталоги.

Примените патч:

```powershell
python .\tools\apply_stage1.py --apply
```

Проверка:

```powershell
python .\tools\apply_stage1.py --check
python -m unittest tests.test_stage1_scientific_contract
python -m unittest tests.test_ss7_regressions
```

Синтаксис:

```powershell
python -m py_compile .\diameter\diameter_sim_v1.py
python -m py_compile .\sip\sip_sim_v4_claude.py
python -m py_compile .\ss7\canonical.py
```

## Повтор SS7 baseline

```powershell
python -m experiments.baseline run --protocol ss7 --seeds 42
```

После успешного smoke-test:

```powershell
python -m experiments.baseline run --protocol ss7 --regen
```

Последний ваш запуск уже показал корректный SS7 null-world до ошибки консоли:
AUC=0.500 для SP и STP. Падение вызвала только Windows cp1251 при печати `→`.

## Важно

После удаления shortcuts метрики Diameter/SIP могут снизиться. Это ожидаемо:
если классификатор ранее использовал `+0.8`, `fail_prob=0.30` или специальную
задержку как подпись класса, такой результат не должен сохраняться.

Этап 1 удаляет известные прямые shortcuts, но ещё не доказывает null-world для
Diameter/SIP. Это будет задачей этапа 2.

Перед изменением файлов патчер сохраняет исходники в `.stage1_backup/`.
