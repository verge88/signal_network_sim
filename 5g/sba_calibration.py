"""
sba_calibration.py
==================
Калибровка симулятора по реальным SBI-трассам (Open5GS / free5GC + UERANSIM)
и анализ чувствительности результатов к параметрам.

Сбор трасс (вне Python; только в собственном изолированном стенде):
  1. развернуть Open5GS или free5GC, подключить UERANSIM, прогнать сценарии
     registration / PDU session establishment / handover / deregistration;
  2. снять трафик SBI:  tshark -i lo -f 'tcp port 7777' -w sbi.pcapng
  3. экспортировать поля HTTP/2 в CSV:
     tshark -r sbi.pcapng -Y http2 -T fields \
       -e frame.time_epoch -e tcp.stream -e http2.streamid -e http2.type \
       -e http2.headers.method -e http2.headers.path -e http2.length \
       -e http2.settings.max_concurrent_streams -E separator=, > sbi.csv

Типы фреймов HTTP/2 (RFC 9113 §6): 0 DATA, 1 HEADERS, 3 RST_STREAM,
4 SETTINGS, 6 PING, 7 GOAWAY, 8 WINDOW_UPDATE.
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, replace
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from sba_sim_v1 import SbaConfig, SbaSimulator

TRACE_COLUMNS = ["t", "tcp_stream", "http2_streamid", "http2_type",
                 "method", "path", "length", "max_concurrent"]


def load_trace(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, header=None, names=TRACE_COLUMNS, low_memory=False)
    df["t"] = pd.to_numeric(df["t"], errors="coerce")
    for c in ("tcp_stream", "http2_streamid", "length",
              "max_concurrent", "http2_type"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["t"]).sort_values("t").reset_index(drop=True)


def estimate_config(trace: pd.DataFrame, base: Optional[SbaConfig] = None
                    ) -> Tuple[SbaConfig, Dict[str, float]]:
    """Оценивает параметры транспорта и нагрузки из реальной трассы."""
    cfg = base or SbaConfig()
    headers = trace[trace["http2_type"] == 1]
    rst = trace[trace["http2_type"] == 3]
    goaway = trace[trace["http2_type"] == 7]
    winupd = trace[trace["http2_type"] == 8]

    duration = float(trace["t"].max() - trace["t"].min()) or 1.0
    client_hdrs = headers[headers["http2_streamid"] % 2 == 1]
    n_req = int(client_hdrs["http2_streamid"].nunique())
    rate = n_req / duration

    mcs = trace["max_concurrent"].dropna()
    max_conc = int(mcs.median()) if len(mcs) else cfg.max_concurrent_streams

    hdr_len = headers["length"].dropna()
    first = float(hdr_len.quantile(0.95)) if len(hdr_len) \
        else cfg.hpack_first_header_bytes
    typical = float(hdr_len.median()) if len(hdr_len) \
        else cfg.hpack_indexed_header_bytes

    # latency: парные HEADERS запрос→ответ по (tcp_stream, streamid)
    lat_list = []
    for _, grp in headers.groupby(["tcp_stream", "http2_streamid"]):
        if len(grp) >= 2:
            lat_list.append((grp["t"].iloc[1] - grp["t"].iloc[0]) * 1000.0)
    lat = np.array([x for x in lat_list if 0 < x < 60000.0])
    if lat.size > 20:
        lmu, lsd = float(np.mean(np.log(lat))), float(np.std(np.log(lat)))
    else:
        lmu, lsd = cfg.latency_mu, cfg.latency_sigma

    # сверхдисперсия потока по односекундным интервалам
    bins = np.arange(trace["t"].min(), trace["t"].max() + 1.0, 1.0)
    counts, _ = np.histogram(client_hdrs["t"], bins=bins) if bins.size > 1 \
        else (np.array([0]), None)
    if counts.size > 3 and counts.mean() > 0:
        disp = float(max(0.0, counts.var() / counts.mean() - 1.0)
                     / max(1e-9, counts.mean()))
    else:
        disp = cfg.rate_dispersion

    est = dict(
        base_rate_hz=float(rate),
        max_concurrent_streams=int(max_conc),
        hpack_first_header_bytes=int(first),
        hpack_indexed_header_bytes=int(max(20.0, typical)),
        rst_stream_rate=float(len(rst) / max(1, n_req)),
        goaway_rate_per_window=float(len(goaway)
                                     / max(1.0, duration / 3600.0)),
        latency_mu=lmu, latency_sigma=lsd,
        rate_dispersion=float(np.clip(disp, 0.01, 3.0)),
        n_connections=int(trace["tcp_stream"].nunique() or 1),
    )
    diag = dict(duration_s=duration, n_requests=n_req,
                n_window_updates=int(len(winupd)), observed_rate_hz=float(rate))
    return replace(cfg, **est), diag


def _ks_statistic(a: np.ndarray, b: np.ndarray) -> float:
    a = np.sort(np.asarray(a, float))
    b = np.sort(np.asarray(b, float))
    if a.size == 0 or b.size == 0:
        return float("nan")
    grid = np.union1d(a, b)
    fa = np.searchsorted(a, grid, side="right") / a.size
    fb = np.searchsorted(b, grid, side="right") / b.size
    return float(np.max(np.abs(fa - fb)))


def compare_marginals(trace: pd.DataFrame, cfg: SbaConfig, seed: int = 0,
                      n_windows: int = 20) -> pd.DataFrame:
    """Таблица «реальное vs синтетическое» по маргинальным распределениям."""
    sim = SbaSimulator(cfg, seed=seed)
    syn_req, syn_bytes = [], []
    for w in range(n_windows):
        cons = sim.topology[w % len(sim.topology)]
        rec = sim.simulate_window(w, cons)
        r = rec.honest_report
        syn_req.append(r.series_requests)
        syn_bytes.append(r.series_bytes / np.maximum(1.0, r.series_requests))
    syn_req = np.concatenate(syn_req)
    syn_bytes = np.concatenate(syn_bytes)
    syn_bytes = syn_bytes[syn_bytes > 0]

    headers = trace[trace["http2_type"] == 1]
    real_len = headers["length"].dropna().to_numpy(float)
    bins = np.arange(trace["t"].min(), trace["t"].max() + 60.0, 60.0)
    real_req, _ = np.histogram(headers["t"], bins=bins)

    def q(x):
        x = np.asarray(x, float)
        return ([float(np.quantile(x, p)) for p in (0.1, 0.5, 0.9)]
                if x.size else [np.nan] * 3)

    rows = []
    for name, real, syn in (("requests_per_minute", real_req, syn_req),
                            ("bytes_per_request", real_len, syn_bytes)):
        rq, sq = q(real), q(syn)
        rows.append(dict(quantity=name,
                         real_p10=rq[0], real_p50=rq[1], real_p90=rq[2],
                         syn_p10=sq[0], syn_p50=sq[1], syn_p90=sq[2],
                         ks_statistic=_ks_statistic(real, syn)))
    return pd.DataFrame(rows)


def sensitivity_analysis(base: SbaConfig, seed: int = 0,
                         factors: Tuple[float, ...] = (0.5, 0.75, 1.0, 1.5, 2.0)
                         ) -> pd.DataFrame:
    """Как меняются итоговые метрики при отклонении ключевых параметров.

    Проверяется главное: воспроизводит ли NoiseModel.sigma() эмпирическую
    дисперсию Δ при H = 0, и держится ли эмпирическая кривая обнаружимости
    рядом с аналитической мощностью. Расхождение > 0.15 означает, что модель
    шума не описывает симулятор.
    """
    from sba_eval import (DatasetSpec, detectability_curve, evaluate_invariant,
                          generate_dataset)

    knobs = ["base_rate_hz", "max_concurrent_streams", "clock_skew_ms",
             "telemetry_export_interval_s", "rate_dispersion",
             "goaway_rate_per_window"]
    spec = DatasetSpec(n_clean_windows=150, n_benign_windows=100,
                       n_attack_windows_per_type=25)
    rows = []
    for knob in knobs:
        v0 = getattr(base, knob)
        for f in factors:
            newv = int(round(v0 * f)) if isinstance(v0, int) else float(v0 * f)
            if isinstance(v0, int):
                newv = max(1, newv)
            cfg = replace(base, **{knob: newv})
            try:
                df = generate_dataset(cfg, spec, seed=seed)
                tbl, det = evaluate_invariant(df)
                curve = detectability_curve(df, det)
            except Exception as exc:                          # noqa: BLE001
                rows.append(dict(knob=knob, factor=f, value=newv,
                                 error=str(exc)))
                continue

            # эмпирическая дисперсия Δ на чистых окнах vs предсказание модели
            clean = df[df["label"] == 0]
            emp_sigma = float(1.4826 * np.median(
                np.abs(clean["inv_delta_core"]
                       - clean["inv_delta_core"].median())))
            model_sigma = float(clean["inv_sigma"].mean())
            fam = tbl[tbl["group"].str.startswith("family:")]

            rows.append(dict(
                knob=knob, factor=f, value=newv,
                mean_recall=float(np.nanmean(fam["recall_combined_1pct"])),
                h_min=float(curve["mean_h_min"].mean()),
                sigma_model=model_sigma,
                sigma_empirical=emp_sigma,
                sigma_ratio=emp_sigma / max(model_sigma, 1e-9),
                mean_abs_theory_error=float(np.nanmean(np.abs(
                    curve["empirical_recall"] - curve["analytic_power"]))),
            ))
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", help="CSV, экспортированный tshark")
    ap.add_argument("--out", default="calibration_sba")
    ap.add_argument("--sensitivity", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    cfg = SbaConfig()
    if a.trace:
        tr = load_trace(a.trace)
        cfg, diag = estimate_config(tr, cfg)
        with open(os.path.join(a.out, "calibrated_config.json"), "w",
                  encoding="utf-8") as fh:
            json.dump(dict(config=asdict(cfg), diagnostics=diag), fh, indent=2)
        cmp_tbl = compare_marginals(tr, cfg)
        cmp_tbl.to_csv(os.path.join(a.out, "marginal_comparison.csv"),
                       index=False)
        print(cmp_tbl.to_string(index=False))

    if a.sensitivity:
        sens = sensitivity_analysis(cfg)
        sens.to_csv(os.path.join(a.out, "sensitivity.csv"), index=False)
        print(sens.to_string(index=False))


if __name__ == "__main__":
    main()
