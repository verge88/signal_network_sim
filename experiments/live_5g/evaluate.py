"""Frozen, run-disjoint evaluation of measured Open5GS SBI observations.

Labels are joined only for evaluation. No scenario identifier, attack label, NF ID,
run identifier, or timestamp enters a fitted detector. Constant evidence features
are checked by explicit rules; an Isolation Forest is never claimed to learn them.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import average_precision_score
from pyod.models.ecod import ECOD
from pyod.models.knn import KNN
from isotree import IsolationForest as ExtendedIsolationForest
from dvmad import DVMAD
from aderh import ADERH

KEYS = ["run_id", "window_id", "nf_id"]
META = KEYS + ["timestamp"]
SBI = ["request_count", "error_rate", "update_count", "interarrival_mean",
       "request_count_10s", "request_count_60s", "error_rate_10s", "error_rate_60s",
       "latency_ms", "heartbeat_age"]
CONTEXT = SBI + ["profile_changes", "forbidden_operation", "profile_policy_violation"]
FULL = CONTEXT + ["route_mismatch", "report_mismatch", "auth_failure"]
METHODS = ["rules", "IF_SBI", "IF_CONTEXT", "IF_FULL", "hybrid_without_KIE", "hybrid"]
TAIL_METHODS = ["robust_tail", "hybrid_tail_without_KIE", "hybrid_tail"]
TAIL_FEATURES = ["request_count", "error_rate", "update_count"]
MODEL_FEATURES = {"IF_SBI": SBI, "IF_CONTEXT": CONTEXT, "IF_FULL": FULL}
PROTOCOL_VERSION = "live-5g-v2-tail"
DETECTOR_CHOICES = ("iforest", "ocsvm_rbf", "lof", "elliptic", "eif", "knn", "ecod", "dvmad", "aderh")
DIRECT_ANOMALY_SCORE_DETECTORS = {"eif", "knn", "ecod", "dvmad", "aderh"}


def build_detector(detector, seed, train_rows):
    """Build an a-priori detector. Held-out attack data never selects parameters."""
    if detector == "iforest":
        return IsolationForest(n_estimators=256, max_samples=min(256, train_rows),
                               contamination="auto", random_state=seed, n_jobs=1)
    if detector == "ocsvm_rbf":
        return make_pipeline(StandardScaler(),
                             OneClassSVM(kernel="rbf", gamma="scale", nu=0.01, cache_size=1000))
    if detector == "lof":
        return make_pipeline(StandardScaler(),
                             LocalOutlierFactor(n_neighbors=35, novelty=True,
                                                contamination="auto", n_jobs=1))
    if detector == "elliptic":
        return make_pipeline(StandardScaler(),
                             EllipticEnvelope(contamination=0.1, support_fraction=None,
                                              random_state=seed))
    if detector == "eif":
        # Reference-style Extended Isolation Forest: hyperplane splits (ndim=2)
        # on standardized numeric inputs. Thresholding is still done solely on
        # the separate normal calibration split below.
        return make_pipeline(
            StandardScaler(),
            ExtendedIsolationForest(
                ntrees=256, sample_size=min(256, train_rows), ndim=2, max_depth=8,
                missing_action="fail", coefs="uniform", standardize_data=False,
                scoring_metric="depth", random_seed=seed, nthreads=1))
    if detector == "knn":
        # k-th-nearest-neighbor distance; larger means more anomalous.
        return make_pipeline(
            StandardScaler(),
            KNN(contamination=0.1, n_neighbors=20, method="largest", n_jobs=1))
    if detector == "ecod":
        # ECOD is parameter-light and rank/tail based. Its built-in contamination
        # threshold is ignored; our held-out normal calibration sets the alarm cut.
        return ECOD(contamination=0.1, n_jobs=1)
    if detector == "dvmad":
        # Fixed, tuning-free one-class discriminant configuration from the ICML
        # reference implementation. Scaling is fixed a-priori because its score
        # uses Euclidean nearest-neighbour distance in the learned subspace.
        return make_pipeline(
            StandardScaler(),
            DVMAD(contamination=0.1, mode="both", eps=0.1,
                  artificial_mode="max", use_faiss=False))
    if detector == "aderh":
        # The reference benchmark uses [0,1] feature scaling. The built-in
        # contamination threshold is ignored in favor of our normal calibration.
        return make_pipeline(
            MinMaxScaler(),
            ADERH(n_estimators=256, n=18, contamination=0.1,
                  random_state=seed))
    raise ValueError(f"Unknown detector {detector!r}; choose one of {DETECTOR_CHOICES}")


EPISODE_COLUMNS = ["run_id", "nf_id", "episode_id", "scenario", "method", "onset", "detected",
                   "first_alarm_at", "delay_seconds", "observed_until", "timing_source"]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def json_safe_parameters(value):
    """Convert sklearn parameter trees to stable JSON-safe metadata."""
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): json_safe_parameters(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe_parameters(item) for item in value]
    if hasattr(value, "get_params"):
        cls = value.__class__
        return {"estimator_class": f"{cls.__module__}.{cls.__name__}"}
    cls = value.__class__
    return {"python_class": f"{cls.__module__}.{cls.__name__}", "repr": repr(value)}


def anomaly_scores(model, values, detector):
    """Return scores with one convention: larger means more anomalous."""
    if detector in DIRECT_ANOMALY_SCORE_DETECTORS:
        return np.asarray(model.decision_function(values), dtype=float)
    return -np.asarray(model.score_samples(values), dtype=float)


def conformal_threshold(scores, alpha=0.001):
    """Finite-sample upper order statistic; ties are never alarms.

    The nominal coverage needs exchangeability. Correlated NF/time windows do not
    justify interpreting this rank choice as an operational error guarantee.
    """
    values = np.asarray(scores, dtype=float)
    if not 0 < alpha < 1 or len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("Invalid calibration scores or alpha")
    rank = math.ceil((len(values) + 1) * (1 - alpha))
    return float(np.sort(values)[rank - 1]) if rank <= len(values) else math.inf


def validate_features(frame):
    missing = set(META + FULL) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing feature columns: {sorted(missing)}")
    if frame.empty or frame[META].isna().any().any():
        raise ValueError("Empty features or missing observation identity")
    if frame.duplicated(KEYS).any():
        raise ValueError("Duplicate NF-window identities")
    if not np.isfinite(frame[FULL].to_numpy(dtype=float)).all():
        raise ValueError("Nonfinite measured features; do not impute evidence failures as normal")
    if not np.isfinite(frame.timestamp.to_numpy(dtype=float)).all():
        raise ValueError("Nonfinite observation timestamp")


def read_dataset(directory, labels_required=True):
    root = Path(directory)
    paths = [root / "features.csv"] if (root / "features.csv").is_file() else sorted(root.rglob("features.csv"))
    if not paths:
        raise ValueError(f"No features.csv beneath {root}")
    frames, labels, provenance = [], [], []
    for path in paths:
        frame = pd.read_csv(path, dtype={"run_id": str, "nf_id": str})
        validate_features(frame)
        frames.append(frame)
        provenance.append({"path": str(path), "sha256": sha256(path), "rows": len(frame)})
        lp = path.with_name("labels.csv")
        if labels_required:
            lab = pd.read_csv(lp, dtype={"run_id": str, "nf_id": str, "episode_id": str})
            required = KEYS + ["attack", "scenario", "onset", "episode_id"]
            if set(required) - set(lab.columns) or lab.duplicated(KEYS).any():
                raise ValueError("Missing label columns or duplicate label identities")
            if not lab.attack.isin([0, 1, False, True]).all():
                raise ValueError("Attack labels must be 0 or 1")
            if lab.loc[lab.attack.astype(bool), ["onset", "episode_id"]].isna().any().any():
                raise ValueError("Attack episodes require measured onset and episode identity")
            labels.append(lab)
            provenance.append({"path": str(lp), "sha256": sha256(lp), "rows": len(lab)})
    frame = pd.concat(frames, ignore_index=True)
    validate_features(frame)
    lab = pd.concat(labels, ignore_index=True) if labels else None
    if lab is not None:
        joined = frame[KEYS].merge(lab, on=KEYS, how="outer", validate="one_to_one", indicator=True)
        if not joined._merge.eq("both").all():
            raise ValueError("Feature and label identities must match exactly")
    return frame, lab, provenance


def require_disjoint(*frames):
    seen = set()
    for frame in frames:
        runs = set(frame.run_id.astype(str))
        if seen & runs:
            raise ValueError(f"Overlapping acquisition runs: {sorted(seen & runs)}")
        seen |= runs


def fit_tail_model(train):
    """Learn a one-sided current-window normal envelope, without attack labels.

    Counts have one-event resolution. The error-fraction floor is the fraction
    represented by one error at the largest training-window request count.
    These floors prevent division by zero; the alarm threshold is independently
    calibrated, never set to an attack amplitude.
    """
    values = train[TAIL_FEATURES].to_numpy(dtype=float)
    center = np.median(values, axis=0)
    mad = np.median(np.abs(values - center), axis=0)
    resolution = np.array([1., 1. / max(1., float(train.request_count.max())), 1.])
    scale = np.maximum(1.4826 * mad, resolution)
    return {"features": TAIL_FEATURES, "center": center.tolist(),
            "scale": scale.tolist(), "resolution_floor": resolution.tolist(),
            "score": "max(0, max_j((x_j - training_median_j) / scale_j))"}


def tail_scores(model, frame):
    values = frame[model["features"]].to_numpy(dtype=float)
    return np.maximum(0., np.max((values - np.asarray(model["center"])) /
                                 np.asarray(model["scale"]), axis=1))


def fit(train_directory, cal_directory, output, alpha=0.001, seed=7143, detector="iforest"):
    train, train_labels, train_files = read_dataset(train_directory)
    cal, cal_labels, cal_files = read_dataset(cal_directory)
    require_disjoint(train, cal)
    if train_labels.attack.any() or cal_labels.attack.any():
        raise ValueError("Training and threshold calibration must contain only normal observations")
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "manifest.json").exists():
        raise ValueError("Frozen model already exists; choose a new output directory")
    if detector not in DETECTOR_CHOICES:
        raise ValueError(f"Unsupported detector: {detector}")
    bundle = {"protocol_version": PROTOCOL_VERSION, "models": {}, "thresholds": {}, "alpha": alpha,
              "detector": detector}
    for name, columns in MODEL_FEATURES.items():
        model = build_detector(detector, seed, len(train))
        model.fit(train[columns].to_numpy(dtype=float))
        calibration_scores = anomaly_scores(model, cal[columns].to_numpy(dtype=float), detector)
        bundle["models"][name] = model
        bundle["thresholds"][name] = conformal_threshold(calibration_scores, alpha)
    bundle["tail_model"] = fit_tail_model(train)
    bundle["thresholds"]["robust_tail"] = conformal_threshold(tail_scores(bundle["tail_model"], cal), alpha)
    manifest = {
        "protocol_version": PROTOCOL_VERSION, "created_at": time.time(), "seed": seed,
        "detector": detector,
        "detector_parameters": json_safe_parameters(
            build_detector(detector, seed, len(train)).get_params(deep=True)),
        "internal_method_alias_note": "IF_* method labels are retained only for compatibility with the frozen live hierarchy; the estimator class is given by detector.",
        "alpha_target": alpha, "calibration_rows": len(cal), "training_rows": len(train),
        "train_runs": sorted(train.run_id.unique().tolist()), "calibration_runs": sorted(cal.run_id.unique().tolist()),
        "features": MODEL_FEATURES, "forbidden_inputs": ["attack", "scenario", "onset", "episode_id"] + META,
        "tail_model": bundle["tail_model"],
        "development_disclosure": "Tail branch introduced after exploratory run 42368716451 missed S4. "
                                  "Original six methods are unchanged. New parameters use only fresh normal "
                                  "training/calibration; confirmation requires a new prospective test run.",
        "thresholds": {k: v if math.isfinite(v) else None for k, v in bundle["thresholds"].items()},
        "threshold_sufficient": all(math.isfinite(v) for v in bundle["thresholds"].values()),
        "score_convention": "Anomaly margin; alarm iff score > 0, strict ties. Rule evidence has margin +1/-1.",
        "rules": {"baseline": "profile_policy_violation > 0 OR heartbeat_age > 30 seconds",
                  "hybrid_without_KIE": "IF_CONTEXT OR baseline OR forbidden_operation > 0 OR route_mismatch > 0",
                  "hybrid": "hybrid_without_KIE OR report_mismatch > 0 OR auth_failure > 0",
                  "hybrid_tail_without_KIE": "robust_tail OR baseline OR forbidden_operation > 0 OR route_mismatch > 0",
                  "hybrid_tail": "hybrid_tail_without_KIE OR report_mismatch > 0 OR auth_failure > 0"},
        "assumptions": ["Fixed operator policy; trusted independent observer and control-channel verifier.",
                        "HMAC authenticates report origin, not the truth of a compromised actor's report.",
                        "A constant-in-training invariant is checked explicitly, not delegated to the statistical detector.",
                        "Correlated windows do not establish an exchangeability-based FPR guarantee.",
                        "No held-out test result is used for threshold selection."],
        "sources": train_files + cal_files,
        "implementation_sha256": {p.name: sha256(p) for p in Path(__file__).parent.iterdir()
                                  if p.is_file() and p.suffix in (".py", ".md", ".txt", ".sh")},
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                     "sklearn": sklearn.__version__},
    }
    bundle["manifest"] = manifest
    cal_predictions = score_frame(bundle, cal)
    manifest["calibration_alarm_rate"] = cal_predictions.groupby("method").alarm.mean().to_dict()
    joblib.dump(bundle, out / "model.joblib")
    manifest["model_sha256"] = sha256(out / "model.joblib")
    dump_json(out / "manifest.json", manifest)
    cal_predictions.to_csv(out / "calibration_predictions.csv", index=False)
    return manifest


def load_frozen(path):
    root = Path(path)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if sha256(root / "model.joblib") != manifest["model_sha256"]:
        raise ValueError("Frozen model hash mismatch")
    bundle = joblib.load(root / "model.joblib")
    if bundle["protocol_version"] not in ("live-5g-v1", PROTOCOL_VERSION):
        raise ValueError("Unsupported protocol version")
    bundle["manifest"] = manifest
    return bundle


def score_frame(bundle, frame):
    frame = pd.DataFrame(frame).copy()
    validate_features(frame)
    margins = {}
    for name, columns in MODEL_FEATURES.items():
        threshold = bundle["thresholds"][name]
        raw = anomaly_scores(bundle["models"][name],
                             frame[columns].to_numpy(dtype=float),
                             bundle.get("detector", bundle.get("manifest", {}).get("detector", "iforest")))
        margins[name] = raw - threshold if math.isfinite(threshold) else np.full(len(frame), -1.)
    baseline = (frame.profile_policy_violation > 0) | (frame.heartbeat_age > 30)
    non_kie = baseline | (frame.forbidden_operation > 0) | (frame.route_mismatch > 0)
    kie = non_kie | (frame.report_mismatch > 0) | (frame.auth_failure > 0)
    margins["rules"] = np.where(baseline, 1., -1.)
    margins["hybrid_without_KIE"] = np.maximum(margins["IF_CONTEXT"], np.where(non_kie, 1., -1.))
    margins["hybrid"] = np.maximum(margins["IF_CONTEXT"], np.where(kie, 1., -1.))
    methods = list(METHODS)
    if "tail_model" in bundle:
        threshold = bundle["thresholds"]["robust_tail"]
        margins["robust_tail"] = (tail_scores(bundle["tail_model"], frame) - threshold
                                  if math.isfinite(threshold) else np.full(len(frame), -1.))
        margins["hybrid_tail_without_KIE"] = np.maximum(margins["robust_tail"], np.where(non_kie, 1., -1.))
        margins["hybrid_tail"] = np.maximum(margins["robust_tail"], np.where(kie, 1., -1.))
        methods += TAIL_METHODS
    predictions = []
    for name in methods:
        part = frame[META].copy()
        part["method"] = name
        part["score"] = margins[name]
        part["threshold"] = 0.
        part["alarm"] = (margins[name] > 0).astype(int)
        predictions.append(part)
    return pd.concat(predictions, ignore_index=True)


def wilson(successes, n):
    if n == 0:
        return [None, None]
    z = 1.959963984540054
    p = successes / n
    denominator = 1 + z * z / n
    middle = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return [max(0., middle - half), min(1., middle + half)]


def metrics(frame):
    truth = frame.attack.to_numpy(dtype=bool)
    pred = frame.alarm.to_numpy(dtype=bool)
    tp, fp = int((truth & pred).sum()), int((~truth & pred).sum())
    fn, tn = int((truth & ~pred).sum()), int((~truth & ~pred).sum())
    return {"n": len(frame), "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "attack_prevalence": float(truth.mean()) if len(frame) else None,
            "recall": tp / (tp + fn) if tp + fn else None,
            "fpr": fp / (fp + tn) if fp + tn else None,
            "precision": tp / (tp + fp) if tp + fp else None,
            "average_precision": float(average_precision_score(truth, frame.score)) if truth.any() else None,
            "fpr_wilson_descriptive": wilson(fp, fp + tn),
            "recall_wilson_descriptive": wilson(tp, tp + fn)}


def score(frozen_directory, test_directory, output):
    bundle = load_frozen(frozen_directory)
    frame, labels, provenance = read_dataset(test_directory)
    manifest = bundle["manifest"]
    trained = set(manifest["train_runs"] + manifest["calibration_runs"])
    if trained & set(frame.run_id):
        raise ValueError("Held-out acquisition overlaps train/calibration runs")
    if frame.timestamp.min() <= manifest["created_at"]:
        raise ValueError("Test observations must be acquired after model freezing")
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    predictions = score_frame(bundle, frame)
    processing_seconds = time.perf_counter() - started
    online_path = Path(test_directory) / "online_predictions.csv"
    timing_source = "feature_completion_proxy"
    if online_path.is_file():
        predictions = attach_online_decisions(predictions, pd.read_csv(online_path,
            dtype={"run_id": str, "nf_id": str}))
        provenance.append({"path": str(online_path), "sha256": sha256(online_path), "rows": len(predictions)})
        timing_source = "online_decision_timestamp"
    else:
        predictions["decision_timestamp"] = predictions.timestamp
    merged = predictions.merge(labels, on=KEYS, validate="many_to_one")
    merged.to_csv(out / "predictions.csv", index=False)
    totals, scenarios, episodes, runs = [], [], [], []
    for name, subset in merged.groupby("method", sort=False):
        totals.append({"method": name, **metrics(subset)})
        for scenario, part in subset.groupby("scenario", dropna=False):
            scenarios.append({"method": name, "scenario": str(scenario), **metrics(part)})
        for run_id, part in subset.groupby("run_id"):
            normal = part.loc[part.attack == 0]
            runs.append({"method": name, "run_id": run_id, "benign_any_alarm": bool(normal.alarm.any()), **metrics(part)})
        positive = subset.loc[subset.attack.astype(bool)]
        for identity, part in positive.groupby(["run_id", "nf_id", "episode_id", "scenario"], dropna=False):
            onset = float(part.onset.min())
            if (part.timestamp < part.onset).any():
                raise ValueError("Attack label predates its measured onset")
            alarms = part.loc[part.alarm.astype(bool)]
            alarm_at = float(alarms.decision_timestamp.min()) if len(alarms) else None
            episodes.append(dict(zip(["run_id", "nf_id", "episode_id", "scenario"], identity), method=name,
                                 onset=onset, detected=alarm_at is not None, first_alarm_at=alarm_at,
                                 delay_seconds=alarm_at - onset if alarm_at is not None else None,
                                 observed_until=float(part.decision_timestamp.max()), timing_source=timing_source))
    episode_frame = pd.DataFrame(episodes, columns=EPISODE_COLUMNS)
    add_episode_metrics(totals, episode_frame)
    report = {"protocol_version": bundle["protocol_version"], "frozen_model_sha256": manifest["model_sha256"],
              "created_at": time.time(), "test_runs": sorted(frame.run_id.unique().tolist()),
              "sources": provenance, "metrics": totals, "per_scenario": scenarios, "per_run": runs,
              "delay_timing_source": timing_source,
              "offline_scoring_seconds": processing_seconds,
              "offline_scoring_microseconds_per_nf_window_all_methods": processing_seconds * 1e6 / len(frame),
              "limitations": ["NF windows within a run are correlated. Wilson intervals are descriptive, not independent-trial guarantees.",
                              "Delayed/missed episodes are retained. Delay percentiles condition on detected episodes.",
                              "Precision and average precision use the observed test prevalence, not operator prevalence.",
                              "Delay uses archived online decisions when present; otherwise only feature-completion proxy timing.",
                              "Successful pipeline completion is not a positive scientific result."]}
    dump_json(out / "report.json", report)
    pd.DataFrame(totals).to_csv(out / "metrics.csv", index=False)
    pd.DataFrame(scenarios).to_csv(out / "per_scenario.csv", index=False)
    pd.DataFrame(runs).to_csv(out / "per_run.csv", index=False)
    episode_frame.to_csv(out / "episodes.csv", index=False)
    write_report(out / "REPORT_RU.md", report)
    return report


def attach_online_decisions(predictions, online):
    """Require exactly the frozen decisions measured while traffic was acquired."""
    keys = KEYS + ["method"]
    required = keys + ["score", "alarm", "decision_timestamp"]
    if set(required) - set(online) or online.duplicated(keys).any():
        raise ValueError("Missing or duplicated online decisions")
    checked = predictions.merge(online[required], on=keys, how="outer", suffixes=("", "_online"),
                                validate="one_to_one", indicator=True)
    if not checked._merge.eq("both").all():
        raise ValueError("Online decisions do not cover exactly the evaluated NF windows")
    if not np.array_equal(checked.alarm, checked.alarm_online) or not np.allclose(
            checked.score, checked.score_online, rtol=1e-10, atol=1e-12):
        raise ValueError("Online decisions disagree with the frozen model replay")
    if not np.isfinite(checked.decision_timestamp).all() or (checked.decision_timestamp < checked.timestamp).any():
        raise ValueError("Online decision precedes available observation")
    return checked.drop(columns=["alarm_online", "score_online", "_merge"])


def add_episode_metrics(totals, episodes):
    for total in totals:
        selected = episodes.loc[episodes.method == total["method"]] if len(episodes) else episodes
        detected = selected.loc[selected.detected.astype(bool)] if len(selected) else selected
        total.update(episodes=len(selected), detected_episodes=len(detected),
                     episode_recall=len(detected) / len(selected) if len(selected) else None,
                     median_delay_detected_seconds=float(detected.delay_seconds.median()) if len(detected) else None,
                     p95_delay_detected_seconds=float(detected.delay_seconds.quantile(.95)) if len(detected) else None,
                     missed_episodes=len(selected) - len(detected))


def write_report(path, report):
    def fmt(value, percent=False):
        return "—" if value is None else f"{100 * value:.3f}%" if percent else f"{value:.4f}"
    lines = ["# Проверка обнаружения аномалий в живом потоке SBI Open5GS", "",
             "Модели обучены на нормальных наблюдениях; пороги зафиксированы до сбора теста. "
             "Результаты относятся к контролируемому стенду и заданным воздействиям.", "",
             "| Метод | TPR | FPR | Precision | AP | Эпизоды | Медиана задержки, с |", "|---|---:|---:|---:|---:|---:|---:|"]
    for item in report["metrics"]:
        lines.append(f"| {item['method']} | {fmt(item['recall'],True)} | {fmt(item['fpr'],True)} | "
                     f"{fmt(item['precision'],True)} | {fmt(item['average_precision'])} | "
                     f"{item.get('detected_episodes',0)}/{item.get('episodes',0)} | {fmt(item.get('median_delay_detected_seconds'))} |")
    lines.extend(["", "AP означает average precision, а не трапецеидальную площадь под PR-кривой. "
                  "Цель FPR ≤ 0,1% не является заранее гарантированным результатом. "
                  "Нулевой счётчик ложных тревог на коррелированных окнах не доказывает такой предел в эксплуатации.", "",
                  "Гибрид объединяет Isolation Forest с явными проверками наблюдаемых нарушений. "
                  "Результаты проверок не выдаются за выученные машинным обучением закономерности. "
                  "Сравнение hybrid и hybrid_without_KIE выделяет добавление контрольных отчётов и проверки подлинности; "
                  "сравнение с правилами показывает вклад статистической ветви.", "",
                  "Варианты robust_tail и hybrid_tail, когда они присутствуют, относятся к следующей "
                  "версии метода. robust_tail обучает одностороннюю робастную границу текущих объёмов "
                  "и ошибок на норме, а hybrid_tail объединяет её с теми же семантическими проверками. "
                  "Исходные шесть методов сохранены без изменений. Это отдельная статистическая ветвь, "
                  "а не улучшение Isolation Forest; решение о её добавлении принято после первой "
                  "исследовательской серии и проверяется на новых проспективных данных.", "",
                  "При наличии online_predictions.csv задержка вычислена по фактическому времени онлайн-решения "
                  "после проверки совпадения с замороженной моделью. В иных случаях используется только время "
                  "завершения наблюдения; источник времени указан в JSON и таблице эпизодов. "
                  "Необнаруженные эпизоды сохранены в episodes.csv; медиана задержки относится только к обнаруженным.", "",
                  "Входные файлы и модель идентифицированы SHA-256 в report.json и manifest.json. "
                  "Отрицательные и нулевые эффекты сохранены в таблицах."])
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def aggregate(input_directory, output, expected_runs=None, required_scenarios=None):
    paths = sorted(Path(input_directory).rglob("report.json"))
    if not paths:
        raise ValueError("No completed held-out reports")
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    reports = [json.loads(p.read_text(encoding="utf-8")) for p in paths]
    identities = [r for report in reports for r in report["test_runs"]]
    if len(set(identities)) != len(identities):
        raise ValueError("Duplicate test runs in aggregation")
    if expected_runs is not None and (expected_runs <= 0 or len(identities) != expected_runs):
        raise ValueError(f"Expected {expected_runs} complete test runs, found {len(identities)}")
    frame = pd.concat([pd.read_csv(path.with_name("predictions.csv")) for path in paths], ignore_index=True)
    if required_scenarios:
        for run_id, run in frame.groupby("run_id"):
            observed = set(run.loc[run.attack.astype(bool), "scenario"])
            missing = set(required_scenarios) - observed
            if missing:
                raise ValueError(f"Run {run_id} is missing positive scenario exposure: {sorted(missing)}")
    episode_parts = []
    for path in paths:
        episode_path = path.with_name("episodes.csv")
        try:
            episode_parts.append(pd.read_csv(episode_path))
        except pd.errors.EmptyDataError:
            # Legacy empty exports may contain LF or CRLF without a header.
            episode_parts.append(pd.DataFrame(columns=EPISODE_COLUMNS))
    nonempty_episodes = [part for part in episode_parts if not part.empty]
    episodes = pd.concat(nonempty_episodes, ignore_index=True) if nonempty_episodes else pd.DataFrame(columns=EPISODE_COLUMNS)
    totals = []
    run_rows = []
    for method, part in frame.groupby("method", sort=False):
        row = {"method": method, **metrics(part)}
        chunks = [{"method": method, "run_id": run_id, **metrics(g)} for run_id, g in part.groupby("run_id")]
        run_rows.extend(chunks)
        row["acquisition_runs"] = len(chunks)
        for metric in ("recall", "fpr", "precision", "average_precision"):
            values = [g[metric] for g in chunks if g[metric] is not None]
            row[f"equal_run_mean_{metric}"] = float(np.mean(values)) if values else None
            row[f"run_range_{metric}"] = [min(values), max(values)] if values else [None, None]
        totals.append(row)
    add_episode_metrics(totals, episodes)
    versions = sorted({r["protocol_version"] for r in reports})
    result = {"protocol_version": versions[0] if len(versions) == 1 else "mixed",
              "source_protocol_versions": versions, "test_runs": sorted(identities), "metrics": totals,
              "per_run": run_rows,
              "completeness_audit": {"expected_runs": expected_runs, "observed_runs": len(identities),
                                     "required_scenarios": list(required_scenarios or [])},
              "source_reports": [{"path": str(p), "sha256": sha256(p)} for p in paths],
              "aggregation": "Micro-pooled counts and equal-run means are both reported. Runs use different actor counts; they are scale conditions, not IID replicas. No three-run confidence claim.",
              "average_precision_scope": "Pooled AP ranks model margins from different fits and is descriptive. Per-run AP and its equal-run mean are preferred.",
              "delay_timing_sources": sorted({r.get("delay_timing_source", "unknown") for r in reports})}
    dump_json(out / "summary.json", result)
    pd.DataFrame(totals).to_csv(out / "metrics.csv", index=False)
    pd.DataFrame(run_rows).to_csv(out / "per_run.csv", index=False)
    episodes.to_csv(out / "episodes.csv", index=False)
    write_report(out / "REPORT_RU.md", result)
    return result


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    fit_parser = commands.add_parser("fit")
    for arg in ("train", "cal", "out"):
        fit_parser.add_argument(f"--{arg}", required=True)
    fit_parser.add_argument("--alpha", type=float, default=.001)
    fit_parser.add_argument("--seed", type=int, default=7143)
    fit_parser.add_argument("--detector", choices=DETECTOR_CHOICES, default="iforest")
    score_parser = commands.add_parser("score")
    for arg in ("frozen", "test", "out"):
        score_parser.add_argument(f"--{arg}", required=True)
    aggregate_parser = commands.add_parser("aggregate")
    aggregate_parser.add_argument("--input", required=True)
    aggregate_parser.add_argument("--out", required=True)
    aggregate_parser.add_argument("--expected-runs", type=int)
    aggregate_parser.add_argument("--require-scenarios", nargs="+", choices=["S1", "S2", "S3", "S4"])
    args = parser.parse_args()
    if args.command == "fit":
        result = fit(args.train, args.cal, args.out, args.alpha, args.seed, args.detector)
    elif args.command == "score":
        result = score(args.frozen, args.test, args.out)
    else:
        result = aggregate(args.input, args.out, args.expected_runs, args.require_scenarios)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
