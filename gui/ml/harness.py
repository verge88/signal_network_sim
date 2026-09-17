"""Subprocess harness for the repository's existing simulation and training code.

Run with: python -m gui.ml.harness path/to/job.json
"""
from __future__ import annotations

import importlib
import inspect
import json
import os
import pickle
import re
import sys
import time
import traceback
from dataclasses import fields as dc_fields
from typing import Any

from .protocol import JobSpec, _write_line, emit

TRAIN_MODULES = {
    "diameter": "diameter_training_v1",
    "ss7": "ss7_training_v7",
    "sip": "sip_training_v4",
    "5g_sba": "sba_eval",
}


class _Capture:
    """Best-effort capture of fitted sklearn estimators."""

    def __init__(self) -> None:
        self.sklearn: dict[str, Any] = {}
        self.torch: dict[str, Any] = {}
        self._installed: list[tuple[type, Any]] = []
        self._count = 0

    def install(self) -> None:
        try:
            import sklearn.ensemble as ensemble
            import sklearn.impute as impute
            import sklearn.preprocessing as preprocessing
            import sklearn.svm as svm
        except ImportError:
            return
        classes = [
            getattr(ensemble, name, None)
            for name in ("IsolationForest", "RandomForestClassifier", "GradientBoostingClassifier")
        ] + [getattr(svm, "OneClassSVM", None), getattr(preprocessing, "StandardScaler", None),
             getattr(impute, "SimpleImputer", None)]
        for cls in (item for item in classes if item is not None):
            if any(original is cls for _, original in self._installed):
                continue
            original = cls.fit
            capture = self

            def fit(instance, *args, __original=original, **kwargs):
                result = __original(instance, *args, **kwargs)
                capture._count += 1
                capture.sklearn[f"{type(instance).__name__}#{capture._count}"] = instance
                return result

            cls.fit = fit
            self._installed.append((cls, original))

    def restore(self) -> None:
        for cls, original in self._installed:
            cls.fit = original
        self._installed.clear()

    def collect_torch(self, module: Any) -> None:
        try:
            import torch.nn as nn
        except ImportError:
            return
        for name in dir(module):
            obj = getattr(module, name, None)
            if isinstance(obj, nn.Module):
                self.torch[name] = obj.state_dict()

    def dump(self, run_dir: str) -> list[str]:
        paths: list[str] = []
        artifact_dir = os.path.join(run_dir, "artifacts")
        os.makedirs(artifact_dir, exist_ok=True)
        if self.sklearn:
            path = os.path.join(artifact_dir, "sklearn_models.pkl")
            try:
                with open(path, "wb") as stream:
                    pickle.dump(self.sklearn, stream)
                paths.append(path)
            except Exception as exc:  # noqa: BLE001
                emit("warn", msg=f"не удалось сохранить sklearn-модели: {exc}")
        if self.torch:
            try:
                import torch
                path = os.path.join(artifact_dir, "torch_state.pt")
                torch.save(self.torch, path)
                paths.append(path)
            except Exception as exc:  # noqa: BLE001
                emit("warn", msg=f"не удалось сохранить torch-модели: {exc}")
        return paths


_EPOCH_RE = re.compile(r"[Ee]poch\s+(\d+)\s*/\s*(\d+)")


class _Tee:
    """Duplicate script output while keeping write() non-throwing."""

    def __init__(self, raw: Any, log_path: str):
        self.raw = raw
        try:
            self.file = open(log_path, "a", encoding="utf-8", errors="replace", newline="")
        except OSError:
            self.file = None

    def write(self, data: str) -> int:
        if not isinstance(data, str):
            data = str(data)
        if self.file is not None:
            try:
                self.file.write(data)
                self.file.flush()
            except Exception:
                pass
        _write_line_nonl(self.raw, data)
        try:
            match = _EPOCH_RE.search(data)
            if match:
                current, total = map(int, match.groups())
                emit("progress", stage="train", value=current / max(1, total),
                     text=f"эпоха {current}/{total}")
        except Exception:
            pass
        return len(data)

    def flush(self) -> None:
        for stream in (self.file, self.raw):
            try:
                if stream is not None:
                    stream.flush()
            except Exception:
                pass

    def close(self) -> None:
        try:
            if self.file is not None:
                self.file.close()
        except Exception:
            pass

    def isatty(self) -> bool:
        return False

    @property
    def encoding(self) -> str:
        return "utf-8"


def _write_line_nonl(stream: Any, text: str) -> None:
    if stream is None:
        return
    try:
        stream.write(text)
        stream.flush()
    except UnicodeEncodeError:
        encoding = getattr(stream, "encoding", None) or "utf-8"
        try:
            stream.write(text.encode(encoding, "replace").decode(encoding, "replace"))
            stream.flush()
        except Exception:
            pass
    except Exception:
        pass


def stage_simulate(job: JobSpec) -> str:
    from ..adapters import run_with_topology
    from ..model import Topology

    topology = Topology.load(job.topology_path)
    topology.sim_params["seed"] = job.seed
    os.makedirs(job.run_dir, exist_ok=True)
    emit("stage", name="simulate", status="start",
         nodes=len(topology.nodes), links=len(topology.links))
    previous = os.getcwd()
    os.chdir(job.run_dir)
    try:
        run_with_topology(topology, job.sim_dir, log=print,
                          module_name=job.sim_module)
    finally:
        os.chdir(previous)
    candidates = [os.path.join(job.run_dir, name)
                  for name in os.listdir(job.run_dir) if name.endswith(".csv")]
    if not candidates:
        raise FileNotFoundError("симулятор не создал CSV-датасет")
    output = max(candidates, key=os.path.getsize)
    emit("stage", name="simulate", status="done", dataset=output,
         size_mb=round(os.path.getsize(output) / 1e6, 2))
    return output


def stage_train(job: JobSpec, dataset: str, capture: _Capture) -> dict[str, Any]:
    module_name = job.train_module or TRAIN_MODULES.get(job.protocol)
    if not module_name:
        raise RuntimeError(f"не задан модуль обучения для {job.protocol}")
    module = importlib.import_module(module_name)
    config_class = getattr(module, "TrainingConfig", None)
    if config_class is None:
        raise RuntimeError(f"{module_name}: нет класса TrainingConfig")
    emit("stage", name="train", status="start", module=module.__file__)
    allowed = {field.name for field in dc_fields(config_class)}
    overrides = {key: value for key, value in job.train_overrides.items() if key in allowed}
    unknown = sorted(set(job.train_overrides) - allowed)
    if unknown:
        emit("warn", msg=f"игнорирую неизвестные параметры: {unknown}")
    overrides.setdefault("seed", job.seed)
    if "data_path" in allowed:
        overrides["data_path"] = dataset
    if "results_dir" in allowed:
        overrides["results_dir"] = os.path.join(job.run_dir, "results")
    if "plots_dir" in allowed:
        overrides["plots_dir"] = os.path.join(job.run_dir, "results", "plots")
    if "out_dir" in allowed:
        overrides["out_dir"] = os.path.join(job.run_dir, "results")
    config = config_class(**overrides)
    module.TrainingConfig = lambda **_kwargs: config
    export_dir = os.path.join(job.run_dir, "artifacts")
    os.makedirs(export_dir, exist_ok=True)
    os.environ["SNS_EXPORT_DIR"] = export_dir
    with open(os.path.join(job.run_dir, "train_config.json"), "w", encoding="utf-8") as stream:
        json.dump({key: getattr(config, key, None) for key in allowed}, stream,
                  ensure_ascii=False, indent=2, default=str)
    if job.capture_models:
        capture.install()
    started = time.time()
    try:
        main = getattr(module, "main", None)
        if main is None:
            raise RuntimeError(f"{module_name}: нет функции main()")
        parameters = list(inspect.signature(main).parameters.values())
        if not parameters:
            result = main()
        elif parameters[0].name in {"cfg", "config", "training_config"}:
            result = main(config)
        else:
            result = main([])
        capture.collect_torch(module)
    finally:
        capture.restore()
    emit("stage", name="train", status="done", seconds=round(time.time() - started, 1))
    return {"result_repr": repr(result)[:2000] if result is not None else None}


def stage_evaluate(job: JobSpec) -> None:
    if not job.test_datasets:
        return
    from .evaluate import evaluate_run
    for dataset in job.test_datasets:
        emit("stage", name="evaluate", status="start", dataset=dataset)
        rows = evaluate_run(job.run_dir, dataset)
        emit("transfer", dataset=dataset, rows=rows)


def main(argv: list[str]) -> int:
    for stream in (sys.__stdout__, sys.__stderr__):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
        except (AttributeError, ValueError, OSError):
            pass
    if len(argv) != 2:
        print("usage: python -m gui.ml.harness JOB_JSON", file=sys.stderr)
        return 2
    with open(argv[1], encoding="utf-8") as stream:
        job = JobSpec.from_json(stream.read())
    os.makedirs(job.run_dir, exist_ok=True)
    if job.threads > 0:
        for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            os.environ[variable] = str(job.threads)
        try:
            import torch
            torch.set_num_threads(job.threads)
        except ImportError:
            pass
    if job.sim_dir and job.sim_dir not in sys.path:
        sys.path.insert(0, os.path.abspath(job.sim_dir))
    sys.stdout = _Tee(sys.__stdout__, os.path.join(job.run_dir, "log.txt"))
    sys.stderr = sys.stdout
    capture = _Capture()
    summary: dict[str, Any] = {"job_id": job.job_id, "seed": job.seed,
                               "protocol": job.protocol,
                               "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    try:
        dataset = stage_simulate(job) if job.make_dataset else os.path.abspath(job.dataset_path)
        summary["dataset"] = dataset
        if job.do_train:
            summary.update(stage_train(job, dataset, capture))
            if job.capture_models:
                summary["artifacts"] = capture.dump(job.run_dir)
        stage_evaluate(job)
        summary["status"] = "ok"
    except BaseException as exc:  # noqa: BLE001
        summary["status"] = "error"
        summary["error"] = f"{type(exc).__name__}: {exc}"
        summary["traceback"] = traceback.format_exc()
        _write_line(sys.__stderr__, summary["traceback"])
        emit("failed", error=summary["error"], traceback=summary["traceback"])
    finally:
        summary["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
        with open(os.path.join(job.run_dir, "summary.json"), "w", encoding="utf-8") as stream:
            json.dump(summary, stream, ensure_ascii=False, indent=2, default=str)
        emit("done", status=summary["status"], run_dir=job.run_dir)
        sys.stdout.flush()
    return 0 if summary["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
