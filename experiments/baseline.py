from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import shlex
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SEEDS = list(range(42, 52))
PACKAGE_NAMES = (
    "numpy",
    "pandas",
    "networkx",
    "scikit-learn",
    "scipy",
    "torch",
)


@dataclass(frozen=True)
class BaselineProfile:
    name: str
    workdir: str
    runner: str
    simulator: str
    training: tuple[str, ...]
    dataset: str
    metrics_file: str = "all_seed_metrics.csv"
    metrics_candidates: tuple[str, ...] = ()
    config_targets: tuple[str, ...] = ()
    extra_sources: tuple[str, ...] = ()
    supports_regen: bool = False

    @property
    def workdir_path(self) -> Path:
        return REPO_ROOT / self.workdir

    @property
    def dataset_path(self) -> Path:
        return self.workdir_path / self.dataset


PROFILES: dict[str, BaselineProfile] = {
    "ss7": BaselineProfile(
        name="ss7",
        workdir="ss7",
        # IMPORTANT: run_seeds_ss7.py is a legacy wrapper and currently expects
        # ss7_training_v7.DataPipeline, which no longer exists.  The current
        # ss7_training_v7.py has its own scientifically corrected multiseed()
        # implementation, so Stage 0 invokes it directly.
        runner="ss7_training_v7.py",
        simulator="fix.py",
        training=("ss7_training_v7.py",),
        dataset="ss7_dataset_v7.csv",
        metrics_file="multiseed_raw.csv",
        metrics_candidates=(
            "multiseed_raw.csv",
            "models.csv",
        ),
        config_targets=(
            "fix:SimConfig",
            "ss7_training_v7:TrainingConfig",
        ),
        extra_sources=("run_seeds_ss7.py", "ms_multiseed_core.py"),
        supports_regen=True,
    ),
    "diameter": BaselineProfile(
        name="diameter",
        workdir="diameter",
        runner="run_seeds_diameter.py",
        simulator="diameter_sim_v1.py",
        training=("diameter_training_v1.py",),
        dataset="diameter_dataset_v1.csv",
        metrics_candidates=("all_seed_metrics.csv",),
        config_targets=(
            "diameter_sim_v1:SimulationConfig",
            "diameter_training_v1:TrainingConfig",
        ),
        extra_sources=("ms_multiseed_core.py",),
        supports_regen=True,
    ),
    "sip": BaselineProfile(
        name="sip",
        workdir="sip",
        runner="run_seeds_ticd.py",
        simulator="sip_sim_v4_claude.py",
        training=("train_ticd.py", "ticd.py", "sip_train_v4_claude.py"),
        dataset="sip_dataset_v4.csv",
        metrics_candidates=("all_seed_metrics.csv",),
        config_targets=("sip_sim_v4_claude:SimulationConfig",),
        supports_regen=False,
    ),
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def canonical_hash(obj: Any) -> str:
    payload = json.dumps(
        obj,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return _sha256_bytes(payload)


def _run_git(*args: str, binary: bool = False) -> bytes | str:
    proc = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    return (
        proc.stdout
        if binary
        else proc.stdout.decode("utf-8", errors="replace").strip()
    )


def _hash_untracked_source_files(
    status_lines: Sequence[str],
) -> dict[str, str]:
    suffixes = {".py", ".json", ".toml", ".yaml", ".yml", ".ini", ".cfg"}
    result: dict[str, str] = {}
    for line in status_lines:
        if not line.startswith("?? "):
            continue
        rel = line[3:].strip()
        path = REPO_ROOT / rel
        if path.is_file() and path.suffix.lower() in suffixes:
            result[rel] = sha256_file(path)
    return result


def git_state() -> dict[str, Any]:
    status = str(
        _run_git("status", "--porcelain=v1", "--untracked-files=all")
    )
    lines = [line for line in status.splitlines() if line.strip()]

    worktree_diff = _run_git("diff", "--binary", "HEAD", binary=True)
    staged_diff = _run_git(
        "diff", "--binary", "--cached", "HEAD", binary=True
    )
    combined = (
        bytes(worktree_diff)
        + b"\0--STAGED--\0"
        + bytes(staged_diff)
    )

    return {
        "commit": _run_git("rev-parse", "HEAD"),
        "branch": _run_git("branch", "--show-current"),
        "describe": _run_git("describe", "--always", "--dirty", "--tags"),
        "dirty": bool(lines),
        "status_porcelain": lines,
        "tracked_diff_sha256": _sha256_bytes(combined),
        "untracked_source_sha256": _hash_untracked_source_files(lines),
    }


def environment_snapshot() -> dict[str, Any]:
    packages: dict[str, str | None] = {}
    for name in PACKAGE_NAMES:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None

    return {
        "python": sys.version,
        "python_executable": sys.executable,
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "packages": packages,
    }


def source_hashes(
    profile: BaselineProfile,
) -> dict[str, str | None]:
    names = [
        profile.runner,
        profile.simulator,
        *profile.training,
        *profile.extra_sources,
    ]
    result: dict[str, str | None] = {}

    for name in names:
        path = profile.workdir_path / name
        rel = str(path.relative_to(REPO_ROOT))
        result[rel] = sha256_file(path) if path.is_file() else None

    return result


_CONFIG_PROBE = r"""
import dataclasses
import enum
import importlib
import json
import pathlib
import sys


def norm(value):
    if dataclasses.is_dataclass(value):
        return {
            field.name: norm(getattr(value, field.name))
            for field in dataclasses.fields(value)
        }
    if isinstance(value, enum.Enum):
        return norm(value.value)
    if isinstance(value, pathlib.Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): norm(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [norm(v) for v in value]
    if (
        isinstance(value, (str, int, float, bool))
        or value is None
    ):
        return value
    if hasattr(value, "tolist"):
        try:
            return norm(value.tolist())
        except Exception:
            pass
    if hasattr(value, "__dict__"):
        return {
            str(k): norm(v)
            for k, v in vars(value).items()
            if not str(k).startswith("_")
        }
    return repr(value)


target = sys.argv[1]
module_name, class_name = target.split(":", 1)
module = importlib.import_module(module_name)
cls = getattr(module, class_name)
obj = cls()
print(
    "__BASELINE_JSON__"
    + json.dumps(norm(obj), ensure_ascii=False, sort_keys=True)
)
"""


def probe_default_config(
    profile: BaselineProfile,
    target: str,
) -> dict[str, Any]:
    proc = subprocess.run(
        [sys.executable, "-c", _CONFIG_PROBE, target],
        cwd=profile.workdir_path,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    payload = None
    for line in proc.stdout.splitlines():
        if line.startswith("__BASELINE_JSON__"):
            payload = line[len("__BASELINE_JSON__"):]

    if proc.returncode != 0 or payload is None:
        return {
            "ok": False,
            "returncode": proc.returncode,
            "stderr": proc.stderr[-4000:],
            "stdout_tail": proc.stdout[-4000:],
        }

    try:
        return {"ok": True, "value": json.loads(payload)}
    except json.JSONDecodeError as exc:
        return {
            "ok": False,
            "error": f"invalid probe JSON: {exc}",
            "raw": payload[-4000:],
        }


def default_config_snapshot(
    profile: BaselineProfile,
) -> dict[str, Any]:
    return {
        target: probe_default_config(profile, target)
        for target in profile.config_targets
    }


def _coerce_csv_value(value: str) -> Any:
    text = value.strip()
    if text == "":
        return None

    low = text.lower()
    if low in {"nan", "na", "null", "none"}:
        return None
    if low in {"true", "false"}:
        return low == "true"

    try:
        if any(ch in text for ch in (".", "e", "E")):
            return float(text)
        return int(text)
    except ValueError:
        return text


def read_metrics_csv(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = [
            {
                str(key): _coerce_csv_value(value or "")
                for key, value in row.items()
            }
            for row in reader
        ]

    return {
        "source": str(path),
        "sha256": sha256_file(path),
        "row_count": len(rows),
        "rows": rows,
    }


def artifact_hashes(
    root: Path,
) -> dict[str, dict[str, Any]]:
    if not root.exists():
        return {}

    result: dict[str, dict[str, Any]] = {}
    for path in sorted(
        p for p in root.rglob("*") if p.is_file()
    ):
        rel = str(path.relative_to(root))
        result[rel] = {
            "size": path.stat().st_size,
            "sha256": sha256_file(path),
        }

    return result


def build_command(
    profile: BaselineProfile,
    seeds: Sequence[int],
    results_dir: Path,
    regen: bool,
) -> tuple[list[str], bool]:
    if profile.name == "ss7":
        # Current ss7_training_v7.py already implements multiseed() and,
        # for every seed, rebuilds topology, traffic and episodes through
        # build_dataset(). Do NOT route this through the stale
        # run_seeds_ss7.py/ms_multiseed_core.py compatibility wrapper.
        if not seeds:
            raise ValueError("SS7 baseline requires at least one seed")

        command = [
            sys.executable,
            profile.runner,
            "--seed",
            str(int(seeds[0])),
            "--seeds",
            *[str(int(seed)) for seed in seeds],
            "--out-dir",
            str(results_dir.resolve()),
        ]

        # multiseed() itself regenerates worlds regardless of this flag.
        # --no-cache additionally forces the preliminary single-seed run
        # in main() not to reuse its cached dataset.
        if regen:
            command.append("--no-cache")

        return command, True

    if profile.name == "diameter":
        command = [
            sys.executable,
            profile.runner,
            *[str(seed) for seed in seeds],
            "--out",
            str(results_dir.resolve()),
        ]

        effective_regen = bool(
            regen and profile.supports_regen
        )
        if effective_regen:
            command.append("--regen")

        return command, effective_regen

    if profile.name == "sip":
        # The current SIP runner has no --out option.
        # Override its module-level BASE without changing legacy code.
        code = (
            "import run_seeds_ticd as r; "
            f"r.BASE={str(results_dir.resolve())!r}; "
            f"r.main("
            f"{list(map(int, seeds))!r}, "
            f"{str(profile.dataset_path.resolve())!r}"
            f")"
        )
        return [sys.executable, "-c", code], False

    raise KeyError(
        f"unsupported baseline profile: {profile.name}"
    )


def run_streamed(
    command: Sequence[str],
    cwd: Path,
    log_path: Path,
) -> tuple[int, float]:
    started = time.monotonic()

    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + shlex.join(command) + "\n\n")
        log.flush()

        child_env = os.environ.copy()
        child_env.setdefault("PYTHONUTF8", "1")
        child_env.setdefault("PYTHONIOENCODING", "utf-8")

        proc = subprocess.Popen(
            list(command),
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            env=child_env,
        )

        assert proc.stdout is not None

        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log.write(line)
            log.flush()

        code = proc.wait()

    return code, time.monotonic() - started


def _write_json(
    path: Path,
    data: Any,
) -> str:
    path.write_text(
        json.dumps(
            data,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )

    digest = sha256_file(path)
    path.with_suffix(
        path.suffix + ".sha256"
    ).write_text(digest + "\n", encoding="ascii")

    return digest


def find_metrics_file(
    profile: BaselineProfile,
    results_dir: Path,
) -> Path | None:
    candidates = profile.metrics_candidates or (profile.metrics_file,)

    # Prefer the newest matching file because current SS7 puts outputs under
    # <out-dir>/<fingerprint>_seed<seed>/.
    found: list[Path] = []
    for name in candidates:
        found.extend(results_dir.rglob(name))

    found = [path for path in found if path.is_file()]
    if not found:
        return None

    found.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return found[0]


def run_protocol(
    profile: BaselineProfile,
    run_root: Path,
    seeds: Sequence[int],
    regen: bool,
    allow_dirty: bool,
) -> dict[str, Any]:
    protocol_root = run_root / profile.name
    results_dir = protocol_root / "results"

    protocol_root.mkdir(
        parents=True,
        exist_ok=False,
    )
    results_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    git = git_state()

    if git["dirty"] and not allow_dirty:
        raise RuntimeError(
            "Git working tree is dirty. Commit/stash changes "
            "or rerun with --allow-dirty. With --allow-dirty "
            "the manifest records status and diff hashes."
        )

    config_defaults = default_config_snapshot(profile)

    command, effective_regen = build_command(
        profile=profile,
        seeds=seeds,
        results_dir=results_dir,
        regen=regen,
    )

    dataset_exists = profile.dataset_path.is_file()
    input_dataset = {
        "path": str(
            profile.dataset_path.relative_to(REPO_ROOT)
        ),
        "exists": dataset_exists,
        "size": (
            profile.dataset_path.stat().st_size
            if dataset_exists
            else None
        ),
        "sha256": (
            sha256_file(profile.dataset_path)
            if dataset_exists
            else None
        ),
    }

    experiment_config = {
        "protocol": profile.name,
        "seeds": list(map(int, seeds)),
        "regen_requested": bool(regen),
        "regen_effective": effective_regen,
        "dataset": profile.dataset,
        "default_configs": config_defaults,
    }

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "kind": "signal_network_sim_baseline",
        "status": "running",
        "created_at_utc": _utc_now(),
        "repository_root": str(REPO_ROOT),
        "protocol": profile.name,
        "git": git,
        "environment": environment_snapshot(),
        "source_sha256": source_hashes(profile),
        "experiment": experiment_config,
        "config_hash": canonical_hash(
            experiment_config
        ),
        "input_dataset": input_dataset,
        "command": list(command),
        "command_display": shlex.join(command),
        "cwd": str(profile.workdir_path),
    }

    _write_json(
        protocol_root / "manifest.partial.json",
        manifest,
    )

    exit_code, duration_s = run_streamed(
        command=command,
        cwd=profile.workdir_path,
        log_path=protocol_root / "run.log",
    )

    artifacts = artifact_hashes(results_dir)
    artifact_tree_sha256 = canonical_hash(artifacts)

    metrics_path = find_metrics_file(
        profile,
        results_dir,
    )

    if metrics_path is not None:
        metrics = read_metrics_csv(metrics_path)
        metrics["relative_source"] = str(
            metrics_path.relative_to(results_dir)
        )
    else:
        metrics = {
            "source": None,
            "searched_for": list(
                profile.metrics_candidates
                or (profile.metrics_file,)
            ),
            "missing": True,
            "rows": [],
            "row_count": 0,
        }

    metrics_digest = _write_json(
        protocol_root / "metrics.json",
        metrics,
    )

    manifest.update(
        {
            "status": (
                "ok"
                if exit_code == 0
                else "failed"
            ),
            "finished_at_utc": _utc_now(),
            "duration_seconds": duration_s,
            "exit_code": exit_code,
            "metrics_json_sha256": metrics_digest,
            "artifacts": artifacts,
            "artifact_tree_sha256": (
                artifact_tree_sha256
            ),
            "git_after": git_state(),
        }
    )

    _write_json(
        protocol_root / "manifest.json",
        manifest,
    )

    (
        protocol_root / "manifest.partial.json"
    ).unlink(missing_ok=True)
    (
        protocol_root / "manifest.partial.json.sha256"
    ).unlink(missing_ok=True)

    if exit_code != 0:
        raise RuntimeError(
            f"{profile.name}: baseline command failed "
            f"with exit code {exit_code}; see "
            f"{protocol_root / 'run.log'}"
        )

    return {
        "protocol": profile.name,
        "status": manifest["status"],
        "manifest": str(
            protocol_root / "manifest.json"
        ),
        "config_hash": manifest["config_hash"],
        "artifact_tree_sha256": (
            artifact_tree_sha256
        ),
        "metrics_json_sha256": (
            metrics_digest
        ),
    }


def _load_json(
    path: Path,
) -> dict[str, Any]:
    return json.loads(
        path.read_text(encoding="utf-8")
    )


def compare_metric_rows(
    a: list[dict[str, Any]],
    b: list[dict[str, Any]],
    atol: float,
) -> dict[str, Any]:
    preferred = ("seed", "name", "family")

    key_fields = tuple(
        key
        for key in preferred
        if a
        and b
        and key in a[0]
        and key in b[0]
    )

    def make_index(
        rows: list[dict[str, Any]],
    ) -> dict[Any, dict[str, Any]]:
        if key_fields:
            return {
                tuple(
                    row.get(key)
                    for key in key_fields
                ): row
                for row in rows
            }

        return {
            i: row
            for i, row in enumerate(rows)
        }

    ia = make_index(a)
    ib = make_index(b)

    only_a = sorted(
        set(ia) - set(ib),
        key=str,
    )
    only_b = sorted(
        set(ib) - set(ia),
        key=str,
    )

    common = set(ia) & set(ib)

    max_abs_delta = 0.0
    numeric_differences = 0
    textual_differences = 0

    for key in common:
        row_a = ia[key]
        row_b = ib[key]

        for column in set(row_a) | set(row_b):
            va = row_a.get(column)
            vb = row_b.get(column)

            numeric = (
                isinstance(va, (int, float))
                and not isinstance(va, bool)
                and isinstance(vb, (int, float))
                and not isinstance(vb, bool)
            )

            if numeric:
                delta = abs(
                    float(va) - float(vb)
                )
                max_abs_delta = max(
                    max_abs_delta,
                    delta,
                )

                if delta > atol:
                    numeric_differences += 1

            elif va != vb:
                textual_differences += 1

    equivalent = (
        not only_a
        and not only_b
        and numeric_differences == 0
        and textual_differences == 0
    )

    return {
        "key_fields": key_fields,
        "rows_a": len(a),
        "rows_b": len(b),
        "only_a": [
            str(value)
            for value in only_a
        ],
        "only_b": [
            str(value)
            for value in only_b
        ],
        "numeric_differences_gt_atol": (
            numeric_differences
        ),
        "textual_differences": (
            textual_differences
        ),
        "max_abs_numeric_delta": (
            max_abs_delta
        ),
        "atol": atol,
        "equivalent": equivalent,
    }


def compare_baselines(
    a_root: Path,
    b_root: Path,
    atol: float,
) -> dict[str, Any]:
    manifest_a = _load_json(
        a_root / "manifest.json"
    )
    manifest_b = _load_json(
        b_root / "manifest.json"
    )

    metrics_a = _load_json(
        a_root / "metrics.json"
    )
    metrics_b = _load_json(
        b_root / "metrics.json"
    )

    return {
        "protocol_a": (
            manifest_a.get("protocol")
        ),
        "protocol_b": (
            manifest_b.get("protocol")
        ),
        "same_protocol": (
            manifest_a.get("protocol")
            == manifest_b.get("protocol")
        ),
        "same_config_hash": (
            manifest_a.get("config_hash")
            == manifest_b.get("config_hash")
        ),
        "same_commit": (
            manifest_a
            .get("git", {})
            .get("commit")
            == manifest_b
            .get("git", {})
            .get("commit")
        ),
        "same_source_hashes": (
            manifest_a.get("source_sha256")
            == manifest_b.get("source_sha256")
        ),
        "same_input_dataset_hash": (
            manifest_a
            .get("input_dataset", {})
            .get("sha256")
            == manifest_b
            .get("input_dataset", {})
            .get("sha256")
        ),
        "same_artifact_tree_hash": (
            manifest_a.get(
                "artifact_tree_sha256"
            )
            == manifest_b.get(
                "artifact_tree_sha256"
            )
        ),
        "metrics": compare_metric_rows(
            metrics_a.get("rows", []),
            metrics_b.get("rows", []),
            atol,
        ),
    }


def _new_run_id() -> str:
    commit = "unknown"

    try:
        commit = str(
            _run_git(
                "rev-parse",
                "--short=8",
                "HEAD",
            )
        )
    except Exception:
        pass

    stamp = datetime.now(
        timezone.utc
    ).strftime("%Y%m%dT%H%M%SZ")

    return (
        f"{stamp}-{commit}-"
        f"{uuid.uuid4().hex[:6]}"
    )


def cmd_run(
    args: argparse.Namespace,
) -> int:
    protocols = (
        list(PROFILES)
        if args.protocol == ["all"]
        else args.protocol
    )

    seeds = (
        args.seeds
        or DEFAULT_SEEDS
    )

    run_id = (
        args.run_id
        or _new_run_id()
    )

    run_root = (
        REPO_ROOT
        / args.output_root
        / run_id
    ).resolve()

    run_root.mkdir(
        parents=True,
        exist_ok=False,
    )

    index: dict[str, Any] = {
        "schema_version": 1,
        "run_id": run_id,
        "created_at_utc": _utc_now(),
        "protocols": [],
    }

    _write_json(
        run_root / "index.partial.json",
        index,
    )

    try:
        for name in protocols:
            result = run_protocol(
                PROFILES[name],
                run_root=run_root,
                seeds=seeds,
                regen=args.regen,
                allow_dirty=args.allow_dirty,
            )

            index["protocols"].append(
                result
            )

            _write_json(
                run_root / "index.partial.json",
                index,
            )
    finally:
        index["finished_at_utc"] = (
            _utc_now()
        )

        _write_json(
            run_root / "index.json",
            index,
        )

        (
            run_root / "index.partial.json"
        ).unlink(missing_ok=True)
        (
            run_root
            / "index.partial.json.sha256"
        ).unlink(missing_ok=True)

    print(
        f"\nBaseline saved to: "
        f"{run_root}"
    )

    return 0


def cmd_compare(
    args: argparse.Namespace,
) -> int:
    report = compare_baselines(
        Path(args.a).resolve(),
        Path(args.b).resolve(),
        atol=args.atol,
    )

    print(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )

    return (
        0
        if report["metrics"]["equivalent"]
        else 1
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Stage 0 baseline recorder "
            "for signal_network_sim"
        )
    )

    sub = parser.add_subparsers(
        dest="command",
        required=True,
    )

    run = sub.add_parser(
        "run",
        help=(
            "run and archive "
            "the current baseline"
        ),
    )

    run.add_argument(
        "--protocol",
        action="append",
        choices=[
            *PROFILES.keys(),
            "all",
        ],
        default=None,
        help=(
            "protocol to run; repeat "
            "for multiple protocols; "
            "default: all"
        ),
    )

    run.add_argument(
        "--seeds",
        nargs="*",
        type=int,
        default=None,
    )

    run.add_argument(
        "--regen",
        action="store_true",
        help=(
            "regenerate per-seed datasets "
            "where the legacy runner supports it"
        ),
    )

    run.add_argument(
        "--allow-dirty",
        action="store_true",
        help=(
            "allow uncommitted changes; "
            "status and diff hashes are recorded"
        ),
    )

    run.add_argument(
        "--run-id",
        default=None,
    )

    run.add_argument(
        "--output-root",
        default="runs/baseline",
        help=(
            "path relative "
            "to repository root"
        ),
    )

    run.set_defaults(
        func=cmd_run
    )

    compare = sub.add_parser(
        "compare",
        help=(
            "compare two protocol "
            "baseline directories"
        ),
    )

    compare.add_argument(
        "a",
        help=(
            "e.g. runs/baseline/"
            "RUN_A/ss7"
        ),
    )

    compare.add_argument(
        "b",
        help=(
            "e.g. runs/baseline/"
            "RUN_B/ss7"
        ),
    )

    compare.add_argument(
        "--atol",
        type=float,
        default=1e-12,
    )

    compare.set_defaults(
        func=cmd_compare
    )

    return parser


def main(
    argv: Sequence[str] | None = None,
) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if (
        args.command == "run"
        and args.protocol is None
    ):
        args.protocol = ["all"]

    if (
        args.command == "run"
        and "all" in args.protocol
        and len(args.protocol) > 1
    ):
        parser.error(
            "--protocol all cannot be combined "
            "with another --protocol"
        )

    return int(
        args.func(args)
    )


if __name__ == "__main__":
    raise SystemExit(main())
