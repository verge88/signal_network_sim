"""Job contract and event protocol for model experiments."""
from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

EVENT_PREFIX = "@@SNS "
STAGES = ("simulate", "train", "evaluate", "persist")


@dataclass
class JobSpec:
    job_id: str
    protocol: str
    sim_dir: str
    run_dir: str
    seed: int = 42
    make_dataset: bool = True
    topology_path: str = ""
    dataset_path: str = ""
    sim_module: Optional[str] = None
    do_train: bool = True
    train_module: Optional[str] = None
    train_overrides: dict[str, Any] = field(default_factory=dict)
    cli_args: list[str] = field(default_factory=list)
    train_script: Optional[str] = None
    train_options: dict[str, Any] = field(default_factory=dict)
    train_extra_args: str = ""
    models: list[str] = field(default_factory=list)
    test_datasets: list[str] = field(default_factory=list)
    capture_models: bool = True
    threads: int = 0

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=2)

    @classmethod
    def from_json(cls, text: str) -> "JobSpec":
        return cls(**json.loads(text.lstrip("\ufeff")))


def _write_line(stream, line: str) -> None:
    """Write one line without allowing output errors to break the job."""
    if stream is None:
        return
    try:
        stream.write(line + "\n")
        stream.flush()
    except UnicodeEncodeError:
        encoding = getattr(stream, "encoding", None) or "utf-8"
        try:
            stream.write(line.encode(encoding, "replace").decode(encoding, "replace") + "\n")
            stream.flush()
        except Exception:
            pass
    except Exception:
        pass


def emit(event: str, **payload: Any) -> None:
    try:
        body = json.dumps({"event": event, **payload}, ensure_ascii=False,
                          default=str)
    except Exception:
        body = json.dumps({"event": event, "payload": "<не сериализуется>"})
    _write_line(sys.__stdout__ or sys.stdout, EVENT_PREFIX + body)


def parse(line: str) -> Optional[dict[str, Any]]:
    if not line.startswith(EVENT_PREFIX):
        return None
    try:
        return json.loads(line[len(EVENT_PREFIX):])
    except ValueError:
        return None
