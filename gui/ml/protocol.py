"""Job contract and event protocol for model experiments."""
from __future__ import annotations

import json
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
    models: list[str] = field(default_factory=list)
    test_datasets: list[str] = field(default_factory=list)
    capture_models: bool = True
    threads: int = 0

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=2)

    @classmethod
    def from_json(cls, text: str) -> "JobSpec":
        return cls(**json.loads(text.lstrip("\ufeff")))


def emit(event: str, **payload: Any) -> None:
    print(EVENT_PREFIX + json.dumps({"event": event, **payload},
                                    ensure_ascii=False, default=str),
          flush=True)


def parse(line: str) -> Optional[dict[str, Any]]:
    if not line.startswith(EVENT_PREFIX):
        return None
    try:
        return json.loads(line[len(EVENT_PREFIX):])
    except ValueError:
        return None
