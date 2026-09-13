"""Subprocess queue with streaming events and cancellation."""
from __future__ import annotations

import os
import queue
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

from .protocol import JobSpec, parse


@dataclass
class JobState:
    spec: JobSpec
    status: str = "queued"
    stage: str = ""
    progress: float = 0.0
    started: float = 0.0
    finished: float = 0.0
    error: str = ""
    proc: Optional[subprocess.Popen] = field(default=None, repr=False)

    @property
    def elapsed(self) -> float:
        end = self.finished or (time.time() if self.status == "running" else 0.0)
        return max(0.0, end - self.started) if self.started else 0.0


class JobRunner:
    def __init__(self, repo_root: str, max_parallel: int = 1):
        self.repo_root = os.path.abspath(repo_root)
        self.max_parallel = max(1, max_parallel)
        self.jobs: list[JobState] = []
        self.events: "queue.Queue[tuple[str, JobState, dict]]" = queue.Queue()
        self._lock = threading.Lock()
        self._stop = False
        threading.Thread(target=self._loop, daemon=True).start()

    def submit(self, spec: JobSpec) -> JobState:
        state = JobState(spec=spec)
        with self._lock:
            self.jobs.append(state)
        self.events.put(("queued", state, {}))
        return state

    def cancel(self, state: JobState) -> None:
        if state.status == "queued":
            state.status = "cancelled"
            self.events.put(("done", state, {"status": "cancelled"}))
        elif state.status == "running" and state.proc:
            self._kill(state.proc)
            state.status = "cancelled"

    def cancel_all(self) -> None:
        for state in list(self.jobs):
            if state.status in ("queued", "running"):
                self.cancel(state)

    def shutdown(self) -> None:
        self._stop = True
        self.cancel_all()

    @staticmethod
    def _kill(proc: subprocess.Popen) -> None:
        try:
            if sys.platform == "win32":
                subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                               capture_output=True, check=False)
            else:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except Exception:  # noqa: BLE001
            proc.kill()

    def _loop(self) -> None:
        while not self._stop:
            with self._lock:
                running = sum(item.status == "running" for item in self.jobs)
                next_job = next((item for item in self.jobs if item.status == "queued"), None)
            if next_job is not None and running < self.max_parallel:
                next_job.status = "running"
                threading.Thread(target=self._run_one, args=(next_job,), daemon=True).start()
            else:
                time.sleep(0.2)

    def _run_one(self, state: JobState) -> None:
        spec = state.spec
        os.makedirs(spec.run_dir, exist_ok=True)
        config_path = os.path.join(spec.run_dir, "job.json")
        with open(config_path, "w", encoding="utf-8") as stream:
            stream.write(spec.to_json())
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join(
            [self.repo_root, env.get("PYTHONPATH", "")]).strip(os.pathsep)
        env["PYTHONUNBUFFERED"] = "1"
        env["MPLBACKEND"] = "Agg"
        kwargs: dict = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if sys.platform == "win32" else {"start_new_session": True}
        state.started = time.time()
        self.events.put(("started", state, {}))
        try:
            state.proc = subprocess.Popen(
                [sys.executable, "-u", "-m", "gui.ml.harness", config_path],
                cwd=self.repo_root, env=env, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, errors="replace", bufsize=1,
                **kwargs)
        except OSError as exc:
            state.status, state.error = "error", str(exc)
            state.finished = time.time()
            self.events.put(("done", state, {"status": "error"}))
            return
        assert state.proc.stdout is not None
        for line in state.proc.stdout:
            line = line.rstrip("\n")
            event = parse(line)
            if event is None:
                self.events.put(("log", state, {"line": line}))
                continue
            kind = event.get("event", "log")
            if kind == "stage":
                state.stage = f"{event.get('name')}:{event.get('status')}"
            elif kind == "progress":
                state.progress = float(event.get("value", 0.0))
            elif kind == "failed":
                state.error = str(event.get("error", ""))
            self.events.put((kind, state, event))
        code = state.proc.wait()
        state.finished = time.time()
        if state.status != "cancelled":
            state.status = "ok" if code == 0 else "error"
            if code != 0 and not state.error:
                state.error = f"процесс завершился с кодом {code}"
        self.events.put(("done", state, {"status": state.status}))

    def drain(self, handler: Callable[[str, JobState, dict], None], limit: int = 400) -> None:
        for _ in range(limit):
            try:
                kind, state, payload = self.events.get_nowait()
            except queue.Empty:
                return
            handler(kind, state, payload)
