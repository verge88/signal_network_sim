"""Constant-rate genuine HTTP/2 GET background load for Open5GS NRF.

Uses curl HTTP/2 prior knowledge (same transport as interventions) and logs
every request timestamp / outcome. This is BENIGN traffic, not an attack.
"""
from __future__ import annotations
import argparse
import csv
from pathlib import Path
import signal
import subprocess
import time

_SHUTDOWN = False

def stop(_signal: int, _frame: object) -> None:
    global _SHUTDOWN
    _SHUTDOWN = True

def run(url: str, rps: float, log_path: Path) -> None:
    if rps <= 0 or rps > 5:
        raise ValueError("background rps must be >0 and <=5")
    log_path.parent.mkdir(parents=True,exist_ok=True)
    step = 1.0 / rps
    deadline = time.monotonic()
    with log_path.open("w",newline="",encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["ts","curl_exit_code","duration_s"])
        stream.flush()
        while not _SHUTDOWN:
            now=time.monotonic()
            if now < deadline:
                time.sleep(min(0.1,deadline-now))
                continue
            started=time.time()
            p=subprocess.run(
                ["curl","--http2-prior-knowledge","-sS","--max-time","4",
                 "-o","/dev/null",url],
                check=False,timeout=6,
            )
            writer.writerow([f"{started:.6f}",p.returncode,
                             f"{time.time()-started:.6f}"])
            stream.flush()
            # If system load delays traffic, do not issue catch-up bursts.
            deadline=max(deadline+step,time.monotonic())

def main() -> None:
    p=argparse.ArgumentParser()
    p.add_argument("--url",required=True)
    p.add_argument("--rps",type=float,required=True)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    signal.signal(signal.SIGTERM,stop)
    signal.signal(signal.SIGINT,stop)
    run(args.url,args.rps,args.output)

if __name__=="__main__":
    main()
