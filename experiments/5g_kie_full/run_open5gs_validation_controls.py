"""Predeclared negative controls for independent Open5GS NRF witness validation.

Runs only AFTER the 12 factorial interventions; controls are never KIE mask
conditions, and deliberate low-rate traffic stays below 20 GET per 2s.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import subprocess
import time

from run_open5gs_randomized_scenarios import atomic_control, endpoint


def run(directory:Path, *, idle_seconds:float=8.0, benign_requests:int=10,
        spacing_seconds:float=0.8, repeats:int=2)->None:
    if idle_seconds<4.0 or benign_requests>12 or spacing_seconds<0.5 or repeats!=2:
        raise ValueError("predeclared negative controls changed")
    url_host,url_port=endpoint(directory/"open5gs_endpoints.json")
    target=f"http://{url_host}:{url_port}/nnrf-nfm/v1/nf-instances"
    atomic_control(directory/"kie_control.json","independent_validation_controls",None)
    path=directory/"independent_negative_controls.csv"
    with path.open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=[
            "block","control_type","start_ts","end_ts","attempted","successful"
        ])
        writer.writeheader()
        for repeat in range(1,repeats+1):
            begin=time.time()
            time.sleep(idle_seconds)
            writer.writerow({
                "block":repeat,"control_type":"idle","start_ts":f"{begin:.6f}",
                "end_ts":f"{time.time():.6f}","attempted":0,"successful":0
            })
            f.flush()
            begin=time.time();successful=0
            for _ in range(benign_requests):
                result=subprocess.run(
                    ["curl","--http2-prior-knowledge","-sS","--max-time","4",
                     "-o","/dev/null",target],timeout=6,check=False
                )
                successful+=(result.returncode==0)
                time.sleep(spacing_seconds)
            writer.writerow({
                "block":repeat,"control_type":"benign_low_rate",
                "start_ts":f"{begin:.6f}","end_ts":f"{time.time():.6f}",
                "attempted":benign_requests,"successful":successful
            })
            f.flush()
            if successful!=benign_requests:
                raise ValueError(f"incomplete independent control: {successful}/{benign_requests}")
    print(path.read_text(),flush=True)


def main()->None:
    p=argparse.ArgumentParser()
    p.add_argument("--output-dir",type=Path,required=True)
    args=p.parse_args()
    run(args.output_dir)


if __name__=="__main__":
    main()
