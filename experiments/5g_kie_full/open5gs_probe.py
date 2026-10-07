from __future__ import annotations

import argparse
import json
import socket
from pathlib import Path
from typing import Any

import yaml


def _as_server_list(server: Any) -> list[dict]:
    if server is None:
        return []
    if isinstance(server, dict):
        return [server]
    if isinstance(server, list):
        return [x for x in server if isinstance(x, dict)]
    return []


def discover_endpoints(config_dir: Path) -> list[dict]:
    """Discover SBI server endpoints from installed Open5GS YAML configs."""
    endpoints: list[dict] = []
    for path in sorted(config_dir.glob("*.yaml")):
        try:
            doc = yaml.safe_load(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(doc, dict):
            continue

        for nf_name, nf_cfg in doc.items():
            if not isinstance(nf_cfg, dict):
                continue
            sbi = nf_cfg.get("sbi")
            if not isinstance(sbi, dict):
                continue
            for server in _as_server_list(sbi.get("server")):
                address = server.get("address")
                port = server.get("port", 7777)
                if isinstance(address, list):
                    addresses = address
                else:
                    addresses = [address]
                for addr in addresses:
                    if not addr:
                        continue
                    try:
                        port_i = int(port)
                    except (TypeError, ValueError):
                        port_i = 7777
                    endpoints.append(
                        {
                            "nf": str(nf_name),
                            "address": str(addr),
                            "port": port_i,
                            "config": str(path),
                        }
                    )

    dedup: dict[tuple[str, str, int], dict] = {}
    for item in endpoints:
        key = (item["nf"], item["address"], item["port"])
        dedup[key] = item
    return list(dedup.values())


def tcp_probe(address: str, port: int, timeout: float) -> tuple[bool, str | None]:
    try:
        with socket.create_connection((address, port), timeout=timeout):
            return True, None
    except OSError as exc:
        return False, str(exc)


def main() -> None:
    p = argparse.ArgumentParser(description="Probe live Open5GS SBI endpoints")
    p.add_argument("--config-dir", type=Path, default=Path("/etc/open5gs"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--timeout", type=float, default=1.0)
    args = p.parse_args()

    endpoints = discover_endpoints(args.config_dir)
    rows = []
    for item in endpoints:
        ok, error = tcp_probe(item["address"], item["port"], args.timeout)
        rows.append({**item, "reachable": bool(ok), "error": error})

    summary = {
        "config_dir": str(args.config_dir),
        "endpoint_count": len(rows),
        "reachable_count": sum(1 for r in rows if r["reachable"]),
        "endpoints": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
