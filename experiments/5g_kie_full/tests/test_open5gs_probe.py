from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from open5gs_probe import discover_endpoints


def test_discovers_sbi_servers_from_open5gs_yaml(tmp_path: Path):
    (tmp_path / "nrf.yaml").write_text(
        """
nrf:
  sbi:
    server:
      - address: 127.0.0.10
        port: 7777
""".strip(),
        encoding="utf-8",
    )
    (tmp_path / "amf.yaml").write_text(
        """
amf:
  sbi:
    server:
      address: 127.0.0.5
      port: 7777
""".strip(),
        encoding="utf-8",
    )

    rows = discover_endpoints(tmp_path)
    got = {(r["nf"], r["address"], r["port"]) for r in rows}
    assert ("nrf", "127.0.0.10", 7777) in got
    assert ("amf", "127.0.0.5", 7777) in got


def test_ignores_non_sbi_config(tmp_path: Path):
    (tmp_path / "upf.yaml").write_text(
        """
upf:
  pfcp:
    server:
      - address: 127.0.0.7
""".strip(),
        encoding="utf-8",
    )
    assert discover_endpoints(tmp_path) == []
