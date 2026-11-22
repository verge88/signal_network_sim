from experiments.distributed_5g.node_trust_replay import CONTROL_NF, assess, episode


def test_trust_lifecycle():
    rows = list(episode())
    assert len({(r["window"], r["nf"]) for r in rows}) == len(rows) == 19 * 32
    assert rows == list(episode())
    target = [r for r in rows if r["nf"] == CONTROL_NF]
    assert [r["decision"] for r in target] == ["consistent"] * 8 + ["alarm"] * 6 + ["unknown"] * 2 + ["consistent"] * 3
    assert sum(r["compromised"] for r in target) == 10
    assert all(r["decision"] == "outside_scope" and not r["compromised"]
               for r in rows if r["nf"] != CONTROL_NF)


def test_external_evidence():
    assert assess("A", "B", "A") == "alarm"
    assert assess("A", "A", None) == "unknown"
    assert assess("A", "A", "A") == "consistent"
