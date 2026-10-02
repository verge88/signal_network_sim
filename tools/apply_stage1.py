from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

DIAMETER_MASK = '    def _masked_report(self, true: Dict, node: DiameterNode,\n                       slave_id: int, interval_idx: int) -> Dict:\n        """Forge benign-looking local telemetry without a hard class signature."""\n        history = self.honest_history.get(slave_id)\n        if history:\n            recent = list(history)[-min(20, len(history)):]\n            base = dict(recent[int(self.rng.integers(0, len(recent)))])\n            return self._honest_report(base)\n        return self._nominal_counterfactual_report(node, interval_idx)\n\n    def _update_honest_history(self, slave_id: int, true: Dict) -> None:\n        self.honest_history[slave_id].append({\n            str(k): v for k, v in true.items() if not str(k).startswith("_")\n        })\n\n    def _nominal_counterfactual_report(\n        self, node: DiameterNode, interval_idx: int\n    ) -> Dict:\n        t_seconds = interval_idx * self.config.interval_s\n        hour = (t_seconds / 3600.0) % 24.0\n        h0 = int(hour) % 24\n        h1 = (h0 + 1) % 24\n        frac = hour - int(hour)\n        profile = DiameterTrafficGenerator.TOD_PROFILE\n        tod = profile[h0] * (1.0 - frac) + profile[h1] * frac\n        total = max(1, int(self.rng.poisson(max(1.0, node.base_rate * tod))))\n\n        names = ["S6a", "Gx", "Rx", "S13", "Other"]\n        p = np.array(\n            [max(0.0, float(node.interface_dist.get(k, 0.0))) for k in names],\n            dtype=float,\n        )\n        if p.sum() <= 0:\n            p = np.ones(len(names), dtype=float)\n        p /= p.sum()\n        counts = self.rng.multinomial(total, p)\n\n        s6a_p = np.array(\n            [max(0.0, float(node.s6a_command_dist.get(k, 0.0)))\n             for k in S6A_COMMANDS],\n            dtype=float,\n        )\n        if s6a_p.sum() <= 0:\n            s6a_p = np.ones(len(S6A_COMMANDS), dtype=float)\n        s6a_p /= s6a_p.sum()\n\n        gx_p = np.array(\n            [max(0.0, float(node.gx_command_dist.get(k, 0.0)))\n             for k in GX_COMMANDS],\n            dtype=float,\n        )\n        if gx_p.sum() <= 0:\n            gx_p = np.ones(len(GX_COMMANDS), dtype=float)\n        gx_p /= gx_p.sum()\n\n        nz = p[p > 0]\n        ent = float(-(nz * np.log2(nz)).sum()) if len(nz) else 0.0\n        ordered = np.sort(s6a_p)[::-1]\n        dominance = float(ordered[0] / max(1e-9, ordered[1:].sum()))\n        top2 = float(ordered[:2].sum() / max(1e-9, ordered[2:].sum()))\n\n        arr = np.sort(s6a_p)\n        idx = np.arange(1, len(arr) + 1)\n        gini = float(\n            (2.0 * np.sum(idx * arr) - (len(arr) + 1) * arr.sum())\n            / (len(arr) * arr.sum())\n        ) if arr.sum() > 0 else 0.0\n\n        base = {\n            "total_messages": total,\n            "s6a_count": int(counts[0]),\n            "gx_count": int(counts[1]),\n            "rx_count": int(counts[2]),\n            "s13_count": int(counts[3]),\n            "other_count": int(counts[4]),\n            "entropy": ent,\n            "inbound_outbound_ratio": float(0.8 + 0.4 * self.rng.random()),\n            "international_fraction": float(0.05 + 0.03 * self.rng.random()),\n            "n_unique_peers": max(5, int(self.rng.normal(20, 6))),\n            "s6a_dominance_ratio": dominance,\n            "s6a_top2_ratio": top2,\n            "s6a_gini": gini,\n        }\n        for cmd, prob in zip(S6A_COMMANDS, s6a_p):\n            base[f"ratio_s6a_{cmd.lower()}"] = float(prob)\n        for cmd, prob in zip(GX_COMMANDS, gx_p):\n            base[f"ratio_gx_{cmd.lower()}"] = float(prob)\n        return self._honest_report(base)\n\n'


def repo_root() -> Path:
    here = Path.cwd().resolve()
    for p in (here, *here.parents):
        if (p / "ss7").is_dir() and (p / "diameter").is_dir() and (p / "sip").is_dir():
            return p
    raise RuntimeError("run from signal_network_sim or a child directory")


def backup(path: Path, root: Path) -> None:
    dst = root / ".stage1_backup" / path.relative_to(root)
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, dst)


def patch_baseline(path: Path) -> bool:
    text = path.read_text(encoding="utf-8")
    before = text
    old = """        proc = subprocess.Popen(
            list(command),
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=os.environ.copy(),
        )"""
    new = """        child_env = os.environ.copy()
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
        )"""
    if old in text:
        text = text.replace(old, new, 1)
    elif 'child_env.setdefault("PYTHONIOENCODING", "utf-8")' not in text:
        raise RuntimeError("baseline.py: expected Stage-0 v2 run_streamed() not found")
    if text != before:
        path.write_text(text, encoding="utf-8")
        return True
    return False


def patch_diameter(path: Path) -> bool:
    text = path.read_text(encoding="utf-8")
    before = text

    if "from collections import defaultdict, deque" not in text:
        text = text.replace(
            "from collections import defaultdict",
            "from collections import defaultdict, deque",
            1,
        )

    text = text.replace(
        """        # Processing delay (ms → seconds for consistency)
        processing_delay_ms = self.cu.processing_delay_ms
        if node.is_compromised:
            processing_delay_ms += self.rng.uniform(0.5, 3.0)
""",
        """        # Compromise alone MUST NOT create an observable delay signature.
        processing_delay_ms = self.cu.processing_delay_ms
""",
        1,
    )

    text = text.replace(
        """        integrity_ok = True
        if cu_delivered:
            fail_prob = self.cu.integrity_background_fail_prob
            if node.is_compromised:
                fail_prob = 0.30
            integrity_ok = self.rng.random() > fail_prob
""",
        """        integrity_ok = True
        if cu_delivered:
            fail_prob = self.cu.integrity_background_fail_prob
            integrity_ok = self.rng.random() > fail_prob
""",
        1,
    )

    anchor = "        self.response_history_window = 20\n"
    added = """        self.response_history_window = 20
        self.honest_history: Dict[int, deque] = defaultdict(
            lambda: deque(maxlen=60)
        )
"""
    if added not in text:
        if anchor not in text:
            raise RuntimeError("diameter: response_history_window anchor not found")
        text = text.replace(anchor, added, 1)

    old_select = """        if cu_delivered:
            if node.is_compromised:
                reported = self._masked_report(true_traffic, node,
                                               interval_idx)
            else:
                reported = self._honest_report(true_traffic)
"""
    new_select = """        if cu_delivered:
            if node.is_compromised:
                reported = self._masked_report(
                    true_traffic, node, slave_id, interval_idx
                )
            else:
                self._update_honest_history(slave_id, true_traffic)
                reported = self._honest_report(true_traffic)
"""
    if old_select in text:
        text = text.replace(old_select, new_select, 1)
    elif "true_traffic, node, slave_id, interval_idx" not in text:
        raise RuntimeError("diameter: report selection block not found")

    pattern = re.compile(
        r"    def _masked_report\(self, true: Dict, node: DiameterNode,\n"
        r"                       interval_idx: int\) -> Dict:\n"
        r".*?(?=    def _empty_report\(self\) -> Dict:)",
        re.S,
    )
    if pattern.search(text):
        text = pattern.sub(DIAMETER_MASK, text, count=1)
    elif "def _nominal_counterfactual_report" not in text:
        raise RuntimeError("diameter: _masked_report block not found")

    text = text.replace(
        "interval_idx - (node.compromised_since or interval_idx)",
        "interval_idx - (node.compromised_since if "
        "node.compromised_since is not None else interval_idx)",
    )

    if text != before:
        path.write_text(text, encoding="utf-8")
        return True
    return False


def patch_sip(path: Path) -> bool:
    text = path.read_text(encoding="utf-8")
    before = text

    delay = re.compile(
        r"        processing_delay = self\.cu\.processing_delay_ms\n"
        r"        if node\.is_compromised:\n"
        r"            cr = node\.covert_redirect_fraction\n"
        r"            soph = node\.attacker_sophistication\n"
        r"            phys_leak = \{\n.*?"
        r"            processing_delay \+= phys_leak \* \(\n"
        r"                self\.rng\.uniform\(0\.1, 0\.5\) \+ 1\.2 \* cr\)\n",
        re.S,
    )
    text = delay.sub(
        """        # Compromise alone does not change processing delay.
        processing_delay = self.cu.processing_delay_ms
""",
        text,
        count=1,
    )

    integ = re.compile(
        r"        integrity_ok = True\n"
        r"        if cu_delivered:\n"
        r"            if node\.is_compromised:\n"
        r"                soph = node\.attacker_sophistication\n"
        r"                fail_prob = \{\n.*?"
        r"                \}\[soph\]\n"
        r"            else:\n"
        r"                fail_prob = self\.cu\.integrity_background_fail_prob\n"
        r"            integrity_ok = self\.rng\.random\(\) > fail_prob\n",
        re.S,
    )
    text = integ.sub(
        """        integrity_ok = True
        if cu_delivered:
            # Key custody is not modelled here yet, so compromise alone
            # must not change the integrity-failure distribution.
            fail_prob = self.cu.integrity_background_fail_prob
            integrity_ok = self.rng.random() > fail_prob
""",
        text,
        count=1,
    )

    text = text.replace(
        "interval_idx - (node.compromised_since or interval_idx)",
        "interval_idx - (node.compromised_since if "
        "node.compromised_since is not None else interval_idx)",
    )

    if text != before:
        path.write_text(text, encoding="utf-8")
        return True
    return False


def patch_legacy_ss7(path: Path) -> bool:
    text = path.read_text(encoding="utf-8")
    before = text
    text = text.replace(
        "interval_idx - (node.compromised_since or interval_idx)",
        "interval_idx - (node.compromised_since if "
        "node.compromised_since is not None else interval_idx)",
    )
    if text != before:
        path.write_text(text, encoding="utf-8")
        return True
    return False


def validate_sources(root: Path) -> list[str]:
    problems = []
    diameter = (root / "diameter" / "diameter_sim_v1.py").read_text(encoding="utf-8")
    sip = (root / "sip" / "sip_sim_v4_claude.py").read_text(encoding="utf-8")
    legacy = (root / "ss7" / "ss7_simulator_v7.py").read_text(encoding="utf-8")

    checks = [
        ("diameter", diameter, [
            "processing_delay_ms += self.rng.uniform(0.5, 3.0)",
            "fail_prob = 0.30",
            "+ drift + 0.8",
            "(node.compromised_since or interval_idx)",
        ]),
        ("sip", sip, [
            "phys_leak = {",
            "(node.compromised_since or interval_idx)",
        ]),
        ("ss7_legacy", legacy, [
            "(node.compromised_since or interval_idx)",
        ]),
    ]
    for name, source, needles in checks:
        for needle in needles:
            if needle in source:
                problems.append(f"{name}: forbidden signature remains: {needle}")

    if "def _nominal_counterfactual_report" not in diameter:
        problems.append("diameter: counterfactual masking not installed")

    baseline = root / "experiments" / "baseline.py"
    if baseline.exists():
        b = baseline.read_text(encoding="utf-8")
        if 'PYTHONIOENCODING", "utf-8"' not in b:
            problems.append("baseline: UTF-8 child-process fix missing")

    if not (root / "ss7" / "canonical.py").exists():
        problems.append("ss7/canonical.py missing")

    return problems


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Apply Stage 1 scientific-correctness patches"
    )
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    root = repo_root()

    if a.apply:
        targets = [
            root / "experiments" / "baseline.py",
            root / "diameter" / "diameter_sim_v1.py",
            root / "sip" / "sip_sim_v4_claude.py",
            root / "ss7" / "ss7_simulator_v7.py",
        ]
        for p in targets:
            if not p.exists():
                raise FileNotFoundError(p)
            backup(p, root)

        changes = {
            "baseline": patch_baseline(targets[0]),
            "diameter": patch_diameter(targets[1]),
            "sip": patch_sip(targets[2]),
            "ss7_legacy": patch_legacy_ss7(targets[3]),
        }
        for name, changed in changes.items():
            print(f"{name:12s}: {'updated' if changed else 'already OK'}")

    problems = validate_sources(root)
    if problems:
        print("\nStage 1 validation: FAILED")
        for p in problems:
            print(" -", p)
        return 2

    print("\nStage 1 validation: OK")
    print("Scientific SS7 source of truth: ss7/fix.py via ss7/canonical.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
