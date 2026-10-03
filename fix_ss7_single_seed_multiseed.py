from pathlib import Path
import shutil

def find_repo_root() -> Path:
    p = Path.cwd().resolve()
    for candidate in (p, *p.parents):
        target = candidate / "ss7" / "ss7_training_v7.py"
        if target.exists():
            return candidate
    raise RuntimeError("Не найден signal_network_sim/ss7/ss7_training_v7.py")

def main() -> int:
    root = find_repo_root()
    path = root / "ss7" / "ss7_training_v7.py"
    backup = root / ".stage1_backup" / "ss7" / "ss7_training_v7.before_single_seed_fix.py"
    backup.parent.mkdir(parents=True, exist_ok=True)
    if not backup.exists():
        shutil.copy2(path, backup)

    text = path.read_text(encoding="utf-8")
    original = text

    old_bootstrap = 'def bootstrap_paired(res: pd.DataFrame, a: str, b: str, metric: str,\n                     n_boot: int = 10000, seed: int = 0) -> Dict[str, float]:\n    pa = res[res.model == a].set_index("seed")[metric]\n    pb = res[res.model == b].set_index("seed")[metric]\n    common = pa.index.intersection(pb.index)\n    d = (pa.loc[common] - pb.loc[common]).dropna().to_numpy(float)\n    if len(d) < 3:\n        return {"n": len(d), "p": np.nan}\n    rng = np.random.default_rng(seed)\n    bs = rng.choice(d, (n_boot, len(d)), replace=True).mean(1)\n    p = 2 * min((bs <= 0).mean(), (bs >= 0).mean())\n    return {"n": int(len(d)), "mean_diff": float(d.mean()),\n            "sd": float(d.std(ddof=1)),\n            "ci_lo": float(np.percentile(bs, 2.5)),\n            "ci_hi": float(np.percentile(bs, 97.5)),\n            "p": float(min(p, 1.0)),\n            "mde_95": float(1.96 * d.std(ddof=1) / math.sqrt(len(d)))}\n'
    new_bootstrap = 'def bootstrap_paired(res: pd.DataFrame, a: str, b: str, metric: str,\n                     n_boot: int = 10000, seed: int = 0) -> Dict[str, float]:\n    """Парный bootstrap по seed с устойчивым schema результата."""\n    pa = res[res.model == a].set_index("seed")[metric]\n    pb = res[res.model == b].set_index("seed")[metric]\n    common = pa.index.intersection(pb.index)\n    d = (pa.loc[common] - pb.loc[common]).dropna().to_numpy(float)\n\n    if len(d) < 3:\n        return {\n            "n": int(len(d)),\n            "mean_diff": float(d.mean()) if len(d) else np.nan,\n            "sd": float(d.std(ddof=1)) if len(d) > 1 else np.nan,\n            "ci_lo": np.nan,\n            "ci_hi": np.nan,\n            "p": np.nan,\n            "mde_95": np.nan,\n            "enough_seeds": False,\n        }\n\n    rng = np.random.default_rng(seed)\n    bs = rng.choice(d, (n_boot, len(d)), replace=True).mean(1)\n    p = 2 * min((bs <= 0).mean(), (bs >= 0).mean())\n\n    return {\n        "n": int(len(d)),\n        "mean_diff": float(d.mean()),\n        "sd": float(d.std(ddof=1)),\n        "ci_lo": float(np.percentile(bs, 2.5)),\n        "ci_hi": float(np.percentile(bs, 97.5)),\n        "p": float(min(p, 1.0)),\n        "mde_95": float(1.96 * d.std(ddof=1) / math.sqrt(len(d))),\n        "enough_seeds": True,\n    }\n'
    old_print = '        if not tests.empty:\n            print("\\n     Бутстрэп по сидам + поправка Холма:")\n            print(tests[["comparison", "n", "mean_diff", "ci_lo", "ci_hi", "p",\n                         "holm_threshold", "significant",\n                         "mde_95"]].to_string(index=False))\n'
    new_print = '        if not tests.empty:\n            enough = (\n                "enough_seeds" in tests.columns\n                and tests["enough_seeds"].fillna(False).any()\n            )\n\n            if enough:\n                print("\\n     Бутстрэп по сидам + поправка Холма:")\n                wanted = [\n                    "comparison", "n", "mean_diff", "ci_lo", "ci_hi", "p",\n                    "holm_threshold", "significant", "mde_95",\n                ]\n                print(\n                    tests[[c for c in wanted if c in tests.columns]]\n                    .to_string(index=False)\n                )\n            else:\n                max_n = (\n                    int(tests["n"].max())\n                    if "n" in tests.columns and tests["n"].notna().any()\n                    else 0\n                )\n                print(\n                    "\\n     Статистические сравнения пропущены: "\n                    f"доступно только {max_n} парных seed; "\n                    "для bootstrap требуется минимум 3, "\n                    "для диссертационного вывода рекомендуется >= 10."\n                )\n'

    if old_bootstrap in text:
        text = text.replace(old_bootstrap, new_bootstrap, 1)
    elif '"enough_seeds": False' not in text:
        raise RuntimeError("Не найден ожидаемый bootstrap_paired().")

    if old_print in text:
        text = text.replace(old_print, new_print, 1)
    elif "Статистические сравнения пропущены" not in text:
        raise RuntimeError("Не найден ожидаемый блок печати multiseed.")

    if text != original:
        path.write_text(text, encoding="utf-8")
        compile(text, str(path), "exec")
        print("Исправлено:", path)
        print("Резервная копия:", backup)
        print("Синтаксис: OK")
    else:
        print("Исправление уже применено.")

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
