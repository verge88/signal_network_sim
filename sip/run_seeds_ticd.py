"""
run_seeds_ticd.py — мультисид. Агрегирует метрики (mean±std) по 10 seed,
Вилкоксон TICD vs unsupervised, сохраняет aggregated.pkl для таблиц/графиков.
Scores-графики строятся по seed 42 (первый), метрики — по всем.
"""
import os, sys, pickle
import numpy as np, pandas as pd
from scipy.stats import wilcoxon
from train_ticd import main as run_one

BASE = "ticd_seed_runs"
UNSUP = ["Isolation Forest", "One-Class SVM", "Autoencoder", "TICD"]
# METRICS = ["auc_roc", "auc_pr", "f1", "fpr", "compromise_recall",
#            "recall_at_1pct_fpr", "soph_naive", "soph_statistical",
#            "soph_adaptive"]
METRICS = ["auc_roc", "auc_pr", "f1", "fpr", "compromise_recall",
           "recall_at_1pct_fpr", "soph_naive", "soph_statistical",
           "soph_adaptive",
           "rec_signaling_dos", "rec_register_hijack",
           "rec_register_bruteforce", "rec_spit", "rec_proxy_compromise"]


COMPARISONS = [
    ("TICD", "Autoencoder", "soph_adaptive"),
    ("TICD", "Isolation Forest", "soph_adaptive"),
    ("TICD", "One-Class SVM", "soph_adaptive"),
    ("TICD", "Autoencoder", "recall_at_1pct_fpr"),
    ("TICD", "Isolation Forest", "compromise_recall"),
]


def main(seeds, csv="sip_dataset_v4.csv"):
    os.makedirs(BASE, exist_ok=True)
    frames = []
    for sd in seeds:
        print(f"\n{'='*60}\n[seed {sd}]\n{'='*60}")
        d = run_one(sd, csv, os.path.join(BASE, f"seed_{sd}"))
        d["seed"] = sd
        frames.append(d)
    alld = pd.concat(frames, ignore_index=True)
    alld.to_csv(f"{BASE}/all_seed_metrics.csv", index=False)

    agg = alld.groupby(["name", "oracle"])[METRICS].agg(["mean", "std"])
    agg.to_csv(f"{BASE}/aggregated.csv")

    sig_rows = []
    for a, b, metric in COMPARISONS:
        va = alld[alld["name"] == a].set_index("seed")[metric]
        vb = alld[alld["name"] == b].set_index("seed")[metric]
        common = va.index.intersection(vb.index)
        va, vb = va.loc[common].dropna(), vb.loc[common].dropna()
        common = va.index.intersection(vb.index)
        va, vb = va.loc[common].to_numpy(), vb.loc[common].to_numpy()
        n = len(va)
        if n < 2 or np.allclose(va - vb, 0):
            continue
        better = int((va > vb).sum())
        try:
            _, p = wilcoxon(va, vb, alternative="two-sided")
        except ValueError:
            p = np.nan
        sig_rows.append(dict(metric=metric, a=a, b=b, n=n,
                             median_a=np.median(va), median_b=np.median(vb),
                             a_better=better, p_value=p))
        print(f"[{metric}] {a} vs {b} (N={n}): "
              f"{np.median(va):.2f} vs {np.median(vb):.2f} | "
              f"{better}/{n} | p={p:.4f}")
    sig = pd.DataFrame(sig_rows)
    sig.to_csv(f"{BASE}/significance.csv", index=False)

    with open(f"{BASE}/aggregated.pkl", "wb") as f:
        pickle.dump(dict(all_metrics=alld, aggregated=agg,
                         significance=sig, seeds=seeds), f)
    print(f"\nГотово: {BASE}/ (aggregated.pkl, значимость, метрики)")


if __name__ == "__main__":
    seeds = [int(s) for s in sys.argv[1:]] or list(range(42, 52))
    main(seeds)
