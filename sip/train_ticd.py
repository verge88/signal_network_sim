"""
train_ticd.py — TICD vs unsupervised/oracle, с сохранением pkl-артефактов.
TICD и unsupervised обучаются ТОЛЬКО на норме. GB/RF помечены [oracle].
Все метрики разбиваются по attacker_sophistication (направление 1).
"""
import numpy as np, pandas as pd, os, sys, pickle, warnings
from sklearn.ensemble import (IsolationForest, RandomForestClassifier,
                              GradientBoostingClassifier)
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             f1_score, confusion_matrix)
import torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from ticd import TICDetector

warnings.filterwarnings("ignore")


def build_feature_cols(df):
    cands = [c for c in df.columns
             if c.startswith(("obs_", "cu_", "var_", "dev_", "z_"))
             and not c.startswith("gt_")]
    for extra in ["zone_io_ratio_rank", "integrity_fail_rate",
                  "autocorr_obs_total_messages", "obs_destinations_cv",
                  "node_type_enc", "mean_txn_state_consistency_error",
                  "mean_cu_processing_delay_ms"]:
        if extra in df.columns and extra not in cands:
            cands.append(extra)
    drop = {"cu_n_ack_observed", "cu_n_invite_200", "cu_dcr_transit"}
    return [c for c in dict.fromkeys(cands) if c not in drop]


class AE(nn.Module):
    def __init__(self, d, h=(64, 32), z=12):
        super().__init__()
        enc, prev = [], d
        for hh in h:
            enc += [nn.Linear(prev, hh), nn.ReLU(), nn.BatchNorm1d(hh)]; prev = hh
        enc += [nn.Linear(prev, z)]
        dec, prev = [], z
        for hh in reversed(h):
            dec += [nn.Linear(prev, hh), nn.ReLU(), nn.BatchNorm1d(hh)]; prev = hh
        dec += [nn.Linear(prev, d)]
        self.e = nn.Sequential(*enc); self.d = nn.Sequential(*dec)
    def forward(self, x): return self.d(self.e(x))


def train_ae(X, epochs=100, bs=256, lr=1e-3):
    m = AE(X.shape[1]); opt = torch.optim.Adam(m.parameters(), lr=lr,
                                               weight_decay=1e-5)
    crit = nn.MSELoss()
    dl = DataLoader(TensorDataset(torch.FloatTensor(X)), batch_size=bs,
                    shuffle=True)
    hist = []
    for _ in range(epochs):
        m.train(); tot = 0.0
        for (b,) in dl:
            loss = crit(m(b), b); opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(b)
        hist.append(tot / len(X))
    return m, hist


def by_attack_type(scores, df, thr):
    """Полнота (recall) обнаружения в разрезе типов атак при пороге thr."""
    yp = (np.nan_to_num(scores) >= thr).astype(int)
    out = {}
    for atk in ["signaling_dos", "register_hijack", "register_bruteforce",
                "spit", "proxy_compromise"]:
        m = (df["attack_type"] == atk).to_numpy()
        if m.sum() > 0:
            out[atk] = 100.0 * yp[m].sum() / m.sum()
    return out


def ae_err(m, X):
    m.eval()
    with torch.no_grad():
        t = torch.FloatTensor(X)
        return torch.mean((t - m(t)) ** 2, dim=1).numpy()


def metrics(scores, y, thr=None, pctile=95):
    scores = np.nan_to_num(np.asarray(scores, float))
    if thr is None:
        thr = np.percentile(scores[y == 0], pctile)
    yp = (scores >= thr).astype(int)
    auc = roc_auc_score(y, scores) if len(np.unique(y)) > 1 else 0.0
    ap = average_precision_score(y, scores) if len(np.unique(y)) > 1 else 0.0
    f1 = f1_score(y, yp, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y, yp, labels=[0, 1]).ravel()
    fpr = fp / (fp + tn) if fp + tn > 0 else 0.0
    return dict(auc_roc=auc, auc_pr=ap, f1=f1, fpr=fpr, thr=thr, yp=yp)


def recall_at_fpr(scores, y, target_fpr=0.01):
    """Recall при фиксированном FPR — честная метрика для статьи."""
    scores = np.nan_to_num(np.asarray(scores, float))
    thr = np.percentile(scores[y == 0], 100 * (1 - target_fpr))
    yp = (scores >= thr).astype(int)
    tp = ((yp == 1) & (y == 1)).sum(); fn = ((yp == 0) & (y == 1)).sum()
    return 100.0 * tp / max(1, tp + fn), thr


def by_soph(scores, df, thr):
    yp = (np.nan_to_num(scores) >= thr).astype(int)
    out = {}
    for s in ["naive", "statistical", "adaptive"]:
        m = (df["attacker_sophistication"] == s).to_numpy()
        if m.sum() > 0:
            out[s] = 100.0 * yp[m].sum() / m.sum()
    return out


def main(seed=42, csv="sip_dataset_v4.csv", out="results_ticd"):
    os.makedirs(out, exist_ok=True)
    df = pd.read_csv(csv)
    if "node_type" in df.columns and "node_type_enc" not in df.columns:
        df["node_type_enc"] = LabelEncoder().fit_transform(df["node_type"])
    feat = build_feature_cols(df)

    tr_val, test = train_test_split(df, test_size=0.25,
                                    stratify=df["is_anomaly"],
                                    random_state=seed)
    train, val = train_test_split(tr_val, test_size=0.2,
                                  stratify=tr_val["is_anomaly"],
                                  random_state=seed)
    imp = SimpleImputer(strategy="median").fit(train[feat])
    sc = StandardScaler().fit(imp.transform(train[feat]))
    Xtr = sc.transform(imp.transform(train[feat]))
    Xte = sc.transform(imp.transform(test[feat]))
    ytr = train["is_anomaly"].values
    yte = test["is_anomaly"].values
    Xtr_norm = Xtr[ytr == 0]
    comp_mask = (test["attack_type"] == "proxy_compromise").to_numpy()

    results, model_scores = [], {}

    # def record(name, scores, oracle=False):
    #     scores = np.asarray(scores, float)
    #     model_scores[name] = scores
    #     m = metrics(scores, yte, thr=(0.5 if oracle else None))
    #     soph = by_soph(scores, test, m["thr"])
    #     r_at_fpr, _ = recall_at_fpr(scores, yte, 0.01)
    #     comp_rec = 100.0 * m["yp"][comp_mask].sum() / max(1, comp_mask.sum())
    #     row = dict(name=name, oracle=oracle, auc_roc=m["auc_roc"],
    #                auc_pr=m["auc_pr"], f1=m["f1"], fpr=m["fpr"],
    #                threshold=m["thr"], compromise_recall=comp_rec,
    #                recall_at_1pct_fpr=r_at_fpr,
    #                soph_naive=soph.get("naive", np.nan),
    #                soph_statistical=soph.get("statistical", np.nan),
    #                soph_adaptive=soph.get("adaptive", np.nan))
    #     results.append(row)
    #     tag = "[oracle]" if oracle else "[unsup] "
    #     print(f"{tag} {name:20s} AUC-PR={m['auc_pr']:.3f} F1={m['f1']:.3f} "
    #           f"comp={comp_rec:.1f}% R@1%FPR={r_at_fpr:.1f}% | "
    #           f"n={soph.get('naive',0):.0f} s={soph.get('statistical',0):.0f} "
    #           f"a={soph.get('adaptive',0):.0f}")

    def record(name, scores, oracle=False):
        scores = np.asarray(scores, float)
        model_scores[name] = scores
        m = metrics(scores, yte, thr=(0.5 if oracle else None))
        soph = by_soph(scores, test, m["thr"])
        atk = by_attack_type(scores, test, m["thr"])          # NEW
        r_at_fpr, _ = recall_at_fpr(scores, yte, 0.01)
        comp_rec = 100.0 * m["yp"][comp_mask].sum() / max(1, comp_mask.sum())
        row = dict(name=name, oracle=oracle, auc_roc=m["auc_roc"],
                   auc_pr=m["auc_pr"], f1=m["f1"], fpr=m["fpr"],
                   threshold=m["thr"], compromise_recall=comp_rec,
                   recall_at_1pct_fpr=r_at_fpr,
                   soph_naive=soph.get("naive", np.nan),
                   soph_statistical=soph.get("statistical", np.nan),
                   soph_adaptive=soph.get("adaptive", np.nan),
                   rec_signaling_dos=atk.get("signaling_dos", np.nan),   # NEW
                   rec_register_hijack=atk.get("register_hijack", np.nan),
                   rec_register_bruteforce=atk.get("register_bruteforce", np.nan),
                   rec_spit=atk.get("spit", np.nan),
                   rec_proxy_compromise=atk.get("proxy_compromise", np.nan))
        results.append(row)
        tag = "[oracle]" if oracle else "[unsup] "
        print(f"{tag} {name:20s} AUC-PR={m['auc_pr']:.3f} "
              f"dos={atk.get('signaling_dos',0):.0f} "
              f"hij={atk.get('register_hijack',0):.0f} "
              f"bf={atk.get('register_bruteforce',0):.0f} "
              f"spit={atk.get('spit',0):.0f} "
              f"comp={atk.get('proxy_compromise',0):.0f}")


    print("\n=== UNSUPERVISED (обучены только на норме) ===")
    iso = IsolationForest(n_estimators=300, contamination=0.05,
                          random_state=seed).fit(Xtr_norm)
    record("Isolation Forest", -iso.score_samples(Xte))
    svm = OneClassSVM(nu=0.05, kernel="rbf", gamma="scale").fit(Xtr_norm)
    record("One-Class SVM", -svm.score_samples(Xte))
    ae, ae_hist = train_ae(Xtr_norm)
    record("Autoencoder", ae_err(ae, Xte))
    ticd = TICDetector(seed=seed).fit(train, val)
    record("TICD", ticd.score(test))

    print("\n=== ORACLE (обучены на метках — верхняя граница) ===")
    rf = RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                random_state=seed, n_jobs=-1).fit(Xtr, ytr)
    record("Random Forest", rf.predict_proba(Xte)[:, 1], oracle=True)
    gb = GradientBoostingClassifier(n_estimators=300, learning_rate=0.05,
                                    max_depth=5, random_state=seed).fit(Xtr, ytr)
    record("Gradient Boosting", gb.predict_proba(Xte)[:, 1], oracle=True)

    # Компонентная абляция TICD
    comps = ticd.score_components(test)
    ticd_ablation = []
    print("\n=== TICD component ablation ===")
    for cname, cs in comps.items():
        thr = np.percentile(cs[yte == 0], 95)
        soph = by_soph(cs, test, thr)
        ap = average_precision_score(yte, np.nan_to_num(cs))
        yp = (np.nan_to_num(cs) >= thr).astype(int)
        f1 = f1_score(yte, yp, zero_division=0)
        comp_rec = 100.0 * yp[comp_mask].sum() / max(1, comp_mask.sum())
        ticd_ablation.append(dict(config=cname, f1=f1, auc_pr=ap,
                                  compromise_recall=comp_rec,
                                  soph_adaptive=soph.get("adaptive", np.nan)))
        print(f"  {cname:12s}: adapt={soph.get('adaptive',0):.1f}% "
              f"AUC-PR={ap:.3f} F1={f1:.3f}")

    res_df = pd.DataFrame(results)
    res_df.to_csv(f"{out}/comparison_seed{seed}.csv", index=False)

    # ── СОХРАНЕНИЕ PKL ДЛЯ ГРАФИКОВ ──
    invariant = dict(
        M=test["cu_ack_closure_mismatch"].to_numpy(),
        delta_I1=(1.0 - (test["obs_dialog_completion_ratio"] +
                         test["obs_dangling_dialog_ratio"])).abs().to_numpy(),
        ddr0=df[df["is_anomaly"] == 0]["obs_dangling_dialog_ratio"].mean(),
        gt_ddr=test.get("gt_dangling_dialog_ratio",
                        pd.Series(np.nan, index=test.index)).to_numpy(),
    )
    artifacts = dict(
        results=results, model_scores=model_scores,
        y_test=yte, attack_types=test["attack_type"].to_numpy(),
        sophistication=test["attacker_sophistication"].to_numpy(),
        ticd_ablation=ticd_ablation, invariant=invariant,
        ae_train_hist=ae_hist, feature_cols=feat, seed=seed,
        rf_importances=rf.feature_importances_,
    )
    with open(f"{out}/ticd_artifacts_seed{seed}.pkl", "wb") as f:
        pickle.dump(artifacts, f)
    print(f"\nСохранено: {out}/ticd_artifacts_seed{seed}.pkl")
    return res_df


if __name__ == "__main__":
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 42
    csv = sys.argv[2] if len(sys.argv) > 2 else "sip_dataset_v4.csv"
    out = sys.argv[3] if len(sys.argv) > 3 else "results_ticd"
    main(seed, csv, out)
