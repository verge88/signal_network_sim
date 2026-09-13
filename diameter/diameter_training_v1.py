"""
Diameter Master-Slave Anomaly Detection Training Pipeline v1
=============================================================
Adapted from SS7 pipeline for Diameter protocol specifics.

Models:
  1. Isolation Forest (unsupervised)
  2. One-Class SVM (unsupervised)
  3. Autoencoder (unsupervised)
  4. Variational Autoencoder (unsupervised)
  5. LSTM-Autoencoder (unsupervised, sequential)
  6. Temporal-Attention-AE (unsupervised, sequential)
  7. Random Forest (supervised)
  8. Gradient Boosting (supervised)
  9. Cascade Compromise Detector (supervised, 2-level)
  10. Specialized Ensemble (unsupervised, per-attack-type)
  11. Meta-Ensemble (stacking)

Plus ablation studies for MS-features and temporal features.
"""

import numpy as np
import pandas as pd
import os
import json
import warnings
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional

from sklearn.ensemble import (
    IsolationForest, RandomForestClassifier, GradientBoostingClassifier
)
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    confusion_matrix, precision_recall_curve, roc_curve
)
from sklearn.impute import SimpleImputer

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch

warnings.filterwarnings("ignore")

plt.rcParams.update({
    'figure.dpi': 150, 'savefig.dpi': 200, 'font.size': 10,
    'axes.titlesize': 12, 'axes.labelsize': 10,
    'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'legend.fontsize': 8, 'figure.facecolor': 'white',
    'axes.facecolor': '#f2f2f2', 'axes.grid': True,
    'grid.alpha': 0.35, 'grid.linestyle': '--',
})

# Diameter-specific command lists (must match simulator)
S6A_COMMANDS = ["AIR", "ULR", "CLR", "IDR", "DSR", "PUR", "RSR", "NOR"]
GX_COMMANDS = ["CCR_I", "CCR_U", "CCR_T", "RAR"]


# ══════════════════════════════════════════════════════════
#  Configuration
# ══════════════════════════════════════════════════════════

@dataclass
class TrainingConfig:
    data_path: str = "diameter_dataset_v1.csv"
    results_dir: str = "results_diameter_v1"
    plots_dir: str = "results_diameter_v1/plots"
    test_size: float = 0.25
    val_size: float = 0.15
    seed: int = 42

    # Unsupervised models
    iso_forest_trees: int = 300
    iso_forest_contamination: float = 0.05
    ocsvm_nu: float = 0.05
    ocsvm_kernel: str = "rbf"

    # Autoencoder
    ae_hidden_dims: List[int] = field(default_factory=lambda: [64, 32])
    ae_latent_dim: int = 12
    ae_epochs: int = 100
    ae_batch: int = 256
    ae_lr: float = 1e-3

    # VAE
    vae_hidden_dims: List[int] = field(default_factory=lambda: [64, 32])
    vae_latent_dim: int = 12
    vae_epochs: int = 120
    vae_batch: int = 256
    vae_lr: float = 1e-3
    vae_kl_weight: float = 0.5

    # LSTM-AE
    lstm_hidden: int = 48
    lstm_layers: int = 2
    lstm_seq_len: int = 10
    lstm_epochs: int = 80
    lstm_batch: int = 256
    lstm_lr: float = 1e-3

    # Supervised
    rf_trees: int = 300
    gb_estimators: int = 300
    gb_lr: float = 0.05

    # Cascade
    cascade_rf_trees: int = 500
    cascade_gb_estimators: int = 500

    threshold_percentile: float = 95.0


# ══════════════════════════════════════════════════════════
#  Data Pipeline — Diameter features
# ══════════════════════════════════════════════════════════

class DataPipeline:
    """
    Feature engineering for Diameter signaling data.

    Feature groups (analogous to SS7 but with Diameter-specific fields):
    1. CU interaction (11 features) — same concept as SS7
    2. Reported by slave (19+ features) — Diameter interfaces & commands
    3. Temporal statistics (7 features) — same concept
    4. Spatial/zone features (13 features) — same concept
    """

    # ── Build feature column lists dynamically ──

    # Group 1: CU interaction features (11)
    CU_FEATURES = [
        "cu_delivered", "cu_rtt_delay_ms", "cu_req_delay_ms",
        "cu_resp_delay_ms", "cu_combined_loss_prob", "integrity_check",
        "consecutive_cu_losses", "cu_processing_delay_ms",
        "cu_response_time_jitter", "cu_staleness", "cu_consistency_error",
    ]

    # Group 2: Reported traffic features
    # Interface counts
    REPORTED_COUNTS = [
        "obs_total_messages", "obs_s6a_count", "obs_gx_count",
        "obs_rx_count", "obs_s13_count", "obs_other_count",
    ]
    REPORTED_STATS = [
        "obs_entropy", "obs_inbound_outbound_ratio",
        "obs_international_fraction", "obs_n_unique_peers",
        "obs_s6a_dominance_ratio", "obs_s6a_top2_ratio", "obs_s6a_gini",
    ]
    # S6a command ratios
    S6A_RATIO_FEATURES = [f"obs_ratio_s6a_{c.lower()}" for c in S6A_COMMANDS]
    # Gx command ratios
    GX_RATIO_FEATURES = [f"obs_ratio_gx_{c.lower()}" for c in GX_COMMANDS]

    # Group 3: Z-score features
    Z_FEATURES = [
        "z_obs_total_messages", "z_obs_s6a_count",
        "z_obs_s6a_dominance_ratio",
        "z_obs_ratio_s6a_ulr", "z_obs_ratio_s6a_idr",
    ]

    # Group 4: Temporal features (7)
    TEMPORAL_FEATURES = [
        "var_obs_total_messages", "var_obs_entropy",
        "var_obs_inbound_outbound_ratio",
        "var_obs_s6a_dominance_ratio",
        "autocorr_obs_total_messages",
        "integrity_fail_rate", "obs_peers_cv",
    ]

    # Group 5: Zone deviation features
    ZONE_FEATURES = [
        "dev_total_messages_vs_zone", "dev_s6a_count_vs_zone",
        "dev_gx_count_vs_zone",
        "zone_io_ratio_rank", "dev_io_ratio_vs_zone",
        "dev_integrity_fail_rate_vs_zone", "dev_var_total_vs_zone",
    ]

    GROUND_TRUTH_COLS = [
        "gt_total_messages", "gt_s6a_count", "gt_gx_count",
        "gt_rx_count", "gt_s13_count", "gt_other_count",
        "gt_entropy", "gt_inbound_outbound_ratio",
        "gt_international_fraction", "gt_n_unique_peers",
    ]

    LABEL_COL = "is_anomaly"
    ATTACK_COL = "attack_type"

    # Compromise-specific features (subset for cascade stage 2)
    COMPROMISE_FEATURES = [
        "obs_inbound_outbound_ratio", "integrity_check",
        "integrity_fail_rate", "cu_consistency_error",
        "cu_processing_delay_ms", "cu_response_time_jitter",
        "var_obs_total_messages", "var_obs_entropy",
        "var_obs_inbound_outbound_ratio", "var_obs_s6a_dominance_ratio",
        "autocorr_obs_total_messages", "obs_peers_cv",
        "zone_io_ratio_rank", "dev_io_ratio_vs_zone",
        "dev_integrity_fail_rate_vs_zone", "dev_var_total_vs_zone",
        "consecutive_cu_losses", "cu_staleness",
        "dev_total_messages_vs_zone", "dev_s6a_count_vs_zone",
    ]

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy="median")
        self.le_node_type = LabelEncoder()
        self.feature_cols: List[str] = []
        self.compromise_scaler = StandardScaler()
        self.compromise_imputer = SimpleImputer(strategy="median")
        self.compromise_feature_cols: List[str] = []

    def _build_feature_cols(self, df: pd.DataFrame):
        """Build feature column lists from what's actually in the data."""
        all_candidate = (
            self.CU_FEATURES + self.REPORTED_COUNTS + self.REPORTED_STATS +
            self.S6A_RATIO_FEATURES + self.GX_RATIO_FEATURES +
            self.Z_FEATURES + self.ZONE_FEATURES + self.TEMPORAL_FEATURES
        )
        self.feature_cols = [c for c in all_candidate if c in df.columns]
        if "node_type_enc" in df.columns:
            self.feature_cols.append("node_type_enc")

        self.compromise_feature_cols = [
            c for c in self.COMPROMISE_FEATURES if c in df.columns
        ]
        if "node_type_enc" in df.columns:
            self.compromise_feature_cols.append("node_type_enc")

    def load_and_prepare(self) -> Tuple[pd.DataFrame, pd.DataFrame,
                                        pd.DataFrame]:
        df = pd.read_csv(self.config.data_path)
        print(f"Loaded {len(df)} records, anomalies: "
              f"{df[self.LABEL_COL].sum()} "
              f"({100 * df[self.LABEL_COL].mean():.2f}%)")
        print(f"Attack distribution:\n{df[self.ATTACK_COL].value_counts()}\n")

        if "node_type" in df.columns and "node_type_enc" not in df.columns:
            df["node_type_enc"] = self.le_node_type.fit_transform(
                df["node_type"])

        self._build_feature_cols(df)

        # Verify no GT leakage
        for c in self.feature_cols:
            if c.startswith("gt_"):
                raise ValueError(f"GT column '{c}' in feature set!")

        print(f"Observable features: {len(self.feature_cols)}")
        print(f"Compromise-specific features: "
              f"{len(self.compromise_feature_cols)}")

        train_val, test = train_test_split(
            df, test_size=self.config.test_size,
            stratify=df[self.LABEL_COL], random_state=self.config.seed)
        train, val = train_test_split(
            train_val,
            test_size=self.config.val_size / (1 - self.config.test_size),
            stratify=train_val[self.LABEL_COL],
            random_state=self.config.seed)

        print(f"Split — Train: {len(train)}, Val: {len(val)}, "
              f"Test: {len(test)}")
        print(f"Train anomaly rate: {train[self.LABEL_COL].mean():.4f}")
        print(f"Val anomaly rate:   {val[self.LABEL_COL].mean():.4f}")
        print(f"Test anomaly rate:  {test[self.LABEL_COL].mean():.4f}\n")
        return train, val, test

    def get_features_labels(self, df, fit=False):
        X = df[self.feature_cols].copy()
        if fit:
            X_imp = self.imputer.fit_transform(X)
            X_sc = self.scaler.fit_transform(X_imp)
        else:
            X_imp = self.imputer.transform(X)
            X_sc = self.scaler.transform(X_imp)
        return X_sc, df[self.LABEL_COL].values

    def get_compromise_features(self, df, fit=False):
        X = df[self.compromise_feature_cols].copy()
        if fit:
            X_imp = self.compromise_imputer.fit_transform(X)
            X_sc = self.compromise_scaler.fit_transform(X_imp)
        else:
            X_imp = self.compromise_imputer.transform(X)
            X_sc = self.compromise_scaler.transform(X_imp)
        y = (df[self.ATTACK_COL] == "node_compromise").astype(int).values
        return X_sc, y

    def get_normal_data(self, df, fit=False):
        normal = df[df[self.LABEL_COL] == 0]
        return self.get_features_labels(normal, fit=fit)

    def get_sequential_data(self, df, seq_len):
        X_all, y_all = self.get_features_labels(df, fit=False)
        seqs, labs = [], []
        for nid in df["node_id"].unique():
            mask = df["node_id"].values == nid
            Xn, yn = X_all[mask], y_all[mask]
            for i in range(len(Xn) - seq_len + 1):
                seqs.append(Xn[i:i + seq_len])
                labs.append(yn[i + seq_len - 1])
        return np.array(seqs), np.array(labs)


# ══════════════════════════════════════════════════════════
#  MS (Master-Slave) Feature List for Ablation
# ══════════════════════════════════════════════════════════

MS_FEATURES = (
    DataPipeline.CU_FEATURES +
    DataPipeline.REPORTED_COUNTS +
    ["obs_inbound_outbound_ratio"] +
    [f for f in DataPipeline.ZONE_FEATURES] +
    DataPipeline.TEMPORAL_FEATURES
)


# ══════════════════════════════════════════════════════════
#  Feature Groups for Specialized Ensemble
# ══════════════════════════════════════════════════════════

FEATURE_GROUPS = {
    "s6a_profile": (
        DataPipeline.S6A_RATIO_FEATURES +
        ["obs_s6a_dominance_ratio", "obs_s6a_top2_ratio", "obs_s6a_gini",
         "z_obs_s6a_dominance_ratio", "z_obs_ratio_s6a_ulr",
         "z_obs_ratio_s6a_idr"]
    ),
    "gx_profile": DataPipeline.GX_RATIO_FEATURES,
    "volume_timing": [
        "obs_total_messages", "obs_s6a_count", "obs_gx_count",
        "obs_rx_count", "obs_s13_count", "obs_entropy",
        "z_obs_total_messages", "z_obs_s6a_count",
    ],
    "control_unit": DataPipeline.CU_FEATURES,
    "international": [
        "obs_international_fraction", "obs_n_unique_peers",
    ],
    "baseline_deviation": [
        "dev_total_messages_vs_zone", "dev_s6a_count_vs_zone",
        "dev_gx_count_vs_zone", "obs_inbound_outbound_ratio",
        "zone_io_ratio_rank", "dev_io_ratio_vs_zone",
    ],
    "compromise_indicators": DataPipeline.COMPROMISE_FEATURES,
}

ATTACK_LABELS = {
    'location_tracking': 'Отслеживание',
    'sms_data_interception': 'Перехват SMS/данных',
    'signaling_dos': 'Сигнальный DoS',
    'fraud_profile': 'Мошенничество (IDR)',
    'node_compromise': 'Компрометация узла',
}


# ══════════════════════════════════════════════════════════
#  Neural Network Models
# ══════════════════════════════════════════════════════════

class Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dims, latent_dim):
        super().__init__()
        enc = []
        prev = input_dim
        for h in hidden_dims:
            enc.extend([nn.Linear(prev, h), nn.ReLU(), nn.BatchNorm1d(h)])
            prev = h
        enc.append(nn.Linear(prev, latent_dim))
        self.encoder = nn.Sequential(*enc)
        dec = []
        prev = latent_dim
        for h in reversed(hidden_dims):
            dec.extend([nn.Linear(prev, h), nn.ReLU(), nn.BatchNorm1d(h)])
            prev = h
        dec.append(nn.Linear(prev, input_dim))
        self.decoder = nn.Sequential(*dec)

    def forward(self, x):
        return self.decoder(self.encoder(x))


class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dims, latent_dim):
        super().__init__()
        enc = []
        prev = input_dim
        for h in hidden_dims:
            enc.extend([nn.Linear(prev, h), nn.ReLU(), nn.BatchNorm1d(h)])
            prev = h
        self.encoder = nn.Sequential(*enc)
        self.fc_mu = nn.Linear(prev, latent_dim)
        self.fc_logvar = nn.Linear(prev, latent_dim)
        dec = []
        prev = latent_dim
        for h in reversed(hidden_dims):
            dec.extend([nn.Linear(prev, h), nn.ReLU(), nn.BatchNorm1d(h)])
            prev = h
        dec.append(nn.Linear(prev, input_dim))
        self.decoder = nn.Sequential(*dec)

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)

    def forward(self, x):
        mu, lv = self.encode(x)
        z = self.reparameterize(mu, lv)
        return self.decoder(z), mu, lv


class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers, seq_len,
                 dropout=0.2):
        super().__init__()
        self.seq_len = seq_len
        self.encoder = nn.LSTM(input_dim, hidden_dim, n_layers,
                               batch_first=True,
                               dropout=dropout if n_layers > 1 else 0)
        self.decoder = nn.LSTM(hidden_dim, hidden_dim, n_layers,
                               batch_first=True,
                               dropout=dropout if n_layers > 1 else 0)
        self.output_layer = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        _, (h, c) = self.encoder(x)
        dec_in = h[-1].unsqueeze(1).repeat(1, self.seq_len, 1)
        dec_out, _ = self.decoder(dec_in, (h, c))
        return self.output_layer(dec_out)


class TemporalAttentionAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_heads=4, n_layers=2,
                 seq_len=10, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_heads,
            dim_feedforward=hidden_dim * 2, dropout=dropout,
            batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, n_layers)
        self.output_proj = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        return self.output_proj(self.transformer(self.input_proj(x)))


# ══════════════════════════════════════════════════════════
#  Trainers
# ══════════════════════════════════════════════════════════

class AETrainer:
    def __init__(self, model, lr, epochs, batch_size, patience=15,
                 device=None):
        self.model = model
        self.lr, self.epochs, self.batch_size = lr, epochs, batch_size
        self.patience = patience
        self.device = device or ("cuda" if torch.cuda.is_available()
                                 else "cpu")
        self.model.to(self.device)
        self.train_losses, self.val_losses = [], []

    def fit(self, X_train, X_val=None):
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr,
                               weight_decay=1e-5)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, self.epochs)
        crit = nn.MSELoss()
        dl = DataLoader(TensorDataset(torch.FloatTensor(X_train)),
                        batch_size=self.batch_size, shuffle=True)
        best_val, pat_cnt, best_state = float("inf"), 0, None
        self.train_losses, self.val_losses = [], []

        for epoch in range(self.epochs):
            self.model.train()
            tl = 0.0
            for (batch,) in dl:
                batch = batch.to(self.device)
                loss = crit(self.model(batch), batch)
                opt.zero_grad(); loss.backward(); opt.step()
                tl += loss.item() * len(batch)
            tl /= len(X_train)
            sched.step()
            self.train_losses.append(tl)

            if X_val is not None:
                vl = self._eval(X_val, crit)
                self.val_losses.append(vl)
                if vl < best_val:
                    best_val, pat_cnt = vl, 0
                    best_state = {k: v.cpu().clone()
                                  for k, v in self.model.state_dict().items()}
                else:
                    pat_cnt += 1
                if (epoch + 1) % 20 == 0:
                    print(f"  Epoch {epoch+1}/{self.epochs}: "
                          f"train={tl:.6f}, val={vl:.6f}")
                if pat_cnt >= self.patience:
                    print(f"  Early stopping at epoch {epoch+1}")
                    break

        if best_state:
            self.model.load_state_dict(best_state)
            self.model.to(self.device)

    def _eval(self, X, crit):
        self.model.eval()
        with torch.no_grad():
            t = torch.FloatTensor(X).to(self.device)
            return crit(self.model(t), t).item()

    def reconstruction_error(self, X):
        self.model.eval()
        with torch.no_grad():
            t = torch.FloatTensor(X).to(self.device)
            err = torch.mean((t - self.model(t)) ** 2, dim=-1)
            if err.dim() > 1:
                err = err.mean(dim=-1)
        return err.cpu().numpy()


class VAETrainer:
    def __init__(self, model, lr, epochs, batch_size, kl_weight=0.5,
                 patience=15, device=None):
        self.model = model
        self.lr, self.epochs, self.batch_size = lr, epochs, batch_size
        self.kl_weight, self.patience = kl_weight, patience
        self.device = device or ("cuda" if torch.cuda.is_available()
                                 else "cpu")
        self.model.to(self.device)
        self.train_losses, self.val_losses = [], []

    def _loss(self, x, x_hat, mu, logvar):
        recon = F.mse_loss(x_hat, x)
        kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return recon + self.kl_weight * kl, recon, kl

    def fit(self, X_train, X_val=None):
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr,
                               weight_decay=1e-5)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, self.epochs)
        dl = DataLoader(TensorDataset(torch.FloatTensor(X_train)),
                        batch_size=self.batch_size, shuffle=True)
        best_val, pat_cnt, best_state = float("inf"), 0, None
        self.train_losses, self.val_losses = [], []

        for epoch in range(self.epochs):
            self.model.train()
            tl = 0.0
            for (batch,) in dl:
                batch = batch.to(self.device)
                x_hat, mu, lv = self.model(batch)
                loss, _, _ = self._loss(batch, x_hat, mu, lv)
                opt.zero_grad(); loss.backward(); opt.step()
                tl += loss.item() * len(batch)
            tl /= len(X_train)
            sched.step()
            self.train_losses.append(tl)

            if X_val is not None:
                vl = self._eval_val(X_val)
                self.val_losses.append(vl)
                if vl < best_val:
                    best_val, pat_cnt = vl, 0
                    best_state = {k: v.cpu().clone()
                                  for k, v in self.model.state_dict().items()}
                else:
                    pat_cnt += 1
                if (epoch + 1) % 20 == 0:
                    print(f"  Epoch {epoch+1}/{self.epochs}: "
                          f"train={tl:.6f}, val={vl:.6f}")
                if pat_cnt >= self.patience:
                    print(f"  Early stopping at epoch {epoch+1}")
                    break

        if best_state:
            self.model.load_state_dict(best_state)
            self.model.to(self.device)

    def _eval_val(self, X):
        self.model.eval()
        with torch.no_grad():
            t = torch.FloatTensor(X).to(self.device)
            x_hat, mu, lv = self.model(t)
            loss, _, _ = self._loss(t, x_hat, mu, lv)
        return loss.item()

    def anomaly_score(self, X):
        self.model.eval()
        with torch.no_grad():
            t = torch.FloatTensor(X).to(self.device)
            x_hat, mu, lv = self.model(t)
            recon = torch.mean((t - x_hat) ** 2, dim=-1)
            kl = -0.5 * torch.sum(1 + lv - mu.pow(2) - lv.exp(), dim=-1)
        return (recon + self.kl_weight * kl).cpu().numpy()


# ══════════════════════════════════════════════════════════
#  Evaluation Utilities
# ══════════════════════════════════════════════════════════

def find_threshold(scores, y_true, percentile=95.0):
    normal_scores = scores[y_true == 0]
    th_pct = np.percentile(normal_scores, percentile)
    prec, rec, ths = precision_recall_curve(y_true, scores)
    f1s = 2 * prec * rec / (prec + rec + 1e-9)
    th_f1 = ths[min(np.argmax(f1s), len(ths) - 1)]
    y_pct = (scores >= th_pct).astype(int)
    y_f1 = (scores >= th_f1).astype(int)
    if f1_score(y_true, y_f1, zero_division=0) >= \
       f1_score(y_true, y_pct, zero_division=0):
        return th_f1, "max_f1"
    return th_pct, "percentile"


def evaluate_model(name, scores, y_true, attack_types,
                   threshold=None, percentile=95.0):
    if threshold is None:
        threshold, method = find_threshold(scores, y_true, percentile)
    else:
        method = "provided"
    y_pred = (scores >= threshold).astype(int)
    auc_roc = (roc_auc_score(y_true, scores)
               if len(np.unique(y_true)) > 1 else 0.)
    auc_pr = (average_precision_score(y_true, scores)
              if len(np.unique(y_true)) > 1 else 0.)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.

    print(f"\n{'='*60}\n  {name}\n{'='*60}")
    print(f"  AUC-ROC: {auc_roc:.4f}  |  AUC-PR: {auc_pr:.4f}  |  "
          f"F1: {f1:.4f}")
    print(f"  FPR: {fpr:.4f}  |  TP: {tp}  FP: {fp}  FN: {fn}  TN: {tn}")
    print(f"  Threshold: {threshold:.4f} ({method})")

    per_attack = {}
    print(f"  Detection by attack type:")
    for at in sorted(set(attack_types)):
        if at == "none":
            continue
        m = attack_types == at
        if m.sum() > 0:
            det = y_pred[m].sum()
            tot = m.sum()
            per_attack[at] = (det, tot, 100 * det / tot)
            label = ATTACK_LABELS.get(at, at)
            print(f"    {label:30s}: {det}/{tot} = "
                  f"{100 * det / tot:.1f}%")

    return {
        "name": name, "auc_roc": auc_roc, "auc_pr": auc_pr, "f1": f1,
        "fpr": fpr, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "threshold": threshold, "method": method,
        "per_attack": per_attack,
        "scores": scores, "y_pred": y_pred,
    }


# ══════════════════════════════════════════════════════════
#  Specialized Ensemble
# ══════════════════════════════════════════════════════════

class CompactAE(nn.Module):
    def __init__(self, input_dim, bottleneck):
        super().__init__()
        mid = max(bottleneck + 1, (input_dim + bottleneck) // 2)
        self.encoder = nn.Sequential(nn.Linear(input_dim, mid), nn.ReLU(),
                                     nn.Linear(mid, bottleneck))
        self.decoder = nn.Sequential(nn.Linear(bottleneck, mid), nn.ReLU(),
                                     nn.Linear(mid, input_dim))

    def forward(self, x):
        return self.decoder(self.encoder(x))


class SpecializedDetector:
    def __init__(self, name, feature_indices, seed=42):
        self.name, self.feature_indices, self.seed = name, feature_indices, seed
        self.models = {}

    def fit(self, X_normal):
        X = X_normal[:, self.feature_indices]
        n = X.shape[1]
        self.models["iso"] = IsolationForest(
            n_estimators=200, contamination=0.05,
            random_state=self.seed).fit(X)
        self.models["svm"] = OneClassSVM(
            nu=0.05, kernel="rbf", gamma="scale").fit(X)
        ae = CompactAE(n, max(2, n // 3))
        trainer = AETrainer(ae, 1e-3, 120, 256, patience=20)
        trainer.fit(X)
        self.models["ae"], self.models["ae_trainer"] = ae, trainer

    def score(self, X):
        Xs = X[:, self.feature_indices]

        def norm(s):
            p1, p99 = np.percentile(s, 1), np.percentile(s, 99)
            return np.clip((s - p1) / (p99 - p1 + 1e-9), 0, 1)

        return np.max(np.column_stack([
            norm(-self.models["iso"].score_samples(Xs)),
            norm(-self.models["svm"].score_samples(Xs)),
            norm(self.models["ae_trainer"].reconstruction_error(Xs)),
        ]), axis=1)


class EnsembleDetector:
    def __init__(self, groups, all_cols, seed=42):
        self.detectors = {}
        for gn, cols in groups.items():
            idx = [all_cols.index(c) for c in cols if c in all_cols]
            if idx:
                self.detectors[gn] = SpecializedDetector(gn, idx, seed)

    def fit(self, X_normal):
        for n, d in self.detectors.items():
            print(f"  Fitting: {n} ({len(d.feature_indices)} features)")
            d.fit(X_normal)

    def score(self, X):
        r = {n: d.score(X) for n, d in self.detectors.items()}
        a = np.column_stack(list(r.values()))
        r["ensemble_max"] = np.max(a, axis=1)
        r["ensemble_mean"] = np.mean(a, axis=1)
        return r


# ══════════════════════════════════════════════════════════
#  Cascade Compromise Detector
# ══════════════════════════════════════════════════════════

class CascadeCompromiseDetector:
    def __init__(self, config, seed=42):
        self.stage1_rf = RandomForestClassifier(
            n_estimators=config.cascade_rf_trees, random_state=seed,
            class_weight="balanced", n_jobs=-1)
        self.stage2_gb = GradientBoostingClassifier(
            n_estimators=config.cascade_gb_estimators, learning_rate=0.03,
            max_depth=4, min_samples_leaf=20, random_state=seed)
        self.stage2_rf = RandomForestClassifier(
            n_estimators=config.cascade_rf_trees, random_state=seed,
            class_weight="balanced_subsample", max_depth=8,
            min_samples_leaf=10, n_jobs=-1)

    def fit(self, X_full, y, X_comp, y_comp):
        print("  Cascade Stage 1: General RF...")
        self.stage1_rf.fit(X_full, y)
        print("  Cascade Stage 2: Compromise-specific GB + RF...")
        self.stage2_gb.fit(X_comp, y_comp)
        self.stage2_rf.fit(X_comp, y_comp)

    def predict_proba(self, X_full, X_comp):
        p1 = self.stage1_rf.predict_proba(X_full)[:, 1]
        p2 = 0.5 * (self.stage2_gb.predict_proba(X_comp)[:, 1] +
                     self.stage2_rf.predict_proba(X_comp)[:, 1])
        return np.maximum(p1, p2)

    def predict_proba_stage1(self, X):
        return self.stage1_rf.predict_proba(X)[:, 1]

    def predict_proba_stage2(self, X):
        return 0.5 * (self.stage2_gb.predict_proba(X)[:, 1] +
                       self.stage2_rf.predict_proba(X)[:, 1])


# ══════════════════════════════════════════════════════════
#  Visualization
# ══════════════════════════════════════════════════════════

class ResultsVisualizer:
    def __init__(self, plots_dir):
        self.plots_dir = plots_dir
        os.makedirs(plots_dir, exist_ok=True)

    def _save(self, fig, name):
        for fmt in ["png", "svg"]:
            path = os.path.join(self.plots_dir, f"{name}.{fmt}")
            fig.savefig(path, bbox_inches='tight', facecolor='white',
                        format=fmt)
        plt.close(fig)
        print(f"  Saved: {name}.png / .svg")

    def plot_roc_curves(self, results_list, y_test, main_models):
        fig, ax = plt.subplots(figsize=(8, 6))
        for r in results_list:
            if r['name'] not in main_models:
                continue
            fpr_arr, tpr_arr, _ = roc_curve(y_test, r['scores'])
            ax.plot(fpr_arr, tpr_arr, linewidth=1.8,
                    label=f"{r['name']} (AUC={r['auc_roc']:.3f})")
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
        ax.set_xlabel('FPR'); ax.set_ylabel('TPR')
        ax.set_title('ROC-кривые — Diameter'); ax.legend(loc='lower right')
        self._save(fig, '01_roc_curves')

    def plot_pr_curves(self, results_list, y_test, main_models):
        fig, ax = plt.subplots(figsize=(8, 6))
        for r in results_list:
            if r['name'] not in main_models:
                continue
            prec, rec, _ = precision_recall_curve(y_test, r['scores'])
            ax.plot(rec, prec, linewidth=1.8,
                    label=f"{r['name']} (AP={r['auc_pr']:.3f})")
        ax.axhline(y=y_test.mean(), color='k', ls='--', alpha=0.3,
                   label=f'Baseline ({y_test.mean():.3f})')
        ax.set_xlabel('Recall'); ax.set_ylabel('Precision')
        ax.set_title('Precision-Recall — Diameter')
        ax.legend(loc='lower left')
        self._save(fig, '02_precision_recall')

    def plot_attack_heatmap(self, results_list, main_models):
        attacks = ['location_tracking', 'sms_data_interception',
                   'signaling_dos', 'fraud_profile', 'node_compromise']
        matrix, names = [], []
        for r in results_list:
            if r['name'] not in main_models:
                continue
            row = [r['per_attack'].get(a, (0, 1, 0))[2] for a in attacks]
            matrix.append(row)
            names.append(r['name'])
        if not matrix:
            return
        mat = np.array(matrix)
        fig, ax = plt.subplots(figsize=(10, 6))
        im = ax.imshow(mat, cmap='Greys', aspect='auto', vmin=0, vmax=100)
        ax.set_xticks(range(len(attacks)))
        ax.set_xticklabels([ATTACK_LABELS.get(a, a) for a in attacks],
                           rotation=30, ha='right')
        ax.set_yticks(range(len(names))); ax.set_yticklabels(names)
        for i in range(len(names)):
            for j in range(len(attacks)):
                v = mat[i, j]
                c = 'white' if v > 60 else 'black'
                ax.text(j, i, f'{v:.1f}%', ha='center', va='center',
                        fontsize=9, fontweight='bold', color=c)
        plt.colorbar(im, ax=ax, label='Detection %', shrink=0.8)
        ax.set_title('Полнота обнаружения по типам атак — Diameter')
        self._save(fig, '03_attack_heatmap')

    def plot_summary_bars(self, results_list, main_models):
        names, rocs, prs, f1s = [], [], [], []
        for r in results_list:
            if r['name'] in main_models:
                names.append(r['name'])
                rocs.append(r['auc_roc'])
                prs.append(r['auc_pr'])
                f1s.append(r['f1'])
        if not names:
            return
        x = np.arange(len(names)); w = 0.25
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.bar(x - w, rocs, w, label='AUC-ROC', color='#2f2f2f', alpha=0.9)
        ax.bar(x, prs, w, label='AUC-PR', color='#7a7a7a', alpha=0.9)
        ax.bar(x + w, f1s, w, label='F1', color='#b0b0b0', alpha=0.95)
        ax.set_xticks(x); ax.set_xticklabels(names, rotation=35, ha='right')
        ax.set_ylabel('Metric Value')
        ax.set_title('Сравнение моделей — Diameter')
        ax.set_ylim([0, 1.08]); ax.legend(loc='lower right')
        for i in range(len(names)):
            ax.text(i + w, f1s[i] + 0.01, f'{f1s[i]:.3f}',
                    ha='center', va='bottom', fontsize=7, fontweight='bold')
        self._save(fig, '04_summary_bars')

    def plot_feature_importance(self, feature_names, importances,
                                ms_features, top_n=20):
        idx = np.argsort(importances)[-top_n:]
        top_names = [feature_names[i] for i in idx]
        top_imps = importances[idx]
        is_ms = [n in ms_features for n in top_names]
        fig, ax = plt.subplots(figsize=(9, 7))
        colors = ['#2f2f2f' if ms else '#a6a6a6' for ms in is_ms]
        ax.barh(range(len(top_names)), top_imps, color=colors, alpha=0.85)
        ax.set_yticks(range(len(top_names))); ax.set_yticklabels(top_names)
        ax.set_xlabel('Gini Importance')
        ax.set_title(f'Top-{top_n} Feature Importance (RF) — Diameter')
        for i, v in enumerate(top_imps):
            ax.text(v + 0.001, i, f'{100*v:.1f}%', va='center', fontsize=7)
        legend_elements = [
            Patch(facecolor='#2f2f2f', alpha=0.85, label='MS features'),
            Patch(facecolor='#a6a6a6', alpha=0.85, label='Other features'),
        ]
        ax.legend(handles=legend_elements, loc='lower right')
        self._save(fig, '05_feature_importance')

    def plot_ablation(self, ablation_results):
        configs = [a['config'] for a in ablation_results]
        f1s = [a['f1'] for a in ablation_results]
        comp_rates = [a['compromise_rate'] for a in ablation_results]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        x = np.arange(len(configs))
        cs = ['#1f1f1f', '#4a4a4a', '#757575', '#9a9a9a', '#c0c0c0']
        ax1.bar(x, f1s, color=cs[:len(x)], alpha=0.85)
        ax1.set_xticks(x); ax1.set_xticklabels(configs, rotation=35,
                                                 ha='right')
        ax1.set_ylabel('F1'); ax1.set_title('Ablation: F1')
        ax1.set_ylim([0, 1.1])
        for i, v in enumerate(f1s):
            ax1.text(i, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold')

        ax2.bar(x, comp_rates, color=cs[:len(x)], alpha=0.85)
        ax2.set_xticks(x); ax2.set_xticklabels(configs, rotation=35,
                                                 ha='right')
        ax2.set_ylabel('Detection %')
        ax2.set_title('Ablation: Node Compromise')
        ax2.set_ylim([0, 110])
        for i, v in enumerate(comp_rates):
            ax2.text(i, v + 1.5, f'{v:.1f}%', ha='center', fontweight='bold')
        fig.tight_layout()
        self._save(fig, '06_ablation')

    def plot_training_curves(self, trainers_dict):
        if not trainers_dict:
            return
        n = len(trainers_dict)
        fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
        if n == 1:
            axes = [axes]
        for ax, (name, trainer) in zip(axes, trainers_dict.items()):
            epochs = range(1, len(trainer.train_losses) + 1)
            ax.plot(epochs, trainer.train_losses, 'k-', lw=1.5,
                    label='Train')
            if trainer.val_losses:
                ve = range(1, len(trainer.val_losses) + 1)
                ax.plot(ve, trainer.val_losses, color='#7a7a7a', ls='--',
                        lw=1.5, label='Val')
            ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
            ax.set_title(name); ax.legend(); ax.set_yscale('log')
        fig.suptitle('Training Curves — Diameter', fontsize=13)
        fig.tight_layout()
        self._save(fig, '07_training_curves')

    def plot_confusion_matrices(self, results_list, y_test, models):
        fig, axes = plt.subplots(1, len(models), figsize=(4 * len(models), 3.5))
        if len(models) == 1:
            axes = [axes]
        for ax, mname in zip(axes, models):
            r = next((r for r in results_list if r['name'] == mname), None)
            if r is None:
                continue
            cm = confusion_matrix(y_test, r['y_pred'], labels=[0, 1])
            ax.imshow(cm, cmap='Greys', aspect='auto')
            for i in range(2):
                for j in range(2):
                    c = 'white' if cm[i, j] > cm.max() * 0.5 else 'black'
                    ax.text(j, i, f'{cm[i,j]:,}', ha='center', va='center',
                            fontsize=11, fontweight='bold', color=c)
            ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
            ax.set_xticklabels(['Normal', 'Anomaly'])
            ax.set_yticklabels(['Normal', 'Anomaly'])
            ax.set_xlabel('Predicted'); ax.set_ylabel('True')
            ax.set_title(f'{mname}\nF1={r["f1"]:.3f}')
        fig.suptitle('Confusion Matrices — Diameter', y=1.02)
        fig.tight_layout()
        self._save(fig, '08_confusion_matrices')


# ══════════════════════════════════════════════════════════
#  MAIN TRAINING & EVALUATION
# ══════════════════════════════════════════════════════════

def main():
    config = TrainingConfig()
    os.makedirs(config.results_dir, exist_ok=True)
    os.makedirs(config.plots_dir, exist_ok=True)

    pipeline = DataPipeline(config)
    train_df, val_df, test_df = pipeline.load_and_prepare()

    X_train, y_train = pipeline.get_features_labels(train_df, fit=True)
    X_val, y_val = pipeline.get_features_labels(val_df)
    X_test, y_test = pipeline.get_features_labels(test_df)

    X_train_comp, y_train_comp = pipeline.get_compromise_features(
        train_df, fit=True)
    X_val_comp, y_val_comp = pipeline.get_compromise_features(val_df)
    X_test_comp, y_test_comp = pipeline.get_compromise_features(test_df)

    X_train_normal, _ = pipeline.get_normal_data(train_df, fit=False)

    attack_types_test = test_df["attack_type"].values
    n_feat = X_train.shape[1]

    results_list = []
    trainers_dict = {}

    print("\n" + "=" * 60)
    print("  Training Models — Diameter v1")
    print("=" * 60)

    # ── 1. Isolation Forest ──
    print("\n[1] Isolation Forest...")
    iso = IsolationForest(n_estimators=config.iso_forest_trees,
                          contamination=config.iso_forest_contamination,
                          random_state=config.seed)
    iso.fit(X_train_normal)
    iso_sv = -iso.score_samples(X_val)
    iso_st = -iso.score_samples(X_test)
    th_iso, _ = find_threshold(iso_sv, y_val, config.threshold_percentile)
    results_list.append(evaluate_model("Isolation Forest", iso_st, y_test,
                                       attack_types_test, threshold=th_iso))

    # ── 2. One-Class SVM ──
    print("\n[2] One-Class SVM...")
    svm = OneClassSVM(nu=config.ocsvm_nu, kernel=config.ocsvm_kernel,
                      gamma="scale")
    svm.fit(X_train_normal)
    svm_sv = -svm.score_samples(X_val)
    svm_st = -svm.score_samples(X_test)
    th_svm, _ = find_threshold(svm_sv, y_val, config.threshold_percentile)
    results_list.append(evaluate_model("One-Class SVM", svm_st, y_test,
                                       attack_types_test, threshold=th_svm))

    # ── 3. Autoencoder ──
    print("\n[3] Autoencoder...")
    ae = Autoencoder(n_feat, config.ae_hidden_dims, config.ae_latent_dim)
    ae_trainer = AETrainer(ae, config.ae_lr, config.ae_epochs,
                           config.ae_batch, patience=15)
    X_val_normal = X_val[y_val == 0]
    ae_trainer.fit(X_train_normal, X_val_normal)
    trainers_dict['Autoencoder'] = ae_trainer
    ae_sv = ae_trainer.reconstruction_error(X_val)
    ae_st = ae_trainer.reconstruction_error(X_test)
    th_ae, _ = find_threshold(ae_sv, y_val, config.threshold_percentile)
    results_list.append(evaluate_model("Autoencoder", ae_st, y_test,
                                       attack_types_test, threshold=th_ae))

    # ── 4. VAE ──
    print("\n[4] VAE...")
    vae = VAE(n_feat, config.vae_hidden_dims, config.vae_latent_dim)
    vae_trainer = VAETrainer(vae, config.vae_lr, config.vae_epochs,
                             config.vae_batch,
                             kl_weight=config.vae_kl_weight, patience=15)
    vae_trainer.fit(X_train_normal, X_val_normal)
    trainers_dict['VAE'] = vae_trainer
    vae_sv = vae_trainer.anomaly_score(X_val)
    vae_st = vae_trainer.anomaly_score(X_test)
    th_vae, _ = find_threshold(vae_sv, y_val, config.threshold_percentile)
    results_list.append(evaluate_model("VAE", vae_st, y_test,
                                       attack_types_test, threshold=th_vae))

    # ── 5. LSTM-AE ──
    print("\n[5] LSTM-Autoencoder...")
    train_normal_df = train_df[train_df[pipeline.LABEL_COL] == 0]
    seq_train, _ = pipeline.get_sequential_data(train_normal_df,
                                                 config.lstm_seq_len)
    lstm_done = False
    if len(seq_train) > 100:
        lstm = LSTMAutoencoder(n_feat, config.lstm_hidden,
                               config.lstm_layers,
                               config.lstm_seq_len, dropout=0.2)
        lstm_trainer = AETrainer(lstm, config.lstm_lr, config.lstm_epochs,
                                 config.lstm_batch, patience=15)
        seq_val, seq_y_val = pipeline.get_sequential_data(
            val_df, config.lstm_seq_len)
        seq_val_normal = seq_val[seq_y_val == 0]
        lstm_trainer.fit(seq_train,
                         seq_val_normal if len(seq_val_normal) > 0 else None)
        trainers_dict['LSTM-AE'] = lstm_trainer

        seq_test, seq_y_test = pipeline.get_sequential_data(
            test_df, config.lstm_seq_len)
        seq_at_test = []
        for nid in test_df["node_id"].unique():
            mask = test_df["node_id"].values == nid
            atypes = test_df.loc[mask, "attack_type"].values
            for i in range(len(atypes) - config.lstm_seq_len + 1):
                seq_at_test.append(atypes[i + config.lstm_seq_len - 1])
        seq_at_test = np.array(seq_at_test)

        lstm_st = lstm_trainer.reconstruction_error(seq_test)
        lstm_sv = lstm_trainer.reconstruction_error(seq_val)
        th_lstm, _ = find_threshold(lstm_sv, seq_y_val,
                                    config.threshold_percentile)
        results_list.append(evaluate_model(
            "LSTM-Autoencoder", lstm_st, seq_y_test,
            seq_at_test, threshold=th_lstm))
        lstm_done = True
    else:
        print("  Skipped: not enough sequences.")

    # ── 6. Temporal-Attention-AE ──
    print("\n[6] Temporal-Attention-AE...")
    if lstm_done and len(seq_train) > 100:
        tat = TemporalAttentionAE(n_feat, 64, 4, 2,
                                  config.lstm_seq_len, 0.15)
        tat_trainer = AETrainer(tat, 5e-4, 100, 256, patience=15)
        tat_trainer.fit(seq_train,
                        seq_val_normal if len(seq_val_normal) > 0 else None)
        trainers_dict['TA-AE'] = tat_trainer
        tat_st = tat_trainer.reconstruction_error(seq_test)
        tat_sv = tat_trainer.reconstruction_error(seq_val)
        th_tat, _ = find_threshold(tat_sv, seq_y_val,
                                   config.threshold_percentile)
        results_list.append(evaluate_model(
            "Temporal-Attention-AE", tat_st, seq_y_test,
            seq_at_test, threshold=th_tat))

    # ── 7. Random Forest ──
    print("\n[7] Random Forest...")
    rf = RandomForestClassifier(n_estimators=config.rf_trees,
                                random_state=config.seed,
                                class_weight="balanced", n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_pt = rf.predict_proba(X_test)[:, 1]
    results_list.append(evaluate_model("Random Forest", rf_pt, y_test,
                                       attack_types_test, threshold=0.5))

    importances = rf.feature_importances_
    print("\n  Feature importance (top 20):")
    feat_imp = sorted(zip(pipeline.feature_cols, importances),
                      key=lambda x: x[1], reverse=True)
    for fn, imp in feat_imp[:20]:
        print(f"    {fn:45s} {imp:.4f} ({100*imp:.1f}%)")

    ms_imp = sum(imp for fn, imp in feat_imp
                 if fn in MS_FEATURES)
    print(f"\n  MS features total importance: {ms_imp:.4f} "
          f"({100 * ms_imp:.1f}%)")

    # ── 8. Gradient Boosting ──
    print("\n[8] Gradient Boosting...")
    gb = GradientBoostingClassifier(n_estimators=config.gb_estimators,
                                    learning_rate=config.gb_lr,
                                    max_depth=5, random_state=config.seed)
    gb.fit(X_train, y_train)
    gb_pt = gb.predict_proba(X_test)[:, 1]
    results_list.append(evaluate_model("Gradient Boosting", gb_pt, y_test,
                                       attack_types_test, threshold=0.5))

    # ── 9. Cascade ──
    print("\n[9] Cascade Compromise Detector...")
    cascade = CascadeCompromiseDetector(config, config.seed)
    cascade.fit(X_train, y_train, X_train_comp, y_train_comp)
    casc_pt = cascade.predict_proba(X_test, X_test_comp)
    results_list.append(evaluate_model("Cascade", casc_pt, y_test,
                                       attack_types_test, threshold=0.5))

    # ── 10. Specialized Ensemble ──
    print("\n[10] Specialized Ensemble...")
    ensemble = EnsembleDetector(FEATURE_GROUPS, pipeline.feature_cols,
                                config.seed)
    ensemble.fit(X_train_normal)
    ens_sv = ensemble.score(X_val)
    ens_st = ensemble.score(X_test)
    ens_results = []
    for en in sorted(ens_st.keys()):
        sv, st = ens_sv[en], ens_st[en]
        th, _ = find_threshold(sv, y_val, config.threshold_percentile)
        r = evaluate_model(f"Ens:{en}", st, y_test, attack_types_test,
                           threshold=th)
        results_list.append(r)
        ens_results.append(r)

    # ── 11. Meta-Ensemble ──
    print("\n[11] Meta-Ensemble...")

    def norm_score(s):
        p1, p99 = np.percentile(s, 1), np.percentile(s, 99)
        return np.clip((s - p1) / (p99 - p1 + 1e-9), 0, 1)

    meta_train = np.column_stack([
        norm_score(-iso.score_samples(X_train)),
        norm_score(-svm.score_samples(X_train)),
        norm_score(ae_trainer.reconstruction_error(X_train)),
        norm_score(vae_trainer.anomaly_score(X_train)),
        norm_score(cascade.predict_proba(X_train, X_train_comp)),
        norm_score(rf.predict_proba(X_train)[:, 1]),
        norm_score(gb.predict_proba(X_train)[:, 1]),
    ])
    meta_test = np.column_stack([
        norm_score(iso_st), norm_score(svm_st),
        norm_score(ae_st), norm_score(vae_st),
        norm_score(casc_pt),
        norm_score(rf_pt), norm_score(gb_pt),
    ])

    meta_gb = GradientBoostingClassifier(
        n_estimators=200, learning_rate=0.05, max_depth=3,
        random_state=config.seed)
    meta_gb.fit(meta_train, y_train)
    meta_pt = meta_gb.predict_proba(meta_test)[:, 1]
    results_list.append(evaluate_model("Meta-Ensemble", meta_pt, y_test,
                                       attack_types_test, threshold=0.5))

    # ══════════════════════════════════════════════════
    #  Ablation Studies
    # ══════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  Ablation Studies")
    print("=" * 60)

    ablation = []
    comp_mask = attack_types_test == "node_compromise"
    comp_total = comp_mask.sum()

    # Full
    comp_full = (rf_pt[comp_mask] >= 0.5).sum()
    rf_r = next(r for r in results_list if r['name'] == 'Random Forest')
    ablation.append({
        'config': f'Full\n({len(pipeline.feature_cols)} feat)',
        'f1': rf_r['f1'],
        'compromise_rate': 100 * comp_full / max(1, comp_total),
    })

    # Without MS features
    non_ms = [c for c in pipeline.feature_cols if c not in MS_FEATURES]
    non_ms_idx = [pipeline.feature_cols.index(c) for c in non_ms]
    rf_no_ms = RandomForestClassifier(
        n_estimators=config.rf_trees, random_state=config.seed,
        class_weight="balanced", n_jobs=-1)
    rf_no_ms.fit(X_train[:, non_ms_idx], y_train)
    rf_no_ms_p = rf_no_ms.predict_proba(X_test[:, non_ms_idx])[:, 1]
    r_no_ms = evaluate_model("RF (no MS)", rf_no_ms_p, y_test,
                             attack_types_test, threshold=0.5)
    comp_no_ms = (rf_no_ms_p[comp_mask] >= 0.5).sum()
    ablation.append({
        'config': f'No MS\n({len(non_ms)} feat)',
        'f1': r_no_ms['f1'],
        'compromise_rate': 100 * comp_no_ms / max(1, comp_total),
    })

    # Without temporal
    temporal = DataPipeline.TEMPORAL_FEATURES
    non_temp = [c for c in pipeline.feature_cols if c not in temporal]
    non_temp_idx = [pipeline.feature_cols.index(c) for c in non_temp]
    rf_no_temp = RandomForestClassifier(
        n_estimators=config.rf_trees, random_state=config.seed,
        class_weight="balanced", n_jobs=-1)
    rf_no_temp.fit(X_train[:, non_temp_idx], y_train)
    rf_no_temp_p = rf_no_temp.predict_proba(X_test[:, non_temp_idx])[:, 1]
    r_no_temp = evaluate_model("RF (no temporal)", rf_no_temp_p, y_test,
                               attack_types_test, threshold=0.5)
    comp_no_temp = (rf_no_temp_p[comp_mask] >= 0.5).sum()
    ablation.append({
        'config': f'No temporal\n({len(non_temp)} feat)',
        'f1': r_no_temp['f1'],
        'compromise_rate': 100 * comp_no_temp / max(1, comp_total),
    })

    # Cascade stage 1 only
    s1_scores = cascade.predict_proba_stage1(X_test)
    r_s1 = evaluate_model("Cascade-S1", s1_scores, y_test,
                          attack_types_test, threshold=0.5)
    comp_s1 = (s1_scores[comp_mask] >= 0.5).sum()
    ablation.append({
        'config': 'Cascade\nStage-1 only',
        'f1': r_s1['f1'],
        'compromise_rate': 100 * comp_s1 / max(1, comp_total),
    })

    # Cascade full
    casc_r = next(r for r in results_list if r['name'] == 'Cascade')
    comp_casc = (casc_pt[comp_mask] >= 0.5).sum()
    ablation.append({
        'config': 'Cascade\nBoth levels',
        'f1': casc_r['f1'],
        'compromise_rate': 100 * comp_casc / max(1, comp_total),
    })

    # ══════════════════════════════════════════════════
    #  Save Results
    # ══════════════════════════════════════════════════
    save_cols = ['name', 'auc_roc', 'auc_pr', 'f1', 'fpr',
                 'tp', 'fp', 'fn', 'tn']
    res_df = pd.DataFrame([{k: r[k] for k in save_cols}
                           for r in results_list])
    res_df.to_csv(os.path.join(config.results_dir,
                               "full_comparison_diameter.csv"), index=False)

    with open(os.path.join(config.results_dir,
                           "feature_columns_diameter.json"), "w") as f:
        json.dump({
            "observable_features": pipeline.feature_cols,
            "compromise_features": pipeline.compromise_feature_cols,
            "ms_features": [c for c in MS_FEATURES
                            if c in pipeline.feature_cols],
            "temporal_features": [c for c in temporal
                                  if c in pipeline.feature_cols],
            "feature_groups": {k: [c for c in v
                                   if c in pipeline.feature_cols]
                               for k, v in FEATURE_GROUPS.items()},
        }, f, indent=2)

    # Summary
    print("\n" + "=" * 70)
    print("  SUMMARY — Diameter v1")
    print("=" * 70)
    print(f"{'Model':<35s} {'AUC-ROC':>8s} {'AUC-PR':>8s} "
          f"{'F1':>8s} {'FPR':>8s}")
    print("-" * 67)
    for r in results_list:
        print(f"{r['name']:<35s} {r['auc_roc']:8.4f} "
              f"{r['auc_pr']:8.4f} {r['f1']:8.4f} {r['fpr']:8.4f}")

    # ══════════════════════════════════════════════════
    #  Generate Plots
    # ══════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  Generating plots...")
    print("=" * 60)

    viz = ResultsVisualizer(config.plots_dir)
    main_models = [
        'Isolation Forest', 'One-Class SVM', 'Autoencoder', 'VAE',
        'Random Forest', 'Gradient Boosting', 'Cascade', 'Meta-Ensemble'
    ]

    viz.plot_roc_curves(results_list, y_test, main_models)
    viz.plot_pr_curves(results_list, y_test, main_models)
    viz.plot_attack_heatmap(results_list, main_models)
    viz.plot_summary_bars(results_list, main_models)
    viz.plot_feature_importance(pipeline.feature_cols, importances,
                                MS_FEATURES)
    viz.plot_ablation(ablation)
    viz.plot_training_curves(trainers_dict)
    viz.plot_confusion_matrices(
        results_list, y_test,
        ['VAE', 'Random Forest', 'Cascade', 'Meta-Ensemble'])

        # ══════════════════════════════════════════════════
    #  Сохранение артефактов для отдельной визуализации
    # ══════════════════════════════════════════════════
    import pickle

    # История обучения нейросетевых моделей
    trainers_history = {}
    for name, trainer in trainers_dict.items():
        trainers_history[name] = {
            "train_losses": list(trainer.train_losses),
            "val_losses": list(trainer.val_losses),
        }

    # Для последовательных моделей y_test отличается (seq_y_test).
    # Приводим scores к длине y_test, где это возможно; иначе скрипт
    # визуализации сам отфильтрует такие модели по длине.
    artifacts = {
        "results_list": results_list,
        "ens_results": ens_results,
        "ablation_results": ablation,
        "y_test": y_test,
        "attack_types_test": attack_types_test,
        "importances": importances,
        "feature_cols": pipeline.feature_cols,
        "cascade_s1": cascade.predict_proba_stage1(X_test),
        "cascade_s2": cascade.predict_proba_stage2(X_test_comp),
        "cascade_comb": casc_pt,
        "trainers_history": trainers_history,
    }

    artifacts_path = os.path.join(config.results_dir, "plot_artifacts.pkl")
    with open(artifacts_path, "wb") as f:
        pickle.dump(artifacts, f)
    print(f"\n  Plot artifacts saved to {artifacts_path}")


    print(f"\n  All results saved to {config.results_dir}/")
    print(f"  Plots saved to {config.plots_dir}/")


if __name__ == "__main__":
    main()
