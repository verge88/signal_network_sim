"""
SIP Master-Slave Anomaly Detection Training Pipeline v4 (без ZCAD)
==================================================================
Адаптация под датасет sip_dataset_v4.csv (симулятор v4, Вариант A:
транзитное наблюдение ACK на SBC). Отличия от v3:

  * Новые физически обоснованные признаки транзитного ACK-инварианта:
      - cu_ack_closure_mismatch  (I2: dcr_report - dcr_transit)
      - dev_ack_mismatch_vs_zone (I2 в зональном контексте)
    Включены в наблюдаемый набор как легитимный сигнал мастера.
  * ИСКЛЮЧЕНЫ сырые транзитные счётчики и прямой транзитный dcr —
    это утечка физической истины (даёт нечестное преимущество supervised):
      cu_dcr_transit, cu_n_ack_transit, cu_n_200_transit.
  * ZCAD полностью удалён (кастомная модель оценивается отдельно).
  * Добавлена разбивка compromise-recall по attacker_sophistication
    (naive/statistical/adaptive) — направление 1.

Модели: Isolation Forest, One-Class SVM, Autoencoder, VAE, LSTM-AE,
Temporal-Attention-AE, Random Forest, Gradient Boosting, Cascade,
Specialized Ensemble, Meta-Ensemble.
"""

import numpy as np
import pandas as pd
import os
import json
import warnings
from dataclasses import dataclass, field
from typing import List

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

SIP_METHODS = ["INVITE", "ACK", "BYE", "REGISTER", "OPTIONS",
               "CANCEL", "SUBSCRIBE", "NOTIFY", "OTHER"]
SIP_RESP_CLASSES = ["resp_1xx", "resp_2xx", "resp_3xx", "resp_4xx",
                    "resp_401_407", "resp_408", "resp_5xx", "resp_6xx"]

# Признаки, которые НЕЛЬЗЯ давать моделям — прямая физическая истина
# (утечка из симулятора Варианта A). Исключаются на этапе отбора.
LEAKY_FEATURES = {
    "cu_dcr_transit", "cu_n_ack_transit", "cu_n_200_transit",
    "cu_n_ack_observed", "cu_n_invite_200",   # старые имена v4-черновика
}


# ══════════════════════════════════════════════════════════
#  Configuration
# ══════════════════════════════════════════════════════════

@dataclass
class TrainingConfig:
    data_path: str = "sip_dataset_v4.csv"
    results_dir: str = "results_sip_v4"
    plots_dir: str = "results_sip_v4/plots"
    test_size: float = 0.25
    val_size: float = 0.15
    seed: int = 42

    iso_forest_trees: int = 300
    iso_forest_contamination: float = 0.05
    ocsvm_nu: float = 0.05
    ocsvm_kernel: str = "rbf"

    ae_hidden_dims: List[int] = field(default_factory=lambda: [64, 32])
    ae_latent_dim: int = 12
    ae_epochs: int = 100
    ae_batch: int = 256
    ae_lr: float = 1e-3

    vae_hidden_dims: List[int] = field(default_factory=lambda: [64, 32])
    vae_latent_dim: int = 12
    vae_epochs: int = 120
    vae_batch: int = 256
    vae_lr: float = 1e-3
    vae_kl_weight: float = 0.5

    lstm_hidden: int = 48
    lstm_layers: int = 2
    lstm_seq_len: int = 10
    lstm_epochs: int = 80
    lstm_batch: int = 256
    lstm_lr: float = 1e-3

    rf_trees: int = 300
    gb_estimators: int = 300
    gb_lr: float = 0.05

    cascade_rf_trees: int = 500
    cascade_gb_estimators: int = 500

    threshold_percentile: float = 95.0


# ══════════════════════════════════════════════════════════
#  Data Pipeline — SIP v4 features
# ══════════════════════════════════════════════════════════

class DataPipeline:
    # v4: добавлен cu_ack_closure_mismatch (I2, транзитный ACK-инвариант)
    CU_FEATURES = [
        "cu_delivered", "cu_rtt_delay_ms", "cu_req_delay_ms",
        "cu_resp_delay_ms", "cu_combined_loss_prob", "integrity_check",
        "consecutive_cu_losses", "cu_processing_delay_ms",
        "cu_response_time_jitter", "cu_staleness", "cu_consistency_error",
        "cu_txn_state_consistency_error",
        "cu_ack_closure_mismatch",     # НОВОЕ v4: I2 (Вариант A)
    ]
    REPORTED_STATS = [
        "obs_total_messages", "obs_resp_total", "obs_entropy",
        "obs_inbound_outbound_ratio", "obs_international_fraction",
        "obs_n_unique_destinations", "obs_method_dominance_ratio",
        "obs_method_gini", "obs_invite_ack_ratio",
    ]
    TRANSACTION_FEATURES = [
        "obs_forking_amplification", "obs_invite_timeout_ratio",
        "obs_invite_mean_retransmits", "obs_invite_frac_retransmitted",
        "obs_dialog_completion_ratio", "obs_dangling_dialog_ratio",
        "obs_noninvite_timeout_ratio", "obs_txn_timeout_ratio",
    ]
    METHOD_RATIO_FEATURES = [f"obs_ratio_{m.lower()}" for m in SIP_METHODS]
    RESP_RATIO_FEATURES = [f"obs_ratio_{rc}" for rc in SIP_RESP_CLASSES]

    Z_FEATURES = [
        "z_obs_total_messages", "z_obs_method_dominance_ratio",
        "z_obs_ratio_invite", "z_obs_ratio_register",
        "z_obs_ratio_resp_401_407", "z_obs_dangling_dialog_ratio",
        "z_obs_invite_timeout_ratio", "z_obs_forking_amplification",
    ]
    TEMPORAL_FEATURES = [
        "var_obs_total_messages", "var_obs_entropy",
        "var_obs_inbound_outbound_ratio", "var_obs_method_dominance_ratio",
        "var_obs_invite_ack_ratio", "var_obs_dialog_completion_ratio",
        "var_obs_forking_amplification", "autocorr_obs_total_messages",
        "integrity_fail_rate", "obs_destinations_cv",
        "mean_txn_state_consistency_error",
    ]
    # v4: добавлено зональное отклонение ACK-инварианта
    ZONE_FEATURES = [
        "dev_total_messages_vs_zone", "dev_ratio_invite_vs_zone",
        "dev_ratio_register_vs_zone", "zone_io_ratio_rank",
        "dev_io_ratio_vs_zone", "dev_integrity_fail_rate_vs_zone",
        "dev_var_total_vs_zone", "dev_dialog_completion_ratio_vs_zone",
        "dev_forking_amplification_vs_zone",
        "dev_dangling_dialog_ratio_vs_zone", "dev_txn_consistency_vs_zone",
        "dev_ack_mismatch_vs_zone",     # НОВОЕ v4: I2 зональный (Вариант A)
    ]

    LABEL_COL = "is_anomaly"
    ATTACK_COL = "attack_type"

    # v4: compromise-набор дополнен ACK-инвариантом (главный рычаг adaptive)
    COMPROMISE_FEATURES = [
        "obs_inbound_outbound_ratio", "integrity_check",
        "integrity_fail_rate", "cu_consistency_error",
        "cu_txn_state_consistency_error", "mean_txn_state_consistency_error",
        "cu_ack_closure_mismatch", "dev_ack_mismatch_vs_zone",  # НОВОЕ v4
        "cu_processing_delay_ms", "cu_response_time_jitter",
        "var_obs_total_messages", "var_obs_entropy",
        "var_obs_inbound_outbound_ratio", "var_obs_method_dominance_ratio",
        "var_obs_invite_ack_ratio", "var_obs_dialog_completion_ratio",
        "var_obs_forking_amplification", "autocorr_obs_total_messages",
        "obs_destinations_cv", "zone_io_ratio_rank", "dev_io_ratio_vs_zone",
        "dev_integrity_fail_rate_vs_zone", "dev_var_total_vs_zone",
        "dev_txn_consistency_vs_zone", "dev_dialog_completion_ratio_vs_zone",
        "dev_dangling_dialog_ratio_vs_zone",
        "consecutive_cu_losses", "cu_staleness", "dev_total_messages_vs_zone",
        "obs_dialog_completion_ratio", "obs_dangling_dialog_ratio",
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

    def _build_feature_cols(self, df):
        all_candidate = (
            self.CU_FEATURES + self.REPORTED_STATS +
            self.TRANSACTION_FEATURES +
            self.METHOD_RATIO_FEATURES + self.RESP_RATIO_FEATURES +
            self.Z_FEATURES + self.ZONE_FEATURES + self.TEMPORAL_FEATURES
        )
        # только присутствующие И не «протекающие» признаки
        self.feature_cols = [c for c in all_candidate
                             if c in df.columns and c not in LEAKY_FEATURES]
        if "node_type_enc" in df.columns:
            self.feature_cols.append("node_type_enc")

        self.compromise_feature_cols = [
            c for c in self.COMPROMISE_FEATURES
            if c in df.columns and c not in LEAKY_FEATURES]
        if "node_type_enc" in df.columns:
            self.compromise_feature_cols.append("node_type_enc")

    def load_and_prepare(self):
        df = pd.read_csv(self.config.data_path)
        print(f"Loaded {len(df)} records, anomalies: "
              f"{df[self.LABEL_COL].sum()} "
              f"({100 * df[self.LABEL_COL].mean():.2f}%)")
        print(f"Attack distribution:\n{df[self.ATTACK_COL].value_counts()}\n")

        if "node_type" in df.columns and "node_type_enc" not in df.columns:
            df["node_type_enc"] = self.le_node_type.fit_transform(
                df["node_type"])

        self._build_feature_cols(df)
        for c in self.feature_cols:
            if c.startswith("gt_"):
                raise ValueError(f"GT column '{c}' in feature set!")
            if c in LEAKY_FEATURES:
                raise ValueError(f"Leaky column '{c}' in feature set!")

        txn_in = [c for c in self.feature_cols
                  if c in (self.TRANSACTION_FEATURES +
                           ["cu_txn_state_consistency_error",
                            "mean_txn_state_consistency_error",
                            "dev_txn_consistency_vs_zone",
                            "cu_ack_closure_mismatch",
                            "dev_ack_mismatch_vs_zone"])]
        print(f"Observable features: {len(self.feature_cols)}")
        print(f"  of which SIP transaction-integrity features: "
              f"{len(txn_in)}")
        ack_in = [c for c in self.feature_cols
                  if c in ("cu_ack_closure_mismatch",
                           "dev_ack_mismatch_vs_zone")]
        print(f"  ACK-invariant (I2, Вариант A) features: {ack_in}")
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
        y = (df[self.ATTACK_COL] == "proxy_compromise").astype(int).values
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


MS_FEATURES = (
    DataPipeline.CU_FEATURES +
    ["obs_total_messages", "obs_resp_total", "obs_inbound_outbound_ratio"] +
    list(DataPipeline.ZONE_FEATURES) +
    DataPipeline.TEMPORAL_FEATURES
)


FEATURE_GROUPS = {
    "method_profile": (
        DataPipeline.METHOD_RATIO_FEATURES +
        ["obs_method_dominance_ratio", "obs_method_gini",
         "obs_invite_ack_ratio", "z_obs_method_dominance_ratio",
         "z_obs_ratio_invite", "z_obs_ratio_register"]
    ),
    "response_profile": (
        DataPipeline.RESP_RATIO_FEATURES + ["z_obs_ratio_resp_401_407"]
    ),
    "volume_timing": [
        "obs_total_messages", "obs_resp_total", "obs_entropy",
        "z_obs_total_messages",
    ],
    "control_unit": DataPipeline.CU_FEATURES,
    "spit_indicators": [
        "obs_international_fraction", "obs_n_unique_destinations",
        "obs_destinations_cv", "obs_forking_amplification",
    ],
    "transaction_integrity": [
        "obs_dialog_completion_ratio", "obs_dangling_dialog_ratio",
        "obs_invite_timeout_ratio", "obs_invite_mean_retransmits",
        "obs_invite_frac_retransmitted", "obs_noninvite_timeout_ratio",
        "obs_txn_timeout_ratio", "obs_forking_amplification",
        "cu_txn_state_consistency_error",
        "cu_ack_closure_mismatch",       # НОВОЕ v4: I2
        "z_obs_dangling_dialog_ratio", "z_obs_invite_timeout_ratio",
    ],
    "baseline_deviation": [
        "dev_total_messages_vs_zone", "dev_ratio_invite_vs_zone",
        "dev_ratio_register_vs_zone", "obs_inbound_outbound_ratio",
        "zone_io_ratio_rank", "dev_io_ratio_vs_zone",
        "dev_txn_consistency_vs_zone",
        "dev_dialog_completion_ratio_vs_zone",
        "dev_dangling_dialog_ratio_vs_zone",
        "dev_ack_mismatch_vs_zone",      # НОВОЕ v4: I2 зональный
    ],
    "compromise_indicators": DataPipeline.COMPROMISE_FEATURES,
}

ATTACK_LABELS = {
    'signaling_dos': 'Сигнальный DoS (INVITE-флуд)',
    'register_hijack': 'Перехват регистрации',
    'register_bruteforce': 'Брутфорс REGISTER',
    'spit': 'SPIT',
    'proxy_compromise': 'Компрометация прокси',
}


# ══════════════════════════════════════════════════════════
#  Neural Network Models (unsupervised)  — без изменений
# ══════════════════════════════════════════════════════════

class Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dims, latent_dim):
        super().__init__()
        enc, prev = [], input_dim
        for h in hidden_dims:
            enc.extend([nn.Linear(prev, h), nn.ReLU(), nn.BatchNorm1d(h)])
            prev = h
        enc.append(nn.Linear(prev, latent_dim))
        self.encoder = nn.Sequential(*enc)
        dec, prev = [], latent_dim
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
        enc, prev = [], input_dim
        for h in hidden_dims:
            enc.extend([nn.Linear(prev, h), nn.ReLU(), nn.BatchNorm1d(h)])
            prev = h
        self.encoder = nn.Sequential(*enc)
        self.fc_mu = nn.Linear(prev, latent_dim)
        self.fc_logvar = nn.Linear(prev, latent_dim)
        dec, prev = [], latent_dim
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
    def __init__(self, input_dim, hidden_dim, n_layers, seq_len, dropout=0.2):
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
#  Trainers — без изменений
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
            lv = torch.clamp(lv, min=-10.0, max=10.0)
            recon = torch.mean((t - x_hat) ** 2, dim=-1)
            kl = -0.5 * torch.sum(1 + lv - mu.pow(2) - lv.exp(), dim=-1)
            score = recon + self.kl_weight * kl
            score = torch.nan_to_num(score, nan=0.0, posinf=1e6, neginf=0.0)
        return score.cpu().numpy()


# ══════════════════════════════════════════════════════════
#  Evaluation Utilities  (+ разбивка по sophistication)
# ══════════════════════════════════════════════════════════

def find_threshold(scores, y_true, percentile=95.0):
    scores = np.asarray(scores, dtype=np.float64)
    if not np.all(np.isfinite(scores)):
        finite = scores[np.isfinite(scores)]
        hi = finite.max() if finite.size else 1.0
        lo = finite.min() if finite.size else 0.0
        scores = np.nan_to_num(scores, nan=lo, posinf=hi, neginf=lo)
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
                   threshold=None, percentile=95.0, sophistication=None):
    scores = np.asarray(scores, dtype=np.float64)
    if not np.all(np.isfinite(scores)):
        finite = scores[np.isfinite(scores)]
        hi = finite.max() if finite.size else 1.0
        lo = finite.min() if finite.size else 0.0
        scores = np.nan_to_num(scores, nan=lo, posinf=hi, neginf=lo)
        print(f"  [warn] {name}: inf/NaN в скорах заменены")

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
            print(f"    {label:32s}: {det}/{tot} = {100 * det / tot:.1f}%")

    # ── Направление 1: compromise-recall по уровням изощрённости ──
    per_soph = {}
    if sophistication is not None:
        soph = np.asarray(sophistication)
        for lv in ["naive", "statistical", "adaptive"]:
            m = (soph == lv)
            if m.sum() > 0:
                per_soph[lv] = 100.0 * y_pred[m].sum() / m.sum()
        if per_soph:
            print("  Compromise recall by sophistication:")
            for lv in ["naive", "statistical", "adaptive"]:
                if lv in per_soph:
                    print(f"    {lv:12s}: {per_soph[lv]:.1f}%")

    return {
        "name": name, "auc_roc": auc_roc, "auc_pr": auc_pr, "f1": f1,
        "fpr": fpr, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "threshold": threshold, "method": method,
        "per_attack": per_attack, "per_soph": per_soph,
        "scores": scores, "y_pred": y_pred,
    }


# ══════════════════════════════════════════════════════════
#  Specialized Ensemble  — без изменений
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
#  Cascade Compromise Detector  — без изменений
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
#  Visualization  (сохранена, ссылки на ZCAD убраны)
# ══════════════════════════════════════════════════════════

class ResultsVisualizer:
    def __init__(self, plots_dir):
        self.plots_dir = plots_dir
        os.makedirs(plots_dir, exist_ok=True)

    def _save(self, fig, name):
        for fmt in ["png", "svg"]:
            fig.savefig(os.path.join(self.plots_dir, f"{name}.{fmt}"),
                        bbox_inches='tight', facecolor='white', format=fmt)
        plt.close(fig)
        print(f"  Saved: {name}.png / .svg")

    def plot_roc_curves(self, results_list, y_test, main_models):
        fig, ax = plt.subplots(figsize=(8, 6))
        for r in results_list:
            if r['name'] not in main_models:
                continue
            if len(np.asarray(r['scores'])) != len(y_test):
                continue
            fpr_arr, tpr_arr, _ = roc_curve(y_test, r['scores'])
            ax.plot(fpr_arr, tpr_arr, linewidth=1.8,
                    label=f"{r['name']} (AUC={r['auc_roc']:.3f})")
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
        ax.set_xlabel('FPR'); ax.set_ylabel('TPR')
        ax.set_title('ROC-кривые — SIP v4'); ax.legend(loc='lower right')
        self._save(fig, '01_roc_curves')

    def plot_pr_curves(self, results_list, y_test, main_models):
        fig, ax = plt.subplots(figsize=(8, 6))
        for r in results_list:
            if r['name'] not in main_models:
                continue
            if len(np.asarray(r['scores'])) != len(y_test):
                continue
            prec, rec, _ = precision_recall_curve(y_test, r['scores'])
            ax.plot(rec, prec, linewidth=1.8,
                    label=f"{r['name']} (AP={r['auc_pr']:.3f})")
        ax.axhline(y=y_test.mean(), color='k', ls='--', alpha=0.3,
                   label=f'Baseline ({y_test.mean():.3f})')
        ax.set_xlabel('Recall'); ax.set_ylabel('Precision')
        ax.set_title('Precision-Recall — SIP v4')
        ax.legend(loc='lower left')
        self._save(fig, '02_precision_recall')

    def plot_summary_bars(self, results_list, main_models):
        names, rocs, prs, f1s = [], [], [], []
        for r in results_list:
            if r['name'] in main_models:
                names.append(r['name'])
                rocs.append(r['auc_roc']); prs.append(r['auc_pr'])
                f1s.append(r['f1'])
        if not names:
            return
        x = np.arange(len(names)); w = 0.25
        fig, ax = plt.subplots(figsize=(13, 5))
        ax.bar(x - w, rocs, w, label='AUC-ROC', color='#2f2f2f', alpha=0.9)
        ax.bar(x, prs, w, label='AUC-PR', color='#7a7a7a', alpha=0.9)
        ax.bar(x + w, f1s, w, label='F1', color='#b0b0b0', alpha=0.95)
        ax.set_xticks(x); ax.set_xticklabels(names, rotation=35, ha='right')
        ax.set_ylabel('Metric Value')
        ax.set_title('Сравнение моделей — SIP v4')
        ax.set_ylim([0, 1.08]); ax.legend(loc='lower right')
        self._save(fig, '04_summary_bars')

    def plot_soph_breakdown(self, results_list, main_models):
        """НОВОЕ v4: compromise-recall по уровням изощрённости (направление 1)."""
        rows, names = [], []
        for r in results_list:
            if r['name'] not in main_models:
                continue
            ps = r.get('per_soph', {})
            if not ps:
                continue
            rows.append([ps.get('naive', 0), ps.get('statistical', 0),
                         ps.get('adaptive', 0)])
            names.append(r['name'])
        if not rows:
            return
        mat = np.array(rows)
        fig, ax = plt.subplots(figsize=(9, 0.5 * len(names) + 3))
        im = ax.imshow(mat, cmap='Greys', aspect='auto', vmin=0, vmax=100)
        ax.set_xticks(range(3))
        ax.set_xticklabels(['naive', 'statistical', 'adaptive'])
        ax.set_yticks(range(len(names))); ax.set_yticklabels(names)
        for i in range(len(names)):
            for j in range(3):
                v = mat[i, j]
                c = 'white' if v > 60 else 'black'
                ax.text(j, i, f'{v:.0f}%', ha='center', va='center',
                        fontsize=9, fontweight='bold', color=c)
        plt.colorbar(im, ax=ax, label='Recall %', shrink=0.8)
        ax.set_title('Полнота обнаружения компрометации\n'
                     'по уровню изощрённости атакующего — SIP v4')
        self._save(fig, '11_sophistication_breakdown')

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
        ax.set_title(f'Top-{top_n} Feature Importance (RF) — SIP v4')
        ax.legend(handles=[
            Patch(facecolor='#2f2f2f', alpha=0.85, label='MS features'),
            Patch(facecolor='#a6a6a6', alpha=0.85, label='Other features'),
        ], loc='lower right')
        self._save(fig, '05_feature_importance')

    def plot_confusion_matrices(self, results_list, y_test, models):
        models = [m for m in models
                  if next((r for r in results_list if r['name'] == m), None)]
        if not models:
            return
        fig, axes = plt.subplots(1, len(models),
                                 figsize=(4 * len(models), 3.5))
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
        fig.suptitle('Confusion Matrices — SIP v4', y=1.02)
        fig.tight_layout()
        self._save(fig, '08_confusion_matrices')


# ══════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════

def main(seed_override=None, results_dir_override=None):
    config = TrainingConfig()
    if seed_override is not None:
        config.seed = seed_override
    if results_dir_override is not None:
        config.results_dir = results_dir_override
        config.plots_dir = os.path.join(results_dir_override, "plots")

    os.makedirs(config.results_dir, exist_ok=True)
    os.makedirs(config.plots_dir, exist_ok=True)

    pipeline = DataPipeline(config)
    train_df, val_df, test_df = pipeline.load_and_prepare()

    X_train, y_train = pipeline.get_features_labels(train_df, fit=True)
    X_val, y_val = pipeline.get_features_labels(val_df)
    X_test, y_test = pipeline.get_features_labels(test_df)

    X_train_comp, y_train_comp = pipeline.get_compromise_features(
        train_df, fit=True)
    X_test_comp, y_test_comp = pipeline.get_compromise_features(test_df)

    X_train_normal, _ = pipeline.get_normal_data(train_df, fit=False)

    attack_types_test = test_df["attack_type"].values
    soph_test = (test_df["attacker_sophistication"].values
                 if "attacker_sophistication" in test_df.columns else None)
    n_feat = X_train.shape[1]

    results_list = []
    trainers_dict = {}

    print("\n" + "=" * 60)
    print("  Training Models — SIP v4 (без ZCAD, transit-ACK aware)")
    print("=" * 60)

    def ev(name, scores, thr):
        return evaluate_model(name, scores, y_test, attack_types_test,
                              threshold=thr, sophistication=soph_test)

    # ── 1. Isolation Forest ──
    print("\n[1] Isolation Forest...")
    iso = IsolationForest(n_estimators=config.iso_forest_trees,
                          contamination=config.iso_forest_contamination,
                          random_state=config.seed)
    iso.fit(X_train_normal)
    iso_sv = -iso.score_samples(X_val)
    iso_st = -iso.score_samples(X_test)
    th_iso, _ = find_threshold(iso_sv, y_val, config.threshold_percentile)
    results_list.append(ev("Isolation Forest", iso_st, th_iso))

    # ── 2. One-Class SVM ──
    print("\n[2] One-Class SVM...")
    svm = OneClassSVM(nu=config.ocsvm_nu, kernel=config.ocsvm_kernel,
                      gamma="scale")
    svm.fit(X_train_normal)
    svm_sv = -svm.score_samples(X_val)
    svm_st = -svm.score_samples(X_test)
    th_svm, _ = find_threshold(svm_sv, y_val, config.threshold_percentile)
    results_list.append(ev("One-Class SVM", svm_st, th_svm))

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
    results_list.append(ev("Autoencoder", ae_st, th_ae))

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
    results_list.append(ev("VAE", vae_st, th_vae))

    # ── 5. LSTM-AE ──
    print("\n[5] LSTM-Autoencoder...")
    train_normal_df = train_df[train_df[pipeline.LABEL_COL] == 0]
    seq_train, _ = pipeline.get_sequential_data(train_normal_df,
                                                 config.lstm_seq_len)
    lstm_done = False
    if len(seq_train) > 100:
        lstm = LSTMAutoencoder(n_feat, config.lstm_hidden, config.lstm_layers,
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
        seq_at_test, seq_soph_test = [], []
        for nid in test_df["node_id"].unique():
            mask = test_df["node_id"].values == nid
            atypes = test_df.loc[mask, "attack_type"].values
            sophs = (test_df.loc[mask, "attacker_sophistication"].values
                     if soph_test is not None
                     else np.array(["none"] * mask.sum()))
            for i in range(len(atypes) - config.lstm_seq_len + 1):
                seq_at_test.append(atypes[i + config.lstm_seq_len - 1])
                seq_soph_test.append(sophs[i + config.lstm_seq_len - 1])
        seq_at_test = np.array(seq_at_test)
        seq_soph_test = np.array(seq_soph_test)

        lstm_st = lstm_trainer.reconstruction_error(seq_test)
        lstm_sv = lstm_trainer.reconstruction_error(seq_val)
        th_lstm, _ = find_threshold(lstm_sv, seq_y_val,
                                    config.threshold_percentile)
        results_list.append(evaluate_model(
            "LSTM-Autoencoder", lstm_st, seq_y_test, seq_at_test,
            threshold=th_lstm, sophistication=seq_soph_test))
        lstm_done = True
    else:
        print("  Skipped: not enough sequences.")

    # ── 6. Temporal-Attention-AE ──
    print("\n[6] Temporal-Attention-AE...")
    if lstm_done and len(seq_train) > 100:
        tat = TemporalAttentionAE(n_feat, 64, 4, 2, config.lstm_seq_len, 0.15)
        tat_trainer = AETrainer(tat, 5e-4, 100, 256, patience=15)
        tat_trainer.fit(seq_train,
                        seq_val_normal if len(seq_val_normal) > 0 else None)
        trainers_dict['TA-AE'] = tat_trainer
        tat_st = tat_trainer.reconstruction_error(seq_test)
        tat_sv = tat_trainer.reconstruction_error(seq_val)
        th_tat, _ = find_threshold(tat_sv, seq_y_val,
                                   config.threshold_percentile)
        results_list.append(evaluate_model(
            "Temporal-Attention-AE", tat_st, seq_y_test, seq_at_test,
            threshold=th_tat, sophistication=seq_soph_test))

    # ── 7. Random Forest ──
    print("\n[7] Random Forest...")
    rf = RandomForestClassifier(n_estimators=config.rf_trees,
                                random_state=config.seed,
                                class_weight="balanced", n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_pt = rf.predict_proba(X_test)[:, 1]
    results_list.append(ev("Random Forest", rf_pt, 0.5))

    importances = rf.feature_importances_
    print("\n  Feature importance (top 20):")
    feat_imp = sorted(zip(pipeline.feature_cols, importances),
                      key=lambda x: x[1], reverse=True)
    for fn, imp in feat_imp[:20]:
        print(f"    {fn:48s} {imp:.4f} ({100*imp:.1f}%)")
    # важность ACK-инварианта отдельно
    ack_imp = sum(imp for fn, imp in feat_imp
                  if fn in ("cu_ack_closure_mismatch",
                            "dev_ack_mismatch_vs_zone"))
    print(f"  ACK-invariant (I2) importance: {ack_imp:.4f} "
          f"({100*ack_imp:.1f}%)")

    # ── 8. Gradient Boosting ──
    print("\n[8] Gradient Boosting...")
    gb = GradientBoostingClassifier(n_estimators=config.gb_estimators,
                                    learning_rate=config.gb_lr,
                                    max_depth=5, random_state=config.seed)
    gb.fit(X_train, y_train)
    gb_pt = gb.predict_proba(X_test)[:, 1]
    results_list.append(ev("Gradient Boosting", gb_pt, 0.5))

    # ── 9. Cascade ──
    print("\n[9] Cascade Compromise Detector...")
    cascade = CascadeCompromiseDetector(config, config.seed)
    cascade.fit(X_train, y_train, X_train_comp, y_train_comp)
    casc_pt = cascade.predict_proba(X_test, X_test_comp)
    results_list.append(ev("Cascade", casc_pt, 0.5))

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
        r = ev(f"Ens:{en}", st, th)
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
        norm_score(casc_pt), norm_score(rf_pt), norm_score(gb_pt),
    ])
    meta_gb = GradientBoostingClassifier(
        n_estimators=200, learning_rate=0.05, max_depth=3,
        random_state=config.seed)
    meta_gb.fit(meta_train, y_train)
    meta_pt = meta_gb.predict_proba(meta_test)[:, 1]
    results_list.append(ev("Meta-Ensemble", meta_pt, 0.5))

    # ══════════════════════════════════════════════════
    #  Ablation Studies (признаки / каскад / ACK-инвариант)
    # ══════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  Ablation Studies")
    print("=" * 60)

    ablation = []
    comp_mask = attack_types_test == "proxy_compromise"
    comp_total = comp_mask.sum()

    rf_r = next(r for r in results_list if r['name'] == 'Random Forest')
    comp_full = (rf_pt[comp_mask] >= 0.5).sum()
    ablation.append({
        'config': f'Full\n({len(pipeline.feature_cols)} feat)',
        'f1': rf_r['f1'],
        'compromise_rate': 100 * comp_full / max(1, comp_total),
    })

    non_ms = [c for c in pipeline.feature_cols if c not in MS_FEATURES]
    non_ms_idx = [pipeline.feature_cols.index(c) for c in non_ms]
    rf_no_ms = RandomForestClassifier(
        n_estimators=config.rf_trees, random_state=config.seed,
        class_weight="balanced", n_jobs=-1)
    rf_no_ms.fit(X_train[:, non_ms_idx], y_train)
    rf_no_ms_p = rf_no_ms.predict_proba(X_test[:, non_ms_idx])[:, 1]
    r_no_ms = ev("RF (no MS)", rf_no_ms_p, 0.5)
    ablation.append({
        'config': f'No MS\n({len(non_ms)} feat)', 'f1': r_no_ms['f1'],
        'compromise_rate': 100 * (rf_no_ms_p[comp_mask] >= 0.5).sum() /
                           max(1, comp_total),
    })

    # НОВОЕ v4: абляция ACK-инварианта — доказательство его вклада
    ack_set = {"cu_ack_closure_mismatch", "dev_ack_mismatch_vs_zone"}
    non_ack = [c for c in pipeline.feature_cols if c not in ack_set]
    non_ack_idx = [pipeline.feature_cols.index(c) for c in non_ack]
    rf_no_ack = RandomForestClassifier(
        n_estimators=config.rf_trees, random_state=config.seed,
        class_weight="balanced", n_jobs=-1)
    rf_no_ack.fit(X_train[:, non_ack_idx], y_train)
    rf_no_ack_p = rf_no_ack.predict_proba(X_test[:, non_ack_idx])[:, 1]
    r_no_ack = ev("RF (no ACK-invariant)", rf_no_ack_p, 0.5)
    ablation.append({
        'config': f'No ACK-inv\n({len(non_ack)} feat)', 'f1': r_no_ack['f1'],
        'compromise_rate': 100 * (rf_no_ack_p[comp_mask] >= 0.5).sum() /
                           max(1, comp_total),
    })

    s1_scores = cascade.predict_proba_stage1(X_test)
    r_s1 = ev("Cascade-S1", s1_scores, 0.5)
    ablation.append({
        'config': 'Cascade\nStage-1 only', 'f1': r_s1['f1'],
        'compromise_rate': 100 * (s1_scores[comp_mask] >= 0.5).sum() /
                           max(1, comp_total),
    })

    casc_r = next(r for r in results_list if r['name'] == 'Cascade')
    ablation.append({
        'config': 'Cascade\nBoth levels', 'f1': casc_r['f1'],
        'compromise_rate': 100 * (casc_pt[comp_mask] >= 0.5).sum() /
                           max(1, comp_total),
    })

    # ══════════════════════════════════════════════════
    #  Save Results
    # ══════════════════════════════════════════════════
    save_cols = ['name', 'auc_roc', 'auc_pr', 'f1', 'fpr',
                 'tp', 'fp', 'fn', 'tn']
    res_df = pd.DataFrame([{k: r[k] for k in save_cols}
                           for r in results_list])
    res_df.to_csv(os.path.join(config.results_dir,
                               "full_comparison_sip.csv"), index=False)

    # seed_metrics.csv — совместим с run_seeds; + колонки по sophistication
    comp_rows = []
    for r in results_list:
        pa = r.get("per_attack", {}).get("proxy_compromise", (0, 1, 0.0))
        ps = r.get("per_soph", {})
        comp_rows.append({
            "name": r["name"],
            "auc_roc": r["auc_roc"], "auc_pr": r["auc_pr"],
            "f1": r["f1"], "fpr": r["fpr"],
            "compromise_recall": pa[2],
            "soph_naive": ps.get("naive", np.nan),
            "soph_statistical": ps.get("statistical", np.nan),
            "soph_adaptive": ps.get("adaptive", np.nan),
        })
    pd.DataFrame(comp_rows).to_csv(
        os.path.join(config.results_dir, "seed_metrics.csv"), index=False)

    pd.DataFrame(ablation).to_csv(
        os.path.join(config.results_dir, "ablation.csv"), index=False)

    with open(os.path.join(config.results_dir,
                           "feature_columns_sip.json"), "w") as f:
        json.dump({
            "observable_features": pipeline.feature_cols,
            "compromise_features": pipeline.compromise_feature_cols,
            "ms_features": [c for c in MS_FEATURES
                            if c in pipeline.feature_cols],
            "ack_invariant_features": [
                c for c in ("cu_ack_closure_mismatch",
                            "dev_ack_mismatch_vs_zone")
                if c in pipeline.feature_cols],
            "excluded_leaky_features": sorted(LEAKY_FEATURES),
            "feature_groups": {k: [c for c in v
                                   if c in pipeline.feature_cols]
                               for k, v in FEATURE_GROUPS.items()},
        }, f, indent=2)

    # Summary
    print("\n" + "=" * 70)
    print("  SUMMARY — SIP v4 (без ZCAD)")
    print("=" * 70)
    print(f"{'Model':<28s} {'AUC-ROC':>8s} {'AUC-PR':>8s} {'F1':>8s} "
          f"{'FPR':>8s} {'naive':>6s} {'stat':>6s} {'adapt':>6s}")
    print("-" * 82)
    for r in results_list:
        ps = r.get("per_soph", {})
        print(f"{r['name']:<28s} {r['auc_roc']:8.4f} {r['auc_pr']:8.4f} "
              f"{r['f1']:8.4f} {r['fpr']:8.4f} "
              f"{ps.get('naive', float('nan')):6.1f} "
              f"{ps.get('statistical', float('nan')):6.1f} "
              f"{ps.get('adaptive', float('nan')):6.1f}")

    print("\n  Feature-ablation (compromise recall):")
    for a in ablation:
        print(f"    {a['config'].replace(chr(10), ' '):26s}  "
              f"F1={a['f1']:.4f}  compromise={a['compromise_rate']:.1f}%")

    # ══════════════════════════════════════════════════
    #  Plots
    # ══════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  Generating plots...")
    print("=" * 60)

    viz = ResultsVisualizer(config.plots_dir)
    main_models = [
        'Isolation Forest', 'One-Class SVM', 'Autoencoder', 'VAE',
        'Random Forest', 'Gradient Boosting', 'Cascade', 'Meta-Ensemble',
    ]
    viz.plot_roc_curves(results_list, y_test, main_models)
    viz.plot_pr_curves(results_list, y_test, main_models)
    viz.plot_summary_bars(results_list, main_models)
    viz.plot_soph_breakdown(results_list, main_models)
    viz.plot_feature_importance(pipeline.feature_cols, importances,
                                MS_FEATURES)
    viz.plot_confusion_matrices(
        results_list, y_test,
        ['VAE', 'Random Forest', 'Cascade', 'Meta-Ensemble'])

    # ── Артефакты для plot_results_print_sip ──
    import pickle
    trainers_history = {}
    for name, trainer in trainers_dict.items():
        trainers_history[name] = {
            "train_losses": list(trainer.train_losses),
            "val_losses": list(trainer.val_losses),
        }
    artifacts = {
        "results_list": results_list,
        "ens_results": ens_results,
        "ablation_results": ablation,
        "zc_ablation": [],  # ZCAD отсутствует в v4-пайплайне
        "y_test": y_test,
        "attack_types_test": attack_types_test,
        "importances": importances,
        "feature_cols": pipeline.feature_cols,
        "cascade_s1": cascade.predict_proba_stage1(X_test),
        "cascade_s2": cascade.predict_proba_stage2(X_test_comp),
        "cascade_comb": casc_pt,
        "trainers_history": trainers_history,
    }
    with open(os.path.join(config.results_dir, "plot_artifacts.pkl"),
              "wb") as f:
        pickle.dump(artifacts, f)

    print(f"\n  All results saved to {config.results_dir}/")


import sys
if __name__ == "__main__":
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 42
    rdir = sys.argv[2] if len(sys.argv) > 2 else "results_sip_v4"
    main(seed_override=seed, results_dir_override=rdir)
