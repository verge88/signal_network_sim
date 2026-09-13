# """
# SS7 Master-Slave Anomaly Detection Training Pipeline v6
# ========================================================
# Улучшения для компрометации слейва:
#   1. Расширенный набор признаков (temporal, cross-zone, consistency)
#   2. Двухуровневый каскадный классификатор: общий детектор + специализированный
#      детектор компрометации
#   3. Focal Loss для автоэнкодера (больший вес на "трудные" примеры)
#   4. Variational Autoencoder (VAE) — лучше моделирует распределение нормального
#      трафика, аномалии получают высокий KL-divergence
#   5. Временно́й attention-автоэнкодер для захвата паттернов стабильности
#   6. Оптимизация порогов отдельно для каждого типа атаки
# """

# import numpy as np
# import pandas as pd
# import os
# import json
# import warnings
# from dataclasses import dataclass, field
# from typing import List, Dict, Tuple, Optional

# from sklearn.ensemble import IsolationForest, RandomForestClassifier, GradientBoostingClassifier
# from sklearn.svm import OneClassSVM
# from sklearn.preprocessing import StandardScaler, LabelEncoder
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import (
#     roc_auc_score, average_precision_score, f1_score,
#     confusion_matrix, classification_report, precision_recall_curve
# )
# from sklearn.impute import SimpleImputer

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.utils.data import DataLoader, TensorDataset

# warnings.filterwarnings("ignore")

# # ──────────────────────────────────────────────────────────
# #  Configuration
# # ──────────────────────────────────────────────────────────

# @dataclass
# class TrainingConfig:
#     data_path: str = "ss7_dataset_v6.csv"
#     results_dir: str = "results_v6"
#     test_size: float = 0.25
#     val_size: float = 0.15
#     seed: int = 42

#     iso_forest_trees: int = 300
#     iso_forest_contamination: float = 0.05
#     ocsvm_nu: float = 0.05
#     ocsvm_kernel: str = "rbf"

#     ae_hidden_dims: List[int] = field(default_factory=lambda: [64, 32])
#     ae_latent_dim: int = 12
#     ae_epochs: int = 100
#     ae_batch: int = 256
#     ae_lr: float = 1e-3

#     # v6: VAE параметры
#     vae_hidden_dims: List[int] = field(default_factory=lambda: [64, 32])
#     vae_latent_dim: int = 12
#     vae_epochs: int = 120
#     vae_batch: int = 256
#     vae_lr: float = 1e-3
#     vae_kl_weight: float = 0.5

#     lstm_hidden: int = 48
#     lstm_layers: int = 2
#     lstm_seq_len: int = 10
#     lstm_epochs: int = 80
#     lstm_batch: int = 256
#     lstm_lr: float = 1e-3

#     rf_trees: int = 300
#     gb_estimators: int = 300
#     gb_lr: float = 0.05

#     # v6: Каскадный классификатор
#     cascade_rf_trees: int = 500
#     cascade_gb_estimators: int = 500

#     threshold_percentile: float = 95.0

# # ──────────────────────────────────────────────────────────
# #  Data Pipeline v6
# # ──────────────────────────────────────────────────────────

# class DataPipeline:
#     """v6: Расширенный набор признаков с temporal/cross-zone/consistency."""

#     # v6: Расширенные наблюдаемые признаки
#     OBSERVABLE_FEATURE_COLS = [
#         # CU-related (расширено)
#         "cu_delivered", "cu_rtt_delay", "cu_req_delay", "cu_resp_delay",
#         "cu_combined_loss_prob", "integrity_check", "consecutive_cu_losses",
#         "cu_processing_delay", "cu_response_time_jitter",  # v6 новые
#         "cu_staleness", "cu_consistency_error",  # v6 новые
#         # Reported by slave
#         "obs_total_messages", "obs_map_count", "obs_isup_count", "obs_tcap_count",
#         "obs_entropy", "obs_inbound_outbound_ratio",
#         "obs_international_fraction", "obs_n_unique_destinations",
#         "obs_map_dominance_ratio", "obs_map_top2_ratio", "obs_map_gini",
#         # MAP subtype ratios
#         "obs_ratio_map_sri", "obs_ratio_map_sri_sm", "obs_ratio_map_psi",
#         "obs_ratio_map_ati", "obs_ratio_map_update_loc",
#         "obs_ratio_map_insert_sub", "obs_ratio_map_send_auth",
#         "obs_ratio_map_other_map",
#         # Z-scores
#         "z_obs_total_messages", "z_obs_map_count",
#         "z_obs_map_dominance_ratio", "z_obs_ratio_map_sri_sm",
#         "z_obs_ratio_map_psi",
#         # Zone deviations (расширено)
#         "dev_total_messages_vs_zone_median",
#         "dev_map_count_vs_zone_median",
#         "dev_isup_count_vs_zone_median",
#         "zone_io_ratio_rank",  # v6 новый
#         "dev_io_ratio_vs_zone_median",  # v6 новый
#         "dev_integrity_fail_rate_vs_zone",  # v6 новый
#         "dev_var_total_vs_zone",  # v6 новый
#         # v6: Temporal features (ключевые для компрометации)
#         "var_obs_total_messages",
#         "var_obs_entropy",
#         "var_obs_inbound_outbound_ratio",
#         "var_obs_map_dominance_ratio",
#         "autocorr_obs_total_messages",
#         "integrity_fail_rate",
#         "obs_destinations_cv",
#     ]

#     GROUND_TRUTH_COLS = [
#         "gt_total_messages", "gt_map_count", "gt_isup_count",
#         "gt_tcap_count", "gt_entropy", "gt_inbound_outbound_ratio",
#         "gt_international_fraction", "gt_n_unique_destinations",
#     ]

#     LABEL_COL = "is_anomaly"
#     ATTACK_COL = "attack_type"

#     # v6: Признаки, наиболее информативные для компрометации
#     COMPROMISE_FEATURES = [
#         "obs_inbound_outbound_ratio", "integrity_check",
#         "integrity_fail_rate", "cu_consistency_error",
#         "cu_processing_delay", "cu_response_time_jitter",
#         "var_obs_total_messages", "var_obs_entropy",
#         "var_obs_inbound_outbound_ratio", "var_obs_map_dominance_ratio",
#         "autocorr_obs_total_messages", "obs_destinations_cv",
#         "zone_io_ratio_rank", "dev_io_ratio_vs_zone_median",
#         "dev_integrity_fail_rate_vs_zone", "dev_var_total_vs_zone",
#         "consecutive_cu_losses", "cu_staleness",
#         "dev_total_messages_vs_zone_median",
#         "dev_map_count_vs_zone_median",
#     ]

#     def __init__(self, config: TrainingConfig):
#         self.config = config
#         self.scaler = StandardScaler()
#         self.imputer = SimpleImputer(strategy="median")
#         self.le_node_type = LabelEncoder()
#         self.feature_cols: List[str] = []
#         # v6: Отдельный scaler для compromise-детектора
#         self.compromise_scaler = StandardScaler()
#         self.compromise_imputer = SimpleImputer(strategy="median")
#         self.compromise_feature_cols: List[str] = []

#     def load_and_prepare(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
#         df = pd.read_csv(self.config.data_path)
#         print(f"Loaded {len(df)} records, anomalies: {df[self.LABEL_COL].sum()} "
#               f"({100*df[self.LABEL_COL].mean():.2f}%)")
#         print(f"Attack distribution:\n{df[self.ATTACK_COL].value_counts()}\n")

#         if "node_type" in df.columns:
#             df["node_type_enc"] = self.le_node_type.fit_transform(df["node_type"])
#         else:
#             df["node_type_enc"] = 0

#         # Build feature columns
#         self.feature_cols = []
#         for col in self.OBSERVABLE_FEATURE_COLS:
#             if col in df.columns:
#                 self.feature_cols.append(col)
#         self.feature_cols.append("node_type_enc")

#         # v6: Build compromise-specific feature columns
#         self.compromise_feature_cols = []
#         for col in self.COMPROMISE_FEATURES:
#             if col in df.columns:
#                 self.compromise_feature_cols.append(col)
#         self.compromise_feature_cols.append("node_type_enc")

#         # Verify NO ground-truth leakage
#         for col in self.feature_cols:
#             if col.startswith("gt_"):
#                 raise ValueError(f"Ground-truth column '{col}' found in feature set!")

#         print(f"Observable features: {len(self.feature_cols)}")
#         print(f"Compromise-specific features: {len(self.compromise_feature_cols)}")
#         print(f"Ground-truth columns (excluded): {len([c for c in self.GROUND_TRUTH_COLS if c in df.columns])}")

#         train_val, test = train_test_split(
#             df, test_size=self.config.test_size,
#             stratify=df[self.LABEL_COL], random_state=self.config.seed
#         )
#         train, val = train_test_split(
#             train_val, test_size=self.config.val_size / (1 - self.config.test_size),
#             stratify=train_val[self.LABEL_COL], random_state=self.config.seed
#         )

#         print(f"Split — Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
#         print(f"Train anomaly rate: {train[self.LABEL_COL].mean():.4f}")
#         print(f"Val anomaly rate:   {val[self.LABEL_COL].mean():.4f}")
#         print(f"Test anomaly rate:  {test[self.LABEL_COL].mean():.4f}\n")

#         return train, val, test

#     def get_features_labels(self, df: pd.DataFrame, fit: bool = False):
#         X = df[self.feature_cols].copy()
#         if fit:
#             X_imp = self.imputer.fit_transform(X)
#             X_scaled = self.scaler.fit_transform(X_imp)
#         else:
#             X_imp = self.imputer.transform(X)
#             X_scaled = self.scaler.transform(X_imp)
#         y = df[self.LABEL_COL].values
#         return X_scaled, y

#     def get_compromise_features(self, df: pd.DataFrame, fit: bool = False):
#         """v6: Extract compromise-specific features."""
#         X = df[self.compromise_feature_cols].copy()
#         if fit:
#             X_imp = self.compromise_imputer.fit_transform(X)
#             X_scaled = self.compromise_scaler.fit_transform(X_imp)
#         else:
#             X_imp = self.compromise_imputer.transform(X)
#             X_scaled = self.compromise_scaler.transform(X_imp)
#         # v6: Бинарная метка — именно компрометация
#         y = (df[self.ATTACK_COL] == "slave_compromise").astype(int).values
#         return X_scaled, y

#     def get_normal_data(self, df: pd.DataFrame, fit: bool = False):
#         normal = df[df[self.LABEL_COL] == 0]
#         return self.get_features_labels(normal, fit=fit)

#     def get_sequential_data(self, df: pd.DataFrame, seq_len: int):
#         X_all, y_all = self.get_features_labels(df, fit=False)
#         sequences = []
#         labels = []
#         node_ids = df["node_id"].unique()
#         for nid in node_ids:
#             mask = df["node_id"].values == nid
#             X_node = X_all[mask]
#             y_node = y_all[mask]
#             for i in range(len(X_node) - seq_len + 1):
#                 sequences.append(X_node[i:i + seq_len])
#                 labels.append(y_node[i + seq_len - 1])
#         return np.array(sequences), np.array(labels)


# # ──────────────────────────────────────────────────────────
# #  Neural network models
# # ──────────────────────────────────────────────────────────

# class Autoencoder(nn.Module):
#     def __init__(self, input_dim: int, hidden_dims: List[int], latent_dim: int):
#         super().__init__()
#         enc_layers = []
#         prev = input_dim
#         for h in hidden_dims:
#             enc_layers.extend([nn.Linear(prev, h), nn.ReLU(), nn.BatchNorm1d(h)])
#             prev = h
#         enc_layers.append(nn.Linear(prev, latent_dim))
#         self.encoder = nn.Sequential(*enc_layers)

#         dec_layers = []
#         prev = latent_dim
#         for h in reversed(hidden_dims):
#             dec_layers.extend([nn.Linear(prev, h), nn.ReLU(), nn.BatchNorm1d(h)])
#             prev = h
#         dec_layers.append(nn.Linear(prev, input_dim))
#         self.decoder = nn.Sequential(*dec_layers)

#     def forward(self, x):
#         z = self.encoder(x)
#         x_hat = self.decoder(z)
#         return x_hat


# # ─── v6: Variational Autoencoder ───

# class VAE(nn.Module):
#     """
#     Variational Autoencoder: моделирует распределение нормального трафика.
#     Аномалии получают высокий reconstruction error + высокий KL-divergence,
#     что даёт более чувствительный скор для «тихих» аномалий типа компрометации.
#     """
#     def __init__(self, input_dim: int, hidden_dims: List[int], latent_dim: int):
#         super().__init__()
#         # Encoder
#         enc_layers = []
#         prev = input_dim
#         for h in hidden_dims:
#             enc_layers.extend([nn.Linear(prev, h), nn.ReLU(), nn.BatchNorm1d(h)])
#             prev = h
#         self.encoder = nn.Sequential(*enc_layers)
#         self.fc_mu = nn.Linear(prev, latent_dim)
#         self.fc_logvar = nn.Linear(prev, latent_dim)

#         # Decoder
#         dec_layers = []
#         prev = latent_dim
#         for h in reversed(hidden_dims):
#             dec_layers.extend([nn.Linear(prev, h), nn.ReLU(), nn.BatchNorm1d(h)])
#             prev = h
#         dec_layers.append(nn.Linear(prev, input_dim))
#         self.decoder = nn.Sequential(*dec_layers)

#     def encode(self, x):
#         h = self.encoder(x)
#         return self.fc_mu(h), self.fc_logvar(h)

#     def reparameterize(self, mu, logvar):
#         std = torch.exp(0.5 * logvar)
#         eps = torch.randn_like(std)
#         return mu + eps * std

#     def decode(self, z):
#         return self.decoder(z)

#     def forward(self, x):
#         mu, logvar = self.encode(x)
#         z = self.reparameterize(mu, logvar)
#         x_hat = self.decode(z)
#         return x_hat, mu, logvar


# class LSTMAutoencoder(nn.Module):
#     def __init__(self, input_dim: int, hidden_dim: int, n_layers: int, seq_len: int,
#                  dropout: float = 0.2):  # v6: добавлен dropout
#         super().__init__()
#         self.seq_len = seq_len
#         self.hidden_dim = hidden_dim
#         self.n_layers = n_layers

#         self.encoder = nn.LSTM(input_dim, hidden_dim, n_layers,
#                                batch_first=True, dropout=dropout if n_layers > 1 else 0)
#         self.decoder = nn.LSTM(hidden_dim, hidden_dim, n_layers,
#                                batch_first=True, dropout=dropout if n_layers > 1 else 0)
#         self.output_layer = nn.Linear(hidden_dim, input_dim)

#     def forward(self, x):
#         _, (h, c) = self.encoder(x)
#         dec_input = h[-1].unsqueeze(1).repeat(1, self.seq_len, 1)
#         dec_out, _ = self.decoder(dec_input, (h, c))
#         out = self.output_layer(dec_out)
#         return out


# # ─── v6: Temporal Attention Autoencoder ───

# class TemporalAttentionAE(nn.Module):
#     """
#     Автоэнкодер с self-attention по временно́й оси.
#     Захватывает паттерны «аномальной стабильности» компрометированных узлов:
#     если последовательность слишком однородна (маскированные значения),
#     attention-механизм это распознаёт.
#     """
#     def __init__(self, input_dim: int, hidden_dim: int, n_heads: int = 4,
#                  n_layers: int = 2, seq_len: int = 10, dropout: float = 0.1):
#         super().__init__()
#         self.seq_len = seq_len
#         self.input_proj = nn.Linear(input_dim, hidden_dim)

#         encoder_layer = nn.TransformerEncoderLayer(
#             d_model=hidden_dim, nhead=n_heads,
#             dim_feedforward=hidden_dim * 2,
#             dropout=dropout, batch_first=True
#         )
#         self.transformer_encoder = nn.TransformerEncoder(encoder_layer, n_layers)
#         self.output_proj = nn.Linear(hidden_dim, input_dim)

#     def forward(self, x):
#         # x: (batch, seq_len, input_dim)
#         h = self.input_proj(x)
#         h = self.transformer_encoder(h)
#         out = self.output_proj(h)
#         return out


# class AETrainer:
#     def __init__(self, model, lr, epochs, batch_size, patience=15, device=None):
#         self.model = model
#         self.lr = lr
#         self.epochs = epochs
#         self.batch_size = batch_size
#         self.patience = patience
#         self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
#         self.model.to(self.device)

#     def fit(self, X_train, X_val=None):
#         optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr,
#                                      weight_decay=1e-5)  # v6: weight decay
#         scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.epochs)
#         criterion = nn.MSELoss()

#         train_ds = TensorDataset(torch.FloatTensor(X_train))
#         train_dl = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)

#         best_val_loss = float("inf")
#         patience_count = 0
#         best_state = None

#         for epoch in range(self.epochs):
#             self.model.train()
#             train_loss = 0.0
#             for (batch,) in train_dl:
#                 batch = batch.to(self.device)
#                 x_hat = self.model(batch)
#                 loss = criterion(x_hat, batch)
#                 optimizer.zero_grad()
#                 loss.backward()
#                 optimizer.step()
#                 train_loss += loss.item() * len(batch)
#             train_loss /= len(X_train)
#             scheduler.step()

#             if X_val is not None:
#                 val_loss = self._evaluate(X_val, criterion)
#                 if val_loss < best_val_loss:
#                     best_val_loss = val_loss
#                     patience_count = 0
#                     best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
#                 else:
#                     patience_count += 1

#                 if (epoch + 1) % 20 == 0:
#                     print(f"  Epoch {epoch+1}/{self.epochs}: "
#                           f"train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")

#                 if patience_count >= self.patience:
#                     print(f"  Early stopping at epoch {epoch+1}")
#                     break

#         if best_state is not None:
#             self.model.load_state_dict(best_state)
#             self.model.to(self.device)

#     def _evaluate(self, X, criterion):
#         self.model.eval()
#         with torch.no_grad():
#             X_t = torch.FloatTensor(X).to(self.device)
#             x_hat = self.model(X_t)
#             loss = criterion(x_hat, X_t)
#         return loss.item()

#     def reconstruction_error(self, X):
#         self.model.eval()
#         with torch.no_grad():
#             X_t = torch.FloatTensor(X).to(self.device)
#             x_hat = self.model(X_t)
#             errors = torch.mean((X_t - x_hat) ** 2, dim=-1)
#             if errors.dim() > 1:
#                 errors = errors.mean(dim=-1)
#         return errors.cpu().numpy()


# class VAETrainer:
#     """v6: Специализированный тренер для VAE с KL-loss."""

#     def __init__(self, model, lr, epochs, batch_size, kl_weight=0.5,
#                  patience=15, device=None):
#         self.model = model
#         self.lr = lr
#         self.epochs = epochs
#         self.batch_size = batch_size
#         self.kl_weight = kl_weight
#         self.patience = patience
#         self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
#         self.model.to(self.device)

#     def _vae_loss(self, x, x_hat, mu, logvar):
#         recon = F.mse_loss(x_hat, x, reduction='mean')
#         kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
#         return recon + self.kl_weight * kl, recon, kl

#     def fit(self, X_train, X_val=None):
#         optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr,
#                                      weight_decay=1e-5)
#         scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.epochs)

#         train_ds = TensorDataset(torch.FloatTensor(X_train))
#         train_dl = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)

#         best_val_loss = float("inf")
#         patience_count = 0
#         best_state = None

#         for epoch in range(self.epochs):
#             self.model.train()
#             train_loss = 0.0
#             for (batch,) in train_dl:
#                 batch = batch.to(self.device)
#                 x_hat, mu, logvar = self.model(batch)
#                 loss, _, _ = self._vae_loss(batch, x_hat, mu, logvar)
#                 optimizer.zero_grad()
#                 loss.backward()
#                 optimizer.step()
#                 train_loss += loss.item() * len(batch)
#             train_loss /= len(X_train)
#             scheduler.step()

#             if X_val is not None:
#                 val_loss = self._evaluate_val(X_val)
#                 if val_loss < best_val_loss:
#                     best_val_loss = val_loss
#                     patience_count = 0
#                     best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
#                 else:
#                     patience_count += 1

#                 if (epoch + 1) % 20 == 0:
#                     print(f"  Epoch {epoch+1}/{self.epochs}: "
#                           f"train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")

#                 if patience_count >= self.patience:
#                     print(f"  Early stopping at epoch {epoch+1}")
#                     break

#         if best_state is not None:
#             self.model.load_state_dict(best_state)
#             self.model.to(self.device)

#     def _evaluate_val(self, X):
#         self.model.eval()
#         with torch.no_grad():
#             X_t = torch.FloatTensor(X).to(self.device)
#             x_hat, mu, logvar = self.model(X_t)
#             loss, _, _ = self._vae_loss(X_t, x_hat, mu, logvar)
#         return loss.item()

#     def anomaly_score(self, X):
#         """
#         v6: Anomaly score = reconstruction_error + kl_weight * KL-divergence per sample.
#         KL-divergence is especially informative for compromise: the compromised node's
#         masked values fall in a different region of latent space.
#         """
#         self.model.eval()
#         with torch.no_grad():
#             X_t = torch.FloatTensor(X).to(self.device)
#             x_hat, mu, logvar = self.model(X_t)
#             # Per-sample reconstruction error
#             recon = torch.mean((X_t - x_hat) ** 2, dim=-1)
#             # Per-sample KL divergence
#             kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
#             scores = recon + self.kl_weight * kl
#         return scores.cpu().numpy()


# # ──────────────────────────────────────────────────────────
# #  Evaluation (без изменений)
# # ──────────────────────────────────────────────────────────

# def find_threshold(scores, y_true, percentile=95.0):
#     normal_scores = scores[y_true == 0]
#     th_pct = np.percentile(normal_scores, percentile)

#     precision, recall, thresholds = precision_recall_curve(y_true, scores)
#     f1s = 2 * precision * recall / (precision + recall + 1e-9)
#     best_idx = np.argmax(f1s)
#     th_f1 = thresholds[min(best_idx, len(thresholds) - 1)]

#     y_pct = (scores >= th_pct).astype(int)
#     y_f1 = (scores >= th_f1).astype(int)
#     f1_pct = f1_score(y_true, y_pct, zero_division=0)
#     f1_f1 = f1_score(y_true, y_f1, zero_division=0)

#     if f1_f1 >= f1_pct:
#         return th_f1, "max_f1"
#     else:
#         return th_pct, "percentile"


# def evaluate_model(name, scores, y_true, attack_types, threshold=None, percentile=95.0):
#     if threshold is None:
#         threshold, method = find_threshold(scores, y_true, percentile)
#     else:
#         method = "provided"

#     y_pred = (scores >= threshold).astype(int)

#     auc_roc = roc_auc_score(y_true, scores) if len(np.unique(y_true)) > 1 else 0.0
#     auc_pr = average_precision_score(y_true, scores) if len(np.unique(y_true)) > 1 else 0.0
#     f1 = f1_score(y_true, y_pred, zero_division=0)
#     tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
#     fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

#     print(f"\n{'='*60}")
#     print(f"  {name}")
#     print(f"{'='*60}")
#     print(f"  AUC-ROC: {auc_roc:.4f}  |  AUC-PR: {auc_pr:.4f}  |  F1: {f1:.4f}")
#     print(f"  FPR: {fpr:.4f}  |  TP: {tp}  FP: {fp}  FN: {fn}  TN: {tn}")
#     print(f"  Threshold: {threshold:.4f} ({method})")

#     print(f"  Detection by attack type:")
#     for atype in sorted(set(attack_types)):
#         if atype == "none":
#             continue
#         mask = attack_types == atype
#         if mask.sum() > 0:
#             detected = y_pred[mask].sum()
#             total = mask.sum()
#             print(f"    {atype:25s}: {detected}/{total} = {100*detected/total:.1f}%")

#     return {
#         "name": name, "auc_roc": auc_roc, "auc_pr": auc_pr, "f1": f1,
#         "fpr": fpr, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
#         "threshold": threshold, "method": method
#     }


# # ──────────────────────────────────────────────────────────
# #  Specialized Ensemble v6
# # ──────────────────────────────────────────────────────────

# class CompactAE(nn.Module):
#     def __init__(self, input_dim, bottleneck):
#         super().__init__()
#         mid = max(bottleneck + 1, (input_dim + bottleneck) // 2)
#         self.encoder = nn.Sequential(
#             nn.Linear(input_dim, mid), nn.ReLU(),
#             nn.Linear(mid, bottleneck)
#         )
#         self.decoder = nn.Sequential(
#             nn.Linear(bottleneck, mid), nn.ReLU(),
#             nn.Linear(mid, input_dim)
#         )

#     def forward(self, x):
#         return self.decoder(self.encoder(x))


# class SpecializedDetector:
#     def __init__(self, name: str, feature_indices: List[int], seed: int = 42):
#         self.name = name
#         self.feature_indices = feature_indices
#         self.seed = seed
#         self.models = {}

#     def fit(self, X_normal):
#         X = X_normal[:, self.feature_indices]
#         n_feat = X.shape[1]

#         iso = IsolationForest(n_estimators=200, contamination=0.05,
#                               random_state=self.seed)
#         iso.fit(X)
#         self.models["iso"] = iso

#         svm = OneClassSVM(nu=0.05, kernel="rbf", gamma="scale")
#         svm.fit(X)
#         self.models["svm"] = svm

#         bottleneck = max(2, n_feat // 3)
#         ae = CompactAE(n_feat, bottleneck)
#         device = "cuda" if torch.cuda.is_available() else "cpu"
#         trainer = AETrainer(ae, lr=1e-3, epochs=120, batch_size=256,
#                             patience=20, device=device)
#         trainer.fit(X)
#         self.models["ae"] = ae
#         self.models["ae_trainer"] = trainer

#     def score(self, X):
#         X_sub = X[:, self.feature_indices]

#         iso_scores = -self.models["iso"].score_samples(X_sub)
#         svm_scores = -self.models["svm"].score_samples(X_sub)
#         ae_scores = self.models["ae_trainer"].reconstruction_error(X_sub)

#         def norm(s):
#             p1, p99 = np.percentile(s, 1), np.percentile(s, 99)
#             if p99 - p1 < 1e-9:
#                 return np.zeros_like(s)
#             return np.clip((s - p1) / (p99 - p1), 0, 1)

#         scores = np.column_stack([norm(iso_scores), norm(svm_scores), norm(ae_scores)])
#         return np.max(scores, axis=1)


# class EnsembleDetector:
#     def __init__(self, feature_groups: Dict[str, List[str]],
#                  all_feature_cols: List[str], seed: int = 42):
#         self.detectors: Dict[str, SpecializedDetector] = {}
#         for gname, cols in feature_groups.items():
#             indices = [all_feature_cols.index(c) for c in cols if c in all_feature_cols]
#             if indices:
#                 self.detectors[gname] = SpecializedDetector(gname, indices, seed)

#     def fit(self, X_normal):
#         for name, det in self.detectors.items():
#             print(f"  Fitting specialized detector: {name} ({len(det.feature_indices)} features)")
#             det.fit(X_normal)

#     def score(self, X) -> Dict[str, np.ndarray]:
#         results = {}
#         for name, det in self.detectors.items():
#             results[name] = det.score(X)

#         all_scores = np.column_stack(list(results.values()))
#         results["ensemble_max"] = np.max(all_scores, axis=1)
#         results["ensemble_mean"] = np.mean(all_scores, axis=1)
#         return results


# # ──────────────────────────────────────────────────────────
# #  Feature Groups v6 (расширены)
# # ──────────────────────────────────────────────────────────

# FEATURE_GROUPS = {
#     "map_profile": [
#         "obs_ratio_map_sri", "obs_ratio_map_sri_sm", "obs_ratio_map_psi",
#         "obs_ratio_map_ati", "obs_ratio_map_update_loc",
#         "obs_ratio_map_insert_sub", "obs_ratio_map_send_auth",
#         "obs_ratio_map_other_map",
#         "obs_map_dominance_ratio", "obs_map_top2_ratio", "obs_map_gini",
#         "z_obs_map_dominance_ratio", "z_obs_ratio_map_sri_sm",
#         "z_obs_ratio_map_psi",
#     ],
#     "volume_timing": [
#         "obs_total_messages", "obs_map_count", "obs_isup_count",
#         "obs_tcap_count", "obs_entropy",
#         "z_obs_total_messages", "z_obs_map_count",
#     ],
#     "international": [
#         "obs_international_fraction", "obs_n_unique_destinations",
#     ],
#     "control_unit": [
#         "cu_delivered", "cu_rtt_delay", "cu_req_delay", "cu_resp_delay",
#         "cu_combined_loss_prob", "integrity_check", "consecutive_cu_losses",
#         "cu_processing_delay", "cu_response_time_jitter",  # v6
#         "cu_staleness", "cu_consistency_error",  # v6
#     ],
#     "baseline_deviation": [
#         "dev_total_messages_vs_zone_median",
#         "dev_map_count_vs_zone_median",
#         "dev_isup_count_vs_zone_median",
#         "obs_inbound_outbound_ratio",
#         "zone_io_ratio_rank",  # v6
#         "dev_io_ratio_vs_zone_median",  # v6
#     ],
#     # v6: НОВАЯ ГРУППА — специально для компрометации
#     "compromise_indicators": [
#         "obs_inbound_outbound_ratio",
#         "integrity_fail_rate",
#         "var_obs_total_messages",
#         "var_obs_entropy",
#         "var_obs_inbound_outbound_ratio",
#         "var_obs_map_dominance_ratio",
#         "autocorr_obs_total_messages",
#         "obs_destinations_cv",
#         "cu_consistency_error",
#         "cu_processing_delay",
#         "cu_response_time_jitter",
#         "zone_io_ratio_rank",
#         "dev_io_ratio_vs_zone_median",
#         "dev_integrity_fail_rate_vs_zone",
#         "dev_var_total_vs_zone",
#     ],
# }

# MS_FEATURES = [
#     "cu_delivered", "cu_rtt_delay", "cu_req_delay", "cu_resp_delay",
#     "cu_combined_loss_prob", "integrity_check", "consecutive_cu_losses",
#     "cu_processing_delay", "cu_response_time_jitter",  # v6
#     "cu_staleness", "cu_consistency_error",  # v6
#     "obs_total_messages", "obs_map_count", "obs_isup_count", "obs_tcap_count",
#     "obs_inbound_outbound_ratio",
#     "dev_total_messages_vs_zone_median",
#     "dev_map_count_vs_zone_median",
#     "dev_isup_count_vs_zone_median",
#     "zone_io_ratio_rank",  # v6
#     "dev_io_ratio_vs_zone_median",  # v6
#     "dev_integrity_fail_rate_vs_zone",  # v6
#     "dev_var_total_vs_zone",  # v6
#     # v6: temporal features are also MS-specific
#     "var_obs_total_messages",
#     "var_obs_entropy",
#     "var_obs_inbound_outbound_ratio",
#     "var_obs_map_dominance_ratio",
#     "autocorr_obs_total_messages",
#     "integrity_fail_rate",
#     "obs_destinations_cv",
# ]


# # ──────────────────────────────────────────────────────────
# #  v6: Cascade Classifier for Compromise Detection
# # ──────────────────────────────────────────────────────────

# class CascadeCompromiseDetector:
#     """
#     Двухуровневый каскадный детектор:
    
#     Уровень 1: Общий детектор аномалий (RF/GB на полном наборе признаков)
#                → отфильтровывает очевидные аномалии (SMS, DoS, IRSF, tracking)
    
#     Уровень 2: Специализированный детектор компрометации на записях,
#                НЕ обнаруженных уровнем 1 как явная аномалия.
#                Использует ТОЛЬКО compromise-specific features.
#                Обучен на дисбалансированной выборке с SMOTE-подобной стратегией
#                (class_weight='balanced_subsample').
    
#     Финальное решение: OR(уровень1, уровень2)
#     """

#     def __init__(self, config: TrainingConfig, seed: int = 42):
#         self.config = config
#         self.seed = seed
#         # Уровень 1: общий RF
#         self.stage1_rf = RandomForestClassifier(
#             n_estimators=config.cascade_rf_trees,
#             random_state=seed,
#             class_weight="balanced",
#             n_jobs=-1
#         )
#         # Уровень 2: специализированный GB для компрометации
#         self.stage2_gb = GradientBoostingClassifier(
#             n_estimators=config.cascade_gb_estimators,
#             learning_rate=0.03,
#             max_depth=4,
#             min_samples_leaf=20,
#             random_state=seed
#         )
#         # Уровень 2 альтернатива: RF с балансировкой
#         self.stage2_rf = RandomForestClassifier(
#             n_estimators=config.cascade_rf_trees,
#             random_state=seed,
#             class_weight="balanced_subsample",
#             max_depth=8,
#             min_samples_leaf=10,
#             n_jobs=-1
#         )

#     def fit(self, X_train_full, y_train, X_train_compromise, y_train_compromise):
#         """
#         X_train_full: полный набор признаков, y_train: бинарная метка аномалии
#         X_train_compromise: compromise-specific признаки
#         y_train_compromise: бинарная метка именно компрометации
#         """
#         print("  Cascade Stage 1: General anomaly detector (RF)...")
#         self.stage1_rf.fit(X_train_full, y_train)

#         print("  Cascade Stage 2: Compromise-specific detector (GB + RF)...")
#         # Обучаем на ВСЕХ данных, но с меткой именно компрометации
#         self.stage2_gb.fit(X_train_compromise, y_train_compromise)
#         self.stage2_rf.fit(X_train_compromise, y_train_compromise)

#     def predict_proba(self, X_full, X_compromise):
#         """
#         Возвращает combined probability:
#         P(anomaly) = max(P_stage1(anomaly), P_stage2(compromise))
#         """
#         p1 = self.stage1_rf.predict_proba(X_full)[:, 1]

#         p2_gb = self.stage2_gb.predict_proba(X_compromise)[:, 1]
#         p2_rf = self.stage2_rf.predict_proba(X_compromise)[:, 1]
#         # Среднее двух stage-2 моделей
#         p2 = 0.5 * p2_gb + 0.5 * p2_rf

#         # Комбинируем: max
#         combined = np.maximum(p1, p2)
#         return combined

#     def predict_proba_stage1(self, X_full):
#         return self.stage1_rf.predict_proba(X_full)[:, 1]

#     def predict_proba_stage2(self, X_compromise):
#         p2_gb = self.stage2_gb.predict_proba(X_compromise)[:, 1]
#         p2_rf = self.stage2_rf.predict_proba(X_compromise)[:, 1]
#         return 0.5 * p2_gb + 0.5 * p2_rf


# # ──────────────────────────────────────────────────────────
# #  Main Training & Evaluation v6
# # ──────────────────────────────────────────────────────────

# def main():
#     config = TrainingConfig()
#     os.makedirs(config.results_dir, exist_ok=True)

#     pipeline = DataPipeline(config)
#     train_df, val_df, test_df = pipeline.load_and_prepare()

#     # Get features
#     X_train, y_train = pipeline.get_features_labels(train_df, fit=True)
#     X_val, y_val = pipeline.get_features_labels(val_df)
#     X_test, y_test = pipeline.get_features_labels(test_df)

#     # v6: Compromise-specific features
#     X_train_comp, y_train_comp = pipeline.get_compromise_features(train_df, fit=True)
#     X_val_comp, y_val_comp = pipeline.get_compromise_features(val_df)
#     X_test_comp, y_test_comp = pipeline.get_compromise_features(test_df)

#     X_train_normal, _ = pipeline.get_normal_data(train_df, fit=False)

#     attack_types_test = test_df["attack_type"].values
#     attack_types_val = val_df["attack_type"].values

#     results_list = []

#     print("\n" + "=" * 60)
#     print("  Training Models v6 (observable features only)")
#     print("=" * 60)

#     # ── 1. Isolation Forest ──
#     print("\n[1] Isolation Forest...")
#     iso = IsolationForest(
#         n_estimators=config.iso_forest_trees,
#         contamination=config.iso_forest_contamination,
#         random_state=config.seed
#     )
#     iso.fit(X_train_normal)
#     iso_scores_val = -iso.score_samples(X_val)
#     iso_scores_test = -iso.score_samples(X_test)
#     th_iso, _ = find_threshold(iso_scores_val, y_val, config.threshold_percentile)
#     r = evaluate_model("Isolation Forest", iso_scores_test, y_test,
#                        attack_types_test, threshold=th_iso)
#     results_list.append(r)

#     # ── 2. One-Class SVM ──
#     print("\n[2] One-Class SVM...")
#     svm = OneClassSVM(nu=config.ocsvm_nu, kernel=config.ocsvm_kernel, gamma="scale")
#     svm.fit(X_train_normal)
#     svm_scores_val = -svm.score_samples(X_val)
#     svm_scores_test = -svm.score_samples(X_test)
#     th_svm, _ = find_threshold(svm_scores_val, y_val, config.threshold_percentile)
#     r = evaluate_model("One-Class SVM", svm_scores_test, y_test,
#                        attack_types_test, threshold=th_svm)
#     results_list.append(r)

#     # ── 3. Autoencoder ──
#     print("\n[3] Autoencoder...")
#     n_feat = X_train.shape[1]
#     ae = Autoencoder(n_feat, config.ae_hidden_dims, config.ae_latent_dim)
#     ae_trainer = AETrainer(ae, config.ae_lr, config.ae_epochs,
#                            config.ae_batch, patience=15)
#     X_val_normal = X_val[y_val == 0]
#     ae_trainer.fit(X_train_normal, X_val_normal)
#     ae_scores_val = ae_trainer.reconstruction_error(X_val)
#     ae_scores_test = ae_trainer.reconstruction_error(X_test)
#     th_ae, _ = find_threshold(ae_scores_val, y_val, config.threshold_percentile)
#     r = evaluate_model("Autoencoder", ae_scores_test, y_test,
#                        attack_types_test, threshold=th_ae)
#     results_list.append(r)

#     # ── 4. v6: Variational Autoencoder ──
#     print("\n[4] Variational Autoencoder (v6)...")
#     vae = VAE(n_feat, config.vae_hidden_dims, config.vae_latent_dim)
#     vae_trainer = VAETrainer(vae, config.vae_lr, config.vae_epochs,
#                              config.vae_batch, kl_weight=config.vae_kl_weight,
#                              patience=15)
#     vae_trainer.fit(X_train_normal, X_val_normal)
#     vae_scores_val = vae_trainer.anomaly_score(X_val)
#     vae_scores_test = vae_trainer.anomaly_score(X_test)
#     th_vae, _ = find_threshold(vae_scores_val, y_val, config.threshold_percentile)
#     r = evaluate_model("VAE", vae_scores_test, y_test,
#                        attack_types_test, threshold=th_vae)
#     results_list.append(r)

#     # ── 5. LSTM-Autoencoder (v6: с dropout) ──
#     print("\n[5] LSTM-Autoencoder...")
#     train_normal_df = train_df[train_df[pipeline.LABEL_COL] == 0]
#     seq_train, _ = pipeline.get_sequential_data(train_normal_df, config.lstm_seq_len)

#     if len(seq_train) > 100:
#         lstm = LSTMAutoencoder(n_feat, config.lstm_hidden, config.lstm_layers,
#                                config.lstm_seq_len, dropout=0.2)  # v6: dropout
#         lstm_trainer = AETrainer(lstm, config.lstm_lr, config.lstm_epochs,
#                                  config.lstm_batch, patience=15)

#         seq_val, seq_y_val = pipeline.get_sequential_data(val_df, config.lstm_seq_len)
#         seq_val_normal = seq_val[seq_y_val == 0]

#         lstm_trainer.fit(seq_train, seq_val_normal if len(seq_val_normal) > 0 else None)

#         seq_test, seq_y_test = pipeline.get_sequential_data(test_df, config.lstm_seq_len)
#         seq_attack_types_test = []
#         node_ids_test = test_df["node_id"].unique()
#         for nid in node_ids_test:
#             mask = test_df["node_id"].values == nid
#             atypes = test_df.loc[mask, "attack_type"].values
#             for i in range(len(atypes) - config.lstm_seq_len + 1):
#                 seq_attack_types_test.append(atypes[i + config.lstm_seq_len - 1])
#         seq_attack_types_test = np.array(seq_attack_types_test)

#         lstm_scores_val = lstm_trainer.reconstruction_error(seq_val)
#         lstm_scores_test = lstm_trainer.reconstruction_error(seq_test)
#         th_lstm, _ = find_threshold(lstm_scores_val, seq_y_val, config.threshold_percentile)
#         r = evaluate_model("LSTM-Autoencoder", lstm_scores_test, seq_y_test,
#                            seq_attack_types_test, threshold=th_lstm)
#         results_list.append(r)
#     else:
#         print("  Not enough sequences for LSTM-AE, skipping.")

#     # ── 6. v6: Temporal Attention Autoencoder ──
#     print("\n[6] Temporal Attention AE (v6)...")
#     if len(seq_train) > 100:
#         tat_ae = TemporalAttentionAE(
#             input_dim=n_feat, hidden_dim=64, n_heads=4,
#             n_layers=2, seq_len=config.lstm_seq_len, dropout=0.15
#         )
#         tat_trainer = AETrainer(tat_ae, lr=5e-4, epochs=100,
#                                 batch_size=256, patience=15)
#         tat_trainer.fit(seq_train, seq_val_normal if len(seq_val_normal) > 0 else None)

#         tat_scores_val = tat_trainer.reconstruction_error(seq_val)
#         tat_scores_test = tat_trainer.reconstruction_error(seq_test)
#         th_tat, _ = find_threshold(tat_scores_val, seq_y_val, config.threshold_percentile)
#         r = evaluate_model("Temporal-Attention-AE", tat_scores_test, seq_y_test,
#                            seq_attack_types_test, threshold=th_tat)
#         results_list.append(r)
#     else:
#         print("  Not enough sequences, skipping.")

#     # ── 7. Random Forest (supervised) ──
#     print("\n[7] Random Forest (supervised)...")
#     rf = RandomForestClassifier(
#         n_estimators=config.rf_trees, random_state=config.seed,
#         class_weight="balanced", n_jobs=-1
#     )
#     rf.fit(X_train, y_train)
#     rf_proba_test = rf.predict_proba(X_test)[:, 1]
#     r = evaluate_model("Random Forest", rf_proba_test, y_test,
#                        attack_types_test, threshold=0.5)
#     results_list.append(r)

#     # Feature importance
#     print("\n  Feature importance (top 20):")
#     importances = rf.feature_importances_
#     feat_imp = sorted(zip(pipeline.feature_cols, importances),
#                       key=lambda x: x[1], reverse=True)
#     for fname, imp in feat_imp[:20]:
#         print(f"    {fname:45s} {imp:.4f} ({100*imp:.1f}%)")

#     ms_importance = sum(imp for fname, imp in feat_imp
#                         if fname in MS_FEATURES)
#     print(f"\n  Master-slave features total importance: {ms_importance:.4f} ({100*ms_importance:.1f}%)")

#     # ── 8. Gradient Boosting (supervised) ──
#     print("\n[8] Gradient Boosting (supervised)...")
#     gb = GradientBoostingClassifier(
#         n_estimators=config.gb_estimators, learning_rate=config.gb_lr,
#         max_depth=5, random_state=config.seed
#     )
#     gb.fit(X_train, y_train)
#     gb_proba_test = gb.predict_proba(X_test)[:, 1]
#     r = evaluate_model("Gradient Boosting", gb_proba_test, y_test,
#                        attack_types_test, threshold=0.5)
#     results_list.append(r)

#     # ── 9. v6: Cascade Compromise Detector ──
#     print("\n[9] Cascade Compromise Detector (v6)...")
#     cascade = CascadeCompromiseDetector(config, config.seed)
#     cascade.fit(X_train, y_train, X_train_comp, y_train_comp)

#     cascade_scores_test = cascade.predict_proba(X_test, X_test_comp)
#     r = evaluate_model("Cascade (v6)", cascade_scores_test, y_test,
#                        attack_types_test, threshold=0.5)
#     results_list.append(r)

#     # v6: Evaluate stage 2 separately on compromise
#     print("\n  --- Cascade Stage 2 (compromise-specific) ---")
#     s2_scores_test = cascade.predict_proba_stage2(X_test_comp)
#     r_s2 = evaluate_model("Cascade-Stage2-only", s2_scores_test, y_test,
#                           attack_types_test)
#     results_list.append(r_s2)

#     # ── 10. Specialized Ensemble ──
#     print("\n[10] Specialized Ensemble (v6)...")
#     ensemble = EnsembleDetector(FEATURE_GROUPS, pipeline.feature_cols, config.seed)
#     ensemble.fit(X_train_normal)

#     ens_scores_val = ensemble.score(X_val)
#     ens_scores_test = ensemble.score(X_test)

#     for ens_name in sorted(ens_scores_test.keys()):
#         s_val = ens_scores_val[ens_name]
#         s_test = ens_scores_test[ens_name]
#         th, _ = find_threshold(s_val, y_val, config.threshold_percentile)
#         r = evaluate_model(f"Ensemble:{ens_name}", s_test, y_test,
#                            attack_types_test, threshold=th)
#         results_list.append(r)

#     # ── 11. v6: Meta-ensemble — combine all unsupervised scores + cascade ──
#     print("\n[11] Meta-Ensemble (v6)...")
#     # Normalize all scores to [0,1]
#     def norm_score(s):
#         p1, p99 = np.percentile(s, 1), np.percentile(s, 99)
#         if p99 - p1 < 1e-9:
#             return np.zeros_like(s)
#         return np.clip((s - p1) / (p99 - p1), 0, 1)

#     # Собираем скоры от всех unsupervised моделей + cascade для тестовой выборки
#     meta_features_test = np.column_stack([
#         norm_score(iso_scores_test),
#         norm_score(svm_scores_test),
#         norm_score(ae_scores_test),
#         norm_score(vae_scores_test),
#         norm_score(cascade_scores_test),
#         norm_score(rf_proba_test),
#         norm_score(gb_proba_test),
#     ])
#     meta_features_val = np.column_stack([
#         norm_score(iso_scores_val),
#         norm_score(svm_scores_val),
#         norm_score(ae_scores_val),
#         norm_score(vae_scores_val),
#         norm_score(cascade.predict_proba(X_val, X_val_comp)),
#         norm_score(rf.predict_proba(X_val)[:, 1]),
#         norm_score(gb.predict_proba(X_val)[:, 1]),
#     ])
#     meta_features_train = np.column_stack([
#         norm_score(-iso.score_samples(X_train)),
#         norm_score(-svm.score_samples(X_train)),
#         norm_score(ae_trainer.reconstruction_error(X_train)),
#         norm_score(vae_trainer.anomaly_score(X_train)),
#         norm_score(cascade.predict_proba(X_train, X_train_comp)),
#         norm_score(rf.predict_proba(X_train)[:, 1]),
#         norm_score(gb.predict_proba(X_train)[:, 1]),
#     ])

#     # Meta-classifier: GradientBoosting на скорах базовых моделей
#     print("  Training meta-classifier on base model scores...")
#     meta_gb = GradientBoostingClassifier(
#         n_estimators=200, learning_rate=0.05,
#         max_depth=3, random_state=config.seed
#     )
#     meta_gb.fit(meta_features_train, y_train)
#     meta_scores_test = meta_gb.predict_proba(meta_features_test)[:, 1]
#     r = evaluate_model("Meta-Ensemble (v6)", meta_scores_test, y_test,
#                        attack_types_test, threshold=0.5)
#     results_list.append(r)

#     # ── 12. Ablation Study ──
#     print("\n" + "=" * 60)
#     print("  Ablation: Random Forest WITHOUT master-slave features")
#     print("=" * 60)

#     non_ms_cols = [c for c in pipeline.feature_cols if c not in MS_FEATURES]
#     non_ms_indices = [pipeline.feature_cols.index(c) for c in non_ms_cols]

#     X_train_no_ms = X_train[:, non_ms_indices]
#     X_test_no_ms = X_test[:, non_ms_indices]

#     rf_no_ms = RandomForestClassifier(
#         n_estimators=config.rf_trees, random_state=config.seed,
#         class_weight="balanced", n_jobs=-1
#     )
#     rf_no_ms.fit(X_train_no_ms, y_train)
#     rf_no_ms_proba = rf_no_ms.predict_proba(X_test_no_ms)[:, 1]
#     r_no_ms = evaluate_model("RF (no MS features)", rf_no_ms_proba, y_test,
#                              attack_types_test, threshold=0.5)

#     rf_full = next(r for r in results_list if r["name"] == "Random Forest")
#     delta_auc = r_no_ms["auc_roc"] - rf_full["auc_roc"]
#     delta_f1 = r_no_ms["f1"] - rf_full["f1"]
#     print(f"\n  Δ AUC-ROC: {delta_auc:+.4f}")
#     print(f"  Δ F1:      {delta_f1:+.4f}")
#     if delta_f1 < 0:
#         print("  → Removing MS features HURTS performance → MS features are valuable")
#     else:
#         print("  → Removing MS features does not hurt → investigate feature overlap")

#     # ── 13. v6: Ablation — без temporal features ──
#     print("\n" + "=" * 60)
#     print("  Ablation v6: Random Forest WITHOUT temporal features")
#     print("=" * 60)

#     temporal_features = [
#         "var_obs_total_messages", "var_obs_entropy",
#         "var_obs_inbound_outbound_ratio", "var_obs_map_dominance_ratio",
#         "autocorr_obs_total_messages", "integrity_fail_rate",
#         "obs_destinations_cv",
#     ]
#     non_temp_cols = [c for c in pipeline.feature_cols if c not in temporal_features]
#     non_temp_indices = [pipeline.feature_cols.index(c) for c in non_temp_cols]

#     X_train_no_temp = X_train[:, non_temp_indices]
#     X_test_no_temp = X_test[:, non_temp_indices]

#     rf_no_temp = RandomForestClassifier(
#         n_estimators=config.rf_trees, random_state=config.seed,
#         class_weight="balanced", n_jobs=-1
#     )
#     rf_no_temp.fit(X_train_no_temp, y_train)
#     rf_no_temp_proba = rf_no_temp.predict_proba(X_test_no_temp)[:, 1]
#     r_no_temp = evaluate_model("RF (no temporal features)", rf_no_temp_proba, y_test,
#                                attack_types_test, threshold=0.5)

#     delta_f1_temp = r_no_temp["f1"] - rf_full["f1"]
#     print(f"\n  Δ F1 (removing temporal): {delta_f1_temp:+.4f}")

#     # Per-attack comparison: compromise specifically
#     comp_mask = attack_types_test == "slave_compromise"
#     if comp_mask.sum() > 0:
#         comp_full = (rf_proba_test[comp_mask] >= 0.5).sum()
#         comp_no_temp = (rf_no_temp_proba[comp_mask] >= 0.5).sum()
#         comp_total = comp_mask.sum()
#         print(f"  Compromise detection: full={comp_full}/{comp_total} "
#               f"({100*comp_full/comp_total:.1f}%), "
#               f"no_temp={comp_no_temp}/{comp_total} "
#               f"({100*comp_no_temp/comp_total:.1f}%)")
#         print(f"  Δ Compromise detection: {comp_full - comp_no_temp:+d} records "
#               f"({100*(comp_full - comp_no_temp)/comp_total:+.1f} p.p.)")

#     # ── 14. v6: Ablation — без cascade (Stage 2) ──
#     print("\n" + "=" * 60)
#     print("  Ablation v6: Cascade vs Stage-1 only")
#     print("=" * 60)

#     s1_only_scores = cascade.predict_proba_stage1(X_test)
#     r_s1 = evaluate_model("Cascade-Stage1-only", s1_only_scores, y_test,
#                           attack_types_test, threshold=0.5)

#     cascade_r = next(r for r in results_list if r["name"] == "Cascade (v6)")
#     print(f"\n  Δ F1 (Cascade vs Stage-1 only): "
#           f"{cascade_r['f1'] - r_s1['f1']:+.4f}")

#     if comp_mask.sum() > 0:
#         comp_cascade = (cascade_scores_test[comp_mask] >= 0.5).sum()
#         comp_s1 = (s1_only_scores[comp_mask] >= 0.5).sum()
#         print(f"  Compromise: Cascade={comp_cascade}/{comp_total} "
#               f"({100*comp_cascade/comp_total:.1f}%), "
#               f"Stage1-only={comp_s1}/{comp_total} "
#               f"({100*comp_s1/comp_total:.1f}%)")

#     # ── Save Results ──
#     results_df = pd.DataFrame(results_list)
#     results_path = os.path.join(config.results_dir, "full_comparison_v6.csv")
#     results_df.to_csv(results_path, index=False)
#     print(f"\nResults saved to {results_path}")

#     feat_path = os.path.join(config.results_dir, "feature_columns_v6.json")
#     with open(feat_path, "w") as f:
#         json.dump({
#             "observable_features": pipeline.feature_cols,
#             "compromise_features": pipeline.compromise_feature_cols,
#             "ms_features": MS_FEATURES,
#             "temporal_features": temporal_features,
#             "ground_truth_cols": pipeline.GROUND_TRUTH_COLS,
#             "feature_groups": {k: v for k, v in FEATURE_GROUPS.items()},
#         }, f, indent=2)
#     print(f"Feature config saved to {feat_path}")

#     # Summary
#     print("\n" + "=" * 70)
#     print("  SUMMARY v6 — improved compromise detection")
#     print("=" * 70)
#     print(f"{'Model':<35s} {'AUC-ROC':>8s} {'AUC-PR':>8s} {'F1':>8s} {'FPR':>8s}")
#     print("-" * 67)
#     for r in results_list:
#         print(f"{r['name']:<35s} {r['auc_roc']:8.4f} {r['auc_pr']:8.4f} "
#               f"{r['f1']:8.4f} {r['fpr']:8.4f}")

#     # v6: Detailed compromise detection summary
#     print("\n" + "=" * 70)
#     print("  COMPROMISE DETECTION SUMMARY (v6)")
#     print("=" * 70)
#     if comp_mask.sum() > 0:
#         comp_total = comp_mask.sum()
#         print(f"  Total compromise records in test: {comp_total}")
#         print(f"  {'Model':<35s} {'Detected':>10s} {'Rate':>8s}")
#         print(f"  {'-'*55}")

#         # Evaluate all models on compromise specifically
#         model_scores = {
#             "Isolation Forest": iso_scores_test,
#             "One-Class SVM": svm_scores_test,
#             "Autoencoder": ae_scores_test,
#             "VAE": vae_scores_test,
#             "Random Forest": rf_proba_test,
#             "Gradient Boosting": gb_proba_test,
#             "Cascade (v6)": cascade_scores_test,
#             "Meta-Ensemble (v6)": meta_scores_test,
#         }

#         for mname, mscores in model_scores.items():
#             # Find the threshold used for this model
#             matching = [r for r in results_list if r["name"] == mname]
#             if matching:
#                 th = matching[0]["threshold"]
#                 detected = (mscores[comp_mask] >= th).sum()
#                 print(f"  {mname:<35s} {detected:>5d}/{comp_total:<4d} "
#                       f"{100*detected/comp_total:>6.1f}%")


# if __name__ == "__main__":
#     main()


"""
SS7 Master-Slave Anomaly Detection Training Pipeline v6
========================================================
С графиками визуализации результатов.
"""

import numpy as np
import pandas as pd
import os
import json
import warnings
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional

from sklearn.ensemble import IsolationForest, RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    confusion_matrix, classification_report, precision_recall_curve,
    roc_curve
)
from sklearn.impute import SimpleImputer

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import matplotlib
matplotlib.use('Agg')  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.ticker as mticker

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────
#  Matplotlib global style
# ──────────────────────────────────────────────────────────

# plt.rcParams.update({
#     'figure.dpi': 150,
#     'savefig.dpi': 200,
#     'font.size': 10,
#     'axes.titlesize': 12,
#     'axes.labelsize': 10,
#     'xtick.labelsize': 8,
#     'ytick.labelsize': 8,
#     'legend.fontsize': 8,
#     'figure.facecolor': 'white',
#     'axes.facecolor': '#fafafa',
#     'axes.grid': True,
#     'grid.alpha': 0.3,
#     'grid.linestyle': '--',
# })

# ──────────────────────────────────────────────────────────
#  Matplotlib global style — GRAYSCALE
# ──────────────────────────────────────────────────────────

plt.rcParams.update({
    'figure.dpi': 150,
    'savefig.dpi': 200,
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.facecolor': 'white',
    'axes.facecolor': '#f2f2f2',
    'axes.grid': True,
    'grid.alpha': 0.35,
    'grid.linestyle': '--',
    'grid.color': '#b5b5b5',
    'axes.edgecolor': '#4d4d4d',
    'text.color': '#111111',
    'axes.labelcolor': '#111111',
    'xtick.color': '#111111',
    'ytick.color': '#111111',
    'svg.fonttype': 'none',   # текст в SVG сохраняется как текст
})

# Colour palette
COLORS = {
    'primary': '#2563EB',
    'secondary': '#7C3AED',
    'success': '#059669',
    'danger': '#DC2626',
    'warning': '#D97706',
    'info': '#0891B2',
    'muted': '#6B7280',
    'bg_highlight': '#EFF6FF',
}

MODEL_COLORS = {
    'Isolation Forest': '#3B82F6',
    'One-Class SVM': '#8B5CF6',
    'Autoencoder': '#10B981',
    'VAE': '#F59E0B',
    'LSTM-Autoencoder': '#EF4444',
    'Temporal-Attention-AE': '#EC4899',
    'Random Forest': '#06B6D4',
    'Gradient Boosting': '#8B5CF6',
    'Cascade (v6)': '#059669',
    'Cascade-Stage2-only': '#84CC16',
    'Meta-Ensemble (v6)': '#F97316',
}

ATTACK_COLORS = {
    'sms_intercept': '#3B82F6',
    'location_track': '#8B5CF6',
    'signaling_dos': '#EF4444',
    'irsf': '#F59E0B',
    'slave_compromise': '#059669',
}

ATTACK_LABELS = {
    'sms_intercept': 'Перехват SMS',
    'location_track': 'Отслеживание',
    'signaling_dos': 'Сигнальный DoS',
    'irsf': 'IRSF',
    'slave_compromise': 'Компрометация',
}


# ──────────────────────────────────────────────────────────
#  Configuration
# ──────────────────────────────────────────────────────────

@dataclass
class TrainingConfig:
    data_path: str = "ss7_dataset_v6.csv"
    results_dir: str = "results_v6"
    plots_dir: str = "results_v6/plots"
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


# ──────────────────────────────────────────────────────────
#  Data Pipeline v6 (без изменений)
# ──────────────────────────────────────────────────────────

class DataPipeline:
    OBSERVABLE_FEATURE_COLS = [
        "cu_delivered", "cu_rtt_delay", "cu_req_delay", "cu_resp_delay",
        "cu_combined_loss_prob", "integrity_check", "consecutive_cu_losses",
        "cu_processing_delay", "cu_response_time_jitter",
        "cu_staleness", "cu_consistency_error",
        "obs_total_messages", "obs_map_count", "obs_isup_count", "obs_tcap_count",
        "obs_entropy", "obs_inbound_outbound_ratio",
        "obs_international_fraction", "obs_n_unique_destinations",
        "obs_map_dominance_ratio", "obs_map_top2_ratio", "obs_map_gini",
        "obs_ratio_map_sri", "obs_ratio_map_sri_sm", "obs_ratio_map_psi",
        "obs_ratio_map_ati", "obs_ratio_map_update_loc",
        "obs_ratio_map_insert_sub", "obs_ratio_map_send_auth",
        "obs_ratio_map_other_map",
        "z_obs_total_messages", "z_obs_map_count",
        "z_obs_map_dominance_ratio", "z_obs_ratio_map_sri_sm",
        "z_obs_ratio_map_psi",
        "dev_total_messages_vs_zone_median",
        "dev_map_count_vs_zone_median",
        "dev_isup_count_vs_zone_median",
        "zone_io_ratio_rank",
        "dev_io_ratio_vs_zone_median",
        "dev_integrity_fail_rate_vs_zone",
        "dev_var_total_vs_zone",
        "var_obs_total_messages",
        "var_obs_entropy",
        "var_obs_inbound_outbound_ratio",
        "var_obs_map_dominance_ratio",
        "autocorr_obs_total_messages",
        "integrity_fail_rate",
        "obs_destinations_cv",
    ]

    GROUND_TRUTH_COLS = [
        "gt_total_messages", "gt_map_count", "gt_isup_count",
        "gt_tcap_count", "gt_entropy", "gt_inbound_outbound_ratio",
        "gt_international_fraction", "gt_n_unique_destinations",
    ]

    LABEL_COL = "is_anomaly"
    ATTACK_COL = "attack_type"

    COMPROMISE_FEATURES = [
        "obs_inbound_outbound_ratio", "integrity_check",
        "integrity_fail_rate", "cu_consistency_error",
        "cu_processing_delay", "cu_response_time_jitter",
        "var_obs_total_messages", "var_obs_entropy",
        "var_obs_inbound_outbound_ratio", "var_obs_map_dominance_ratio",
        "autocorr_obs_total_messages", "obs_destinations_cv",
        "zone_io_ratio_rank", "dev_io_ratio_vs_zone_median",
        "dev_integrity_fail_rate_vs_zone", "dev_var_total_vs_zone",
        "consecutive_cu_losses", "cu_staleness",
        "dev_total_messages_vs_zone_median",
        "dev_map_count_vs_zone_median",
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

    def load_and_prepare(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        df = pd.read_csv(self.config.data_path)
        print(f"Loaded {len(df)} records, anomalies: {df[self.LABEL_COL].sum()} "
              f"({100*df[self.LABEL_COL].mean():.2f}%)")
        print(f"Attack distribution:\n{df[self.ATTACK_COL].value_counts()}\n")

        if "node_type" in df.columns:
            df["node_type_enc"] = self.le_node_type.fit_transform(df["node_type"])
        else:
            df["node_type_enc"] = 0

        self.feature_cols = []
        for col in self.OBSERVABLE_FEATURE_COLS:
            if col in df.columns:
                self.feature_cols.append(col)
        self.feature_cols.append("node_type_enc")

        self.compromise_feature_cols = []
        for col in self.COMPROMISE_FEATURES:
            if col in df.columns:
                self.compromise_feature_cols.append(col)
        self.compromise_feature_cols.append("node_type_enc")

        for col in self.feature_cols:
            if col.startswith("gt_"):
                raise ValueError(f"Ground-truth column '{col}' found in feature set!")

        print(f"Observable features: {len(self.feature_cols)}")
        print(f"Compromise-specific features: {len(self.compromise_feature_cols)}")
        print(f"Ground-truth columns (excluded): "
              f"{len([c for c in self.GROUND_TRUTH_COLS if c in df.columns])}")

        train_val, test = train_test_split(
            df, test_size=self.config.test_size,
            stratify=df[self.LABEL_COL], random_state=self.config.seed
        )
        train, val = train_test_split(
            train_val, test_size=self.config.val_size / (1 - self.config.test_size),
            stratify=train_val[self.LABEL_COL], random_state=self.config.seed
        )

        print(f"Split — Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
        return train, val, test

    def get_features_labels(self, df, fit=False):
        X = df[self.feature_cols].copy()
        if fit:
            X_imp = self.imputer.fit_transform(X)
            X_scaled = self.scaler.fit_transform(X_imp)
        else:
            X_imp = self.imputer.transform(X)
            X_scaled = self.scaler.transform(X_imp)
        y = df[self.LABEL_COL].values
        return X_scaled, y

    def get_compromise_features(self, df, fit=False):
        X = df[self.compromise_feature_cols].copy()
        if fit:
            X_imp = self.compromise_imputer.fit_transform(X)
            X_scaled = self.compromise_scaler.fit_transform(X_imp)
        else:
            X_imp = self.compromise_imputer.transform(X)
            X_scaled = self.compromise_scaler.transform(X_imp)
        y = (df[self.ATTACK_COL] == "slave_compromise").astype(int).values
        return X_scaled, y

    def get_normal_data(self, df, fit=False):
        normal = df[df[self.LABEL_COL] == 0]
        return self.get_features_labels(normal, fit=fit)

    def get_sequential_data(self, df, seq_len):
        X_all, y_all = self.get_features_labels(df, fit=False)
        sequences, labels = [], []
        for nid in df["node_id"].unique():
            mask = df["node_id"].values == nid
            X_node, y_node = X_all[mask], y_all[mask]
            for i in range(len(X_node) - seq_len + 1):
                sequences.append(X_node[i:i + seq_len])
                labels.append(y_node[i + seq_len - 1])
        return np.array(sequences), np.array(labels)


# ──────────────────────────────────────────────────────────
#  Neural network models (без изменений)
# ──────────────────────────────────────────────────────────

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
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar


class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers, seq_len, dropout=0.2):
        super().__init__()
        self.seq_len = seq_len
        self.encoder = nn.LSTM(input_dim, hidden_dim, n_layers,
                               batch_first=True, dropout=dropout if n_layers > 1 else 0)
        self.decoder = nn.LSTM(hidden_dim, hidden_dim, n_layers,
                               batch_first=True, dropout=dropout if n_layers > 1 else 0)
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
            dim_feedforward=hidden_dim * 2, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, n_layers)
        self.output_proj = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        return self.output_proj(self.transformer(self.input_proj(x)))


class AETrainer:
    def __init__(self, model, lr, epochs, batch_size, patience=15, device=None):
        self.model = model
        self.lr, self.epochs, self.batch_size = lr, epochs, batch_size
        self.patience = patience
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.train_losses = []
        self.val_losses = []

    def fit(self, X_train, X_val=None):
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-5)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, self.epochs)
        crit = nn.MSELoss()
        dl = DataLoader(TensorDataset(torch.FloatTensor(X_train)),
                        batch_size=self.batch_size, shuffle=True)
        best_val, patience_cnt, best_state = float("inf"), 0, None
        self.train_losses, self.val_losses = [], []

        for epoch in range(self.epochs):
            self.model.train()
            tloss = 0.0
            for (batch,) in dl:
                batch = batch.to(self.device)
                loss = crit(self.model(batch), batch)
                opt.zero_grad(); loss.backward(); opt.step()
                tloss += loss.item() * len(batch)
            tloss /= len(X_train)
            sched.step()
            self.train_losses.append(tloss)

            if X_val is not None:
                vloss = self._evaluate(X_val, crit)
                self.val_losses.append(vloss)
                if vloss < best_val:
                    best_val, patience_cnt = vloss, 0
                    best_state = {k: v.cpu().clone()
                                  for k, v in self.model.state_dict().items()}
                else:
                    patience_cnt += 1
                if (epoch + 1) % 20 == 0:
                    print(f"  Epoch {epoch+1}/{self.epochs}: "
                          f"train={tloss:.6f}, val={vloss:.6f}")
                if patience_cnt >= self.patience:
                    print(f"  Early stopping at epoch {epoch+1}")
                    break

        if best_state:
            self.model.load_state_dict(best_state)
            self.model.to(self.device)

    def _evaluate(self, X, crit):
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
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.train_losses, self.val_losses = [], []

    def _loss(self, x, x_hat, mu, logvar):
        recon = F.mse_loss(x_hat, x)
        kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return recon + self.kl_weight * kl, recon, kl

    def fit(self, X_train, X_val=None):
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-5)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, self.epochs)
        dl = DataLoader(TensorDataset(torch.FloatTensor(X_train)),
                        batch_size=self.batch_size, shuffle=True)
        best_val, patience_cnt, best_state = float("inf"), 0, None
        self.train_losses, self.val_losses = [], []

        for epoch in range(self.epochs):
            self.model.train()
            tloss = 0.0
            for (batch,) in dl:
                batch = batch.to(self.device)
                x_hat, mu, logvar = self.model(batch)
                loss, _, _ = self._loss(batch, x_hat, mu, logvar)
                opt.zero_grad(); loss.backward(); opt.step()
                tloss += loss.item() * len(batch)
            tloss /= len(X_train)
            sched.step()
            self.train_losses.append(tloss)

            if X_val is not None:
                vloss = self._eval_val(X_val)
                self.val_losses.append(vloss)
                if vloss < best_val:
                    best_val, patience_cnt = vloss, 0
                    best_state = {k: v.cpu().clone()
                                  for k, v in self.model.state_dict().items()}
                else:
                    patience_cnt += 1
                if (epoch + 1) % 20 == 0:
                    print(f"  Epoch {epoch+1}/{self.epochs}: "
                          f"train={tloss:.6f}, val={vloss:.6f}")
                if patience_cnt >= self.patience:
                    print(f"  Early stopping at epoch {epoch+1}")
                    break

        if best_state:
            self.model.load_state_dict(best_state)
            self.model.to(self.device)

    def _eval_val(self, X):
        self.model.eval()
        with torch.no_grad():
            t = torch.FloatTensor(X).to(self.device)
            x_hat, mu, logvar = self.model(t)
            loss, _, _ = self._loss(t, x_hat, mu, logvar)
        return loss.item()

    def anomaly_score(self, X):
        self.model.eval()
        with torch.no_grad():
            t = torch.FloatTensor(X).to(self.device)
            x_hat, mu, logvar = self.model(t)
            recon = torch.mean((t - x_hat) ** 2, dim=-1)
            kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
        return (recon + self.kl_weight * kl).cpu().numpy()


# ──────────────────────────────────────────────────────────
#  Evaluation helpers
# ──────────────────────────────────────────────────────────

def find_threshold(scores, y_true, percentile=95.0):
    normal_scores = scores[y_true == 0]
    th_pct = np.percentile(normal_scores, percentile)
    prec, rec, ths = precision_recall_curve(y_true, scores)
    f1s = 2 * prec * rec / (prec + rec + 1e-9)
    th_f1 = ths[min(np.argmax(f1s), len(ths) - 1)]
    y_pct = (scores >= th_pct).astype(int)
    y_f1 = (scores >= th_f1).astype(int)
    if f1_score(y_true, y_f1, zero_division=0) >= f1_score(y_true, y_pct, zero_division=0):
        return th_f1, "max_f1"
    return th_pct, "percentile"


def evaluate_model(name, scores, y_true, attack_types, threshold=None, percentile=95.0):
    if threshold is None:
        threshold, method = find_threshold(scores, y_true, percentile)
    else:
        method = "provided"
    y_pred = (scores >= threshold).astype(int)
    auc_roc = roc_auc_score(y_true, scores) if len(np.unique(y_true)) > 1 else 0.
    auc_pr = average_precision_score(y_true, scores) if len(np.unique(y_true)) > 1 else 0.
    f1 = f1_score(y_true, y_pred, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.

    print(f"\n{'='*60}\n  {name}\n{'='*60}")
    print(f"  AUC-ROC: {auc_roc:.4f}  |  AUC-PR: {auc_pr:.4f}  |  F1: {f1:.4f}")
    print(f"  FPR: {fpr:.4f}  |  TP: {tp}  FP: {fp}  FN: {fn}  TN: {tn}")
    print(f"  Threshold: {threshold:.4f} ({method})")
    print(f"  Detection by attack type:")
    per_attack = {}
    for at in sorted(set(attack_types)):
        if at == "none":
            continue
        m = attack_types == at
        if m.sum() > 0:
            det = y_pred[m].sum()
            tot = m.sum()
            per_attack[at] = (det, tot, 100 * det / tot)
            print(f"    {at:25s}: {det}/{tot} = {100*det/tot:.1f}%")

    return {
        "name": name, "auc_roc": auc_roc, "auc_pr": auc_pr, "f1": f1,
        "fpr": fpr, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "threshold": threshold, "method": method, "per_attack": per_attack,
        "scores": scores, "y_pred": y_pred,
    }


# ──────────────────────────────────────────────────────────
#  Specialized ensemble (compact)
# ──────────────────────────────────────────────────────────

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
        self.models["iso"] = IsolationForest(n_estimators=200, contamination=0.05,
                                             random_state=self.seed).fit(X)
        self.models["svm"] = OneClassSVM(nu=0.05, kernel="rbf", gamma="scale").fit(X)
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


# ──────────────────────────────────────────────────────────
#  Feature groups / MS features
# ──────────────────────────────────────────────────────────

FEATURE_GROUPS = {
    "map_profile": [
        "obs_ratio_map_sri", "obs_ratio_map_sri_sm", "obs_ratio_map_psi",
        "obs_ratio_map_ati", "obs_ratio_map_update_loc",
        "obs_ratio_map_insert_sub", "obs_ratio_map_send_auth",
        "obs_ratio_map_other_map",
        "obs_map_dominance_ratio", "obs_map_top2_ratio", "obs_map_gini",
        "z_obs_map_dominance_ratio", "z_obs_ratio_map_sri_sm",
        "z_obs_ratio_map_psi",
    ],
    "volume_timing": [
        "obs_total_messages", "obs_map_count", "obs_isup_count",
        "obs_tcap_count", "obs_entropy",
        "z_obs_total_messages", "z_obs_map_count",
    ],
    "international": [
        "obs_international_fraction", "obs_n_unique_destinations",
    ],
    "control_unit": [
        "cu_delivered", "cu_rtt_delay", "cu_req_delay", "cu_resp_delay",
        "cu_combined_loss_prob", "integrity_check", "consecutive_cu_losses",
        "cu_processing_delay", "cu_response_time_jitter",
        "cu_staleness", "cu_consistency_error",
    ],
    "baseline_deviation": [
        "dev_total_messages_vs_zone_median", "dev_map_count_vs_zone_median",
        "dev_isup_count_vs_zone_median", "obs_inbound_outbound_ratio",
        "zone_io_ratio_rank", "dev_io_ratio_vs_zone_median",
    ],
    "compromise_indicators": [
        "obs_inbound_outbound_ratio", "integrity_fail_rate",
        "var_obs_total_messages", "var_obs_entropy",
        "var_obs_inbound_outbound_ratio", "var_obs_map_dominance_ratio",
        "autocorr_obs_total_messages", "obs_destinations_cv",
        "cu_consistency_error", "cu_processing_delay",
        "cu_response_time_jitter", "zone_io_ratio_rank",
        "dev_io_ratio_vs_zone_median", "dev_integrity_fail_rate_vs_zone",
        "dev_var_total_vs_zone",
    ],
}

MS_FEATURES = [
    "cu_delivered", "cu_rtt_delay", "cu_req_delay", "cu_resp_delay",
    "cu_combined_loss_prob", "integrity_check", "consecutive_cu_losses",
    "cu_processing_delay", "cu_response_time_jitter",
    "cu_staleness", "cu_consistency_error",
    "obs_total_messages", "obs_map_count", "obs_isup_count", "obs_tcap_count",
    "obs_inbound_outbound_ratio",
    "dev_total_messages_vs_zone_median", "dev_map_count_vs_zone_median",
    "dev_isup_count_vs_zone_median",
    "zone_io_ratio_rank", "dev_io_ratio_vs_zone_median",
    "dev_integrity_fail_rate_vs_zone", "dev_var_total_vs_zone",
    "var_obs_total_messages", "var_obs_entropy",
    "var_obs_inbound_outbound_ratio", "var_obs_map_dominance_ratio",
    "autocorr_obs_total_messages", "integrity_fail_rate", "obs_destinations_cv",
]


# ──────────────────────────────────────────────────────────
#  Cascade Compromise Detector
# ──────────────────────────────────────────────────────────

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
#  VISUALIZATION MODULE — GRAYSCALE + PNG/SVG
# ══════════════════════════════════════════════════════════

class ResultsVisualizer:
    """Generates all publication-quality plots in grayscale."""

    def __init__(self, plots_dir: str):
        self.plots_dir = plots_dir
        os.makedirs(plots_dir, exist_ok=True)

        # Greyscale styles for models
        self.model_colors = {
            'Isolation Forest': '#111111',
            'One-Class SVM': '#2a2a2a',
            'Autoencoder': '#3f3f3f',
            'VAE': '#555555',
            'LSTM-Autoencoder': '#6b6b6b',
            'Temporal-Attention-AE': '#808080',
            'Random Forest': '#949494',
            'Gradient Boosting': '#aaaaaa',
            'Cascade (v6)': '#4a4a4a',
            'Cascade-Stage2-only': '#707070',
            'Meta-Ensemble (v6)': '#202020',
        }

        self.model_linestyles = {
            'Isolation Forest': '-',
            'One-Class SVM': '--',
            'Autoencoder': '-.',
            'VAE': ':',
            'LSTM-Autoencoder': (0, (5, 1)),
            'Temporal-Attention-AE': (0, (3, 1, 1, 1)),
            'Random Forest': '-',
            'Gradient Boosting': '--',
            'Cascade (v6)': '-.',
            'Cascade-Stage2-only': ':',
            'Meta-Ensemble (v6)': (0, (7, 2)),
        }

        # Greyscale styles for attacks
        self.attack_colors = {
            'sms_intercept': '#2b2b2b',
            'location_track': '#4b4b4b',
            'signaling_dos': '#6a6a6a',
            'irsf': '#8a8a8a',
            'slave_compromise': '#111111',
        }

        self.attack_linestyles = {
            'sms_intercept': '-',
            'location_track': '--',
            'signaling_dos': '-.',
            'irsf': ':',
            'slave_compromise': (0, (5, 1)),
        }

        self.attack_hatches = {
            'sms_intercept': '////',
            'location_track': '\\\\\\\\',
            'signaling_dos': 'xxxx',
            'irsf': '....',
            'slave_compromise': '----',
        }

        self.attack_markers = {
            'sms_intercept': 'o',
            'location_track': 's',
            'signaling_dos': '^',
            'irsf': 'D',
            'slave_compromise': 'x',
        }

        self.attack_labels = {
            'sms_intercept': 'Перехват SMS',
            'location_track': 'Отслеживание',
            'signaling_dos': 'Сигнальный DoS',
            'irsf': 'IRSF',
            'slave_compromise': 'Компрометация',
        }

    def _save(self, fig, name):
        """Save every figure to both PNG and SVG."""
        stem = os.path.splitext(name)[0]

        png_path = os.path.join(self.plots_dir, f"{stem}.png")
        svg_path = os.path.join(self.plots_dir, f"{stem}.svg")

        fig.savefig(
            png_path,
            bbox_inches='tight',
            facecolor='white',
            format='png'
        )
        fig.savefig(
            svg_path,
            bbox_inches='tight',
            facecolor='white',
            format='svg'
        )

        plt.close(fig)
        print(f"  Saved: {png_path}")
        print(f"  Saved: {svg_path}")

    def _model_style(self, name: str) -> Dict:
        return {
            "color": self.model_colors.get(name, '#5c5c5c'),
            "linestyle": self.model_linestyles.get(name, '-')
        }

    def _attack_style(self, name: str) -> Dict:
        return {
            "color": self.attack_colors.get(name, '#7a7a7a'),
            "linestyle": self.attack_linestyles.get(name, '-'),
            "hatch": self.attack_hatches.get(name, '')
        }

    def _hist_step(self, ax, data, bins, color, linestyle, label, density=True, linewidth=1.8, alpha=1.0):
        """Helper: step histogram with optional custom linestyle."""
        if len(data) == 0:
            return
        _, _, patches = ax.hist(
            data, bins=bins, histtype='step',
            linewidth=linewidth, color=color,
            label=label, density=density, alpha=alpha
        )
        for p in patches:
            if hasattr(p, "set_linestyle"):
                p.set_linestyle(linestyle)

    # ── 1. ROC Curves ──────────────────────────────────

    def plot_roc_curves(self, results_list, y_test):
        fig, ax = plt.subplots(figsize=(8, 6))

        main_models = [
            'Isolation Forest', 'One-Class SVM', 'Autoencoder', 'VAE',
            'Random Forest', 'Gradient Boosting', 'Cascade (v6)', 'Meta-Ensemble (v6)'
        ]

        for r in results_list:
            if r['name'] not in main_models:
                continue
            fpr_arr, tpr_arr, _ = roc_curve(y_test, r['scores'])
            style = self._model_style(r['name'])
            ax.plot(
                fpr_arr, tpr_arr,
                color=style['color'],
                linestyle=style['linestyle'],
                linewidth=1.8,
                label=f"{r['name']} (AUC={r['auc_roc']:.3f})"
            )

        ax.plot([0, 1], [0, 1], color='#9a9a9a', linestyle='--', alpha=0.8, linewidth=1)
        ax.set_xlabel('Доля ложных тревог (FPR)')
        ax.set_ylabel('Доля истинных обнаружений (TPR)')
        ax.set_title('ROC-кривые моделей обнаружения аномалий')
        ax.legend(loc='lower right', framealpha=0.9)
        ax.set_xlim([-0.01, 1.01])
        ax.set_ylim([-0.01, 1.01])
        self._save(fig, '01_roc_curves.png')

    # ── 2. ROC zoom (low FPR) ──────────────────────────

    def plot_roc_zoom(self, results_list, y_test):
        fig, ax = plt.subplots(figsize=(8, 6))

        main_models = [
            'Autoencoder', 'VAE', 'Random Forest',
            'Gradient Boosting', 'Cascade (v6)', 'Meta-Ensemble (v6)'
        ]

        for r in results_list:
            if r['name'] not in main_models:
                continue
            fpr_arr, tpr_arr, _ = roc_curve(y_test, r['scores'])
            style = self._model_style(r['name'])
            ax.plot(
                fpr_arr, tpr_arr,
                color=style['color'],
                linestyle=style['linestyle'],
                linewidth=2,
                label=f"{r['name']} (AUC={r['auc_roc']:.4f})"
            )

        ax.set_xlabel('Доля ложных тревог (FPR)')
        ax.set_ylabel('Доля истинных обнаружений (TPR)')
        ax.set_title('ROC-кривые (увеличенная область FPR < 10%)')
        ax.set_xlim([-0.002, 0.10])
        ax.set_ylim([0.85, 1.005])
        ax.legend(loc='lower right', framealpha=0.9)
        self._save(fig, '02_roc_zoom.png')

    # ── 3. Precision-Recall Curves ─────────────────────

    def plot_pr_curves(self, results_list, y_test):
        fig, ax = plt.subplots(figsize=(8, 6))

        main_models = [
            'Isolation Forest', 'One-Class SVM', 'Autoencoder', 'VAE',
            'Random Forest', 'Gradient Boosting', 'Cascade (v6)', 'Meta-Ensemble (v6)'
        ]

        for r in results_list:
            if r['name'] not in main_models:
                continue
            prec, rec, _ = precision_recall_curve(y_test, r['scores'])
            style = self._model_style(r['name'])
            ax.plot(
                rec, prec,
                color=style['color'],
                linestyle=style['linestyle'],
                linewidth=1.8,
                label=f"{r['name']} (AP={r['auc_pr']:.3f})"
            )

        baseline = y_test.mean()
        ax.axhline(
            y=baseline, color='#8c8c8c', linestyle='--', alpha=0.9,
            label=f'Базовый уровень ({baseline:.3f})'
        )
        ax.set_xlabel('Полнота (Recall)')
        ax.set_ylabel('Точность (Precision)')
        ax.set_title('Кривые Precision-Recall')
        ax.legend(loc='lower left', framealpha=0.9)
        ax.set_xlim([-0.01, 1.01])
        ax.set_ylim([-0.01, 1.05])
        self._save(fig, '03_precision_recall.png')

    # ── 4. Per-attack detection heatmap ────────────────

    def plot_attack_detection_heatmap(self, results_list):
        main_models = [
            'Isolation Forest', 'One-Class SVM', 'Autoencoder', 'VAE',
            'Random Forest', 'Gradient Boosting', 'Cascade (v6)', 'Meta-Ensemble (v6)'
        ]
        attacks = ['sms_intercept', 'location_track', 'signaling_dos',
                   'irsf', 'slave_compromise']

        matrix = []
        model_names = []
        for r in results_list:
            if r['name'] not in main_models:
                continue
            row = []
            for at in attacks:
                if at in r['per_attack']:
                    row.append(r['per_attack'][at][2])
                else:
                    row.append(0.0)
            matrix.append(row)
            model_names.append(r['name'])

        if not matrix:
            return

        matrix = np.array(matrix)

        fig, ax = plt.subplots(figsize=(10, 6))
        im = ax.imshow(matrix, cmap=plt.cm.Greys, aspect='auto', vmin=0, vmax=100)

        ax.set_xticks(range(len(attacks)))
        ax.set_xticklabels([self.attack_labels.get(a, a) for a in attacks],
                           rotation=30, ha='right')
        ax.set_yticks(range(len(model_names)))
        ax.set_yticklabels(model_names)

        for i in range(len(model_names)):
            for j in range(len(attacks)):
                val = matrix[i, j]
                color = 'white' if val > 60 else 'black'
                ax.text(j, i, f'{val:.1f}%', ha='center', va='center',
                        fontsize=9, fontweight='bold', color=color)

        plt.colorbar(im, ax=ax, label='Доля обнаружения (%)', shrink=0.8)
        ax.set_title('Полнота обнаружения по типам атак и моделям')
        self._save(fig, '04_attack_heatmap.png')

    # ── 5. Summary bar chart ───────────────────────────

    def plot_summary_bars(self, results_list):
        main_models = [
            'Isolation Forest', 'One-Class SVM', 'Autoencoder', 'VAE',
            'Random Forest', 'Gradient Boosting', 'Cascade (v6)', 'Meta-Ensemble (v6)'
        ]

        names, auc_rocs, auc_prs, f1s = [], [], [], []
        for r in results_list:
            if r['name'] in main_models:
                names.append(r['name'])
                auc_rocs.append(r['auc_roc'])
                auc_prs.append(r['auc_pr'])
                f1s.append(r['f1'])

        if not names:
            return

        x = np.arange(len(names))
        w = 0.25

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.bar(x - w, auc_rocs, w, label='AUC-ROC', color='#2f2f2f', alpha=0.9, hatch='////', edgecolor='#111111')
        ax.bar(x, auc_prs, w, label='AUC-PR', color='#7a7a7a', alpha=0.9, hatch='....', edgecolor='#111111')
        ax.bar(x + w, f1s, w, label='F1', color='#b0b0b0', alpha=0.95, hatch='xxxx', edgecolor='#2f2f2f')

        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=35, ha='right')
        ax.set_ylabel('Значение метрики')
        ax.set_title('Сравнение моделей по основным метрикам')
        ax.set_ylim([0, 1.08])
        ax.legend(loc='lower right')

        for i in range(len(names)):
            ax.text(i + w, f1s[i] + 0.01, f'{f1s[i]:.3f}',
                    ha='center', va='bottom', fontsize=7, fontweight='bold')
        self._save(fig, '05_summary_bars.png')

    # ── 6. Feature importance ──────────────────────────

    def plot_feature_importance(self, feature_names, importances, ms_features, top_n=20):
        if len(importances) == 0:
            return

        idx = np.argsort(importances)[-top_n:]
        top_names = [feature_names[i] for i in idx]
        top_imps = importances[idx]
        is_ms = [n in ms_features for n in top_names]

        fig, ax = plt.subplots(figsize=(9, 7))
        colors = ['#2f2f2f' if ms else '#a6a6a6' for ms in is_ms]
        ax.barh(range(len(top_names)), top_imps, color=colors, alpha=0.85, edgecolor='#1f1f1f')

        ax.set_yticks(range(len(top_names)))
        ax.set_yticklabels(top_names)
        ax.set_xlabel('Важность признака (Gini importance)')
        ax.set_title(f'Топ-{len(top_names)} признаков по важности (Random Forest)')

        for i, v in enumerate(top_imps):
            ax.text(v + 0.001, i, f'{100*v:.1f}%', va='center', fontsize=7)

        legend_elements = [
            Patch(facecolor='#2f2f2f', alpha=0.85, label='Мастер-слейв признаки'),
            Patch(facecolor='#a6a6a6', alpha=0.85, label='Прочие признаки'),
        ]
        ax.legend(handles=legend_elements, loc='lower right')
        self._save(fig, '06_feature_importance.png')

    # ── 7. Ablation study ──────────────────────────────

    def plot_ablation(self, ablation_results):
        if not ablation_results:
            return

        configs = [a['config'] for a in ablation_results]
        f1s = [a['f1'] for a in ablation_results]
        comp_rates = [a['compromise_rate'] for a in ablation_results]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        colors = ['#1f1f1f', '#4a4a4a', '#757575', '#9a9a9a', '#c0c0c0']
        x = np.arange(len(configs))

        ax1.bar(x, f1s, color=colors[:len(configs)], alpha=0.85, edgecolor='#111111')
        ax1.set_xticks(x)
        ax1.set_xticklabels(configs, rotation=35, ha='right')
        ax1.set_ylabel('F1-мера')
        ax1.set_title('Абляционное исследование: F1-мера')
        ax1.set_ylim([0, 1.1])
        for i, v in enumerate(f1s):
            ax1.text(i, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold', fontsize=9)

        ax2.bar(x, comp_rates, color=colors[:len(configs)], alpha=0.85, edgecolor='#111111')
        ax2.set_xticks(x)
        ax2.set_xticklabels(configs, rotation=35, ha='right')
        ax2.set_ylabel('Доля обнаружения (%)')
        ax2.set_title('Абляционное исследование: компрометация слейва')
        ax2.set_ylim([0, 110])
        for i, v in enumerate(comp_rates):
            ax2.text(i, v + 1.5, f'{v:.1f}%', ha='center', fontweight='bold', fontsize=9)

        fig.tight_layout()
        self._save(fig, '07_ablation.png')

    # ── 8. Score distributions ─────────────────────────

    def plot_score_distributions(self, results_list, y_test, attack_types):
        models_to_plot = ['VAE', 'Random Forest', 'Cascade (v6)']
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

        for ax, mname in zip(axes, models_to_plot):
            r = next((r for r in results_list if r['name'] == mname), None)
            if r is None:
                continue

            scores = r['scores']
            th = r['threshold']

            normal_scores = scores[y_test == 0]
            ax.hist(
                normal_scores,
                bins=80,
                alpha=0.65,
                color='#d9d9d9',
                edgecolor='#2f2f2f',
                label='Норма',
                density=True
            )

            for at in self.attack_colors.keys():
                mask = attack_types == at
                if mask.sum() == 0:
                    continue
                style = self._attack_style(at)
                self._hist_step(
                    ax,
                    scores[mask],
                    bins=40,
                    color=style['color'],
                    linestyle=style['linestyle'],
                    label=self.attack_labels.get(at, at),
                    density=True,
                    linewidth=1.8
                )

            ax.axvline(th, color='#111111', linestyle='--', linewidth=1.5,
                       label=f'Порог ({th:.3f})')
            ax.set_title(mname)
            ax.set_xlabel('Скор аномальности')
            ax.set_ylabel('Плотность')
            ax.legend(fontsize=6, loc='upper right')

        fig.suptitle('Распределения скоров аномальности по классам', fontsize=13)
        fig.tight_layout()
        self._save(fig, '08_score_distributions.png')

    # ── 9. Confusion matrices ──────────────────────────

    def plot_confusion_matrices(self, results_list, y_test):
        models = ['VAE', 'Random Forest', 'Cascade (v6)', 'Meta-Ensemble (v6)']
        fig, axes = plt.subplots(1, 4, figsize=(16, 3.5))

        for ax, mname in zip(axes, models):
            r = next((r for r in results_list if r['name'] == mname), None)
            if r is None:
                continue
            cm = confusion_matrix(y_test, r['y_pred'], labels=[0, 1])
            ax.imshow(cm, cmap='Greys', aspect='auto')

            for i in range(2):
                for j in range(2):
                    color = 'white' if cm[i, j] > cm.max() * 0.5 else 'black'
                    ax.text(j, i, f'{cm[i,j]:,}', ha='center', va='center',
                            fontsize=11, fontweight='bold', color=color)

            ax.set_xticks([0, 1])
            ax.set_yticks([0, 1])
            ax.set_xticklabels(['Норма', 'Аномалия'])
            ax.set_yticklabels(['Норма', 'Аномалия'])
            ax.set_xlabel('Предсказание')
            ax.set_ylabel('Истина')
            ax.set_title(f'{mname}\nF1={r["f1"]:.3f}')

        fig.suptitle('Матрицы ошибок', fontsize=13, y=1.02)
        fig.tight_layout()
        self._save(fig, '09_confusion_matrices.png')

    # ── 10. Compromise detection comparison ────────────

    def plot_compromise_comparison(self, results_list, attack_types):
        comp_mask = attack_types == 'slave_compromise'
        comp_total = comp_mask.sum()
        if comp_total == 0:
            return

        main_models = [
            'Isolation Forest', 'One-Class SVM', 'Autoencoder', 'VAE',
            'Random Forest', 'Gradient Boosting', 'Cascade (v6)', 'Meta-Ensemble (v6)'
        ]

        names, rates = [], []
        for r in results_list:
            if r['name'] not in main_models:
                continue
            detected = r['y_pred'][comp_mask].sum()
            rate = 100 * detected / comp_total
            names.append(r['name'])
            rates.append(rate)

        fig, ax = plt.subplots(figsize=(10, 5))
        colors = [self.model_colors.get(n, '#6B7280') for n in names]
        ax.bar(range(len(names)), rates, color=colors, alpha=0.9, edgecolor='#2f2f2f')

        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=35, ha='right')
        ax.set_ylabel('Доля обнаружения (%)')
        ax.set_title(f'Обнаружение компрометации слейва ({comp_total} записей)')
        ax.set_ylim([0, 110])
        ax.axhline(100, color='#666666', linestyle=':', alpha=0.7)

        for i, v in enumerate(rates):
            ax.text(i, v + 1.5, f'{v:.1f}%', ha='center',
                    fontweight='bold', fontsize=9)
        self._save(fig, '10_compromise_detection.png')

    # ── 11. Training curves ────────────────────────────

    def plot_training_curves(self, trainers_dict):
        if not trainers_dict:
            return

        n = len(trainers_dict)
        fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
        if n == 1:
            axes = [axes]

        for ax, (name, trainer) in zip(axes, trainers_dict.items()):
            epochs = range(1, len(trainer.train_losses) + 1)
            ax.plot(
                epochs, trainer.train_losses,
                color='#1f1f1f', linewidth=1.5,
                linestyle='-', label='Train'
            )
            if trainer.val_losses:
                val_epochs = range(1, len(trainer.val_losses) + 1)
                ax.plot(
                    val_epochs, trainer.val_losses,
                    color='#7a7a7a', linewidth=1.5,
                    linestyle='--', label='Validation'
                )
            ax.set_xlabel('Эпоха')
            ax.set_ylabel('Loss')
            ax.set_title(name)
            ax.legend()
            ax.set_yscale('log')

        fig.suptitle('Кривые обучения нейронных моделей', fontsize=13)
        fig.tight_layout()
        self._save(fig, '11_training_curves.png')

    # ── 12. Ensemble specialized detectors ─────────────

    def plot_ensemble_results(self, ens_results, y_test, attack_types):
        if not ens_results:
            return

        attacks = ['sms_intercept', 'location_track', 'signaling_dos',
                   'irsf', 'slave_compromise']
        detectors = ['map_profile', 'volume_timing', 'control_unit',
                     'international', 'baseline_deviation', 'compromise_indicators']

        fig, ax = plt.subplots(figsize=(10, 6))

        det_attack_matrix = []
        det_names = []
        for r in ens_results:
            dname = r['name'].replace('Ensemble:', '')
            if dname not in detectors:
                continue
            row = []
            for at in attacks:
                if at in r['per_attack']:
                    row.append(r['per_attack'][at][2])
                else:
                    row.append(0.0)
            det_attack_matrix.append(row)
            det_names.append(dname)

        if not det_attack_matrix:
            plt.close(fig)
            return

        matrix = np.array(det_attack_matrix)
        im = ax.imshow(matrix, cmap=plt.cm.Greys, aspect='auto', vmin=0, vmax=100)

        ax.set_xticks(range(len(attacks)))
        ax.set_xticklabels([self.attack_labels.get(a, a) for a in attacks],
                           rotation=30, ha='right')
        ax.set_yticks(range(len(det_names)))
        ax.set_yticklabels(det_names)

        for i in range(len(det_names)):
            for j in range(len(attacks)):
                v = matrix[i, j]
                color = 'white' if v > 60 else 'black'
                ax.text(j, i, f'{v:.0f}%', ha='center', va='center',
                        fontsize=9, fontweight='bold', color=color)

        plt.colorbar(im, ax=ax, label='Доля обнаружения (%)', shrink=0.8)
        ax.set_title('Специализация детекторов ансамбля по типам атак')
        self._save(fig, '12_ensemble_specialization.png')

    # ── 13. Dataset overview ───────────────────────────

    def plot_dataset_overview(self, df):
        fig = plt.figure(figsize=(14, 8))
        gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.3)

        # (a) Attack distribution
        ax1 = fig.add_subplot(gs[0, 0])
        atk_counts = df['attack_type'].value_counts()
        labels = [self.attack_labels.get(a, a) if a != 'none' else 'Нормальный\nтрафик'
                  for a in atk_counts.index]
        colors_bar = ['#d9d9d9' if a == 'none' else self.attack_colors.get(a, '#7a7a7a')
                      for a in atk_counts.index]

        bars = ax1.bar(
            range(len(atk_counts)), atk_counts.values,
            color=colors_bar, edgecolor='#374151', linewidth=0.5, alpha=0.9
        )
        ax1.set_xticks(range(len(atk_counts)))
        ax1.set_xticklabels(labels, rotation=30, ha='right', fontsize=8)
        ax1.set_ylabel('Количество записей')
        ax1.set_title('Распределение классов')

        total = atk_counts.sum()
        for bar, val in zip(bars, atk_counts.values):
            pct = val / total * 100
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + total * 0.005,
                f'{val}\n({pct:.1f}%)',
                ha='center', va='bottom', fontsize=7
            )
        ax1.set_ylim(0, atk_counts.max() * 1.15)

        # (b) Attack timeline
        ax2 = fig.add_subplot(gs[0, 1])
        for at, color in self.attack_colors.items():
            mask = df['attack_type'] == at
            if mask.sum() == 0:
                continue
            sub = df[mask]
            ax2.scatter(
                sub['interval'], sub['node_id'],
                c=color, s=6, alpha=0.55,
                marker=self.attack_markers.get(at, 'o'),
                label=self.attack_labels.get(at, at)
            )
        ax2.set_xlabel('Интервал')
        ax2.set_ylabel('ID узла')
        ax2.set_title('Временна́я карта атак')
        ax2.legend(fontsize=7, loc='upper right', markerscale=2)

        # (c) Feature contrast
        ax3 = fig.add_subplot(gs[1, 0])
        contrast_features = [
            'var_obs_inbound_outbound_ratio',
            'dev_io_ratio_vs_zone_median',
            'integrity_fail_rate',
            'cu_response_time_jitter',
            'autocorr_obs_total_messages',
        ]
        existing_cf = [f for f in contrast_features if f in df.columns]
        norm_means = [df.loc[df['is_anomaly'] == 0, f].mean() for f in existing_cf]
        comp_means = [df.loc[df['attack_type'] == 'slave_compromise', f].mean()
                      for f in existing_cf]

        x = np.arange(len(existing_cf))
        ax3.barh(x - 0.2, norm_means, 0.35, color='#c9c9c9', alpha=0.9,
                 edgecolor='#2f2f2f', label='Нормальный')
        ax3.barh(x + 0.2, comp_means, 0.35, color='#4a4a4a', alpha=0.9,
                 edgecolor='#2f2f2f', label='Компрометация')
        ax3.set_yticks(x)
        ax3.set_yticklabels([f.replace('_', '\n', 1) for f in existing_cf], fontsize=7)
        ax3.set_xlabel('Среднее значение')
        ax3.set_title('Контраст признаков: норма vs компрометация')
        ax3.legend(fontsize=8)

        # (d) CU delivery statistics
        ax4 = fig.add_subplot(gs[1, 1])
        cu_by_attack = df.groupby('attack_type')['cu_delivered'].mean()
        atk_order = ['none', 'sms_intercept', 'location_track',
                     'signaling_dos', 'irsf', 'slave_compromise']
        existing_atk = [a for a in atk_order if a in cu_by_attack.index]
        vals = [cu_by_attack[a] * 100 for a in existing_atk]
        colors_cu = ['#d9d9d9' if a == 'none' else self.attack_colors.get(a, '#6B7280')
                     for a in existing_atk]
        labels_cu = [self.attack_labels.get(a, a) if a != 'none' else 'Норма'
                     for a in existing_atk]

        ax4.bar(
            range(len(existing_atk)), vals,
            color=colors_cu, alpha=0.9,
            edgecolor='#374151', linewidth=0.5
        )
        ax4.set_xticks(range(len(existing_atk)))
        ax4.set_xticklabels(labels_cu, rotation=30, ha='right', fontsize=8)
        ax4.set_ylabel('Доставка КЕ (%)')
        ax4.set_title('Доставка контрольных единиц по типам')
        ax4.set_ylim([0, 105])
        for i, v in enumerate(vals):
            ax4.text(i, v + 1, f'{v:.1f}%', ha='center', fontsize=8)

        fig.suptitle('Обзор набора данных SS7', fontsize=14, y=1.01)
        self._save(fig, '13_dataset_overview.png')

    # ── 14. Cascade analysis ──────────────────────────

    def plot_cascade_analysis(self, stage1_scores, stage2_scores,
                              combined_scores, y_test, attack_types):
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
        comp_mask = attack_types == 'slave_compromise'
        norm_mask = y_test == 0
        other_atk = (y_test == 1) & ~comp_mask

        for ax, scores, title in zip(
            axes,
            [stage1_scores, stage2_scores, combined_scores],
            ['Уровень 1 (общий RF)', 'Уровень 2 (компрометация)', 'Каскад (OR)']
        ):
            ax.hist(
                scores[norm_mask], bins=60, alpha=0.55, color='#d0d0d0',
                edgecolor='#2f2f2f', label='Норма', density=True
            )

            if other_atk.sum() > 0:
                self._hist_step(
                    ax, scores[other_atk], bins=30,
                    color='#7a7a7a', linestyle='--',
                    label='Прочие атаки', density=True, linewidth=1.6
                )

            if comp_mask.sum() > 0:
                self._hist_step(
                    ax, scores[comp_mask], bins=30,
                    color='#111111', linestyle='-',
                    label='Компрометация', density=True, linewidth=2.0
                )

            ax.axvline(0.5, color='#111111', linestyle=':', linewidth=1.5,
                       label='Порог 0.5')
            ax.set_title(title)
            ax.set_xlabel('P(аномалия)')
            ax.legend(fontsize=7)

        fig.suptitle('Анализ каскадного детектора', fontsize=13)
        fig.tight_layout()
        self._save(fig, '14_cascade_analysis.png')


# ══════════════════════════════════════════════════════════
#  VISUALIZATION MODULE
# ══════════════════════════════════════════════════════════

# class ResultsVisualizer:
#     """Generates all publication-quality plots."""

#     def __init__(self, plots_dir: str):
#         self.plots_dir = plots_dir
#         os.makedirs(plots_dir, exist_ok=True)

#     def _save(self, fig, name):
#         path = os.path.join(self.plots_dir, name)
#         fig.savefig(path, bbox_inches='tight', facecolor='white')
#         plt.close(fig)
#         print(f"  Saved: {path}")

#     # ── 1. ROC Curves ──────────────────────────────────

#     def plot_roc_curves(self, results_list, y_test):
#         """ROC curves for main models."""
#         fig, ax = plt.subplots(figsize=(8, 6))

#         main_models = [
#             'Isolation Forest', 'One-Class SVM', 'Autoencoder', 'VAE',
#             'Random Forest', 'Gradient Boosting', 'Cascade (v6)', 'Meta-Ensemble (v6)'
#         ]

#         for r in results_list:
#             if r['name'] not in main_models:
#                 continue
#             fpr_arr, tpr_arr, _ = roc_curve(y_test, r['scores'])
#             color = MODEL_COLORS.get(r['name'], '#6B7280')
#             ax.plot(fpr_arr, tpr_arr, color=color, linewidth=1.8,
#                     label=f"{r['name']} (AUC={r['auc_roc']:.3f})")

#         ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, linewidth=1)
#         ax.set_xlabel('Доля ложных тревог (FPR)')
#         ax.set_ylabel('Доля истинных обнаружений (TPR)')
#         ax.set_title('ROC-кривые моделей обнаружения аномалий')
#         ax.legend(loc='lower right', framealpha=0.9)
#         ax.set_xlim([-0.01, 1.01])
#         ax.set_ylim([-0.01, 1.01])
#         self._save(fig, '01_roc_curves.png')

#     # ── 2. ROC zoom (low FPR) ──────────────────────────

#     def plot_roc_zoom(self, results_list, y_test):
#         """ROC curves zoomed to FPR < 0.1."""
#         fig, ax = plt.subplots(figsize=(8, 6))

#         main_models = [
#             'Autoencoder', 'VAE', 'Random Forest',
#             'Gradient Boosting', 'Cascade (v6)', 'Meta-Ensemble (v6)'
#         ]

#         for r in results_list:
#             if r['name'] not in main_models:
#                 continue
#             fpr_arr, tpr_arr, _ = roc_curve(y_test, r['scores'])
#             color = MODEL_COLORS.get(r['name'], '#6B7280')
#             ax.plot(fpr_arr, tpr_arr, color=color, linewidth=2,
#                     label=f"{r['name']} (AUC={r['auc_roc']:.4f})")

#         ax.set_xlabel('Доля ложных тревог (FPR)')
#         ax.set_ylabel('Доля истинных обнаружений (TPR)')
#         ax.set_title('ROC-кривые (увеличенная область FPR < 10%)')
#         ax.set_xlim([-0.002, 0.10])
#         ax.set_ylim([0.85, 1.005])
#         ax.legend(loc='lower right', framealpha=0.9)
#         self._save(fig, '02_roc_zoom.png')

#     # ── 3. Precision-Recall Curves ─────────────────────

#     def plot_pr_curves(self, results_list, y_test):
#         """Precision-Recall curves."""
#         fig, ax = plt.subplots(figsize=(8, 6))

#         main_models = [
#             'Isolation Forest', 'One-Class SVM', 'Autoencoder', 'VAE',
#             'Random Forest', 'Gradient Boosting', 'Cascade (v6)', 'Meta-Ensemble (v6)'
#         ]

#         for r in results_list:
#             if r['name'] not in main_models:
#                 continue
#             prec, rec, _ = precision_recall_curve(y_test, r['scores'])
#             color = MODEL_COLORS.get(r['name'], '#6B7280')
#             ax.plot(rec, prec, color=color, linewidth=1.8,
#                     label=f"{r['name']} (AP={r['auc_pr']:.3f})")

#         baseline = y_test.mean()
#         ax.axhline(y=baseline, color='k', linestyle='--', alpha=0.3,
#                    label=f'Базовый уровень ({baseline:.3f})')
#         ax.set_xlabel('Полнота (Recall)')
#         ax.set_ylabel('Точность (Precision)')
#         ax.set_title('Кривые Precision-Recall')
#         ax.legend(loc='lower left', framealpha=0.9)
#         ax.set_xlim([-0.01, 1.01])
#         ax.set_ylim([-0.01, 1.05])
#         self._save(fig, '03_precision_recall.png')

#     # ── 4. Per-attack detection heatmap ────────────────

#     def plot_attack_detection_heatmap(self, results_list):
#         """Heatmap of detection rates per model per attack type."""
#         main_models = [
#             'Isolation Forest', 'One-Class SVM', 'Autoencoder', 'VAE',
#             'Random Forest', 'Gradient Boosting', 'Cascade (v6)', 'Meta-Ensemble (v6)'
#         ]
#         attacks = ['sms_intercept', 'location_track', 'signaling_dos',
#                    'irsf', 'slave_compromise']

#         matrix = []
#         model_names = []
#         for r in results_list:
#             if r['name'] not in main_models:
#                 continue
#             row = []
#             for at in attacks:
#                 if at in r['per_attack']:
#                     row.append(r['per_attack'][at][2])
#                 else:
#                     row.append(0.0)
#             matrix.append(row)
#             model_names.append(r['name'])

#         matrix = np.array(matrix)

#         fig, ax = plt.subplots(figsize=(10, 6))
#         cmap = LinearSegmentedColormap.from_list('rg',
#             ['#FEE2E2', '#FEF3C7', '#D1FAE5', '#059669'], N=256)
#         im = ax.imshow(matrix, cmap=cmap, aspect='auto', vmin=0, vmax=100)

#         ax.set_xticks(range(len(attacks)))
#         ax.set_xticklabels([ATTACK_LABELS.get(a, a) for a in attacks],
#                            rotation=30, ha='right')
#         ax.set_yticks(range(len(model_names)))
#         ax.set_yticklabels(model_names)

#         for i in range(len(model_names)):
#             for j in range(len(attacks)):
#                 val = matrix[i, j]
#                 color = 'white' if val > 60 else 'black'
#                 ax.text(j, i, f'{val:.1f}%', ha='center', va='center',
#                         fontsize=9, fontweight='bold', color=color)

#         plt.colorbar(im, ax=ax, label='Доля обнаружения (%)', shrink=0.8)
#         ax.set_title('Полнота обнаружения по типам атак и моделям')
#         self._save(fig, '04_attack_heatmap.png')

#     # ── 5. Summary bar chart ───────────────────────────

#     def plot_summary_bars(self, results_list):
#         """Grouped bar chart: AUC-ROC, AUC-PR, F1 for main models."""
#         main_models = [
#             'Isolation Forest', 'One-Class SVM', 'Autoencoder', 'VAE',
#             'Random Forest', 'Gradient Boosting', 'Cascade (v6)', 'Meta-Ensemble (v6)'
#         ]

#         names, auc_rocs, auc_prs, f1s = [], [], [], []
#         for r in results_list:
#             if r['name'] in main_models:
#                 names.append(r['name'])
#                 auc_rocs.append(r['auc_roc'])
#                 auc_prs.append(r['auc_pr'])
#                 f1s.append(r['f1'])

#         x = np.arange(len(names))
#         w = 0.25

#         fig, ax = plt.subplots(figsize=(12, 5))
#         ax.bar(x - w, auc_rocs, w, label='AUC-ROC', color='#3B82F6', alpha=0.85)
#         ax.bar(x, auc_prs, w, label='AUC-PR', color='#8B5CF6', alpha=0.85)
#         ax.bar(x + w, f1s, w, label='F1', color='#059669', alpha=0.85)

#         ax.set_xticks(x)
#         ax.set_xticklabels(names, rotation=35, ha='right')
#         ax.set_ylabel('Значение метрики')
#         ax.set_title('Сравнение моделей по основным метрикам')
#         ax.set_ylim([0, 1.08])
#         ax.legend(loc='lower right')

#         for i in range(len(names)):
#             ax.text(i + w, f1s[i] + 0.01, f'{f1s[i]:.3f}',
#                     ha='center', va='bottom', fontsize=7, fontweight='bold')
#         self._save(fig, '05_summary_bars.png')

#     # ── 6. Feature importance ──────────────────────────

#     def plot_feature_importance(self, feature_names, importances, ms_features,
#                                 top_n=20):
#         """Horizontal bar chart of RF feature importances."""
#         idx = np.argsort(importances)[-top_n:]
#         top_names = [feature_names[i] for i in idx]
#         top_imps = importances[idx]
#         is_ms = [n in ms_features for n in top_names]

#         fig, ax = plt.subplots(figsize=(9, 7))
#         colors = ['#059669' if ms else '#6B7280' for ms in is_ms]
#         bars = ax.barh(range(top_n), top_imps, color=colors, alpha=0.85)

#         ax.set_yticks(range(top_n))
#         ax.set_yticklabels(top_names)
#         ax.set_xlabel('Важность признака (Gini importance)')
#         ax.set_title(f'Топ-{top_n} признаков по важности (Random Forest)')

#         for i, v in enumerate(top_imps):
#             ax.text(v + 0.001, i, f'{100*v:.1f}%', va='center', fontsize=7)

#         legend_elements = [
#             Patch(facecolor='#059669', alpha=0.85,
#                   label='Мастер-слейв признаки'),
#             Patch(facecolor='#6B7280', alpha=0.85,
#                   label='Прочие признаки'),
#         ]
#         ax.legend(handles=legend_elements, loc='lower right')
#         self._save(fig, '06_feature_importance.png')

#     # ── 7. Ablation study ──────────────────────────────

#     def plot_ablation(self, ablation_results):
#         """Bar chart for ablation study results."""
#         configs = [a['config'] for a in ablation_results]
#         f1s = [a['f1'] for a in ablation_results]
#         comp_rates = [a['compromise_rate'] for a in ablation_results]

#         fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

#         colors = ['#059669', '#EF4444', '#F59E0B', '#3B82F6', '#8B5CF6']
#         x = np.arange(len(configs))

#         bars1 = ax1.bar(x, f1s, color=colors[:len(configs)], alpha=0.85)
#         ax1.set_xticks(x)
#         ax1.set_xticklabels(configs, rotation=35, ha='right')
#         ax1.set_ylabel('F1-мера')
#         ax1.set_title('Абляционное исследование: F1-мера')
#         ax1.set_ylim([0, 1.1])
#         for i, v in enumerate(f1s):
#             ax1.text(i, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold', fontsize=9)

#         bars2 = ax2.bar(x, comp_rates, color=colors[:len(configs)], alpha=0.85)
#         ax2.set_xticks(x)
#         ax2.set_xticklabels(configs, rotation=35, ha='right')
#         ax2.set_ylabel('Доля обнаружения (%)')
#         ax2.set_title('Абляционное исследование: компрометация слейва')
#         ax2.set_ylim([0, 110])
#         for i, v in enumerate(comp_rates):
#             ax2.text(i, v + 1.5, f'{v:.1f}%', ha='center', fontweight='bold', fontsize=9)

#         fig.tight_layout()
#         self._save(fig, '07_ablation.png')

#     # ── 8. Score distributions ─────────────────────────

#     def plot_score_distributions(self, results_list, y_test, attack_types):
#         """Score distribution per class for best models."""
#         models_to_plot = ['VAE', 'Random Forest', 'Cascade (v6)']

#         fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

#         for ax, mname in zip(axes, models_to_plot):
#             r = next((r for r in results_list if r['name'] == mname), None)
#             if r is None:
#                 continue
#             scores = r['scores']
#             th = r['threshold']

#             # Normal
#             normal_scores = scores[y_test == 0]
#             ax.hist(normal_scores, bins=80, alpha=0.5, color='#3B82F6',
#                     label='Норма', density=True)

#             # Per attack
#             for at, color in ATTACK_COLORS.items():
#                 mask = attack_types == at
#                 if mask.sum() == 0:
#                     continue
#                 ax.hist(scores[mask], bins=40, alpha=0.6, color=color,
#                         label=ATTACK_LABELS.get(at, at), density=True)

#             ax.axvline(th, color='red', linestyle='--', linewidth=1.5,
#                        label=f'Порог ({th:.3f})')
#             ax.set_title(mname)
#             ax.set_xlabel('Скор аномальности')
#             ax.set_ylabel('Плотность')
#             ax.legend(fontsize=6, loc='upper right')

#         fig.suptitle('Распределения скоров аномальности по классам', fontsize=13)
#         fig.tight_layout()
#         self._save(fig, '08_score_distributions.png')

#     # ── 9. Confusion matrices ──────────────────────────

#     def plot_confusion_matrices(self, results_list, y_test):
#         """Confusion matrices for top models."""
#         models = ['VAE', 'Random Forest', 'Cascade (v6)', 'Meta-Ensemble (v6)']
#         fig, axes = plt.subplots(1, 4, figsize=(16, 3.5))

#         for ax, mname in zip(axes, models):
#             r = next((r for r in results_list if r['name'] == mname), None)
#             if r is None:
#                 continue
#             cm = confusion_matrix(y_test, r['y_pred'], labels=[0, 1])
#             im = ax.imshow(cm, cmap='Blues', aspect='auto')

#             for i in range(2):
#                 for j in range(2):
#                     color = 'white' if cm[i, j] > cm.max() * 0.5 else 'black'
#                     ax.text(j, i, f'{cm[i,j]:,}', ha='center', va='center',
#                             fontsize=11, fontweight='bold', color=color)

#             ax.set_xticks([0, 1])
#             ax.set_yticks([0, 1])
#             ax.set_xticklabels(['Норма', 'Аномалия'])
#             ax.set_yticklabels(['Норма', 'Аномалия'])
#             ax.set_xlabel('Предсказание')
#             ax.set_ylabel('Истина')
#             ax.set_title(f'{mname}\nF1={r["f1"]:.3f}')

#         fig.suptitle('Матрицы ошибок', fontsize=13, y=1.02)
#         fig.tight_layout()
#         self._save(fig, '09_confusion_matrices.png')

#     # ── 10. Compromise detection comparison ────────────

#     def plot_compromise_comparison(self, results_list, attack_types):
#         """Bar chart: compromise detection rate across all models."""
#         comp_mask = attack_types == 'slave_compromise'
#         comp_total = comp_mask.sum()
#         if comp_total == 0:
#             return

#         main_models = [
#             'Isolation Forest', 'One-Class SVM', 'Autoencoder', 'VAE',
#             'Random Forest', 'Gradient Boosting', 'Cascade (v6)', 'Meta-Ensemble (v6)'
#         ]

#         names, rates = [], []
#         for r in results_list:
#             if r['name'] not in main_models:
#                 continue
#             detected = r['y_pred'][comp_mask].sum()
#             rate = 100 * detected / comp_total
#             names.append(r['name'])
#             rates.append(rate)

#         fig, ax = plt.subplots(figsize=(10, 5))
#         colors = [MODEL_COLORS.get(n, '#6B7280') for n in names]
#         bars = ax.bar(range(len(names)), rates, color=colors, alpha=0.85)

#         ax.set_xticks(range(len(names)))
#         ax.set_xticklabels(names, rotation=35, ha='right')
#         ax.set_ylabel('Доля обнаружения (%)')
#         ax.set_title(f'Обнаружение компрометации слейва ({comp_total} записей)')
#         ax.set_ylim([0, 110])
#         ax.axhline(100, color='green', linestyle=':', alpha=0.5)

#         for i, v in enumerate(rates):
#             ax.text(i, v + 1.5, f'{v:.1f}%', ha='center',
#                     fontweight='bold', fontsize=9)
#         self._save(fig, '10_compromise_detection.png')

#     # ── 11. Training curves ────────────────────────────

#     def plot_training_curves(self, trainers_dict):
#         """Training and validation loss curves for neural models."""
#         n = len(trainers_dict)
#         fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
#         if n == 1:
#             axes = [axes]

#         for ax, (name, trainer) in zip(axes, trainers_dict.items()):
#             epochs = range(1, len(trainer.train_losses) + 1)
#             ax.plot(epochs, trainer.train_losses, color='#3B82F6',
#                     linewidth=1.5, label='Train')
#             if trainer.val_losses:
#                 val_epochs = range(1, len(trainer.val_losses) + 1)
#                 ax.plot(val_epochs, trainer.val_losses, color='#EF4444',
#                         linewidth=1.5, label='Validation')
#             ax.set_xlabel('Эпоха')
#             ax.set_ylabel('Loss')
#             ax.set_title(name)
#             ax.legend()
#             ax.set_yscale('log')

#         fig.suptitle('Кривые обучения нейронных моделей', fontsize=13)
#         fig.tight_layout()
#         self._save(fig, '11_training_curves.png')

#     # ── 12. Ensemble specialized detectors ─────────────

#     def plot_ensemble_results(self, ens_results, y_test, attack_types):
#         """Spider/radar chart for specialized detectors."""
#         attacks = ['sms_intercept', 'location_track', 'signaling_dos',
#                    'irsf', 'slave_compromise']
#         detectors = ['map_profile', 'volume_timing', 'control_unit',
#                      'international', 'baseline_deviation', 'compromise_indicators']

#         existing_dets = [d for d in detectors if d in [r['name'].replace('Ensemble:', '')
#                          for r in ens_results]]

#         fig, ax = plt.subplots(figsize=(10, 6))

#         det_attack_matrix = []
#         det_names = []
#         for r in ens_results:
#             dname = r['name'].replace('Ensemble:', '')
#             if dname not in detectors:
#                 continue
#             row = []
#             for at in attacks:
#                 if at in r['per_attack']:
#                     row.append(r['per_attack'][at][2])
#                 else:
#                     row.append(0.0)
#             det_attack_matrix.append(row)
#             det_names.append(dname)

#         if not det_attack_matrix:
#             return

#         matrix = np.array(det_attack_matrix)

#         cmap = LinearSegmentedColormap.from_list('rg',
#             ['#FEE2E2', '#FEF3C7', '#D1FAE5', '#059669'], N=256)
#         im = ax.imshow(matrix, cmap=cmap, aspect='auto', vmin=0, vmax=100)

#         ax.set_xticks(range(len(attacks)))
#         ax.set_xticklabels([ATTACK_LABELS.get(a, a) for a in attacks],
#                            rotation=30, ha='right')
#         ax.set_yticks(range(len(det_names)))
#         ax.set_yticklabels(det_names)

#         for i in range(len(det_names)):
#             for j in range(len(attacks)):
#                 v = matrix[i, j]
#                 color = 'white' if v > 60 else 'black'
#                 ax.text(j, i, f'{v:.0f}%', ha='center', va='center',
#                         fontsize=9, fontweight='bold', color=color)

#         plt.colorbar(im, ax=ax, label='Доля обнаружения (%)', shrink=0.8)
#         ax.set_title('Специализация детекторов ансамбля по типам атак')
#         self._save(fig, '12_ensemble_specialization.png')

#     # ── 13. Dataset overview ───────────────────────────

#     def plot_dataset_overview(self, df):
#         """Overview of the dataset: attack distribution, timeline."""
#         fig = plt.figure(figsize=(14, 8))
#         gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.3)

#         # (a) Attack distribution — histogram (bar chart)
#         ax1 = fig.add_subplot(gs[0, 0])
#         atk_counts = df['attack_type'].value_counts()
#         labels = [ATTACK_LABELS.get(a, a) if a != 'none' else 'Нормальный\nтрафик'
#                 for a in atk_counts.index]
#         colors_bar = ['#E5E7EB' if a == 'none' else ATTACK_COLORS.get(a, '#6B7280')
#                     for a in atk_counts.index]

#         bars = ax1.bar(range(len(atk_counts)), atk_counts.values,
#                     color=colors_bar, edgecolor='#374151', linewidth=0.5,
#                     alpha=0.85)
#         ax1.set_xticks(range(len(atk_counts)))
#         ax1.set_xticklabels(labels, rotation=30, ha='right', fontsize=8)
#         ax1.set_ylabel('Количество записей')
#         ax1.set_title('Распределение классов')

#         # подписи значений и процентов над столбцами
#         total = atk_counts.sum()
#         for i, (bar, val) in enumerate(zip(bars, atk_counts.values)):
#             pct = val / total * 100
#             ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + total * 0.005,
#                     f'{val}\n({pct:.1f}%)', ha='center', va='bottom', fontsize=7)

#         # небольшой зазор сверху для подписей
#         ax1.set_ylim(0, atk_counts.max() * 1.15)

#         # (b) Attack timeline
#         ax2 = fig.add_subplot(gs[0, 1])
#         for at, color in ATTACK_COLORS.items():
#             mask = df['attack_type'] == at
#             if mask.sum() == 0:
#                 continue
#             sub = df[mask]
#             ax2.scatter(sub['interval'], sub['node_id'],
#                         c=color, s=1, alpha=0.5, label=ATTACK_LABELS.get(at, at))
#         ax2.set_xlabel('Интервал')
#         ax2.set_ylabel('ID узла')
#         ax2.set_title('Временна́я карта атак')
#         ax2.legend(fontsize=7, loc='upper right', markerscale=5)

#         # (c) Feature contrast: normal vs compromise
#         ax3 = fig.add_subplot(gs[1, 0])
#         contrast_features = [
#             'var_obs_inbound_outbound_ratio',
#             'dev_io_ratio_vs_zone_median',
#             'integrity_fail_rate',
#             'cu_response_time_jitter',
#             'autocorr_obs_total_messages',
#         ]
#         existing_cf = [f for f in contrast_features if f in df.columns]
#         norm_means = [df.loc[df['is_anomaly'] == 0, f].mean() for f in existing_cf]
#         comp_means = [df.loc[df['attack_type'] == 'slave_compromise', f].mean()
#                     for f in existing_cf]

#         x = np.arange(len(existing_cf))
#         ax3.barh(x - 0.2, norm_means, 0.35, color='#3B82F6', alpha=0.8,
#                 label='Нормальный')
#         ax3.barh(x + 0.2, comp_means, 0.35, color='#EF4444', alpha=0.8,
#                 label='Компрометация')
#         ax3.set_yticks(x)
#         ax3.set_yticklabels([f.replace('_', '\n', 1) for f in existing_cf], fontsize=7)
#         ax3.set_xlabel('Среднее значение')
#         ax3.set_title('Контраст признаков: норма vs компрометация')
#         ax3.legend(fontsize=8)

#         # (d) CU delivery statistics
#         ax4 = fig.add_subplot(gs[1, 1])
#         cu_by_attack = df.groupby('attack_type')['cu_delivered'].mean()
#         atk_order = ['none', 'sms_intercept', 'location_track',
#                     'signaling_dos', 'irsf', 'slave_compromise']
#         existing_atk = [a for a in atk_order if a in cu_by_attack.index]
#         vals = [cu_by_attack[a] * 100 for a in existing_atk]
#         colors_cu = ['#E5E7EB' if a == 'none' else ATTACK_COLORS.get(a, '#6B7280')
#                     for a in existing_atk]
#         labels_cu = [ATTACK_LABELS.get(a, a) if a != 'none' else 'Норма'
#                     for a in existing_atk]
#         ax4.bar(range(len(existing_atk)), vals, color=colors_cu, alpha=0.85,
#                 edgecolor='#374151', linewidth=0.5)
#         ax4.set_xticks(range(len(existing_atk)))
#         ax4.set_xticklabels(labels_cu, rotation=30, ha='right', fontsize=8)
#         ax4.set_ylabel('Доставка КЕ (%)')
#         ax4.set_title('Доставка контрольных единиц по типам')
#         ax4.set_ylim([0, 105])
#         for i, v in enumerate(vals):
#             ax4.text(i, v + 1, f'{v:.1f}%', ha='center', fontsize=8)

#         fig.suptitle('Обзор набора данных SS7', fontsize=14, y=1.01)
#         self._save(fig, '13_dataset_overview.png')


#     # ── 14. Cascade analysis ──────────────────────────

#     def plot_cascade_analysis(self, stage1_scores, stage2_scores,
#                               combined_scores, y_test, attack_types):
#         """Visualization of cascade detector stages."""
#         fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
#         comp_mask = attack_types == 'slave_compromise'
#         norm_mask = y_test == 0
#         other_atk = (y_test == 1) & ~comp_mask

#         for ax, scores, title in zip(axes,
#             [stage1_scores, stage2_scores, combined_scores],
#             ['Уровень 1 (общий RF)', 'Уровень 2 (компрометация)', 'Каскад (OR)']):

#             ax.hist(scores[norm_mask], bins=60, alpha=0.4, color='#3B82F6',
#                     label='Норма', density=True)
#             if other_atk.sum() > 0:
#                 ax.hist(scores[other_atk], bins=30, alpha=0.5, color='#F59E0B',
#                         label='Прочие атаки', density=True)
#             if comp_mask.sum() > 0:
#                 ax.hist(scores[comp_mask], bins=30, alpha=0.6, color='#059669',
#                         label='Компрометация', density=True)
#             ax.axvline(0.5, color='red', linestyle='--', linewidth=1.5,
#                        label='Порог 0.5')
#             ax.set_title(title)
#             ax.set_xlabel('P(аномалия)')
#             ax.legend(fontsize=7)

#         fig.suptitle('Анализ каскадного детектора', fontsize=13)
#         fig.tight_layout()
#         self._save(fig, '14_cascade_analysis.png')


# ══════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════

def main():
    config = TrainingConfig()
    os.makedirs(config.results_dir, exist_ok=True)
    os.makedirs(config.plots_dir, exist_ok=True)

    pipeline = DataPipeline(config)
    train_df, val_df, test_df = pipeline.load_and_prepare()

    # Load full dataset for overview plot
    full_df = pd.read_csv(config.data_path)

    X_train, y_train = pipeline.get_features_labels(train_df, fit=True)
    X_val, y_val = pipeline.get_features_labels(val_df)
    X_test, y_test = pipeline.get_features_labels(test_df)

    X_train_comp, y_train_comp = pipeline.get_compromise_features(train_df, fit=True)
    X_val_comp, y_val_comp = pipeline.get_compromise_features(val_df)
    X_test_comp, y_test_comp = pipeline.get_compromise_features(test_df)

    X_train_normal, _ = pipeline.get_normal_data(train_df, fit=False)

    attack_types_test = test_df["attack_type"].values
    n_feat = X_train.shape[1]

    results_list = []
    trainers_dict = {}  # for training curves

    print("\n" + "=" * 60)
    print("  Training Models v6")
    print("=" * 60)

    # ── 1. Isolation Forest ──
    print("\n[1] Isolation Forest...")
    iso = IsolationForest(n_estimators=config.iso_forest_trees,
                          contamination=config.iso_forest_contamination,
                          random_state=config.seed)
    iso.fit(X_train_normal)
    iso_scores_val = -iso.score_samples(X_val)
    iso_scores_test = -iso.score_samples(X_test)
    th_iso, _ = find_threshold(iso_scores_val, y_val, config.threshold_percentile)
    r = evaluate_model("Isolation Forest", iso_scores_test, y_test,
                       attack_types_test, threshold=th_iso)
    results_list.append(r)

    # ── 2. One-Class SVM ──
    print("\n[2] One-Class SVM...")
    svm = OneClassSVM(nu=config.ocsvm_nu, kernel=config.ocsvm_kernel, gamma="scale")
    svm.fit(X_train_normal)
    svm_scores_val = -svm.score_samples(X_val)
    svm_scores_test = -svm.score_samples(X_test)
    th_svm, _ = find_threshold(svm_scores_val, y_val, config.threshold_percentile)
    r = evaluate_model("One-Class SVM", svm_scores_test, y_test,
                       attack_types_test, threshold=th_svm)
    results_list.append(r)

    # ── 3. Autoencoder ──
    print("\n[3] Autoencoder...")
    ae = Autoencoder(n_feat, config.ae_hidden_dims, config.ae_latent_dim)
    ae_trainer = AETrainer(ae, config.ae_lr, config.ae_epochs,
                           config.ae_batch, patience=15)
    X_val_normal = X_val[y_val == 0]
    ae_trainer.fit(X_train_normal, X_val_normal)
    trainers_dict['Autoencoder'] = ae_trainer
    ae_scores_val = ae_trainer.reconstruction_error(X_val)
    ae_scores_test = ae_trainer.reconstruction_error(X_test)
    th_ae, _ = find_threshold(ae_scores_val, y_val, config.threshold_percentile)
    r = evaluate_model("Autoencoder", ae_scores_test, y_test,
                       attack_types_test, threshold=th_ae)
    results_list.append(r)

    # ── 4. VAE ──
    print("\n[4] VAE...")
    vae = VAE(n_feat, config.vae_hidden_dims, config.vae_latent_dim)
    vae_trainer = VAETrainer(vae, config.vae_lr, config.vae_epochs,
                             config.vae_batch, kl_weight=config.vae_kl_weight,
                             patience=15)
    vae_trainer.fit(X_train_normal, X_val_normal)
    trainers_dict['VAE'] = vae_trainer
    vae_scores_val = vae_trainer.anomaly_score(X_val)
    vae_scores_test = vae_trainer.anomaly_score(X_test)
    th_vae, _ = find_threshold(vae_scores_val, y_val, config.threshold_percentile)
    r = evaluate_model("VAE", vae_scores_test, y_test,
                       attack_types_test, threshold=th_vae)
    results_list.append(r)

    # ── 5. LSTM-AE ──
    print("\n[5] LSTM-Autoencoder...")
    train_normal_df = train_df[train_df[pipeline.LABEL_COL] == 0]
    seq_train, _ = pipeline.get_sequential_data(train_normal_df, config.lstm_seq_len)

    lstm_result = None
    if len(seq_train) > 100:
        lstm = LSTMAutoencoder(n_feat, config.lstm_hidden, config.lstm_layers,
                               config.lstm_seq_len, dropout=0.2)
        lstm_trainer = AETrainer(lstm, config.lstm_lr, config.lstm_epochs,
                                 config.lstm_batch, patience=15)
        seq_val, seq_y_val = pipeline.get_sequential_data(val_df, config.lstm_seq_len)
        seq_val_normal = seq_val[seq_y_val == 0]
        lstm_trainer.fit(seq_train, seq_val_normal if len(seq_val_normal) > 0 else None)
        trainers_dict['LSTM-AE'] = lstm_trainer

        seq_test, seq_y_test = pipeline.get_sequential_data(test_df, config.lstm_seq_len)
        seq_attack_types_test = []
        for nid in test_df["node_id"].unique():
            mask = test_df["node_id"].values == nid
            atypes = test_df.loc[mask, "attack_type"].values
            for i in range(len(atypes) - config.lstm_seq_len + 1):
                seq_attack_types_test.append(atypes[i + config.lstm_seq_len - 1])
        seq_attack_types_test = np.array(seq_attack_types_test)

        lstm_scores_test = lstm_trainer.reconstruction_error(seq_test)
        lstm_scores_val = lstm_trainer.reconstruction_error(seq_val)
        th_lstm, _ = find_threshold(lstm_scores_val, seq_y_val, config.threshold_percentile)
        lstm_result = evaluate_model("LSTM-Autoencoder", lstm_scores_test, seq_y_test,
                                     seq_attack_types_test, threshold=th_lstm)
        results_list.append(lstm_result)

    # ── 6. Temporal Attention AE ──
    print("\n[6] Temporal Attention AE...")
    if len(seq_train) > 100:
        tat = TemporalAttentionAE(n_feat, 64, 4, 2, config.lstm_seq_len, 0.15)
        tat_trainer = AETrainer(tat, 5e-4, 100, 256, patience=15)
        tat_trainer.fit(seq_train, seq_val_normal if len(seq_val_normal) > 0 else None)
        trainers_dict['Temporal-Attn-AE'] = tat_trainer
        tat_scores_test = tat_trainer.reconstruction_error(seq_test)
        tat_scores_val = tat_trainer.reconstruction_error(seq_val)
        th_tat, _ = find_threshold(tat_scores_val, seq_y_val, config.threshold_percentile)
        r = evaluate_model("Temporal-Attention-AE", tat_scores_test, seq_y_test,
                           seq_attack_types_test, threshold=th_tat)
        results_list.append(r)

    # ── 7. Random Forest ──
    print("\n[7] Random Forest...")
    rf = RandomForestClassifier(n_estimators=config.rf_trees, random_state=config.seed,
                                class_weight="balanced", n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_proba_test = rf.predict_proba(X_test)[:, 1]
    r = evaluate_model("Random Forest", rf_proba_test, y_test,
                       attack_types_test, threshold=0.5)
    results_list.append(r)

    print("\n  Feature importance (top 20):")
    importances = rf.feature_importances_
    feat_imp = sorted(zip(pipeline.feature_cols, importances),
                      key=lambda x: x[1], reverse=True)
    for fn, imp in feat_imp[:20]:
        print(f"    {fn:45s} {imp:.4f} ({100*imp:.1f}%)")

    # ── 8. Gradient Boosting ──
    print("\n[8] Gradient Boosting...")
    gb = GradientBoostingClassifier(n_estimators=config.gb_estimators,
                                    learning_rate=config.gb_lr,
                                    max_depth=5, random_state=config.seed)
    gb.fit(X_train, y_train)
    gb_proba_test = gb.predict_proba(X_test)[:, 1]
    r = evaluate_model("Gradient Boosting", gb_proba_test, y_test,
                       attack_types_test, threshold=0.5)
    results_list.append(r)

    # ── 9. Cascade ──
    print("\n[9] Cascade Compromise Detector...")
    cascade = CascadeCompromiseDetector(config, config.seed)
    cascade.fit(X_train, y_train, X_train_comp, y_train_comp)
    cascade_scores_test = cascade.predict_proba(X_test, X_test_comp)
    r = evaluate_model("Cascade (v6)", cascade_scores_test, y_test,
                       attack_types_test, threshold=0.5)
    results_list.append(r)

    s2_scores_test = cascade.predict_proba_stage2(X_test_comp)
    r_s2 = evaluate_model("Cascade-Stage2-only", s2_scores_test, y_test,
                          attack_types_test)
    results_list.append(r_s2)

    # ── 10. Specialized Ensemble ──
    print("\n[10] Specialized Ensemble...")
    ensemble = EnsembleDetector(FEATURE_GROUPS, pipeline.feature_cols, config.seed)
    ensemble.fit(X_train_normal)
    ens_scores_val = ensemble.score(X_val)
    ens_scores_test = ensemble.score(X_test)

    ens_results = []
    for en in sorted(ens_scores_test.keys()):
        sv, st = ens_scores_val[en], ens_scores_test[en]
        th, _ = find_threshold(sv, y_val, config.threshold_percentile)
        r = evaluate_model(f"Ensemble:{en}", st, y_test,
                           attack_types_test, threshold=th)
        results_list.append(r)
        ens_results.append(r)

    # ── 11. Meta-Ensemble ──
    print("\n[11] Meta-Ensemble...")
    def norm_score(s):
        p1, p99 = np.percentile(s, 1), np.percentile(s, 99)
        return np.clip((s - p1) / (p99 - p1 + 1e-9), 0, 1)

    meta_features_train = np.column_stack([
        norm_score(-iso.score_samples(X_train)),
        norm_score(-svm.score_samples(X_train)),
        norm_score(ae_trainer.reconstruction_error(X_train)),
        norm_score(vae_trainer.anomaly_score(X_train)),
        norm_score(cascade.predict_proba(X_train, X_train_comp)),
        norm_score(rf.predict_proba(X_train)[:, 1]),
        norm_score(gb.predict_proba(X_train)[:, 1]),
    ])
    meta_features_test = np.column_stack([
        norm_score(iso_scores_test), norm_score(svm_scores_test),
        norm_score(ae_scores_test), norm_score(vae_scores_test),
        norm_score(cascade_scores_test),
        norm_score(rf_proba_test), norm_score(gb_proba_test),
    ])

    meta_gb = GradientBoostingClassifier(n_estimators=200, learning_rate=0.05,
                                         max_depth=3, random_state=config.seed)
    meta_gb.fit(meta_features_train, y_train)
    meta_scores_test = meta_gb.predict_proba(meta_features_test)[:, 1]
    r = evaluate_model("Meta-Ensemble (v6)", meta_scores_test, y_test,
                       attack_types_test, threshold=0.5)
    results_list.append(r)

    # ── 12. Ablation studies ──
    print("\n" + "=" * 60)
    print("  Ablation Studies")
    print("=" * 60)

    ablation_results = []
    comp_mask = attack_types_test == 'slave_compromise'
    comp_total = comp_mask.sum()

    # Full model
    comp_full = (rf_proba_test[comp_mask] >= 0.5).sum()
    ablation_results.append({
        'config': 'Полный набор\n(50 призн.)',
        'f1': next(r for r in results_list if r['name'] == 'Random Forest')['f1'],
        'compromise_rate': 100 * comp_full / comp_total if comp_total > 0 else 0,
    })

    # Without MS features
    non_ms = [c for c in pipeline.feature_cols if c not in MS_FEATURES]
    non_ms_idx = [pipeline.feature_cols.index(c) for c in non_ms]
    rf_no_ms = RandomForestClassifier(n_estimators=config.rf_trees,
                                      random_state=config.seed,
                                      class_weight="balanced", n_jobs=-1)
    rf_no_ms.fit(X_train[:, non_ms_idx], y_train)
    rf_no_ms_p = rf_no_ms.predict_proba(X_test[:, non_ms_idx])[:, 1]
    r_no_ms = evaluate_model("RF (no MS)", rf_no_ms_p, y_test,
                             attack_types_test, threshold=0.5)
    comp_no_ms = (rf_no_ms_p[comp_mask] >= 0.5).sum()
    ablation_results.append({
        'config': 'Без MS-призн.\n(20 призн.)',
        'f1': r_no_ms['f1'],
        'compromise_rate': 100 * comp_no_ms / comp_total if comp_total > 0 else 0,
    })

    # Without temporal features
    temporal_features = [
        "var_obs_total_messages", "var_obs_entropy",
        "var_obs_inbound_outbound_ratio", "var_obs_map_dominance_ratio",
        "autocorr_obs_total_messages", "integrity_fail_rate", "obs_destinations_cv",
    ]
    non_temp = [c for c in pipeline.feature_cols if c not in temporal_features]
    non_temp_idx = [pipeline.feature_cols.index(c) for c in non_temp]
    rf_no_temp = RandomForestClassifier(n_estimators=config.rf_trees,
                                        random_state=config.seed,
                                        class_weight="balanced", n_jobs=-1)
    rf_no_temp.fit(X_train[:, non_temp_idx], y_train)
    rf_no_temp_p = rf_no_temp.predict_proba(X_test[:, non_temp_idx])[:, 1]
    r_no_temp = evaluate_model("RF (no temporal)", rf_no_temp_p, y_test,
                               attack_types_test, threshold=0.5)
    comp_no_temp = (rf_no_temp_p[comp_mask] >= 0.5).sum()
    ablation_results.append({
        'config': 'Без temporal\n(43 призн.)',
        'f1': r_no_temp['f1'],
        'compromise_rate': 100 * comp_no_temp / comp_total if comp_total > 0 else 0,
    })

    # Cascade Stage 1 only
    s1_scores = cascade.predict_proba_stage1(X_test)
    comp_s1 = (s1_scores[comp_mask] >= 0.5).sum()
    r_s1 = evaluate_model("Cascade Stage-1", s1_scores, y_test,
                          attack_types_test, threshold=0.5)
    ablation_results.append({
        'config': 'Каскад:\nтолько ур.1',
        'f1': r_s1['f1'],
        'compromise_rate': 100 * comp_s1 / comp_total if comp_total > 0 else 0,
    })

    # Cascade full
    comp_cascade = (cascade_scores_test[comp_mask] >= 0.5).sum()
    ablation_results.append({
        'config': 'Каскад:\nоба уровня',
        'f1': next(r for r in results_list if r['name'] == 'Cascade (v6)')['f1'],
        'compromise_rate': 100 * comp_cascade / comp_total if comp_total > 0 else 0,
    })

    # ── Save results ──
    save_cols = ['name', 'auc_roc', 'auc_pr', 'f1', 'fpr', 'tp', 'fp', 'fn', 'tn']
    results_df = pd.DataFrame([{k: r[k] for k in save_cols} for r in results_list])
    results_df.to_csv(os.path.join(config.results_dir, "full_comparison_v6.csv"),
                      index=False)

    with open(os.path.join(config.results_dir, "feature_columns_v6.json"), "w") as f:
        json.dump({
            "observable_features": pipeline.feature_cols,
            "compromise_features": pipeline.compromise_feature_cols,
            "ms_features": MS_FEATURES,
            "temporal_features": temporal_features,
            "ground_truth_cols": pipeline.GROUND_TRUTH_COLS,
            "feature_groups": FEATURE_GROUPS,
        }, f, indent=2)

    # Summary table
    print("\n" + "=" * 70)
    print("  SUMMARY v6")
    print("=" * 70)
    print(f"{'Model':<35s} {'AUC-ROC':>8s} {'AUC-PR':>8s} {'F1':>8s} {'FPR':>8s}")
    print("-" * 67)
    for r in results_list:
        print(f"{r['name']:<35s} {r['auc_roc']:8.4f} {r['auc_pr']:8.4f} "
              f"{r['f1']:8.4f} {r['fpr']:8.4f}")

    # ══════════════════════════════════════════════════
    #  GENERATE ALL PLOTS
    # ══════════════════════════════════════════════════

    print("\n" + "=" * 60)
    print("  Generating plots...")
    print("=" * 60)

    viz = ResultsVisualizer(config.plots_dir)

    # 1. ROC curves
    viz.plot_roc_curves(results_list, y_test)

    # 2. ROC zoom
    viz.plot_roc_zoom(results_list, y_test)

    # 3. Precision-Recall
    viz.plot_pr_curves(results_list, y_test)

    # 4. Attack detection heatmap
    viz.plot_attack_detection_heatmap(results_list)

    # 5. Summary bars
    viz.plot_summary_bars(results_list)

    # 6. Feature importance
    viz.plot_feature_importance(pipeline.feature_cols, importances, MS_FEATURES)

    # 7. Ablation
    viz.plot_ablation(ablation_results)

    # 8. Score distributions
    viz.plot_score_distributions(results_list, y_test, attack_types_test)

    # 9. Confusion matrices
    viz.plot_confusion_matrices(results_list, y_test)

    # 10. Compromise comparison
    viz.plot_compromise_comparison(results_list, attack_types_test)

    # 11. Training curves
    viz.plot_training_curves(trainers_dict)

    # 12. Ensemble specialization
    viz.plot_ensemble_results(ens_results, y_test, attack_types_test)

    # 13. Dataset overview
    viz.plot_dataset_overview(full_df)

    # 14. Cascade analysis
    s1_test = cascade.predict_proba_stage1(X_test)
    s2_test = cascade.predict_proba_stage2(X_test_comp)
    viz.plot_cascade_analysis(s1_test, s2_test, cascade_scores_test,
                              y_test, attack_types_test)

    print(f"\n  All plots saved to {config.plots_dir}/")
    print(f"  Total: 14 figures")


if __name__ == "__main__":
    main()
