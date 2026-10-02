"""
SIP Master-Slave Signalling Network Simulator v4
====================================================
Транзакционно-точная имитация SIP (RFC 3261) с РЕАЛИСТИЧНОЙ, МНОГОУРОВНЕВОЙ
моделью КОМПРОМЕТАЦИИ узла (adaptive/осведомлённый атакующий).

ИЗМЕНЕНИЯ v3.1 (относительно v3):
  Цель — сделать ТРАНЗАКЦИОННУЮ ЦЕЛОСТНОСТЬ доминирующим остаточным
  рычагом против adaptive-атакующего (как заявлено в модели угроз),
  вместо случайного «физического чита» по задержке обработки.

  (1) Физический след (processing_delay) ОСЛАБЛЕН и теперь зависит от
      изощрённости: adaptive-атакующий маскирует и задержку → она больше
      не доминирует в feature importance и не делает txn-признаки лишними.
  (2) Нарушение закона сохранения транзакционного автомата стало
      СИСТЕМАТИЧЕСКИМ (направленный дефицит замыкания диалогов,
      пропорциональный доле covert redirect), а не шумовым → признак
      cu_txn_state_consistency_error выучивается моделью.
  (3) Adaptive-атакующий может подогнать НАБЛЮДАЕМЫЕ dcr/ddr под зону, но
      физически не может «вернуть» уведённые вызовы → остаточный дефицит
      замыкания сохраняется даже после подгонки под зональную медиану.

Модель угроз (как в v3):
  NAIVE / STATISTICAL / ADAPTIVE — три уровня мимикрии.

Refs (модель угроз):
  - CVE-2020-7034 (Avaya SBC RCE)
  - Keromytis, "A Comprehensive Survey of VoIP Security Research", 2012
  - RFC 3261 §17 (transactions), §13 (dialogs), RFC 5393 (forking),
    RFC 6026 (2xx INVITE handling)
"""

import numpy as np
import pandas as pd
import networkx as nx
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from collections import defaultdict, deque
import warnings

warnings.filterwarnings("ignore")


# ──────────────────────────────────────────────────────────
#  RFC 3261 §17 — Таймеры
# ──────────────────────────────────────────────────────────

class SIPTimers:
    T1 = 0.5
    T2 = 4.0
    T4 = 5.0
    TIMER_A = T1
    TIMER_B = 64 * T1
    TIMER_D = 32.0
    TIMER_E = T1
    TIMER_F = 64 * T1
    TIMER_G = T1
    TIMER_H = 64 * T1
    TIMER_I = T4
    TIMER_J = 64 * T1
    TIMER_K = T4
    MAX_INVITE_RETRANSMIT = 7
    MAX_NONINVITE_RETRANSMIT = 11


# ──────────────────────────────────────────────────────────
#  Enums
# ──────────────────────────────────────────────────────────

class NodeType(Enum):
    SBC = auto()
    PROXY = auto()
    REGISTRAR = auto()
    UA_GW = auto()
    REDIRECT = auto()


class NodeRole(Enum):
    MASTER = auto()
    SLAVE = auto()


class AttackerSophistication(Enum):
    """Уровень изощрённости компрометированного узла (модель угроз v3)."""
    NAIVE = "naive"
    STATISTICAL = "statistical"
    ADAPTIVE = "adaptive"


class AttackType(Enum):
    NONE = "none"
    SIGNALING_DOS = "signaling_dos"
    REGISTER_HIJACK = "register_hijack"
    REGISTER_BRUTEFORCE = "register_bruteforce"
    SPIT = "spit"
    PROXY_COMPROMISE = "proxy_compromise"


class NormalEventType(Enum):
    NONE = "none"
    TRAFFIC_SPIKE = "traffic_spike"
    MAINTENANCE = "maintenance"
    REROUTE = "reroute"
    PRESENCE_SURGE = "presence_surge"


# ──────────────────────────────────────────────────────────
#  SIP-методы и ответы
# ──────────────────────────────────────────────────────────

SIP_METHODS = ["INVITE", "ACK", "BYE", "REGISTER", "OPTIONS",
               "CANCEL", "SUBSCRIBE", "NOTIFY", "OTHER"]
INVITE_METHODS = {"INVITE"}
NONINVITE_METHODS = {"REGISTER", "OPTIONS", "BYE", "SUBSCRIBE",
                     "NOTIFY", "CANCEL", "OTHER"}
SIP_RESP_CLASSES = ["resp_1xx", "resp_2xx", "resp_3xx", "resp_4xx",
                    "resp_401_407", "resp_408", "resp_5xx", "resp_6xx"]


# ══════════════════════════════════════════════════════════
#  1. Транзакционный автомат — теперь возвращает n_ack явно
# ══════════════════════════════════════════════════════════

class SIPTransactionModel:
    def __init__(self, rng: np.random.Generator):
        self.rng = rng

    def simulate_invite_client(self, n_txn, link_loss, rtt_ms,
                               attack_no_ack=0.0, covert_redirect=0.0,
                               transit_loss=0.0):
        """
        RFC 3261 §13/§17. Истинный автомат: n_200 = n_completed + n_dangling.

        Вариант A (физически корректный): ACK для 2xx — отдельная транзакция,
        маршрутизируемая end-to-end через цепочку прокси/SBC (RFC §13.2.2.4).
        Мастер-SBC лежит на пути и считает проходящие ACK/200 НА ТРАНЗИТЕ,
        независимо от прикладной отчётности узла. transit_loss — канальная
        потеря на пути узел↔SBC (честный шум, атакующий на неё не влияет).
        """
        loss = np.clip(link_loss, 0.0, 0.95)
        if n_txn <= 0:
            return self._empty_invite()
        retransmits = self._truncated_geometric(
            n_txn, loss, SIPTimers.MAX_INVITE_RETRANSMIT)
        p_all_lost = loss ** SIPTimers.MAX_INVITE_RETRANSMIT
        timed_out = self.rng.random(n_txn) < p_all_lost
        n_timeout = int(timed_out.sum())
        n_answered = n_txn - n_timeout
        n_200 = max(0, n_answered)

        ack_loss = loss
        natural_dangling = self.rng.random(n_200) < ack_loss
        attack_dangling = self.rng.random(n_200) < attack_no_ack
        redirect_dangling = self.rng.random(n_200) < covert_redirect
        dangling_mask = natural_dangling | attack_dangling | redirect_dangling
        n_dangling = int(dangling_mask.sum())
        n_completed_dialogs = n_200 - n_dangling

        # ── ТРАНЗИТНОЕ НАБЛЮДЕНИЕ МАСТЕРА (Вариант A) ──
        # 200 OK идут через SBC → мастер видит n_200 на транзите (с потерей).
        # ACK замкнутых диалогов идут через SBC → n_ack на транзите.
        # Атакующий НЕ управляет этими счётчиками: это транспортный учёт.
        tl = np.clip(transit_loss, 0.0, 0.5)
        n_200_transit = int((self.rng.random(n_200) >= tl).sum()) \
            if n_200 > 0 else 0
        n_ack_transit = int(
            (self.rng.random(max(0, n_completed_dialogs)) >= tl).sum()
        ) if n_completed_dialogs > 0 else 0
        # мастер не может увидеть больше ACK, чем прошедших 200 OK
        n_ack_transit = min(n_ack_transit, n_200_transit)

        return {
            "n_invite_txn": n_txn,
            "n_invite_timeout": n_timeout,
            "n_invite_200": n_200,
            "n_dangling_dialogs": n_dangling,
            "n_completed_dialogs": n_completed_dialogs,
            # ── транзитно наблюдаемые (физика, не ground truth) ──
            "n_200_transit": n_200_transit,
            "n_ack_transit": n_ack_transit,
            "invite_total_retransmits": int(retransmits.sum()),
            "invite_mean_retransmits": float(retransmits.mean()),
            "invite_frac_retransmitted": float((retransmits > 0).mean()),
            "invite_timeout_ratio": n_timeout / max(1, n_txn),
            "dialog_completion_ratio": n_completed_dialogs / max(1, n_200),
        }

    def _empty_invite(self):
        return {
            "n_invite_txn": 0, "n_invite_timeout": 0, "n_invite_200": 0,
            "n_dangling_dialogs": 0, "n_completed_dialogs": 0,
            "n_200_transit": 0, "n_ack_transit": 0,
            "invite_total_retransmits": 0, "invite_mean_retransmits": 0.0,
            "invite_frac_retransmitted": 0.0, "invite_timeout_ratio": 0.0,
            "dialog_completion_ratio": 1.0,
        }


    def simulate_noninvite_client(self, n_txn, link_loss):
        loss = np.clip(link_loss, 0.0, 0.95)
        if n_txn <= 0:
            return {"n_noninvite_txn": 0, "n_noninvite_timeout": 0,
                    "noninvite_total_retransmits": 0,
                    "noninvite_timeout_ratio": 0.0}
        retransmits = self._truncated_geometric(
            n_txn, loss, SIPTimers.MAX_NONINVITE_RETRANSMIT)
        p_all_lost = loss ** SIPTimers.MAX_NONINVITE_RETRANSMIT
        n_timeout = int((self.rng.random(n_txn) < p_all_lost).sum())
        return {
            "n_noninvite_txn": n_txn,
            "n_noninvite_timeout": n_timeout,
            "noninvite_total_retransmits": int(retransmits.sum()),
            "noninvite_timeout_ratio": n_timeout / max(1, n_txn),
        }

    def _truncated_geometric(self, n, p, cap):
        if n <= 0:
            return np.array([], dtype=int)
        if p <= 1e-9:
            return np.zeros(n, dtype=int)
        u = self.rng.random(n)
        with np.errstate(divide="ignore"):
            k = np.floor(np.log(u + 1e-12) / np.log(p + 1e-12)).astype(int)
        return np.clip(k, 0, cap)


# ──────────────────────────────────────────────────────────
#  Data Classes
# ──────────────────────────────────────────────────────────

@dataclass
class SIPNode:
    node_id: int
    node_type: NodeType
    role: NodeRole = NodeRole.SLAVE
    hostname: str = ""
    realm: str = "operator.com"
    transport: str = "UDP"
    master_id: Optional[int] = None
    zone_id: int = 0
    base_rate: float = 1000.0
    forking_factor: float = 1.0

    method_dist: Dict[str, float] = field(default_factory=lambda: {
        "INVITE": 0.22, "ACK": 0.20, "BYE": 0.12, "REGISTER": 0.18,
        "OPTIONS": 0.10, "CANCEL": 0.02, "SUBSCRIBE": 0.06,
        "NOTIFY": 0.08, "OTHER": 0.02,
    })
    resp_dist: Dict[str, float] = field(default_factory=lambda: {
        "resp_1xx": 0.33, "resp_2xx": 0.45, "resp_3xx": 0.02,
        "resp_4xx": 0.07, "resp_401_407": 0.04, "resp_408": 0.02,
        "resp_5xx": 0.05, "resp_6xx": 0.02,
    })
    is_compromised: bool = False
    compromised_since: Optional[int] = None
    attacker_sophistication: AttackerSophistication = \
        AttackerSophistication.ADAPTIVE
    covert_redirect_fraction: float = 0.0


@dataclass
class SIPLink:
    src: int
    dst: int
    capacity_mbps: float = 100.0
    propagation_delay_ms: float = 2.0
    base_loss_prob: float = 0.0005
    current_load_mbps: float = 0.0
    maintenance_loss_mult: float = 1.0
    maintenance_delay_mult: float = 1.0

    @property
    def rho(self):
        if self.capacity_mbps <= 0:
            return 999.0
        return self.current_load_mbps / self.capacity_mbps

    @property
    def effective_delay_ms(self):
        base = self.propagation_delay_ms * self.maintenance_delay_mult
        r = self.rho
        if r > 0.7:
            base *= (1.0 + 3.0 * (r - 0.7) / 0.3)
        if r > 1.0:
            base *= (1.0 + 8.0 * (r - 1.0))
        return base

    @property
    def effective_loss(self):
        p = self.base_loss_prob * self.maintenance_loss_mult
        r = self.rho
        if r > 0.8:
            p += 0.03 * ((r - 0.8) / 0.2) ** 2
        if r > 1.0:
            p += 0.25 * (r - 1.0)
        return min(p, 0.90)


@dataclass
class ControlUnitConfig:
    poll_interval_s: float = 20.0
    cu_message_size_bytes: int = 350
    processing_delay_ms: float = 2.0
    integrity_background_fail_prob: float = 0.01


@dataclass
class AttackScenario:
    attack_type: AttackType
    target_nodes: List[int]
    start_interval: int
    end_interval: int
    intensity: float = 1.0
    sophistication: AttackerSophistication = AttackerSophistication.ADAPTIVE
    covert_redirect_fraction: float = 0.15


@dataclass
class NormalScenario:
    event_type: NormalEventType
    target_nodes: List[int]
    start_interval: int
    end_interval: int
    intensity: float = 1.0


@dataclass
class SimulationConfig:
    duration_hours: float = 24.0
    interval_s: float = 20.0
    n_sbc: int = 2
    n_proxy: int = 3
    n_registrar: int = 2
    n_ua_gw: int = 4
    n_redirect: int = 2
    master_node_ids: List[int] = field(default_factory=lambda: [0, 1])
    sbc_sbc_capacity_mbps: float = 1000.0
    sbc_node_capacity_mbps: float = 100.0
    ar_phi: float = 0.85
    ar_sigma: float = 0.18
    avg_message_size_bytes: int = 700
    seed: int = 42
    cu_config: ControlUnitConfig = field(default_factory=ControlUnitConfig)


# ──────────────────────────────────────────────────────────
#  Topology Builder
# ──────────────────────────────────────────────────────────

class SIPNetworkTopology:
    def __init__(self, config):
        self.config = config
        self.graph = nx.Graph()
        self.nodes: Dict[int, SIPNode] = {}
        self.links: Dict[Tuple[int, int], SIPLink] = {}
        self._build()

    def _build(self):
        rng = np.random.default_rng(self.config.seed)
        node_id = 0

        sbc_ids = []
        for i in range(self.config.n_sbc):
            n = SIPNode(node_id=node_id, node_type=NodeType.SBC,
                        hostname=f"sbc-{i}.operator.com", transport="TLS",
                        base_rate=rng.uniform(3000, 6000))
            self.nodes[node_id] = n
            self.graph.add_node(node_id)
            sbc_ids.append(node_id)
            node_id += 1

        proxy_ids = []
        for i in range(self.config.n_proxy):
            n = SIPNode(node_id=node_id, node_type=NodeType.PROXY,
                        hostname=f"proxy-{i}.operator.com", transport="UDP",
                        forking_factor=rng.uniform(1.2, 1.8),
                        base_rate=rng.uniform(2000, 4500))
            self.nodes[node_id] = n
            self.graph.add_node(node_id)
            proxy_ids.append(node_id)
            node_id += 1

        registrar_ids = []
        for i in range(self.config.n_registrar):
            n = SIPNode(node_id=node_id, node_type=NodeType.REGISTRAR,
                        hostname=f"registrar-{i}.operator.com",
                        transport="UDP", base_rate=rng.uniform(1500, 3500))
            n.method_dist = {
                "INVITE": 0.05, "ACK": 0.04, "BYE": 0.03, "REGISTER": 0.55,
                "OPTIONS": 0.10, "CANCEL": 0.01, "SUBSCRIBE": 0.08,
                "NOTIFY": 0.12, "OTHER": 0.02,
            }
            n.resp_dist = {
                "resp_1xx": 0.08, "resp_2xx": 0.55, "resp_3xx": 0.01,
                "resp_4xx": 0.04, "resp_401_407": 0.25, "resp_408": 0.02,
                "resp_5xx": 0.03, "resp_6xx": 0.02,
            }
            self.nodes[node_id] = n
            self.graph.add_node(node_id)
            registrar_ids.append(node_id)
            node_id += 1

        ua_ids = []
        for i in range(self.config.n_ua_gw):
            n = SIPNode(node_id=node_id, node_type=NodeType.UA_GW,
                        hostname=f"uagw-{i}.operator.com", transport="UDP",
                        base_rate=rng.uniform(1500, 4000))
            n.method_dist = {
                "INVITE": 0.30, "ACK": 0.27, "BYE": 0.16, "REGISTER": 0.06,
                "OPTIONS": 0.08, "CANCEL": 0.03, "SUBSCRIBE": 0.04,
                "NOTIFY": 0.04, "OTHER": 0.02,
            }
            self.nodes[node_id] = n
            self.graph.add_node(node_id)
            ua_ids.append(node_id)
            node_id += 1

        redirect_ids = []
        for i in range(self.config.n_redirect):
            n = SIPNode(node_id=node_id, node_type=NodeType.REDIRECT,
                        hostname=f"redirect-{i}.operator.com",
                        transport="UDP", base_rate=rng.uniform(500, 1500))
            n.resp_dist = {
                "resp_1xx": 0.13, "resp_2xx": 0.20, "resp_3xx": 0.45,
                "resp_4xx": 0.08, "resp_401_407": 0.03, "resp_408": 0.02,
                "resp_5xx": 0.07, "resp_6xx": 0.02,
            }
            self.nodes[node_id] = n
            self.graph.add_node(node_id)
            redirect_ids.append(node_id)
            node_id += 1

        for i, s1 in enumerate(sbc_ids):
            for s2 in sbc_ids[i + 1:]:
                self._add_link(s1, s2, self.config.sbc_sbc_capacity_mbps,
                               delay_ms=1.0 + rng.uniform(0, 1.0))
        for nid in proxy_ids + registrar_ids:
            for sbc in sbc_ids:
                self._add_link(nid, sbc, self.config.sbc_node_capacity_mbps,
                               delay_ms=1.5 + rng.uniform(0, 2.5))
        for nid in ua_ids + redirect_ids:
            chosen = rng.choice(sbc_ids, size=min(2, len(sbc_ids)),
                                replace=False)
            for sbc in chosen:
                self._add_link(nid, sbc, self.config.sbc_node_capacity_mbps,
                               delay_ms=2.0 + rng.uniform(0, 3.0))

        for mid in self.config.master_node_ids:
            if mid in self.nodes:
                self.nodes[mid].role = NodeRole.MASTER
        for nid, node in self.nodes.items():
            if node.role == NodeRole.MASTER:
                node.master_id = nid
                node.zone_id = nid
                continue
            best_master, best_dist = None, float("inf")
            for mid in self.config.master_node_ids:
                try:
                    d = nx.shortest_path_length(self.graph, nid, mid)
                    if d < best_dist:
                        best_dist, best_master = d, mid
                except nx.NetworkXNoPath:
                    pass
            node.master_id = (best_master if best_master is not None
                              else self.config.master_node_ids[0])
            node.zone_id = node.master_id

    def _add_link(self, src, dst, capacity, delay_ms=2.0):
        key = (min(src, dst), max(src, dst))
        if key not in self.links:
            self.links[key] = SIPLink(src=key[0], dst=key[1],
                                      capacity_mbps=capacity,
                                      propagation_delay_ms=delay_ms)
            self.graph.add_edge(key[0], key[1])

    def get_link(self, a, b):
        return self.links.get((min(a, b), max(a, b)))

    def get_path(self, src, dst):
        try:
            return nx.shortest_path(self.graph, src, dst)
        except nx.NetworkXNoPath:
            return []

    def reset_link_loads(self):
        for link in self.links.values():
            link.current_load_mbps = 0.0

    def path_loss(self, path):
        survive = 1.0
        for i in range(len(path) - 1):
            link = self.get_link(path[i], path[i + 1])
            survive *= (1.0 - (link.effective_loss if link else 0.5))
        return 1.0 - survive

    def path_delay(self, path):
        d = 0.0
        for i in range(len(path) - 1):
            link = self.get_link(path[i], path[i + 1])
            d += link.effective_delay_ms if link else 5.0
        return d


# ──────────────────────────────────────────────────────────
#  Traffic Generator
# ──────────────────────────────────────────────────────────

class SIPTrafficGenerator:
    TOD_PROFILE = [
        0.3, 0.2, 0.15, 0.15, 0.2, 0.4,
        0.7, 0.9, 1.0, 1.0, 0.95, 0.9,
        0.85, 0.9, 0.95, 1.0, 1.0, 0.95,
        0.9, 0.8, 0.7, 0.6, 0.5, 0.4
    ]

    def __init__(self, topology, config):
        self.topo = topology
        self.config = config
        self.rng = np.random.default_rng(config.seed + 1)
        self.txn_model = SIPTransactionModel(self.rng)
        self.ar_state = {nid: 0.0 for nid in topology.nodes}

    def time_of_day_factor(self, interval_idx):
        t_seconds = interval_idx * self.config.interval_s
        hour = (t_seconds / 3600.0) % 24.0
        h_low = int(hour) % 24
        h_high = (h_low + 1) % 24
        frac = hour - int(hour)
        return (self.TOD_PROFILE[h_low] * (1 - frac) +
                self.TOD_PROFILE[h_high] * frac)

    def _target_sbc(self, node):
        sbc_ids = [n.node_id for n in self.topo.nodes.values()
                   if n.node_type == NodeType.SBC]
        if not sbc_ids:
            return None
        return min(sbc_ids, key=lambda s:
                   nx.shortest_path_length(self.topo.graph, node.node_id, s)
                   if nx.has_path(self.topo.graph, node.node_id, s)
                   else 9999)

    def _path_loss_to_master(self, node):
        target = self._target_sbc(node)
        if target is None:
            return 0.001
        path = self.topo.get_path(node.node_id, target)
        return self.topo.path_loss(path) if len(path) >= 2 else 0.001

    def _path_rtt_to_master(self, node):
        target = self._target_sbc(node)
        if target is None:
            return 4.0
        path = self.topo.get_path(node.node_id, target)
        return 2 * self.topo.path_delay(path) if len(path) >= 2 else 4.0

    def generate_node_traffic(self, node, interval_idx, override_params=None):
        phi, sigma = self.config.ar_phi, self.config.ar_sigma
        x_t = phi * self.ar_state[node.node_id] + sigma * self.rng.standard_normal()
        self.ar_state[node.node_id] = x_t
        ar_mult = np.exp(x_t)

        tod = self.time_of_day_factor(interval_idx)
        base_rate = node.base_rate
        method_dist = dict(node.method_dist)
        resp_dist = dict(node.resp_dist)
        forking_factor = node.forking_factor
        attack_no_ack = 0.0
        covert_redirect = node.covert_redirect_fraction if node.is_compromised else 0.0

        international_fraction = 0.04 + 0.03 * self.rng.random()
        n_unique_destinations = max(3, int(self.rng.normal(15, 5)))

        if override_params:
            if "base_rate_mult" in override_params:
                base_rate *= override_params["base_rate_mult"]
            if "ar_mult_override" in override_params:
                ar_mult = override_params["ar_mult_override"]
            if "method_dist" in override_params:
                method_dist.update(override_params["method_dist"])
            if "resp_dist" in override_params:
                resp_dist.update(override_params["resp_dist"])
            if "international_fraction" in override_params:
                international_fraction = override_params["international_fraction"]
            if "n_unique_destinations" in override_params:
                n_unique_destinations = override_params["n_unique_destinations"]
            if "forking_factor" in override_params:
                forking_factor = override_params["forking_factor"]
            if "attack_no_ack" in override_params:
                attack_no_ack = override_params["attack_no_ack"]
            if "covert_redirect" in override_params:
                covert_redirect = override_params["covert_redirect"]

        m_total = sum(method_dist.values())
        method_dist = {k: v / m_total for k, v in method_dist.items()}
        r_total = sum(resp_dist.values())
        resp_dist = {k: v / r_total for k, v in resp_dist.items()}

        rate = base_rate * tod * ar_mult
        total_messages = max(1, int(self.rng.poisson(max(1, rate))))

        method_counts = {}
        remaining = total_messages
        items = list(method_dist.items())
        for i, (m, p) in enumerate(items):
            if i == len(items) - 1:
                method_counts[m] = remaining
            else:
                c = int(total_messages * p)
                method_counts[m] = c
                remaining -= c

        link_loss = self._path_loss_to_master(node)
        rtt_ms = self._path_rtt_to_master(node)
        # transit_loss: односторонняя потеря узел→SBC для транзитного
        # учёта ACK/200 мастером (Вариант A). Берём канальную потерю пути.
        transit_loss = link_loss

        n_invite = method_counts.get("INVITE", 0)
        eff_forking = forking_factor * (1.0 + covert_redirect)
        n_invite_branches = int(n_invite * eff_forking)
        inv = self.txn_model.simulate_invite_client(
            n_invite, link_loss, rtt_ms, attack_no_ack=attack_no_ack,
            covert_redirect=covert_redirect, transit_loss=transit_loss)


        n_noninvite = sum(method_counts.get(m, 0)
                          for m in NONINVITE_METHODS if m != "ACK")
        ninv = self.txn_model.simulate_noninvite_client(n_noninvite, link_loss)

        forking_amplification = n_invite_branches / max(1, n_invite)

        n_408 = inv["n_invite_timeout"] + ninv["n_noninvite_timeout"]
        resp_total = int(total_messages * 0.9)
        resp_counts = {}
        remaining = resp_total
        items = list(resp_dist.items())
        for i, (rc, p) in enumerate(items):
            if rc == "resp_408":
                resp_counts[rc] = n_408
                remaining -= n_408
                continue
            if i == len(items) - 1:
                resp_counts[rc] = max(0, remaining)
            else:
                c = int(resp_total * p)
                resp_counts[rc] = c
                remaining -= c
        resp_total = sum(resp_counts.values())

        method_ratios = {f"ratio_{m.lower()}":
                         method_counts.get(m, 0) / max(1, total_messages)
                         for m in SIP_METHODS}
        resp_ratios = {f"ratio_{rc}":
                       resp_counts.get(rc, 0) / max(1, resp_total)
                       for rc in SIP_RESP_CLASSES}

        entropy = self._entropy(list(method_counts.values()))
        if override_params and "entropy_mult" in override_params:
            entropy = max(0.0, entropy * override_params["entropy_mult"])

        sorted_m = sorted(
            [method_ratios[f"ratio_{m.lower()}"] for m in SIP_METHODS],
            reverse=True)
        method_dominance = (sorted_m[0] / max(0.001, sum(sorted_m[1:]))
                            if len(sorted_m) > 1 else 0.0)
        method_gini = self._gini(
            [method_ratios[f"ratio_{m.lower()}"] for m in SIP_METHODS])

        invite_r = method_ratios.get("ratio_invite", 0.0)
        ack_r = method_ratios.get("ratio_ack", 0.0)
        invite_ack_ratio = invite_r / max(0.001, ack_r)

        inbound_outbound_ratio = 0.8 + 0.4 * self.rng.random()
        if override_params and "inbound_outbound_ratio" in override_params:
            inbound_outbound_ratio = override_params["inbound_outbound_ratio"]

        result = {
            "total_messages": total_messages,
            "resp_total": resp_total,
            "entropy": entropy,
            "international_fraction": international_fraction,
            "n_unique_destinations": n_unique_destinations,
            "inbound_outbound_ratio": inbound_outbound_ratio,
            "method_dominance_ratio": method_dominance,
            "method_gini": method_gini,
            "invite_ack_ratio": invite_ack_ratio,
            "forking_amplification": forking_amplification,
            "invite_timeout_ratio": inv["invite_timeout_ratio"],
            "invite_mean_retransmits": inv["invite_mean_retransmits"],
            "invite_frac_retransmitted": inv["invite_frac_retransmitted"],
            "dialog_completion_ratio": inv["dialog_completion_ratio"],
            "dangling_dialog_ratio": (inv["n_dangling_dialogs"] /
                                      max(1, inv["n_invite_200"])),
            "noninvite_timeout_ratio": ninv["noninvite_timeout_ratio"],
            "txn_timeout_ratio": n_408 / max(1, total_messages),
            # НОВОЕ (Вариант A): абсолютные счётчики автомата для физпотолка
            # Транзитно наблюдаемые мастером счётчики (Вариант A, физика)
            "n_invite_200": inv["n_invite_200"],
            "n_completed_dialogs": inv["n_completed_dialogs"],
            "n_dangling_dialogs": inv["n_dangling_dialogs"],
            "n_200_transit": inv["n_200_transit"],
            "n_ack_transit": inv["n_ack_transit"],

        }
        result.update(method_ratios)
        result.update(resp_ratios)
        for m in SIP_METHODS:
            result[f"{m.lower()}_count"] = method_counts.get(m, 0)
        for rc in SIP_RESP_CLASSES:
            result[f"{rc}_count"] = resp_counts.get(rc, 0)
        return result

    def add_traffic_load_to_links(self, node, traffic):
        target = self._target_sbc(node)
        if target is None:
            return
        path = self.topo.get_path(node.node_id, target)
        if len(path) < 2:
            return
        retransmit_overhead = 1.0 + 0.3 * traffic.get(
            "invite_frac_retransmitted", 0.0)
        load_mbps = (traffic["total_messages"] * retransmit_overhead *
                     self.config.avg_message_size_bytes * 8) / (
                         self.config.interval_s * 1_000_000.0)
        for i in range(len(path) - 1):
            link = self.topo.get_link(path[i], path[i + 1])
            if link:
                link.current_load_mbps += load_mbps

    @staticmethod
    def _entropy(counts):
        total = sum(counts)
        if total == 0:
            return 0.0
        probs = [c / total for c in counts if c > 0]
        return -sum(p * np.log2(p) for p in probs)

    @staticmethod
    def _gini(values):
        if not values or sum(values) == 0:
            return 0.0
        arr = np.array(sorted(values))
        n = len(arr)
        index = np.arange(1, n + 1)
        s = np.sum(arr)
        return float((2 * np.sum(index * arr) - (n + 1) * s) /
                     (n * s)) if s > 0 else 0.0


# ══════════════════════════════════════════════════════════
#  Control Unit Engine v4 — физический потолок ACK (Вариант A)
# ══════════════════════════════════════════════════════════

class SIPControlUnitEngine:
    """
    v4: закон сохранения автомата воспроизводится, а НЕ постулируется.

    Ключевое отличие от v3.1: удалён искусственный closure_deficit.
    Мастер независимо наблюдает поток ACK (n_ack_observed) на сигнальном
    пути (Вариант A). При маскировке скомпрометированный узел ФИЗИЧЕСКИ
    ограничен: заявленное число замкнутых диалогов не может превысить
    реально прошедшие ACK. Дефицит замыкания Δ=|1-(dcr+ddr)| возникает
    как СЛЕДСТВИЕ этого ограничения (RFC 3261 §13), а не как вписанная
    константа.
    """

    TXN_KEYS = [
        "forking_amplification", "invite_timeout_ratio",
        "invite_mean_retransmits", "invite_frac_retransmitted",
        "dialog_completion_ratio", "dangling_dialog_ratio",
        "noninvite_timeout_ratio", "txn_timeout_ratio",
    ]
    SCALAR_KEYS = [
        "total_messages", "resp_total", "entropy",
        "inbound_outbound_ratio", "international_fraction",
        "n_unique_destinations", "method_dominance_ratio",
        "method_gini", "invite_ack_ratio",
    ]

    def __init__(self, topology, config):
        self.topo = topology
        self.config = config
        self.cu = config.cu_config
        self.rng = np.random.default_rng(config.seed + 100)
        self.last_successful_report = {}
        self.consecutive_cu_losses = defaultdict(int)
        self.intervals_since_last_cu = defaultdict(int)
        self.response_delay_history = defaultdict(list)
        self.response_history_window = 20
        self.honest_history = defaultdict(lambda: deque(maxlen=60))
        self._zone_medians: Dict[int, Dict[str, float]] = {}

    def add_cu_load_to_links(self, master_id, slave_ids):
        cu_load = (self.cu.cu_message_size_bytes * 8) / (
            self.config.interval_s * 1_000_000.0)
        for sid in slave_ids:
            for path in [self.topo.get_path(master_id, sid),
                         self.topo.get_path(sid, master_id)]:
                for i in range(len(path) - 1):
                    link = self.topo.get_link(path[i], path[i + 1])
                    if link:
                        link.current_load_mbps += cu_load

    def set_zone_medians(self, zone_medians):
        self._zone_medians = zone_medians

    def update_honest_history(self, slave_id, true_traffic):
        snap = {k: true_traffic.get(k, 0.0)
                for k in self.SCALAR_KEYS + self.TXN_KEYS}
        self.honest_history[slave_id].append(snap)

    def poll_slave(self, master_id, slave_id, true_traffic, node, interval_idx):
        path_req = self.topo.get_path(master_id, slave_id)
        req_delay, req_loss = self._path_impairment(path_req)
        path_resp = self.topo.get_path(slave_id, master_id)
        resp_delay, resp_loss = self._path_impairment(path_resp)

        processing_delay = self.cu.processing_delay_ms
        if node.is_compromised:
            cr = node.covert_redirect_fraction
            soph = node.attacker_sophistication
            phys_leak = {
                AttackerSophistication.NAIVE: 1.0,
                AttackerSophistication.STATISTICAL: 0.4,
                AttackerSophistication.ADAPTIVE: 0.15,
            }[soph]
            processing_delay += phys_leak * (
                self.rng.uniform(0.1, 0.5) + 1.2 * cr)

        rtt = req_delay + processing_delay + resp_delay
        combined_loss = 1.0 - (1.0 - req_loss) * (1.0 - resp_loss)
        cu_delivered = self.rng.random() > combined_loss

        integrity_ok = True
        if cu_delivered:
            if node.is_compromised:
                soph = node.attacker_sophistication
                fail_prob = {
                    AttackerSophistication.NAIVE: 0.30,
                    AttackerSophistication.STATISTICAL: 0.06,
                    AttackerSophistication.ADAPTIVE: 0.025,
                }[soph]
            else:
                fail_prob = self.cu.integrity_background_fail_prob
            integrity_ok = self.rng.random() > fail_prob

        if cu_delivered:
            if node.is_compromised:
                reported = self._masked_report(
                    true_traffic, node, slave_id, interval_idx)
            else:
                reported = self._honest_report(true_traffic)
                self.update_honest_history(slave_id, true_traffic)
            self.last_successful_report[slave_id] = dict(reported)
            self.consecutive_cu_losses[slave_id] = 0
            self.intervals_since_last_cu[slave_id] = 0
            self.response_delay_history[slave_id].append(
                resp_delay + processing_delay)
            if len(self.response_delay_history[slave_id]) > \
                    self.response_history_window:
                self.response_delay_history[slave_id] = \
                    self.response_delay_history[slave_id][
                        -self.response_history_window:]
        else:
            self.consecutive_cu_losses[slave_id] += 1
            self.intervals_since_last_cu[slave_id] += 1
            reported = (dict(self.last_successful_report[slave_id])
                        if slave_id in self.last_successful_report
                        else self._empty_report())

        hist = self.response_delay_history.get(slave_id, [])
        jitter = float(np.std(hist[-10:])) if len(hist) >= 3 else 0.0

        method_sum = sum(reported.get(f"{m.lower()}_count", 0)
                         for m in SIP_METHODS)
        consistency_error = abs(reported["total_messages"] - method_sum)

        # Закон сохранения автомата: наблюдаемый дефицит замыкания (I1).
        dcr = reported.get("dialog_completion_ratio", 0.0)
        ddr = reported.get("dangling_dialog_ratio", 0.0)
        txn_state_consistency_error = abs(1.0 - (dcr + ddr))

        # ── I2, Вариант A: ТРАНЗИТНОЕ наблюдение мастера ──
        # Мастер-SBC считает 200 OK и ACK на транспортном пути НЕЗАВИСИМО
        # от отчётности узла. Атакующий контролирует reported dcr, но НЕ
        # транзитный dcr_transit = n_ack_transit / n_200_transit.
        # Разрыв = неустранимый физический след covert redirect (RFC §13).
        n_ack_tr = float(true_traffic.get("n_ack_transit", 0))
        n_200_tr = float(true_traffic.get("n_200_transit", 0))
        dcr_transit = n_ack_tr / max(1.0, n_200_tr)
        ack_closure_mismatch = max(0.0, dcr - dcr_transit)


        result = {
            "cu_delivered": 1 if cu_delivered else 0,
            "cu_rtt_delay_ms": rtt if cu_delivered else 0.0,
            "cu_req_delay_ms": req_delay,
            "cu_resp_delay_ms": resp_delay,
            "cu_combined_loss_prob": combined_loss,
            "integrity_check": 1 if integrity_ok else 0,
            "consecutive_cu_losses": self.consecutive_cu_losses[slave_id],
            "cu_processing_delay_ms": processing_delay if cu_delivered else 0.0,
            "cu_response_time_jitter": jitter,
            "cu_staleness": self.intervals_since_last_cu[slave_id],
            "cu_consistency_error": consistency_error,

            # НОВОЕ (Вариант A): физически наблюдаемые инварианты
            "cu_txn_state_consistency_error": txn_state_consistency_error,
            # I2 (Вариант A): транзитный ACK-инвариант — главный рычаг
            "cu_dcr_transit": dcr_transit if cu_delivered else 1.0,
            "cu_ack_closure_mismatch": ack_closure_mismatch if cu_delivered else 0.0,

        }
        for key in self.SCALAR_KEYS + self.TXN_KEYS:
            result[f"reported_{key}"] = reported.get(key, 0.0)
        for m in SIP_METHODS:
            k = f"ratio_{m.lower()}"
            result[f"reported_{k}"] = reported.get(k, 0.0)
        for rc in SIP_RESP_CLASSES:
            k = f"ratio_{rc}"
            result[f"reported_{k}"] = reported.get(k, 0.0)
        return result

    def _path_impairment(self, path):
        total_delay, survive = 0.0, 1.0
        for i in range(len(path) - 1):
            link = self.topo.get_link(path[i], path[i + 1])
            if link:
                total_delay += link.effective_delay_ms
                survive *= (1.0 - link.effective_loss)
            else:
                survive *= 0.5
                total_delay += 5.0
        return total_delay, 1.0 - survive

    def _honest_report(self, true):
        noise = lambda v: max(0, v + self.rng.normal(0, max(1, v * 0.02)))
        r = {}
        for key in ["total_messages", "resp_total"]:
            r[key] = int(noise(true.get(key, 0)))
        for key in (["entropy", "inbound_outbound_ratio",
                     "international_fraction", "n_unique_destinations",
                     "method_dominance_ratio", "method_gini",
                     "invite_ack_ratio"] + self.TXN_KEYS):
            r[key] = true.get(key, 0.0) + self.rng.normal(0, 0.01)
        for m in SIP_METHODS:
            r[f"ratio_{m.lower()}"] = true.get(f"ratio_{m.lower()}", 0.0)
            r[f"{m.lower()}_count"] = true.get(f"{m.lower()}_count", 0)
        for rc in SIP_RESP_CLASSES:
            r[f"ratio_{rc}"] = true.get(f"ratio_{rc}", 0.0)
        return r

    # ────────────────────────────────────────────────────
    #  v4: МАСКИРОВКА — дефицит возникает из физпотолка ACK
    # ────────────────────────────────────────────────────

    def _masked_report(self, true, node, slave_id, interval_idx):
        soph = node.attacker_sophistication
        slip_prob = {
            AttackerSophistication.NAIVE: 0.08,
            AttackerSophistication.STATISTICAL: 0.03,
            AttackerSophistication.ADAPTIVE: 0.015,
        }[soph]
        if self.rng.random() < slip_prob:
            return dict(true)

        if soph == AttackerSophistication.NAIVE:
            return self._mask_naive(true, node, interval_idx)
        elif soph == AttackerSophistication.STATISTICAL:
            return self._mask_statistical(true, node, slave_id, interval_idx)
        else:
            return self._mask_adaptive(true, node, slave_id, interval_idx)

    def _physical_dcr_ceiling(self, true):
        """
        RFC §13: замкнутый диалог требует ACK. Мастер видит ACK на транзите.
        Но АТАКУЮЩИЙ (на уровне приложения узла) знает только собственные
        замкнутые диалоги — не транзитный учёт мастера. Для симуляции его
        рапорта берём истинный автомат: атакующий физически не может
        сгенерировать ACK для уведённых вызовов. Разрыв с ТРАНЗИТОМ мастера
        (вычисляется в poll_slave) и есть неустранимый инвариант.
        """
        n_ack = float(true.get("n_completed_dialogs", 0))
        n_200 = float(true.get("n_invite_200", 0))
        return n_ack / max(1.0, n_200)


    def _mask_naive(self, true, node, interval_idx):
        fake_total = int(node.base_rate * (0.95 + 0.10 * self.rng.random()))
        method_counts = {}
        for m in SIP_METHODS:
            method_counts[f"{m.lower()}_count"] = int(
                fake_total * node.method_dist.get(m, 0.05) *
                (1.0 + self.rng.normal(0, 0.01)))
        if self.rng.random() < 0.12:
            method_counts["other_count"] = method_counts.get(
                "other_count", 0) + int(self.rng.integers(-5, 6))
        time_comp = interval_idx - (node.compromised_since or interval_idx)
        drift = min(0.4, time_comp * 0.0004)
        io_ratio = 0.9 + 0.2 * self.rng.random() + drift + 0.8

        # NAIVE даже не пытается согласовать автомат — рапортует «идеал».
        # Физпотолок игнорируется атакующим, но дефицит виден мастеру
        # через ack_closure_mismatch (dcr заявлен выше реального ACK-потока).
        r = {
            "total_messages": fake_total,
            "resp_total": int(fake_total * 0.9),
            "entropy": 1.6 + self.rng.normal(0, 0.02),
            "inbound_outbound_ratio": io_ratio,
            "international_fraction": 0.04 + 0.02 * self.rng.random(),
            "n_unique_destinations": int(self.rng.normal(15, 2)),
            "method_dominance_ratio": 0.3 + 0.02 * self.rng.random(),
            "method_gini": 0.3 + 0.02 * self.rng.random(),
            "invite_ack_ratio": 1.0 + 0.05 * self.rng.random(),
            "forking_amplification": 1.0 + 0.05 * self.rng.random(),
            "invite_timeout_ratio": 0.01 + 0.01 * self.rng.random(),
            "invite_mean_retransmits": 0.1 + 0.05 * self.rng.random(),
            "invite_frac_retransmitted": 0.05 + 0.02 * self.rng.random(),
            "dialog_completion_ratio": 0.95 + 0.03 * self.rng.random(),
            "dangling_dialog_ratio": 0.10 + 0.05 * self.rng.random(),
            "noninvite_timeout_ratio": 0.01 + 0.01 * self.rng.random(),
            "txn_timeout_ratio": 0.01 + 0.01 * self.rng.random(),
        }
        self._fill_counts_ratios(r, method_counts, node)
        return r

    def _replay_honest_value(self, slave_id, key, fallback):
        hist = self.honest_history.get(slave_id)
        if hist and len(hist) >= 3:
            vals = [h[key] for h in hist if key in h]
            if vals:
                base = vals[-1]
                std = np.std(vals[-20:]) if len(vals) >= 3 else abs(base) * 0.05
                return base + self.rng.normal(0, max(std, abs(base) * 0.02 + 1e-6))
        return fallback

    def _mask_statistical(self, true, node, slave_id, interval_idx):
        """
        STATISTICAL: реплей исторических честных значений + попытка показать
        «здоровый» автомат. КЛЮЧЕВОЕ v4: атакующий хочет показать высокий
        dcr, но физически ограничен реальным ACK-потоком (dcr_ceiling).
        Дефицит НЕ вписывается константой — он равен тому, насколько
        желаемый dcr превышает физпотолок.
        """
        fake_total = int(max(1, self._replay_honest_value(
            slave_id, "total_messages", node.base_rate)))
        r = {"total_messages": fake_total,
             "resp_total": int(fake_total * 0.9)}
        for key in ["entropy", "inbound_outbound_ratio",
                    "international_fraction", "n_unique_destinations",
                    "method_dominance_ratio", "method_gini",
                    "invite_ack_ratio"]:
            r[key] = self._replay_honest_value(slave_id, key, true.get(key, 0.0))

        # Атакующий ЖЕЛАЕТ показать здоровый автомат: dcr≈1, ddr≈мало.
        desired_dcr = 0.95 + 0.02 * self.rng.random()
        fake_ddr = 0.03 + 0.02 * self.rng.random()

        # ФИЗИЧЕСКОЕ ОГРАНИЧЕНИЕ (RFC §13): нельзя заявить больше замкнутых
        # диалогов, чем реально прошло ACK. Дефицит = следствие, не аксиома.
        dcr_ceiling = self._physical_dcr_ceiling(true)
        fake_dcr = min(desired_dcr, dcr_ceiling)

        r["dialog_completion_ratio"] = fake_dcr
        r["dangling_dialog_ratio"] = fake_ddr
        r["forking_amplification"] = (1.0 + 0.05 * self.rng.random() +
                                      0.30 * node.covert_redirect_fraction)
        r["invite_timeout_ratio"] = 0.01 + 0.01 * self.rng.random()
        r["invite_mean_retransmits"] = self._replay_honest_value(
            slave_id, "invite_mean_retransmits", 0.15)
        r["invite_frac_retransmitted"] = self._replay_honest_value(
            slave_id, "invite_frac_retransmitted", 0.06)
        r["noninvite_timeout_ratio"] = 0.01 + 0.01 * self.rng.random()
        r["txn_timeout_ratio"] = 0.01 + 0.01 * self.rng.random()

        method_counts = {}
        for m in SIP_METHODS:
            method_counts[f"{m.lower()}_count"] = int(
                fake_total * node.method_dist.get(m, 0.05) *
                (1.0 + self.rng.normal(0, 0.03)))
        self._fill_counts_ratios(r, method_counts, node)
        return r

    def _mask_adaptive(self, true, node, slave_id, interval_idx):
        """
        ADAPTIVE: как STATISTICAL + подгонка НАБЛЮДАЕМЫХ dcr/ddr под зону.
        Но подгонка ddr под зональную норму лишь УХУДШАЕТ дилемму:
        занижая ddr, атакующий обязан завысить dcr, а dcr уже упёрт в
        физпотолок ACK. Итог: закон сохранения нарушается автоматически,
        БЕЗ вписанного closure_deficit.
        """
        r = self._mask_statistical(true, node, slave_id, interval_idx)
        dcr_ceiling = self._physical_dcr_ceiling(true)

        zmed = self._zone_medians.get(node.zone_id)
        if zmed:
            for key in ["inbound_outbound_ratio", "entropy",
                        "method_dominance_ratio", "invite_ack_ratio",
                        "total_messages"]:
                if key in zmed:
                    target = zmed[key]
                    cur = r.get(key, target)
                    blended = 0.3 * cur + 0.7 * target
                    val = blended + self.rng.normal(0, abs(target) * 0.03 + 1e-6)
                    r[key] = int(max(1, val)) if key == "total_messages" else val
            r["resp_total"] = int(r["total_messages"] * 0.9)

            # Подгонка ddr под зональную медиану (маскировка «висячих»).
            if "dangling_dialog_ratio" in zmed:
                r["dangling_dialog_ratio"] = (
                    0.5 * zmed["dangling_dialog_ratio"] + 0.02 +
                    self.rng.normal(0, 0.01))
            # Атакующий хочет подогнать и dcr под зону, НО ограничен
            # физпотолком ACK. Это и порождает дефицит как следствие.
            if "dialog_completion_ratio" in zmed:
                desired = 0.4 * r["dialog_completion_ratio"] + \
                          0.6 * zmed["dialog_completion_ratio"]
                r["dialog_completion_ratio"] = min(desired, dcr_ceiling)

        return r

    def _fill_counts_ratios(self, r, method_counts, node):
        r.update(method_counts)
        for m in SIP_METHODS:
            r[f"ratio_{m.lower()}"] = (node.method_dist.get(m, 0.05) +
                                       self.rng.normal(0, 0.01))
        for rc in SIP_RESP_CLASSES:
            r[f"ratio_{rc}"] = (node.resp_dist.get(rc, 0.05) +
                                self.rng.normal(0, 0.01))

    def _empty_report(self):
        r = {k: 0 for k in ["total_messages", "resp_total"]}
        r.update({k: 0.0 for k in (
            ["entropy", "inbound_outbound_ratio", "international_fraction",
             "n_unique_destinations", "method_dominance_ratio",
             "method_gini", "invite_ack_ratio"] + self.TXN_KEYS)})
        for m in SIP_METHODS:
            r[f"ratio_{m.lower()}"] = 0.0
            r[f"{m.lower()}_count"] = 0
        for rc in SIP_RESP_CLASSES:
            r[f"ratio_{rc}"] = 0.0
        return r


    

# ──────────────────────────────────────────────────────────
#  Scenario Injector
# ──────────────────────────────────────────────────────────

class SIPScenarioInjector:
    def __init__(self, rng):
        self.rng = rng

    def get_attack_overrides(self, attack):
        intensity = attack.intensity
        atype = attack.attack_type
        params = {}

        if atype == AttackType.SIGNALING_DOS:
            params["base_rate_mult"] = 30.0 + 10.0 * intensity / 10.0
            params["ar_mult_override"] = 1.0
            params["entropy_mult"] = self.rng.uniform(0.3, 0.5)
            params["attack_no_ack"] = 0.6 + 0.03 * intensity
            params["method_dist"] = {
                "INVITE": 0.65 + 0.05 * intensity / 10.0,
                "ACK": 0.03, "BYE": 0.02, "REGISTER": 0.03,
                "OPTIONS": 0.05, "CANCEL": 0.05, "SUBSCRIBE": 0.02,
                "NOTIFY": 0.03, "OTHER": 0.12,
            }
            params["resp_dist"] = {
                "resp_1xx": 0.45, "resp_2xx": 0.10, "resp_3xx": 0.01,
                "resp_4xx": 0.05, "resp_401_407": 0.02,
                "resp_408": 0.10, "resp_5xx": 0.23, "resp_6xx": 0.04,
            }

        elif atype == AttackType.REGISTER_HIJACK:
            params["base_rate_mult"] = 2.0 + 0.5 * intensity / 10.0
            params["entropy_mult"] = self.rng.uniform(0.4, 0.6)
            params["method_dist"] = {
                "REGISTER": 0.55 + 0.05 * intensity / 10.0,
                "INVITE": 0.08, "ACK": 0.05, "BYE": 0.03,
                "OPTIONS": 0.08, "CANCEL": 0.01, "SUBSCRIBE": 0.05,
                "NOTIFY": 0.10, "OTHER": 0.05,
            }
            params["resp_dist"] = {
                "resp_1xx": 0.06, "resp_2xx": 0.42, "resp_3xx": 0.01,
                "resp_4xx": 0.05, "resp_401_407": 0.40 + 0.03 * intensity / 10.0,
                "resp_408": 0.01, "resp_5xx": 0.03, "resp_6xx": 0.02,
            }

        elif atype == AttackType.REGISTER_BRUTEFORCE:
            params["base_rate_mult"] = 1.8 + 0.4 * intensity / 10.0
            params["method_dist"] = {
                "REGISTER": 0.70 + 0.05 * intensity / 10.0,
                "INVITE": 0.03, "ACK": 0.02, "BYE": 0.01,
                "OPTIONS": 0.05, "CANCEL": 0.01, "SUBSCRIBE": 0.03,
                "NOTIFY": 0.05, "OTHER": 0.10,
            }
            params["resp_dist"] = {
                "resp_1xx": 0.04, "resp_2xx": 0.07,
                "resp_401_407": 0.75 + 0.05 * intensity / 10.0,
                "resp_3xx": 0.01, "resp_4xx": 0.05,
                "resp_408": 0.01, "resp_5xx": 0.05, "resp_6xx": 0.02,
            }
            params["n_unique_destinations"] = int(self.rng.integers(1, 3))

        elif atype == AttackType.SPIT:
            params["method_dist"] = {
                "INVITE": 0.55 + 0.05 * intensity / 10.0,
                "ACK": 0.15, "BYE": 0.12, "REGISTER": 0.03,
                "OPTIONS": 0.05, "CANCEL": 0.03, "SUBSCRIBE": 0.02,
                "NOTIFY": 0.02, "OTHER": 0.03,
            }
            params["base_rate_mult"] = 3.0 + 1.0 * intensity / 10.0
            params["international_fraction"] = 0.30 + 0.5 * intensity / 10.0
            params["n_unique_destinations"] = int(self.rng.integers(20, 60))
            params["entropy_mult"] = self.rng.uniform(0.5, 0.7)
            params["forking_factor"] = 1.5 + 0.5 * intensity / 10.0

        elif atype == AttackType.PROXY_COMPROMISE:
            params["covert_redirect"] = attack.covert_redirect_fraction

        return params

    def get_normal_overrides(self, event):
        params = {}
        etype = event.event_type
        if etype == NormalEventType.TRAFFIC_SPIKE:
            params["base_rate_mult"] = 2.0 + 0.5 * event.intensity / 10.0
        elif etype == NormalEventType.MAINTENANCE:
            params["base_rate_mult"] = 0.3
        elif etype == NormalEventType.REROUTE:
            params["method_dist"] = {
                "INVITE": 0.25 + 0.05 * self.rng.random(),
                "REGISTER": 0.20 + 0.05 * self.rng.random(),
            }
        elif etype == NormalEventType.PRESENCE_SURGE:
            params["method_dist"] = {
                "SUBSCRIBE": 0.20 + 0.05 * self.rng.random(),
                "NOTIFY": 0.25 + 0.05 * self.rng.random(),
            }
            params["base_rate_mult"] = 1.4 + 0.3 * event.intensity / 10.0
        return params


# ──────────────────────────────────────────────────────────
#  Main Simulator
# ──────────────────────────────────────────────────────────

class SIPSimulator:
    def __init__(self, config, attacks, normal_events):
        self.config = config
        self.attacks = attacks
        self.normal_events = normal_events
        self.rng = np.random.default_rng(config.seed)
        self.topology = SIPNetworkTopology(config)
        self.traffic_gen = SIPTrafficGenerator(self.topology, config)
        self.cu_engine = SIPControlUnitEngine(self.topology, config)
        self.injector = SIPScenarioInjector(self.rng)
        self.n_intervals = int(config.duration_hours * 3600 / config.interval_s)
        self.records = []
        self.node_history = defaultdict(list)
        self.z_window = 20
        self.node_temporal_history = defaultdict(list)
        self.temporal_window = 30

    def run(self):
        slave_ids_by_master = defaultdict(list)
        for nid, node in self.topology.nodes.items():
            if node.role == NodeRole.SLAVE:
                slave_ids_by_master[node.master_id].append(nid)

        for t in range(self.n_intervals):
            self.topology.reset_link_loads()
            active_attacks = [a for a in self.attacks
                              if a.start_interval <= t < a.end_interval]
            active_events = [e for e in self.normal_events
                             if e.start_interval <= t < e.end_interval]
            self._apply_maintenance(active_events)

            node_traffic = {}
            for nid, node in self.topology.nodes.items():
                if node.role == NodeRole.MASTER:
                    continue
                override = {}
                attack_type = AttackType.NONE
                event_type = NormalEventType.NONE

                for a in active_attacks:
                    if nid in a.target_nodes:
                        override.update(self.injector.get_attack_overrides(a))
                        attack_type = a.attack_type
                        if attack_type == AttackType.PROXY_COMPROMISE:
                            if not node.is_compromised:
                                node.is_compromised = True
                                node.compromised_since = t
                                node.attacker_sophistication = a.sophistication
                                node.covert_redirect_fraction = \
                                    a.covert_redirect_fraction
                for e in active_events:
                    if nid in e.target_nodes:
                        override.update(self.injector.get_normal_overrides(e))
                        event_type = e.event_type

                traffic = self.traffic_gen.generate_node_traffic(
                    node, t, override if override else None)
                traffic["_attack_type"] = attack_type
                traffic["_event_type"] = event_type
                node_traffic[nid] = traffic
                self.traffic_gen.add_traffic_load_to_links(node, traffic)

            zone_medians = self._compute_zone_medians(node_traffic)
            self.cu_engine.set_zone_medians(zone_medians)

            for mid, slaves in slave_ids_by_master.items():
                self.cu_engine.add_cu_load_to_links(mid, slaves)

            for mid, slaves in slave_ids_by_master.items():
                for sid in slaves:
                    node = self.topology.nodes[sid]
                    true_traffic = node_traffic.get(sid, {})
                    attack_type = true_traffic.get("_attack_type", AttackType.NONE)
                    event_type = true_traffic.get("_event_type", NormalEventType.NONE)
                    cu_result = self.cu_engine.poll_slave(
                        mid, sid, true_traffic, node, t)
                    record = self._build_record(
                        t, sid, node, mid, true_traffic, cu_result,
                        attack_type, event_type)
                    self.records.append(record)

            self._reset_maintenance()

    def _compute_zone_medians(self, node_traffic):
        keys = ["inbound_outbound_ratio", "entropy", "method_dominance_ratio",
                "invite_ack_ratio", "total_messages",
                "dialog_completion_ratio", "dangling_dialog_ratio"]
        by_zone = defaultdict(lambda: defaultdict(list))
        for nid, tr in node_traffic.items():
            node = self.topology.nodes[nid]
            if node.is_compromised:
                continue
            for k in keys:
                if k in tr:
                    by_zone[node.zone_id][k].append(tr[k])
        medians = {}
        for zone, d in by_zone.items():
            medians[zone] = {k: float(np.median(v)) for k, v in d.items() if v}
        return medians

    def _build_record(self, t, sid, node, master_id, true_traffic,
                      cu_result, attack_type, event_type):
        obs = {
            "interval": t, "time_s": t * self.config.interval_s,
            "node_id": sid, "node_type": node.node_type.name,
            "transport": node.transport,
            "master_id": master_id, "zone_id": node.zone_id,
        }
        obs.update(cu_result)

        for key in (SIPControlUnitEngine.SCALAR_KEYS +
                    SIPControlUnitEngine.TXN_KEYS):
            obs[f"obs_{key}"] = cu_result.get(f"reported_{key}", 0.0)
        for m in SIP_METHODS:
            k = f"ratio_{m.lower()}"
            obs[f"obs_{k}"] = cu_result.get(f"reported_{k}", 0.0)
        for rc in SIP_RESP_CLASSES:
            k = f"ratio_{rc}"
            obs[f"obs_{k}"] = cu_result.get(f"reported_{k}", 0.0)

        z_keys = ["obs_total_messages", "obs_method_dominance_ratio",
                  "obs_ratio_invite", "obs_ratio_register",
                  "obs_ratio_resp_401_407", "obs_dangling_dialog_ratio",
                  "obs_invite_timeout_ratio", "obs_forking_amplification"]
        self.node_history[sid].append({k: obs.get(k, 0.0) for k in z_keys})
        if len(self.node_history[sid]) > self.z_window:
            self.node_history[sid] = self.node_history[sid][-self.z_window:]
        for key in z_keys:
            hist = [h[key] for h in self.node_history[sid]]
            if len(hist) >= 3:
                vals, cur = hist[:-1], hist[-1]
                m, s = np.mean(vals), np.std(vals)
                obs[f"z_{key}"] = (cur - m) / s if s > 1e-6 else 0.0
            else:
                obs[f"z_{key}"] = 0.0

        self.node_temporal_history[sid].append({
            "obs_total_messages": obs["obs_total_messages"],
            "obs_entropy": obs["obs_entropy"],
            "obs_inbound_outbound_ratio": obs["obs_inbound_outbound_ratio"],
            "obs_method_dominance_ratio": obs["obs_method_dominance_ratio"],
            "obs_invite_ack_ratio": obs["obs_invite_ack_ratio"],
            "obs_n_unique_destinations": obs["obs_n_unique_destinations"],
            "obs_dialog_completion_ratio": obs["obs_dialog_completion_ratio"],
            "obs_dangling_dialog_ratio": obs["obs_dangling_dialog_ratio"],
            "obs_forking_amplification": obs["obs_forking_amplification"],
            "integrity_check": cu_result["integrity_check"],
            "cu_txn_state_consistency_error":
                cu_result["cu_txn_state_consistency_error"],
            "cu_processing_delay_ms": cu_result["cu_processing_delay_ms"],
        })
        if len(self.node_temporal_history[sid]) > self.temporal_window:
            self.node_temporal_history[sid] = \
                self.node_temporal_history[sid][-self.temporal_window:]
        obs.update(self._compute_temporal(sid))

        gt = {}
        for key in ["total_messages", "entropy", "inbound_outbound_ratio",
                    "international_fraction", "n_unique_destinations",
                    "invite_ack_ratio", "dialog_completion_ratio",
                    "dangling_dialog_ratio", "forking_amplification",
                    "invite_timeout_ratio",
                    "n_invite_200", "n_completed_dialogs",
                    "n_dangling_dialogs", "n_200_transit", "n_ack_transit"]:
            gt[f"gt_{key}"] = true_traffic.get(key, 0)


        is_anomaly = 1 if attack_type != AttackType.NONE else 0
        labels = {
            "attack_type": (attack_type.value
                            if isinstance(attack_type, AttackType)
                            else str(attack_type)),
            "normal_event": (event_type.value
                             if isinstance(event_type, NormalEventType)
                             else str(event_type)),
            "is_anomaly": is_anomaly,
            "is_compromised": 1 if node.is_compromised else 0,
            "attacker_sophistication": (
                node.attacker_sophistication.value
                if node.is_compromised else "none"),
        }

        record = {}
        record.update(obs)
        record.update(gt)
        record.update(labels)
        return record

    def _compute_temporal(self, node_id):
        history = self.node_temporal_history[node_id]
        result = {}
        keys = ["var_obs_total_messages", "var_obs_entropy",
                "var_obs_inbound_outbound_ratio",
                "var_obs_method_dominance_ratio", "var_obs_invite_ack_ratio",
                "var_obs_dialog_completion_ratio",
                "var_obs_forking_amplification",
                "autocorr_obs_total_messages", "integrity_fail_rate",
                "obs_destinations_cv", "mean_txn_state_consistency_error",
                "mean_cu_processing_delay_ms"]
        if len(history) < 5:
            return {k: 0.0 for k in keys}

        for key in ["obs_total_messages", "obs_entropy",
                    "obs_inbound_outbound_ratio",
                    "obs_method_dominance_ratio", "obs_invite_ack_ratio",
                    "obs_dialog_completion_ratio",
                    "obs_forking_amplification"]:
            vals = [h[key] for h in history]
            m, s = np.mean(vals), np.std(vals)
            result[f"var_{key}"] = s / (abs(m) + 1e-9)

        vals = [h["obs_total_messages"] for h in history]
        if len(vals) >= 5 and np.std(vals) > 1e-6:
            arr = np.array(vals, dtype=float)
            m = np.mean(arr)
            d = np.sum((arr - m) ** 2)
            result["autocorr_obs_total_messages"] = (
                np.sum((arr[1:] - m) * (arr[:-1] - m)) / d if d > 1e-9 else 0.0)
        else:
            result["autocorr_obs_total_messages"] = 0.0

        result["integrity_fail_rate"] = 1.0 - np.mean(
            [h["integrity_check"] for h in history])
        dest = [h["obs_n_unique_destinations"] for h in history]
        result["obs_destinations_cv"] = np.std(dest) / (abs(np.mean(dest)) + 1e-9)
        result["mean_txn_state_consistency_error"] = float(np.mean(
            [h["cu_txn_state_consistency_error"] for h in history]))
        result["mean_cu_processing_delay_ms"] = float(np.mean(
            [h["cu_processing_delay_ms"] for h in history]))
        return result

    def _apply_maintenance(self, active_events):
        for event in active_events:
            if isinstance(event, NormalScenario) and \
                    event.event_type == NormalEventType.MAINTENANCE:
                for nid in event.target_nodes:
                    for (a, b), link in self.topology.links.items():
                        if a == nid or b == nid:
                            link.maintenance_loss_mult = 3.0
                            link.maintenance_delay_mult = 2.0

    def _reset_maintenance(self):
        for link in self.topology.links.values():
            link.maintenance_loss_mult = 1.0
            link.maintenance_delay_mult = 1.0

    def to_dataframe(self):
        return pd.DataFrame(self.records)


# ──────────────────────────────────────────────────────────
#  Zone deviation features
# ──────────────────────────────────────────────────────────

def add_zone_deviation_features(df):
    for col in ["obs_total_messages", "obs_ratio_invite",
                "obs_ratio_register", "obs_dialog_completion_ratio",
                "obs_forking_amplification", "obs_dangling_dialog_ratio"]:
        if col not in df.columns:
            continue
        med = df.groupby(["interval", "zone_id"])[col].transform("median")
        df[f"dev_{col.replace('obs_', '')}_vs_zone"] = df[col] - med

    if "obs_inbound_outbound_ratio" in df.columns:
        df["zone_io_ratio_rank"] = df.groupby(["interval", "zone_id"])[
            "obs_inbound_outbound_ratio"].rank(pct=True)
        med = df.groupby(["interval", "zone_id"])[
            "obs_inbound_outbound_ratio"].transform("median")
        df["dev_io_ratio_vs_zone"] = df["obs_inbound_outbound_ratio"] - med

    if "integrity_fail_rate" in df.columns:
        zmean = df.groupby(["interval", "zone_id"])[
            "integrity_fail_rate"].transform("mean")
        df["dev_integrity_fail_rate_vs_zone"] = (
            df["integrity_fail_rate"] - zmean)

    if "var_obs_total_messages" in df.columns:
        zmed = df.groupby(["interval", "zone_id"])[
            "var_obs_total_messages"].transform("median")
        df["dev_var_total_vs_zone"] = df["var_obs_total_messages"] - zmed

    if "mean_txn_state_consistency_error" in df.columns:
        zmean = df.groupby(["interval", "zone_id"])[
            "mean_txn_state_consistency_error"].transform("mean")
        df["dev_txn_consistency_vs_zone"] = (
            df["mean_txn_state_consistency_error"] - zmean)

    if "mean_cu_processing_delay_ms" in df.columns:
        zmed = df.groupby(["interval", "zone_id"])[
            "mean_cu_processing_delay_ms"].transform("median")
        df["dev_processing_delay_vs_zone"] = (
            df["mean_cu_processing_delay_ms"] - zmed)

    if "cu_ack_closure_mismatch" in df.columns:
        zmed = df.groupby(["interval", "zone_id"])[
            "cu_ack_closure_mismatch"].transform("median")
        df["dev_ack_mismatch_vs_zone"] = (
            df["cu_ack_closure_mismatch"] - zmed)
        
    return df


# ──────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────

def main():
    config = SimulationConfig(
        duration_hours=24.0, interval_s=20.0,
        n_sbc=2, n_proxy=3, n_registrar=2, n_ua_gw=4, n_redirect=2,
        master_node_ids=[0, 1], seed=42,
    )
    n_intervals = int(config.duration_hours * 3600 / config.interval_s)

    proxy_ids = list(range(2, 5))
    registrar_ids = list(range(5, 7))
    ua_ids = list(range(7, 11))
    redirect_ids = list(range(11, 13))

    attacks = [
        AttackScenario(AttackType.SIGNALING_DOS, [7, 8], 600, 660, intensity=8.0),
        AttackScenario(AttackType.SIGNALING_DOS, [3], 2000, 2040, intensity=9.0),
        AttackScenario(AttackType.REGISTER_HIJACK, [5], 400, 470, intensity=7.0),
        AttackScenario(AttackType.REGISTER_HIJACK, [6], 1800, 1860, intensity=8.0),
        AttackScenario(AttackType.REGISTER_BRUTEFORCE, [6], 800, 880, intensity=7.0),
        AttackScenario(AttackType.REGISTER_BRUTEFORCE, [5], 2300, 2360, intensity=8.0),
        AttackScenario(AttackType.SPIT, [9], 300, 400, intensity=7.0),
        AttackScenario(AttackType.SPIT, [10], 1500, 1580, intensity=6.0),
        AttackScenario(AttackType.PROXY_COMPROMISE, [4], 500, 1200,
                       intensity=5.0,
                       sophistication=AttackerSophistication.NAIVE,
                       covert_redirect_fraction=0.10),
        AttackScenario(AttackType.PROXY_COMPROMISE, [2], 1300, 2000,
                       intensity=5.0,
                       sophistication=AttackerSophistication.STATISTICAL,
                       covert_redirect_fraction=0.15),
        AttackScenario(AttackType.PROXY_COMPROMISE, [9, 10], 1900, 2800,
                       intensity=6.0,
                       sophistication=AttackerSophistication.ADAPTIVE,
                       covert_redirect_fraction=0.20),
    ]

    normal_events = [
        NormalScenario(NormalEventType.TRAFFIC_SPIKE, [7, 8], 100, 140, intensity=5.0),
        NormalScenario(NormalEventType.TRAFFIC_SPIKE, ua_ids, 2900, 2940, intensity=4.0),
        NormalScenario(NormalEventType.MAINTENANCE, [6], 1400, 1460, intensity=3.0),
        NormalScenario(NormalEventType.REROUTE, [2, 3], 2100, 2150, intensity=3.0),
        NormalScenario(NormalEventType.PRESENCE_SURGE, registrar_ids, 1000, 1060, intensity=5.0),
    ]

    print("=" * 66)
    print("  SIP Master-Slave Simulator v3.1 (txn-integrity dominant)")
    print("  Threat model: NAIVE / STATISTICAL / ADAPTIVE attacker mimicry")
    print("=" * 66)
    total_nodes = (config.n_sbc + config.n_proxy + config.n_registrar +
                   config.n_ua_gw + config.n_redirect)
    print(f"Nodes: {total_nodes} | Masters: {config.master_node_ids}")
    print(f"Intervals: {n_intervals} ({config.duration_hours}h, "
          f"Δt={config.interval_s}s)")
    print(f"Attacks: {len(attacks)} (3 compromise sophistication levels)\n")

    sim = SIPSimulator(config, attacks, normal_events)
    sim.run()

    df = sim.to_dataframe()
    df = add_zone_deviation_features(df)

    from sklearn.preprocessing import LabelEncoder
    df["node_type_enc"] = LabelEncoder().fit_transform(df["node_type"])

    print(f"Total records: {len(df)}")
    print(f"Anomalies: {df['is_anomaly'].sum()} "
          f"({100 * df['is_anomaly'].mean():.2f}%)")
    print(f"\nAttack distribution:\n{df['attack_type'].value_counts()}")
    if "attacker_sophistication" in df.columns:
        print(f"\nCompromise by sophistication:\n"
              f"{df[df['is_compromised']==1]['attacker_sophistication'].value_counts()}")

    print("\n--- Compromise detectability by sophistication level ---")
    print("(средние значения остаточных рычагов: чем ближе к норме, "
          "тем сложнее обнаружить)")
    cols = ["cu_txn_state_consistency_error", "dev_txn_consistency_vs_zone",
            "dev_io_ratio_vs_zone", "var_obs_inbound_outbound_ratio",
            "dev_processing_delay_vs_zone", "obs_dangling_dialog_ratio"]
    norm = df[df["is_anomaly"] == 0]
    for col in cols:
        if col not in df.columns:
            continue
        nm = norm[col].mean()
        line = f"  {col:38s} norm={nm:8.4f}"
        for soph in ["naive", "statistical", "adaptive"]:
            m = (df["attacker_sophistication"] == soph)
            v = df.loc[m, col].mean() if m.sum() > 0 else float("nan")
            line += f"  {soph[:4]}={v:8.4f}"
        print(line)

    output = "sip_dataset_v4.csv"
    df.to_csv(output, index=False)
    print(f"\nDataset saved to {output}\nShape: {df.shape}")

    print("\n--- Проверка инварианта I2 (транзитный ACK, Вариант A) ---")
    df["_M"] = df["cu_ack_closure_mismatch"]
    for soph in ["none", "naive", "statistical", "adaptive"]:
        m = df["attacker_sophistication"] == soph
        if m.sum() > 0:
            print(f"  {soph:12s}: E[M(I2)]={df.loc[m,'_M'].mean():.4f}  "
                  f"E[Δ(I1)]={(1.0-(df.loc[m,'obs_dialog_completion_ratio']+df.loc[m,'obs_dangling_dialog_ratio'])).abs().mean():.4f}")


    return df


if __name__ == "__main__":
    df = main()
