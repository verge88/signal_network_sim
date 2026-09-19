"""
SS7 Master-Slave Signalling Network Simulator v6
=================================================
Улучшения для обнаружения компрометации слейва:
  1. Добавлены микро-утечки в маскированные отчёты (реалистичная неидеальная маскировка)
  2. Добавлены временны́е признаки: дисперсия и автокорреляция сообщённых значений
  3. Добавлены кросс-корреляционные признаки между слейвами зоны
  4. Добавлен признак «стеленности» (staleness) при потере КЕ
  5. Добавлена consistency-проверка: мастер сравнивает obs_total ≈ obs_map + obs_isup + obs_tcap
  6. Компрометированный узел периодически «проскальзывает» с реальными значениями
"""

import json
import numpy as np
import pandas as pd
import networkx as nx
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set
from collections import defaultdict
import warnings
import os

_GUI_TOPOLOGY = None

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────
#  Enums (без изменений)
# ──────────────────────────────────────────────────────────

class NodeType(Enum):
    SP = auto()
    STP = auto()
    SCP = auto()

class NodeRole(Enum):
    MASTER = auto()
    SLAVE = auto()

class AttackType(Enum):
    NONE = "none"
    SMS_INTERCEPT = "sms_intercept"
    LOCATION_TRACK = "location_track"
    SIGNALING_DOS = "signaling_dos"
    IRSF = "irsf"
    SLAVE_COMPROMISE = "slave_compromise"

class NormalEventType(Enum):
    NONE = "none"
    TRAFFIC_SPIKE = "traffic_spike"
    MAINTENANCE = "maintenance"
    REROUTE = "reroute"

# ──────────────────────────────────────────────────────────
#  Data Classes
# ──────────────────────────────────────────────────────────

@dataclass
class SignallingNode:
    node_id: int
    node_type: NodeType
    role: NodeRole = NodeRole.SLAVE
    point_code: str = ""
    master_id: Optional[int] = None
    zone_id: int = 0
    base_rate: float = 100.0
    msg_type_dist: Dict[str, float] = field(default_factory=lambda: {
        "ISUP": 0.55, "MAP": 0.30, "TCAP": 0.15
    })
    map_subtype_dist: Dict[str, float] = field(default_factory=lambda: {
        "SRI": 0.25, "SRI_SM": 0.18, "PSI": 0.12,
        "ATI": 0.08, "UPDATE_LOC": 0.17, "INSERT_SUB": 0.10,
        "SEND_AUTH": 0.05, "OTHER_MAP": 0.05
    })
    is_compromised: bool = False
    compromised_since: Optional[int] = None

@dataclass
class SignallingLink:
    src: int
    dst: int
    capacity_kbps: float = 64.0
    propagation_delay_s: float = 0.002
    base_loss_prob: float = 0.001
    current_load_kbps: float = 0.0
    maintenance_loss_mult: float = 1.0
    maintenance_delay_mult: float = 1.0

    @property
    def rho(self) -> float:
        if self.capacity_kbps <= 0:
            return 999.0
        return self.current_load_kbps / self.capacity_kbps

    @property
    def effective_delay(self) -> float:
        base = self.propagation_delay_s * self.maintenance_delay_mult
        r = self.rho
        if r > 0.8:
            base *= (1.0 + 2.0 * (r - 0.8) / 0.2)
        if r > 1.0:
            base *= (1.0 + 5.0 * (r - 1.0))
        return base

    @property
    def effective_loss(self) -> float:
        p = self.base_loss_prob * self.maintenance_loss_mult
        r = self.rho
        if r > 0.85:
            p += 0.05 * ((r - 0.85) / 0.15) ** 2
        if r > 1.0:
            p += 0.3 * (r - 1.0)
        return min(p, 0.95)

@dataclass
class ControlUnitConfig:
    poll_interval_s: float = 30.0
    cu_message_size_bytes: int = 50
    processing_delay_s: float = 0.005
    integrity_background_fail_prob: float = 0.015

@dataclass
class AttackScenario:
    attack_type: AttackType
    target_nodes: List[int]
    start_interval: int
    end_interval: int
    intensity: float = 1.0

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
    interval_s: float = 30.0
    n_sp: int = 15
    n_stp: int = 3
    n_scp: int = 2
    master_node_ids: List[int] = field(default_factory=lambda: [0, 1])
    stp_capacity_kbps: float = 128.0
    sp_capacity_kbps: float = 64.0
    scp_capacity_kbps: float = 64.0
    ar_phi: float = 0.85
    ar_sigma: float = 0.18
    avg_message_size_bytes: int = 100
    seed: int = 42
    cu_config: ControlUnitConfig = field(default_factory=ControlUnitConfig)

# ──────────────────────────────────────────────────────────
#  Topology Builder (без изменений)
# ──────────────────────────────────────────────────────────

class SS7NetworkTopology:
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.graph = nx.Graph()
        self.nodes: Dict[int, SignallingNode] = {}
        self.links: Dict[Tuple[int, int], SignallingLink] = {}
        self._build()

    def _build(self):
        rng = np.random.default_rng(self.config.seed)
        node_id = 0

        stp_ids = []
        for _ in range(self.config.n_stp):
            n = SignallingNode(
                node_id=node_id, node_type=NodeType.STP,
                point_code=f"STP-{node_id:03d}",
                base_rate=rng.uniform(150, 300)
            )
            self.nodes[node_id] = n
            self.graph.add_node(node_id)
            stp_ids.append(node_id)
            node_id += 1

        scp_ids = []
        for _ in range(self.config.n_scp):
            n = SignallingNode(
                node_id=node_id, node_type=NodeType.SCP,
                point_code=f"SCP-{node_id:03d}",
                base_rate=rng.uniform(50, 120)
            )
            self.nodes[node_id] = n
            self.graph.add_node(node_id)
            scp_ids.append(node_id)
            node_id += 1

        sp_ids = []
        for _ in range(self.config.n_sp):
            n = SignallingNode(
                node_id=node_id, node_type=NodeType.SP,
                point_code=f"SP-{node_id:03d}",
                base_rate=rng.uniform(60, 200)
            )
            self.nodes[node_id] = n
            self.graph.add_node(node_id)
            sp_ids.append(node_id)
            node_id += 1

        for i, s1 in enumerate(stp_ids):
            for s2 in stp_ids[i + 1:]:
                self._add_link(s1, s2, self.config.stp_capacity_kbps,
                               delay=0.001 + rng.uniform(0, 0.002))

        for scp in scp_ids:
            for stp in stp_ids:
                self._add_link(scp, stp, self.config.scp_capacity_kbps,
                               delay=0.002 + rng.uniform(0, 0.002))

        for sp in sp_ids:
            chosen_stps = rng.choice(stp_ids, size=min(2, len(stp_ids)), replace=False)
            for stp in chosen_stps:
                self._add_link(sp, stp, self.config.sp_capacity_kbps,
                               delay=0.002 + rng.uniform(0, 0.003))

        for mid in self.config.master_node_ids:
            if mid in self.nodes:
                self.nodes[mid].role = NodeRole.MASTER

        for nid, node in self.nodes.items():
            if node.role == NodeRole.MASTER:
                node.master_id = nid
                node.zone_id = nid
                continue
            best_master = None
            best_dist = float("inf")
            for mid in self.config.master_node_ids:
                try:
                    d = nx.shortest_path_length(self.graph, nid, mid)
                    if d < best_dist:
                        best_dist = d
                        best_master = mid
                except nx.NetworkXNoPath:
                    pass
            node.master_id = best_master if best_master is not None else self.config.master_node_ids[0]
            node.zone_id = node.master_id

    def _add_link(self, src: int, dst: int, capacity: float, delay: float = 0.002):
        key = (min(src, dst), max(src, dst))
        if key not in self.links:
            link = SignallingLink(
                src=key[0], dst=key[1],
                capacity_kbps=capacity,
                propagation_delay_s=delay
            )
            self.links[key] = link
            self.graph.add_edge(key[0], key[1])

    def get_link(self, a: int, b: int) -> Optional[SignallingLink]:
        key = (min(a, b), max(a, b))
        return self.links.get(key)

    def get_path(self, src: int, dst: int) -> List[int]:
        try:
            return nx.shortest_path(self.graph, src, dst)
        except nx.NetworkXNoPath:
            return []

    def reset_link_loads(self):
        for link in self.links.values():
            link.current_load_kbps = 0.0

# ──────────────────────────────────────────────────────────
#  Traffic Generator (без изменений)
# ──────────────────────────────────────────────────────────

class TrafficGenerator:
    TOD_PROFILE = [
        0.3, 0.2, 0.15, 0.15, 0.2, 0.4,
        0.7, 0.9, 1.0, 1.0, 0.95, 0.9,
        0.85, 0.9, 0.95, 1.0, 1.0, 0.95,
        0.9, 0.8, 0.7, 0.6, 0.5, 0.4
    ]

    MAP_SUBTYPES = ["SRI", "SRI_SM", "PSI", "ATI", "UPDATE_LOC",
                    "INSERT_SUB", "SEND_AUTH", "OTHER_MAP"]

    def __init__(self, topology: SS7NetworkTopology, config: SimulationConfig):
        self.topo = topology
        self.config = config
        self.rng = np.random.default_rng(config.seed + 1)
        self.ar_state: Dict[int, float] = {nid: 0.0 for nid in topology.nodes}

    def time_of_day_factor(self, interval_idx: int) -> float:
        t_seconds = interval_idx * self.config.interval_s
        hour = (t_seconds / 3600.0) % 24.0
        h_low = int(hour) % 24
        h_high = (h_low + 1) % 24
        frac = hour - int(hour)
        return self.TOD_PROFILE[h_low] * (1 - frac) + self.TOD_PROFILE[h_high] * frac

    def generate_node_traffic(
        self, node: SignallingNode, interval_idx: int,
        override_params: Optional[Dict] = None
    ) -> Dict:
        phi = self.config.ar_phi
        sigma = self.config.ar_sigma
        prev_x = self.ar_state[node.node_id]
        x_t = phi * prev_x + sigma * self.rng.standard_normal()
        self.ar_state[node.node_id] = x_t
        ar_multiplier = np.exp(x_t)

        tod = self.time_of_day_factor(interval_idx)

        base_rate = node.base_rate
        msg_dist = dict(node.msg_type_dist)
        map_dist = dict(node.map_subtype_dist)
        international_fraction = 0.03 + 0.02 * self.rng.random()
        n_unique_destinations = max(3, int(self.rng.normal(12, 4)))

        if override_params:
            if "base_rate_mult" in override_params:
                base_rate *= override_params["base_rate_mult"]
            if "ar_multiplier_override" in override_params:
                ar_multiplier = override_params["ar_multiplier_override"]
            if "msg_dist" in override_params:
                msg_dist.update(override_params["msg_dist"])
            if "map_dist" in override_params:
                map_dist.update(override_params["map_dist"])
            if "international_fraction" in override_params:
                international_fraction = override_params["international_fraction"]
            if "n_unique_destinations" in override_params:
                n_unique_destinations = override_params["n_unique_destinations"]

        msg_total = sum(msg_dist.values())
        msg_dist = {k: v / msg_total for k, v in msg_dist.items()}
        map_total = sum(map_dist.values())
        map_dist = {k: v / map_total for k, v in map_dist.items()}

        rate = base_rate * tod * ar_multiplier
        total_messages = max(1, int(self.rng.poisson(max(1, rate))))

        isup_count = int(total_messages * msg_dist.get("ISUP", 0.55))
        map_count = int(total_messages * msg_dist.get("MAP", 0.30))
        tcap_count = total_messages - isup_count - map_count

        map_sub_counts = {}
        remaining_map = map_count
        for i, (st, p) in enumerate(map_dist.items()):
            if i == len(map_dist) - 1:
                map_sub_counts[st] = remaining_map
            else:
                c = int(map_count * p)
                map_sub_counts[st] = c
                remaining_map -= c

        map_ratios = {}
        for st in self.MAP_SUBTYPES:
            cnt = map_sub_counts.get(st, 0)
            map_ratios[f"ratio_map_{st.lower()}"] = cnt / max(1, map_count)

        ratio_vals = [map_ratios[f"ratio_map_{st.lower()}"] for st in self.MAP_SUBTYPES]
        sorted_ratios = sorted(ratio_vals, reverse=True)
        map_dominance = sorted_ratios[0] / max(0.001, sum(sorted_ratios[1:])) if len(sorted_ratios) > 1 else 0.0
        map_top2 = sum(sorted_ratios[:2]) / max(0.001, sum(sorted_ratios[2:])) if len(sorted_ratios) > 2 else 0.0
        map_gini = self._gini(ratio_vals)

        all_counts = [isup_count, map_count, tcap_count]
        entropy = self._entropy(all_counts)

        if total_messages > 1:
            intervals = self.rng.exponential(
                self.config.interval_s / total_messages, size=total_messages
            )
            mean_interval = float(np.mean(intervals))
            std_interval = float(np.std(intervals))
        else:
            mean_interval = self.config.interval_s
            std_interval = 0.0

        if override_params:
            if "entropy_mult" in override_params:
                entropy = max(0.0, entropy * override_params["entropy_mult"])
            if "n_unique_destinations" in override_params:
                n_unique_destinations = override_params["n_unique_destinations"]

        inbound_outbound_ratio = 0.8 + 0.4 * self.rng.random()
        if override_params and "inbound_outbound_ratio" in override_params:
            inbound_outbound_ratio = override_params["inbound_outbound_ratio"]

        result = {
            "total_messages": total_messages,
            "isup_count": isup_count,
            "map_count": map_count,
            "tcap_count": tcap_count,
            "entropy": entropy,
            "mean_interval": mean_interval,
            "std_interval": std_interval,
            "international_fraction": international_fraction,
            "n_unique_destinations": n_unique_destinations,
            "inbound_outbound_ratio": inbound_outbound_ratio,
            "map_dominance_ratio": map_dominance,
            "map_top2_ratio": map_top2,
            "map_gini": map_gini,
        }
        result.update(map_ratios)
        for st in self.MAP_SUBTYPES:
            result[f"map_{st.lower()}_count"] = map_sub_counts.get(st, 0)

        return result

    def add_traffic_load_to_links(self, node: SignallingNode, traffic: Dict):
        stp_ids = [n.node_id for n in self.topo.nodes.values()
                   if n.node_type == NodeType.STP]
        if not stp_ids:
            return
        target_stp = min(stp_ids,
                         key=lambda s: nx.shortest_path_length(
                             self.topo.graph, node.node_id, s)
                         if nx.has_path(self.topo.graph, node.node_id, s) else 9999)
        path = self.topo.get_path(node.node_id, target_stp)
        if len(path) < 2:
            return
        load_kbps = (traffic["total_messages"] * self.config.avg_message_size_bytes * 8) / \
                    (self.config.interval_s * 1000.0)
        for i in range(len(path) - 1):
            link = self.topo.get_link(path[i], path[i + 1])
            if link:
                link.current_load_kbps += load_kbps

    @staticmethod
    def _entropy(counts) -> float:
        total = sum(counts)
        if total == 0:
            return 0.0
        probs = [c / total for c in counts if c > 0]
        return -sum(p * np.log2(p) for p in probs)

    @staticmethod
    def _gini(values) -> float:
        if not values or sum(values) == 0:
            return 0.0
        arr = np.array(sorted(values))
        n = len(arr)
        index = np.arange(1, n + 1)
        return float((2 * np.sum(index * arr) - (n + 1) * np.sum(arr)) /
                      (n * np.sum(arr))) if np.sum(arr) > 0 else 0.0

# ──────────────────────────────────────────────────────────
#  Control Unit Engine v6 — улучшенная модель компрометации
# ──────────────────────────────────────────────────────────

class ControlUnitEngine:
    """
    v6 Улучшения:
    1. Компрометированный узел имеет «проскальзывания» — периодически
       реальные значения просачиваются в отчёт
    2. Маскированные отчёты имеют аномально низкую вариацию (слишком стабильные)
    3. Мастер вычисляет response_time_jitter — у компрометированного узла
       обработка КЕ занимает чуть больше (из-за дополнительного ПО)
    4. Consistency check: obs_total ≈ obs_map + obs_isup + obs_tcap
       (компрометированный узел иногда не согласует эти значения точно)
    """

    def __init__(self, topology: SS7NetworkTopology, config: SimulationConfig):
        self.topo = topology
        self.config = config
        self.cu = config.cu_config
        self.rng = np.random.default_rng(config.seed + 100)
        self.last_successful_report: Dict[int, Dict] = {}
        self.consecutive_cu_losses: Dict[int, int] = defaultdict(int)
        # v6: Счётчик интервалов с момента последнего успешного КЕ
        self.intervals_since_last_cu: Dict[int, int] = defaultdict(int)
        # v6: История задержек ответов для вычисления jitter
        self.response_delay_history: Dict[int, List[float]] = defaultdict(list)
        self.response_history_window = 20

    def add_cu_load_to_links(self, master_id: int, slave_ids: List[int]):
        cu_load_per_msg_kbps = (self.cu.cu_message_size_bytes * 8) / \
                               (self.config.interval_s * 1000.0)
        for sid in slave_ids:
            path_req = self.topo.get_path(master_id, sid)
            for i in range(len(path_req) - 1):
                link = self.topo.get_link(path_req[i], path_req[i + 1])
                if link:
                    link.current_load_kbps += cu_load_per_msg_kbps
            path_resp = self.topo.get_path(sid, master_id)
            for i in range(len(path_resp) - 1):
                link = self.topo.get_link(path_resp[i], path_resp[i + 1])
                if link:
                    link.current_load_kbps += cu_load_per_msg_kbps

    def poll_slave(
        self, master_id: int, slave_id: int,
        true_traffic: Dict, node: SignallingNode,
        interval_idx: int  # v6: добавлен для вычисления длительности компрометации
    ) -> Dict:
        path_req = self.topo.get_path(master_id, slave_id)
        req_delay, req_loss = self._path_impairment(path_req)

        path_resp = self.topo.get_path(slave_id, master_id)
        resp_delay, resp_loss = self._path_impairment(path_resp)

        # v6: Компрометированный узел добавляет обработку (~1-5 мс дополнительно)
        processing_delay = self.cu.processing_delay_s
        if node.is_compromised:
            processing_delay += self.rng.uniform(0.001, 0.005)

        rtt = req_delay + processing_delay + resp_delay

        combined_loss_prob = 1.0 - (1.0 - req_loss) * (1.0 - resp_loss)
        cu_delivered = self.rng.random() > combined_loss_prob

        integrity_ok = True
        if cu_delivered:
            integrity_ok = self.rng.random() > self.cu.integrity_background_fail_prob

        if node.is_compromised and cu_delivered:
            integrity_fail_prob = 0.35
            integrity_ok = self.rng.random() > integrity_fail_prob

        if cu_delivered:
            if node.is_compromised:
                reported = self._generate_masked_report_v6(
                    true_traffic, node, interval_idx
                )
            else:
                reported = self._generate_honest_report(true_traffic)

            self.last_successful_report[slave_id] = dict(reported)
            self.consecutive_cu_losses[slave_id] = 0
            self.intervals_since_last_cu[slave_id] = 0

            # v6: Сохраняем задержку ответа для вычисления jitter
            self.response_delay_history[slave_id].append(resp_delay + processing_delay)
            if len(self.response_delay_history[slave_id]) > self.response_history_window:
                self.response_delay_history[slave_id] = \
                    self.response_delay_history[slave_id][-self.response_history_window:]
        else:
            self.consecutive_cu_losses[slave_id] += 1
            self.intervals_since_last_cu[slave_id] += 1
            if slave_id in self.last_successful_report:
                reported = dict(self.last_successful_report[slave_id])
            else:
                reported = self._empty_report()

        # v6: Вычисляем response_time_jitter
        resp_history = self.response_delay_history.get(slave_id, [])
        if len(resp_history) >= 3:
            response_time_jitter = float(np.std(resp_history[-10:]))
        else:
            response_time_jitter = 0.0

        # v6: Consistency error — разница между obs_total и суммой компонент
        consistency_error = abs(
            reported["total_messages"] -
            (reported["map_count"] + reported["isup_count"] + reported["tcap_count"])
        )

        result = {
            "cu_delivered": 1 if cu_delivered else 0,
            "cu_rtt_delay": rtt if cu_delivered else 0.0,
            "cu_req_delay": req_delay,
            "cu_resp_delay": resp_delay,
            "cu_combined_loss_prob": combined_loss_prob,
            "integrity_check": 1 if integrity_ok else 0,
            "consecutive_cu_losses": self.consecutive_cu_losses[slave_id],
            # v6: Новые CU-признаки
            "cu_processing_delay": processing_delay if cu_delivered else 0.0,
            "cu_response_time_jitter": response_time_jitter,
            "cu_staleness": self.intervals_since_last_cu[slave_id],
            "cu_consistency_error": consistency_error,
            # Reported values
            "reported_total_messages": reported["total_messages"],
            "reported_map_count": reported["map_count"],
            "reported_isup_count": reported["isup_count"],
            "reported_tcap_count": reported["tcap_count"],
            "reported_entropy": reported["entropy"],
            "reported_inbound_outbound_ratio": reported["inbound_outbound_ratio"],
            "reported_international_fraction": reported["international_fraction"],
            "reported_n_unique_destinations": reported["n_unique_destinations"],
            "reported_map_dominance_ratio": reported.get("map_dominance_ratio", 0.0),
            "reported_map_top2_ratio": reported.get("map_top2_ratio", 0.0),
            "reported_map_gini": reported.get("map_gini", 0.0),
        }

        for st in TrafficGenerator.MAP_SUBTYPES:
            key = f"ratio_map_{st.lower()}"
            result[f"reported_{key}"] = reported.get(key, 0.0)

        return result

    def _path_impairment(self, path: List[int]) -> Tuple[float, float]:
        total_delay = 0.0
        survive_prob = 1.0
        for i in range(len(path) - 1):
            link = self.topo.get_link(path[i], path[i + 1])
            if link:
                total_delay += link.effective_delay
                survive_prob *= (1.0 - link.effective_loss)
            else:
                survive_prob *= 0.5
                total_delay += 0.01
        loss_prob = 1.0 - survive_prob
        return total_delay, loss_prob

    def _generate_honest_report(self, true_traffic: Dict) -> Dict:
        noise = lambda v: max(0, v + self.rng.normal(0, max(1, v * 0.02)))
        r = {
            "total_messages": int(noise(true_traffic["total_messages"])),
            "map_count": int(noise(true_traffic["map_count"])),
            "isup_count": int(noise(true_traffic["isup_count"])),
            "tcap_count": int(noise(true_traffic["tcap_count"])),
            "entropy": true_traffic["entropy"] + self.rng.normal(0, 0.01),
            "inbound_outbound_ratio": true_traffic["inbound_outbound_ratio"],
            "international_fraction": true_traffic["international_fraction"],
            "n_unique_destinations": true_traffic["n_unique_destinations"],
            "map_dominance_ratio": true_traffic.get("map_dominance_ratio", 0.0),
            "map_top2_ratio": true_traffic.get("map_top2_ratio", 0.0),
            "map_gini": true_traffic.get("map_gini", 0.0),
        }
        for st in TrafficGenerator.MAP_SUBTYPES:
            key = f"ratio_map_{st.lower()}"
            r[key] = true_traffic.get(key, 0.0)
        return r

    def _generate_masked_report_v6(
        self, true_traffic: Dict, node: SignallingNode, interval_idx: int
    ) -> Dict:
        """
        v6: Улучшенная маскировка с реалистичными микро-утечками.
        
        Ключевые улучшения:
        1. «Проскальзывание» — с вероятностью 8% в каждом интервале часть реальных
           значений просачивается в отчёт (злоумышленник не успевает перехватить)
        2. Маскированные значения слишком стабильные — низкая вариация
           (используется узкий рандомизатор вместо нормального AR-шума)
        3. Иногда компоненты (map+isup+tcap) не точно сходятся к total
        4. Inbound/outbound ratio «дрейфует» — растёт со временем компрометации
        """
        # Вероятность проскальзывания (реальные данные просачиваются)
        slip_prob = 0.08
        is_slipping = self.rng.random() < slip_prob

        if is_slipping:
            # Частичное проскальзывание: некоторые поля реальные, некоторые фейковые
            fake_total = int(node.base_rate * (0.9 + 0.2 * self.rng.random()))
            r = {
                "total_messages": true_traffic["total_messages"],  # РЕАЛЬНОЕ
                "map_count": int(fake_total * node.msg_type_dist.get("MAP", 0.30)),
                "isup_count": int(fake_total * node.msg_type_dist.get("ISUP", 0.55)),
                "tcap_count": fake_total - int(fake_total * 0.30) - int(fake_total * 0.55),
                "entropy": true_traffic["entropy"],  # РЕАЛЬНОЕ
                "inbound_outbound_ratio": true_traffic["inbound_outbound_ratio"],
                "international_fraction": true_traffic["international_fraction"],
                "n_unique_destinations": true_traffic["n_unique_destinations"],
                "map_dominance_ratio": true_traffic.get("map_dominance_ratio", 0.0),
                "map_top2_ratio": true_traffic.get("map_top2_ratio", 0.0),
                "map_gini": true_traffic.get("map_gini", 0.0),
            }
            for st in TrafficGenerator.MAP_SUBTYPES:
                key = f"ratio_map_{st.lower()}"
                r[key] = true_traffic.get(key, 0.0)
            return r

        # Стандартная маскировка — но с аномально малым шумом
        # (нормальный узел имеет AR(1) вариацию, а маскированный — почти константу)
        fake_total = int(node.base_rate * (0.95 + 0.10 * self.rng.random()))
        fake_map = int(fake_total * node.msg_type_dist.get("MAP", 0.30)
                       * (1.0 + self.rng.normal(0, 0.01)))  # очень малый шум
        fake_isup = int(fake_total * node.msg_type_dist.get("ISUP", 0.55)
                        * (1.0 + self.rng.normal(0, 0.01)))
        fake_tcap = fake_total - fake_map - fake_isup

        # v6: Иногда компоненты не сходятся (ошибка маскировки)
        if self.rng.random() < 0.12:
            fake_tcap += int(self.rng.integers(-3, 4))

        # v6: Inbound/outbound ratio дрейфует вверх со временем
        time_compromised = interval_idx - (node.compromised_since or interval_idx)
        drift = min(0.4, time_compromised * 0.0005)  # медленный рост
        io_ratio = 0.9 + 0.2 * self.rng.random() + drift
        # Базовый повышенный io_ratio из-за атаки
        io_ratio += 0.8 + 0.4 * (1.0 / 10.0)  # intensity effect

        r = {
            "total_messages": fake_total,
            "map_count": fake_map,
            "isup_count": fake_isup,
            "tcap_count": fake_tcap,
            "entropy": 1.4 + self.rng.normal(0, 0.02),  # v6: ещё меньше вариации
            "inbound_outbound_ratio": io_ratio,
            "international_fraction": 0.03 + 0.02 * self.rng.random(),
            "n_unique_destinations": int(self.rng.normal(12, 1.5)),  # v6: слишком стабильно
            "map_dominance_ratio": 0.3 + 0.02 * self.rng.random(),  # v6: малая вариация
            "map_top2_ratio": 0.8 + 0.05 * self.rng.random(),
            "map_gini": 0.3 + 0.02 * self.rng.random(),
        }
        for st in TrafficGenerator.MAP_SUBTYPES:
            key = f"ratio_map_{st.lower()}"
            r[key] = node.map_subtype_dist.get(st, 0.1) + self.rng.normal(0, 0.01)
        return r

    def _empty_report(self) -> Dict:
        r = {
            "total_messages": 0,
            "map_count": 0,
            "isup_count": 0,
            "tcap_count": 0,
            "entropy": 0.0,
            "inbound_outbound_ratio": 0.0,
            "international_fraction": 0.0,
            "n_unique_destinations": 0,
            "map_dominance_ratio": 0.0,
            "map_top2_ratio": 0.0,
            "map_gini": 0.0,
        }
        for st in TrafficGenerator.MAP_SUBTYPES:
            r[f"ratio_map_{st.lower()}"] = 0.0
        return r


# ──────────────────────────────────────────────────────────
#  Scenario Injector (без изменений)
# ──────────────────────────────────────────────────────────

class ScenarioInjector:
    def __init__(self, rng: np.random.Generator):
        self.rng = rng

    def get_attack_overrides(self, attack: AttackScenario) -> Dict:
        intensity = attack.intensity
        atype = attack.attack_type
        params: Dict = {}

        if atype == AttackType.SMS_INTERCEPT:
            params["msg_dist"] = {"MAP": 0.65 + 0.05 * intensity / 10.0}
            params["map_dist"] = {
                "SRI_SM": 0.45 + 0.03 * intensity / 10.0,
                "UPDATE_LOC": 0.25 + 0.02 * intensity / 10.0,
                "SRI": 0.05, "PSI": 0.03, "ATI": 0.02,
                "INSERT_SUB": 0.02, "SEND_AUTH": 0.01, "OTHER_MAP": 0.01,
            }
            params["base_rate_mult"] = 2.0 + 0.5 * intensity / 10.0
            params["entropy_mult"] = self.rng.uniform(0.3, 0.5)
            params["n_unique_destinations"] = int(self.rng.integers(1, 4))

        elif atype == AttackType.LOCATION_TRACK:
            params["msg_dist"] = {"MAP": 0.70 + 0.05 * intensity / 10.0}
            params["map_dist"] = {
                "PSI": 0.40 + 0.03 * intensity / 10.0,
                "ATI": 0.30 + 0.02 * intensity / 10.0,
                "SRI": 0.05, "SRI_SM": 0.03, "UPDATE_LOC": 0.03,
                "INSERT_SUB": 0.02, "SEND_AUTH": 0.01, "OTHER_MAP": 0.01,
            }
            params["base_rate_mult"] = 1.5 + 0.3 * intensity / 10.0
            params["entropy_mult"] = self.rng.uniform(0.3, 0.5)
            params["n_unique_destinations"] = int(self.rng.integers(1, 4))

        elif atype == AttackType.SIGNALING_DOS:
            params["base_rate_mult"] = 30.0 + 10.0 * intensity / 10.0
            params["ar_multiplier_override"] = 1.0
            params["entropy_mult"] = self.rng.uniform(0.4, 0.6)

        elif atype == AttackType.IRSF:
            params["msg_dist"] = {"ISUP": 0.85 + 0.05 * intensity / 10.0}
            params["international_fraction"] = 0.25 + 0.6 * intensity / 10.0
            params["n_unique_destinations"] = int(self.rng.integers(20, 60))
            params["base_rate_mult"] = 3.0 + 1.0 * intensity / 10.0

        elif atype == AttackType.SLAVE_COMPROMISE:
            params["inbound_outbound_ratio"] = 1.8 + 0.4 * intensity / 10.0
            params["base_rate_mult"] = 1.1 + 0.15 * intensity / 10.0

        return params

    def get_normal_overrides(self, event: NormalScenario) -> Dict:
        params: Dict = {}
        etype = event.event_type

        if etype == NormalEventType.TRAFFIC_SPIKE:
            params["base_rate_mult"] = 2.0 + 0.5 * event.intensity / 10.0

        elif etype == NormalEventType.MAINTENANCE:
            params["base_rate_mult"] = 0.3

        elif etype == NormalEventType.REROUTE:
            params["msg_dist"] = {
                "ISUP": 0.45 + 0.1 * self.rng.random(),
                "MAP": 0.35 + 0.05 * self.rng.random(),
            }

        return params


# ──────────────────────────────────────────────────────────
#  Main Simulator v6
# ──────────────────────────────────────────────────────────

class SS7Simulator:
    def __init__(
        self,
        config: SimulationConfig,
        attacks: List[AttackScenario],
        normal_events: List[NormalScenario],
    ):
        self.config = config
        self.attacks = attacks
        self.normal_events = normal_events
        self.rng = np.random.default_rng(config.seed)

        self.topology = SS7NetworkTopology(config)
        self.traffic_gen = TrafficGenerator(self.topology, config)
        self.cu_engine = ControlUnitEngine(self.topology, config)
        self.injector = ScenarioInjector(self.rng)

        self.n_intervals = int(config.duration_hours * 3600 / config.interval_s)
        self.records: List[Dict] = []

        self.node_history: Dict[int, List[Dict]] = defaultdict(list)
        self.z_window = 20

        # v6: расширенная история для временны́х признаков
        self.node_temporal_history: Dict[int, List[Dict]] = defaultdict(list)
        self.temporal_window = 30

    def run(self):
        slave_ids_by_master: Dict[int, List[int]] = defaultdict(list)
        for nid, node in self.topology.nodes.items():
            if node.role == NodeRole.SLAVE:
                slave_ids_by_master[node.master_id].append(nid)

        for t in range(self.n_intervals):
            self.topology.reset_link_loads()

            active_attacks = self._active_scenarios(t, self.attacks)
            active_events = self._active_scenarios(t, self.normal_events)

            self._apply_maintenance(active_events)

            node_traffic: Dict[int, Dict] = {}
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
                        if attack_type == AttackType.SLAVE_COMPROMISE:
                            if not node.is_compromised:
                                node.is_compromised = True
                                node.compromised_since = t

                for e in active_events:
                    if nid in e.target_nodes:
                        override.update(self.injector.get_normal_overrides(e))
                        event_type = e.event_type

                traffic = self.traffic_gen.generate_node_traffic(
                    node, t, override if override else None
                )
                traffic["_attack_type"] = attack_type
                traffic["_event_type"] = event_type
                node_traffic[nid] = traffic

                self.traffic_gen.add_traffic_load_to_links(node, traffic)

            for mid, slaves in slave_ids_by_master.items():
                self.cu_engine.add_cu_load_to_links(mid, slaves)

            # Собираем obs-значения текущего интервала для зонных медиан
            interval_obs: Dict[int, Dict] = {}

            for mid, slaves in slave_ids_by_master.items():
                for sid in slaves:
                    node = self.topology.nodes[sid]
                    true_traffic = node_traffic.get(sid, {})
                    attack_type = true_traffic.get("_attack_type", AttackType.NONE)
                    event_type = true_traffic.get("_event_type", NormalEventType.NONE)

                    cu_result = self.cu_engine.poll_slave(
                        mid, sid, true_traffic, node, t  # v6: передаём interval_idx
                    )

                    record = self._build_record(
                        t, sid, node, mid, true_traffic, cu_result,
                        attack_type, event_type
                    )
                    self.records.append(record)
                    interval_obs[sid] = record

            self._reset_maintenance()

    def _build_record(
        self, t: int, sid: int, node: SignallingNode, master_id: int,
        true_traffic: Dict, cu_result: Dict,
        attack_type: AttackType, event_type: NormalEventType
    ) -> Dict:
        obs = {
            "interval": t,
            "time_s": t * self.config.interval_s,
            "node_id": sid,
            "node_type": node.node_type.name,
            "master_id": master_id,
            "zone_id": node.zone_id,
        }

        obs.update(cu_result)

        obs["obs_total_messages"] = cu_result["reported_total_messages"]
        obs["obs_map_count"] = cu_result["reported_map_count"]
        obs["obs_isup_count"] = cu_result["reported_isup_count"]
        obs["obs_tcap_count"] = cu_result["reported_tcap_count"]
        obs["obs_entropy"] = cu_result["reported_entropy"]
        obs["obs_inbound_outbound_ratio"] = cu_result["reported_inbound_outbound_ratio"]
        obs["obs_international_fraction"] = cu_result["reported_international_fraction"]
        obs["obs_n_unique_destinations"] = cu_result["reported_n_unique_destinations"]
        obs["obs_map_dominance_ratio"] = cu_result["reported_map_dominance_ratio"]
        obs["obs_map_top2_ratio"] = cu_result["reported_map_top2_ratio"]
        obs["obs_map_gini"] = cu_result["reported_map_gini"]

        for st in TrafficGenerator.MAP_SUBTYPES:
            key = f"ratio_map_{st.lower()}"
            obs[f"obs_{key}"] = cu_result.get(f"reported_{key}", 0.0)

        # Z-score features
        self.node_history[sid].append({
            "obs_total_messages": obs["obs_total_messages"],
            "obs_map_count": obs["obs_map_count"],
            "obs_map_dominance_ratio": obs["obs_map_dominance_ratio"],
            "obs_ratio_map_sri_sm": obs.get("obs_ratio_map_sri_sm", 0.0),
            "obs_ratio_map_psi": obs.get("obs_ratio_map_psi", 0.0),
        })
        if len(self.node_history[sid]) > self.z_window:
            self.node_history[sid] = self.node_history[sid][-self.z_window:]

        z_scores = self._compute_z_scores(sid)
        obs.update(z_scores)

        # ────── v6: НОВЫЕ ВРЕМЕННЫ́Е ПРИЗНАКИ ──────

        # Сохраняем в расширенную историю
        self.node_temporal_history[sid].append({
            "obs_total_messages": obs["obs_total_messages"],
            "obs_map_count": obs["obs_map_count"],
            "obs_entropy": obs["obs_entropy"],
            "obs_inbound_outbound_ratio": obs["obs_inbound_outbound_ratio"],
            "obs_map_dominance_ratio": obs["obs_map_dominance_ratio"],
            "obs_n_unique_destinations": obs["obs_n_unique_destinations"],
            "integrity_check": cu_result["integrity_check"],
        })
        if len(self.node_temporal_history[sid]) > self.temporal_window:
            self.node_temporal_history[sid] = \
                self.node_temporal_history[sid][-self.temporal_window:]

        temporal = self._compute_temporal_features(sid)
        obs.update(temporal)

        # ────── Ground truth ──────
        gt = {
            "gt_total_messages": true_traffic.get("total_messages", 0),
            "gt_map_count": true_traffic.get("map_count", 0),
            "gt_isup_count": true_traffic.get("isup_count", 0),
            "gt_tcap_count": true_traffic.get("tcap_count", 0),
            "gt_entropy": true_traffic.get("entropy", 0.0),
            "gt_inbound_outbound_ratio": true_traffic.get("inbound_outbound_ratio", 0.0),
            "gt_international_fraction": true_traffic.get("international_fraction", 0.0),
            "gt_n_unique_destinations": true_traffic.get("n_unique_destinations", 0),
        }

        is_anomaly = 1 if attack_type != AttackType.NONE else 0
        labels = {
            "attack_type": attack_type.value if isinstance(attack_type, AttackType) else str(attack_type),
            "normal_event": event_type.value if isinstance(event_type, NormalEventType) else str(event_type),
            "is_anomaly": is_anomaly,
            "is_compromised": 1 if node.is_compromised else 0,
        }

        record = {}
        record.update(obs)
        record.update(gt)
        record.update(labels)

        return record

    def _compute_temporal_features(self, node_id: int) -> Dict:
        """
        v6: Вычисляет временны́е признаки, которые помогают обнаружить компрометацию.
        
        Ключевая идея: компрометированный узел генерирует слишком стабильные отчёты
        (низкая дисперсия), потому что маскировка использует фиксированные значения
        вместо естественного AR(1)-шума. Нормальный узел имеет вариацию из-за
        суточного профиля и стохастического процесса.
        
        Также: низкая автокорреляция (маскированные значения не коррелируют
        между соседними интервалами, в отличие от реального AR(1) с φ=0.85).
        """
        history = self.node_temporal_history[node_id]
        result = {}

        if len(history) < 5:
            result["var_obs_total_messages"] = 0.0
            result["var_obs_entropy"] = 0.0
            result["var_obs_inbound_outbound_ratio"] = 0.0
            result["var_obs_map_dominance_ratio"] = 0.0
            result["autocorr_obs_total_messages"] = 0.0
            result["integrity_fail_rate"] = 0.0
            result["obs_destinations_cv"] = 0.0
            return result

        # Дисперсия (coefficient of variation) — компрометация → аномально низкая
        for key in ["obs_total_messages", "obs_entropy",
                    "obs_inbound_outbound_ratio", "obs_map_dominance_ratio"]:
            vals = [h[key] for h in history]
            mean_val = np.mean(vals)
            std_val = np.std(vals)
            # Coefficient of variation (нормализованная дисперсия)
            cv = std_val / (abs(mean_val) + 1e-9)
            result[f"var_{key}"] = cv

        # Автокорреляция lag-1 для obs_total_messages
        # Нормальный AR(1) φ=0.85 → autocorr ≈ 0.85
        # Маскированный → autocorr ≈ 0 (независимые uniform)
        vals = [h["obs_total_messages"] for h in history]
        if len(vals) >= 5 and np.std(vals) > 1e-6:
            vals_arr = np.array(vals, dtype=float)
            mean_v = np.mean(vals_arr)
            denom = np.sum((vals_arr - mean_v) ** 2)
            if denom > 1e-9:
                numer = np.sum((vals_arr[1:] - mean_v) * (vals_arr[:-1] - mean_v))
                result["autocorr_obs_total_messages"] = numer / denom
            else:
                result["autocorr_obs_total_messages"] = 0.0
        else:
            result["autocorr_obs_total_messages"] = 0.0

        # Integrity fail rate (скользящее среднее)
        integrity_vals = [h["integrity_check"] for h in history]
        result["integrity_fail_rate"] = 1.0 - np.mean(integrity_vals)

        # CV числа уникальных адресатов
        dest_vals = [h["obs_n_unique_destinations"] for h in history]
        mean_d = np.mean(dest_vals)
        std_d = np.std(dest_vals)
        result["obs_destinations_cv"] = std_d / (abs(mean_d) + 1e-9)

        return result

    def _compute_z_scores(self, node_id: int) -> Dict:
        history = self.node_history[node_id]
        result = {}
        if len(history) < 3:
            for key in ["obs_total_messages", "obs_map_count",
                        "obs_map_dominance_ratio",
                        "obs_ratio_map_sri_sm", "obs_ratio_map_psi"]:
                result[f"z_{key}"] = 0.0
            return result

        for key in ["obs_total_messages", "obs_map_count",
                    "obs_map_dominance_ratio",
                    "obs_ratio_map_sri_sm", "obs_ratio_map_psi"]:
            vals = [h[key] for h in history[:-1]]
            current = history[-1][key]
            mean = np.mean(vals)
            std = np.std(vals)
            if std > 1e-6:
                result[f"z_{key}"] = (current - mean) / std
            else:
                result[f"z_{key}"] = 0.0

        return result

    def _active_scenarios(self, t, scenarios):
        return [s for s in scenarios if s.start_interval <= t < s.end_interval]

    def _apply_maintenance(self, active_events):
        for event in active_events:
            if not isinstance(event, NormalScenario):
                continue
            if event.event_type == NormalEventType.MAINTENANCE:
                for nid in event.target_nodes:
                    node = self.topology.nodes.get(nid)
                    if node:
                        for (a, b), link in self.topology.links.items():
                            if a == nid or b == nid:
                                link.maintenance_loss_mult = 3.0
                                link.maintenance_delay_mult = 2.0

    def _reset_maintenance(self):
        for link in self.topology.links.values():
            link.maintenance_loss_mult = 1.0
            link.maintenance_delay_mult = 1.0

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.records)


# ──────────────────────────────────────────────────────────
#  Zone-median + v6: cross-node features
# ──────────────────────────────────────────────────────────

def add_zone_deviation_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    v6: Расширены зонные признаки:
    1. Отклонения от медианы (как раньше)
    2. Ранг внутри зоны — позиция узла по obs_inbound_outbound_ratio
    3. Отклонение integrity_fail_rate от зонного среднего
    """
    obs_cols = ["obs_total_messages", "obs_map_count", "obs_isup_count"]
    for col in obs_cols:
        median_col = f"zone_median_{col}"
        df[median_col] = df.groupby(["interval", "zone_id"])[col].transform("median")
        dev_col = col.replace("obs_", "dev_") + "_vs_zone_median"
        df[dev_col] = df[col] - df[median_col]
        df.drop(columns=[median_col], inplace=True)

    # v6: Ранг по inbound_outbound_ratio внутри зоны (нормализованный 0-1)
    df["zone_io_ratio_rank"] = df.groupby(["interval", "zone_id"])[
        "obs_inbound_outbound_ratio"
    ].rank(pct=True)

    # v6: Отклонение obs_inbound_outbound_ratio от зонной медианы
    zone_io_median = df.groupby(["interval", "zone_id"])[
        "obs_inbound_outbound_ratio"
    ].transform("median")
    df["dev_io_ratio_vs_zone_median"] = df["obs_inbound_outbound_ratio"] - zone_io_median

    # v6: Отклонение integrity_fail_rate от зонного среднего
    if "integrity_fail_rate" in df.columns:
        zone_ifr_mean = df.groupby(["interval", "zone_id"])[
            "integrity_fail_rate"
        ].transform("mean")
        df["dev_integrity_fail_rate_vs_zone"] = df["integrity_fail_rate"] - zone_ifr_mean

    # v6: Отклонение var_obs_total_messages от зонной медианы
    if "var_obs_total_messages" in df.columns:
        zone_var_med = df.groupby(["interval", "zone_id"])[
            "var_obs_total_messages"
        ].transform("median")
        df["dev_var_total_vs_zone"] = df["var_obs_total_messages"] - zone_var_med

    return df


# ──────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────

def main():
    config = SimulationConfig(
        duration_hours=24.0,
        interval_s=30.0,
        n_sp=15,
        n_stp=3,
        n_scp=2,
        master_node_ids=[0, 1],
        seed=42,
    )

    n_intervals = int(config.duration_hours * 3600 / config.interval_s)

    sp_ids = list(range(5, 20))

    attacks = [
        AttackScenario(AttackType.SMS_INTERCEPT, [7, 8], 500, 560, intensity=7.0),
        AttackScenario(AttackType.SMS_INTERCEPT, [12], 1800, 1850, intensity=8.0),
        AttackScenario(AttackType.LOCATION_TRACK, [9], 1000, 1040, intensity=6.0),
        AttackScenario(AttackType.LOCATION_TRACK, [14], 2200, 2230, intensity=7.0),
        AttackScenario(AttackType.SIGNALING_DOS, [10, 11], 700, 750, intensity=8.0),
        AttackScenario(AttackType.SIGNALING_DOS, [6], 2000, 2030, intensity=9.0),
        AttackScenario(AttackType.IRSF, [13], 300, 380, intensity=7.0),
        AttackScenario(AttackType.IRSF, [15, 16], 1500, 1560, intensity=6.0),
        AttackScenario(AttackType.SLAVE_COMPROMISE, [17], 400, 1200, intensity=5.0),
        AttackScenario(AttackType.SLAVE_COMPROMISE, [18, 19], 1600, 2400, intensity=6.0),
    ]

    normal_events = [
        NormalScenario(NormalEventType.TRAFFIC_SPIKE, [5, 6, 7], 100, 130, intensity=5.0),
        NormalScenario(NormalEventType.TRAFFIC_SPIKE, sp_ids[:5], 2500, 2530, intensity=4.0),
        NormalScenario(NormalEventType.MAINTENANCE, [8], 1300, 1350, intensity=3.0),
        NormalScenario(NormalEventType.REROUTE, [10, 11], 1900, 1940, intensity=3.0),
    ]

    print("=" * 60)
    print("  SS7 Master-Slave Simulator v6")
    print("  Улучшения: temporal features, micro-leaks, consistency")
    print("=" * 60)
    print(f"Nodes: {config.n_sp} SP + {config.n_stp} STP + {config.n_scp} SCP")
    print(f"Masters: {config.master_node_ids}")
    print(f"Intervals: {n_intervals} ({config.duration_hours}h, Δt={config.interval_s}s)")
    print(f"Attacks: {len(attacks)}, Normal events: {len(normal_events)}")
    print()

    sim = SS7Simulator(config, attacks, normal_events)
    sim.run()

    df = sim.to_dataframe()
    df = add_zone_deviation_features(df)

    print(f"Total records: {len(df)}")
    print(f"Anomalies: {df['is_anomaly'].sum()} ({100*df['is_anomaly'].mean():.2f}%)")
    print(f"\nAttack distribution:")
    print(df["attack_type"].value_counts())

    # v6: Показываем контраст для новых признаков
    print("\n--- Feature contrast v6 (anomaly vs normal) ---")
    v6_features = [
        "obs_inbound_outbound_ratio", "integrity_fail_rate",
        "var_obs_total_messages", "var_obs_entropy",
        "autocorr_obs_total_messages", "cu_consistency_error",
        "cu_processing_delay", "cu_response_time_jitter",
        "cu_staleness", "zone_io_ratio_rank",
        "dev_io_ratio_vs_zone_median", "dev_var_total_vs_zone",
        "obs_destinations_cv",
    ]
    for col in v6_features:
        if col in df.columns:
            norm_mean = df.loc[df["is_anomaly"] == 0, col].mean()
            anom_mean = df.loc[df["is_anomaly"] == 1, col].mean()
            # Отдельно для компрометации
            comp_mask = df["attack_type"] == "slave_compromise"
            comp_mean = df.loc[comp_mask, col].mean() if comp_mask.sum() > 0 else 0.0
            print(f"  {col:45s}  normal={norm_mean:8.4f}  anomaly={anom_mean:8.4f}  "
                  f"compromise={comp_mean:8.4f}")

    output_path = "ss7_dataset_v6.csv"
    df.to_csv(output_path, index=False)
    print(f"\nDataset saved to {output_path}")
    print(f"Shape: {df.shape}")

    return df


# ---------------------------------------------------------------------------
# Исправленный backend
# ---------------------------------------------------------------------------
# Реализация из fix.py разделена по логическим модулям внутри одного файла.
# Этот фасад сохраняет имя текущего симулятора и контракт main() -> DataFrame,
# которым пользуется ss7/ms_multiseed_core.py.
try:
    from . import fix as _fixed
except ImportError:
    import fix as _fixed

RngBook = _fixed.RngBook
Priors = _fixed.Priors
SimConfig = _fixed.SimConfig
SimulationConfig = _fixed.SimConfig
FixedNodeType = _fixed.NodeType
FixedNodeRole = _fixed.NodeRole
KeyCustody = _fixed.KeyCustody
Sophistication = _fixed.Sophistication
Ss7Sim = _fixed.Ss7Sim
Episode = _fixed.Episode
FEATURE_SETS = _fixed.FEATURE_SETS

_NTYPE_MAP = {
    "STP": _fixed.NodeType.STP,
    "SCP": _fixed.NodeType.SCP,
    "HLR": _fixed.NodeType.SCP,
    "SP": _fixed.NodeType.SP,
    "MSC": _fixed.NodeType.SP,
    "VLR": _fixed.NodeType.SP,
    "IGW": _fixed.NodeType.IGW,
}


def set_gui_topology(topology) -> None:
    """Вызывается из gui/adapters перед main(); None — сбросить."""
    global _GUI_TOPOLOGY
    _GUI_TOPOLOGY = topology


def topology_from_gui(gui, *, link_kbps: float = 64.0, buffer_msgs: int = 400,
                      n_external: int = 4, log=print) -> "_fixed.Topology":
    """gui — объект gui.model.Topology (нужны .nodes и .links)."""
    Node = _fixed.Node
    Linkset = _fixed.Linkset
    NodeType = _fixed.NodeType
    NodeRole = _fixed.NodeRole

    zmap = {z: i for i, z in enumerate(sorted({int(n.zone_id) for n in gui.nodes.values()}))}

    nodes, ext, igw = [], [], []
    for n in sorted(gui.nodes.values(), key=lambda x: x.node_id):
        extra = dict(getattr(n, "extra", {}) or {})
        raw = str(extra.get("ss7_type") or n.node_type).upper()
        is_ext = bool(extra.get("external")) or raw in ("EXT", "EXTERNAL")
        ntype = NodeType.IGW if is_ext else _NTYPE_MAP.get(raw)
        if ntype is None:
            raise ValueError(f"узел {n.node_id}: тип {raw!r} не отображается на SS7")
        zone = -2 if is_ext else (-1 if ntype is NodeType.IGW else zmap[int(n.zone_id)])
        role = NodeRole.MASTER if str(n.role).upper() == "MASTER" else NodeRole.SLAVE
        nid = int(n.node_id)
        nodes.append(Node(nid, getattr(n, "hostname", "") or f"{raw}{nid}",
                          ntype, role, zone, is_ext))
        if is_ext:
            ext.append(nid)
        elif ntype is NodeType.IGW:
            igw.append(nid)

    links = []
    for lk in gui.links.values():
        mbps = float(getattr(lk, "capacity_mbps", 0.064) or 0.064)
        n_links = max(1, min(16, int(round(mbps * 1000.0 / link_kbps))))
        links.append(Linkset(int(lk.src), int(lk.dst), n_links, link_kbps, buffer_msgs))

    free = max((n.nid for n in nodes), default=-1) + 1
    if not igw:
        stp = [n.nid for n in nodes if n.ntype is NodeType.STP]
        if not stp:
            raise ValueError("нет ни одного STP — не к чему подключить интерконнект")
        for i in range(2):
            nodes.append(Node(free, f"IGW{i}", NodeType.IGW, NodeRole.SLAVE, -1, False))
            igw.append(free)
            free += 1
        for g in igw:
            for s in stp:
                links.append(Linkset(min(g, s), max(g, s), 3, link_kbps, buffer_msgs))
        links.append(Linkset(min(igw), max(igw), 4, link_kbps, buffer_msgs))
        log(f"добавлены служебные IGW {igw}: B-линки к STP {stp}")
    if not ext:
        for i in range(int(n_external)):
            nodes.append(Node(free, f"EXT{i}", NodeType.IGW, NodeRole.SLAVE, -2, True))
            ext.append(free)
            free += 1
        for e in ext:
            for g in igw:
                links.append(Linkset(min(e, g), max(e, g), 2, link_kbps, buffer_msgs))
        log(f"добавлены внешние партнёры {ext}: источник SRI-SM/MT-FSM")

    topo = _fixed.Topology.from_spec(nodes, links)
    log(f"SS7-топология: {len(topo.nodes)} узлов, {len(topo.linksets)} linkset'ов, "
        f"зон {len(topo.zone_sp)}, HLR {topo.hlr}, внешних {topo.external}, "
        f"мастера {topo.masters}")
    return topo


def topology_spec(topo) -> dict:
    return {
        "nodes": [{"nid": n.nid, "name": n.name, "ntype": n.ntype.name,
                   "role": n.role.name, "zone": n.zone, "external": n.external}
                  for n in topo.nodes.values()],
        "linksets": [{"a": l.a, "b": l.b, "n_links": l.n_links,
                      "link_kbps": l.link_kbps, "buffer_msgs": l.buffer_msgs}
                     for l in topo.linksets.values()],
    }


def topology_from_spec_dict(spec: dict):
    Node, Linkset = _fixed.Node, _fixed.Linkset
    nodes = [Node(int(d["nid"]), d["name"], _fixed.NodeType[d["ntype"]],
                  _fixed.NodeRole[d["role"]], int(d["zone"]), bool(d["external"]))
             for d in spec["nodes"]]
    links = [Linkset(int(d["a"]), int(d["b"]), int(d["n_links"]),
                     float(d["link_kbps"]), int(d["buffer_msgs"]))
             for d in spec["linksets"]]
    return _fixed.Topology.from_spec(nodes, links)


def main(config=None, topology=None, argv=None):
    """Generate the corrected SS7 dataset while preserving the old API."""
    src = topology if topology is not None else _GUI_TOPOLOGY
    params = dict(getattr(src, "sim_params", {}) or {}) if src is not None else {}
    if isinstance(config, _fixed.SimConfig):
        cfg = config
    else:
        cfg = _fixed.SimConfig(
            seed=int(params.get("seed", 42)),
            days=float(params.get("duration_hours", 144.0)) / 24.0)
    topo = None
    if src is not None:
        topo = src if isinstance(src, _fixed.Topology) else topology_from_gui(src)
    dataframe, sim, _ = _fixed.build_dataset(
        cfg,
        topo=topo,
        size_capacities=bool(params.get("size_capacities", True)))
    out = os.environ.get("SNS_DATASET_PATH", "ss7_dataset_v7.csv")
    dataframe.to_csv(out, index=False)
    sidecar = os.path.splitext(out)[0] + "_topology.json"
    with open(sidecar, "w", encoding="utf-8") as fh:
        json.dump({"topology": topology_spec(sim.topo),
                   "days": cfg.days, "seed": cfg.seed},
                  fh, ensure_ascii=False, indent=2)
    print(f"Dataset saved to {os.path.abspath(out)}; shape={dataframe.shape}")
    print(f"Topology sidecar saved to {os.path.abspath(sidecar)}")
    return dataframe


if __name__ == "__main__":
    result = main()
    output_path = "ss7_dataset_v7.csv"
    result.to_csv(output_path, index=False)
    print(f"Dataset saved to {output_path}; shape={result.shape}")