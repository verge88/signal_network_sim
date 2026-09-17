"""
gui/model.py — протокольно-независимая модель сигнальной сети
для редактора топологий signal_network_sim.

Модель намеренно «плоская» и сериализуемая в JSON, чтобы её можно было
хранить в git, диффить и подставлять в любой из симуляторов
(ss7 / diameter / sip / 5g) через gui/adapters.py.
"""
from __future__ import annotations

import json
import math
import random
from collections import deque
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple

SCHEMA_VERSION = 1

# ──────────────────────────────────────────────────────────────────────
#  Протокольные профили: типы узлов, дефолты линков, таксономии атак
# ──────────────────────────────────────────────────────────────────────

PROTOCOLS: Dict[str, dict] = {
    "diameter": {
        "label": "Diameter (4G/LTE, 3GPP TS 29.272)",
        "module": "diameter_sim_v1",
        "topology_class": "DiameterNetworkTopology",
        "node_class": "DiameterNode",
        "link_class": "DiameterLink",
        "hub_types": ["DRA"],
        "node_types": {
            "DRA":  {"rate": (2000, 5000), "color": "#3f7fc4",
                     "iface": {"S6a": .55, "Gx": .25, "Rx": .10, "S13": .05, "Other": .05}},
            "DEA":  {"rate": (800, 2000),  "color": "#2fa3a3",
                     "iface": {"S6a": .60, "Gx": .15, "Rx": .05, "S13": .10, "Other": .10}},
            "HSS":  {"rate": (3000, 6000), "color": "#b5487f",
                     "iface": {"S6a": .85, "Gx": .00, "Rx": .00, "S13": .05, "Other": .10}},
            "MME":  {"rate": (1500, 4000), "color": "#e0873a",
                     "iface": {"S6a": .65, "Gx": .10, "Rx": .05, "S13": .15, "Other": .05}},
            "PCRF": {"rate": (1000, 3000), "color": "#7f5ec2",
                     "iface": {"S6a": .00, "Gx": .65, "Rx": .25, "S13": .00, "Other": .10}},
            "SGW":  {"rate": (500, 1500),  "color": "#5ba85b",
                     "iface": {"S6a": .20, "Gx": .50, "Rx": .10, "S13": .00, "Other": .20}},
            "PGW":  {"rate": (500, 1500),  "color": "#3f9e6e",
                     "iface": {"S6a": .05, "Gx": .60, "Rx": .15, "S13": .00, "Other": .20}},
        },
        "link_defaults": {"capacity_mbps": 100.0, "propagation_delay_ms": 2.0,
                          "base_loss_prob": 0.0005},
        "backbone_link": {"capacity_mbps": 1000.0, "propagation_delay_ms": 1.0,
                          "base_loss_prob": 0.0002},
        "attacks": ["location_tracking", "sms_data_interception", "signaling_dos",
                    "fraud_profile", "node_compromise"],
        "events": ["traffic_spike", "maintenance", "reroute", "roaming_surge"],
        "sim_defaults": {"duration_hours": 24.0, "interval_s": 15.0, "ar_phi": 0.85,
                         "ar_sigma": 0.18, "avg_message_size_bytes": 500, "seed": 42},
    },
    "ss7": {
        "label": "SS7 (2G/3G, MAP over TDM)",
        "module": "ss7_simulator_v7",
        "topology_class": "SS7NetworkTopology",
        "node_class": "SignallingNode",
        "link_class": "SignallingLink",
        "hub_types": ["STP"],
        "node_types": {
            "STP": {"rate": (150, 300), "color": "#3f7fc4", "iface": {}},
            "SCP": {"rate": (100, 250), "color": "#b5487f", "iface": {}},
            "SP":  {"rate": (60, 180),  "color": "#e0873a", "iface": {}},
        },
        # SS7-линк: 64 kbps = 0.064 Mbps
        "link_defaults": {"capacity_mbps": 0.064, "propagation_delay_ms": 8.0,
                          "base_loss_prob": 0.001},
        "backbone_link": {"capacity_mbps": 0.128, "propagation_delay_ms": 5.0,
                          "base_loss_prob": 0.0008},
        "attacks": ["NONE", "SMS_INTERCEPT", "LOCATION_TRACK", "SIGNALING_DOS",
                "IRSF", "SLAVE_COMPROMISE"],
        "events": ["NONE", "TRAFFIC_SPIKE", "MAINTENANCE", "REROUTE"],
        "sim_defaults": {"duration_hours": 24.0, "interval_s": 30.0, "ar_phi": 0.85,
                         "ar_sigma": 0.18, "avg_message_size_bytes": 100, "seed": 42},
    },
    "sip": {
        "label": "SIP / VoLTE (RFC 3261)",
        "module": "sip_sim_v4_claude",
        "topology_class": "SIPNetworkTopology",
        "node_class": "SIPNode",
        "link_class": "SIPLink",
        "hub_types": ["SBC", "PROXY"],
        "node_types": {
            "SBC":       {"rate": (1500, 4000), "color": "#3f7fc4", "iface": {}},
            "PROXY":     {"rate": (1000, 3000), "color": "#7f5ec2", "iface": {}},
            "REGISTRAR": {"rate": (800, 2000),  "color": "#b5487f", "iface": {}},
            "UA_GW":     {"rate": (400, 1200),  "color": "#5ba85b", "iface": {}},
            "REDIRECT":  {"rate": (200, 800),   "color": "#e0873a", "iface": {}},
        },
        "link_defaults": {"capacity_mbps": 100.0, "propagation_delay_ms": 3.0,
                          "base_loss_prob": 0.0005},
        "backbone_link": {"capacity_mbps": 1000.0, "propagation_delay_ms": 1.5,
                          "base_loss_prob": 0.0002},
        "attacks": ["signaling_dos", "register_hijack", "register_bruteforce",
                    "spit", "proxy_compromise"],
        "events": ["traffic_spike", "maintenance", "reroute", "presence_surge"],
        "sim_defaults": {"duration_hours": 24.0, "interval_s": 15.0, "ar_phi": 0.85,
                         "ar_sigma": 0.18, "avg_message_size_bytes": 700, "seed": 42},
    },
    "5g_sba": {
        "label": "5G SBA (HTTP/2, TS 29.510) — логическая схема",
        "module": "sba_sim_v1",
        "topology_class": None,      # SBA-симулятор графом не оперирует
        "node_class": None,
        "link_class": None,
        "hub_types": ["SCP", "NRF"],
        "node_types": {
            "NRF":         {"rate": (200, 600),   "color": "#b5487f", "iface": {}},
            "SCP":         {"rate": (3000, 8000), "color": "#3f7fc4", "iface": {}},
            "AMF":         {"rate": (1000, 3000), "color": "#e0873a", "iface": {}},
            "SMF":         {"rate": (800, 2500),  "color": "#5ba85b", "iface": {}},
            "UDM":         {"rate": (1200, 3500), "color": "#7f5ec2", "iface": {}},
            "AUSF":        {"rate": (600, 1800),  "color": "#2fa3a3", "iface": {}},
            "PCF":         {"rate": (500, 1500),  "color": "#3f9e6e", "iface": {}},
            "NF_CONSUMER": {"rate": (200, 900),   "color": "#8a8a8a", "iface": {}},
        },
        "link_defaults": {"capacity_mbps": 1000.0, "propagation_delay_ms": 0.5,
                          "base_loss_prob": 0.0001},
        "backbone_link": {"capacity_mbps": 10000.0, "propagation_delay_ms": 0.2,
                          "base_loss_prob": 0.00005},
        "attacks": ["no_token", "scp_bypass", "notify_abuse"],
        "events": ["diurnal_peak", "scale_in", "api_change", "code_upgrade"],
        "sim_defaults": {"duration_hours": 24.0, "interval_s": 600.0, "ar_phi": 0.85,
                         "ar_sigma": 0.18, "avg_message_size_bytes": 900, "seed": 42},
    },
}

LinkKey = Tuple[int, int]


def link_key(a: int, b: int) -> LinkKey:
    return (min(a, b), max(a, b))


# ──────────────────────────────────────────────────────────────────────
#  Сущности
# ──────────────────────────────────────────────────────────────────────

@dataclass
class Node:
    node_id: int
    node_type: str
    hostname: str = ""
    role: str = "SLAVE"                     # MASTER | SLAVE
    realm: str = "operator.com"
    master_id: Optional[int] = None
    zone_id: int = 0
    base_rate: float = 1000.0
    is_compromised: bool = False
    x: float = 0.0
    y: float = 0.0
    interface_dist: Dict[str, float] = field(default_factory=dict)
    extra: Dict[str, object] = field(default_factory=dict)

    @property
    def is_master(self) -> bool:
        return self.role.upper() == "MASTER"


@dataclass
class Link:
    src: int
    dst: int
    capacity_mbps: float = 100.0
    propagation_delay_ms: float = 2.0
    base_loss_prob: float = 0.0005
    extra: Dict[str, object] = field(default_factory=dict)

    @property
    def key(self) -> LinkKey:
        return link_key(self.src, self.dst)


@dataclass
class Scenario:
    """Окно атаки или нормального (объясняющего) события."""
    kind: str = "attack"                    # attack | normal
    name: str = "signaling_dos"
    target_nodes: List[int] = field(default_factory=list)
    start_interval: int = 0
    end_interval: int = 100
    intensity: float = 1.0


class Topology:
    """Редактируемая сигнальная сеть: узлы, линки, сценарии, параметры прогона."""

    def __init__(self, protocol: str = "diameter"):
        if protocol not in PROTOCOLS:
            raise ValueError(f"unknown protocol: {protocol}")
        self.protocol = protocol
        self.nodes: Dict[int, Node] = {}
        self.links: Dict[LinkKey, Link] = {}
        self.scenarios: List[Scenario] = []
        self.sim_params: Dict[str, object] = dict(
            PROTOCOLS[protocol]["sim_defaults"])
        self.notes: str = ""

    # ── профиль протокола ──────────────────────────────────────────
    @property
    def profile(self) -> dict:
        return PROTOCOLS[self.protocol]

    def node_type_names(self) -> List[str]:
        return list(self.profile["node_types"].keys())

    def color_of(self, node: Node) -> str:
        spec = self.profile["node_types"].get(node.node_type)
        return spec["color"] if spec else "#999999"

    # ── редактирование ─────────────────────────────────────────────
    def next_id(self) -> int:
        return (max(self.nodes) + 1) if self.nodes else 0

    def add_node(self, node_type: str, x: float, y: float,
                 rng: Optional[random.Random] = None) -> Node:
        rng = rng or random
        spec = self.profile["node_types"].get(node_type, {"rate": (500, 1500),
                                                          "iface": {}})
        lo, hi = spec.get("rate", (500, 1500))
        nid = self.next_id()
        idx = sum(1 for n in self.nodes.values() if n.node_type == node_type)
        node = Node(
            node_id=nid,
            node_type=node_type,
            hostname=f"{node_type.lower()}-{idx}.operator.com",
            base_rate=round(rng.uniform(lo, hi), 1),
            x=float(x), y=float(y),
            interface_dist=dict(spec.get("iface", {})),
        )
        self.nodes[nid] = node
        return node

    def remove_node(self, nid: int) -> None:
        self.nodes.pop(nid, None)
        for k in [k for k in self.links if nid in k]:
            del self.links[k]
        for sc in self.scenarios:
            sc.target_nodes = [t for t in sc.target_nodes if t != nid]
        for n in self.nodes.values():
            if n.master_id == nid:
                n.master_id = None

    def add_link(self, a: int, b: int, backbone: bool = False) -> Optional[Link]:
        if a == b or a not in self.nodes or b not in self.nodes:
            return None
        k = link_key(a, b)
        if k in self.links:
            return self.links[k]
        d = dict(self.profile["backbone_link" if backbone else "link_defaults"])
        link = Link(src=k[0], dst=k[1], **d)
        self.links[k] = link
        return link

    def remove_link(self, a: int, b: int) -> None:
        self.links.pop(link_key(a, b), None)

    def neighbors(self, nid: int) -> List[int]:
        out = []
        for (a, b) in self.links:
            if a == nid:
                out.append(b)
            elif b == nid:
                out.append(a)
        return out

    def masters(self) -> List[int]:
        return sorted(n.node_id for n in self.nodes.values() if n.is_master)

    # ── зоны мониторинга (BFS до ближайшего мастера) ───────────────
    def assign_zones(self) -> None:
        ms = self.masters()
        if not ms:
            return
        dist: Dict[int, Tuple[int, int]] = {}
        for m in ms:
            q = deque([(m, 0)])
            seen = {m}
            while q:
                cur, d = q.popleft()
                best = dist.get(cur)
                if best is None or d < best[0]:
                    dist[cur] = (d, m)
                for nb in self.neighbors(cur):
                    if nb not in seen:
                        seen.add(nb)
                        q.append((nb, d + 1))
        for n in self.nodes.values():
            if n.is_master:
                n.master_id = n.node_id
                n.zone_id = n.node_id
            else:
                d = dist.get(n.node_id)
                n.master_id = d[1] if d else ms[0]
                n.zone_id = n.master_id

    # ── метрики и валидация ────────────────────────────────────────
    def metrics(self) -> Dict[str, object]:
        n, m = len(self.nodes), len(self.links)
        deg = (2 * m / n) if n else 0.0
        comps = self._components()
        return {"nodes": n, "links": m, "avg_degree": round(deg, 2),
                "components": len(comps),
                "largest_component": max((len(c) for c in comps), default=0),
                "masters": len(self.masters()),
                "compromised": sum(1 for x in self.nodes.values()
                                   if x.is_compromised)}

    def _components(self) -> List[List[int]]:
        seen, out = set(), []
        for nid in self.nodes:
            if nid in seen:
                continue
            q, comp = deque([nid]), []
            seen.add(nid)
            while q:
                cur = q.popleft()
                comp.append(cur)
                for nb in self.neighbors(cur):
                    if nb not in seen:
                        seen.add(nb)
                        q.append(nb)
            out.append(comp)
        return out

    def validate(self) -> List[str]:
        problems: List[str] = []
        if not self.nodes:
            problems.append("ERROR: сеть пуста")
        if not self.masters():
            problems.append("ERROR: не задан ни один MASTER-узел "
                            "(мониторинг master-slave невозможен)")
        comps = self._components()
        if len(comps) > 1:
            problems.append(f"ERROR: граф несвязен ({len(comps)} компонент)")
        for n in self.nodes.values():
            if not self.neighbors(n.node_id):
                problems.append(f"ERROR: узел {n.node_id} ({n.hostname}) изолирован")
            if n.base_rate <= 0:
                problems.append(f"WARN: узел {n.node_id}: base_rate <= 0")
            if n.interface_dist:
                s = sum(n.interface_dist.values())
                if abs(s - 1.0) > 1e-3:
                    problems.append(f"WARN: узел {n.node_id}: "
                                    f"сумма interface_dist = {s:.3f} != 1.0")
        for k, l in self.links.items():
            if l.capacity_mbps <= 0:
                problems.append(f"ERROR: линк {k}: capacity_mbps <= 0")
            if not (0.0 <= l.base_loss_prob < 1.0):
                problems.append(f"ERROR: линк {k}: base_loss_prob вне [0,1)")
            if l.propagation_delay_ms < 0:
                problems.append(f"ERROR: линк {k}: отрицательная задержка")
        hubs = self.profile.get("hub_types") or []
        if hubs and not any(n.node_type in hubs for n in self.nodes.values()):
            problems.append(f"WARN: нет ни одного транзитного узла типа {hubs}")
        for i, sc in enumerate(self.scenarios):
            if sc.end_interval <= sc.start_interval:
                problems.append(f"ERROR: сценарий #{i}: end <= start")
            miss = [t for t in sc.target_nodes if t not in self.nodes]
            if miss:
                problems.append(f"ERROR: сценарий #{i}: нет узлов {miss}")
            if not sc.target_nodes:
                problems.append(f"WARN: сценарий #{i}: не выбраны целевые узлы")
        for m in self.masters():
            node = self.nodes[m]
            if node.is_compromised:
                problems.append(f"WARN: узел {m} одновременно MASTER и compromised — "
                                "мастер считается доверенным в модели угроз")
        return problems

    # ── раскладка ──────────────────────────────────────────────────
    def auto_layout(self, kind: str = "spring", width: float = 1100.0,
                    height: float = 760.0, seed: int = 42) -> None:
        if not self.nodes:
            return
        ids = sorted(self.nodes)
        pos: Dict[int, Tuple[float, float]] = {}
        try:
            import networkx as nx  # используется самим проектом
            g = self.to_networkx()
            if kind == "spring":
                p = nx.spring_layout(g, seed=seed, iterations=200)
            elif kind == "circular":
                p = nx.circular_layout(g)
            elif kind == "kamada":
                p = nx.kamada_kawai_layout(g)
            else:
                hubs = self.profile.get("hub_types") or []
                shells = [[i for i in ids if self.nodes[i].node_type in hubs],
                          [i for i in ids if self.nodes[i].node_type not in hubs]]
                shells = [s for s in shells if s]
                p = nx.shell_layout(g, nlist=shells)
            pos = {i: (float(v[0]), float(v[1])) for i, v in p.items()}
        except Exception:
            pos = self._fr_layout(ids, seed)

        xs = [v[0] for v in pos.values()] or [0.0]
        ys = [v[1] for v in pos.values()] or [0.0]
        sx = (max(xs) - min(xs)) or 1.0
        sy = (max(ys) - min(ys)) or 1.0
        pad = 70.0
        for i in ids:
            px, py = pos[i]
            self.nodes[i].x = pad + (px - min(xs)) / sx * (width - 2 * pad)
            self.nodes[i].y = pad + (py - min(ys)) / sy * (height - 2 * pad)

    def _fr_layout(self, ids: List[int], seed: int) -> Dict[int, Tuple[float, float]]:
        """Fruchterman-Reingold на случай отсутствия networkx."""
        rnd = random.Random(seed)
        pos = {i: [rnd.uniform(-1, 1), rnd.uniform(-1, 1)] for i in ids}
        k = math.sqrt(1.0 / max(1, len(ids)))
        for step in range(250):
            t = 0.1 * (1.0 - step / 250.0)
            disp = {i: [0.0, 0.0] for i in ids}
            for a in ids:
                for b in ids:
                    if a >= b:
                        continue
                    dx = pos[a][0] - pos[b][0]
                    dy = pos[a][1] - pos[b][1]
                    d2 = max(dx * dx + dy * dy, 1e-6)
                    f = (k * k) / d2
                    disp[a][0] += dx * f
                    disp[a][1] += dy * f
                    disp[b][0] -= dx * f
                    disp[b][1] -= dy * f
            for (a, b) in self.links:
                dx = pos[a][0] - pos[b][0]
                dy = pos[a][1] - pos[b][1]
                d = math.sqrt(dx * dx + dy * dy) or 1e-6
                f = d / k
                disp[a][0] -= dx / d * f * 0.1
                disp[a][1] -= dy / d * f * 0.1
                disp[b][0] += dx / d * f * 0.1
                disp[b][1] += dy / d * f * 0.1
            for i in ids:
                dx, dy = disp[i]
                d = math.sqrt(dx * dx + dy * dy) or 1e-6
                pos[i][0] += dx / d * min(d, t)
                pos[i][1] += dy / d * min(d, t)
        return {i: (pos[i][0], pos[i][1]) for i in ids}

    # ── экспорт/импорт ─────────────────────────────────────────────
    def to_networkx(self):
        import networkx as nx
        g = nx.Graph()
        for n in self.nodes.values():
            g.add_node(n.node_id, node_type=n.node_type, role=n.role,
                       hostname=n.hostname, zone_id=n.zone_id,
                       base_rate=n.base_rate, is_compromised=n.is_compromised)
        for l in self.links.values():
            g.add_edge(l.src, l.dst, capacity_mbps=l.capacity_mbps,
                       delay_ms=l.propagation_delay_ms,
                       loss=l.base_loss_prob)
        return g

    def to_dict(self) -> dict:
        return {
            "schema_version": SCHEMA_VERSION,
            "protocol": self.protocol,
            "sim_params": self.sim_params,
            "notes": self.notes,
            "nodes": [asdict(n) for n in
                      sorted(self.nodes.values(), key=lambda x: x.node_id)],
            "links": [asdict(l) for l in self.links.values()],
            "scenarios": [asdict(s) for s in self.scenarios],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Topology":
        topo = cls(d.get("protocol", "diameter"))
        topo.sim_params.update(d.get("sim_params", {}))
        topo.notes = d.get("notes", "")
        for nd in d.get("nodes", []):
            n = Node(**nd)
            topo.nodes[n.node_id] = n
        for ld in d.get("links", []):
            l = Link(**ld)
            topo.links[l.key] = l
        for sd in d.get("scenarios", []):
            topo.scenarios.append(Scenario(**sd))
        return topo

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> "Topology":
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(json.load(f))


# ──────────────────────────────────────────────────────────────────────
#  Шаблоны, повторяющие дефолтные топологии симуляторов
# ──────────────────────────────────────────────────────────────────────

def template(protocol: str, seed: int = 42) -> Topology:
    rng = random.Random(seed)
    t = Topology(protocol)

    def group(kind: str, count: int) -> List[int]:
        return [t.add_node(kind, 0, 0, rng).node_id for _ in range(count)]

    if protocol == "diameter":
        dra = group("DRA", 3)
        dea = group("DEA", 2)
        hss = group("HSS", 2)
        mme = group("MME", 4)
        pcrf = group("PCRF", 2)
        sgw = group("SGW", 2)
        pgw = group("PGW", 2)
        for i, a in enumerate(dra):                 # full-mesh DRA
            for b in dra[i + 1:]:
                t.add_link(a, b, backbone=True)
        for nid in dea + hss:                       # DEA/HSS → все DRA
            for d in dra:
                t.add_link(nid, d)
        for nid in mme + pcrf + sgw + pgw:          # остальные → 2 DRA
            for d in rng.sample(dra, min(2, len(dra))):
                t.add_link(nid, d)
        for m in dra[:2]:
            t.nodes[m].role = "MASTER"

    elif protocol == "ss7":
        stp = group("STP", 4)
        scp = group("SCP", 3)
        sp = group("SP", 10)
        for i, a in enumerate(stp):
            for b in stp[i + 1:]:
                t.add_link(a, b, backbone=True)
        for nid in scp + sp:
            for s in rng.sample(stp, 2):
                t.add_link(nid, s)
        for m in stp[:2]:
            t.nodes[m].role = "MASTER"

    elif protocol == "sip":
        sbc = group("SBC", 2)
        prx = group("PROXY", 3)
        reg = group("REGISTRAR", 2)
        gw = group("UA_GW", 6)
        red = group("REDIRECT", 1)
        for i, a in enumerate(sbc + prx):
            for b in (sbc + prx)[i + 1:]:
                t.add_link(a, b, backbone=True)
        for nid in reg + gw + red:
            for p in rng.sample(prx, min(2, len(prx))):
                t.add_link(nid, p)
            t.add_link(nid, rng.choice(sbc))
        for m in sbc:
            t.nodes[m].role = "MASTER"

    else:  # 5g_sba
        nrf = group("NRF", 1)
        scp = group("SCP", 2)
        prod = group("AMF", 2) + group("SMF", 2) + group("UDM", 2) + \
            group("AUSF", 1) + group("PCF", 1)
        cons = group("NF_CONSUMER", 6)
        for a in scp:
            for b in nrf:
                t.add_link(a, b, backbone=True)
        for i, a in enumerate(scp):
            for b in scp[i + 1:]:
                t.add_link(a, b, backbone=True)
        for nid in prod + cons:
            for s in scp:
                t.add_link(nid, s)
        for m in scp:
            t.nodes[m].role = "MASTER"

    t.assign_zones()
    t.auto_layout("spring", seed=seed)
    return t
