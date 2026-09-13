"""
Diameter Master-Slave Signalling Network Simulator v1
======================================================
Adaptation of SS7 hierarchical monitoring concept to Diameter (4G/LTE).

Design based on:
  - ENISA "Signalling Security in Telecom SS7/Diameter/5G" (2018)
  - GSMA FS.19 "Diameter Interconnect Security" categories 0-3
  - ITU-T Q.3057 "Signalling requirements for interconnection between
    trustable network entities" (2020)
  - Kotte B.T. "Analysis and Experimental Verification of Diameter
    Attacks in Long Term Evolution Networks" (Aalto/KTH, 2016)
  - P1 Security "Understanding Vulnerabilities of Diameter in 4G" (2024)
  - Cellusys "Introduction to Diameter Security" (2018)
  - Enea/AdaptiveMobile "Measuring the Diameter – Protecting 4G" (2018)
  - Diametriq "Measuring the Explosion of LTE Signaling Traffic" (2012)
  - 3GPP TS 29.272 (S6a/S6d), TS 29.212 (Gx), TS 29.214 (Rx),
    TS 29.229 (Cx/Dx), TS 29.336 (S6t)
  - RFC 6733 (Diameter Base Protocol)

Key architectural differences from SS7 simulator:
  1. Nodes: MME, HSS, PCRF, S-GW/P-GW, DRA, DEA (vs STP, SCP, SP)
  2. Interfaces: S6a, Gx, Rx, Cx, S13 (vs MAP subtypes)
  3. IP-based transport (TCP/SCTP over IP) — higher bandwidth, different
     loss/delay characteristics vs 64 kbps TDM links
  4. Master nodes placed on DRA (max connectivity) — analogous to STP
  5. Diameter messages are ~500 bytes avg (vs ~100 bytes MSU in SS7)
  6. Traffic volume: 500-5000 msgs/interval per node (vs 60-300 in SS7)
  7. Attack types adapted to Diameter-specific commands per GSMA FS.19
"""

import numpy as np
import pandas as pd
import networkx as nx
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set
from collections import defaultdict
import warnings
import os

warnings.filterwarnings("ignore")


# ──────────────────────────────────────────────────────────
#  Enums
# ──────────────────────────────────────────────────────────

class NodeType(Enum):
    """
    LTE/EPC node types.
    Based on 3GPP TS 23.401 (EPC architecture) and
    Kotte (2016) ch. 2.1.3.
    """
    MME = auto()     # Mobility Management Entity
    HSS = auto()     # Home Subscriber Server
    PCRF = auto()    # Policy and Charging Rules Function
    SGW = auto()     # Serving Gateway
    PGW = auto()     # PDN Gateway
    DRA = auto()     # Diameter Routing Agent (transit/routing hub)
    DEA = auto()     # Diameter Edge Agent (border/interconnect)


class NodeRole(Enum):
    MASTER = auto()
    SLAVE = auto()


class AttackType(Enum):
    """
    Diameter attack taxonomy based on:
    - GSMA FS.19 categories 1-3
    - ENISA (2018) Table 1: attack types
    - P1 Security (2024): 5 vulnerability classes
    - Kotte (2016) ch. 5.4-5.6: Phase 1 & Phase 2 attacks
    - Cellusys (2018): attack scenarios
    """
    NONE = "none"
    SUBSCRIBER_LOCATION_TRACKING = "location_tracking"
    # Exploits S6a AIR + SRR to discover serving node/IMSI,
    # then IDR/UDR for fine-grained location.
    # Refs: Kotte 5.5.3, 5.6.4; ENISA Table 1; P1 Security
    SMS_DATA_INTERCEPTION = "sms_data_interception"
    # Exploits S6a ULR to redirect subscriber to attacker's
    # fake MME, then intercepts SMS/data.
    # Refs: Kotte 5.6.2; Cellusys; Enea (2018) "Random Walk"
    SIGNALING_DOS = "signaling_dos"
    # Flood of CLR/RSR/PUR to deny service to subscribers
    # or RSR to reset entire MME pools.
    # Refs: Kotte 5.6.1, 5.6.5, 5.6.7; ENISA; P1 Security
    FRAUD_PROFILE_MANIPULATION = "fraud_profile"
    # Exploits IDR to modify subscriber profile (ODB, QoS),
    # enables unauthorized services or billing fraud.
    # Refs: Kotte 5.6.3; ENISA "Subscriber Fraud"; P1 Security
    NODE_COMPROMISE = "node_compromise"
    # Compromised DRA/DEA/MME sends falsified monitoring reports.
    # Novel attack class from original SS7 paper, adapted to Diameter.


class NormalEventType(Enum):
    NONE = "none"
    TRAFFIC_SPIKE = "traffic_spike"      # Flash crowd / mass event
    MAINTENANCE = "maintenance"          # Planned node maintenance
    REROUTE = "reroute"                  # DRA policy change / failover
    ROAMING_SURGE = "roaming_surge"      # Roaming traffic increase


# ──────────────────────────────────────────────────────────
#  Diameter Command Codes (simplified)
#  Based on 3GPP TS 29.272 (S6a), TS 29.212 (Gx),
#  TS 29.214 (Rx), TS 29.229 (Cx)
# ──────────────────────────────────────────────────────────

# S6a interface commands (MME <-> HSS)
S6A_COMMANDS = [
    "AIR",   # Authentication-Information-Request (cmd 318)
    "ULR",   # Update-Location-Request (cmd 316)
    "CLR",   # Cancel-Location-Request (cmd 317)
    "IDR",   # Insert-Subscriber-Data-Request (cmd 319)
    "DSR",   # Delete-Subscriber-Data-Request (cmd 320)
    "PUR",   # Purge-UE-Request (cmd 321)
    "RSR",   # Reset-Request (cmd 322)
    "NOR",   # Notify-Request (cmd 323)
]

# Gx interface commands (PCEF/P-GW <-> PCRF)
GX_COMMANDS = [
    "CCR_I",  # CC-Request Initial (cmd 272, CC-Request-Type=1)
    "CCR_U",  # CC-Request Update
    "CCR_T",  # CC-Request Terminate
    "RAR",    # Re-Auth-Request (cmd 258)
]

# Rx interface commands (AF <-> PCRF)
RX_COMMANDS = [
    "AAR",    # AA-Request (cmd 265)
    "STR",    # Session-Termination-Request (cmd 275)
    "ASR",    # Abort-Session-Request (cmd 274)
]

# S13 interface (MME <-> EIR)
S13_COMMANDS = ["ECR"]  # ME-Identity-Check-Request (cmd 324)

# Combined for traffic distribution
ALL_DIAMETER_COMMANDS = S6A_COMMANDS + GX_COMMANDS + RX_COMMANDS + S13_COMMANDS


# ──────────────────────────────────────────────────────────
#  Data Classes
# ──────────────────────────────────────────────────────────

@dataclass
class DiameterNode:
    node_id: int
    node_type: NodeType
    role: NodeRole = NodeRole.SLAVE
    hostname: str = ""
    realm: str = "operator.com"
    master_id: Optional[int] = None
    zone_id: int = 0
    base_rate: float = 1000.0   # msgs per interval (higher than SS7)
    # Diameter interface distribution for this node
    # Based on Diametriq (2012) traffic model:
    # S6a dominates (~55%), Gx (~25%), Rx (~10%), S13 (~5%), other (~5%)
    interface_dist: Dict[str, float] = field(default_factory=lambda: {
        "S6a": 0.55, "Gx": 0.25, "Rx": 0.10, "S13": 0.05, "Other": 0.05
    })
    # S6a command subtype distribution (normal operation)
    # Based on Kotte (2016) and 3GPP attach/detach/TAU procedures
    s6a_command_dist: Dict[str, float] = field(default_factory=lambda: {
        "AIR": 0.30,          # Most frequent: every attach
        "ULR": 0.25,          # Every attach + TAU
        "CLR": 0.05,          # Detach-initiated
        "IDR": 0.12,          # Profile updates (HSS-initiated)
        "DSR": 0.03,          # Data deletion
        "PUR": 0.05,          # Purge on detach
        "RSR": 0.02,          # Rare: network reset
        "NOR": 0.18,          # Notifications
    })
    # Gx command distribution
    gx_command_dist: Dict[str, float] = field(default_factory=lambda: {
        "CCR_I": 0.35,        # Session creation
        "CCR_U": 0.30,        # QoS modifications
        "CCR_T": 0.25,        # Session termination
        "RAR": 0.10,          # Re-authorization
    })
    is_compromised: bool = False
    compromised_since: Optional[int] = None


@dataclass
class DiameterLink:
    """
    Represents a Diameter peer connection (TCP/SCTP over IP).
    
    Key difference from SS7: IP links have much higher capacity
    but are subject to IP-layer congestion, TCP retransmissions.
    
    Typical Diameter link: 1 Gbps Ethernet (vs 64 kbps SS7 link).
    Average Diameter message: ~500 bytes (RFC 6733 header 20 bytes
    + AVPs; typical S6a msg 300-800 bytes per F5/Diametriq).
    """
    src: int
    dst: int
    capacity_mbps: float = 100.0      # Much higher than SS7's 64 kbps
    propagation_delay_ms: float = 2.0  # Intra-DC: 1-2ms, inter-DC: 5-20ms
    base_loss_prob: float = 0.0005     # IP networks: lower base loss
    current_load_mbps: float = 0.0
    maintenance_loss_mult: float = 1.0
    maintenance_delay_mult: float = 1.0

    @property
    def rho(self) -> float:
        if self.capacity_mbps <= 0:
            return 999.0
        return self.current_load_mbps / self.capacity_mbps

    @property
    def effective_delay_ms(self) -> float:
        base = self.propagation_delay_ms * self.maintenance_delay_mult
        r = self.rho
        # M/M/1 queueing delay behavior at high utilization
        if r > 0.7:
            base *= (1.0 + 3.0 * (r - 0.7) / 0.3)
        if r > 1.0:
            base *= (1.0 + 8.0 * (r - 1.0))
        return base

    @property
    def effective_loss(self) -> float:
        p = self.base_loss_prob * self.maintenance_loss_mult
        r = self.rho
        # TCP retransmission / SCTP congestion at high utilization
        if r > 0.8:
            p += 0.03 * ((r - 0.8) / 0.2) ** 2
        if r > 1.0:
            p += 0.25 * (r - 1.0)
        return min(p, 0.90)


@dataclass
class ControlUnitConfig:
    """
    Control Unit for Diameter monitoring.
    
    In Diameter, CU can be implemented as:
    - Vendor-specific Diameter message (Application-Id = vendor)
    - Or piggybacked on Device-Watchdog-Request (DWR, cmd 280)
    
    CU message size: ~200 bytes (Diameter header 20 bytes +
    monitoring AVPs with reported counters).
    """
    poll_interval_s: float = 15.0      # Faster than SS7 (30s) — IP allows it
    cu_message_size_bytes: int = 200   # Larger than SS7's 50 bytes (more AVPs)
    processing_delay_ms: float = 2.0   # Faster processing (modern hardware)
    integrity_background_fail_prob: float = 0.01  # Lower than SS7 (TLS optional)


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
    """
    Simulation parameters.
    
    Key parameter sources:
    - Duration/interval: same 24h cycle, but 15s intervals (IP allows faster)
    - Node counts: medium LTE operator topology per 3GPP TS 23.401
      * 4 MME (serving different tracking areas)
      * 2 HSS (geo-redundant pair)
      * 2 PCRF (geo-redundant pair)
      * 2 S-GW (regional)
      * 2 P-GW (regional)
      * 3 DRA (routing mesh, full connectivity)
      * 2 DEA (border/interconnect with IPX/roaming)
    - Link capacities: 100 Mbps internal, 1 Gbps DRA-DRA
      (conservative for signaling VLAN)
    - Message size: ~500 bytes average per Diametriq/F5
    """
    duration_hours: float = 24.0
    interval_s: float = 15.0           # 15s polling (faster than SS7's 30s)
    # Node counts for medium LTE operator
    n_mme: int = 4                     # MME pool
    n_hss: int = 2                     # HSS pair
    n_pcrf: int = 2                    # PCRF pair
    n_sgw: int = 2                     # Serving Gateways
    n_pgw: int = 2                     # PDN Gateways
    n_dra: int = 3                     # DRA mesh (routing hubs)
    n_dea: int = 2                     # DEA (edge/border)
    master_node_ids: List[int] = field(default_factory=lambda: [0, 1])
    # Link capacities (Mbps)
    dra_dra_capacity_mbps: float = 1000.0
    dra_node_capacity_mbps: float = 100.0
    dea_dra_capacity_mbps: float = 100.0
    # Traffic parameters
    ar_phi: float = 0.85               # Same AR(1) autocorrelation
    ar_sigma: float = 0.18
    avg_message_size_bytes: int = 500  # Diameter msg avg (vs 100 for SS7)
    seed: int = 42
    cu_config: ControlUnitConfig = field(default_factory=ControlUnitConfig)


# ──────────────────────────────────────────────────────────
#  Topology Builder — LTE/EPC Diameter Network
# ──────────────────────────────────────────────────────────

class DiameterNetworkTopology:
    """
    Builds realistic LTE EPC Diameter topology.
    
    Architecture based on:
    - 3GPP TS 23.401 (EPC architecture)
    - 3GPP TS 29.272 (S6a interface)
    - Diametriq (2012) "Role of DSC in LTE and VoLTE"
    - Kotte (2016) Figure 2.3 (EPC architecture)
    
    Topology:
    - DRA nodes form full-mesh (analogous to STP full-mesh in SS7)
    - All other nodes connect to at least 2 DRAs (redundancy)
    - DEA nodes connect to all DRAs (border routing)
    - HSS connects to all DRAs (central database)
    - MME, PCRF, S-GW, P-GW connect to 2 DRAs each
    """

    def __init__(self, config: SimulationConfig):
        self.config = config
        self.graph = nx.Graph()
        self.nodes: Dict[int, DiameterNode] = {}
        self.links: Dict[Tuple[int, int], DiameterLink] = {}
        self._build()

    def _build(self):
        rng = np.random.default_rng(self.config.seed)
        node_id = 0

        # ── DRA nodes (routing hubs) ──
        dra_ids = []
        for i in range(self.config.n_dra):
            n = DiameterNode(
                node_id=node_id, node_type=NodeType.DRA,
                hostname=f"dra-{i}.operator.com",
                base_rate=rng.uniform(2000, 5000)  # High traffic: transit
            )
            self.nodes[node_id] = n
            self.graph.add_node(node_id)
            dra_ids.append(node_id)
            node_id += 1

        # ── DEA nodes (edge/border) ──
        dea_ids = []
        for i in range(self.config.n_dea):
            n = DiameterNode(
                node_id=node_id, node_type=NodeType.DEA,
                hostname=f"dea-{i}.operator.com",
                base_rate=rng.uniform(800, 2000)  # Roaming/interconnect
            )
            # DEA has higher roaming/international traffic
            n.interface_dist = {"S6a": 0.60, "Gx": 0.15, "Rx": 0.05,
                                "S13": 0.10, "Other": 0.10}
            self.nodes[node_id] = n
            self.graph.add_node(node_id)
            dea_ids.append(node_id)
            node_id += 1

        # ── HSS nodes ──
        hss_ids = []
        for i in range(self.config.n_hss):
            n = DiameterNode(
                node_id=node_id, node_type=NodeType.HSS,
                hostname=f"hss-{i}.operator.com",
                base_rate=rng.uniform(3000, 6000)  # Highest: all S6a terminates
            )
            n.interface_dist = {"S6a": 0.85, "Gx": 0.0, "Rx": 0.0,
                                "S13": 0.05, "Other": 0.10}
            self.nodes[node_id] = n
            self.graph.add_node(node_id)
            hss_ids.append(node_id)
            node_id += 1

        # ── MME nodes ──
        mme_ids = []
        for i in range(self.config.n_mme):
            n = DiameterNode(
                node_id=node_id, node_type=NodeType.MME,
                hostname=f"mme-{i}.operator.com",
                base_rate=rng.uniform(1500, 4000)
            )
            # MME: heavy S6a (attach/TAU) + S13 (IMEI check)
            n.interface_dist = {"S6a": 0.65, "Gx": 0.10, "Rx": 0.05,
                                "S13": 0.15, "Other": 0.05}
            self.nodes[node_id] = n
            self.graph.add_node(node_id)
            mme_ids.append(node_id)
            node_id += 1

        # ── PCRF nodes ──
        pcrf_ids = []
        for i in range(self.config.n_pcrf):
            n = DiameterNode(
                node_id=node_id, node_type=NodeType.PCRF,
                hostname=f"pcrf-{i}.operator.com",
                base_rate=rng.uniform(1000, 3000)
            )
            n.interface_dist = {"S6a": 0.0, "Gx": 0.65, "Rx": 0.25,
                                "S13": 0.0, "Other": 0.10}
            # PCRF has Gx command distribution
            n.s6a_command_dist = {}  # Not applicable
            self.nodes[node_id] = n
            self.graph.add_node(node_id)
            pcrf_ids.append(node_id)
            node_id += 1

        # ── S-GW nodes ──
        sgw_ids = []
        for i in range(self.config.n_sgw):
            n = DiameterNode(
                node_id=node_id, node_type=NodeType.SGW,
                hostname=f"sgw-{i}.operator.com",
                base_rate=rng.uniform(500, 1500)
            )
            n.interface_dist = {"S6a": 0.20, "Gx": 0.50, "Rx": 0.10,
                                "S13": 0.0, "Other": 0.20}
            self.nodes[node_id] = n
            self.graph.add_node(node_id)
            sgw_ids.append(node_id)
            node_id += 1

        # ── P-GW nodes ──
        pgw_ids = []
        for i in range(self.config.n_pgw):
            n = DiameterNode(
                node_id=node_id, node_type=NodeType.PGW,
                hostname=f"pgw-{i}.operator.com",
                base_rate=rng.uniform(500, 1500)
            )
            n.interface_dist = {"S6a": 0.05, "Gx": 0.60, "Rx": 0.15,
                                "S13": 0.0, "Other": 0.20}
            self.nodes[node_id] = n
            self.graph.add_node(node_id)
            pgw_ids.append(node_id)
            node_id += 1

        # ── Build links ──
        # DRA full-mesh (high-speed backbone)
        for i, d1 in enumerate(dra_ids):
            for d2 in dra_ids[i + 1:]:
                self._add_link(d1, d2, self.config.dra_dra_capacity_mbps,
                               delay_ms=1.0 + rng.uniform(0, 1.0))

        # DEA → all DRAs
        for dea in dea_ids:
            for dra in dra_ids:
                self._add_link(dea, dra, self.config.dea_dra_capacity_mbps,
                               delay_ms=2.0 + rng.uniform(0, 3.0))

        # HSS → all DRAs (central, must be reachable from everywhere)
        for hss in hss_ids:
            for dra in dra_ids:
                self._add_link(hss, dra, self.config.dra_node_capacity_mbps,
                               delay_ms=1.0 + rng.uniform(0, 2.0))

        # MME, PCRF, S-GW, P-GW → 2 DRAs each
        for nid_list in [mme_ids, pcrf_ids, sgw_ids, pgw_ids]:
            for nid in nid_list:
                chosen = rng.choice(dra_ids,
                                    size=min(2, len(dra_ids)),
                                    replace=False)
                for dra in chosen:
                    self._add_link(nid, dra, self.config.dra_node_capacity_mbps,
                                   delay_ms=2.0 + rng.uniform(0, 3.0))

        # ── Assign master/slave roles ──
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
            node.master_id = (best_master if best_master is not None
                              else self.config.master_node_ids[0])
            node.zone_id = node.master_id

    def _add_link(self, src: int, dst: int, capacity: float,
                  delay_ms: float = 2.0):
        key = (min(src, dst), max(src, dst))
        if key not in self.links:
            link = DiameterLink(
                src=key[0], dst=key[1],
                capacity_mbps=capacity,
                propagation_delay_ms=delay_ms
            )
            self.links[key] = link
            self.graph.add_edge(key[0], key[1])

    def get_link(self, a: int, b: int) -> Optional[DiameterLink]:
        key = (min(a, b), max(a, b))
        return self.links.get(key)

    def get_path(self, src: int, dst: int) -> List[int]:
        try:
            return nx.shortest_path(self.graph, src, dst)
        except nx.NetworkXNoPath:
            return []

    def reset_link_loads(self):
        for link in self.links.values():
            link.current_load_mbps = 0.0


# ──────────────────────────────────────────────────────────
#  Traffic Generator — Diameter
# ──────────────────────────────────────────────────────────

class DiameterTrafficGenerator:
    """
    Generates Diameter signaling traffic with realistic profiles.
    
    Traffic composition based on Diametriq (2012) traffic model:
    - Attach generates: 1 AIR + 1 ULR + 1 ECR + 1 CCR_I
    - TAU generates: 1 ULR + 1 NOR
    - Service Request: 1 CCR_U
    - Detach: 1 PUR + 1 CCR_T
    - VoLTE call: 1 AAR + 1 CCR_I + 1 CCR_T + 1 STR
    
    For medium operator with 5M LTE subscribers:
    ~100K-500K msgs/sec total → ~1500-7500 msgs per 15s per node
    """

    TOD_PROFILE = [
        0.3, 0.2, 0.15, 0.15, 0.2, 0.4,
        0.7, 0.9, 1.0, 1.0, 0.95, 0.9,
        0.85, 0.9, 0.95, 1.0, 1.0, 0.95,
        0.9, 0.8, 0.7, 0.6, 0.5, 0.4
    ]

    INTERFACES = ["S6a", "Gx", "Rx", "S13", "Other"]

    def __init__(self, topology: DiameterNetworkTopology,
                 config: SimulationConfig):
        self.topo = topology
        self.config = config
        self.rng = np.random.default_rng(config.seed + 1)
        self.ar_state: Dict[int, float] = {nid: 0.0
                                            for nid in topology.nodes}

    def time_of_day_factor(self, interval_idx: int) -> float:
        t_seconds = interval_idx * self.config.interval_s
        hour = (t_seconds / 3600.0) % 24.0
        h_low = int(hour) % 24
        h_high = (h_low + 1) % 24
        frac = hour - int(hour)
        return (self.TOD_PROFILE[h_low] * (1 - frac) +
                self.TOD_PROFILE[h_high] * frac)

    def generate_node_traffic(
        self, node: DiameterNode, interval_idx: int,
        override_params: Optional[Dict] = None
    ) -> Dict:
        # AR(1) process for traffic variability
        phi = self.config.ar_phi
        sigma = self.config.ar_sigma
        prev_x = self.ar_state[node.node_id]
        x_t = phi * prev_x + sigma * self.rng.standard_normal()
        self.ar_state[node.node_id] = x_t
        ar_mult = np.exp(x_t)

        tod = self.time_of_day_factor(interval_idx)
        base_rate = node.base_rate
        iface_dist = dict(node.interface_dist)
        s6a_dist = dict(node.s6a_command_dist)
        gx_dist = dict(node.gx_command_dist)

        # Diameter-specific: roaming fraction, unique peers
        international_fraction = 0.05 + 0.03 * self.rng.random()
        n_unique_peers = max(5, int(self.rng.normal(20, 6)))

        if override_params:
            if "base_rate_mult" in override_params:
                base_rate *= override_params["base_rate_mult"]
            if "ar_mult_override" in override_params:
                ar_mult = override_params["ar_mult_override"]
            if "iface_dist" in override_params:
                iface_dist.update(override_params["iface_dist"])
            if "s6a_dist" in override_params:
                s6a_dist.update(override_params["s6a_dist"])
            if "gx_dist" in override_params:
                gx_dist.update(override_params["gx_dist"])
            if "international_fraction" in override_params:
                international_fraction = override_params[
                    "international_fraction"]
            if "n_unique_peers" in override_params:
                n_unique_peers = override_params["n_unique_peers"]

        # Normalize distributions
        iface_total = sum(iface_dist.values())
        iface_dist = {k: v / iface_total for k, v in iface_dist.items()}
        if s6a_dist:
            s6a_total = sum(s6a_dist.values())
            s6a_dist = {k: v / s6a_total for k, v in s6a_dist.items()}
        if gx_dist:
            gx_total = sum(gx_dist.values())
            gx_dist = {k: v / gx_total for k, v in gx_dist.items()}

        # Generate total messages
        rate = base_rate * tod * ar_mult
        total_messages = max(1, int(self.rng.poisson(max(1, rate))))

        # Distribute by interface
        s6a_count = int(total_messages * iface_dist.get("S6a", 0.55))
        gx_count = int(total_messages * iface_dist.get("Gx", 0.25))
        rx_count = int(total_messages * iface_dist.get("Rx", 0.10))
        s13_count = int(total_messages * iface_dist.get("S13", 0.05))
        other_count = total_messages - s6a_count - gx_count - rx_count - s13_count

        # S6a command subtypes
        s6a_sub_counts = {}
        remaining = s6a_count
        for i, (cmd, p) in enumerate(s6a_dist.items()):
            if i == len(s6a_dist) - 1:
                s6a_sub_counts[cmd] = remaining
            else:
                c = int(s6a_count * p)
                s6a_sub_counts[cmd] = c
                remaining -= c

        # Gx command subtypes
        gx_sub_counts = {}
        remaining = gx_count
        for i, (cmd, p) in enumerate(gx_dist.items()):
            if i == len(gx_dist) - 1:
                gx_sub_counts[cmd] = remaining
            else:
                c = int(gx_count * p)
                gx_sub_counts[cmd] = c
                remaining -= c

        # Compute ratios
        s6a_ratios = {}
        for cmd in S6A_COMMANDS:
            cnt = s6a_sub_counts.get(cmd, 0)
            s6a_ratios[f"ratio_s6a_{cmd.lower()}"] = (
                cnt / max(1, s6a_count))

        gx_ratios = {}
        for cmd in GX_COMMANDS:
            cnt = gx_sub_counts.get(cmd, 0)
            gx_ratios[f"ratio_gx_{cmd.lower()}"] = (
                cnt / max(1, gx_count))

        # Entropy of interface distribution
        all_counts = [s6a_count, gx_count, rx_count, s13_count, other_count]
        entropy = self._entropy(all_counts)

        # S6a dominance metrics
        s6a_ratio_vals = [s6a_ratios.get(f"ratio_s6a_{c.lower()}", 0)
                          for c in S6A_COMMANDS]
        sorted_r = sorted(s6a_ratio_vals, reverse=True)
        s6a_dominance = (sorted_r[0] / max(0.001, sum(sorted_r[1:]))
                         if len(sorted_r) > 1 else 0.0)
        s6a_top2 = (sum(sorted_r[:2]) / max(0.001, sum(sorted_r[2:]))
                    if len(sorted_r) > 2 else 0.0)
        s6a_gini = self._gini(s6a_ratio_vals)

        inbound_outbound_ratio = 0.8 + 0.4 * self.rng.random()
        if override_params and "inbound_outbound_ratio" in override_params:
            inbound_outbound_ratio = override_params[
                "inbound_outbound_ratio"]

        # Apply entropy override
        if override_params and "entropy_mult" in override_params:
            entropy = max(0.0, entropy * override_params["entropy_mult"])

        result = {
            "total_messages": total_messages,
            "s6a_count": s6a_count,
            "gx_count": gx_count,
            "rx_count": rx_count,
            "s13_count": s13_count,
            "other_count": other_count,
            "entropy": entropy,
            "international_fraction": international_fraction,
            "n_unique_peers": n_unique_peers,
            "inbound_outbound_ratio": inbound_outbound_ratio,
            "s6a_dominance_ratio": s6a_dominance,
            "s6a_top2_ratio": s6a_top2,
            "s6a_gini": s6a_gini,
        }
        result.update(s6a_ratios)
        result.update(gx_ratios)
        for cmd in S6A_COMMANDS:
            result[f"s6a_{cmd.lower()}_count"] = s6a_sub_counts.get(cmd, 0)
        for cmd in GX_COMMANDS:
            result[f"gx_{cmd.lower()}_count"] = gx_sub_counts.get(cmd, 0)

        return result

    def add_traffic_load_to_links(self, node: DiameterNode, traffic: Dict):
        dra_ids = [n.node_id for n in self.topo.nodes.values()
                   if n.node_type == NodeType.DRA]
        if not dra_ids:
            return
        target = min(dra_ids,
                     key=lambda s: nx.shortest_path_length(
                         self.topo.graph, node.node_id, s)
                     if nx.has_path(self.topo.graph, node.node_id, s)
                     else 9999)
        path = self.topo.get_path(node.node_id, target)
        if len(path) < 2:
            return
        load_mbps = (traffic["total_messages"] *
                     self.config.avg_message_size_bytes * 8) / (
                         self.config.interval_s * 1_000_000.0)
        for i in range(len(path) - 1):
            link = self.topo.get_link(path[i], path[i + 1])
            if link:
                link.current_load_mbps += load_mbps

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
        s = np.sum(arr)
        return float((2 * np.sum(index * arr) - (n + 1) * s) /
                     (n * s)) if s > 0 else 0.0


# ──────────────────────────────────────────────────────────
#  Control Unit Engine — Diameter version
# ──────────────────────────────────────────────────────────

class DiameterControlUnitEngine:
    """
    CU engine adapted for Diameter.
    Same masking/compromise model as SS7 version.
    """

    def __init__(self, topology: DiameterNetworkTopology,
                 config: SimulationConfig):
        self.topo = topology
        self.config = config
        self.cu = config.cu_config
        self.rng = np.random.default_rng(config.seed + 100)
        self.last_successful_report: Dict[int, Dict] = {}
        self.consecutive_cu_losses: Dict[int, int] = defaultdict(int)
        self.intervals_since_last_cu: Dict[int, int] = defaultdict(int)
        self.response_delay_history: Dict[int, List[float]] = defaultdict(
            list)
        self.response_history_window = 20

    def add_cu_load_to_links(self, master_id: int, slave_ids: List[int]):
        cu_load_mbps = (self.cu.cu_message_size_bytes * 8) / (
            self.config.interval_s * 1_000_000.0)
        for sid in slave_ids:
            for path in [self.topo.get_path(master_id, sid),
                         self.topo.get_path(sid, master_id)]:
                for i in range(len(path) - 1):
                    link = self.topo.get_link(path[i], path[i + 1])
                    if link:
                        link.current_load_mbps += cu_load_mbps

    def poll_slave(self, master_id: int, slave_id: int,
                   true_traffic: Dict, node: DiameterNode,
                   interval_idx: int) -> Dict:
        path_req = self.topo.get_path(master_id, slave_id)
        req_delay, req_loss = self._path_impairment(path_req)

        path_resp = self.topo.get_path(slave_id, master_id)
        resp_delay, resp_loss = self._path_impairment(path_resp)

        # Processing delay (ms → seconds for consistency)
        processing_delay_ms = self.cu.processing_delay_ms
        if node.is_compromised:
            processing_delay_ms += self.rng.uniform(0.5, 3.0)

        rtt_ms = req_delay + processing_delay_ms + resp_delay

        combined_loss = 1.0 - (1.0 - req_loss) * (1.0 - resp_loss)
        cu_delivered = self.rng.random() > combined_loss

        integrity_ok = True
        if cu_delivered:
            fail_prob = self.cu.integrity_background_fail_prob
            if node.is_compromised:
                fail_prob = 0.30
            integrity_ok = self.rng.random() > fail_prob

        if cu_delivered:
            if node.is_compromised:
                reported = self._masked_report(true_traffic, node,
                                               interval_idx)
            else:
                reported = self._honest_report(true_traffic)
            self.last_successful_report[slave_id] = dict(reported)
            self.consecutive_cu_losses[slave_id] = 0
            self.intervals_since_last_cu[slave_id] = 0
            self.response_delay_history[slave_id].append(
                resp_delay + processing_delay_ms)
            if len(self.response_delay_history[slave_id]) > \
                    self.response_history_window:
                self.response_delay_history[slave_id] = \
                    self.response_delay_history[slave_id][
                        -self.response_history_window:]
        else:
            self.consecutive_cu_losses[slave_id] += 1
            self.intervals_since_last_cu[slave_id] += 1
            if slave_id in self.last_successful_report:
                reported = dict(self.last_successful_report[slave_id])
            else:
                reported = self._empty_report()

        # Jitter
        hist = self.response_delay_history.get(slave_id, [])
        jitter = float(np.std(hist[-10:])) if len(hist) >= 3 else 0.0

        # Consistency error
        consistency_error = abs(
            reported["total_messages"] -
            (reported["s6a_count"] + reported["gx_count"] +
             reported["rx_count"] + reported["s13_count"] +
             reported["other_count"])
        )

        result = {
            "cu_delivered": 1 if cu_delivered else 0,
            "cu_rtt_delay_ms": rtt_ms if cu_delivered else 0.0,
            "cu_req_delay_ms": req_delay,
            "cu_resp_delay_ms": resp_delay,
            "cu_combined_loss_prob": combined_loss,
            "integrity_check": 1 if integrity_ok else 0,
            "consecutive_cu_losses": self.consecutive_cu_losses[slave_id],
            "cu_processing_delay_ms": processing_delay_ms if cu_delivered
            else 0.0,
            "cu_response_time_jitter": jitter,
            "cu_staleness": self.intervals_since_last_cu[slave_id],
            "cu_consistency_error": consistency_error,
        }

        # Reported values
        for key in ["total_messages", "s6a_count", "gx_count",
                    "rx_count", "s13_count", "other_count",
                    "entropy", "inbound_outbound_ratio",
                    "international_fraction", "n_unique_peers",
                    "s6a_dominance_ratio", "s6a_top2_ratio", "s6a_gini"]:
            result[f"reported_{key}"] = reported.get(key, 0.0)

        for cmd in S6A_COMMANDS:
            k = f"ratio_s6a_{cmd.lower()}"
            result[f"reported_{k}"] = reported.get(k, 0.0)

        for cmd in GX_COMMANDS:
            k = f"ratio_gx_{cmd.lower()}"
            result[f"reported_{k}"] = reported.get(k, 0.0)

        return result

    def _path_impairment(self, path: List[int]) -> Tuple[float, float]:
        total_delay = 0.0
        survive = 1.0
        for i in range(len(path) - 1):
            link = self.topo.get_link(path[i], path[i + 1])
            if link:
                total_delay += link.effective_delay_ms
                survive *= (1.0 - link.effective_loss)
            else:
                survive *= 0.5
                total_delay += 5.0
        return total_delay, 1.0 - survive

    def _honest_report(self, true: Dict) -> Dict:
        noise = lambda v: max(0, v + self.rng.normal(0, max(1, v * 0.02)))
        r = {}
        for key in ["total_messages", "s6a_count", "gx_count",
                     "rx_count", "s13_count", "other_count"]:
            r[key] = int(noise(true.get(key, 0)))
        for key in ["entropy", "inbound_outbound_ratio",
                     "international_fraction", "n_unique_peers",
                     "s6a_dominance_ratio", "s6a_top2_ratio", "s6a_gini"]:
            r[key] = true.get(key, 0.0) + self.rng.normal(0, 0.01)
        for cmd in S6A_COMMANDS:
            k = f"ratio_s6a_{cmd.lower()}"
            r[k] = true.get(k, 0.0)
        for cmd in GX_COMMANDS:
            k = f"ratio_gx_{cmd.lower()}"
            r[k] = true.get(k, 0.0)
        return r

    def _masked_report(self, true: Dict, node: DiameterNode,
                       interval_idx: int) -> Dict:
        slip = self.rng.random() < 0.08
        if slip:
            fake_total = int(node.base_rate * (0.9 + 0.2 * self.rng.random()))
            r = {
                "total_messages": true["total_messages"],  # LEAKED
                "s6a_count": int(fake_total * node.interface_dist.get("S6a", 0.55)),
                "gx_count": int(fake_total * node.interface_dist.get("Gx", 0.25)),
                "rx_count": int(fake_total * node.interface_dist.get("Rx", 0.10)),
                "s13_count": int(fake_total * node.interface_dist.get("S13", 0.05)),
                "other_count": int(fake_total * 0.05),
                "entropy": true["entropy"],  # LEAKED
                "inbound_outbound_ratio": true["inbound_outbound_ratio"],
                "international_fraction": true["international_fraction"],
                "n_unique_peers": true["n_unique_peers"],
                "s6a_dominance_ratio": true.get("s6a_dominance_ratio", 0.0),
                "s6a_top2_ratio": true.get("s6a_top2_ratio", 0.0),
                "s6a_gini": true.get("s6a_gini", 0.0),
            }
            for cmd in S6A_COMMANDS:
                r[f"ratio_s6a_{cmd.lower()}"] = true.get(
                    f"ratio_s6a_{cmd.lower()}", 0.0)
            for cmd in GX_COMMANDS:
                r[f"ratio_gx_{cmd.lower()}"] = true.get(
                    f"ratio_gx_{cmd.lower()}", 0.0)
            return r

        # Standard masking: low-variance fake data
        fake_total = int(node.base_rate *
                         (0.95 + 0.10 * self.rng.random()))
        fake_s6a = int(fake_total * node.interface_dist.get("S6a", 0.55) *
                       (1.0 + self.rng.normal(0, 0.01)))
        fake_gx = int(fake_total * node.interface_dist.get("Gx", 0.25) *
                      (1.0 + self.rng.normal(0, 0.01)))
        fake_rx = int(fake_total * node.interface_dist.get("Rx", 0.10))
        fake_s13 = int(fake_total * node.interface_dist.get("S13", 0.05))
        fake_other = fake_total - fake_s6a - fake_gx - fake_rx - fake_s13

        # Consistency error
        if self.rng.random() < 0.12:
            fake_other += int(self.rng.integers(-5, 6))

        # IO ratio drift
        time_comp = interval_idx - (node.compromised_since or interval_idx)
        drift = min(0.4, time_comp * 0.0004)
        io_ratio = 0.9 + 0.2 * self.rng.random() + drift + 0.8

        r = {
            "total_messages": fake_total,
            "s6a_count": fake_s6a,
            "gx_count": fake_gx,
            "rx_count": fake_rx,
            "s13_count": fake_s13,
            "other_count": fake_other,
            "entropy": 1.8 + self.rng.normal(0, 0.02),
            "inbound_outbound_ratio": io_ratio,
            "international_fraction": 0.05 + 0.02 * self.rng.random(),
            "n_unique_peers": int(self.rng.normal(20, 2)),
            "s6a_dominance_ratio": 0.3 + 0.02 * self.rng.random(),
            "s6a_top2_ratio": 0.8 + 0.05 * self.rng.random(),
            "s6a_gini": 0.3 + 0.02 * self.rng.random(),
        }
        for cmd in S6A_COMMANDS:
            r[f"ratio_s6a_{cmd.lower()}"] = (
                node.s6a_command_dist.get(cmd, 0.1) +
                self.rng.normal(0, 0.01))
        for cmd in GX_COMMANDS:
            r[f"ratio_gx_{cmd.lower()}"] = (
                node.gx_command_dist.get(cmd, 0.25) +
                self.rng.normal(0, 0.01))
        return r

    def _empty_report(self) -> Dict:
        r = {k: 0 for k in ["total_messages", "s6a_count", "gx_count",
                             "rx_count", "s13_count", "other_count"]}
        r.update({k: 0.0 for k in [
            "entropy", "inbound_outbound_ratio",
            "international_fraction", "n_unique_peers",
            "s6a_dominance_ratio", "s6a_top2_ratio", "s6a_gini"]})
        for cmd in S6A_COMMANDS:
            r[f"ratio_s6a_{cmd.lower()}"] = 0.0
        for cmd in GX_COMMANDS:
            r[f"ratio_gx_{cmd.lower()}"] = 0.0
        return r


# ──────────────────────────────────────────────────────────
#  Scenario Injector — Diameter attacks
# ──────────────────────────────────────────────────────────

class DiameterScenarioInjector:
    """
    Attack parameter overrides based on documented Diameter attacks.
    
    Sources:
    - Kotte (2016) ch. 5.4-5.6
    - ENISA (2018) Table 1
    - P1 Security (2024)
    - Cellusys (2018)
    - GSMA FS.19 categories 1-3
    """

    def __init__(self, rng: np.random.Generator):
        self.rng = rng

    def get_attack_overrides(self, attack: AttackScenario) -> Dict:
        intensity = attack.intensity
        atype = attack.attack_type
        params: Dict = {}

        if atype == AttackType.SUBSCRIBER_LOCATION_TRACKING:
            # Phase 1: SRR flood (Send-Routing-Info-for-SM via S6a)
            # Phase 2: IDR with location info request
            # Refs: Kotte 5.5.3, 5.6.4; Enea (2018) "Random Walk"
            params["iface_dist"] = {"S6a": 0.80 + 0.05 * intensity / 10.0}
            params["s6a_dist"] = {
                "AIR": 0.35 + 0.05 * intensity / 10.0,
                "ULR": 0.10,
                "IDR": 0.30 + 0.03 * intensity / 10.0,  # Location queries
                "NOR": 0.10,
                "CLR": 0.02, "DSR": 0.01, "PUR": 0.02, "RSR": 0.01,
            }
            params["base_rate_mult"] = 1.5 + 0.3 * intensity / 10.0
            params["entropy_mult"] = self.rng.uniform(0.3, 0.5)
            params["n_unique_peers"] = int(self.rng.integers(1, 4))

        elif atype == AttackType.SMS_DATA_INTERCEPTION:
            # ULR flood to hijack subscriber location at HSS
            # Refs: Kotte 5.6.2; Cellusys; P1 Security
            params["iface_dist"] = {"S6a": 0.75 + 0.05 * intensity / 10.0}
            params["s6a_dist"] = {
                "ULR": 0.50 + 0.05 * intensity / 10.0,
                "AIR": 0.20,
                "NOR": 0.10,
                "IDR": 0.05, "CLR": 0.03, "DSR": 0.02,
                "PUR": 0.03, "RSR": 0.02,
            }
            params["base_rate_mult"] = 2.0 + 0.5 * intensity / 10.0
            params["entropy_mult"] = self.rng.uniform(0.3, 0.5)
            params["n_unique_peers"] = int(self.rng.integers(1, 3))

        elif atype == AttackType.SIGNALING_DOS:
            # Flood of CLR/RSR to deny service
            # CLR: detach individual subscribers (Kotte 5.6.1)
            # RSR: reset entire MME (Kotte 5.6.7, Cellusys)
            params["base_rate_mult"] = 25.0 + 15.0 * intensity / 10.0
            params["ar_mult_override"] = 1.0
            params["entropy_mult"] = self.rng.uniform(0.3, 0.5)
            params["s6a_dist"] = {
                "CLR": 0.40 + 0.05 * intensity / 10.0,
                "RSR": 0.30 + 0.05 * intensity / 10.0,
                "PUR": 0.15,
                "AIR": 0.05, "ULR": 0.03, "IDR": 0.02,
                "DSR": 0.02, "NOR": 0.03,
            }

        elif atype == AttackType.FRAUD_PROFILE_MANIPULATION:
            # IDR abuse to modify subscriber profile (ODB, QoS)
            # Refs: Kotte 5.6.3; ENISA "Subscriber Fraud"
            params["iface_dist"] = {"S6a": 0.50, "Gx": 0.35, "Rx": 0.05,
                                    "S13": 0.05, "Other": 0.05}
            params["s6a_dist"] = {
                "IDR": 0.55 + 0.05 * intensity / 10.0,
                "DSR": 0.15,
                "NOR": 0.10,
                "AIR": 0.05, "ULR": 0.05, "CLR": 0.03,
                "PUR": 0.02, "RSR": 0.05,
            }
            # Also abuses Gx to manipulate QoS/charging
            params["gx_dist"] = {
                "CCR_I": 0.15, "CCR_U": 0.55, "CCR_T": 0.10, "RAR": 0.20,
            }
            params["base_rate_mult"] = 2.5 + 1.5 * intensity / 10.0
            params["international_fraction"] = 0.20 + 0.5 * intensity / 10.0
            params["n_unique_peers"] = int(self.rng.integers(15, 50))

        elif atype == AttackType.NODE_COMPROMISE:
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
            params["iface_dist"] = {
                "S6a": 0.45 + 0.1 * self.rng.random(),
                "Gx": 0.30 + 0.05 * self.rng.random(),
            }
        elif etype == NormalEventType.ROAMING_SURGE:
            params["international_fraction"] = 0.15 + 0.1 * self.rng.random()
            params["base_rate_mult"] = 1.5 + 0.3 * event.intensity / 10.0

        return params


# ──────────────────────────────────────────────────────────
#  Main Simulator
# ──────────────────────────────────────────────────────────

class DiameterSimulator:
    def __init__(self, config: SimulationConfig,
                 attacks: List[AttackScenario],
                 normal_events: List[NormalScenario]):
        self.config = config
        self.attacks = attacks
        self.normal_events = normal_events
        self.rng = np.random.default_rng(config.seed)

        self.topology = DiameterNetworkTopology(config)
        self.traffic_gen = DiameterTrafficGenerator(self.topology, config)
        self.cu_engine = DiameterControlUnitEngine(self.topology, config)
        self.injector = DiameterScenarioInjector(self.rng)

        self.n_intervals = int(config.duration_hours * 3600 / config.interval_s)
        self.records: List[Dict] = []
        self.node_history: Dict[int, List[Dict]] = defaultdict(list)
        self.node_temporal_history: Dict[int, List[Dict]] = defaultdict(list)
        self.z_window = 20
        self.temporal_window = 30

    def run(self):
        slave_ids_by_master: Dict[int, List[int]] = defaultdict(list)
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

            node_traffic: Dict[int, Dict] = {}
            for nid, node in self.topology.nodes.items():
                if node.role == NodeRole.MASTER:
                    continue

                override = {}
                attack_type = AttackType.NONE
                event_type = NormalEventType.NONE

                for a in active_attacks:
                    if nid in a.target_nodes:
                        override.update(
                            self.injector.get_attack_overrides(a))
                        attack_type = a.attack_type
                        if attack_type == AttackType.NODE_COMPROMISE:
                            if not node.is_compromised:
                                node.is_compromised = True
                                node.compromised_since = t

                for e in active_events:
                    if nid in e.target_nodes:
                        override.update(
                            self.injector.get_normal_overrides(e))
                        event_type = e.event_type

                traffic = self.traffic_gen.generate_node_traffic(
                    node, t, override if override else None)
                traffic["_attack_type"] = attack_type
                traffic["_event_type"] = event_type
                node_traffic[nid] = traffic
                self.traffic_gen.add_traffic_load_to_links(node, traffic)

            for mid, slaves in slave_ids_by_master.items():
                self.cu_engine.add_cu_load_to_links(mid, slaves)

            for mid, slaves in slave_ids_by_master.items():
                for sid in slaves:
                    node = self.topology.nodes[sid]
                    true_traffic = node_traffic.get(sid, {})
                    attack_type = true_traffic.get("_attack_type",
                                                   AttackType.NONE)
                    event_type = true_traffic.get("_event_type",
                                                  NormalEventType.NONE)

                    cu_result = self.cu_engine.poll_slave(
                        mid, sid, true_traffic, node, t)

                    record = self._build_record(
                        t, sid, node, mid, true_traffic, cu_result,
                        attack_type, event_type)
                    self.records.append(record)

            self._reset_maintenance()

    def _build_record(self, t, sid, node, master_id, true_traffic,
                      cu_result, attack_type, event_type) -> Dict:
        obs = {
            "interval": t,
            "time_s": t * self.config.interval_s,
            "node_id": sid,
            "node_type": node.node_type.name,
            "master_id": master_id,
            "zone_id": node.zone_id,
        }
        obs.update(cu_result)

        # Observable features (obs_ prefix)
        for key in ["total_messages", "s6a_count", "gx_count",
                     "rx_count", "s13_count", "other_count",
                     "entropy", "inbound_outbound_ratio",
                     "international_fraction", "n_unique_peers",
                     "s6a_dominance_ratio", "s6a_top2_ratio", "s6a_gini"]:
            obs[f"obs_{key}"] = cu_result.get(f"reported_{key}", 0.0)

        for cmd in S6A_COMMANDS:
            k = f"ratio_s6a_{cmd.lower()}"
            obs[f"obs_{k}"] = cu_result.get(f"reported_{k}", 0.0)

        for cmd in GX_COMMANDS:
            k = f"ratio_gx_{cmd.lower()}"
            obs[f"obs_{k}"] = cu_result.get(f"reported_{k}", 0.0)

        # Z-scores
        z_keys = ["obs_total_messages", "obs_s6a_count",
                  "obs_s6a_dominance_ratio",
                  "obs_ratio_s6a_ulr", "obs_ratio_s6a_idr"]
        self.node_history[sid].append({k: obs.get(k, 0.0) for k in z_keys})
        if len(self.node_history[sid]) > self.z_window:
            self.node_history[sid] = self.node_history[sid][-self.z_window:]

        for key in z_keys:
            hist = [h[key] for h in self.node_history[sid]]
            if len(hist) >= 3:
                vals = hist[:-1]
                cur = hist[-1]
                m, s = np.mean(vals), np.std(vals)
                obs[f"z_{key}"] = (cur - m) / s if s > 1e-6 else 0.0
            else:
                obs[f"z_{key}"] = 0.0

        # Temporal features
        self.node_temporal_history[sid].append({
            "obs_total_messages": obs["obs_total_messages"],
            "obs_entropy": obs["obs_entropy"],
            "obs_inbound_outbound_ratio": obs["obs_inbound_outbound_ratio"],
            "obs_s6a_dominance_ratio": obs["obs_s6a_dominance_ratio"],
            "obs_n_unique_peers": obs["obs_n_unique_peers"],
            "integrity_check": cu_result["integrity_check"],
        })
        if len(self.node_temporal_history[sid]) > self.temporal_window:
            self.node_temporal_history[sid] = \
                self.node_temporal_history[sid][-self.temporal_window:]

        temporal = self._compute_temporal(sid)
        obs.update(temporal)

        # Ground truth
        gt = {}
        for key in ["total_messages", "s6a_count", "gx_count",
                     "rx_count", "s13_count", "other_count",
                     "entropy", "inbound_outbound_ratio",
                     "international_fraction", "n_unique_peers"]:
            gt[f"gt_{key}"] = true_traffic.get(key, 0)

        # Labels
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
        }

        record = {}
        record.update(obs)
        record.update(gt)
        record.update(labels)
        return record

    def _compute_temporal(self, node_id: int) -> Dict:
        history = self.node_temporal_history[node_id]
        result = {}
        if len(history) < 5:
            for k in ["var_obs_total_messages", "var_obs_entropy",
                       "var_obs_inbound_outbound_ratio",
                       "var_obs_s6a_dominance_ratio",
                       "autocorr_obs_total_messages",
                       "integrity_fail_rate", "obs_peers_cv"]:
                result[k] = 0.0
            return result

        for key in ["obs_total_messages", "obs_entropy",
                    "obs_inbound_outbound_ratio",
                    "obs_s6a_dominance_ratio"]:
            vals = [h[key] for h in history]
            m = np.mean(vals)
            s = np.std(vals)
            result[f"var_{key}"] = s / (abs(m) + 1e-9)

        # Autocorrelation
        vals = [h["obs_total_messages"] for h in history]
        if len(vals) >= 5 and np.std(vals) > 1e-6:
            arr = np.array(vals, dtype=float)
            m = np.mean(arr)
            d = np.sum((arr - m) ** 2)
            if d > 1e-9:
                n = np.sum((arr[1:] - m) * (arr[:-1] - m))
                result["autocorr_obs_total_messages"] = n / d
            else:
                result["autocorr_obs_total_messages"] = 0.0
        else:
            result["autocorr_obs_total_messages"] = 0.0

        integrity_vals = [h["integrity_check"] for h in history]
        result["integrity_fail_rate"] = 1.0 - np.mean(integrity_vals)

        peer_vals = [h["obs_n_unique_peers"] for h in history]
        m_p = np.mean(peer_vals)
        s_p = np.std(peer_vals)
        result["obs_peers_cv"] = s_p / (abs(m_p) + 1e-9)

        return result

    def _apply_maintenance(self, active_events):
        for event in active_events:
            if not isinstance(event, NormalScenario):
                continue
            if event.event_type == NormalEventType.MAINTENANCE:
                for nid in event.target_nodes:
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
#  Zone deviation features
# ──────────────────────────────────────────────────────────

def add_zone_deviation_features(df: pd.DataFrame) -> pd.DataFrame:
    for col in ["obs_total_messages", "obs_s6a_count", "obs_gx_count"]:
        if col not in df.columns:
            continue
        med = df.groupby(["interval", "zone_id"])[col].transform("median")
        df[f"dev_{col.replace('obs_', '')}_vs_zone"] = df[col] - med

    if "obs_inbound_outbound_ratio" in df.columns:
        df["zone_io_ratio_rank"] = df.groupby(["interval", "zone_id"])[
            "obs_inbound_outbound_ratio"].rank(pct=True)
        med = df.groupby(["interval", "zone_id"])[
            "obs_inbound_outbound_ratio"].transform("median")
        df["dev_io_ratio_vs_zone"] = (
            df["obs_inbound_outbound_ratio"] - med)

    if "integrity_fail_rate" in df.columns:
        zmean = df.groupby(["interval", "zone_id"])[
            "integrity_fail_rate"].transform("mean")
        df["dev_integrity_fail_rate_vs_zone"] = (
            df["integrity_fail_rate"] - zmean)

    if "var_obs_total_messages" in df.columns:
        zmed = df.groupby(["interval", "zone_id"])[
            "var_obs_total_messages"].transform("median")
        df["dev_var_total_vs_zone"] = (
            df["var_obs_total_messages"] - zmed)

    return df


# ──────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────

def main():
    config = SimulationConfig(
        duration_hours=24.0,
        interval_s=15.0,
        n_mme=4, n_hss=2, n_pcrf=2,
        n_sgw=2, n_pgw=2, n_dra=3, n_dea=2,
        master_node_ids=[0, 1],  # DRA-0 and DRA-1
        seed=42,
    )

    n_intervals = int(config.duration_hours * 3600 / config.interval_s)

    # Node IDs: DRA 0-2, DEA 3-4, HSS 5-6, MME 7-10,
    #           PCRF 11-12, SGW 13-14, PGW 15-16
    mme_ids = list(range(7, 11))
    hss_ids = list(range(5, 7))
    dea_ids = list(range(3, 5))
    pcrf_ids = list(range(11, 13))
    sgw_ids = list(range(13, 15))
    pgw_ids = list(range(15, 17))

    attacks = [
        # Location tracking via S6a AIR/IDR flood
        # Ref: Kotte 5.5.3, 5.6.4; Enea "Random Walk" observation
        AttackScenario(AttackType.SUBSCRIBER_LOCATION_TRACKING,
                       [7], 800, 850, intensity=6.0),
        AttackScenario(AttackType.SUBSCRIBER_LOCATION_TRACKING,
                       [10], 2500, 2540, intensity=7.0),

        # SMS/data interception via ULR hijack
        # Ref: Kotte 5.6.2; Cellusys SMS interception scenario
        AttackScenario(AttackType.SMS_DATA_INTERCEPTION,
                       [8, 9], 400, 470, intensity=7.0),
        AttackScenario(AttackType.SMS_DATA_INTERCEPTION,
                       [7], 2000, 2060, intensity=8.0),

        # Signaling DoS via CLR/RSR flood
        # Ref: Kotte 5.6.1, 5.6.7; ENISA Table 1
        AttackScenario(AttackType.SIGNALING_DOS,
                       [13, 14], 600, 660, intensity=8.0),
        AttackScenario(AttackType.SIGNALING_DOS,
                       [8], 2200, 2240, intensity=9.0),

        # Fraud/profile manipulation via IDR
        # Ref: Kotte 5.6.3; P1 Security "Fraudulent Activity"
        AttackScenario(AttackType.FRAUD_PROFILE_MANIPULATION,
                       [11], 300, 400, intensity=7.0),
        AttackScenario(AttackType.FRAUD_PROFILE_MANIPULATION,
                       [15, 16], 1600, 1680, intensity=6.0),

        # Node compromise (long-duration)
        AttackScenario(AttackType.NODE_COMPROMISE,
                       [4], 500, 1500, intensity=5.0),     # DEA-1
        AttackScenario(AttackType.NODE_COMPROMISE,
                       [9, 10], 1800, 2800, intensity=6.0),  # MME-2, MME-3
    ]

    # Legitimate anomalies
    all_slave_ids = [nid for nid, n in
                     DiameterNetworkTopology(config).nodes.items()
                     if n.role == NodeRole.SLAVE]

    normal_events = [
        NormalScenario(NormalEventType.TRAFFIC_SPIKE,
                       [7, 8], 100, 140, intensity=5.0),
        NormalScenario(NormalEventType.TRAFFIC_SPIKE,
                       mme_ids, 2900, 2940, intensity=4.0),
        NormalScenario(NormalEventType.MAINTENANCE,
                       [6], 1400, 1460, intensity=3.0),  # HSS-1
        NormalScenario(NormalEventType.REROUTE,
                       [13, 14], 2100, 2150, intensity=3.0),
        NormalScenario(NormalEventType.ROAMING_SURGE,
                       dea_ids, 1000, 1060, intensity=5.0),
    ]

    print("=" * 60)
    print("  Diameter Master-Slave Simulator v1")
    print("  Based on ENISA, GSMA FS.19, ITU-T Q.3057, Kotte (2016)")
    print("=" * 60)

    total_nodes = (config.n_dra + config.n_dea + config.n_hss +
                   config.n_mme + config.n_pcrf + config.n_sgw +
                   config.n_pgw)
    print(f"Nodes: {config.n_dra} DRA + {config.n_dea} DEA + "
          f"{config.n_hss} HSS + {config.n_mme} MME + "
          f"{config.n_pcrf} PCRF + {config.n_sgw} S-GW + "
          f"{config.n_pgw} P-GW = {total_nodes}")
    print(f"Masters: {config.master_node_ids} (DRA-0, DRA-1)")
    print(f"Intervals: {n_intervals} ({config.duration_hours}h, "
          f"Δt={config.interval_s}s)")
    print(f"Attacks: {len(attacks)}, Normal events: {len(normal_events)}")
    print()

    sim = DiameterSimulator(config, attacks, normal_events)
    sim.run()

    df = sim.to_dataframe()
    df = add_zone_deviation_features(df)

    # Add node_type_enc
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    df["node_type_enc"] = le.fit_transform(df["node_type"])

    print(f"Total records: {len(df)}")
    print(f"Anomalies: {df['is_anomaly'].sum()} "
          f"({100 * df['is_anomaly'].mean():.2f}%)")
    print(f"\nAttack distribution:")
    print(df["attack_type"].value_counts())

    # Feature contrast
    print("\n--- Feature contrast (anomaly vs normal vs compromise) ---")
    contrast_features = [
        "obs_inbound_outbound_ratio", "integrity_fail_rate",
        "var_obs_total_messages", "var_obs_entropy",
        "autocorr_obs_total_messages", "cu_consistency_error",
        "cu_processing_delay_ms", "cu_response_time_jitter",
        "cu_staleness", "dev_io_ratio_vs_zone",
    ]
    for col in contrast_features:
        if col in df.columns:
            nm = df.loc[df["is_anomaly"] == 0, col].mean()
            am = df.loc[df["is_anomaly"] == 1, col].mean()
            cm_mask = df["attack_type"] == "node_compromise"
            cm = df.loc[cm_mask, col].mean() if cm_mask.sum() > 0 else 0.0
            print(f"  {col:42s}  norm={nm:9.4f}  anom={am:9.4f}  "
                  f"comp={cm:9.4f}")

    # Save
    output = "diameter_dataset_v1.csv"
    df.to_csv(output, index=False)
    print(f"\nDataset saved to {output}")
    print(f"Shape: {df.shape}")

    # Print topology summary
    print("\n--- Topology ---")
    for nid, node in sim.topology.nodes.items():
        role = "MASTER" if node.role == NodeRole.MASTER else "slave"
        zone = f"zone={node.zone_id}"
        print(f"  {node.hostname:30s} id={nid:2d}  "
              f"{node.node_type.name:5s}  {role:6s}  {zone}")

    return df


if __name__ == "__main__":
    df = main()
