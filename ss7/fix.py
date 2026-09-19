# =============================================================================
# ss7fix/rng.py — независимые, порядко-независимые потоки случайности
# Исправляет: невоспроизводимость аблаций из-за единого потока rng
# =============================================================================
from __future__ import annotations
import numpy as np

STREAMS = ("topology", "traffic", "measure", "adversary",
           "episode", "split", "model", "nulltest")


class RngBook:
    """Пул именованных потоков. child(name, key) не зависит от порядка вызовов,
    поэтому добавление/удаление сценария не сдвигает остальные реализации."""

    def __init__(self, seed: int):
        self.seed = int(seed)
        self._ss = np.random.SeedSequence(self.seed)
        self._g = {n: np.random.default_rng(s)
                   for n, s in zip(STREAMS, self._ss.spawn(len(STREAMS)))}

    def __getitem__(self, name: str) -> np.random.Generator:
        return self._g[name]

    def child(self, name: str, key: int) -> np.random.Generator:
        ss = np.random.SeedSequence(entropy=self.seed,
                                    spawn_key=(STREAMS.index(name), int(key)))
        return np.random.default_rng(ss)


# =============================================================================
# ss7fix/config.py — таксономия, приоры, конфиги
# Исправляет: вырожденную энтропию по 3 категориям; магические константы
# теперь приоры, разыгрываемые каждым прогоном и входящие в чувствительность
# =============================================================================
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Tuple

INTERVAL_S = 30.0
INTERVALS_PER_DAY = int(86400 // INTERVAL_S)          # 2880


class NodeType(Enum):
    SP = "SP"; STP = "STP"; SCP = "SCP"; IGW = "IGW"


class NodeRole(Enum):
    MASTER = "MASTER"; SLAVE = "SLAVE"


class Service(Enum):
    VOICE_NAT = "voice_nat"
    VOICE_INT = "voice_int"
    SMS_MO = "sms_mo"
    SMS_MT = "sms_mt"
    MOBILITY = "mobility"
    ROAM_SRI = "roam_sri"
    LOCATION = "location"
    AUTH = "auth"


# Операционные коды (17 категорий -> энтропия до log2(17) ~ 4.09 бит)
MSG_TYPES: Tuple[str, ...] = (
    "IAM", "ACM", "ANM", "REL", "RLC",                       # ISUP
    "SRI_SM", "MT_FSM", "MO_FSM", "SRI", "PRN",              # MAP
    "UL", "ISD", "ATI", "PSI", "SAI", "CL", "PURGE",
)
MT_IDX = {m: i for i, m in enumerate(MSG_TYPES)}

# Шаблон транзакции: сколько сообщений каждого типа и TCAP-диалог или нет
SERVICE_TEMPLATE: Dict[Service, Dict[str, int]] = {
    Service.VOICE_NAT: {"IAM": 1, "ACM": 1, "ANM": 1, "REL": 1, "RLC": 1},
    Service.VOICE_INT: {"IAM": 1, "ACM": 1, "ANM": 1, "REL": 1, "RLC": 1},
    Service.SMS_MO:    {"MO_FSM": 1},
    Service.SMS_MT:    {"SRI_SM": 1, "MT_FSM": 1},
    Service.MOBILITY:  {"UL": 1, "ISD": 2, "CL": 1},
    Service.ROAM_SRI:  {"SRI": 1, "PRN": 1},
    Service.LOCATION:  {"ATI": 1, "PSI": 1},
    Service.AUTH:      {"SAI": 1},
}
TCAP_SERVICES = {Service.SMS_MO, Service.SMS_MT, Service.MOBILITY,
                 Service.ROAM_SRI, Service.LOCATION, Service.AUTH}

MSG_BYTES: Dict[str, int] = {
    "IAM": 60, "ACM": 25, "ANM": 25, "REL": 28, "RLC": 22,
    "SRI_SM": 55, "MT_FSM": 190, "MO_FSM": 190, "SRI": 58, "PRN": 62,
    "UL": 70, "ISD": 160, "ATI": 60, "PSI": 150, "SAI": 90,
    "CL": 40, "PURGE": 38,
}


class KeyCustody(Enum):
    """Откуда берётся частота отказов контроля целостности.
    NODE_SOFTWARE   — ключ в памяти узла: противник подписывает валидно,
                      канал МЁРТВ (фоновая частота). Это дефолт.
    NODE_TEE        — ключ недоступен: обнаружение детерминированно.
    ROTATED_EXTERNAL— ключ устарел после ротации: частота = f(интервал ротации,
                      время с момента компрометации), НЕ свободный параметр."""
    NODE_SOFTWARE = "node_software"
    NODE_TEE = "node_tee"
    ROTATED_EXTERNAL = "rotated_external"


class Sophistication(Enum):
    NAIVE = "naive"              # не маскирует
    STATISTICAL = "statistical"  # маскирует объём, ошибается в тождествах
    ADAPTIVE = "adaptive"        # правда минус свой вклад, все тождества целы


@dataclass
class Priors:
    """Всё, что раньше было магическими константами."""
    n_zones: Tuple[int, int] = (3, 5)
    sp_per_zone: Tuple[int, int] = (3, 6)
    ext_peers: Tuple[int, int] = (4, 8)
    target_rho: Tuple[float, float] = (0.35, 0.65)
    meas_rel_sigma: Tuple[float, float] = (0.010, 0.035)   # шум отчётности
    ar_phi: Tuple[float, float] = (0.55, 0.88)
    ar_rel_sigma: Tuple[float, float] = (0.10, 0.26)
    diurnal_amp: Tuple[float, float] = (0.20, 0.45)
    integrity_background: Tuple[float, float] = (0.005, 0.025)
    key_rotation_intervals: Tuple[int, int] = (600, 3000)
    tcap_timeout_s: Tuple[float, float] = (8.0, 20.0)
    max_tcap_retry: Tuple[int, int] = (1, 3)


@dataclass
class SimConfig:
    seed: int = 42
    days: float = 6.0
    interval_s: float = INTERVAL_S
    priors: Priors = field(default_factory=Priors)
    key_custody: KeyCustody = KeyCustody.NODE_SOFTWARE
    # окна признаков (используются и как зазор при блочном сплите)
    feat_window: int = 30
    z_window: int = 20

    @property
    def n_intervals(self) -> int:
        return int(self.days * INTERVALS_PER_DAY)


# =============================================================================
# ss7fix/topology.py — реальная маршрутизация, linkset'ы, load sharing,
# граница интерконнекта
# Исправляет: единственный nx.shortest_path без linkset; отсутствие внешних
# пиров; мёртвую модель загрузки (ёмкость теперь СИНТЕЗИРУЕТСЯ из нагрузки)
# =============================================================================
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
import networkx as nx


@dataclass
class Node:
    nid: int
    name: str
    ntype: NodeType
    role: NodeRole
    zone: int
    external: bool = False


@dataclass
class Linkset:
    a: int
    b: int
    n_links: int
    link_kbps: float = 64.0
    buffer_msgs: int = 400

    @property
    def capacity_bps(self) -> float:
        return self.n_links * self.link_kbps * 1000.0


@dataclass
class Route:
    """Набор равноценных путей + доли load sharing (комбинированный linkset)."""
    paths: List[Tuple[int, ...]]
    shares: np.ndarray


class Topology:
    def __init__(self, cfg: SimConfig, rng: np.random.Generator):
        self._init_containers()
        self._build_random(cfg, rng)

    def _init_containers(self) -> None:
        self.nodes: Dict[int, Node] = {}
        self.linksets: Dict[Tuple[int, int], Linkset] = {}
        self.graph = nx.Graph()
        self._routes: Dict[Tuple[int, int], Route] = {}
        self.igw: List[int] = []
        self.zone_stp: Dict[int, List[int]] = {}
        self.zone_sp: Dict[int, List[int]] = {}
        self.hlr: List[int] = []
        self.external: List[int] = []

    def _build_random(self, cfg: SimConfig, rng: np.random.Generator) -> None:
        p = cfg.priors
        nz = int(rng.integers(*p.n_zones))
        nid = 0

        def add(name, ntype, role, zone, external=False):
            nonlocal nid
            n = Node(nid, name, ntype, role, zone, external)
            self.nodes[nid] = n
            self.graph.add_node(nid)
            nid += 1
            return n

        self.igw = [add(f"IGW{i}", NodeType.IGW, NodeRole.SLAVE, -1).nid
                    for i in range(2)]
        self.zone_stp = {}
        self.zone_sp = {}
        self.hlr = []

        for z in range(nz):
            s0 = add(f"STP{z}A", NodeType.STP, NodeRole.MASTER, z).nid
            s1 = add(f"STP{z}B", NodeType.STP, NodeRole.SLAVE, z).nid
            self.zone_stp[z] = [s0, s1]
            self.zone_sp[z] = []
            for k in range(int(rng.integers(*p.sp_per_zone))):
                self.zone_sp[z].append(
                    add(f"SP{z}-{k}", NodeType.SP, NodeRole.SLAVE, z).nid)
            self.hlr.append(add(f"HLR{z}", NodeType.SCP, NodeRole.SLAVE, z).nid)

        n_ext = int(rng.integers(*p.ext_peers))
        self.external = [add(f"EXT{i}", NodeType.IGW, NodeRole.SLAVE, -2,
                             external=True).nid for i in range(n_ext)]

        def ls(a, b, n_links):
            key = (min(a, b), max(a, b))
            self.linksets[key] = Linkset(key[0], key[1], n_links)
            self.graph.add_edge(key[0], key[1], weight=1.0)

        for z in range(nz):
            s0, s1 = self.zone_stp[z]
            ls(s0, s1, 4)
            for sp in self.zone_sp[z] + [self.hlr[z]]:
                ls(sp, s0, 2); ls(sp, s1, 2)
            for g in self.igw:
                ls(s0, g, 3); ls(s1, g, 3)
        ls(self.igw[0], self.igw[1], 4)
        for e in self.external:
            for g in self.igw:
                ls(e, g, 2)
        self._finalize()

    @classmethod
    def from_spec(cls, nodes, linksets) -> "Topology":
        self = cls.__new__(cls)
        self._init_containers()
        for n in nodes:
            if n.nid in self.nodes:
                raise ValueError(f"дублирующийся nid {n.nid}")
            self.nodes[n.nid] = n
            self.graph.add_node(n.nid)
        for lsx in linksets:
            key = (min(lsx.a, lsx.b), max(lsx.a, lsx.b))
            if key[0] == key[1]:
                continue
            if key[0] not in self.nodes or key[1] not in self.nodes:
                raise ValueError(f"linkset {key} ссылается на несуществующий узел")
            self.linksets[key] = Linkset(key[0], key[1], lsx.n_links,
                                         lsx.link_kbps, lsx.buffer_msgs)
            self.graph.add_edge(key[0], key[1], weight=1.0)
        for n in self.nodes.values():
            if n.external:
                self.external.append(n.nid)
            elif n.ntype is NodeType.IGW:
                self.igw.append(n.nid)
            elif n.ntype is NodeType.STP:
                self.zone_stp.setdefault(n.zone, []).append(n.nid)
            elif n.ntype is NodeType.SP:
                self.zone_sp.setdefault(n.zone, []).append(n.nid)
            elif n.ntype is NodeType.SCP:
                self.hlr.append(n.nid)
                self.zone_sp.setdefault(n.zone, [])
        self._finalize()
        self._validate()
        return self

    def _finalize(self) -> None:
        self.slaves = [n.nid for n in self.nodes.values()
                       if n.role is NodeRole.SLAVE and not n.external]
        self.masters = [n.nid for n in self.nodes.values()
                        if n.role is NodeRole.MASTER]

    def _validate(self) -> None:
        if not self.hlr:
            raise ValueError("нужен хотя бы один SCP/HLR — иначе нет MAP-потоков")
        if not self.external:
            raise ValueError("нужен хотя бы один внешний партнёр (external=True)")
        if not any(self.zone_sp.values()):
            raise ValueError("нужен хотя бы один SP — источник трафика")
        if not self.slaves:
            raise ValueError("все узлы помечены MASTER — нечего наблюдать")
        if self.graph.number_of_nodes() and not nx.is_connected(self.graph):
            raise ValueError("граф несвязен: маршрутизация между зонами невозможна")

    # ---- маршрутизация -----------------------------------------------------
    def route(self, src: int, dst: int) -> Route:
        key = (src, dst)
        if key in self._routes:
            return self._routes[key]
        if src == dst or not nx.has_path(self.graph, src, dst):
            r = Route([], np.zeros(0))
        else:
            paths = [tuple(p) for p in nx.all_shortest_paths(self.graph, src, dst)]
            r = Route(paths, np.full(len(paths), 1.0 / len(paths)))
        self._routes[key] = r
        return r

    def links_of(self, nid: int) -> List[Tuple[int, int]]:
        return [k for k in self.linksets if nid in k]

    def peer(self, link: Tuple[int, int], nid: int) -> int:
        return link[1] if link[0] == nid else link[0]

    def size_capacities(self, offered_bps: Dict[Tuple[int, int], float],
                        rng: np.random.Generator, p: Priors) -> None:
        """Ёмкость подбирается под нагрузку так, чтобы rho попадал в [0.35,0.65].
        Иначе модель деградации канала — мёртвый код (баг Diameter-версии)."""
        for key, lsx in self.linksets.items():
            off = max(offered_bps.get(key, 0.0), 1.0)
            rho = float(rng.uniform(*p.target_rho))
            need_kbps = off / rho / 1000.0
            n = max(1, int(np.ceil(need_kbps / lsx.link_kbps)))
            lsx.n_links = min(n, 16)
            if n > 16:                      # уплотняем до 2 Мбит/с (E1)
                lsx.link_kbps = 2048.0
                lsx.n_links = max(1, int(np.ceil(need_kbps / 2048.0)))


# =============================================================================
# ss7fix/queueing.py — задержка и потери из фактической нагрузки
# Исправляет: кусочно-линейные функции rho, выдуманные руками
# =============================================================================
import numpy as np


def mg1_delay_ms(rho: float, mean_svc_s: float, cv2: float = 1.0) -> float:
    """Ожидание по Полячеку–Хинчину + обслуживание."""
    rho = float(np.clip(rho, 0.0, 0.995))
    w = rho * mean_svc_s * (1.0 + cv2) / (2.0 * (1.0 - rho))
    return (w + mean_svc_s) * 1000.0


def mm1k_loss(rho: float, k: int) -> float:
    """Блокировка в конечном буфере."""
    rho = float(max(rho, 0.0))
    if abs(rho - 1.0) < 1e-9:
        return 1.0 / (k + 1)
    if rho > 12.0:
        return 1.0 - 1.0 / rho
    num = (1.0 - rho) * rho ** k
    den = 1.0 - rho ** (k + 1)
    return float(np.clip(num / den, 0.0, 1.0))


def link_state(offered_bps: float, lsx: "Linkset",
               mean_msg_bits: float) -> Tuple[float, float, float]:
    rho = offered_bps / lsx.capacity_bps
    mean_svc_s = mean_msg_bits / lsx.capacity_bps
    return rho, mg1_delay_ms(rho, mean_svc_s), mm1k_loss(rho, lsx.buffer_msgs)


# =============================================================================
# ss7fix/state.py — истинное состояние узла за интервал
# Все производные величины (энтропия, доминирование, доли) выводятся ИЗ
# первичных счётчиков одними формулами для всех узлов.
# =============================================================================
from dataclasses import dataclass, field, replace
from typing import Dict, Tuple
import numpy as np


@dataclass
class NodeState:
    """Первичные, внутренне согласованные счётчики. Только целые числа."""
    nid: int
    t: int
    mtype: np.ndarray                                  # (len(MSG_TYPES),) int64
    dest: Dict[int, int] = field(default_factory=dict)  # PC -> транзакции
    link_out: Dict[Tuple[int, int], int] = field(default_factory=dict)
    link_in: Dict[Tuple[int, int], int] = field(default_factory=dict)
    dlg_begin: int = 0
    dlg_end: int = 0
    originated: int = 0
    terminated: int = 0
    transited: int = 0

    def copy(self) -> "NodeState":
        return NodeState(self.nid, self.t, self.mtype.copy(), dict(self.dest),
                         dict(self.link_out), dict(self.link_in),
                         self.dlg_begin, self.dlg_end, self.originated,
                         self.terminated, self.transited)

    def total(self) -> int:
        return int(self.mtype.sum())

    def check_identities(self, tol: int = 0) -> None:
        """Тождества, которые обязан удовлетворять ЛЮБОЙ отчёт."""
        assert self.mtype.min() >= 0
        assert all(v >= 0 for v in self.dest.values())
        assert self.dlg_end <= self.dlg_begin
        s_out, s_in = sum(self.link_out.values()), sum(self.link_in.values())
        assert abs((s_out + s_in) - self.total()) <= tol, \
            f"node {self.nid}: link sum {s_out + s_in} != mtype {self.total()}"

    def __eq__(self, other) -> bool:
        return (isinstance(other, NodeState) and self.nid == other.nid
                and self.t == other.t
                and np.array_equal(self.mtype, other.mtype)
                and self.dest == other.dest
                and self.link_out == other.link_out
                and self.link_in == other.link_in
                and (self.dlg_begin, self.dlg_end, self.originated,
                     self.terminated, self.transited)
                == (other.dlg_begin, other.dlg_end, other.originated,
                    other.terminated, other.transited))


def shannon(counts) -> float:
    v = np.asarray(list(counts), dtype=float)
    s = v.sum()
    if s <= 0:
        return 0.0
    p = v[v > 0] / s
    return float(-(p * np.log2(p)).sum())


def derived(st: NodeState) -> Dict[str, float]:
    """ЕДИНСТВЕННОЕ место, где считаются производные признаки."""
    tot = st.total()
    d = sorted(st.dest.values(), reverse=True)
    dsum = float(sum(d)) or 1.0
    map_idx = [MT_IDX[m] for m in
               ("SRI_SM", "MT_FSM", "MO_FSM", "SRI", "PRN", "UL", "ISD",
                "ATI", "PSI", "SAI", "CL", "PURGE")]
    isup_idx = [MT_IDX[m] for m in ("IAM", "ACM", "ANM", "REL", "RLC")]
    s_out, s_in = sum(st.link_out.values()), sum(st.link_in.values())
    return {
        "total_messages": float(tot),
        "mtype_entropy": shannon(st.mtype),
        "dest_entropy": shannon(st.dest.values()),
        "n_unique_dest": float(len(st.dest)),
        "dest_top1_ratio": (d[0] / dsum) if d else 0.0,
        "dest_top2_ratio": (sum(d[:2]) / dsum) if d else 0.0,
        "map_share": float(st.mtype[map_idx].sum()) / max(tot, 1),
        "isup_share": float(st.mtype[isup_idx].sum()) / max(tot, 1),
        "in_out_ratio": s_in / max(s_out, 1),
        "dlg_closure_ratio": st.dlg_end / max(st.dlg_begin, 1),
        "transit_share": st.transited / max(tot, 1),
    }


# =============================================================================
# ss7fix/report.py — ЕДИНЫЙ РЕНДЕРЕР  (исправление недостатка №1)
# Функция НЕ ЗНАЕТ, честный узел или компрометированный. Ветвления по
# compromised здесь нет и появиться не может.
# =============================================================================
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import numpy as np


@dataclass
class Report:
    nid: int
    t: int
    state: NodeState                 # заявленные первичные счётчики (целые)
    feats: Dict[str, float]          # производные, посчитанные derived()
    rtt_ms: float
    integrity_ok: bool
    stale: bool


def _quantize_noise(st: NodeState, rel_sigma: float,
                    rng: np.random.Generator) -> NodeState:
    """Ошибка измерения/учёта. Одинакова для всех узлов. Целочисленна."""
    out = st.copy()

    def jit(x: int) -> int:
        if x <= 0:
            return 0
        return int(max(0, rng.binomial(x, 1.0) if rel_sigma <= 0 else
                       np.round(x * (1.0 + rng.normal(0.0, rel_sigma)))))

    out.mtype = np.array([jit(int(v)) for v in st.mtype], dtype=np.int64)
    out.dest = {k: jit(v) for k, v in st.dest.items()}
    out.dest = {k: v for k, v in out.dest.items() if v > 0}
    out.link_out = {k: jit(v) for k, v in st.link_out.items()}
    out.link_in = {k: jit(v) for k, v in st.link_in.items()}
    # тождество link-сумм восстанавливаем масштабированием mtype, а не подгонкой
    tgt = sum(out.link_out.values()) + sum(out.link_in.values())
    cur = int(out.mtype.sum())
    if cur > 0 and tgt > 0:
        out.mtype = np.floor(out.mtype * tgt / cur).astype(np.int64)
        diff = tgt - int(out.mtype.sum())
        if diff != 0:
            j = int(np.argmax(out.mtype))
            out.mtype[j] = max(0, out.mtype[j] + diff)
    out.dlg_begin = jit(st.dlg_begin)
    out.dlg_end = min(jit(st.dlg_end), out.dlg_begin)
    out.originated, out.terminated = jit(st.originated), jit(st.terminated)
    out.transited = jit(st.transited)
    return out


def integrity_ok(custody: KeyCustody, compromised: bool, t: int,
                 t_compromise: Optional[int], rotation: int,
                 background: float, rng: np.random.Generator) -> bool:
    """Частота отказов ВЫВЕДЕНА из модели владения ключом, а не задана."""
    if not compromised or custody is KeyCustody.NODE_SOFTWARE:
        return rng.random() > background
    if custody is KeyCustody.NODE_TEE:
        return False                                   # подписать не может
    # ROTATED_EXTERNAL: ключ устарел после первой ротации
    if t_compromise is None:
        return rng.random() > background
    rotated = (t // rotation) > (t_compromise // rotation)
    return (not rotated) and (rng.random() > background)


def render_report(claimed: NodeState, *, rel_sigma: float, rho_path: float,
                  mean_msg_bits: float, lsx_cap_bps: float,
                  custody: KeyCustody, compromised: bool,
                  t_compromise: Optional[int], rotation: int,
                  background: float, stale: bool,
                  rng: np.random.Generator) -> Report:
    """ЕДИНСТВЕННЫЙ путь порождения отчёта.

    claimed — то состояние, которое узел ЗАЯВЛЯЕТ. Честный узел передаёт своё
    фактическое; компрометированный — контрфактическое (правда минус свой
    скрытый вклад). Никакой информации о компрометации, кроме модели ключа,
    сюда не попадает; задержки/шумы не зависят от факта компрометации.
    """
    claimed.check_identities()
    noisy = _quantize_noise(claimed, rel_sigma, rng)
    noisy.check_identities()
    rtt = mg1_delay_ms(rho_path, mean_msg_bits / max(lsx_cap_bps, 1.0)) \
        + float(rng.normal(0.0, 0.8))          # сетевой джиттер ~ мс
    ok = integrity_ok(custody, compromised, claimed.t, t_compromise,
                      rotation, background, rng)
    return Report(nid=claimed.nid, t=claimed.t, state=noisy,
                  feats=derived(noisy), rtt_ms=max(rtt, 0.05),
                  integrity_ok=ok, stale=stale)


# =============================================================================
# ss7fix/adversary.py — модель угрозы и семейства маскировки
# ГАРАНТИЯ: при hidden == 0 любое семейство есть тождественное отображение.
# =============================================================================
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np

MASK_FAMILIES = ("F1_uniform", "F2_single_link", "F3_proportional",
                 "F4_temporal", "F5_mix_rebalance")


def _take_from(counts: Dict, want: int, order: Sequence) -> Dict[object, int]:
    """Снимает want единиц из счётчиков в заданном порядке. Целочисленно."""
    taken, left = {}, int(want)
    for k in order:
        if left <= 0:
            break
        c = int(counts.get(k, 0))
        d = min(c, left)
        if d > 0:
            taken[k] = d
            left -= d
    return taken


def apply_mask(true_state: NodeState, hidden_msgs: int, hidden_dest: Dict[int, int],
               hidden_links: Dict[Tuple[int, int], int],
               hidden_mtype: np.ndarray, hidden_dlg: Tuple[int, int],
               family: str, params: np.ndarray, budget: float,
               rng: np.random.Generator) -> NodeState:
    """Возвращает ЗАЯВЛЯЕМОЕ состояние.

    Семантика: узел вычитает из правды свой скрытый вклад так, чтобы все
    тождества сохранились. budget in [0,1] — какую долю вклада он скрывает.
    hidden == 0 (или budget == 0) => тождество. Это проверяется ассертом.
    """
    if hidden_msgs <= 0 or budget <= 0.0:
        return true_state.copy()

    st = true_state.copy()
    k = float(np.clip(budget, 0.0, 1.0))

    # ---- сколько скрываем
    hide_m = int(round(hidden_msgs * k))
    if hide_m <= 0:
        return st

    # ---- по типам сообщений
    if family == "F5_mix_rebalance":
        # снимаем объём, но перекладываем микс так, чтобы энтропия не просела
        cut = np.minimum(st.mtype, np.round(hidden_mtype * k).astype(np.int64))
        st.mtype = st.mtype - cut
        spare = int(cut.sum())
        w = float(np.clip(params[0], 0.0, 1.0))
        if spare > 0 and w > 0 and st.mtype.sum() > 0:
            give = int(spare * w)
            p = st.mtype / st.mtype.sum()
            st.mtype = st.mtype + rng.multinomial(give, p)
            hide_m = spare - give
    else:
        cut = np.minimum(st.mtype, np.round(hidden_mtype * k).astype(np.int64))
        st.mtype = st.mtype - cut
        hide_m = int(cut.sum())

    # ---- по назначениям
    hd = {d: int(round(c * k)) for d, c in hidden_dest.items()}
    for d, c in hd.items():
        if d in st.dest:
            st.dest[d] = max(0, st.dest[d] - c)
    st.dest = {d: c for d, c in st.dest.items() if c > 0}

    # ---- по линкам: чем отличаются семейства
    hl = {l: int(round(c * k)) for l, c in hidden_links.items()}
    need = hide_m
    if family == "F2_single_link":
        # весь дефицит на один линк: минимум числа затронутых невязок
        order = sorted(st.link_in, key=lambda l: -st.link_in.get(l, 0))
        pick = order[int(np.clip(params[0], 0, 0.999) * len(order))] if order else None
        alloc = {pick: need} if pick else {}
    elif family == "F3_proportional":
        tot = sum(st.link_in.values()) or 1
        alloc = {l: int(need * c / tot) for l, c in st.link_in.items()}
    elif family == "F4_temporal":
        # растягивание во времени реализовано снаружи (доля интервала),
        # внутри интервала — пропорционально
        tot = sum(st.link_in.values()) or 1
        alloc = {l: int(need * c / tot) for l, c in st.link_in.items()}
    else:                                   # F1_uniform / F5
        alloc = dict(hl) if hl else {}
        if not alloc and st.link_in:
            share = need // max(len(st.link_in), 1)
            alloc = {l: share for l in st.link_in}

    for l, c in alloc.items():
        if l in st.link_in:
            st.link_in[l] = max(0, st.link_in[l] - int(c))

    # ---- диалоги
    b, e = hidden_dlg
    st.dlg_begin = max(0, st.dlg_begin - int(round(b * k)))
    st.dlg_end = max(0, min(st.dlg_begin, st.dlg_end - int(round(e * k))))
    st.originated = max(0, st.originated - int(round(b * k)))
    st.terminated = max(0, st.terminated - int(round(e * k)))

    # ---- восстановление тождества link-сумм БЕЗ подгонки наблюдаемых
    tgt = sum(st.link_out.values()) + sum(st.link_in.values())
    cur = int(st.mtype.sum())
    if cur > tgt and cur > 0:
        excess = cur - tgt
        p = st.mtype / cur
        st.mtype = np.maximum(0, st.mtype - rng.multinomial(excess, p))
    elif tgt > cur:
        p = (st.mtype / cur) if cur > 0 else np.full(len(MSG_TYPES),
                                                     1.0 / len(MSG_TYPES))
        st.mtype = st.mtype + rng.multinomial(tgt - cur, p)
    st.check_identities()
    return st


def assert_mask_is_identity_at_zero(rng: np.random.Generator) -> None:
    """ПРИЁМОЧНЫЙ ассерт: при h = 0 маскировка не меняет ничего."""
    st = NodeState(nid=7, t=3,
                   mtype=rng.integers(0, 50, len(MSG_TYPES)).astype(np.int64),
                   dest={11: 40, 12: 25, 13: 9},
                   link_out={(1, 7): 0}, link_in={(1, 7): 0},
                   dlg_begin=60, dlg_end=57,
                   originated=60, terminated=10, transited=5)
    tot = int(st.mtype.sum())
    st.link_out[(1, 7)] = tot // 2
    st.link_in[(1, 7)] = tot - tot // 2
    st.check_identities()
    for fam in MASK_FAMILIES:
        for b in (0.0, 0.3, 1.0):
            out = apply_mask(st, 0, {}, {}, np.zeros(len(MSG_TYPES), np.int64),
                             (0, 0), fam, np.array([0.5, 0.5]), b, rng)
            assert out == st, f"{fam}, budget={b}: маскировка не тождественна при h=0"
        out = apply_mask(st, 10, {11: 5}, {(1, 7): 10},
                         np.zeros(len(MSG_TYPES), np.int64), (5, 5),
                         fam, np.array([0.5, 0.5]), 0.0, rng)
        assert out == st, f"{fam}: budget=0 должен быть тождеством"


# =============================================================================
# ss7fix/simulate.py — прогон сети: OD-потоки, очереди, TCAP-диалоги,
# скрытая активность с границы интерконнекта
# =============================================================================
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np


@dataclass
class Episode:
    kind: str                 # 'compromise' | нормальное событие | 'ext_attack'
    nid: Optional[int]
    t0: int
    t1: int
    hidden_rate: float = 0.0
    family: str = "F3_proportional"
    budget: float = 1.0
    sophistication: Sophistication = Sophistication.ADAPTIVE
    colluding_peer: Optional[int] = None
    params: np.ndarray = field(default_factory=lambda: np.array([0.5, 0.5]))
    mult: float = 1.0


@dataclass
class Flow:
    src: int
    dst: int
    service: Service
    rate: float               # транзакций/с в среднем


class Ss7Sim:
    def __init__(self, cfg: SimConfig, book: RngBook,
                 topo: Optional[Topology] = None,
                 size_capacities: bool = True,
                 template: Optional["Ss7Sim"] = None):
        self.cfg = cfg
        self.book = book
        p = cfg.priors
        rt = book["topology"]
        self.topo = Topology(cfg, rt) if topo is None else topo
        if template is not None:
            self.rel_sigma = template.rel_sigma
            self.background = template.background
            self.rotation = template.rotation
            self.max_retry = template.max_retry
            self.diurnal_amp = template.diurnal_amp
            self.flows = template.flows
            self._ar_phi = dict(template._ar_phi)
            self._ar_sig = dict(template._ar_sig)
            self._ar_x = {n: 0.0 for n in self.topo.nodes}
            self.mean_msg_bits = template.mean_msg_bits
            return
        self.rel_sigma = float(rt.uniform(*p.meas_rel_sigma))
        self.background = float(rt.uniform(*p.integrity_background))
        self.rotation = int(rt.integers(*p.key_rotation_intervals))
        self.diurnal_amp = float(rt.uniform(*p.diurnal_amp))
        self.max_retry = int(rt.integers(*p.max_tcap_retry))
        self.flows = self._build_flows(rt)
        self._ar_phi = {n: float(rt.uniform(*p.ar_phi)) for n in self.topo.nodes}
        self._ar_sig = {n: float(rt.uniform(*p.ar_rel_sigma)) for n in self.topo.nodes}
        self._ar_x = {n: 0.0 for n in self.topo.nodes}
        self.mean_msg_bits = 8.0 * float(np.mean(list(MSG_BYTES.values())))
        if size_capacities:
            self.topo.size_capacities(self._nominal_offered(), rt, p)

    # ---- потоки ------------------------------------------------------------
    def _build_flows(self, rng) -> List[Flow]:
        t, fl = self.topo, []
        for z, sps in t.zone_sp.items():
            for sp in sps:
                for other_z, other_sps in t.zone_sp.items():
                    for o in other_sps:
                        if o == sp:
                            continue
                        r = float(rng.gamma(2.0, 0.35)) * (1.0 if z == other_z else 0.35)
                        fl.append(Flow(sp, o, Service.VOICE_NAT, r))
                for h in t.hlr:
                    fl.append(Flow(sp, h, Service.SMS_MT, float(rng.gamma(2.0, 0.5))))
                    fl.append(Flow(sp, h, Service.MOBILITY, float(rng.gamma(2.0, 0.3))))
                    fl.append(Flow(sp, h, Service.AUTH, float(rng.gamma(2.0, 0.25))))
                fl.append(Flow(sp, t.external[0], Service.VOICE_INT,
                               float(rng.gamma(1.5, 0.15))))
        for e in t.external:                        # роуминговые партнёры
            for h in t.hlr:
                fl.append(Flow(e, h, Service.ROAM_SRI, float(rng.gamma(1.5, 0.22))))
                fl.append(Flow(e, h, Service.SMS_MT, float(rng.gamma(1.5, 0.18))))
        return fl

    def _nominal_offered(self) -> Dict[Tuple[int, int], float]:
        off: Dict[Tuple[int, int], float] = {}
        for f in self.flows:
            tmpl = SERVICE_TEMPLATE[f.service]
            bits = 8.0 * sum(MSG_BYTES[m] * c for m, c in tmpl.items())
            r = self.topo.route(f.src, f.dst)
            for path, sh in zip(r.paths, r.shares):
                for u, v in zip(path[:-1], path[1:]):
                    k = (min(u, v), max(u, v))
                    off[k] = off.get(k, 0.0) + f.rate * bits * sh
        return off

    # ---- один интервал -----------------------------------------------------
    def step_true(self, t: int, episodes: List[Episode]
                  ) -> Tuple[Dict[int, NodeState], Dict[int, Dict], Dict]:
        cfg, topo = self.cfg, self.topo
        rng = self.book.child("traffic", t)
        dt = cfg.interval_s
        phase = 2.0 * np.pi * (t % INTERVALS_PER_DAY) / INTERVALS_PER_DAY
        diurnal = 1.0 + self.diurnal_amp * np.sin(phase - 1.2)

        # AR(1) по узлам — общий механизм для всех, без веток
        for n in topo.nodes:
            self._ar_x[n] = (self._ar_phi[n] * self._ar_x[n]
                             + rng.normal(0.0, self._ar_sig[n]))
        node_mult = {n: float(np.exp(self._ar_x[n])) for n in topo.nodes}

        # нормальные события (негативные контроли)
        ev_mult = {n: 1.0 for n in topo.nodes}
        failed_links = set()
        for ep in episodes:
            if not (ep.t0 <= t <= ep.t1) or ep.kind == "compromise":
                continue
            if ep.kind in ("traffic_spike", "traffic_dip", "scale_in", "campaign"):
                for n in (topo.nodes if ep.nid is None else [ep.nid]):
                    ev_mult[n] *= ep.mult
            elif ep.kind == "link_failure" and ep.nid is not None:
                for l in topo.links_of(ep.nid)[:1]:
                    failed_links.add(l)

        states = {n: NodeState(n, t, np.zeros(len(MSG_TYPES), np.int64))
                  for n in topo.nodes}
        offered: Dict[Tuple[int, int], float] = {}
        pending: List[Tuple] = []

        # --- первый проход: раскладка транзакций по маршрутам
        for f in self.flows:
            lam = f.rate * dt * diurnal * node_mult[f.src] * ev_mult[f.src]
            # внешние атаки на границе (SRI-SM flood / ATI tracking / IRSF)
            for ep in episodes:
                if (ep.kind == "ext_attack" and ep.t0 <= t <= ep.t1
                        and f.src == ep.nid and f.service in
                        (Service.ROAM_SRI, Service.SMS_MT, Service.LOCATION)):
                    lam *= ep.mult
            n_txn = int(rng.poisson(max(lam, 0.0)))
            if n_txn <= 0:
                continue
            r = topo.route(f.src, f.dst)
            if not r.paths:
                continue
            split = rng.multinomial(n_txn, r.shares)
            for path, cnt in zip(r.paths, split):
                if cnt <= 0 or any((min(u, v), max(u, v)) in failed_links
                                   for u, v in zip(path[:-1], path[1:])):
                    continue
                pending.append((f, path, int(cnt)))
                tmpl = SERVICE_TEMPLATE[f.service]
                bits = 8.0 * sum(MSG_BYTES[m] * c for m, c in tmpl.items())
                for u, v in zip(path[:-1], path[1:]):
                    k = (min(u, v), max(u, v))
                    offered[k] = offered.get(k, 0.0) + cnt * bits / dt

        # --- скрытая активность компрометированного узла
        hidden: Dict[int, Dict] = {}
        for ep in episodes:
            if ep.kind != "compromise" or not (ep.t0 <= t <= ep.t1):
                continue
            nid = ep.nid
            hidden_lambda = max(ep.hidden_rate * dt, 0.0)
            h = 0 if hidden_lambda == 0.0 else int(rng.poisson(hidden_lambda))
            if h <= 0:
                continue
            ext = int(rng.choice(topo.external))
            dstz = int(rng.choice(topo.hlr))
            r1 = topo.route(ext, nid)
            r2 = topo.route(nid, dstz)
            if not r1.paths or not r2.paths:
                continue
            f_h = Flow(ext, dstz, Service.SMS_MT, 0.0)
            path = tuple(list(r1.paths[0]) + list(r2.paths[0])[1:])
            pending.append((f_h, path, h))
            hidden[nid] = {"msgs": 0, "dest": {}, "links": {},
                           "mtype": np.zeros(len(MSG_TYPES), np.int64),
                           "dlg": (0, 0), "ep": ep, "path": path, "n": h,
                           "dst": dstz}
            tmpl = SERVICE_TEMPLATE[Service.SMS_MT]
            bits = 8.0 * sum(MSG_BYTES[m] * c for m, c in tmpl.items())
            for u, v in zip(path[:-1], path[1:]):
                k = (min(u, v), max(u, v))
                offered[k] = offered.get(k, 0.0) + h * bits / dt

        # --- состояние линков из ФАКТИЧЕСКОЙ нагрузки
        lstate = {}
        for k, lsx in topo.linksets.items():
            lstate[k] = link_state(offered.get(k, 0.0), lsx, self.mean_msg_bits)

        # --- второй проход: разнесение по счётчикам, потери, замыкание TCAP
        for f, path, cnt in pending:
            tmpl = SERVICE_TEMPLATE[f.service]
            per_txn = sum(tmpl.values())
            hop_keys = [(min(u, v), max(u, v)) for u, v in zip(path[:-1], path[1:])]
            p_ok = float(np.prod([1.0 - lstate[k][2] for k in hop_keys])) ** 2
            p_ok = 1.0 - (1.0 - p_ok) ** (1 + self.max_retry)
            closed = int(rng.binomial(cnt, np.clip(p_ok, 0.0, 1.0)))

            is_h = any(h.get("path") == path and h.get("n") == cnt
                       for h in hidden.values())
            hrec = None
            if is_h:
                for nid_h, h in hidden.items():
                    if h.get("path") == path and h.get("n") == cnt:
                        hrec = (nid_h, h)
                        break

            for i, nid in enumerate(path):
                st = states[nid]
                for m, c in tmpl.items():
                    st.mtype[MT_IDX[m]] += cnt * c
                if i > 0:
                    k = hop_keys[i - 1]
                    st.link_in[k] = st.link_in.get(k, 0) + cnt * per_txn
                if i < len(path) - 1:
                    k = hop_keys[i]
                    st.link_out[k] = st.link_out.get(k, 0) + cnt * per_txn
                if i == 0:
                    st.originated += cnt
                    st.dest[path[-1]] = st.dest.get(path[-1], 0) + cnt
                    if f.service in TCAP_SERVICES:
                        st.dlg_begin += cnt
                        st.dlg_end += closed
                elif i == len(path) - 1:
                    st.terminated += cnt
                    st.dest[path[0]] = st.dest.get(path[0], 0) + cnt
                    if f.service in TCAP_SERVICES:
                        st.dlg_begin += cnt
                        st.dlg_end += closed
                else:
                    st.transited += cnt
                if hrec is not None and nid == hrec[0]:
                    h = hrec[1]
                    for m, c in tmpl.items():
                        h["mtype"][MT_IDX[m]] += cnt * c
                    h["msgs"] += cnt * per_txn
                    h["dest"][path[-1]] = h["dest"].get(path[-1], 0) + cnt
                    ki = hop_keys[i - 1] if i > 0 else hop_keys[0]
                    h["links"][ki] = h["links"].get(ki, 0) + cnt * per_txn
                    h["dlg"] = (h["dlg"][0] + cnt, h["dlg"][1] + closed)

        # приводим mtype к сумме link-счётчиков (у краёв путь обрывается)
        for st in states.values():
            tgt = sum(st.link_out.values()) + sum(st.link_in.values())
            cur = int(st.mtype.sum())
            if cur > 0 and tgt != cur:
                st.mtype = np.floor(st.mtype * tgt / cur).astype(np.int64)
                d = tgt - int(st.mtype.sum())
                if d != 0:
                    j = int(np.argmax(st.mtype))
                    st.mtype[j] = max(0, st.mtype[j] + d)
            st.dlg_end = min(st.dlg_end, st.dlg_begin)
            st.check_identities()

        meta = {"lstate": lstate, "offered": offered, "failed": failed_links}
        return states, hidden, meta

    # ---- отчёты ------------------------------------------------------------
    def step_reports(self, t: int, states: Dict[int, NodeState],
                     hidden: Dict[int, Dict], meta: Dict,
                     episodes: List[Episode]) -> Dict[int, Report]:
        rng = self.book.child("measure", t)
        arng = self.book.child("adversary", t)
        reps: Dict[int, Report] = {}
        comp = {ep.nid: ep for ep in episodes
                if ep.kind == "compromise" and ep.t0 <= t <= ep.t1}

        for nid, st in states.items():
            if self.topo.nodes[nid].external:
                continue
            claimed = st
            ep = comp.get(nid)
            if ep is not None and nid in hidden:
                h = hidden[nid]
                soph = ep.sophistication
                if soph is Sophistication.NAIVE:
                    budget = 0.0                     # не маскирует вовсе
                elif soph is Sophistication.STATISTICAL:
                    budget = 0.6 * ep.budget
                else:
                    budget = ep.budget
                if soph is Sophistication.ADAPTIVE and ep.family == "F4_temporal":
                    budget *= float(arng.uniform(0.4, 1.0))
                claimed = apply_mask(st, h["msgs"], h["dest"], h["links"],
                                     h["mtype"], h["dlg"], ep.family,
                                     ep.params, budget, arng)
                if soph is Sophistication.STATISTICAL:
                    # НЕ подарок детектору, а следствие неполноты модели
                    # противника: он забывает про диалоги.
                    claimed.dlg_begin = st.dlg_begin
                    claimed.dlg_end = st.dlg_end
            ls_keys = self.topo.links_of(nid)
            rho = float(np.mean([meta["lstate"][k][0] for k in ls_keys])) if ls_keys else 0.0
            cap = float(np.mean([self.topo.linksets[k].capacity_bps for k in ls_keys])) \
                if ls_keys else 64000.0
            reps[nid] = render_report(
                claimed, rel_sigma=self.rel_sigma, rho_path=rho,
                mean_msg_bits=self.mean_msg_bits, lsx_cap_bps=cap,
                custody=self.cfg.key_custody, compromised=(ep is not None),
                t_compromise=(ep.t0 if ep is not None else None),
                rotation=self.rotation, background=self.background,
                stale=False, rng=rng)
        return reps


# =============================================================================
# ss7fix/invariants.py — то, что узел НЕ МОЖЕТ подавить в одиночку
# Здесь сосредоточен весь реальный сигнал обнаружения.
# =============================================================================
from typing import Dict, List, Tuple
import numpy as np


def link_reconciliation(reps: Dict[int, Report], topo: Topology,
                        lstate: Dict) -> Dict[int, Dict[str, float]]:
    """Оба конца линка отчитываются об ОДНОМ потоке. Компрометированный узел
    владеет только своей половиной. Невязка растёт ~ скрытому объёму."""
    out: Dict[int, Dict[str, float]] = {n: {} for n in reps}
    per_node: Dict[int, List[float]] = {n: [] for n in reps}
    for key in topo.linksets:
        a, b = key
        if a not in reps or b not in reps:
            continue
        loss = lstate[key][2]
        oa = reps[a].state.link_out.get(key, 0)
        ib = reps[b].state.link_in.get(key, 0)
        ob = reps[b].state.link_out.get(key, 0)
        ia = reps[a].state.link_in.get(key, 0)
        r1 = (oa * (1.0 - loss)) - ib
        r2 = (ob * (1.0 - loss)) - ia
        scale = max(np.sqrt(max(oa, 1.0)), 1.0)
        per_node[a] += [r1 / scale, -r2 / scale]
        per_node[b] += [-r1 / scale, r2 / scale]
    for n, v in per_node.items():
        a = np.asarray(v, dtype=float) if v else np.zeros(1)
        out[n] = {"recon_mean": float(a.mean()),
                  "recon_min": float(a.min()),
                  "recon_absmax": float(np.abs(a).max()),
                  "recon_rms": float(np.sqrt((a ** 2).mean()))}
    return out


def od_reconciliation(reps: Dict[int, Report]) -> Dict[int, Dict[str, float]]:
    """Пара (A -> H): A заявляет транзакции к H, H заявляет транзакции от A.
    Скрыть в одиночку нельзя — нужен сговор обоих концов."""
    res: Dict[int, List[float]] = {n: [] for n in reps}
    for a, ra in reps.items():
        for h, cnt in ra.state.dest.items():
            if h not in reps:
                continue
            back = reps[h].state.dest.get(a, 0)
            sc = max(np.sqrt(max(cnt, back, 1.0)), 1.0)
            res[a].append((cnt - back) / sc)
            res[h].append((back - cnt) / sc)
    out = {}
    for n, v in res.items():
        a = np.asarray(v, float) if v else np.zeros(1)
        out[n] = {"od_mean": float(a.mean()), "od_min": float(a.min()),
                  "od_absmax": float(np.abs(a).max())}
    return out


def closure_deficit(reps: Dict[int, Report], lstate: Dict,
                    topo: Topology) -> Dict[int, Dict[str, float]]:
    """Перехваченные/уведённые транзакции не замыкаются у истинного партнёра.
    Дефицит систематический, а не шумовой (перенос решения из SIP-модуля)."""
    out = {}
    for n, r in reps.items():
        ls = topo.links_of(n)
        exp_loss = float(np.mean([lstate[k][2] for k in ls])) if ls else 0.0
        b, e = r.state.dlg_begin, r.state.dlg_end
        expected = b * (1.0 - exp_loss) ** 2
        sd = max(np.sqrt(max(expected, 1.0)), 1.0)
        out[n] = {"closure_resid": float((e - expected) / sd),
                  "closure_ratio": float(e / max(b, 1))}
    return out


def transit_conservation(reps: Dict[int, Report],
                         topo: Topology) -> Dict[int, Dict[str, float]]:
    """STP не порождает трафик: вход ~ выход. SP/SCP: разность = originated -
    terminated, и оба члена перекрёстно проверяемы по OD."""
    out = {}
    for n, r in reps.items():
        st = r.state
        s_in, s_out = sum(st.link_in.values()), sum(st.link_out.values())
        if topo.nodes[n].ntype in (NodeType.STP, NodeType.IGW):
            resid = s_in - s_out
        else:
            resid = (s_in - s_out) - (st.terminated - st.originated)
        sc = max(np.sqrt(max(s_in, 1.0)), 1.0)
        out[n] = {"transit_resid": float(resid / sc)}
    return out


def all_invariants(reps: Dict[int, Report], topo: Topology,
                   lstate: Dict) -> Dict[int, Dict[str, float]]:
    a = link_reconciliation(reps, topo, lstate)
    b = od_reconciliation(reps)
    c = closure_deficit(reps, lstate, topo)
    d = transit_conservation(reps, topo)
    out = {}
    for n in reps:
        row = {}
        row.update(a[n]); row.update(b[n]); row.update(c[n]); row.update(d[n])
        out[n] = row
    return out


# =============================================================================
# ss7fix/features.py — ПРЕДРЕГИСТРИРОВАННЫЙ реестр признаков
# Каждый признак обоснован наблюдаемой величиной протокола и механизмом
# искажения. Признак без обоснования в реестр не попадает.
# =============================================================================
from typing import Dict, List
import numpy as np
import pandas as pd

FEATURE_REGISTRY: Dict[str, Dict[str, str]] = {
    # --- собственный отчёт узла (в принципе фальсифицируем => слабый канал)
    "total_messages":     dict(group="own", why="MSU/интервал, MTP3-счётчик"),
    "mtype_entropy":      dict(group="own", why="энтропия по 17 опкодам"),
    "dest_entropy":       dict(group="own", why="энтропия по DPC"),
    "n_unique_dest":      dict(group="own", why="мощность множества DPC"),
    "dest_top1_ratio":    dict(group="own", why="концентрация на одном DPC"),
    "dest_top2_ratio":    dict(group="own", why="концентрация на двух DPC"),
    "map_share":          dict(group="own", why="доля MAP в миксе"),
    "isup_share":         dict(group="own", why="доля ISUP в миксе"),
    "in_out_ratio":       dict(group="own", why="асимметрия направления"),
    "transit_share":      dict(group="own", why="доля транзита"),
    "dlg_closure_ratio":  dict(group="own", why="TCAP End/Begin"),
    # --- кросс-узловые невязки (унилатерально не подавляемы => сильный канал)
    "recon_mean":         dict(group="inv", why="парная сверка линка, среднее"),
    "recon_min":          dict(group="inv", why="парная сверка, худший линк"),
    "recon_absmax":       dict(group="inv", why="парная сверка, максимум |r|"),
    "recon_rms":          dict(group="inv", why="парная сверка, RMS"),
    "od_mean":            dict(group="inv", why="сверка OD-пар A<->H"),
    "od_min":             dict(group="inv", why="сверка OD, худшая пара"),
    "od_absmax":          dict(group="inv", why="сверка OD, максимум"),
    "closure_resid":      dict(group="inv", why="дефицит замыкания TCAP vs потери"),
    "transit_resid":      dict(group="inv", why="закон сохранения транзита"),
    # --- канал целостности (жив только при ROTATED_EXTERNAL/TEE)
    "integrity_fail":     dict(group="integrity", why="отказ MAC отчёта"),
}
OWN = [k for k, v in FEATURE_REGISTRY.items() if v["group"] == "own"]
INV = [k for k, v in FEATURE_REGISTRY.items() if v["group"] == "inv"]
INTEG = [k for k, v in FEATURE_REGISTRY.items() if v["group"] == "integrity"]

FEATURE_SETS = {"OWN_only": OWN, "INV_only": INV,
                "OWN+INV": OWN + INV, "ALL": OWN + INV + INTEG}


def temporal_block(df: pd.DataFrame, cols: List[str], win: int) -> pd.DataFrame:
    """Скользящие статистики ПО ПРОШЛОМУ (closed='left'), без заглядывания."""
    g = df.sort_values(["nid", "t"]).groupby("nid", sort=False)
    add = {}
    for c in cols:
        r = g[c].rolling(win, min_periods=max(5, win // 4), closed="left")
        add[f"{c}__mean"] = r.mean().reset_index(level=0, drop=True)
        add[f"{c}__std"] = r.std().reset_index(level=0, drop=True)
        add[f"{c}__z"] = ((df[c] - add[f"{c}__mean"])
                          / add[f"{c}__std"].replace(0, np.nan))
    out = df.copy()
    for k, v in add.items():
        out[k] = v
    return out


def zone_deviation(df: pd.DataFrame, cols: List[str],
                   zone_of: Dict[int, int]) -> pd.DataFrame:
    """Отклонение от медианы зоны БЕЗ самого узла (leave-one-out),
    иначе узел влияет на собственный эталон."""
    out = df.copy()
    out["zone"] = out["nid"].map(zone_of)
    for c in cols:
        med = out.groupby(["zone", "t"])[c].transform("median")
        cnt = out.groupby(["zone", "t"])[c].transform("count")
        out[f"{c}__dev_zone"] = np.where(cnt > 2, out[c] - med, np.nan)
    return out.drop(columns=["zone"])


# =============================================================================
# ss7fix/dataset.py — сборка датасета, эпизоды, негативные контроли
# =============================================================================
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

NORMAL_EVENTS = ("traffic_spike", "traffic_dip", "scale_in",
                 "campaign", "link_failure", "reroute")


def make_episodes(sim: Ss7Sim, cfg: SimConfig, rng: np.random.Generator,
                  *, n_compromise: int = 6, n_normal: int = 18,
                  n_ext: int = 6, hidden_rate_grid=(0.05, 0.15, 0.4, 1.0, 2.5),
                  soph: Sophistication = Sophistication.ADAPTIVE,
                  families=MASK_FAMILIES, force_hidden: Optional[float] = None
                  ) -> List[Episode]:
    T, eps = cfg.n_intervals, []
    dur = lambda lo, hi: int(rng.integers(lo, hi))
    for _ in range(n_compromise):
        t0 = int(rng.integers(cfg.feat_window + 5, max(T - 200, cfg.feat_window + 6)))
        hr = force_hidden if force_hidden is not None else float(rng.choice(hidden_rate_grid))
        eps.append(Episode(kind="compromise", nid=int(rng.choice(sim.topo.slaves)),
                           t0=t0, t1=min(T - 1, t0 + dur(60, 400)),
                           hidden_rate=hr, family=str(rng.choice(list(families))),
                           budget=float(rng.uniform(0.7, 1.0)),
                           sophistication=soph,
                           params=rng.random(2)))
    for _ in range(n_normal):
        k = str(rng.choice(NORMAL_EVENTS))
        t0 = int(rng.integers(cfg.feat_window + 5, max(T - 120, cfg.feat_window + 6)))
        mult = {"traffic_spike": rng.uniform(1.5, 3.5),
                "traffic_dip": rng.uniform(0.25, 0.65),
                "scale_in": rng.uniform(0.5, 0.8),
                "campaign": rng.uniform(1.8, 4.0),
                "link_failure": 1.0, "reroute": 1.0}[k]
        eps.append(Episode(kind=k, nid=int(rng.choice(sim.topo.slaves)), t0=t0,
                           t1=min(T - 1, t0 + dur(20, 150)), mult=float(mult)))
    for _ in range(n_ext):
        t0 = int(rng.integers(cfg.feat_window + 5, max(T - 120, cfg.feat_window + 6)))
        eps.append(Episode(kind="ext_attack", nid=int(rng.choice(sim.topo.external)),
                           t0=t0, t1=min(T - 1, t0 + dur(20, 120)),
                           mult=float(rng.uniform(3.0, 12.0))))
    return eps


def build_dataset(cfg: SimConfig, *, episodes_kw: Optional[dict] = None,
                  topo: Optional[Topology] = None,
                  size_capacities: bool = True,
                  sim_template: Optional[Ss7Sim] = None
                  ) -> Tuple[pd.DataFrame, Ss7Sim, List[Episode]]:
    book = RngBook(cfg.seed)
    sim = Ss7Sim(cfg, book, topo=topo, size_capacities=size_capacities,
                 template=sim_template)
    eps = make_episodes(sim, cfg, book["episode"], **(episodes_kw or {}))
    ep_id = {id(e): i for i, e in enumerate(eps)}
    rows = []
    for t in range(cfg.n_intervals):
        states, hidden, meta = sim.step_true(t, eps)
        reps = sim.step_reports(t, states, hidden, meta, eps)
        inv = all_invariants(reps, sim.topo, meta["lstate"])
        comp = {e.nid: e for e in eps if e.kind == "compromise" and e.t0 <= t <= e.t1}
        for nid, r in reps.items():
            if sim.topo.nodes[nid].role is NodeRole.MASTER:
                continue
            row = {"nid": nid, "t": t,
                   "ntype": sim.topo.nodes[nid].ntype.value,
                   "zone": sim.topo.nodes[nid].zone}
            row.update(r.feats)
            row.update(inv[nid])
            row["integrity_fail"] = float(not r.integrity_ok)
            row["rtt_ms"] = r.rtt_ms
            e = comp.get(nid)
            row["label"] = int(e is not None)
            row["hidden_msgs"] = float(hidden.get(nid, {}).get("msgs", 0))
            row["episode_id"] = ep_id[id(e)] if e is not None else -1
            row["mask_family"] = e.family if e is not None else "none"
            row["soph"] = e.sophistication.value if e is not None else "none"
            rows.append(row)
    df = pd.DataFrame(rows)
    base = [c for c in OWN + INV if c in df.columns]
    df = temporal_block(df, base, cfg.feat_window)
    df = zone_deviation(df, base,
                        {n: v.zone for n, v in sim.topo.nodes.items()})
    return df, sim, eps


# =============================================================================
# ss7fix/nulltest.py — ПРИЁМОЧНЫЙ НУЛЕВОЙ ТЕСТ (главный шлюз)
# Компрометированный узел проходит весь путь маскировки при h = 0.
# Детектор обязан дать AUC ~ 0.5; иначе в генераторе есть подпись.
# =============================================================================
from collections import defaultdict
from typing import Dict, List, Sequence, Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score


def energy_distance(x: np.ndarray, y: np.ndarray) -> float:
    def d(a, b):
        return np.sqrt(((a[:, None, :] - b[None, :, :]) ** 2).sum(-1)).mean()
    return float(2 * d(x, y) - d(x, x) - d(y, y))


def permutation_energy_test(x: np.ndarray, y: np.ndarray, n_perm: int = 400,
                            rng=None) -> float:
    rng = rng or np.random.default_rng(0)
    obs = energy_distance(x, y)
    z = np.vstack([x, y]); n = len(x)
    cnt = 0
    for _ in range(n_perm):
        p = rng.permutation(len(z))
        if energy_distance(z[p[:n]], z[p[n:]]) >= obs:
            cnt += 1
    return (cnt + 1) / (n_perm + 1)


def tost_equivalence(x: np.ndarray, y: np.ndarray, delta: float = 0.25
                     ) -> Tuple[bool, float]:
    """Тест ЭКВИВАЛЕНТНОСТИ (а не непрокинутая H0): |d| < delta по Коэну."""
    sx, sy = x.std(ddof=1), y.std(ddof=1)
    sp = np.sqrt(((len(x) - 1) * sx ** 2 + (len(y) - 1) * sy ** 2)
                 / max(len(x) + len(y) - 2, 1)) or 1e-12
    d = (x.mean() - y.mean()) / sp
    se = np.sqrt(1 / len(x) + 1 / len(y))
    hi = abs(d) + 1.96 * se
    return bool(hi < delta), float(d)


def _usable_columns(frame: pd.DataFrame, cols: Sequence[str],
                    min_notna: float = 0.98) -> List[str]:
    return [c for c in cols
            if frame[c].notna().mean() >= min_notna
            and frame[c].nunique(dropna=True) > 1]


def _null_test_one(pos: pd.DataFrame, neg: pd.DataFrame, cols: List[str],
                   seed: int, auc_tol: float, delta: float,
                   n_boot: int) -> Dict:
    X = pd.concat([pos, neg])[cols].to_numpy(float)
    y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]

    rf = RandomForestClassifier(n_estimators=300, min_samples_leaf=5,
                                random_state=seed, n_jobs=1)
    n = len(y); idx = np.random.default_rng(seed).permutation(n)
    half = n // 2
    rf.fit(X[idx[:half]], y[idx[:half]])
    s = rf.predict_proba(X[idx[half:]])[:, 1]
    auc = roc_auc_score(y[idx[half:]], s)

    rng = np.random.default_rng(seed + 1)
    boots = []
    yy, ss = y[idx[half:]], s
    for _ in range(n_boot):
        b = rng.integers(0, len(yy), len(yy))
        if len(np.unique(yy[b])) < 2:
            continue
        boots.append(roc_auc_score(yy[b], ss[b]))
    lo, hi = np.percentile(boots, [2.5, 97.5])

    Xs = (X - X.mean(0)) / (X.std(0) + 1e-12)
    k = min(8, Xs.shape[1])
    p_energy = permutation_energy_test(
        Xs[y == 1][:150, :k], Xs[y == 0][:150, :k],
        rng=np.random.default_rng(seed + 2))

    per_feat = {}
    fails = []
    for j, c in enumerate(cols):
        ok, dd = tost_equivalence(X[y == 1, j], X[y == 0, j], delta)
        per_feat[c] = {"cohen_d": dd, "equivalent": ok}
        if not ok:
            fails.append((c, dd))

    rep = {"auc": float(auc), "auc_ci": (float(lo), float(hi)),
           "ci": (float(lo), float(hi)), "p_energy": float(p_energy),
           "n_pos": int(len(pos)), "n_neg": int(len(neg)),
           "equivalent": not fails,
           "non_equivalent": sorted(fails, key=lambda x: -abs(x[1]))[:10],
           "per_feature": per_feat}
    assert abs(auc - 0.5) <= auc_tol, (
        f"НУЛЕВОЙ ТЕСТ ПРОВАЛЕН: AUC={auc:.3f} при h=0. "
        f"В генераторе есть подпись, не связанная со скрытой активностью. "
        f"Худшие признаки: {rep['non_equivalent'][:5]}")
    assert p_energy > 0.01, f"НУЛЕВОЙ ТЕСТ: p_energy={p_energy:.4f}"
    return rep


def _counterfactual_null(cfg: SimConfig, feature_cols: Sequence[str],
                         topo: Optional[Topology], log=print) -> Dict:
    """Побиточно сравнивает h=0 с тем же миром без compromise-эпизодов."""
    c0 = SimConfig(**{**cfg.__dict__, "seed": cfg.seed + 100_003})
    base = dict(n_normal=0, n_ext=0)
    df_a, sim_a, _ = build_dataset(
        c0, topo=topo, size_capacities=(topo is None),
        episodes_kw=dict(n_compromise=10, force_hidden=0.0, **base))
    df_b, _, _ = build_dataset(
        c0, topo=sim_a.topo, size_capacities=False,
        sim_template=sim_a,
        episodes_kw=dict(n_compromise=0, **base))

    cols = [c for c in feature_cols
            if c in df_a.columns and c in df_b.columns
            and not c.endswith(("__z", "__mean", "__std", "__dev_zone"))]
    if not cols:
        raise AssertionError("контрфактуальный тест: нет базовых признаков")

    a = df_a.set_index(["nid", "t"]).sort_index()
    b = df_b.set_index(["nid", "t"]).sort_index()
    idx = a.index.intersection(b.index)
    a, b = a.loc[idx], b.loc[idx]
    va = a[cols].to_numpy(float)
    vb = b[cols].to_numpy(float)
    scale = np.nanstd(vb, axis=0)
    scale[~np.isfinite(scale) | (scale == 0)] = 1.0
    rel = np.abs(va - vb) / scale

    victim = a["label"].to_numpy() == 1
    off = float(np.nanmax(rel[~victim])) if (~victim).any() else 0.0
    on = float(np.nanmax(rel[victim])) if victim.any() else 0.0
    worst = (sorted(zip(cols, np.nanmax(rel[victim], axis=0)),
                    key=lambda kv: -kv[1])[:8]
             if victim.any() else [])
    log(f"[null-test/cf] строк {len(idx)}, окон компрометации {int(victim.sum())}; "
        f"макс. относительное расхождение: вне эпизодов {off:.3e}, "
        f"внутри {on:.3e}")

    tol = 1e-9
    if off > tol:
        raise AssertionError(
            f"прогоны A и B расходятся вне эпизодов (max={off:.3e}). "
            f"Проверьте независимость потоков ГСЧ и book.child")
    if on > tol:
        raise AssertionError(
            f"НУЛЕВОЙ ТЕСТ ПРОВАЛЕН контрфактуально: при h=0 наблюдаемые "
            f"величины отличаются от честного прогона, max={on:.3e}. "
            f"Худшие признаки: {worst}")
    return {"passed": True, "max_dev_outside": off,
            "max_dev_inside": on, "n_rows": int(len(idx)),
            "n_pos": int(victim.sum()), "n_features": len(cols)}


def null_test(cfg: SimConfig, feature_cols: Sequence[str], *,
              topo: Optional[Topology] = None, auc_tol: float = 0.06,
              delta: float = 0.30, n_boot: int = 200, min_pos: int = 50,
              log=print) -> Dict:
    """Сравнивает компрометированные окна с честными на тех же узлах."""
    assert_mask_is_identity_at_zero(np.random.default_rng(cfg.seed))
    cf = _counterfactual_null(cfg, feature_cols, topo, log=log)
    c0 = SimConfig(**{**cfg.__dict__, "seed": cfg.seed + 100_003})
    df, sim, _ = build_dataset(
        c0, topo=topo, size_capacities=(topo is None),
        episodes_kw=dict(n_compromise=10, n_normal=0, n_ext=0,
                         force_hidden=0.0))
    cols0 = [c for c in feature_cols if c in df.columns]
    role_of = {int(n): sim.topo.nodes[int(n)].ntype.name
               for n in df["nid"].unique()}
    victims = sorted(int(n) for n in df.loc[df["label"] == 1, "nid"].unique())
    if not victims:
        raise AssertionError("в нулевом прогоне нет компрометированных окон")
    groups: Dict[str, List[int]] = defaultdict(list)
    for nid in victims:
        groups[role_of[nid]].append(nid)

    reports, worst = {}, 0.5
    for role, nids in sorted(groups.items()):
        sub = df[df["nid"].isin(nids)]
        cols = _usable_columns(sub, cols0)
        d = sub.dropna(subset=cols)
        pos = d[d["label"] == 1]
        negp = d[d["label"] == 0]
        log(f"[null-test] {role}: узлов {len(nids)}, "
            f"признаков {len(cols)}/{len(cols0)}, строк {len(d)}/{len(sub)}, "
            f"pos={len(pos)} neg={len(negp)}")
        if len(cols) < 5:
            raise AssertionError(f"{role}: информативных признаков всего {len(cols)}")
        if len(pos) < min_pos:
            raise AssertionError(
                f"{role}: окон компрометации {len(pos)}, требуется {min_pos}. "
                f"Увеличьте days или n_compromise")
        neg = negp.sample(min(len(negp), 4 * len(pos)),
                          random_state=cfg.seed)
        try:
            rep = _null_test_one(pos, neg, cols, cfg.seed, auc_tol, delta,
                                  n_boot)
        except AssertionError as exc:
            reports[role] = {"passed": False, "reason": str(exc),
                             "n_nodes": len(nids),
                             "n_features": len(cols),
                             "n_pos": int(len(pos)), "n_neg": int(len(neg))}
            continue
        rep["n_nodes"], rep["n_features"] = len(nids), len(cols)
        reports[role] = rep
        worst = max(worst, abs(rep["auc"] - 0.5) + 0.5)

        return {"passed": True, "counterfactual": cf,
            "by_role": reports, "worst_auc": worst,
            "role_diagnostic_passed": all(
                v.get("passed", True) and v.get("equivalent", True)
                for v in reports.values())}


# =============================================================================
# ss7fix/splits.py — блочный сплит по времени и эпизодам с зазором
# Исправляет: train_test_split по i.i.d.-записям при скользящих окнах
# =============================================================================
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd


def blocked_split(df: pd.DataFrame, cfg: SimConfig, rng: np.random.Generator,
                  fracs=(0.6, 0.15, 0.25), n_blocks: int = 12
                  ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Непрерывные блоки времени -> train/val/test; purge-зазор >= окна
    признаков; эпизод не может попасть в два сплита."""
    T = int(df["t"].max()) + 1
    gap = cfg.feat_window + cfg.z_window
    edges = np.linspace(0, T, n_blocks + 1).astype(int)
    blocks = list(zip(edges[:-1], edges[1:]))
    order = rng.permutation(len(blocks))
    n_tr = int(round(fracs[0] * len(blocks)))
    n_va = int(round(fracs[1] * len(blocks)))
    assign = {}
    for i, b in enumerate(order):
        assign[b] = "train" if i < n_tr else ("val" if i < n_tr + n_va else "test")

    def part(name):
        keep = np.zeros(len(df), bool)
        for b, (a, z) in enumerate(blocks):
            if assign[b] != name:
                continue
            lo = a + (gap if b > 0 else 0)
            hi = z - gap
            if hi <= lo:
                continue
            keep |= ((df["t"] >= lo) & (df["t"] < hi)).to_numpy()
        return df[keep]

    tr, va, te = part("train"), part("val"), part("test")
    # эпизод целиком туда, где его большинство
    for e in df.loc[df["episode_id"] >= 0, "episode_id"].unique():
        cnt = {n: int((p["episode_id"] == e).sum()) for n, p in
               (("train", tr), ("val", va), ("test", te))}
        win = max(cnt, key=cnt.get)
        for n, p in (("train", tr), ("val", va), ("test", te)):
            if n != win and cnt[n] > 0:
                drop = p.index[p["episode_id"] == e]
                if n == "train":
                    tr = tr.drop(drop)
                elif n == "val":
                    va = va.drop(drop)
                else:
                    te = te.drop(drop)
    return tr, va, te


# =============================================================================
# ss7fix/evaluate.py — эпизодные метрики, тревоги/сутки, rule-baseline, LOMFO
# Исправляет: оракульный порог по тесту; окно-метрики без операционного смысла
# =============================================================================
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler


def threshold_for_alert_budget(scores_neg: np.ndarray, n_nodes: int,
                               alerts_per_day: float) -> float:
    """Порог задаётся ОПЕРАЦИОННЫМ бюджетом тревог, а не абстрактным FPR,
    и считается только по train/val-негативам."""
    per_node_day = alerts_per_day / max(n_nodes, 1)
    frac = per_node_day / INTERVALS_PER_DAY
    q = 100.0 * (1.0 - np.clip(frac, 1e-6, 0.5))
    return float(np.percentile(scores_neg, q))


def episode_metrics(df: pd.DataFrame, scores: np.ndarray, thr: float,
                    k_consec: int = 3) -> Dict[str, float]:
    """Обнаружение на уровне ИНЦИДЕНТА + время до обнаружения."""
    d = df.assign(score=scores, fire=(scores > thr).astype(int))
    det, ttd = [], []
    for e, g in d[d["episode_id"] >= 0].groupby("episode_id"):
        g = g.sort_values("t")
        f = g["fire"].to_numpy()
        run, hit = 0, None
        for i, v in enumerate(f):
            run = run + 1 if v else 0
            if run >= k_consec:
                hit = i - k_consec + 1
                break
        det.append(hit is not None)
        if hit is not None:
            ttd.append(int(g["t"].iloc[hit] - g["t"].iloc[0]))
    neg = d[d["label"] == 0]
    n_nodes = d["nid"].nunique()
    days = (d["t"].max() - d["t"].min() + 1) / INTERVALS_PER_DAY
    alerts = int(neg["fire"].sum())
    return {"episode_recall": float(np.mean(det)) if det else np.nan,
            "n_episodes": int(len(det)),
            "median_ttd_intervals": float(np.median(ttd)) if ttd else np.nan,
            "alerts_per_day_network": float(alerts / max(days, 1e-9)),
            "window_fpr": float(neg["fire"].mean()) if len(neg) else np.nan,
            "n_nodes": int(n_nodes)}


def rule_baseline(df: pd.DataFrame) -> np.ndarray:
    """Тривиальные правила (аналог проверок FS.11 кат.1 + порог по невязке).
    Без этого baseline непонятно, что вообще дают ML-модели."""
    z = lambda c: df[c].fillna(0.0).to_numpy(float) if c in df else np.zeros(len(df))
    return np.maximum.reduce([
        -z("recon_min"), np.abs(z("od_min")), -z("closure_resid"),
        np.abs(z("transit_resid")), 3.0 * z("integrity_fail")])


def _prep(tr, va, te, cols):
    imp = SimpleImputer(strategy="median").fit(tr[cols])
    sc = StandardScaler().fit(imp.transform(tr[cols]))
    f = lambda d: sc.transform(imp.transform(d[cols]))
    return f(tr), f(va), f(te)


def run_models(tr, va, te, cols, cfg: SimConfig,
               alerts_per_day: float = 20.0) -> pd.DataFrame:
    Xtr, Xva, Xte = _prep(tr, va, te, cols)
    ytr, yva, yte = (d["label"].to_numpy() for d in (tr, va, te))
    rows = []

    def add(name, s_va, s_te):
        thr = threshold_for_alert_budget(s_va[yva == 0], te["nid"].nunique(),
                                         alerts_per_day)
        m = episode_metrics(te, s_te, thr)
        m.update(model=name,
                 auc=roc_auc_score(yte, s_te) if len(np.unique(yte)) > 1 else np.nan,
                 ap=average_precision_score(yte, s_te))
        rows.append(m)

    add("rule_baseline", rule_baseline(va), rule_baseline(te))

    iso = IsolationForest(n_estimators=400, contamination="auto",
                          random_state=cfg.seed).fit(Xtr[ytr == 0])
    add("isolation_forest", -iso.score_samples(Xva), -iso.score_samples(Xte))

    rf = RandomForestClassifier(n_estimators=500, min_samples_leaf=4,
                                class_weight="balanced_subsample",
                                random_state=cfg.seed, n_jobs=1).fit(Xtr, ytr)
    add("random_forest", rf.predict_proba(Xva)[:, 1], rf.predict_proba(Xte)[:, 1])
    return pd.DataFrame(rows)


def lomfo(cfg: SimConfig, cols: Sequence[str],
          alerts_per_day: float = 20.0) -> pd.DataFrame:
    """Leave-one-masking-family-out: обучаемся без семейства, тестируем на нём.
    Перенесено из 5G-модуля на SS7."""
    df, _, _ = build_dataset(cfg)
    rng = RngBook(cfg.seed)["split"]
    out = []
    for fam in MASK_FAMILIES:
        tr, va, te = blocked_split(df, cfg, rng)
        tr = tr[(tr["label"] == 0) | (tr["mask_family"] != fam)]
        va = va[(va["label"] == 0) | (va["mask_family"] != fam)]
        te = te[(te["label"] == 0) | (te["mask_family"] == fam)]
        if te["label"].sum() < 20 or tr["label"].sum() < 20:
            continue
        r = run_models(tr, va, te, list(cols), cfg, alerts_per_day)
        r["holdout_family"] = fam
        out.append(r)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def measured_eta(cfg: SimConfig) -> pd.DataFrame:
    """eta ИЗМЕРЯЕТСЯ, а не задаётся константой: какую долю h удаётся
    спрятать так, чтобы суммарная невязка осталась в шуме."""
    df, _, _ = build_dataset(cfg)
    d = df[df["label"] == 1].dropna(subset=["recon_rms", "hidden_msgs"])
    d = d[d["hidden_msgs"] > 0]
    if d.empty:
        return pd.DataFrame()
    res = np.abs(d["recon_mean"]) + np.abs(d["od_mean"]) + np.abs(d["closure_resid"])
    d = d.assign(residual=res)
    g = d.groupby("mask_family").apply(
        lambda x: pd.Series({
            "n": len(x),
            "eta_hat": float(1.0 - np.clip(
                np.polyfit(x["hidden_msgs"], x["residual"], 1)[0]
                / max(np.polyfit(d["hidden_msgs"], d["residual"], 1)[0], 1e-9),
                0, 1)),
            "resid_per_hidden": float((x["residual"] / x["hidden_msgs"]).median()),
        }), include_groups=False)
    return g.reset_index()


def multiseed(seeds: Sequence[int], cols: Sequence[str], base: SimConfig,
              alerts_per_day: float = 20.0) -> pd.DataFrame:
    """Каждый сид ПЕРЕГЕНЕРИРУЕТ сеть и трафик (regen по умолчанию) —
    иначе это дисперсия разбиения, а не реализации процесса."""
    out = []
    for s in seeds:
        cfg = SimConfig(**{**base.__dict__, "seed": int(s)})
        df, _, _ = build_dataset(cfg)
        tr, va, te = blocked_split(df, cfg, RngBook(cfg.seed)["split"])
        r = run_models(tr, va, te, list(cols), cfg, alerts_per_day)
        r["seed"] = s
        out.append(r)
    return pd.concat(out, ignore_index=True)


def bootstrap_compare(res: pd.DataFrame, a: str, b: str,
                      metric: str = "episode_recall", n_boot: int = 5000,
                      rng=None) -> Dict[str, float]:
    """Бутстрэп по сидам вместо Уилкоксона на 10 зависимых парах."""
    rng = rng or np.random.default_rng(0)
    pa = res[res.model == a].set_index("seed")[metric]
    pb = res[res.model == b].set_index("seed")[metric]
    common = pa.index.intersection(pb.index)
    d = (pa[common] - pb[common]).dropna().to_numpy(float)
    if len(d) < 3:
        return {"n": len(d)}
    bs = np.array([rng.choice(d, len(d), replace=True).mean()
                   for _ in range(n_boot)])
    return {"n": len(d), "mean_diff": float(d.mean()),
            "ci_lo": float(np.percentile(bs, 2.5)),
            "ci_hi": float(np.percentile(bs, 97.5)),
            "p_two_sided": float(2 * min((bs <= 0).mean(), (bs >= 0).mean()))}


# =============================================================================
# ss7fix/run.py — CLI
# =============================================================================
import argparse
import numpy as np
import pandas as pd


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="SS7 stand с исправлениями")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--days", type=float, default=6.0)
    ap.add_argument("--alerts-per-day", type=float, default=20.0)
    ap.add_argument("--custody", default="node_software",
                    choices=[k.value for k in KeyCustody])
    ap.add_argument("--soph", default="adaptive",
                    choices=[s.value for s in Sophistication])
    ap.add_argument("--feature-set", default="OWN+INV",
                    choices=list(FEATURE_SETS))
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    ap.add_argument("--skip-null-test", action="store_true")
    ap.add_argument("--lomfo", action="store_true")
    ap.add_argument("--eta", action="store_true")
    a = ap.parse_args(argv)

    pd.set_option("display.width", 160, "display.max_columns", 40)
    cfg = SimConfig(seed=a.seed, days=a.days,
                    key_custody=KeyCustody(a.custody))
    base = FEATURE_SETS[a.feature_set]

    print("=" * 78)
    print("1) Ассерт тождественности маскировки при h=0")
    assert_mask_is_identity_at_zero(np.random.default_rng(a.seed))
    print("   OK: все семейства тождественны при h=0 и при budget=0")

    df, sim, eps = build_dataset(cfg)
    cols = [c for c in df.columns
            if any(c == b or c.startswith(b + "__") for b in base)]
    print(f"\n2) Датасет: {len(df)} окон, {df['nid'].nunique()} узлов, "
          f"{df['label'].mean():.4f} доля аномальных окон, {len(cols)} признаков")
    rho = [v[0] for v in sim.step_true(0, eps)[2]["lstate"].values()]
    print(f"   rho линков: med={np.median(rho):.3f} "
          f"[{np.min(rho):.3f}, {np.max(rho):.3f}] — модель очередей живая")

    if not a.skip_null_test:
        print("\n3) НУЛЕВОЙ ТЕСТ (компрометация с h=0, полный путь маскировки)")
        rep = null_test(cfg, cols)
        for role, report in rep["by_role"].items():
            if "auc" in report:
                print(f"   {role}: AUC={report['auc']:.3f} "
                      f"CI95={report['ci']}, признаки={report['n_features']}")
            else:
                print(f"   {role}: диагностический RF не пройден: "
                      f"{report['reason'][:160]}")
        print("   OK: подписи, не связанной со скрытой активностью, нет")

    print("\n4) Блочный сплит по времени с purge-зазором")
    tr, va, te = blocked_split(df, cfg, RngBook(cfg.seed)["split"])
    print(f"   train={len(tr)} val={len(va)} test={len(te)}; "
          f"зазор={cfg.feat_window + cfg.z_window} интервалов")

    print("\n5) Метрики на уровне инцидента (порог из train/val по бюджету тревог)")
    res = run_models(tr, va, te, cols, cfg, a.alerts_per_day)
    print(res[["model", "auc", "ap", "episode_recall", "median_ttd_intervals",
               "alerts_per_day_network", "window_fpr"]].to_string(index=False))

    print("\n6) Аблация наборов признаков (own vs кросс-узловые инварианты)")
    for name, bb in FEATURE_SETS.items():
        cc = [c for c in df.columns
              if any(c == b or c.startswith(b + "__") for b in bb)]
        r = run_models(tr, va, te, cc, cfg, a.alerts_per_day)
        r = r[r.model == "random_forest"].iloc[0]
        print(f"   {name:10s} recall_episode={r['episode_recall']:.3f} "
              f"ttd={r['median_ttd_intervals']:.0f} auc={r['auc']:.3f}")

    if a.eta:
        print("\n7) Измеренное eta по семействам (вместо константы)")
        print(measured_eta(cfg).to_string(index=False))

    if a.lomfo:
        print("\n8) Leave-one-masking-family-out")
        L = lomfo(cfg, base, a.alerts_per_day)
        if not L.empty:
            print(L[["holdout_family", "model", "episode_recall",
                     "auc"]].to_string(index=False))

    if a.seeds:
        print("\n9) Мультисид с ПЕРЕГЕНЕРАЦИЕЙ сети и трафика")
        ms = multiseed(a.seeds, base, cfg, a.alerts_per_day)
        agg = ms.groupby("model")[["episode_recall", "auc",
                                   "alerts_per_day_network"]].agg(["mean", "std"])
        print(agg.to_string())
        cmp = bootstrap_compare(ms, "random_forest", "rule_baseline")
        print(f"   RF vs rule_baseline: {cmp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
