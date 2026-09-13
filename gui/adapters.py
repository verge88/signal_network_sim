"""
gui/adapters.py — мост между GUI-моделью и симуляторами репозитория.

Идея: не менять код симуляторов. Мы создаём объект их класса топологии
через __new__ (минуя процедурный _build), заполняем nodes/links/graph
из нарисованной сети и подменяем класс в модуле фабрикой, возвращающей
этот объект. Дальше вызывается штатный main() модуля.
"""
from __future__ import annotations

import importlib
import os
import sys
from typing import Callable, Optional

from .model import PROTOCOLS, Topology


def _enum(enum_cls, name: str):
    try:
        return enum_cls[name]
    except KeyError:
        for member in enum_cls:
            if str(getattr(member, "value", "")).lower() == name.lower():
                return member
        raise ValueError(f"{enum_cls.__name__}: нет варианта {name!r}")


def build_sim_topology(topo: Topology, module, config=None):
    """Собирает объект *NetworkTopology симулятора из GUI-топологии."""
    import networkx as nx

    prof = PROTOCOLS[topo.protocol]
    TopoCls = getattr(module, prof["topology_class"])
    NodeCls = getattr(module, prof["node_class"])
    LinkCls = getattr(module, prof["link_class"])
    NodeType = getattr(module, "NodeType")
    NodeRole = getattr(module, "NodeRole")

    cfg = config if config is not None else module.SimulationConfig()
    for k, v in topo.sim_params.items():
        if hasattr(cfg, k):
            try:
                setattr(cfg, k, type(getattr(cfg, k))(v))
            except (TypeError, ValueError):
                setattr(cfg, k, v)
    if hasattr(cfg, "master_node_ids"):
        cfg.master_node_ids = topo.masters()

    obj = TopoCls.__new__(TopoCls)          # _build() не вызывается
    obj.config = cfg
    obj.graph = nx.Graph()
    obj.nodes = {}
    obj.links = {}

    topo.assign_zones()
    for n in topo.nodes.values():
        kwargs = dict(node_id=n.node_id, node_type=_enum(NodeType, n.node_type),
                      base_rate=float(n.base_rate))
        sim_node = NodeCls(**kwargs)
        for attr, value in (("hostname", n.hostname), ("realm", n.realm),
                            ("zone_id", int(n.zone_id)),
                            ("master_id", n.master_id),
                            ("is_compromised", bool(n.is_compromised))):
            if hasattr(sim_node, attr):
                setattr(sim_node, attr, value)
        sim_node.role = _enum(NodeRole, "MASTER" if n.is_master else "SLAVE")
        if n.interface_dist and hasattr(sim_node, "interface_dist"):
            sim_node.interface_dist = dict(n.interface_dist)
        obj.nodes[n.node_id] = sim_node
        obj.graph.add_node(n.node_id)

    for l in topo.links.values():
        key = (min(l.src, l.dst), max(l.src, l.dst))
        link = LinkCls(src=key[0], dst=key[1])
        for attr, value in (("capacity_mbps", l.capacity_mbps),
                            ("propagation_delay_ms", l.propagation_delay_ms),
                            ("base_loss_prob", l.base_loss_prob)):
            if hasattr(link, attr):
                setattr(link, attr, value)
        obj.links[key] = link
        obj.graph.add_edge(*key)

    return obj


def build_scenarios(topo: Topology, module):
    """Преобразует GUI-сценарии в AttackScenario / NormalScenario модуля."""
    attacks, normals = [], []
    AttackScenario = getattr(module, "AttackScenario", None)
    NormalScenario = getattr(module, "NormalScenario", None)
    AttackType = getattr(module, "AttackType", None)
    NormalEventType = getattr(module, "NormalEventType", None)
    for sc in topo.scenarios:
        if sc.kind == "attack" and AttackScenario and AttackType:
            attacks.append(AttackScenario(
                attack_type=_enum(AttackType, sc.name),
                target_nodes=list(sc.target_nodes),
                start_interval=sc.start_interval,
                end_interval=sc.end_interval,
                intensity=sc.intensity))
        elif NormalScenario and NormalEventType:
            normals.append(NormalScenario(
                event_type=_enum(NormalEventType, sc.name),
                target_nodes=list(sc.target_nodes),
                start_interval=sc.start_interval,
                end_interval=sc.end_interval,
                intensity=sc.intensity))
    return attacks, normals


def run_with_topology(topo: Topology, sim_dir: str,
                      log: Optional[Callable[[str], None]] = None,
                      module_name: Optional[str] = None):
    """Импортирует модуль симулятора из sim_dir и выполняет main()
    на нарисованной топологии."""
    log = log or print
    prof = PROTOCOLS[topo.protocol]
    if prof["topology_class"] is None:
        raise RuntimeError(
            "Симулятор 5G SBA не использует граф узлов: экспортируйте JSON "
            "и используйте его как логическую схему (число consumers/producers).")

    sim_dir = os.path.abspath(sim_dir)
    if sim_dir not in sys.path:
        sys.path.insert(0, sim_dir)
    name = module_name or prof["module"]
    module = importlib.import_module(name)
    log(f"модуль загружен: {module.__file__}")

    sim_topo = build_sim_topology(topo, module)
    log(f"топология подставлена: {len(sim_topo.nodes)} узлов, "
        f"{len(sim_topo.links)} линков, мастера {topo.masters()}")

    orig_cls = getattr(module, prof["topology_class"])
    attacks, normals = build_scenarios(topo, module)
    patched_gen = None
    for gen_name in ("generate_scenarios", "build_scenarios", "make_scenarios"):
        if hasattr(module, gen_name):
            patched_gen = gen_name
            break

    def factory(*_args, **_kwargs):
        return sim_topo

    setattr(module, prof["topology_class"], factory)
    orig_gen = getattr(module, patched_gen, None) if patched_gen else None
    if patched_gen and (attacks or normals):
        setattr(module, patched_gen, lambda *a, **k: (attacks, normals))
        log(f"сценарии подставлены через {patched_gen}(): "
            f"{len(attacks)} атак, {len(normals)} нормальных событий")
    elif topo.scenarios:
        log("ВНИМАНИЕ: в модуле нет хука генерации сценариев — "
            "сценарии из GUI сохранены в JSON, но не переданы в main()")

    try:
        result = module.main()
    finally:
        setattr(module, prof["topology_class"], orig_cls)
        if patched_gen and orig_gen is not None:
            setattr(module, patched_gen, orig_gen)
    log("main() завершён")
    return result
