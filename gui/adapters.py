"""
gui/adapters.py — мост между GUI-моделью и симуляторами репозитория.

Идея: не менять код симуляторов. Мы создаём объект их класса топологии
через __new__ (минуя процедурный _build), заполняем nodes/links/graph
из нарисованной сети и подменяем класс в модуле фабрикой, возвращающей
этот объект. Дальше вызывается штатный main() модуля.
"""
from __future__ import annotations

import importlib
import inspect
import dataclasses
import os
import sys
from typing import Any, Callable, Dict, Optional

from .model import PROTOCOLS, Topology


def _enum(enum_cls, name: str):
    try:
        return enum_cls[name]
    except KeyError:
        for member in enum_cls:
            if str(getattr(member, "value", "")).lower() == name.lower():
                return member
        raise ValueError(f"{enum_cls.__name__}: нет варианта {name!r}")


LINK_MAP = {
    "capacity_mbps": [("capacity_mbps", lambda value: value),
                      ("capacity_kbps", lambda value: value * 1000.0),
                      ("capacity_bps", lambda value: value * 1e6)],
    "propagation_delay_ms": [("propagation_delay_ms", lambda value: value),
                              ("propagation_delay_s", lambda value: value / 1000.0),
                              ("delay_s", lambda value: value / 1000.0)],
    "base_loss_prob": [("base_loss_prob", lambda value: value),
                        ("loss_prob", lambda value: value)],
}

NODE_MAP = {
    "hostname": [("hostname", lambda value: value), ("point_code", lambda value: value),
                 ("fqdn", lambda value: value)],
    "realm": [("realm", lambda value: value)],
    "zone_id": [("zone_id", int)],
    "master_id": [("master_id", lambda value: value)],
    "is_compromised": [("is_compromised", bool)],
    "base_rate": [("base_rate", float)],
    "interface_dist": [("interface_dist", dict), ("msg_type_dist", dict),
                        ("iface_dist", dict)],
}


def _field_names(cls) -> set[str]:
    if dataclasses.is_dataclass(cls):
        return {item.name for item in dataclasses.fields(cls)}
    try:
        return set(inspect.signature(cls.__init__).parameters) - {"self"}
    except (TypeError, ValueError):
        return set()


def _apply(target, cls_fields: set[str], mapping: Dict[str, list],
           values: Dict[str, Any], log, skipped: Optional[Dict[str, int]] = None) -> None:
    for gui_key, candidates in mapping.items():
        value = values.get(gui_key)
        if gui_key not in values or value in (None, "", {}):
            continue
        for sim_key, converter in candidates:
            if sim_key in cls_fields or hasattr(target, sim_key):
                try:
                    setattr(target, sim_key, converter(value))
                except (TypeError, ValueError) as exc:
                    log(f"ВНИМАНИЕ: {sim_key}: {exc}")
                break
        else:
            if skipped is not None:
                skipped[gui_key] = skipped.get(gui_key, 0) + 1


def build_sim_topology(topo: Topology, module, config=None, log=print):
    """Собирает объект *NetworkTopology симулятора из GUI-топологии."""
    import networkx as nx

    prof = PROTOCOLS[topo.protocol]
    TopoCls = getattr(module, prof["topology_class"])
    NodeCls = getattr(module, prof["node_class"])
    LinkCls = getattr(module, prof["link_class"])
    NodeType = getattr(module, "NodeType")
    NodeRole = getattr(module, "NodeRole")
    node_fields, link_fields = _field_names(NodeCls), _field_names(LinkCls)
    skipped: Dict[str, int] = {}

    cfg = config if config is not None else module.SimulationConfig()
    for k, v in topo.sim_params.items():
        if hasattr(cfg, k):
            try:
                setattr(cfg, k, type(getattr(cfg, k))(v))
            except (TypeError, ValueError):
                setattr(cfg, k, v)
    if hasattr(cfg, "master_node_ids"):
        cfg.master_node_ids = topo.masters()

    counts: Dict[str, int] = {}
    for node in topo.nodes.values():
        counts[node.node_type] = counts.get(node.node_type, 0) + 1
    for node_type, count in counts.items():
        for attr in (f"n_{node_type.lower()}", f"n_{node_type.lower()}s"):
            if hasattr(cfg, attr):
                setattr(cfg, attr, count)
                break

    obj = TopoCls.__new__(TopoCls)          # _build() не вызывается
    obj.config = cfg
    obj.graph = nx.Graph()
    obj.nodes = {}
    obj.links = {}

    topo.assign_zones()
    for n in topo.nodes.values():
        sim_node = NodeCls(node_id=n.node_id,
                           node_type=_enum(NodeType, n.node_type))
        sim_node.role = _enum(NodeRole, "MASTER" if n.is_master else "SLAVE")
        _apply(sim_node, node_fields, NODE_MAP,
               {"hostname": n.hostname or f"{n.node_type}-{n.node_id:03d}",
                "realm": n.realm, "zone_id": n.zone_id, "master_id": n.master_id,
                "is_compromised": n.is_compromised, "base_rate": n.base_rate,
                "interface_dist": n.interface_dist}, log, skipped)
        if n.is_compromised and "compromised_since" in node_fields:
            sim_node.compromised_since = 0
        obj.nodes[n.node_id] = sim_node
        obj.graph.add_node(n.node_id)

    for l in topo.links.values():
        key = (min(l.src, l.dst), max(l.src, l.dst))
        link = LinkCls(src=key[0], dst=key[1])
        _apply(link, link_fields, LINK_MAP,
               {"capacity_mbps": l.capacity_mbps,
                "propagation_delay_ms": l.propagation_delay_ms,
                "base_loss_prob": l.base_loss_prob}, log, skipped)
        obj.links[key] = link
        obj.graph.add_edge(*key)

    log(f"узлов {len(obj.nodes)}, линков {len(obj.links)}, мастера {topo.masters()}, "
        f"поля узла: {sorted(node_fields)}")
    irrelevant = {"ss7": {"realm", "capacity_mbps"}, "diameter": set(), "sip": set()}
    for name, count in sorted(skipped.items()):
        level = "инфо" if name in irrelevant.get(topo.protocol, set()) else "ВНИМАНИЕ"
        log(f"{level}: поле «{name}» не поддерживается "
            f"{NodeCls.__name__ if name in NODE_MAP else LinkCls.__name__} — "
            f"пропущено ({count} объектов)")
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

    sim_topo = build_sim_topology(topo, module, log=log)
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
        try:
            parameters = inspect.signature(module.main).parameters
        except (TypeError, ValueError):
            parameters = {}
        if parameters:
            config = getattr(sim_topo, "config", None)
            result = module.main(config) if config is not None else module.main()
        else:
            result = module.main()
    finally:
        setattr(module, prof["topology_class"], orig_cls)
        if patched_gen and orig_gen is not None:
            setattr(module, patched_gen, orig_gen)
    log("main() завершён")
    return result
