"""Post-final V5 attribution ablation and structural-generalization diagnostic.

This analysis does not alter the frozen V5 model. It replays complete archived
final acquisitions and explicitly separates:
- the budgeted non-ML hierarchy,
- the learned V4 logistic probability without V4 structural gating,
- the frozen V4 core,
- the handcrafted V5 rescue,
- their frozen V5 union.

Structural robustness is evaluated by exact within-zone identifier permutations
and by counterfactual group-size interventions that reuse measured S5 trajectories.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import hashlib
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from experiments.distributed_5g.temporal_graph_v4 import (
    CommonModeResidualTransformer,
    FEATURES_V4,
)

KEYS = ["run_id", "window_id", "nf_id"]
ALPHA = 0.00015
EXPECTED_SEEDS = (90401, 90402, 90405, 90406, 90407, 90408, 90409, 90410)
ABLATION_METHODS = (
    "BUDGETED_WITHOUT_MINING",
    "LEARNED_PROBABILITY_ONLY",
    "V4_CORE_ONLY",
    "RESCUE_ONLY",
    "FULL_V5",
)
BRANCH_METHODS = (
    "LEARNED_PROBABILITY_ONLY",
    "TG_V4",
    "TG_RESCUE",
    "TG_V5",
)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_bundle(root):
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    if sha256(root / "model.joblib") != manifest["model_sha256"]:
        raise ValueError("Frozen model hash mismatch")
    bundle = joblib.load(root / "model.joblib")
    bundle["manifest"] = manifest
    return bundle


def conformal_threshold(scores, alpha):
    values = np.asarray(scores, dtype=float)
    rank = math.ceil((len(values) + 1) * (1 - alpha))
    return float(np.sort(values)[rank - 1]) if rank <= len(values) else math.inf


def slot_number(value):
    text = str(value)
    if not text.startswith("slot-"):
        raise ValueError(f"Unexpected NF identity: {text!r}")
    result = int(text.split("-", 1)[1])
    if not 0 <= result < 32:
        raise ValueError(result)
    return result


def metrics(truth, pred):
    truth = np.asarray(truth, dtype=bool)
    pred = np.asarray(pred, dtype=bool)
    tp = int((truth & pred).sum())
    fp = int((~truth & pred).sum())
    fn = int((truth & ~pred).sum())
    tn = int((~truth & ~pred).sum())
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "recall": tp / (tp + fn) if tp + fn else None,
        "fpr": fp / (fp + tn) if fp + tn else None,
        "precision": tp / (tp + fp) if tp + fp else None,
    }


def episode_summary(frame, pred):
    pred = np.asarray(pred, dtype=bool)
    positive = frame.attack.astype(bool) & frame.scenario.astype(str).eq("S5")
    ids = sorted(set(frame.loc[positive, "episode_id"].astype(str)) - {"", "nan"})
    detected = 0
    for episode in ids:
        mask = positive.to_numpy() & frame.episode_id.astype(str).eq(episode).to_numpy()
        detected += int(pred[mask].any())
    return {
        "detected": detected,
        "total": len(ids),
        "recall": detected / len(ids) if ids else None,
    }


def paired_greater(values):
    values = np.asarray(values, dtype=float)
    if not len(values) or np.allclose(values, 0):
        return {"statistic": 0.0, "pvalue_one_sided": 1.0}
    result = wilcoxon(values, alternative="greater", zero_method="wilcox", method="auto")
    return {
        "statistic": float(result.statistic),
        "pvalue_one_sided": float(result.pvalue),
    }


def load_frame(root):
    features = pd.read_csv(
        root / "test" / "features.csv",
        dtype={"run_id": str, "nf_id": str},
        float_precision="round_trip",
    )
    labels = pd.read_csv(
        root / "test" / "labels.csv",
        dtype={"run_id": str, "nf_id": str, "episode_id": str},
    ).fillna({"episode_id": "", "scenario": "N1", "variant": "normal"})
    if features.duplicated(KEYS).any() or labels.duplicated(KEYS).any():
        raise ValueError("Duplicate test keys")
    frame = features.merge(labels, on=KEYS, validate="one_to_one")
    if len(frame) != 115200:
        raise ValueError(f"Expected 115200 final-test rows, got {len(frame)}")
    return frame


def read_budgeted(root):
    rows = []
    with (root / "test" / "distributed_predictions.jsonl").open() as stream:
        for line in stream:
            result = json.loads(line)
            for nf, decision in result["decisions"].items():
                if "budgeted_without_ml" not in decision:
                    raise ValueError("Archived decision lacks budgeted_without_ml")
                rows.append(
                    {
                        "window_id": int(result["window"]),
                        "nf_id": str(nf),
                        "budgeted": bool(decision["budgeted_without_ml"]),
                    }
                )
    result = pd.DataFrame(rows)
    if len(result) != 115200 or result.duplicated(["window_id", "nf_id"]).any():
        raise ValueError("Unexpected distributed decision coverage")
    return result


def online_alarm(root, method):
    online = pd.read_csv(
        root / "test" / "online_predictions.csv",
        dtype={"run_id": str, "nf_id": str},
    )
    part = online.loc[online.method.astype(str).eq(method), KEYS + ["alarm"]].copy()
    if len(part) != 115200 or part.duplicated(KEYS).any():
        raise ValueError(f"Unexpected online coverage for {method}")
    return part.rename(columns={"alarm": method})


def learned_probability(bundle, frame):
    transformer = CommonModeResidualTransformer(history=8)
    graph = transformer.transform(frame)
    model = bundle["temporal_graph_v4"].model
    return model.predict_proba(graph[:, : len(FEATURES_V4)])[:, 1]


def v4_probability_and_evidence(bundle, frame):
    """Replay V4 once and expose both its learned probability and final evidence.

    This is algebraically the same path as TemporalGraphV4Detector.decision_function,
    but avoids recomputing the residual-temporal transform solely for the
    LEARNED_PROBABILITY_ONLY diagnostic.
    """
    detector = bundle["temporal_graph_v4"].clean_copy()
    transformer = CommonModeResidualTransformer(history=detector.history_size)
    graph = transformer.transform(frame)
    probability = detector.model.predict_proba(graph)[:, 1]
    structural = detector._structural(frame, probability, graph)
    evidence = np.zeros(len(frame), dtype=float)
    state = {}
    for pos, nf in enumerate(frame.nf_id.astype(str).to_numpy()):
        value = detector.decay * state.get(nf, 0.0) + float(structural[pos])
        state[nf] = value
        evidence[pos] = value
    return probability, evidence


def learned_threshold(root, bundle):
    cal = pd.read_csv(
        root / "cal" / "features.csv",
        dtype={"run_id": str, "nf_id": str},
        float_precision="round_trip",
    )
    score = learned_probability(bundle, cal)
    return conformal_threshold(score, ALPHA)


def ablation_seed(root, seed):
    frame = load_frame(root)
    bundle = load_bundle(root / "frozen")
    threshold = learned_threshold(root, bundle)
    learned_alarm = learned_probability(bundle, frame) > threshold

    table = frame[KEYS + ["attack", "scenario", "episode_id"]].copy()
    table = table.merge(read_budgeted(root), on=["window_id", "nf_id"], validate="one_to_one")
    for method in ("TG_V4", "TG_RESCUE", "TG_V5"):
        table = table.merge(online_alarm(root, method), on=KEYS, validate="one_to_one")
    table["LEARNED_POINT"] = learned_alarm.astype(int)

    predictions = {
        "BUDGETED_WITHOUT_MINING": table.budgeted.to_numpy(bool),
        "LEARNED_PROBABILITY_ONLY": (
            table.budgeted.to_numpy(bool) | table.LEARNED_POINT.to_numpy(bool)
        ),
        "V4_CORE_ONLY": table.budgeted.to_numpy(bool) | table.TG_V4.to_numpy(bool),
        "RESCUE_ONLY": table.budgeted.to_numpy(bool) | table.TG_RESCUE.to_numpy(bool),
        "FULL_V5": table.budgeted.to_numpy(bool) | table.TG_V5.to_numpy(bool),
    }
    truth = table.attack.to_numpy(bool)
    records = {}
    for method, pred in predictions.items():
        records[method] = {
            **metrics(truth, pred),
            "s5": episode_summary(table, pred),
        }
    return {
        "seed": int(seed),
        "learned_probability_threshold": float(threshold),
        "methods": records,
    }


def build_triplet_mapping(target):
    source = [1, 2, 3]
    target = list(target)
    if len(target) != 3 or len(set(target)) != 3 or not all(0 <= x < 8 for x in target):
        raise ValueError(target)
    source_rest = [x for x in range(8) if x not in source]
    target_rest = [x for x in range(8) if x not in target]
    mapping = dict(zip(source, target))
    mapping.update(dict(zip(source_rest, target_rest)))
    if set(mapping) != set(range(8)) or set(mapping.values()) != set(range(8)):
        raise AssertionError(mapping)
    return mapping


def permute_within_zone(frame, mapping):
    out = frame.copy()

    def mapped(value):
        number = slot_number(value)
        zone, rel = divmod(number, 8)
        return f"slot-{zone * 8 + mapping[rel]}"

    out["nf_id"] = out.nf_id.map(mapped)
    if out.duplicated(KEYS).any():
        raise ValueError("Permutation produced duplicate identities")
    return out


def original_s5_episodes(frame):
    positive = frame.attack.astype(bool) & frame.scenario.astype(str).eq("S5")
    episodes = []
    for episode, part in frame.loc[positive].groupby("episode_id", sort=True):
        members = sorted({slot_number(x) for x in part.nf_id.astype(str)})
        windows = sorted(part.window_id.astype(int).unique().tolist())
        if len(members) != 3 or len(windows) != 30:
            raise ValueError(f"Unexpected S5 episode geometry {episode}: {members}, {len(windows)}")
        zones = {x // 8 for x in members}
        if len(zones) != 1:
            raise ValueError("Original S5 episode crosses zones")
        episodes.append(
            {
                "episode": str(episode),
                "members": members,
                "zone": next(iter(zones)),
                "windows": windows,
                "onset": float(pd.to_numeric(part.onset, errors="coerce").dropna().min()),
            }
        )
    if len(episodes) != 4:
        raise ValueError(f"Expected four S5 episodes, got {len(episodes)}")
    return episodes


def counterfactual_group(frame, target_relative, variant_name):
    """Change only request_count and S5 labels, reusing measured S5 trajectories."""
    out = frame.copy()
    episodes = original_s5_episodes(frame)
    target_relative = list(target_relative)
    if not target_relative or len(set(target_relative)) != len(target_relative):
        raise ValueError(target_relative)
    if not all(0 <= x < 8 for x in target_relative):
        raise ValueError(target_relative)

    # Build the row locator once. The previous implementation recomputed an
    # all-frame NF map for every S5 window; that was equivalent but unnecessarily
    # expensive for the 8-seed diagnostic replay.
    numeric_slots = out.nf_id.astype(str).str.split("-", n=1).str[1].astype(int).to_numpy()
    windows_all = out.window_id.astype(int).to_numpy()
    locator = {
        (int(window), int(slot)): int(idx)
        for idx, window, slot in zip(out.index.to_numpy(), windows_all, numeric_slots)
    }

    for item in episodes:
        zone = item["zone"]
        original = item["members"]
        target = [zone * 8 + x for x in target_relative]
        windows = set(item["windows"])
        episode = item["episode"]

        # Clear original S5 labels for this episode; targets are assigned below.
        original_label_mask = out.episode_id.astype(str).eq(episode)
        out.loc[original_label_mask, "attack"] = 0
        out.loc[original_label_mask, "scenario"] = "N1"
        out.loc[original_label_mask, "episode_id"] = ""
        if "variant" in out:
            out.loc[original_label_mask, "variant"] = "normal"

        zone_members = list(range(zone * 8, zone * 8 + 8))
        for window in sorted(windows):
            by_slot = {member: locator[(int(window), member)] for member in zone_members}
            attack_values = [
                float(out.at[by_slot[member], "request_count"]) for member in original
            ]
            unaffected = [
                float(out.at[idx, "request_count"])
                for member, idx in by_slot.items()
                if member not in original
            ]
            replacement = float(np.median(unaffected))

            for member in original:
                out.at[by_slot[member], "request_count"] = replacement
            for j, member in enumerate(target):
                out.at[by_slot[member], "request_count"] = attack_values[j % len(attack_values)]

            for member in target:
                idx = by_slot[member]
                out.at[idx, "attack"] = 1
                out.at[idx, "scenario"] = "S5"
                out.at[idx, "episode_id"] = episode
                if "variant" in out:
                    out.at[idx, "variant"] = variant_name
                if "onset" in out:
                    out.at[idx, "onset"] = item["onset"]
    return out


def branch_predictions(bundle, frame, point_threshold):
    probability, v4_evidence = v4_probability_and_evidence(bundle, frame)
    learned = probability > point_threshold
    v4_threshold = float(bundle["thresholds"]["TG_V4"])
    v4_alarm = v4_evidence > v4_threshold

    rescue = bundle["cyclic_rescue"].clean_copy()
    rescue_threshold = float(bundle["thresholds"]["TG_RESCUE"])
    rescue_alarm = rescue.decision_function(frame) > rescue_threshold

    return {
        "LEARNED_PROBABILITY_ONLY": np.asarray(learned, dtype=bool),
        "TG_V4": np.asarray(v4_alarm, dtype=bool),
        "TG_RESCUE": np.asarray(rescue_alarm, dtype=bool),
        "TG_V5": np.asarray(v4_alarm | rescue_alarm, dtype=bool),
    }


def structural_seed(root, seed):
    frame = load_frame(root)
    bundle = load_bundle(root / "frozen")
    point_threshold = learned_threshold(root, bundle)

    variants = {
        "original": frame,
        "id_permutation_evenodd": permute_within_zone(
            frame, {0: 0, 1: 2, 2: 4, 3: 6, 4: 1, 5: 3, 6: 5, 7: 7}
        ),
        "contiguous_left": permute_within_zone(frame, build_triplet_mapping([0, 1, 2])),
        "contiguous_middle": permute_within_zone(frame, build_triplet_mapping([2, 3, 4])),
        "contiguous_right": permute_within_zone(frame, build_triplet_mapping([5, 6, 7])),
        "group_k2": counterfactual_group(frame, [1, 2], "counterfactual_k2"),
        "group_k3": counterfactual_group(frame, [1, 2, 3], "counterfactual_k3"),
        "group_k4": counterfactual_group(frame, [1, 2, 3, 4], "counterfactual_k4"),
        "group_k5": counterfactual_group(frame, [1, 2, 3, 4, 5], "counterfactual_k5"),
        "spaced_triple": counterfactual_group(frame, [0, 3, 6], "counterfactual_spaced"),
    }

    records = []
    for variant, variant_frame in variants.items():
        predictions = branch_predictions(bundle, variant_frame, point_threshold)
        truth = variant_frame.attack.astype(bool).to_numpy()
        s5 = (
            variant_frame.attack.astype(bool)
            & variant_frame.scenario.astype(str).eq("S5")
        ).to_numpy()
        normal = ~truth
        for method, pred in predictions.items():
            episode = episode_summary(variant_frame, pred)
            records.append(
                {
                    "seed": int(seed),
                    "variant": variant,
                    "method": method,
                    "s5_detected": int(episode["detected"]),
                    "s5_total": int(episode["total"]),
                    "s5_episode_recall": episode["recall"],
                    "s5_window_tp": int((pred & s5).sum()),
                    "s5_window_total": int(s5.sum()),
                    "s5_window_recall": (
                        float((pred & s5).sum() / s5.sum()) if s5.sum() else None
                    ),
                    "fp": int((pred & normal).sum()),
                    "normal_total": int(normal.sum()),
                    "fpr": float((pred & normal).sum() / normal.sum()),
                }
            )
    return records


def pooled_method(seed_records, method):
    items = [r["methods"][method] for r in seed_records]
    total = {
        key: sum(int(x[key]) for x in items)
        for key in ("tp", "fp", "fn", "tn")
    }
    total["recall"] = total["tp"] / (total["tp"] + total["fn"])
    total["fpr"] = total["fp"] / (total["fp"] + total["tn"])
    total["precision"] = total["tp"] / (total["tp"] + total["fp"])
    total["s5_detected"] = sum(int(x["s5"]["detected"]) for x in items)
    total["s5_total"] = sum(int(x["s5"]["total"]) for x in items)
    total["s5_episode_recall"] = total["s5_detected"] / total["s5_total"]
    return total


def aggregate_generalization(records):
    frame = pd.DataFrame(records)
    result = []
    for (variant, method), part in frame.groupby(["variant", "method"], sort=True):
        detected = int(part.s5_detected.sum())
        episodes = int(part.s5_total.sum())
        s5_tp = int(part.s5_window_tp.sum())
        s5_n = int(part.s5_window_total.sum())
        fp = int(part.fp.sum())
        normal = int(part.normal_total.sum())
        result.append(
            {
                "variant": variant,
                "method": method,
                "s5_detected": detected,
                "s5_total": episodes,
                "s5_episode_recall": detected / episodes if episodes else None,
                "s5_window_recall": s5_tp / s5_n if s5_n else None,
                "fpr": fp / normal if normal else None,
            }
        )
    return result


def report_markdown(summary):
    lines = [
        "# V5 ablation and structural-generalization diagnostic",
        "",
        "Frozen V5 parameters were not changed. Eight complete archived final acquisitions were replayed; "
        "seeds 90403 and 90404 are absent because their technical retries retained summary artifacts but not full evidence.",
        "",
        "## Attribution ablation",
        "",
        "| Method | Recall | FPR | Precision | S5 episodes |",
        "|---|---:|---:|---:|---:|",
    ]
    for method in ABLATION_METHODS:
        item = summary["ablation"]["pooled"][method]
        lines.append(
            f"| {method} | {item['recall']:.6f} | {item['fpr']:.6f} | "
            f"{item['precision']:.6f} | {item['s5_detected']}/{item['s5_total']} |"
        )

    lines += [
        "",
        "Paired seed-level tests are diagnostic/post-final, not a new preregistered prospective confirmation.",
        "",
        "## Structural generalization",
        "",
        "| Variant | Method | S5 episode recall | S5 NF-window recall | FPR |",
        "|---|---|---:|---:|---:|",
    ]
    for item in summary["generalization"]["pooled"]:
        lines.append(
            f"| {item['variant']} | {item['method']} | "
            f"{item['s5_episode_recall']:.3f} | {item['s5_window_recall']:.3f} | {item['fpr']:.6f} |"
        )
    lines += [
        "",
        "Exact identifier/location permutations preserve measured feature values and only rename NF identities. "
        "Group-size counterfactuals change request_count only, because TG_V4/TG_RESCUE derive their runtime evidence from request_count.",
        "",
        "A failure under a variant is reported as a limitation of frozen V5; no post-hoc V5 parameter change is applied.",
    ]
    return "\n".join(lines) + "\n"


def run(root, output, seeds=EXPECTED_SEEDS):
    root = Path(root)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)

    seed_records = []
    generalization_records = []
    for seed in seeds:
        seed_root = root / str(seed)
        if not seed_root.exists():
            raise ValueError(f"Missing input seed {seed}")
        seed_records.append(ablation_seed(seed_root, seed))
        generalization_records.extend(structural_seed(seed_root, seed))

    pooled = {method: pooled_method(seed_records, method) for method in ABLATION_METHODS}

    paired_tests = {}
    comparisons = [
        ("LEARNED_PROBABILITY_ONLY", "BUDGETED_WITHOUT_MINING"),
        ("V4_CORE_ONLY", "BUDGETED_WITHOUT_MINING"),
        ("V4_CORE_ONLY", "LEARNED_PROBABILITY_ONLY"),
        ("RESCUE_ONLY", "BUDGETED_WITHOUT_MINING"),
        ("FULL_V5", "V4_CORE_ONLY"),
        ("FULL_V5", "BUDGETED_WITHOUT_MINING"),
    ]
    for upper, lower in comparisons:
        differences = [
            r["methods"][upper]["recall"] - r["methods"][lower]["recall"]
            for r in seed_records
        ]
        paired_tests[f"{upper}_gt_{lower}"] = {
            "mean_recall_delta": float(np.mean(differences)),
            "per_seed_deltas": [float(x) for x in differences],
            **paired_greater(differences),
        }

    # Two-block Shapley attribution at the system-recall level: V4 core vs explicit rescue.
    b = pooled["BUDGETED_WITHOUT_MINING"]["recall"]
    v = pooled["V4_CORE_ONLY"]["recall"]
    r = pooled["RESCUE_ONLY"]["recall"]
    f = pooled["FULL_V5"]["recall"]
    shapley = {
        "v4_core_recall_contribution": 0.5 * ((v - b) + (f - r)),
        "explicit_rescue_recall_contribution": 0.5 * ((r - b) + (f - v)),
        "total_recall_gain": f - b,
        "note": (
            "V4 core is not purely learned: it includes learned logistic probabilities plus "
            "a fixed structural score and temporal accumulation. LEARNED_PROBABILITY_ONLY "
            "is reported separately to expose that distinction."
        ),
    }

    generalization_pooled = aggregate_generalization(generalization_records)
    summary = {
        "protocol": {
            "seeds": list(seeds),
            "frozen_v5_unchanged": True,
            "point_ablation_alpha": ALPHA,
            "raw_evidence_seed_count": len(seeds),
            "excluded_raw_seeds": [90403, 90404],
            "post_final_diagnostic": True,
        },
        "ablation": {
            "pooled": pooled,
            "paired_tests": paired_tests,
            "shapley_two_block_recall": shapley,
            "per_seed": seed_records,
        },
        "generalization": {
            "pooled": generalization_pooled,
            "per_seed": generalization_records,
            "interpretation_rule": (
                "Negative frozen-V5 variants narrow the claim and are not repaired by changing "
                "the already-frozen V5 parameters in this analysis."
            ),
        },
    }

    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n"
    )

    ablation_rows = []
    for record in seed_records:
        for method, item in record["methods"].items():
            ablation_rows.append(
                {
                    "seed": record["seed"],
                    "method": method,
                    "learned_probability_threshold": record["learned_probability_threshold"],
                    **{k: item[k] for k in ("tp", "fp", "fn", "tn", "recall", "fpr", "precision")},
                    "s5_detected": item["s5"]["detected"],
                    "s5_total": item["s5"]["total"],
                    "s5_episode_recall": item["s5"]["recall"],
                }
            )
    pd.DataFrame(ablation_rows).to_csv(output / "ablation_per_seed.csv", index=False)
    pd.DataFrame(generalization_records).to_csv(
        output / "generalization_per_seed.csv", index=False
    )
    (output / "REPORT.md").write_text(report_markdown(summary))

    print(
        json.dumps(
            {
                "ablation_pooled": pooled,
                "paired_tests": paired_tests,
                "shapley": shapley,
                "generalization": generalization_pooled,
            },
            allow_nan=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seeds", type=int, nargs="*", default=None)
    args = parser.parse_args()
    selected = tuple(args.seeds) if args.seeds else EXPECTED_SEEDS
    unknown = sorted(set(selected) - set(EXPECTED_SEEDS))
    if unknown:
        parser.error(f"Unsupported diagnostic seeds: {unknown}")
    run(args.root, args.output, seeds=selected)
