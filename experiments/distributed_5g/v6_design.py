"""Predeclared V6 prospective structural-generalization schedule.

This module is imported only by the collector/auditor. Detector code never sees
scenario names, labels, membership sets, or the random permutation used here.
"""
from __future__ import annotations

import random


V6_WINDOWS=600
V6_EPISODE_STARTS=(30,100,170,240,310,380,450,520)
V6_GROUP_SIZES=(2,3,4,5,2,3,4,5)


def v6_attack_schedule(windows, seed):
    if int(windows)!=V6_WINDOWS:
        raise ValueError(f"V6 structural trial requires {V6_WINDOWS} windows")
    rng=random.Random(int(seed)+6_000_006)
    sizes=list(V6_GROUP_SIZES)
    rng.shuffle(sizes)
    result=[]
    for episode,(start,k) in enumerate(zip(V6_EPISODE_STARTS,sizes)):
        zone=(episode+int(seed))%4
        members=sorted(rng.sample(list(range(zone*8,(zone+1)*8)),k))
        variant=f"k{k}_members_"+"-".join(str(x) for x in members)
        # slot field remains a conventional representative only; membership is
        # encoded in the preregistered variant and recovered by v6_members().
        result.append((start,start+30,"S6",members[0],0,variant))
    return result


def v6_members(entry):
    if entry[2]!="S6":
        return ()
    variant=str(entry[5])
    if "_members_" not in variant:
        raise ValueError(f"Invalid V6 variant {variant!r}")
    prefix,raw=variant.split("_members_",1)
    if not prefix.startswith("k"):
        raise ValueError(variant)
    k=int(prefix[1:])
    members=tuple(int(x) for x in raw.split("-") if x!="")
    if len(members)!=k or len(set(members))!=k:
        raise ValueError(f"Invalid V6 members {members}")
    if not all(0<=x<32 for x in members):
        raise ValueError(members)
    zones={x//8 for x in members}
    if len(zones)!=1:
        raise ValueError("V6 coordinated group must remain within one eight-NF zone")
    return members


def v6_count(entry, window, index):
    members=v6_members(entry)
    if index not in members:
        raise ValueError("V6 count requested for non-member")
    elapsed=int(window)-int(entry[0])
    if not 0<=elapsed<30:
        raise ValueError("V6 count requested outside episode")
    wave=(0,1,1,2,1,0)[(elapsed//3)%6]
    rank=members.index(index)
    # Same benign-range low-and-slow waveform as S5, with small deterministic
    # offsets so equality is not a shortcut. All counts stay in [2,5].
    return int(2+wave+(rank%2))
