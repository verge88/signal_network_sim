"""Predeclared V7 prospective FPR-constrained structural schedule.

Imported only by collection/audit code. Detector code never receives scenario
labels, membership sets, episode timing, or this seeded randomization.
"""
from __future__ import annotations

import random

V7_WINDOWS=600
V7_EPISODE_STARTS=(30,100,170,240,310,380,450,520)
V7_GROUP_SIZES=(2,3,4,5,2,3,4,5)


def v7_attack_schedule(windows, seed):
    if int(windows)!=V7_WINDOWS:
        raise ValueError(f"V7 structural trial requires {V7_WINDOWS} windows")
    rng=random.Random(int(seed)+7_000_007)
    sizes=list(V7_GROUP_SIZES)
    rng.shuffle(sizes)
    result=[]
    for episode,(start,k) in enumerate(zip(V7_EPISODE_STARTS,sizes)):
        zone=(episode+int(seed)+1)%4
        members=sorted(rng.sample(list(range(zone*8,(zone+1)*8)),k))
        variant=f"k{k}_members_"+"-".join(str(x) for x in members)
        result.append((start,start+30,"S7",members[0],0,variant))
    return result


def v7_members(entry):
    if entry[2]!="S7":
        return ()
    variant=str(entry[5])
    if "_members_" not in variant:
        raise ValueError(f"Invalid V7 variant {variant!r}")
    prefix,raw=variant.split("_members_",1)
    if not prefix.startswith("k"):
        raise ValueError(variant)
    k=int(prefix[1:])
    members=tuple(int(x) for x in raw.split("-") if x!="")
    if len(members)!=k or len(set(members))!=k:
        raise ValueError(f"Invalid V7 members {members}")
    if not all(0<=x<32 for x in members):
        raise ValueError(members)
    if len({x//8 for x in members})!=1:
        raise ValueError("V7 coordinated group must remain within one eight-NF zone")
    return members


def v7_count(entry, window, index):
    members=v7_members(entry)
    if index not in members:
        raise ValueError("V7 count requested for non-member")
    elapsed=int(window)-int(entry[0])
    if not 0<=elapsed<30:
        raise ValueError("V7 count requested outside episode")
    wave=(0,1,1,2,1,0)[(elapsed//3)%6]
    rank=members.index(index)
    return int(2+wave+(rank%2))
