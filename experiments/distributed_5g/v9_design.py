"""Predeclared V9 prospective specialized-fusion structural schedule."""
from __future__ import annotations
import random

V9_WINDOWS=700
V9_EPISODE_STARTS=(30,100,170,240,310,380,450,520,590,660)
V9_GROUP_SIZES=(2,2,2,2,3,3,4,4,5,5)


def v9_attack_schedule(windows,seed):
    if int(windows)!=V9_WINDOWS:
        raise ValueError(f"V9 structural trial requires {V9_WINDOWS} windows")
    rng=random.Random(int(seed)+9_000_009)
    sizes=list(V9_GROUP_SIZES);rng.shuffle(sizes)
    result=[]
    for episode,(start,k) in enumerate(zip(V9_EPISODE_STARTS,sizes)):
        zone=(episode+int(seed)+3)%4
        members=sorted(rng.sample(list(range(zone*8,(zone+1)*8)),k))
        variant=f"k{k}_members_"+"-".join(str(x) for x in members)
        result.append((start,start+30,"S9",members[0],0,variant))
    return result


def v9_members(entry):
    if entry[2]!="S9": return ()
    variant=str(entry[5])
    prefix,raw=variant.split("_members_",1)
    k=int(prefix[1:]);members=tuple(int(x) for x in raw.split("-") if x!="")
    if len(members)!=k or len(set(members))!=k or len({x//8 for x in members})!=1:
        raise ValueError(f"Invalid V9 members {members}")
    return members


def v9_count(entry,window,index):
    members=v9_members(entry)
    if index not in members: raise ValueError("V9 count requested for non-member")
    elapsed=int(window)-int(entry[0])
    if not 0<=elapsed<30: raise ValueError("V9 count requested outside episode")
    wave=(0,1,1,2,1,0)[(elapsed//3)%6]
    rank=members.index(index)
    return int(2+wave+(rank%2))
