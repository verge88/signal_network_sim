"""Predeclared test-only distribution shifts; never imported by detectors."""
import random

def regime(window):
    return ['reference','higher_load','cyclic','bursty_recovery'][min(3,window//900)]

def normal_count(rng,index,window):
    phase=regime(window)
    if phase=='reference':
        count=rng.randrange(1,7)
        if rng.random()<.04: count=rng.randrange(12,37)
    elif phase=='higher_load':
        count=rng.randrange(4,13)
        if rng.random()<.06: count=rng.randrange(24,49)
    elif phase=='cyclic':
        count=2+(window%120)//8
        if rng.random()<.04: count=rng.randrange(12,37)
    else:
        count=rng.randrange(1,7)
        if rng.random()<.08: count=rng.randrange(12,37)
    if index==3 and 40<=window%300<140:
        count=4+(window%300-40)//4
    return count

def attack_schedule(windows,seed=0):
    if windows!=3600:
        raise ValueError('Preregistered one-hour schedule requires 3600 windows')
    cases=[(name,slot,0,'standard') for name,slot in
        [('S1',0),('S2',1),('S3',2),('S4',3),('C',4),('F',4),('R',4),('D',4)] for _ in range(4)]
    cases += [('G',3,2,'fast'),('G',3,1,'slow'),('G',3,1,'intermittent'),('G',3,1,'inside_permit')]
    random.Random(seed).shuffle(cases)
    # A grant is issued independently every cycle at window 40, for 100 seconds.
    idx=next(i for i,c in enumerate(cases) if c[3]=='inside_permit')
    cases[0],cases[idx]=cases[idx],cases[0]
    for variant in ['slow','intermittent']:
        idx=next(i for i,c in enumerate(cases) if c[3]==variant)
        if idx%3==0:
            target=next(i for i,c in enumerate(cases) if i%3!=0 and c[0]!='G')
            cases[idx],cases[target]=cases[target],cases[idx]
    result=[]
    for i,(name,slot,step,variant) in enumerate(cases):
        start=300*(i//3)+[60,150,240][i%3]
        duration=60 if variant in ['slow','intermittent'] else 20 if variant=='inside_permit' else 10
        result.append((start,start+duration,name,slot,step,variant))
    return result

def growth_count(entry,window):
    elapsed=window-entry[0]; variant=entry[5]
    if variant=='fast': return 4+entry[4]*elapsed
    if variant in ['slow','inside_permit']: return 4+elapsed//6
    if variant=='intermittent': return 4+elapsed//6 if elapsed%6<3 else 2
    raise ValueError(variant)
