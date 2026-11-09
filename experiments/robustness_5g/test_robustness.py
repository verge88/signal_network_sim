import random
from experiments.robustness_5g.design import attack_schedule,normal_count,growth_count
from experiments.robustness_5g.control import Control

def test_hour_schedule_has_every_family_and_nonoverlapping_variants():
    for seed in [44101,44201,44301]:
        schedule=attack_schedule(3600,seed)
        assert len(schedule)==36
        assert all(sum(e[2]==name for e in schedule)==4 for name in ['S1','S2','S3','S4','C','F','R','D','G'])
        assert all(a[1]<=b[0] for a,b in zip(schedule,schedule[1:]))
        permit=next(e for e in schedule if e[5]=='inside_permit')
        assert 40<=permit[0]%300<permit[1]%300<140
        assert all(growth_count(permit,w)+1<=48 for w in range(permit[0],permit[1]))
        for e in schedule:
            if e[5] in ['slow','intermittent']:
                assert e[0]%300 in [150,240]

def test_new_load_regimes_include_shift_without_changing_reference_bounds():
    rng=random.Random(12)
    reference=[normal_count(rng,5,w) for w in range(900)]
    shifted=[normal_count(rng,5,w) for w in range(900,1800)]
    assert max(reference)<=36 and max(shifted)>37
    assert all(normal_count(rng,3,w)<=48 for w in range(40,140))

def test_observer_really_stays_down_through_extended_benign_outage(tmp_path):
    root=tmp_path/'test'; root.mkdir()
    c=Control(root,'instance')
    try:
        c.prepare(80,'N1')
        assert not c.observer.is_alive() and c.down_until==100
        c.prepare(88,'N1')
        assert not c.observer.is_alive()
        c.prepare(100,'N1')
        assert c.observer.is_alive() and c.observe(100,'N1')['original'] is not None
    finally:
        c.close()
