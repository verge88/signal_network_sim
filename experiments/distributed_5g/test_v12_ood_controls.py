"""Counterfactual non-S9 robustness controls for the frozen V12 implementation."""
from experiments.distributed_5g.v11_staircase import PERIOD, WAVE
from experiments.distributed_5g.v12_staircase import ConfirmedStaircaseDetector

NF_IDS = tuple(f"slot-{i}" for i in range(8))
ZONES = {nf: 0 for nf in NF_IDS}


def scenario(members=(0,1), bases=(2,3), wave=WAVE, onset=7, duration=30):
    rows = []
    for w in range(46):
        counts = {nf: 8 + (i%3) for i,nf in enumerate(NF_IDS)}
        if onset <= w < onset+duration:
            offset = wave[((w-onset)//PERIOD)%len(wave)]
            for i,slot in enumerate(members):
                counts[f"slot-{slot}"] = bases[i] + offset
        rows.append(counts)
    return rows


def signal_count(rows):
    detector = ConfirmedStaircaseDetector(ZONES)
    alarms = [detector.step(w,row) for w,row in enumerate(rows)]
    return sum(v["slot-0"]+v["slot-1"] for v in alarms[7:37])


def test_known_s9_signature_has_exactly_27_per_nf():
    assert signal_count(scenario()) == 54


def test_benign_observational_copy_has_identical_false_alerts():
    """Without an external witness, counts alone cannot separate benign mimicry."""
    identical_but_labeled_benign = scenario()
    assert signal_count(identical_but_labeled_benign) == 54


def test_one_nf_attack_not_detected_by_zone_cooccurrence():
    assert signal_count(scenario(members=(0,),bases=(2,))) == 0


def test_shift_to_higher_baselines_undetectable():
    assert signal_count(scenario(bases=(5,6))) == 0


def test_changed_waveform_not_full_recall():
    assert signal_count(scenario(wave=(0,1,2,3,2,1))) < 54
