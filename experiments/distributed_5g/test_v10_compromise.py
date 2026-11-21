"""V10 integrity contracts, deliberately independent of Open5GS host privileges."""
import copy
import hashlib
import json

import joblib
import numpy as np
import pandas as pd
import pytest

from experiments.distributed_5g.v10_design import (
    ALL_SCENARIOS, V10_WINDOWS, V10_COMPROMISE_VARIANTS,
    STRUCTURAL_SCENARIOS, s10_control_mode, v10_attack_schedule,
    v10_count, v10_members,
)
from experiments.distributed_5g.v10_model import V10MultiThreatFusion, derive_from_v9
from experiments.integrated_5g.control import combine
from experiments.live_5g.collect import sign_report
from experiments.live_5g.evaluate import PROTOCOL_VERSION, load_frozen


def features(count=4, error=.0, **flags):
    row = {n:0 for n in V10MultiThreatFusion.RULE_COLUMNS}
    row.update(request_count=count,error_rate=error)
    row.update(flags)
    return pd.DataFrame([row])


def test_schedule_preserves_every_original_attack_and_appends_new_families():
    from experiments.robustness_5g.design import attack_schedule
    for seed in (91401,91402,91403):
        old = attack_schedule(V10_WINDOWS,seed)
        got = v10_attack_schedule(V10_WINDOWS,seed)
        assert all(x in got for x in old) and len(got)==len(old)+19
        assert {x[2] for x in got} == ALL_SCENARIOS
        assert all(a[1] <= b[0] for a,b in zip(got,got[1:]))
        assert len([e for e in got if e[2]=='S10'])==3
        assert {e[5] for e in got if e[2]=='S10'} == set(V10_COMPROMISE_VARIANTS)
        for scenario in STRUCTURAL_SCENARIOS:
            sizes = sorted(len(v10_members(e)) for e in got if e[2]==scenario)
            assert sizes==[2,3,4,5]
        for e in got:
            assert 0<=e[0]<e[1]<=V10_WINDOWS


def test_structural_attack_variants_have_bounded_30_window_lab_traffic():
    plan = v10_attack_schedule(V10_WINDOWS, 91403)
    for entry in plan:
        if entry[2] not in STRUCTURAL_SCENARIOS:
            continue
        assert entry[1]-entry[0]==30
        members=v10_members(entry)
        assert len({slot//8 for slot in members})==1
        for w in range(entry[0],entry[1]):
            for member in members:
                assert 2 <= v10_count(entry,w,member) <= 5


def test_v10_schedule_and_compromise_variants_are_deterministic():
    for seed in (91401,91402):
        assert v10_attack_schedule(V10_WINDOWS,seed)==v10_attack_schedule(V10_WINDOWS,seed)
    assert v10_attack_schedule(V10_WINDOWS,91401)!=v10_attack_schedule(V10_WINDOWS,91402)
    assert s10_control_mode("node_config")=="C"
    assert s10_control_mode("node_report_tamper")=="F"
    assert s10_control_mode("node_sbi_abuse")=="C"
    with pytest.raises(ValueError):
        s10_control_mode("stolen_admin_key")
    with pytest.raises(ValueError):
        v10_attack_schedule(700,91401)


def test_s10_restricted_to_sandboxed_research_nf_slot4():
    for entry in v10_attack_schedule(V10_WINDOWS,91401):
        if entry[2]=="S10":
            assert v10_members(entry)==(4,)
            assert entry[1]-entry[0]==30
            bad=list(entry);bad[3]=0
            with pytest.raises(ValueError):
                v10_members(tuple(bad))


@pytest.mark.parametrize("flag",V10MultiThreatFusion.RULE_COLUMNS)
def test_traffic_policy_and_integrity_flags_raise_alarm(flag):
    v10 = V10MultiThreatFusion()
    assert v10.combine_traffic(np.array([-1.0]),features(**{flag:1}))[0] > 0
    assert v10.combine_traffic(np.array([-1.0]),features())[0] < 0


def test_v10_burst_error_conjunction_not_simple_high_load_trigger():
    v10=V10MultiThreatFusion()
    assert v10.combine_traffic([-1.0],features(count=25,error=.28))[0]>0
    assert v10.combine_traffic([-1.0],features(count=40,error=.0))[0]<0
    assert v10.combine_traffic([-1.0],features(count=4,error=.45))[0]<0
    assert v10.combine_traffic([.25],features())[0] == .25


def test_v10_rejects_missing_or_nonfinite_evidence():
    v10=V10MultiThreatFusion()
    with pytest.raises(ValueError):
        v10.combine_traffic([-1.0],features().drop(columns=['auth_failure']))
    with pytest.raises(ValueError):
        v10.combine_traffic([-1.0],features(error=np.nan))
    with pytest.raises(ValueError):
        v10.combine_traffic([1.0,1.0],features())


def observation(config_hash, expected, key, nonce="nonce", timestamp=100.0):
    payload=dict(nonce=nonce,instance="research-slot-4",
                 timestamp=timestamp,config_hash=config_hash)
    envelope=dict(report=payload,signature=sign_report(payload,key))
    return dict(delivered=envelope,nonce=nonce,
                expected=expected,instance="research-slot-4")


def test_compromised_nf_config_detected_by_independent_authenticated_witness():
    key=b"isolated-lab-control-key"
    v10=V10MultiThreatFusion()
    normal=observation("allowed","allowed",key)
    changed=observation("malicious-config","allowed",key)
    assert v10.assess_external(False,normal,key,100.1)["control_state"]=="normal"
    out=v10.assess_external(False,changed,key,100.1)
    assert out["security_alarm"] and not out["availability_unknown"]


def test_compromised_local_report_detected_without_stealing_trusted_key():
    key=b"isolated-lab-control-key"
    v10=V10MultiThreatFusion()
    changed=observation("malicious-config","allowed",key)
    forged=copy.deepcopy(changed)
    forged["delivered"]["report"]["config_hash"]="allowed" # signature invalid
    out=v10.assess_external(False,forged,key,100.1)
    assert out["security_alarm"]
    assert out["control_state"]=="alarm"


def test_missing_or_stale_report_unknown_not_automatic_attack():
    key=b"isolated-lab-control-key"
    v10=V10MultiThreatFusion()
    dropped=observation("allowed","allowed",key);dropped["delivered"]=None
    out=v10.assess_external(False,dropped,key,100.1)
    assert out["availability_unknown"] and not out["security_alarm"]
    stale=observation("allowed","allowed",key,nonce="old")
    stale["nonce"]="fresh"
    assert v10.assess_external(False,stale,key,100.1)["control_state"]=="unknown"
    assert v10.assess_external(True,dropped,key,100.1)["security_alarm"]


def test_fully_compromised_external_signer_is_out_of_scope_and_cannot_be_claimed_detected():
    key=b"out-of-scope-stolen-witness-key"
    # If an attacker controlled BOTH the external signing key and the report,
    # a newly valid forged normal report is observationally indistinguishable
    # to this research guard. This is a mandatory honest negative control.
    normal_looking=observation("allowed","allowed",key)
    out=V10MultiThreatFusion().assess_external(False,normal_looking,key,100.1)
    assert not out["security_alarm"] and not out["availability_unknown"]


class ResetStub:
    def reset(self):
        pass


def test_separate_frozen_v10_manifest_keeps_original_v9_bytes(tmp_path):
    src=tmp_path/"v9";src.mkdir()
    bundle={"protocol_version":PROTOCOL_VERSION,"ml_method":"TG_V9",
            "v9_pair_rescue":ResetStub()}
    joblib.dump(bundle,src/"model.joblib")
    before=hashlib.sha256((src/"model.joblib").read_bytes()).hexdigest()
    manifest={"model_sha256":before,"ml_method":"TG_V9",
              "temporal_graph_v9":{"heldout_v9_test_used_for_selection":False},
              "seed":91201}
    (src/"manifest.json").write_text(json.dumps(manifest))
    out=tmp_path/"v10"
    version=derive_from_v9(src,out)
    assert version["ml_method"]=="TG_V10"
    assert version["temporal_graph_v10"]["new_training"] is False
    assert version["temporal_graph_v10"]["new_calibration"] is False
    assert version["temporal_graph_v10"]["parent_frozen_model_sha256"]==before
    assert (src/"model.joblib").read_bytes()==(src/"model.joblib").read_bytes()
    assert hashlib.sha256((src/"model.joblib").read_bytes()).hexdigest()==before
    frozen=load_frozen(out)
    assert isinstance(frozen["v10_multithreat"],V10MultiThreatFusion)
    with pytest.raises(FileExistsError):
        derive_from_v9(src,out)


def test_v9_must_be_verified_to_create_a_v10_model(tmp_path):
    src=tmp_path/"bad";src.mkdir()
    joblib.dump({"protocol_version":PROTOCOL_VERSION,"ml_method":"TG_V9",
                 "v9_pair_rescue":ResetStub()},src/"model.joblib")
    (src/"manifest.json").write_text(json.dumps({
        "model_sha256":"0"*64,"ml_method":"TG_V9",
        "temporal_graph_v9":{"heldout_v9_test_used_for_selection":False}}))
    with pytest.raises(ValueError,match="hash mismatch"):
        derive_from_v9(src,tmp_path/"new")
