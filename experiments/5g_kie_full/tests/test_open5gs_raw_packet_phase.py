from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

from audit_open5gs_raw_packet_phase import (
    max_count_in_sliding_window,max_count_in_fixed_bins,
    count_trial,audit_one_artifact,summarize,RUNNERS,
)


def test_fixed_bin_aliasing_is_measurable_without_changing_detector():
    origin=100.
    # Twenty NRF GETs all in one 0.6s short burst but straddling the
    # global 2s-bin boundary at ts=102; old 10/s rate cannot trigger.
    ts=np.concatenate([np.linspace(101.6,101.99,10),
                       np.linspace(102.01,102.4,10)])
    result=count_trial(ts,origin,101.5,102.5)
    assert result["raw_request_count"]==20
    assert result["peak_fixed_2s_count"]==10
    assert result["peak_sliding_2s_count"]==20
    assert result["fixed_2s_threshold_20"] is False
    assert result["sliding_2s_threshold_20"] is True
    assert result["sliding_1s_threshold_10"] is True


def test_true_lack_of_passive_capture_remains_missing():
    m=count_trial(np.array([100.1, 101.1]),100.,100.0,102.)
    assert m["raw_request_count"]==2
    assert m["raw_5_get_coverage"] is False
    assert m["fixed_2s_threshold_20"] is False


def test_raw_burst_not_granted_to_nearby_trial():
    ts=np.array([100.02,100.03,100.04,100.05,100.06,102.2,102.3])
    a=count_trial(ts,100.,100.,101.)
    b=count_trial(ts,100.,102.,103.)
    assert a["raw_request_count"]==5
    assert b["raw_request_count"]==2
    assert a["raw_5_get_coverage"] is True
    assert b["raw_5_get_coverage"] is False


def test_empty_array_and_invalid_window():
    assert max_count_in_sliding_window(np.array([]),1.)==0
    assert max_count_in_fixed_bins(np.array([]),0.,2.)==0
    with pytest.raises(ValueError,match="positive"):
        max_count_in_sliding_window(np.array([1.]),0.)
    with pytest.raises(ValueError,match="invalid"):
        count_trial(np.array([1.]),0.,2.,1.)


def test_deduplicated_raw_http2_requests(tmp_path:Path):
    from audit_open5gs_raw_packet_phase import _collected_requests
    ts=tmp_path/"raw.tsv"
    mapfile=tmp_path/"endpoint.json"
    mapfile.write_text(json.dumps({"endpoints":[{"nf":"nrf","address":"127.0.0.10"}]}))
    columns=["frame.time_epoch","ip.src","ip.dst","tcp.stream",
             "tcp.flags.reset","tcp.len","http2.streamid","http2.type",
             "http2.header.name","http2.header.value"]
    rows=[
        ["100.01","127.0.0.1","127.0.0.10","1","0","60","3","1",
         ":method,:path","GET,/nnrf-nfm/v1/nf-instances"],
        ["100.02","127.0.0.1","127.0.0.10","1","0","60","3","1",
         ":method,:path","GET,/nnrf-nfm/v1/nf-instances"],
        ["100.03","127.0.0.1","127.0.0.10","2","0","60","3","1",
         ":method,:path","GET,/nnrf-nfm/v1/nf-instances"],
        ["100.04","127.0.0.1","127.0.0.11","3","0","60","3","1",
         ":method,:path","GET,/nnrf-nfm/v1/nf-instances"],
    ]
    pd.DataFrame(rows,columns=columns).to_csv(ts,sep="\t",index=False)
    out=_collected_requests(ts,mapfile)
    assert len(out)==2
    assert out.ts.tolist()==[100.01,100.03]


def test_missing_replication_artifacts_fail_closed(tmp_path):
    with pytest.raises(ValueError,match="missing or duplicate runner"):
        summarize(tmp_path,tmp_path/"out",runners={31011:1.})


def test_four_independent_environment_seeds_per_cadence():
    from collections import Counter
    assert len(RUNNERS)==12
    assert Counter(RUNNERS.values())=={1.:4,2.:4,4.:4}
