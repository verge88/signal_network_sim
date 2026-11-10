from experiments.accumulation_5g.detector import advance, peer_residuals

def test_plateau_keeps_accumulated_evidence():
    value=0.
    for _ in range(10):value=advance(value,1.,False)
    assert value>4.
    assert 0<advance(value,0.,False)<value
    assert advance(value,1.,True)==0.

def test_common_load_shift_and_isolated_outlier():
    rows=[dict(nf_id=str(i),request_count=10.) for i in range(8)]
    assert set(peer_residuals(rows).values())=={0.}
    rows[-1]['request_count']=15.
    result=peer_residuals(rows)
    assert result['7']==5. and all(result[str(i)]==0. for i in range(7))
