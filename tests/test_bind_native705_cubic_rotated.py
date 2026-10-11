import copy,sys
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'tools/reviews'),str(ROOT/'src')]
import bind_native705_cubic_rotated as C


def test_actual_independent_changed_gauge_join_without_math_or_native(monkeypatch):
    def no(*a,**k):raise AssertionError('no proof arithmetic/native acquisition')
    monkeypatch.setattr(C.G.P,'small_matrix',no);monkeypatch.setattr(C.G,'exact_inverse',no);monkeypatch.setattr(C.G.B,'TwoRateCuspMap',no)
    index=C.read(ROOT/C.M.CARD)
    row,run,ir,rt,domain,inp,proof,ib=C.M.metadata(index,35,lambda:None)
    previous=C.M.select(index,34);authority=dict(index=previous,complete_complement_endpoints=previous['parameter_endpoints'])
    result=C.rotated_join(index,row,ir,proof,ib,inp,authority,lambda:None)
    assert result['known_parameter']is None and len(result['projection'])==2
    left,right=map(C.F,result['projection']);assert C.F(-1,2048)<=left<=right<=C.F(63,2048)
    bad=dict(authority,complete_complement_endpoints=['0','1'])
    with pytest.raises(ValueError,match='preceding chart34'):C.rotated_join(index,row,ir,proof,ib,inp,bad,lambda:None)
    badproof=copy.deepcopy(proof);badproof['membership']['known_parameter']='115/512'
    with pytest.raises(ValueError,match='changed-gauge uniqueness'):C.rotated_join(index,row,ir,badproof,ib,inp,authority,lambda:None)


def test_missing_rotated_artifact_never_supplies_join(tmp_path):
    with pytest.raises(ValueError,match='literal rotated-join'):C.checked_artifact(tmp_path,{},'fresh-endpoint-localization.json')
