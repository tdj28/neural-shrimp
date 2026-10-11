import importlib.util
from pathlib import Path
import numpy as np
import pytest
s=importlib.util.spec_from_file_location('n6p24',Path(__file__).parents[1]/'tools/refine_clark_rnn_n6_p24.py');M=importlib.util.module_from_spec(s);s.loader.exec_module(M)
def test_fixed_selection_both_starts_cap_and_no_other_period():
    p=np.repeat(np.array([24]*100+[48]*100+[3]*20),2)
    selected=M.selection(p)
    assert len(selected)==96 and set(p[selected])=={24,48}
    assert all(selected[i+1]==selected[i]+1 for i in range(0,len(selected),2))
def test_disagreement_excluded():
    assert M.selection(np.array([24,48,24,24])).tolist()==[2,3]
def test_scaling_has_horizon_and_batch():
    r=M.projection({'batch_seconds':1.,'period64_unit_seconds':.001,'peak_rss_kib':50000})
    assert r['projected_seconds']==pytest.approx(2*(2601/81)*(34192/21192)+96*8*.001+20)
    assert r['conservative_raw_bytes']<256*1024**2


def test_single_seed_identity_binds_folder_and_regeneration(tmp_path,monkeypatch):
    assert M.SEED==28184 and M.target_folder(tmp_path)==tmp_path/'run/n6-s28184'
    source=Path(M.__file__).read_text()
    assert 'folder=target_folder(raw)' in source and 'inputs(6,SEED)' in source
    assert "run/n6-s28181" not in source


def test_fixed_component_ranking_and_eligibility():
 def row(i,p,c,margin=2,halo=3):return dict(id=i,period=p,count=c,grid_edge_margin=margin,halo_both_start_positive_growth=halo,selected_for_native_qualification=True)
 rows=[row(67,24,7),row(1,6,20,1),row(2,10,8,2,0),row(73,26,4)]
 assert M.choose_component(rows)['id']==67
 with pytest.raises(ValueError):M.choose_component([row(1,6,20,1)])

def test_launcher_byte_cap(tmp_path):
 import importlib.util
 p=Path(__file__).parents[1]/'tools/launch_clark_rnn_n6_p24.py';s=importlib.util.spec_from_file_location('p24launch',p);a=importlib.util.module_from_spec(s);s.loader.exec_module(a)
 (tmp_path/'process.log').write_bytes(b'x'*5000)
 assert not a.final_byte_gate(tmp_path,{},launcher_cap=1024)['passed']
 assert a.final_byte_gate(tmp_path,{})['passed']
