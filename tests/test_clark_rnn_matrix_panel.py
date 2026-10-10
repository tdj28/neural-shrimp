import importlib.util
from pathlib import Path
import numpy as np
import pytest
S=importlib.util.spec_from_file_location('panel',Path(__file__).parents[1]/'tools/explore_clark_rnn_matrix_panel.py');M=importlib.util.module_from_spec(S);S.loader.exec_module(M)
def fixture(n):
 p=np.zeros((n,n,2),dtype=int);growth=np.ones((32,n*n*2));collapse=np.zeros(n*n*2,dtype=int);grid=np.ones((n*n*2,2));return p,growth,collapse,grid
def test_component_margin_and_both_starts_retained():
 p,g,c,x=fixture(7);p[3,3]=7;p[0,0]=5
 cs,ind=M.components(p.ravel(),g,c,x,7)
 assert len(cs)==2 and sum(q['count'] for q in cs)==2
 center=next(q for q in cs if q['period']==7)
 assert center['grid_edge_margin']==3 and center['halo_both_start_positive_growth']==24
 assert set(ind)=={0,1,48,49}
def test_disagreement_and_collapsed_growth_are_not_context():
 p,g,c,x=fixture(5);p[2,2]=[7,5];c[:]=1
 cs,ind=M.components(p.ravel(),g,c,x,5);assert cs==[] and len(ind)==0
 p[2,2]=7;cs,_=M.components(p.ravel(),g,c,x,5);assert cs[0]['halo_any_start_positive_growth']==0
def test_fixed_ranking_prefers_margin_before_large_edge():
 p,g,c,x=fixture(7);p[0,:,0]=p[0,:,1]=3;p[3,3]=9
 cs,_=M.components(p.ravel(),g,c,x,7)
 assert len(cs)==2 and all(q['selected_for_native_qualification'] for q in cs)
def test_projection_counts_all_six_matrices_and_target_batch():
 z=M.projection({'batch_seconds':1.,'period64_unit_seconds':.001,'peak_rss_kib':50000})
 assert z['projected_seconds']==pytest.approx(1.8*6*961/81+192*8*.001+20)
 assert z['conservative_raw_bytes']<256*1024**2
@pytest.mark.parametrize('v',[float('nan'),float('inf'),0,-1])
def test_nonfinite_resource_rejected(v):
 with pytest.raises(ValueError):M.projection({'batch_seconds':v,'period64_unit_seconds':.001,'peak_rss_kib':50000})
