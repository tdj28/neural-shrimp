import copy,subprocess,sys
from pathlib import Path
import pytest
from flint import arb,arb_mat
from neural_shrimp.rigorous_orbits import ball_json
ROOT=Path(__file__).parents[1];sys.path.insert(0,str(ROOT/'tools'))
import probe_native705_cubic_inverse as W
import launch_native705_cubic_inverse as L

def valid():
 e={'direction_return':180,'pal_path_phase_return':108,'pal_chart_return':1,'gradient_path_input':144,'gradient_path_return':144,'gradient_path_assembly_return':4,'cubic_correlated_chart_return':2,'cubic_correlated_domain':2,'cubic_path_input':144,'cubic_path_return':144,'cubic_path_assembly_return':4,'profile_gradient_input':24,'profile_gradient_return':24,'profile_quartic_input':24,'profile_quartic_return':24}
 out=dict(passed=True,target_acquired=False,ticks=629,events=e,elapsed_seconds=40,peak_rss_kib=1000000,stored_bytes_before_final=1000000,chart_times=[10.,11.],gradient_times=[.4]*12,path_times=[.05]*12)
 out['projection']=W.P.projection(out['chart_times'],out['gradient_times'],out['path_times'],out['peak_rss_kib'],out['stored_bytes_before_final']);return out

def test_complete_new_native_profile_cannot_use_old_quadratic_events():
 assert W.admit_profile(valid())['resource_admitted']
 for k in ['cubic_path_input','cubic_path_return','profile_quartic_return','gradient_path_return']:
  old=valid();old['events'][k]-=1
  with pytest.raises(ValueError):W.admit_profile(old)
 old=valid();old['projection']['one_fresh_chart_seconds']=1
 with pytest.raises(ValueError):W.admit_profile(old)

def test_cubic_proposal_is_only_finite_left_solve_and_all_coordinates_retained():
 C=arb_mat([[1,2],[3,4]]);coeff=[[ball_json(arb(0))for _ in range(4)]for _ in range(2)];coeff[0][3]=ball_json(arb(2));coeff[1][3]=ball_json(arb(3));ticks=[]
 got=W.cubic_proposal(C,coeff,lambda:ticks.append(1));assert got==[arb(-8),arb(-18)]and len(ticks)==1
 with pytest.raises(ValueError):W.cubic_proposal(C,coeff[:1])

def test_fixed_scientific_scales_same_anchor_and_archive_closure():
 assert W.SETTINGS['halfwidth_exponent']==-3 and W.SETTINGS['radius_exponent']==-8 and W.SETTINGS['newton_steps']==0
 assert W.TARGET==L.BASE/'attempt001-target/run'and W.BASE.name=='cubic001'
 assert set(W.P.SOURCES)<=set(W.SOURCES)and all((ROOT/n).is_file()for n in W.SOURCES)
 assert W.C.APINS['endpoint']=='d56ed2992697924b767e16b76e7cd1271384899b039e52e8d416f790318ac9be'
 r=subprocess.run([sys.executable,str(ROOT/'tools/launch_native705_cubic_inverse.py'),'--help'],capture_output=True,text=True);assert r.returncode==0 and '--profile-worker'in r.stdout
