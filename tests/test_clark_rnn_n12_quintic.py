import importlib.util,subprocess,sys
from pathlib import Path
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[1]
s=importlib.util.spec_from_file_location('quintic_driver',ROOT/'tools/explore_clark_rnn_n12_quintic.py');Q=importlib.util.module_from_spec(s);s.loader.exec_module(Q)

def test_fixed_nine_gain_geometry():
 t=Q.targets(4.075278242797678,4.075278245521262);by=dict(t);c=by['center'];assert len(t)==len(by)==9 and by['minus_h']==c-Q.H and by['plus_half']==c+Q.H/2
 assert by['fold_near']==4.0756 and by['fold_far']==4.08
 with pytest.raises(ValueError):Q.targets(4.,4.01)

def test_physical_unfolding_sign_and_negative_gate():
 t=Q.targets(4.075278242797678,4.075278245521262);c=t[0][1];points=[dict(tag=k,normal_cubic=-.3*(g-c),quintic=.2,alpha_delta=2.)for k,g in t];r=Q.unfolding(points);assert r['candidate']and abs(r['physical_determinant']-.6)<1e-10
 points[3]['normal_cubic']+=.02;assert not Q.unfolding(points)['candidate']

def test_asymptotic_fold_comparison_explicit_coefficients():
 row=dict(left=np.eye(12)[0],normal_cubic=-.02,quintic=.5,alpha_delta=2.,delta=.8);u=np.sqrt(.02);pair=np.r_[np.eye(12)[0]*u,-np.eye(12)[0]*u,.8001];r=Q.fold_prediction(row,pair)
 assert abs(r['predicted_squared_half_amplitude']-.02)<1e-15 and abs(r['predicted_leak_gap']-.0001)<1e-15 and abs(r['leak_ratio']-1)<1e-12

def test_caps_cover_fixed_worst_case_and_reject_next_work(tmp_path):
 w=Q.Work(tmp_path);w.augmented_calls=Q.CAPS['half'];w.path5_calls=Q.CAPS['path5'];w.solve_calls=Q.CAPS['solve']
 with pytest.raises(ValueError):w.charge(1)
 with pytest.raises(ValueError):w.path(np.zeros((6,12)),np.eye(12),1.,1.,'bad')
 with pytest.raises(ValueError):w.charge_solve()
 assert 17*12+9+20<=Q.CAPS['half']and 9*5+2*5<=Q.CAPS['path5']and 17*11+16+9+36+18<=Q.CAPS['solve']

def test_entrypoint_source_paths_and_small_launcher():
 r=subprocess.run([sys.executable,str(ROOT/'tools/explore_clark_rnn_n12_quintic.py'),'--help'],text=True,capture_output=True,timeout=10);assert r.returncode==0,r.stderr
 assert all((ROOT/n).exists()for n in Q.EXTRA)
 s=(ROOT/'tools/launch_clark_rnn_n12_quintic.py').read_text();assert 'tools/explore_clark_rnn_n12_quintic.py'in s and '"external_seconds":75'in s and 'science_cap=64*1024**2'in s


def test_literal_independent_receipt_schema_and_manifest_mutation():
 import json,copy
 a=json.loads((ROOT/'docs/reviews/clark-rnn-n12-organizer-connection-001.json').read_text())['independent_saved_audit'];Q.audit_binding(a)
 b=copy.deepcopy(a);b['worker']['manifest_sha256']='0'*64
 with pytest.raises(ValueError):Q.audit_binding(b)
