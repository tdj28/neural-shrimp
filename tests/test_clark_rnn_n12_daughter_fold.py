import importlib.util
from pathlib import Path
import numpy as np
import pytest

def module():
 p=Path(__file__).parents[1]/'tools/qualify_clark_rnn_n12_daughter_fold.py';s=importlib.util.spec_from_file_location('daughter_fold',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def test_full_native_augmented_profile_has_nonsingular_solve():
 m=module();j=np.eye(12)*.2+np.ones((12,12))*.025;x=np.arange(1,13)/20;h,b,a=m.M.augmented(np.r_[x,.8,np.eye(12)[0]],j,2.,6,1);v=np.linalg.solve(b,-h);assert max(abs(b@v+h))<1e-8

def test_native_normal_fold_coefficients_and_degenerate_rejection():
 m=module();q=np.eye(12)[0];a=dict(monodromy=np.diag([1]+[.2]*11),first=np.zeros((12,14)),second=np.zeros((12,14,14)));a['first'][0,13]=2.;a['second'][0,0,0]=6
 report,w,bqq=m.fold_coefficients(a,q);assert report['passed']and report['parameter']==2 and report['quadratic']==6
 a['second'][0,0,0]=0;assert not m.fold_coefficients(a,q)[0]['passed']

def test_native_augmented_cap_precedes_model(tmp_path):
 m=module();w=m.Work(tmp_path);w.native_aug_calls=32
 with pytest.raises(ValueError,match='32 full-native'):w.aug(np.zeros(25),np.eye(12),2.,'synthetic')
 assert w.calls==0
