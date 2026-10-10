import importlib.util
from pathlib import Path
import numpy as np
import pytest

def module():
 p=Path(__file__).resolve().parents[1]/'tools/associate_clark_rnn_n6_p7_folds.py'
 s=importlib.util.spec_from_file_location('assoc',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def test_known_scalar_fold_both_signs():
 m=module();a,qs=m.predictors(np.zeros(1),np.ones(1),1.,2.,-.0001,1.)
 assert a==.01
 for q in qs:assert abs(q[0]**2-.0001)<1e-15
 np.testing.assert_array_equal(qs[0],-qs[1])

def test_vector_sign_cannot_change_candidate_set():
 m=module();q=np.array([.6,.8]);a,x=m.predictors(np.zeros(2),q,2.,4.,-.0001,1.);b,y=m.predictors(np.zeros(2),-q,2.,4.,-.0001,1.)
 np.testing.assert_array_equal(x[0],y[1]);np.testing.assert_array_equal(x[1],y[0])

@pytest.mark.parametrize('a,c,change',[(0.,2.,-.001),(1.,0.,-.001),(1.,2.,.001),(np.nan,2.,-.001),(1.,np.inf,-.001)])
def test_nonreal_or_degenerate_is_not_repaired(a,c,change):
 with pytest.raises(ValueError):module().predictors(np.zeros(1),np.ones(1),a,c,change,1.)

def test_association_needs_original_full_native_cycle():
 m=module();p=np.arange(42.).reshape(7,6)
 assert m.associated({'cycle':p.copy()},p)['passed']
 for x in [-p,np.roll(p,1,axis=0),p+.001]:assert not m.associated({'cycle':x},p)['passed']
 assert not m.associated(None,p)['passed']

def test_boundaries_are_exactly_three_and_resource_cap_fixed():
 m=module();assert m.TAGS==['g16-dir-1-fold','g16-dir1-fold','g25-dir1-fold']
 assert '120*max(dur)+5' in Path(m.__file__).read_text()

@pytest.mark.parametrize('field,value',[('passed',False),('best_sign',-1),('best_shift',2),('native_error',np.nan),('native_error',.06)])
def test_original_native_identity_cannot_be_repaired(field,value):
 m=module();good={'passed':True,'best_error':.01,'best_sign':1,'best_shift':0,'native_error':.01};bad=dict(good);bad[field]=value
 with pytest.raises(ValueError):m.original_identity({'native_identity':bad},good)
 with pytest.raises(ValueError):m.original_identity({'native_identity':good},bad)
 m.original_identity({'native_identity':good},good)
