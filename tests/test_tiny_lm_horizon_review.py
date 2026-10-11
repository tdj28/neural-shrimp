import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parents[1]/'tools'))
import review_tiny_lm_horizon as A
import numpy as np
import pytest

def test_strict_crossing_and_nested_gate_integrity():
 assert A.crossing([0.,.2,.20000000001,1.])==3
 assert A.crossing([0.,.2,.2,.2])is None
 with pytest.raises(ValueError):A.same({'gate':True},{'gate':False})
 with pytest.raises(ValueError):A.same({'gate':False,'missing':0},{'gate':False})

def test_curvature_denominator_and_early_defect():
 u=np.array([1.,0.]);v=np.array([2.,0.]);early=np.array([u,v,2*v,3*v,u,2*u]);p=A.prediction(early,[2.,3.])
 assert p['directional_curvature']==.5 and p['early_defect']==.5
 assert p['scores']['drift']==[0.,.5,1.5,3.]
 assert p['scores']['static_curvature']==[.5,1.,1.5,2.]
 assert p['scores']['baseline_composite']==[1.,2.,3.,4.]

def test_origin_failure_excludes_good_later_responses():
 u=np.ones(1);small=np.ones((5,1));large=small.copy();large[0]=1.1
 p=A.prediction(np.array([u,u,u,u,u,u]),[1.,1.]);e=A.evaluation(p,u,small,small,large,small)
 assert not e['resolved']and not e['future_warning_eligible']and e['observed_crossing']is None

def test_already_failing_is_not_a_future_warning_case():
 u=np.ones(1);s=np.ones((5,1));f=s.copy();f[2]=1.3
 p=A.prediction(np.array([u,u,u,u,u,u]),[1.,1.]);e=A.evaluation(p,u,f,s,s,s)
 assert e['resolved']and e['observed_crossing']==2 and not e['future_warning_eligible']and not e['future_failure']
 f[2]=1.;f[4]=1.3;e=A.evaluation(p,u,f,s,s,s)
 assert e['future_warning_eligible']and e['future_failure']and e['observed_crossing']==4

def test_confusion_tables_include_misses_false_warnings_and_exclusions():
 def row(w,f,eligible=True,resolved=True):return dict(evaluation=dict(future_warning_eligible=eligible,resolved=resolved,future_failure=f,warnings=dict.fromkeys(A.METHODS,w)))
 rows=[row(True,True),row(True,False),row(False,True),row(False,False),row(True,False,False)]+[row(False,False,False,False)for _ in range(27)]
 s=A.summary(rows);assert s['future_eligible_rows']==4 and s['already_failing_rows']==1 and s['resolved_rows']==5
 assert s['future_warning_counts']['drift']==dict(true_warning=1,false_warning=1,missed_failure=1,correct_quiet=1)
 assert s['drift_only_caught_failures']==0

def test_corrupt_native_action_and_nonfinite_norm_rejected():
 with pytest.raises(ValueError):A.exact(np.ones(2),np.array([1.,1.00001]),'corrupt action')
 with pytest.raises(ValueError):A.A.norm(np.array([np.nan]))

def test_audit_has_no_producer_or_native_import():
 import ast
 tree=ast.parse(Path(A.__file__).read_text());names=[n.module for n in ast.walk(tree)if isinstance(n,ast.ImportFrom)]+[a.name for n in ast.walk(tree)if isinstance(n,ast.Import)for a in n.names]
 assert not any(n and ('torch'in n or 'neural_shrimp'in n or 'explore_'in n)for n in names)
