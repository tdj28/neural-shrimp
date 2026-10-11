import numpy as np
import pytest
from neural_shrimp.response_time_lines import factor_lines,solve_lines,setup,probe
from neural_shrimp.response_operator_blocks import Operator,ActionBudget
from neural_shrimp.response_spectrum import difference_matrix


def test_exact_free_coordinate_principal_blocks():
    maps=np.array([[[.9,.4],[-.2,1.1]],[[.8,-.3],[.5,.95]]]);op=Operator(maps,8,1000)
    lower,pivot=setup(op);k=difference_matrix(maps,8,-.05,'free');h=(k.T@k).toarray()+.0001*np.eye(18)
    rhs=np.arange(18)/19
    x=solve_lines(lower,pivot,rhs).reshape(9,2)
    for j in range(2):
        ix=np.arange(j,18,2)
        np.testing.assert_allclose(x[:,j],np.linalg.solve(h[np.ix_(ix,ix)],rhs[ix]),rtol=1e-12,atol=1e-11)
    assert op.counts=={'JVP':4,'VJP':0}


def test_scalar_exact_line_converges_one_iteration_and_counts_setup():
    rows=[];out=probe(np.array([[[.87]],[[1.06]]]),10000,lambda k,v:rows.append((k,v)))
    assert out['solve_qualified'] and out['iterations']==1 and out['setup_jvps']==2
    assert out['counts']=={'JVP':66,'VJP':96}
    assert sum(k=='line_setup_column' for k,v in rows)==2
    assert rows[-1][0]=='terminal'


def test_matched_budget_fails_closed_with_terminal_reserve():
    out=probe(np.array([[[.9,.4],[-.2,1.1]],[[.8,-.3],[.5,.95]]]),200)
    assert out['status']=='action_budget_exhausted' and not out['solve_qualified']
    assert sum(out['counts'].values())<=200


def test_line_factor_rejects_nonpositive_and_nonfinite():
    with pytest.raises(ValueError):factor_lines(np.array([[1.],[1.]]),np.array([[2.]]))
    with pytest.raises(ValueError):factor_lines(np.array([[np.nan],[1.]]),np.array([[0.]]))


def test_durable_setup_failure_propagates():
    def fail(k,v):
        if k=='line_setup_column':raise OSError('storage failure')
    with pytest.raises(OSError):probe(np.ones((2,1,1)),10000,fail)


def test_coupled_solve_matches_direct_normal():
    maps=np.array([[[.9,.1],[-.1,.8]],[[.7,-.2],[.15,.85]]]);records=[]
    out=probe(maps,10000,lambda k,v:records.append((k,v)))
    assert out['solve_qualified']
    terminal=next(v for k,v in records if k=='terminal');k=difference_matrix(maps,32,-.05,'free');b=np.zeros(64);b[32]=1
    x=np.linalg.solve((k.T@k).toarray()+.0001*np.eye(66),k.T@b)
    np.testing.assert_allclose(terminal['state'],x,rtol=1e-7,atol=1e-8)
