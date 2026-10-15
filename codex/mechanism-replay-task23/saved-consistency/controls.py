"""Tiny manufactured saved streams only; never loads a model or target result."""
import argparse
import copy
import importlib.util
import json
from pathlib import Path
import time
import numpy as np


def fixture(core):
    n, dimension = 3, 2
    base0 = np.asarray([[1.,0.],[2.,0.],[3.,0.],[4.,0.]])
    base = np.stack([base0.copy() for _ in range(n+1)])
    rng = np.random.default_rng(0)
    directions = np.stack([rng.normal(size=(201,dimension))[core.INDICES] for _ in range(2)])
    norms = np.linalg.norm(base0,axis=1)
    twins = [base0 + directions[k]/np.linalg.norm(directions[k],axis=1)[:,None]*eps*norms[:,None]
             for k,eps in enumerate(core.EPS)]
    states = np.concatenate([base0,*twins])
    dx = np.stack([np.linalg.norm(twin-base0,axis=1) for twin in twins])
    initialization = {'measured_initial_states':states.copy(), 'directions':directions, 'prefix_ids':np.array([1,2])}
    final = {'states':states.copy(), 'tail':base[-n:].copy()}
    growth = np.log((2*dx)/dx)
    net = np.log(((3*dx)/(2*norms))/(dx/norms))
    read = np.log(((2*dx)/norms)/((3*dx)/(2*norms)))
    residual = growth-net-read
    rows = [{'step':i, 'numerically_defined':True,
             'x_norm':norms.tolist(), 'h_norm':(2*norms).tolist(), 'F_norm':norms.tolist(),
             'dx':dx.tolist(), 'dh':(3*dx).tolist(), 'dF':(2*dx).tolist(), 'next_dx':dx.tolist(),
             'log_growth':growth.tolist(), 'log_g_net':net.tolist(), 'log_g_read':read.tolist(),
             'boundary_step':[0.]*4, 'identity_residual':residual.tolist(),
             'entropy':[1.]*4, 'gap':[.5]*4, 'p_top':[.5]*4,
             'top2_logits':[[1.5,1.]]*4, 'top2_ids':[[1,2]]*4} for i in range(n)]
    means = [sum((a.copy() for _ in range(n)),np.zeros_like(a))/n for a in (growth,net,read)]
    summaries = [{'measurement_steps':n,'eps':core.EPS,'numerically_defined':True,
                  'lyapunov':means[0][:,c].tolist(), 'both_twin_mean_log_g_net':means[1][:,c].tolist(),
                  'both_twin_mean_log_g_read':means[2][:,c].tolist(),
                  'mean_log_g_net':float(means[1][0,c]), 'mean_log_g_read':float(means[2][0,c]),
                  'boundary_term':0., 'identity_residual':(means[0]-means[1]-means[2])[:,c].tolist(),
                  'identity_abs_tolerance':1e-10, 'max_step_identity_abs':np.abs(residual[:,c]).tolist(),
                  'nonzero_step_identity_count':((residual[:,c]!=0)*n).tolist(),
                  'mean_entropy':1., 'mean_gap':.5, 'mean_p_top':.5,
                  'frac_gap_below_1':1.,'frac_gap_below_2':1.,'tail_states':n,
                  'lag_distances':[0.,0.], 'threshold':1e-8*norms[c], 'period':1,
                  'literal_chaotic':False} for c in range(4)]
    return [base,initialization,final,rows,summaries,np.ones(4),n]


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',required=True);args=parser.parse_args()
    out=Path(args.out);out.mkdir(parents=False,exist_ok=False)
    start=time.perf_counter()
    spec=importlib.util.spec_from_file_location('own_saved_consistency',Path(__file__).with_name('audit.py'))
    core=importlib.util.module_from_spec(spec);spec.loader.exec_module(core)
    try:
        data=fixture(core)
        original=core.consistency(*data)
        assert original['source_identical_arithmetic_and_state_links_exact']
        assert original['steps']==3 and original['cells']==4 and original['dimension']==2
        checks=['complete tiny known factor-two stream, both twins, period-one tail and means']
        for mutate in (lambda d:d[3].pop(),lambda d:d[3][1].update(step=0),
                       lambda d:d[3][0].update(numerically_defined=False),
                       lambda d:d[3][0]['dx'][0].__setitem__(0,0.)):
            changed=copy.deepcopy(data);mutate(changed)
            try:core.consistency(*changed)
            except ValueError:pass
            else:raise AssertionError('structural refusal missing')
        checks.append('missing/duplicate/undefined/zero-operand stream refuses')
        for mutate in (lambda d:d[4][0].update(mean_entropy=2.),
                       lambda d:d[2]['tail'].__setitem__((0,0,0),9.),
                       lambda d:d[1]['directions'].__setitem__((0,0,0),9.),
                       lambda d:d[3][1]['log_g_net'][0].__setitem__(0,d[3][1]['log_g_net'][0][0]+1e-12)):
            changed=copy.deepcopy(data);mutate(changed)
            mismatch=core.consistency(*changed)
            assert not mismatch['source_identical_arithmetic_and_state_links_exact']
            assert mismatch['all_nonexact_comparisons']
        checks.append('summary/tail/RNG/step tampering retained as exact mismatches')
        ledger=core.Ledger();ledger.norm('manufactured different norm reduction',np.array([[1.,0.]]),[1.+1e-15])
        assert ledger.norm_diagnostics and not ledger.nonexact
        checks.append('norm reduction differences retained without an invented gate')
        changed=copy.deepcopy(data);changed[2]['states']=changed[2]['states'].astype(np.float32)
        try:core.consistency(*changed)
        except ValueError:pass
        else:raise AssertionError('dtype refusal missing')
        checks.append('native float32 substitution refuses')
        temps=(10. ** np.asarray(core.LOG_T,dtype=np.float64)).tolist()
        metadata={'alpha':.0655,'eps':core.EPS,'physical_temperatures':temps,
                  'log10_temperature_hex':[x.hex() for x in core.LOG_T],
                  'physical_temperature_hex':[x.hex() for x in temps]}
        card={'source_indices':core.INDICES,'log10_temperatures':core.LOG_T,'alpha':.0655,'eps':core.EPS}
        summaries=[{'source_index':core.INDICES[c],'log10_temperature':core.LOG_T[c],
                    'physical_temperature':temps[c],'alpha':.0655,'beta':0.,'eps':core.EPS} for c in range(4)]
        assert core.check_contract(metadata,card,summaries)==temps
        for kind in ('metadata','summary'):
            m,c,s=copy.deepcopy([metadata,card,summaries])
            if kind=='metadata':m['physical_temperatures'][1]+=1e-12
            else:s[1]['beta']=.1
            try:core.check_contract(m,c,s)
            except ValueError:pass
            else:raise AssertionError('physical-contract refusal missing')
        for residual in (np.array([1e-9]),np.array([np.nan])):
            try:core.require_summed_identity(residual)
            except ValueError:pass
            else:raise AssertionError('summed accounting refusal missing')
        checks.append('fixed physical metadata and original summed accounting predicate')
        ledger=core.Ledger();a=np.zeros((66,4,896));b=a.copy();b[1,2,3]=1.
        ledger.exact('single large-tail difference',a,b)
        assert ledger.nonexact[0]['mismatch_indices']==[[1,2,3]]
        assert ledger.nonexact[0]['saved']==[1.] and ledger.nonexact[0]['recomputed']==[0.]
        checks.append('every mismatch retained sparsely without duplicating equal native states')
        core.save(out/'fixture.json',original)
        core.save(out/'report.json',{'schema':'task23-saved-consistency-controls-v1',
                                    'all_passed':True,'passed':len(checks),'failed':0,'checks':checks,
                                    'source_sha256':core.sha(core.__file__),
                                    'seconds_before_report':time.perf_counter()-start,
                                    'scope':'manufactured saved data only, no target/model imports or reads'})
    except Exception as error:
        core.save(out/'failure.json',{'type':type(error).__name__,'error':str(error)})
        raise


if __name__=='__main__':main()
