"""Synthetic saved trajectory corruption checks; no target model or evidence."""
import copy
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest

HERE=Path(__file__).resolve().parents[1]/'codex/hostile-fold-crosscheck'
sys.path.insert(0,str(HERE))
import passage_worker as worker
import passage_audit as auditor


def fixture(censor=False):
    f=dict(passed=True,stage='qualified',certificate_passed=False,
        passage_trajectories_evaluated=False,solver_history_complete=True,return_map='G=F^2',
        active_parameter_units='log10T',alpha=.083,gates={'fold':True},
        x_c=np.array([0.,0.]),r=np.array([1.,0.]),l=np.array([1.,2.]),
        parameter_c=-.04,a=-10000.,b=2.,U=.05,ghost_parameter_direction=-1)
    design=worker.design(f);data={};results=[];events=[]
    for row in design['rows']:
        g=0
        def step(state,p):
            nonlocal g
            g+=1;prefix=row['id']+f'/step-{g:04d}-'
            data[prefix+'input.npz']=dict(state=state.copy(),parameter=np.array(p))
            value=state.copy() if censor else state+np.array([f['a']*(p-f['parameter_c'])+f['b']*state[0]**2,0.])
            data[prefix+'output.npz']=dict(state=value.copy())
            events.append(dict(row=row['id'],G=g,F=2*g,seconds=.01))
            return value
        def save(trace):data[row['id']+'/trace.npz']=copy.deepcopy(trace)
        results.append(worker.run_row(step,row,f,save=save))
    status=dict(rows=results,model_G_calls=len(events),primary_all_pass=all(r['primary_pass'] for r in results),
        all_rows_observed=all(r['status']=='observed' for r in results))
    return dict(design=design),status,{k:f[k] for k in ['x_c','r','l']},data,events


def check(parts):
    p,s,m,d,e=parts
    return auditor.numeric(p,s,m,lambda name:copy.deepcopy(d[name]),e,dimension=2)


def test_exact_quadratic_six_rows_pass_independent_check():
    r=check(fixture())
    assert r['audit_passed'] and r['scientific_primary_all_pass'] and len(r['rows'])==6
    assert r['certificate_passed'] is False and r['natural_reinjection_measured'] is False
    assert r['total_F']==2*r['total_G']


def test_faithful_censor_is_completed_audit_not_successful_hypothesis():
    r=check(fixture(censor=True))
    assert r['audit_passed'] and not r['scientific_primary_all_pass'] and not r['all_rows_observed']
    assert all(row['status']=='censored' for row in r['rows'])


@pytest.mark.parametrize('kind',['chain','parameter','u','transverse','count','clock','forecast','aggregate','events','initial','stop','mode'])
def test_independent_corruption_refusals(kind):
    parts=fixture();p,s,m,d,e=parts;label='e0-full'
    if kind=='chain':d[label+'/step-0001-input.npz']['state'][1]=.01
    elif kind=='parameter':d[label+'/step-0001-input.npz']['parameter']=np.array(-.02)
    elif kind=='u':d[label+'/trace.npz']['u'][1]+=.01
    elif kind=='transverse':d[label+'/trace.npz']['transverse'][1]+=.01
    elif kind=='count':s['rows'][0]['measured_G']+=1
    elif kind=='clock':s['rows'][0]['measured_F']+=1
    elif kind=='forecast':p['design']['rows'][0]['quadratic_G']+=1
    elif kind=='aggregate':s['primary_all_pass']=False
    elif kind=='events':e.reverse()
    elif kind=='initial':d[label+'/trace.npz']['initial_states'][0]+=.01
    elif kind=='stop':d[label+'/trace.npz']['states_at_stop'][0]+=.01
    elif kind=='mode':m['l'][1]=3.
    assert not check(parts)['audit_passed']


@pytest.mark.parametrize('kind',['missing','shape','nonfinite','rows'])
def test_hard_missing_or_invalid_input_refused(kind):
    parts=fixture();p,s,m,d,e=parts
    if kind=='missing':del d['e0-full/step-0001-output.npz']
    elif kind=='shape':d['e0-full/step-0001-output.npz']['state']=np.zeros(3)
    elif kind=='nonfinite':d['e0-full/trace.npz']['state_trace'][0,0]=np.nan
    elif kind=='rows':s['rows'].pop()
    with pytest.raises((ValueError,KeyError)):check(parts)


def test_isolated_help_no_model_or_target_import():
    p=subprocess.run([sys.executable,'-I','-B',str(HERE/'passage_audit.py'),'--help'],capture_output=True,text=True,timeout=10)
    assert p.returncode==0 and '--receipt-sha256' in p.stdout
