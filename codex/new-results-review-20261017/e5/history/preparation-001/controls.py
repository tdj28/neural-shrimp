"""Manufactured controls only; run once by the admitted saved-data adapter."""
import importlib.util
from pathlib import Path
import math
import numpy as np
P=Path(__file__).resolve().parent/'arithmetic.py'
spec=importlib.util.spec_from_file_location('e5_control_arithmetic',P)
a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)


def require(condition,message):
    if not condition:raise AssertionError(message)


def fixture_loop():
    n=4
    z={k:np.zeros(n,dtype=np.int16) for k in ('period','q_transient','q_end','q_supplementary','retire_q','retire_q_tol','lead_period','cell_prompt','cell_tindex')}
    z.update(label=np.array(['periodic','chaotic','irregular','periodic']),raw_class=np.array(['periodic','hidden chaos','irregular, exponent not positive at both offsets','periodic']),period=np.array([2,0,0,3]),q_end=np.array([2,0,0,-1]),retire_q_tol=np.array([0,0,0,3]),lead_period=np.array([0,1,0,0]),twins=np.array([False,True,True,False]),supplementary=np.zeros(n,bool),retired=np.array([False,False,False,True]),retire_step=np.array([-1,-1,-1,150]),lyapunov=np.array([[np.nan,.01,.001,np.nan],[np.nan,.02,.002,np.nan]]),collapses=np.zeros((2,n),int),entropy_mean=np.ones(n),entropy_sd=np.ones(n),residual=np.array([1e-10,np.nan,np.nan,np.nan]),x_end=np.ones((n,3)),lead=np.stack([np.full(200,7),np.full(200,8),np.arange(200),np.full(200,-1)],axis=1))
    return z,dict(qmax=64,tol=1e-8,lambda_min=.001,retire=True)


def c_periods():
    x=np.stack([np.tile(np.arange(48),6)[:256],np.tile([1,2],128),np.arange(256),np.ones(256,int)])
    require(a.periods(x,96,32).tolist()==[0,2,0,1],'32 period cutoff')
    require(a.periods(x,96,48).tolist()==[48,2,0,1],'48 and primitive period')
    require(a.tails(x,a.periods(x,96,48)).tolist()==[256,256,0,256],'whole sequence tails')
    x[1,0]=99;require(a.tails(x,a.periods(x,96,48))[1]==255,'broken initial token tail')


def c_onsets():
    g=[0.,1.,2.,3.]
    require(a.rep_onset(g,[1,.5,.25,.6])==dict(value=1.,censored=None,analysis_value=1.,index=2,monotone=False),'strict threshold and later recross')
    require(a.rep_onset(g,[.25,.1,.2,0])['analysis_value']==-1,'lower censor')
    require(a.rep_onset(g,[.5,.5,.5,.5])['analysis_value']==4,'upper censor')
    require(a.label_onset(g,['irregular']*4,'chaotic')['censored']=='above','irregular retained')
    require(a.label_onset(g,['chaotic']*4,'chaotic')['censored'] is None,'first grid chaos not lower censored')


def c_ranks():
    require(a.ranks([3,1,1,7]).tolist()==[3,1.5,1.5,4],'average tie ranks')
    require(float(a.spearman([3,1,1,7],[3,1,1,7]))==1,'tied self correlation')
    require(np.isnan(a.spearman([1,1,1],[1,2,3])),'constant undefined')
    require(np.isnan(a.partial([1,2,3],[1,2,3],[1,2,3])),'perfect conditioning undefined')
    require(np.isnan(a.spearman([1,np.nan,3],[1,2,3])),'nonfinite undefined')


def c_bootstrap():
    idx=a.draw_indices(4,5);rng=np.random.default_rng(20261607)
    require(all(np.array_equal(i,rng.integers(0,4,4)) for i in idx),'source iid reset schedule')
    cl=a.draw_indices(4,5,[2,2,5,5])
    require(all(all(list(i).count(j)%2==0 or list(i).count(j)==1 for j in range(4)) for i in cl),'cluster rows retained')
    require(all(list(i).count(0)==list(i).count(1) and list(i).count(2)==list(i).count(3) for i in cl),'whole clusters')
    got=a.interval([0,1,np.nan,3]);require(got['nonfinite_draws']==1 and np.allclose(got['ci'],[.05,2.9]),'finite-only percentile')
    require(a.interval([np.nan])['ci']==[None,None],'undefined interval')


def c_matched():
    f=np.array([[1.,.2],[0.,.4],[.7,.8]])
    l=np.array([['chaotic','periodic'],['periodic','periodic'],['irregular','irregular']])
    val,parts=a.matched(f,l);require(float(val)==1 and parts[-1].tolist()==[.5,0.],'only comparable temperature')
    require(np.isnan(a.matched(f,np.full(l.shape,'irregular'))[0]),'no overlap undefined')
    val2=a.matched(f[np.array([[0,1,2],[0,0,1]])],l[np.array([[0,1,2],[0,0,1]])])[0]
    require(np.allclose(val2,[1,1]),'resampling duplicates both label and fraction')


def c_entropy():
    z=np.array([math.log(3),0.]);target=-(.75*math.log(.75)+.25*math.log(.25))
    r=a.entropy_root(z,target);require(r['defined'] and abs(r['T']-1)<1e-12,'known binary entropy root')
    require(not a.entropy_root(np.array([0.,0.]),.2)['defined'],'unattainable tied logits')
    require(abs(a.entropy(np.array([0.,0.]),3)-math.log(2))<1e-15,'uniform entropy')


def c_gates():
    z,cfg=fixture_loop();r=a.loop_evidence(z,cfg)
    require(r['source_gate_pass'] and not r['classification_mismatches'] and not r['lead_mismatches'],'base source gates')
    z['residual'][0]=1e-8;r=a.loop_evidence(z,cfg);require(not r['source_gate_pass'],'strict residual cutoff')
    z['supplementary'][0]=True;z['q_supplementary'][0]=2;r=a.loop_evidence(z,cfg);require(r['source_gate_pass'],'source supplementary exemption')
    cfg['retire']=False;require(not a.loop_evidence(z,cfg)['source_gate_pass'],'forbidden retirement')
    cfg['retire']=True;z['retired'][0]=False;z['retire_step'][0]=120;z['retire_q_tol'][0]=3
    require(len(a.loop_evidence(z,cfg)['virtual_retirement_mismatches'])==1,'virtual period mismatch')


def c_samples():
    z=dict(tokens=np.zeros((2,3,100),np.int32),mean_entropy=np.zeros((2,3)),prefix=np.array([1,2]),T_H_float32=np.array(1.))
    require(a.sample_evidence(z,(2,3,100),151936)['source_gate_pass'],'sample gate baseline')
    z['tokens'][0,0,0]=151645;z['mean_entropy'][0,0]=np.nan
    r=a.sample_evidence(z,(2,3,100),151936);require(len(r['gate_failures'])==2,'EOS and nonfinite failures retained')


def c_compare():
    r=a.comparison({'a':1.,'extra':[False]}, {'a':1.+1e-12,'missing':None})
    require({x['status'] for x in r}=={'different','own_only','missing_own'},'all comparison dispositions')
    require(next(x for x in r if x['path']=='/a')['signed_difference']<0,'signed discrepancy')


def c_stat_pairing():
    x=np.array([1,2,3,4,5.]);y=np.array([4,1,5,2,3.]);z=np.array([1,3,5,2,4.]);idx=a.draw_indices(5,8)
    b=a.boot_values(lambda i:a.spearman(z[i],y[i]),idx)
    oracle=np.array([float(a.spearman(z[i],y[i])) for i in idx])
    require(np.array_equal(b,oracle,equal_nan=True),'batch arithmetic preserves paired indices')
    require(not np.array_equal(b,np.array([float(a.spearman(z[i],y)) for i in idx]),equal_nan=True),'unpaired bug distinguished')


def checks():
    functions=[c_periods,c_onsets,c_ranks,c_bootstrap,c_matched,c_entropy,c_gates,c_samples,c_compare,c_stat_pairing]
    rows=[]
    for f in functions:
        try:f();rows.append(dict(name=f.__name__,passed=True))
        except Exception as e:rows.append(dict(name=f.__name__,passed=False,type=type(e).__name__,message=str(e)))
    return dict(schema='e5-manufactured-controls-v1',passed=sum(r['passed'] for r in rows),failed=sum(not r['passed'] for r in rows),groups=rows)
