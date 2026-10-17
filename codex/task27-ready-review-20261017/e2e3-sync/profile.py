"""Prepared manufactured-only full-shape measurement. Not yet executed.

CLI --out FRESH_DIR, CPU1. No target path/card, tokenizer or model access.
Runs accepted controls once, then full admitted-shape own arithmetic/serialization.
"""
import argparse
import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import resource
import sys
import time

import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import e2e3 as e
import synchronization as s
import run as adapter
import controls


def manufacture():
    groups=[{'length':81+g,'indices':list(range(2000+16*g,2016+16*g))} for g in range(32)]
    indices=[i for g in groups for i in g['indices']]
    membership={'groups':groups,'indices':indices,'start':2000,'end':None,'split':'train','eligible_lengths':77,
                'sha256_indices':hashlib.sha256(json.dumps(indices).encode()).hexdigest()}
    runs={}
    def one(name,mode,layouts,cap=0.,temperature=.6,perturb=0.):
        args={'layouts':layouts,'temperature':temperature,'gain_cap':cap,'top_n':15 if mode=='soft' else 0,
              'soft_steps':64,'out':name,'mode':mode,'max_new':320,'perturb':perturb}
        rows=[]
        for layout in layouts:
            b=0
            for group in groups:
                for start in range(0,16,layout):
                    for pos,i in enumerate(group['indices'][start:start+layout]):
                        answer='1' if (i+layout+len(name))%5 else '2'
                        n=64 if mode=='soft' else 320
                        row={'index':i,'layout':layout,'batch':b,'row':pos,'prompt_len':group['length'],
                             'answer':answer,'gold':'1','correct':answer=='1','text':'x'*370+' The answer is '+answer,
                             'tokens':[1]*12 if mode=='soft' else [1]*320,'top1':[1]*n,
                             'pmax':[.56789]*n,'entropy':[1.23456]*n,'gap':[.1234]*n}
                        if mode=='soft': row['gain']=[.01234]*64
                        if cap: row['temp']=[.54321]*64
                        if perturb: row['perturb_realized']=.000834567
                        rows.append(row)
                    b+=1
        runs[name]={'id':name,'args':args,'records':rows}
    names=['uncapped','cap0.37','cap0.10','cap0.015','t0.5']
    for name,cap in zip(names,[0.,.37,.1,.015,0.]): one(name,'soft',[8,16],cap,.5 if name=='t0.5' else .6)
    one('soft0','soft',[16]);one('greedy','hard',[16],temperature=0.)
    for i in range(1,9):
        one('soft-k'+str(i),'soft',[16],perturb=.001)
        one('sample-k'+str(i),'hard',[16],temperature=.7)
    assoc={'e2':{'arms':[{'name':n,'run':n} for n in names],
                 'contrast_key_order':{'contrasts_vs_t0.5':[n+' - t0.5' for n in names[1:4]]}},
           'e3':{'metadata':{'runs':{'soft0':'soft0','greedy':'greedy','soft':['soft-k'+str(i) for i in range(1,9)],
                                    'sample':['sample-k'+str(i) for i in range(1,9)]}}}}
    sync={};cells=[{'index':i,'log10_temperature':.1+i*.01,'alpha':.05} for i in range(3)]
    tp=[{'cell':i,'c':c} for i in range(3) for c in [0.,.05,.1,.15,.2,.25,.3]]
    sync['transverse.json']={'cells':cells,'pairs':tp,'commit':{'tools_dirty':True}}
    sync['transverse.npz']={'jvp_logs':np.tile(np.sin(np.arange(3000)/20)*.1+.2,(3,1)),
                            'logs':np.array([np.sin(np.arange(2000)/20)*.1+.2+np.log(abs(1-2*p['c'])) for p in tp])}
    for name,nc,steps in [('sweep-near',31,3000),('sweep-far',16,2000),('sweep-refine',8,10000)]:
        pairs=[{'cell':i,'c':float(c)} for i in range(3) for c in np.linspace(0,.3,nc)]
        # Variable low runs and excursions exercise run extraction and censored-tail treatment.
        phase=np.arange(steps+1)
        base=np.where((phase%97)<80,1e-9,2e-3)
        dist=np.tile(base,(len(pairs),1))
        tok=np.tile((np.arange(steps)%19).astype(np.int32),(2*len(pairs),1))
        final=np.ones((2*len(pairs),1,896))
        final[1::2]+=dist[:,-1,None,None]
        sync[name+'.json']={'pairs':pairs,'steps':steps,'complete':True,
                            'args':{'start':'far' if name=='sweep-far' else 'near','record':steps},
                            'commit':{},'sweep_npz_sha256':'manufactured'}
        sync[name+'.npz']={'dist':dist,'tok':tok,'final':final}
    for name,n in [('periodic',14),('periodic-synced',4)]:
        chosen=sync['sweep-near.json']['pairs'][:n]
        sync[name+'.json']={'args':{'max_period':40,'tol':1e-9},'pairs':[
            {'sweep':'sweep-near',**p,'steps_before':3000,'primitive_period':4,'return_distance_at_P':1e-13,
             'min_return':1e-14,'argmin_return':8,'pair_distance_now':1e-9,'top1_A':[1]*4,'top1_B':[2]*4} for p in chosen]}
    sync['check.json']={'cells':[dict(c,fd_h=[.01,.003,.001,.0003,.0001,.00003,.00001,.000001,.0000001],
                                     fd_rel_err=[1e-2,9e-4,1e-4,9e-6,1e-6,9e-8,1e-8,1e-9,1e-8],
                                     fd_ratio_per_decade=[100.,100.,100.],bitwise={'identical_rows_equal':True},
                                     max_abs_diff_rel={'identical_rows_differ':0.}) for c in cells]}
    return runs,membership,assoc,sync,{0:.2,1:.2,2:.2}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);args=ap.parse_args()
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    start=time.perf_counter()
    try:
        stream=io.StringIO()
        with contextlib.redirect_stdout(stream): status=controls.main()
        control=json.loads(stream.getvalue());adapter.write(out/'controls.json',control)
        if status: raise ValueError('manufactured control failed')
        before=time.perf_counter();runs,membership,assoc,sync,pub=manufacture()
        adapter.write(out/'manufactured-records.json',runs)
        # Measured full-shape JSON parse/ownership, not real-source I/O timing.
        runs=json.loads((out/'manufactured-records.json').read_text())
        fixture_seconds=time.perf_counter()-before
        before=time.perf_counter();result=adapter.calculate(e,s,runs,membership,assoc,sync,pub)
        arithmetic_seconds=time.perf_counter()-before
        before=time.perf_counter();adapter.write(out/'evidence.json',result)
        serialization_seconds=time.perf_counter()-before
        files=[{'path':str(p),'bytes':p.stat().st_size,'sha256':adapter.sha(p)} for p in sorted(out.glob('*.json'))]
        report={'schema':'task27-e2e3-sync-profile-v1','computational_status':'complete','controls':control,
                'shape':{'e2_rows':5120,'e3_rows':9216,'sweep_pair_steps':615000,'transverse_logs':42000,'jvp_logs':9000},
                'fixture_build_write_decode_seconds':fixture_seconds,'arithmetic_seconds':arithmetic_seconds,
                'serialization_seconds':serialization_seconds,'total_before_final_report_seconds':time.perf_counter()-start,
                'observed_peak_kib_before_final_report':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                'files':files,'bytes_before_final_report':sum(p['bytes'] for p in files),
                'limitations':'manufactured strings/numbers; actual hashes/decompression/comparison/native authority remain unmeasured'}
        adapter.write(out/'report.json',report)
    except Exception as exc:
        adapter.write(out/'failure.json',{'computational_status':'failed','error':repr(exc)})
        raise


if __name__=='__main__': main()
