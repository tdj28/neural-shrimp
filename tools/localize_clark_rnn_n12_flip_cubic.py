#!/usr/bin/env python3
"""Localize only the fixed same-native-phase cubic bracket from retained branch001."""
import argparse
import importlib.util
import json
import os
import resource
import signal
import sys
import time
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('flip_branch',ROOT/'tools/explore_clark_rnn_n12_flip_branch.py')
B=importlib.util.module_from_spec(spec);spec.loader.exec_module(B)
S,R,M=B.S,B.R,B.M
PIN='b4d342b35228940943e509b1871c2637f861ca377cd82433965559979f64d32d'
TAGS=('point01','point02')
ITERATIONS=24
EXTRA=B.EXTRA+['tools/localize_clark_rnn_n12_flip_cubic.py','tools/launch_clark_rnn_n12_flip_cubic.py','tests/test_clark_rnn_n12_flip_cubic.py','docs/plans/clark-rnn-n12-flip-cubic.md']
need=B.need

def split_bracket(left,right,middle):
    need(left['gain']<middle['gain']<right['gain'] and left['cubic']>0>right['cubic'], 'strict ordered signed bracket')
    c=middle['cubic'];need(np.isfinite(c) and c!=0, 'finite nonzero numerical midpoint; zero needs separate error analysis')
    return (middle,right) if c>0 else (left,middle)

def inputs(parent):
    need(S.sha(parent/'manifest.json')==PIN,'fixed retained branch manifest')
    inv=json.loads((parent/'manifest.json').read_text())['files']
    need(set(inv)=={str(p.relative_to(parent))for p in parent.rglob('*')if p.is_file()and p!=parent/'manifest.json'},'complete retained branch inventory')
    for name,d in inv.items():
        p=parent/name
        need(not Path(name).is_absolute()and '..'not in Path(name).parts and not p.is_symlink() and p.stat().st_size==d['bytes']and S.sha(p)==d['sha256']and not p.stat().st_mode&0o222,'immutable input bytes')
    old=json.loads((parent/'run/result.json').read_text())
    need(old['status']=='failed_or_interrupted'and old['error']=='ValueError: independent saved native endpoint join'and not old['endpoint_join']['passed'], 'retain exact original endpoint failure')
    need(len(old['points'])==33 and len(old['associations'])==32 and all(x['passed']and all(x['gates'].values())for x in old['points']), 'all saved numerical points qualified')
    need(all(x['index']==i and x['forward']['passed']and x['reverse']['passed']and x['reverse_native_passed']and x['reverse_q_error']<=1e-7 for i,x in enumerate(old['associations'],1)), 'complete retained native forward/reverse joins')
    binding=json.loads((parent/'run/input-binding.json').read_text());inp=R.RAW/'n12-p6-refinement-001/run/inputs.npz'
    need(S.sha(inp)==binding['matrix_input_sha256'],'same native matrix')
    with np.load(inp)as a:j=a['j'].copy()
    seeds=[]
    for tag in TAGS:
        paths=[p for p in(parent/'run').glob('*branch_point.json')if json.loads(p.read_text())['tag']==tag]
        need(len(paths)==1,'unique fixed local endpoint');p=paths[0];meta=json.loads(p.read_text());arr=p.with_name(meta['array_file'])
        need(meta['passed']and all(meta['gates'].values())and S.sha(arr)==meta['array_sha256'],'qualified local endpoint array')
        with np.load(arr)as a:seeds.append((meta,a['physical'].copy(),a['cycle'].copy()))
    need(seeds[0][0]['cubic']>0>seeds[1][0]['cubic'],'fixed opposite cubic signs')
    return j,seeds,binding

def profile(w,booked,out):
    j=np.eye(12)*.2+np.ones((12,12))*.025;x,q=np.arange(1,13)/20,np.eye(12)[0]
    times={k:[]for k in('native','half','eigen','solve','third')};sizes={k:[]for k in times}
    def measured(kind,f):
        before,t=booked[0],time.monotonic();v=f();times[kind].append(time.monotonic()-t);sizes[kind].append(booked[0]-before);return v
    for i in range(10):
        ret=measured('native',lambda:w.it(x,j,2.,.8,'synthetic_profile'))
        half=measured('half',lambda:w.half(np.r_[x,.8,q],j,2.,-1,'synthetic_profile'))
        def eig():
            e,v=np.linalg.eig(ret[1]);w.emit('profile_eigen',{'matrix':ret[1],'eigenvalues':e,'eigenvectors':v})
        def solve():
            matrix,rhs=half[1],-half[0];sol=np.linalg.solve(matrix,rhs);w.emit('profile_solve',{'matrix':matrix,'rhs':rhs,'solution':sol})
        measured('eigen',eig);measured('solve',solve);measured('third',lambda:w.third(x,q,j,2.,.8,'synthetic_profile'))
    t,b=({k:max(v)for k,v in a.items()}for a in(times,sizes))
    seconds=2*(256*(t['native']+4*t['eigen'])+1024*(t['half']+3*t['solve'])+43*t['third'])+15
    raw=256*(2*b['native']+4*b['eigen']+4096)+1024*(b['half']+3*b['solve']+4096)+43*(b['third']+4096)+8*1024**2
    peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024+128*1024**2
    gate=all(np.isfinite(v)and v>0 for a in times.values()for v in a)and seconds<=120 and raw<=128*1024**2 and peak<=448*1024**2
    S.save(out/'profile.json',{'times':times,'bytes':sizes,'projected_seconds':seconds,'projected_raw_bytes':raw,'projected_peak_bytes':peak,'gate':bool(gate),'target_unopened':True})
    need(gate,'actual measured admission')

def main(out):
    identity=S.source();identity['sha256'].update({p:S.sha(ROOT/p)for p in EXTRA})
    need(os.getpriority(os.PRIO_PROCESS,0)>=10 and all(os.environ.get(k)=='1'for k in R.M.THREADS),'CPU1 nice10')
    resource.setrlimit(resource.RLIMIT_AS,(1024**3,)*2);out.mkdir(parents=True,exist_ok=False);S.ACTIVE_ROOT,S.RAW_CAP=out,128*1024**2
    w,booked=B.Work(out),[0];signal.signal(signal.SIGTERM,lambda *_:setattr(w,'interrupted',True))
    def reserve(path,size,final=False):
        old=Path(path).stat().st_size if Path(path).exists()else 0;new=booked[0]-old+size
        if new>(S.RAW_CAP if final else S.RAW_CAP-S.FINAL_RESERVE):raise OSError('prewrite raw cap')
        booked[0]=new
    S.reserve=reserve
    S.save(out/'runtime-start.json',{'source':identity,'pid':os.getpid(),'nice':os.getpriority(os.PRIO_PROCESS,0),'threads':{k:os.environ[k]for k in R.M.THREADS},'started_unix':time.time(),'python':sys.version,'numpy':np.__version__})
    result={'status':'incomplete','endpoints':[],'midpoints':[],'associations':[]}
    try:
        need(int(next(x.split()[1]for x in Path('/proc/meminfo').read_text().splitlines()if x.startswith('MemAvailable:')))*1024>=1024**3,'1GiB headroom')
        profile(w,booked,out)
        j,seeds,binding=inputs(R.RAW/'n12-p6-flip-branch-001')
        S.save(out/'input-binding.json',{'parent_manifest':PIN,'matrix_input_sha256':binding['matrix_input_sha256'],'fixed_tags':TAGS,'iterations':ITERATIONS,'parent_endpoint_failure_retained':True})
        nodes=[]
        for i,(meta,seed,cycle)in enumerate(seeds):
            tag=f'endpoint{i}';z,h,b,a=w.correct_critical(seed,seed,j,meta['gain'],tag);row,native=w.qualify(z,h,b,a,j,meta['gain'],tag)
            join=B.native_join(cycle,native['cycle'],seed[12],z[12]);need(join['passed'],'original same-native bracket endpoint')
            result['endpoints'].append(dict(row,saved_join=join));nodes.append({'row':row,'z':z,'jac':b,'a':a,'native':native})
        left,right=nodes
        need(left['row']['cubic']>0>right['row']['cubic'],'recomputed strict signs')
        S.save(out/'progress.json',result)
        for i in range(ITERATIONS):
            g=(left['row']['gain']+right['row']['gain'])/2;tag=f'midpoint{i:02d}'
            guess=w.predict(left['z'],left['jac'],left['a'],g-left['row']['gain'],tag)
            z,h,b,a=w.correct_critical(guess,left['z'],j,g,tag);row,native=w.qualify(z,h,b,a,j,g,tag)
            pending={'point':row,'joins':[]};result['pending']=pending;S.save(out/'progress.json',result)
            for side,anchor in [('left',left),('right',right)]:
                forward=M.phase_match(anchor['native']['cycle'],native['cycle']);need(forward['passed'],'native '+side+' phase association')
                reverse_guess=w.predict(z,b,a,anchor['row']['gain']-g,tag+'-'+side+'-reverse')
                rz,rh,rb,ra=w.correct_critical(reverse_guess,z,j,anchor['row']['gain'],tag+'-'+side+'-reverse')
                rn=w.native(rz[:12],j,anchor['row']['gain'],float(rz[12]),tag+'-'+side+'-reverse-native')
                reverse=B.native_join(anchor['native']['cycle'],rn['cycle'],anchor['z'][12],rz[12]);qe=float(np.linalg.norm(anchor['z'][13:]-rz[13:]))
                association={'side':side,'anchor_gain':anchor['row']['gain'],'midpoint_gain':g,'forward':forward,'reverse':reverse,'reverse_q_error':qe,'reverse_native_passed':rn['passed']}
                w.emit('bracket_association',dict(association,anchor_physical=anchor['z'],midpoint_physical=z,reverse_physical=rz,anchor_cycle=anchor['native']['cycle'],midpoint_cycle=native['cycle'],reverse_cycle=rn['cycle']))
                pending['joins'].append(association);S.save(out/'progress.json',result)
                need(reverse['passed']and rn['passed']and qe<=1e-7,'native '+side+' reverse join')
            mid={'row':row,'z':z,'jac':b,'a':a,'native':native}
            newleft,newright=split_bracket(left['row'],right['row'],row)
            left=mid if newleft is row else left;right=mid if newright is row else right
            result['midpoints'].append(row);result['associations'].append(pending['joins']);result.pop('pending')
            result['bracket']=[left['row'],right['row']];S.save(out/'progress.json',result)
        need(len(result['midpoints'])==24 and len(result['associations'])==24 and left['row']['cubic']>0>right['row']['cubic'],'complete strict numerical bracket')
        result['gain_width']=right['row']['gain']-left['row']['gain'];need(result['gain_width']<=3e-9,'fixed resolution attained')
        w.check();result['status']='completed_numerical_cubic_bracket'
    except BaseException as e:result.update(status='failed_or_interrupted',error=type(e).__name__+': '+str(e))
    finally:
        result.update(elapsed_seconds=time.monotonic()-w.start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,native_calls=w.native_calls,half_calls=w.augmented_calls,third_calls=w.third_calls,total_calls=w.calls,
                      scope='Numerical same-native signed-half cubic sign bracket only. No exact zero, generalized-flip nondegeneracy/quintic proof, interval connection, native12/24 daughter, full shrimp or utility claim.')
        S.save(out/'result.json',result,final=True);S.manifest(out)
    return 0 if result['status']=='completed_numerical_cubic_bracket'else 2
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);sys.exit(main(p.parse_args().out))
