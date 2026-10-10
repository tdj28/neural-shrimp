#!/usr/bin/env python3
"""Complete only the two capped native p12 arms inside the unchanged envelope."""
import argparse,importlib.util,json,os,resource,signal,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('compact_parent',ROOT/'tools/continue_clark_rnn_compact_p12.py');P=importlib.util.module_from_spec(spec);spec.loader.exec_module(P)
S=P.S;R=P.R;B=P.B;Work=P.Work;COL=P.COL;ROW=P.ROW;in_domain=P.in_domain
RAW=R.RAW/'compact-p12-boundary-curves-001'
PIN='af1817405d90499cde73d88777a00fcf8076a2f80bc4e8ce50e23d94084f91dd'
SEEDS=[('fold',-1,'g16-dir-1-fold',1),('fold',1,'g16-dir-1-fold',1),('flip',1,'flip:1:192',-1)]
SLICE=R.RAW/'compact-p12-boundary-slices-001';SLICE_PIN='b5a8e4ee85008dfb84b29032c42718b7df42924829e8c7ee2d72af92e12bd6ad'
EXTRA=P.EXTRA+['tools/extend_clark_rnn_compact_p12.py','tools/launch_clark_rnn_compact_p12_extension.py','tests/test_clark_rnn_compact_p12_extension.py','docs/plans/clark-rnn-compact-p12-extension.md']

def endpoint_paths(raw,branch,direction,tag):
    result=json.loads((raw/'run/result.json').read_text())
    rows=[t for t in result['traces'] if t['branch']==branch and t['direction']==direction]
    if len(rows)!=1 or rows[0]['stop']!='node_cap' or rows[0]['nodes'][-1]['tag']!=tag:raise ValueError('fixed capped endpoint')
    paths=[p for p in (raw/'run').glob('*-node.json') if json.loads(p.read_text())['tag']==tag]
    if len(paths)!=1:raise ValueError('unique endpoint record')
    return paths[0]

def associated_fold_path(parent,slices,tag):
    result=json.loads((parent/'run/result.json').read_text());rows=[r for r in result['fold_associations']if r['tag']==tag]
    if len(rows)!=1 or not rows[0]['passed']or len(rows[0]['levels'])!=8 or not rows[0]['root_phase_match']['passed']or rows[0]['root_reverse_error']is None or rows[0]['root_reverse_error']>1e-7:raise ValueError('finer fold association required')
    paths=[p for p in(slices/'run').glob('*-boundary_result.json')if json.loads(p.read_text())['tag']==tag]
    if len(paths)!=1:raise ValueError('unique retained fold')
    m=json.loads(paths[0].read_text())
    if m['residual']>1e-8 or m['critical_error']>1e-7 or m['other_distance']<1e-4 or abs(m['fold_a_delta'])<1e-5 or abs(m['fold_c'])<1e-5 or min(m['proper_divisor_separations'].values())<=1e-6:raise ValueError('retained local fold gates')
    return paths[0]

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);args=ap.parse_args();identity=S.source();identity['sha256'].update({p:S.sha(ROOT/p) for p in EXTRA})
    if os.getpriority(os.PRIO_PROCESS,0)<10 or any(os.environ.get(k)!='1' for k in R.M.THREADS):raise ValueError('CPU1 nice10')
    args.out.mkdir(parents=True,exist_ok=False);S.ACTIVE_ROOT=args.out;S.RAW_CAP=256*1024**2;w=Work(args.out);signal.signal(signal.SIGTERM,lambda *_:setattr(w,'interrupted',True))
    # Sequential immutable record names: reserve a conservative upper byte count,
    # avoiding an O(number-of-files²) walk while still prechecking every write.
    booked=[0]
    def reserve(path,size,final=False):
        old=Path(path).stat().st_size if Path(path).exists() else 0;new=booked[0]-old+size
        if new>(S.RAW_CAP if final else S.RAW_CAP-S.FINAL_RESERVE):raise OSError('prewrite reserved byte cap')
        booked[0]=new
    S.reserve=reserve
    available=int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))*1024
    S.save(args.out/'runtime-start.json',{'source':identity,'pid':os.getpid(),'started_unix':time.time(),'python':sys.version,'numpy':np.__version__,'nice':os.getpriority(os.PRIO_PROCESS,0),'threads':{k:os.environ.get(k) for k in R.M.THREADS},'mem_available_bytes':available,'row_scale':ROW.tolist(),'column_scale':COL.tolist()})
    result={'status':'incomplete','traces':[]}
    try:
        if available<1024**3:raise MemoryError('host headroom')
        tj=np.eye(4)*.2+np.ones((4,4))*.1;ty=np.r_[np.arange(1,5)/10,2.,.5,[1.,0,0,0]]/COL;times=[]
        for k in range(10):times.append(w.full(ty,tj,1,'synthetic_profile')[3])
        projected=9700*max(times)+20;S.save(args.out/'profile.json',{'times':times,'projected_seconds':projected,'gate':np.isfinite(projected) and projected<=120,'target_unopened':True})
        if not np.isfinite(projected) or projected>120:raise RuntimeError('resource screen')
        for name,pin in {RAW.name:PIN,SLICE.name:SLICE_PIN}.items():
            raw=R.RAW/name
            if S.sha(raw/'manifest.json')!=pin:raise ValueError('manifest pin')
            inv=json.loads((raw/'manifest.json').read_text())['files']
            if set(inv)!={str(p.relative_to(raw)) for p in raw.rglob('*') if p.is_file() and p!=raw/'manifest.json'}:raise ValueError('input inventory')
            for name,d in inv.items():
                p=raw/name
                if Path(name).is_absolute() or '..' in Path(name).parts or p.is_symlink() or S.sha(p)!=d['sha256'] or p.stat().st_size!=d['bytes']:raise ValueError('input bytes')
        binding=json.loads((RAW/'run/input-binding.json').read_text());ref=Path(binding['source_input'])
        if S.sha(ref)!=binding['source_input_sha256']:raise ValueError('matrix identity')
        with np.load(ref,allow_pickle=False) as f:j=f['j']
        S.save(args.out/'input-binding.json',{'parent_manifest':PIN,'slice_manifest':SLICE_PIN,'source_input':str(ref),'source_input_sha256':S.sha(ref),'seeds':SEEDS})
        for branch,direction,tag,sign in SEEDS:
            if branch=='fold':
                p=associated_fold_path(RAW,SLICE,tag);meta=json.loads(p.read_text());arr=p.with_name(meta['array_file'])
                if S.sha(arr)!=meta['array_sha256']:raise ValueError('fold array identity')
                with np.load(arr,allow_pickle=False)as f:cur=np.r_[f['theta'],meta['gain'],meta['delta'],f['right_vector']]/COL
                h,b,a,_=w.full(cur,j,sign,tag+'-associated-fold');tan=direction*w.tangent(b,None,tag+'-associated-fold')
            else:
                p=endpoint_paths(RAW,branch,direction,tag);meta=json.loads(p.read_text());arr=p.with_name(meta['array_file'])
                if S.sha(arr)!=meta['array_sha256'] or not meta['qualified_primitive_simple_boundary']:raise ValueError('endpoint identity')
                with np.load(arr,allow_pickle=False) as f:cur=f['y'].copy();tan=f['tangent'].copy()
                h,b,a,_=w.full(cur,j,sign,tag+'-retained-endpoint')
            if not np.isfinite(cur).all() or not np.isfinite(tan).all() or abs(tan@tan-1)>1e-10:raise ValueError('finite unit endpoint tangent')
            if not np.isfinite(h).all() or np.max(np.abs(h))>1e-9:raise ValueError('endpoint residual')
            trace={'branch':branch,'direction':direction,'starting_tag':tag,'source_node_sha256':S.sha(p),'source_array_sha256':S.sha(arr),'step':.01,'nodes':[],'stop':'node_cap'}
            result['traces'].append(trace);oldcycle=a['cycle']
            try:
                trace['nodes'].append(w.node(cur,b,a,tan,sign,f'{branch}:{direction}:extension:seed'))
                for k in range(1,257 if branch=='fold'else 385):
                    pred=cur+.01*tan;tag2=f'{branch}:{direction}:extension:{k}'
                    if not in_domain(pred):trace['stop']='domain_edge';break
                    c,status=w.correct(pred,pred,tan,j,sign,tag2)
                    if c is None:trace['stop']='forward_'+status;break
                    nxt,nb,na=c;nt=w.tangent(nb,tan,tag2)
                    pm=B.phase_match(oldcycle,na['cycle']);w.emit('phase_sign_match',dict(pm,tag=tag2,parent_cycle=oldcycle,child_cycle=na['cycle']))
                    if not pm['passed']:trace['stop']='phase_sign_identity';break
                    reverse,rs=w.correct(nxt-.01*nt,cur,tan,j,sign,tag2+'-reverse')
                    if reverse is None:trace['stop']='reverse_'+rs;break
                    mismatch=float(np.max(np.abs(reverse[0]-cur)));w.emit('reverse_check',{'tag':tag2,'old':cur,'reverse':reverse[0],'mismatch':mismatch})
                    if not np.isfinite(mismatch) or mismatch>1e-7:trace['stop']='reverse_mismatch';break
                    nr=w.node(nxt,nb,na,nt,sign,tag2);nr['reverse_mismatch']=mismatch;nr['phase_sign_match']=pm;trace['nodes'].append(nr);cur=nxt;tan=nt;oldcycle=na['cycle']
                    if k%8==0:S.save(args.out/'progress.json',result)
            except (ValueError,np.linalg.LinAlgError) as e:
                trace['stop']='qualification_or_arithmetic: '+type(e).__name__+': '+str(e)
            S.save(args.out/'progress.json',result)
        w.check();result['status']='completed'
    except BaseException as e:result.update(status='failed_or_interrupted',error=type(e).__name__+': '+str(e))
    finally:
        result.update(elapsed_seconds=time.monotonic()-w.start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,attempted_family_calls=w.calls,scope='fixed associatedfold two-direction and cappedflip native primitive12 traces inside unchanged envelope with reverse, proper-divisor and phase/odd-sign diagnostics; generic/attracting labels separate. No validated curve/common orbit at projected crossing/whole shrimp/ODE claim');S.save(args.out/'result.json',result,final=True);S.manifest(args.out)
    return 0 if result['status']=='completed' else 2
if __name__=='__main__':sys.exit(main())
