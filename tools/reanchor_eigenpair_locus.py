#!/usr/bin/env python3
"""Bounded fixed-parent known-root localization and two tangent-gauge charts."""
import os
THREADS=('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')
for k in THREADS:os.environ[k]='1'
import argparse,gc,gzip,hashlib,json,math,resource,signal,subprocess,sys,time
from pathlib import Path
from fractions import Fraction
from types import SimpleNamespace
import numpy as np,flint
from flint import arb,arb_mat,ctx
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools')]
import continue_eigenpair_locus_chain as C
M=C.M
from neural_shrimp.rigorous_eigenpair_reanchor import reanchor,localization_identity
from neural_shrimp.rigorous_eigenpair_march import compact
from neural_shrimp.rigorous_eigenpair_locus import require
from neural_shrimp.rigorous_cusps import TwoRateCuspMap
from neural_shrimp.rigorous_orbits import ball_json,matrix_json,vector_json,eye
PARENT=C.BASE/'attempt-002-march-target'
PARENT_SOURCE='692943933a061c718bfb8ccdb8d17c8a5c99feb8'
PARENT_MANIFEST='6984dff87e855a4fa43d036358688ebf536d9feac88bbe1a0fcdc2d0fc9a2715'
PARENT_RUN='f1bb8a6acc3b6b50e53d71ef8fa57be1ea1c9035f3dfbec8d5d6cacc1c8a2cad'
EXTRA=['src/neural_shrimp/rigorous_eigenpair_reanchor.py','tools/reanchor_eigenpair_locus.py','tools/launch_eigenpair_reanchor.py','tests/test_eigenpair_reanchor.py','tests/test_eigenpair_reanchor_adapter.py','docs/plans/eigenpair-tangent-reanchor.md','docs/reviews/eigenpair-march-unit002.json']
SOURCES=list(dict.fromkeys(C.ORIGINAL_SOURCES+C.EXTRA+EXTRA))
entry=M.entry;read=M.read


def cost_gate(parent,profile):
    require(parent['passed'] is True and parent['status']=='four_joined_charts' and parent['ticks']==508,'fixed successful unit002 cost')
    require(M.finite(parent['elapsed_seconds'],300) and M.finite(parent['peak_rss_kib'],2*1024**2) and M.finite(parent['stored_bytes_before_final'],768*1024**2),'finite parent cost')
    require(profile['passed'] is True and profile['native_calls']==0 and profile['synthetic_dimension']==470 and profile['intersections']==3,'synthetic localization profile')
    require(M.finite(profile['elapsed_seconds'],20) and M.finite(profile['peak_rss_kib'],512*1024) and M.finite(profile['stored_bytes_before_final'],31*1024**2),'finite profile caps')
    seconds=2*(parent['elapsed_seconds']*326/508+profile['elapsed_seconds'])+20
    storage=parent['stored_bytes_before_final']+6*profile['stored_bytes_before_final']+8*1024**2
    peak=max(parent['peak_rss_kib'],profile['peak_rss_kib'])+256*1024
    require(seconds<180 and storage<504*1024**2 and peak<2*1024**2,'measured reanchor admission')
    return {'seconds':seconds,'raw_bytes':storage,'rss_kib':peak,'formula':'2x unit002 elapsed per508ticks times326 +2profile +20s; one full4chart unit bytes plus6synthetic localization bundles+8MiB; measuredRSS+256MiB. Proposal only; actual caps mandatory.'}


def bind_parent(args):
    # Reuse all original ancestry/source/runtime checks, explicitly selecting unit002.
    call=SimpleNamespace(mode='target',unit=3,parent=PARENT,parent_source=PARENT_SOURCE,parent_manifest=PARENT_MANIFEST,parent_run_manifest=PARENT_RUN,expected_right='19/32768')
    C.install(call);M.SOURCES=SOURCES
    inp,previous,bindings=M.bind_parent(args.profile,args.profile_manifest,args.profile_outer_manifest)
    po=read(args.profile/'outcome.json');parent_out=read(PARENT/'run/outcome.json')
    bindings['reanchor_cost']=cost_gate(parent_out,po)
    desc=parent_out['progress']['last_accepted']
    with gzip.open(C.input_path(PARENT,desc['full_chart_file']),'rt') as f:full=json.load(f)
    require(compact(full)==previous,'full parent for C/localization')
    require(full['parameter_endpoints'][1]=='19/32768','fixed parent endpoint')
    return inp,full,bindings


def synthetic_profile(observe):
    # Dense coupled affine known root atzero; no neural model or target reads.
    size=470;center=[arb(0)]*size;box=[arb(0,arb((1,-20)))]*size
    pre=eye(size);J=arb_mat([[arb(1) if i==j else arb((1,-14)) for j in range(size)] for i in range(size)]);g=arb_mat(size,1)
    observe('synthetic_recipe',{'size':size,'C':'identity','J':'diagonal1,offdiagonal2^-14','known_root':'zero','native_calls':0})
    for k in range(3):
        image,defect=localization_identity(center,g,J,pre,box)
        observe('synthetic_localization_image',{'iteration':k,'prior':vector_json(box),'jacobian':matrix_json(J),'defect':matrix_json(defect),'image':matrix_json(image)})
        box=[box[i].intersection(image[i,0]) for i in range(size)]
        require(all(v.contains(0) for v in box),'known synthetic root retained')
    return {'passed':True,'native_calls':0,'synthetic_dimension':size,'intersections':3,'scope':'Dense470D saved-algebra/serialization cost only; no neural model or target inputs.'}


def main(args):
    require(not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip(),'clean source');require(os.getpriority(os.PRIO_PROCESS,0)>=10,'nice10')
    target=args.mode=='target';limit=180 if target else 20;rss=(2*1024**2 if target else 512*1024);cap=(512 if target else 32)*1024**2;reserve=(8 if target else 1)*1024**2
    out=args.out;out.mkdir(parents=True,exist_ok=False);start=time.monotonic();stop=[False];ticks=[0];serial=[0];total=[0];events={};progress={'localization_file':None,'proposal_file':None,'chart_files':[]}
    signal.signal(signal.SIGTERM,lambda *_:stop.__setitem__(0,True));resource.setrlimit(resource.RLIMIT_AS,((4 if target else 1)*1024**3,)*2)
    def save(name,value,final=False):
        p=out/name;p.parent.mkdir(parents=True,exist_ok=True);before=total[0];bound=cap if final else cap-reserve
        try:
            with p.open('xb') as raw:
                h=gzip.GzipFile(fileobj=raw,mode='wb',mtime=0) if name.endswith('.gz') else raw
                try:
                    chunks=[];size=0
                    for piece in json.JSONEncoder(sort_keys=True,separators=(',',':'),allow_nan=False).iterencode(value):
                        chunks.append(piece);size+=len(piece)
                        if size>=65536:h.write(''.join(chunks).encode());chunks=[];size=0;require(before+raw.tell()<bound,'aggregate raw cap')
                    if chunks:h.write(''.join(chunks).encode())
                finally:
                    if h is not raw:h.close()
                raw.flush();os.fsync(raw.fileno())
            require(before+p.stat().st_size<bound,'aggregate raw cap')
        finally:
            if p.exists():total[0]=before+p.stat().st_size
        return name
    def check():
        if stop[0] or time.monotonic()-start>=limit:raise TimeoutError('fixed continuation deadline')
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss>=rss:raise MemoryError('monitored RSS')
    def tick():
        ticks[0]+=1;check();require(ticks[0]<=360,'fixed360tick cap')
    def observe(kind,value):
        name=f'events/{serial[0]:04d}-{kind}.json.gz';save(name,value);serial[0]+=1;events[kind]=events.get(kind,0)+1
        if kind=='localization_return':progress['localization_file']=name
        if kind=='reanchor_proposal':progress['proposal_file']=name
        if kind=='chart_return':progress['chart_files'].append(name)
        check()
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    save('runtime.json',{'commit':commit,'mode':args.mode,'pid':os.getpid(),'sources':{n:entry(ROOT/n) for n in SOURCES},'python':sys.version,'numpy':np.__version__,'flint':flint.__version__,'flint_library':flint.__FLINT_VERSION__,'threads':dict.fromkeys(THREADS,'1'),'nice':os.getpriority(os.PRIO_PROCESS,0),'precision':256,'seconds_cap':limit,'rss_cap_kib':rss,'tick_cap':360,'raw_cap_bytes':cap,'reserve_bytes':reserve,'direction':'raw full-state tangent at authenticated unit002 endpoint; no assumed cusp association'})
    for n in SOURCES:
        p=out/'source'/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/n).read_bytes());total[0]+=p.stat().st_size
    outcome={'passed':False,'status':'unresolved','scope':'Joined normalized primitive18+1 eigenpair locus only; diagnostics do not certify genericity or attraction.'}
    try:
        with ctx.workprec(256):
            if not target:outcome.update(synthetic_profile(observe),status='synthetic_profile_completed')
            else:
                require(args.profile is not None and args.profile_manifest is not None and args.profile_outer_manifest is not None,'profile pin required')
                inp,parent,bindings=bind_parent(args);save('bindings.json',bindings);save('inputs.json',{'map_inputs':inp,'parent_compact':compact(parent),'localization_iterations':3,'chart_cap':2})
                check();model=TwoRateCuspMap(inp);result=reanchor(model,parent,tick=tick,observe=observe);save('reanchor.json',result)
                outcome.update(passed=result['passed'],status='two_reanchored_charts' if result['passed'] else 'localized_endpoint_then_chart_unresolved',gates=result['gates'])
    except BaseException as exc:outcome.update(status='failed_or_interrupted_with_retained_prefix',failure={'type':type(exc).__name__,'message':str(exc)})
    outcome.update(elapsed_seconds=time.monotonic()-start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,ticks=ticks[0],event_counts=events,stored_bytes_before_final=total[0],progress=progress)
    if outcome['elapsed_seconds']>=limit or outcome['peak_rss_kib']>=rss:outcome['passed']=False
    save('outcome.json',outcome,True);save('manifest.json',{'files':{str(p.relative_to(out)):entry(p) for p in sorted(out.rglob('*')) if p.is_file()}},True)
    for p in sorted(out.rglob('*'),reverse=True):p.chmod(0o555 if p.is_dir() else 0o444)
    out.chmod(0o555);print(json.dumps(outcome));return 0 if outcome['passed'] else 2
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['profile','target'],required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--profile',type=Path);p.add_argument('--profile-manifest');p.add_argument('--profile-outer-manifest');raise SystemExit(main(p.parse_args()))
