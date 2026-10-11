#!/usr/bin/env python3
"""Fixed saved-boundary mode tracking; no model or derivative acquisition."""
import os
THREADS=('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')
for k in THREADS:os.environ[k]='1'
import argparse,gc,gzip,hashlib,json,math,resource,signal,subprocess,sys,time
from pathlib import Path
import numpy as np,flint
from flint import ctx
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools')]
import narrow_eigenpair_locus as N
M=N.M;T=N.T
from neural_shrimp.rigorous_eigenpair_locus import require
from neural_shrimp.rigorous_eigenpair_march import compact,decode,midpoint_spectrum
from neural_shrimp.eigenpair_boundary_diagnostics import fold_indicator
from neural_shrimp.eigenpair_gauge_diagnostics import saved_predictor_diagnostics
from neural_shrimp.eigenpair_boundary_trend import summarize
CARD='docs/plans/eigenpair-boundary-trend-inputs.json'
EXTRA=['src/neural_shrimp/eigenpair_boundary_trend.py','tools/diagnose_eigenpair_boundary_trend.py','tools/launch_eigenpair_boundary_trend.py','tests/test_eigenpair_boundary_trend.py','docs/plans/eigenpair-boundary-trend.md',CARD,'docs/reviews/eigenpair-narrower-outcome.json']
SOURCES=list(dict.fromkeys(N.SOURCES+EXTRA));entry=M.entry


def bound_selected(root,pin,run_pin,relative,identity):
    require(entry(root/'manifest.json')['sha256']==pin and entry(root/'run/manifest.json')['sha256']==run_pin,'exact outer/worker manifest pins')
    outer=M.read(root/'manifest.json')['files'];run=M.read(root/'run/manifest.json')['files']
    require(outer['run/manifest.json']==entry(root/'run/manifest.json'),'nested manifest identity')
    path=T.C.input_path(root,relative)
    require(identity==run[relative]==outer['run/'+relative]==entry(path),'exact selected retained file')
    require(path.stat().st_mode&0o222==0,'read-only selected evidence')
    return path


def acquire_saved(observe):
    specification=M.read(ROOT/CARD);rows=[];bindings=[];previous=None
    for item in specification['roots']:
        root=Path(item['root'])
        require(root.parent==T.BASE and not root.is_symlink(),'owned exact evidence root')
        runtime_path=bound_selected(root,item['outer'],item['run'],'runtime.json',M.read(root/'run/manifest.json')['files']['runtime.json'])
        rt=M.read(runtime_path);require(rt['commit']==item['execution'] and rt['mode']=='target' and rt['precision']==256,'exact prior execution')
        for spec in item['selected']:
            path=bound_selected(root,item['outer'],item['run'],spec['path'],spec['identity'])
            with gzip.open(path,'rt') as f:value=json.load(f)
            if item['kind']=='accepted_saved_diagnostic':chart=value['chart'];diag=value['diagnostics']
            else:
                chart=compact(value)
                require(len(value['coordinate_inclusions'])==470 and all(value['coordinate_inclusions']) and value['anchor_check']['passed'],'full fresh accepted proof')
                diag={'spectrum':midpoint_spectrum(value,observe),'fold':fold_indicator(value,observe),'predictor':saved_predictor_diagnostics(value)}
            require(chart['passed'] and all(chart['gates'].values()) and chart['period']==18 and chart['dimension']==13 and chart['sigma']==1 and chart['shooting_dimension']==470,'accepted native root')
            require(chart['parameter_endpoints']==spec['expected_endpoints'],'selected exact interval')
            if previous is not None:
                require(previous['parameter_endpoints'][1]==chart['parameter_endpoints'][0],'gapless increasing native chain')
                require(previous['normal']==chart['normal'] and previous['origin']==chart['origin'],'same native gauge')
            row={'gauge':chart['parameter_center'],'rates':[float(decode(v).mid()) for v in chart['center'][-2:]],'spectrum':diag['spectrum'],'fold':diag['fold'],'predictor':diag['predictor'],'source':{'root':str(root),'path':spec['path'],'identity':spec['identity']}}
            observe('saved_row',row);rows.append(row);previous=chart
            del value;gc.collect()
        bindings.append({k:item[k] for k in ('root','outer','run','execution','kind')})
    require(len(rows)==52 and previous['parameter_endpoints'][1]=='227/128','fixed all49 plus3 selection')
    return rows,{'inputs':bindings,'scope':'Exact pinned outer/worker manifests and selected immutable diagnostic/full-chart files verified; unrelated raw files are not rehashed here. Scientific proof audit remains separate.'}


def main(args):
    require(not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip(),'clean source');require(os.getpriority(os.PRIO_PROCESS,0)>=10,'nice10')
    target=True;limit=90;rss=1536*1024;cap=32*1024**2;reserve=4*1024**2
    out=args.out;out.mkdir(parents=True,exist_ok=False);start=time.monotonic();stop=[False];ticks=[0];serial=[0];total=[0];events={};progress={'localization_file':None,'proposal_file':None,'chart_files':[]}
    signal.signal(signal.SIGTERM,lambda *_:stop.__setitem__(0,True));resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,)*2)
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
        ticks[0]+=1;check();require(ticks[0]==0,'no model or derivative calls')
    def observe(kind,value):
        name=f'events/{serial[0]:04d}-{kind}.json.gz';save(name,value);serial[0]+=1;events[kind]=events.get(kind,0)+1
        if kind=='chart_return':progress['chart_files'].append(name)
        check()
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    save('runtime.json',{'commit':commit,'mode':args.mode,'pid':os.getpid(),'sources':{n:entry(ROOT/n) for n in SOURCES},'python':sys.version,'numpy':np.__version__,'flint':flint.__version__,'flint_library':flint.__FLINT_VERSION__,'threads':dict.fromkeys(THREADS,'1'),'nice':os.getpriority(os.PRIO_PROCESS,0),'precision':256,'seconds_cap':limit,'rss_cap_kib':rss,'tick_cap':0,'raw_cap_bytes':cap,'reserve_bytes':reserve,'direction':'fixed saved boundary trends, no model/derivative calls or crossing proof'})
    for n in SOURCES:
        p=out/'source'/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/n).read_bytes());total[0]+=p.stat().st_size
    outcome={'passed':False,'status':'unresolved','scope':'Joined normalized primitive18+1 eigenpair locus only; diagnostics do not certify genericity or attraction.'}
    try:
        with ctx.workprec(256):
            rows,bindings=acquire_saved(observe);save('bindings.json',bindings);check()
            result=summarize(rows);save('result.json',result)
            outcome.update(passed=True,status='fixed_saved_boundary_trends_completed',rows=len(rows),scope=result['scope'])
    except BaseException as exc:outcome.update(status='failed_or_interrupted_with_retained_prefix',failure={'type':type(exc).__name__,'message':str(exc)})
    outcome.update(elapsed_seconds=time.monotonic()-start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,ticks=ticks[0],event_counts=events,stored_bytes_before_final=total[0],progress=progress)
    if outcome['elapsed_seconds']>=limit or outcome['peak_rss_kib']>=rss:outcome['passed']=False
    save('outcome.json',outcome,True);save('manifest.json',{'files':{str(p.relative_to(out)):entry(p) for p in sorted(out.rglob('*')) if p.is_file()}},True)
    for p in sorted(out.rglob('*'),reverse=True):p.chmod(0o555 if p.is_dir() else 0o444)
    out.chmod(0o555);print(json.dumps(outcome));return 0 if outcome['passed'] else 2
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['target'],default='target');p.add_argument('--out',type=Path,required=True);raise SystemExit(main(p.parse_args()))
