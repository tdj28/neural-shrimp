#!/usr/bin/env python3
"""One four-chart bounded wider continuation; each chart permits one full repair."""
import os
THREADS=('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')
for k in THREADS:os.environ[k]='1'
import argparse,gc,gzip,hashlib,json,math,resource,signal,subprocess,sys,time
from pathlib import Path
import numpy as np,flint
from flint import ctx
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools')]
import re
from fractions import Fraction
import continue_eigenpair_wider_chain as C
W=C.W
M=W.M;T=W.T
from neural_shrimp.rigorous_eigenpair_locus import require
from neural_shrimp.rigorous_eigenpair_march import compact
from neural_shrimp.rigorous_cusps import TwoRateCuspMap
from neural_shrimp.rigorous_eigenpair_radius_forecast import march
EXTRA=['src/neural_shrimp/rigorous_eigenpair_radius_forecast.py','tools/continue_eigenpair_radius_forecast.py','tools/launch_eigenpair_radius_forecast.py','tests/test_eigenpair_radius_forecast.py','docs/plans/eigenpair-radius-forecast.md','docs/reviews/eigenpair-wider-march-unit003.json']
SOURCES=list(dict.fromkeys(C.SOURCES+EXTRA));entry=M.entry
PINS={'parent_source':'a930234710d0925b5d43cc0d5f811d99c484d9f3','parent_manifest':'50af8672ab8bd3048f4dd4cf5d2bce9a12df409a7325d64a631fea70b479c09f','parent_run_manifest':'251c555e7183e4f8ad916049dd499d188c50fb204a793ab2347b49a168e1d852'}


def install(args):
    C.install(args);require(args.unit==4 and args.expected_right=='71/128','one fixed radius-forecast unit004')
    require(all(getattr(args,k)==v for k,v in PINS.items()),'exact fixed complete unit003 pins')


def radii(chart):
    return [Fraction(v)*Fraction(2)**chart['settings']['radius_exponent'] for v in chart['weights']]


def history_increment(charts):
    require(len(charts)==5,'original parent and four complete charts')
    first=compact(charts[0]);size=len(first['weights']);rows=[]
    for c in charts:
        c=compact(c)
        require(all(c[k]==first[k] for k in ('period','sigma','dimension','shooting_dimension','normal','origin')),'same full native history')
        require(c['settings']['halfwidth_exponent']==-6 and len(c['weights'])==size,'same widths/dimension')
        row=radii(c);require(all(v>0 for v in row),'positive exact historical radii');rows.append(row)
    for a,b in zip(charts,charts[1:]):require(a['parameter_endpoints'][1]==b['parameter_endpoints'][0],'consecutive exact history endpoints')
    increments=[]
    for a,b in zip(rows,rows[1:]):
        require(all(y>=x for x,y in zip(a,b)),'nonshrinking exact history')
        increments.append([y-x for x,y in zip(a,b)])
    delta=[max(row[i] for row in increments) for i in range(size)]
    require(all(2*delta[i]<=rows[-1][i]/16 for i in range(size)),'fixed forecast cap before target')
    return {'history_exact_radii':[list(map(str,row)) for row in rows],'fixed_increment':list(map(str,delta)),'multiplier':2,'relative_increment_cap':'1/16','scope':'Coordinatewise maximum of four exact historical radius increments, fixed for all four pilot segments; no derivative extrapolation and no proof gate relaxation.'}


def bind(args):
    inp,parent,bindings=C.bind(args)
    pi=M.read(args.parent/'run/inputs.json');out=M.read(args.parent/'run/outcome.json');charts=[pi['parent']];identities=[]
    for row in out['progress']['accepted']:
        p=T.C.input_path(args.parent,row['accepted_file'])
        with gzip.open(p,'rt') as f:v=json.load(f)
        require(v['chart']['parameter_endpoints'][1]==row['right'] and v['index']==row['index'] and v['call']==row['call'],'fixed historical accepted event')
        charts.append(v['chart']);identities.append({'path':str(p),'identity':entry(p)})
    require(charts[-1]==parent,'forecast ends at actual proved parent')
    bindings['radius_forecast']=history_increment(charts);bindings['radius_forecast']['history_files']=identities
    return inp,parent,bindings


def main(args):
    require(not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip(),'clean source');require(os.getpriority(os.PRIO_PROCESS,0)>=10,'nice10')
    target=True;limit=400;rss=2*1024**2;cap=512*1024**2;reserve=8*1024**2
    out=args.out;out.mkdir(parents=True,exist_ok=False);start=time.monotonic();stop=[False];ticks=[0];serial=[0];total=[0];events={};progress={'active_call':None,'full_chart_file':None,'accepted':[],'last_accepted':None}
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
        ticks[0]+=1;check();require(ticks[0]<=1040,'fixed1040tick cap')
    def observe(kind,value):
        name=f'events/{serial[0]:04d}-{kind}.json.gz';save(name,value);serial[0]+=1;events[kind]=events.get(kind,0)+1
        if kind=='march_call_start':progress['active_call']=value['call'];progress['full_chart_file']=None
        if kind=='chart_return':progress['full_chart_file']=name
        if kind=='march_accepted':
            require(progress['full_chart_file'] is not None,'durable full chart before accepted reference')
            rec={'call':value['call'],'index':value['index'],'full_chart_file':progress['full_chart_file'],'accepted_file':name,'right':value['chart']['parameter_endpoints'][1]}
            progress['accepted'].append(rec);progress['last_accepted']=rec;progress['active_call']=None
        check()
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    save('runtime.json',{'commit':commit,'unit':args.unit,'mode':args.mode,'pid':os.getpid(),'sources':{n:entry(ROOT/n) for n in SOURCES},'python':sys.version,'numpy':np.__version__,'flint':flint.__version__,'flint_library':flint.__FLINT_VERSION__,'threads':dict.fromkeys(THREADS,'1'),'nice':os.getpriority(os.PRIO_PROCESS,0),'precision':256,'seconds_cap':limit,'rss_cap_kib':rss,'tick_cap':1040,'raw_cap_bytes':cap,'reserve_bytes':reserve,'direction':'positive full-state tangent gauge, four fixed-increment radius forecasts with at most one nonshrinking repair each; diagnostics not proof'})
    for n in SOURCES:
        p=out/'source'/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/n).read_bytes());total[0]+=p.stat().st_size
    outcome={'passed':False,'status':'unresolved','scope':'Joined normalized primitive18+1 eigenpair locus only; diagnostics do not certify genericity or attraction.'}
    try:
        with ctx.workprec(256):
            inp,parent,bindings=bind(args);save('bindings.json',bindings);save('inputs.json',{'map_inputs':inp,'parent':parent,'halfwidth_exponent':-6,'accepted_cap':4,'call_cap':8,'repair_cap_per_chart':1})
            check();model=TwoRateCuspMap(inp);result=march(model,parent,bindings['radius_forecast']['fixed_increment'],tick=tick,observe=observe);save('march.json',result)
            outcome.update(passed=result['completed'],status='four_joined_charts' if result['completed'] else 'qualified_prefix_then_unresolved',calls=result['calls'],repairs=result['repairs'])
    except BaseException as exc:outcome.update(status='failed_or_interrupted_with_retained_prefix',failure={'type':type(exc).__name__,'message':str(exc)})
    outcome.update(elapsed_seconds=time.monotonic()-start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,ticks=ticks[0],event_counts=events,stored_bytes_before_final=total[0],progress=progress)
    if outcome['elapsed_seconds']>=limit or outcome['peak_rss_kib']>=rss:outcome['passed']=False
    save('outcome.json',outcome,True);save('manifest.json',{'files':{str(p.relative_to(out)):entry(p) for p in sorted(out.rglob('*')) if p.is_file()}},True)
    for p in sorted(out.rglob('*'),reverse=True):p.chmod(0o555 if p.is_dir() else 0o444)
    out.chmod(0o555);print(json.dumps(outcome));return 0 if outcome['passed'] else 2
def parser():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['target'],default='target');p.add_argument('--out',type=Path,required=True);p.add_argument('--unit',type=int,required=True);p.add_argument('--parent',type=Path,required=True);p.add_argument('--parent-source',required=True);p.add_argument('--parent-manifest',required=True);p.add_argument('--parent-run-manifest',required=True);p.add_argument('--expected-right',required=True);return p
if __name__=='__main__':
    args=parser().parse_args();install(args);raise SystemExit(main(args))
