"""One fresh-domain rebox of the saved second smaller chart."""
import os
THREADS=('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')
for name in THREADS:os.environ[name]='1'
import argparse,gzip,json,math,platform,resource,signal,subprocess,sys,time
from datetime import datetime,timezone
from fractions import Fraction
from pathlib import Path
import flint,numpy,scipy
from flint import ctx
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'tools'))
import certify_s2_period18_fold_rechart as io
from neural_shrimp.rigorous_arclength_rebox import proposed_radii,rebox
from neural_shrimp.rigorous_reboxing_gradient import ReboxingMeanValueHiddenRateMap
from neural_shrimp.rigorous_connection import require
from neural_shrimp.rigorous_orbits import exact,vector_json,canonical_bytes
from flint import arb
for name in ('entry','read','now','total','inventory','verify_inventory','sha256'):
    globals()[name]=getattr(io,name)
CARD='docs/plans/s2-period18-fold-saved-rebox.md'
SOURCES=list(dict.fromkeys(io.SOURCES+['tools/certify_s2_period18_fold_saved_rebox.py','tests/test_s2_period18_fold_saved_rebox.py',CARD]))
CAP=128*1024**2
PARENT=Path('/data2/neural-shrimp-data/explore/certificates/s2-period18-fold-rechart-2026-10-10/attempt-001-target/run')
PIN='2b900f6b1928ed512d0395200e276849a27c777e4284ed991950077fc0c7e5f1'
EXECUTION='f5a1490549cfc3e44ba045cfcb45d564f862401b'
PROFILE_PARENT=PARENT.parent.parent/'attempt-001-profile/run'
PROFILE_PIN='1a1f3ebae56e9b6c781ba7c5dbcd8a02e325e167489992bdad109935b8aae98a'

def target_pair(result,inp):
    require(not result['passed'] and len(result['charts'])==2,'exact retained partial pair')
    first,failed=result['charts']
    require(first['passed'] and all(first['gates'].values()) and all(first['coordinate_inclusions']) and first['anchor_check']['passed'],'qualified first native chart')
    require(first['anchor']==inp['anchor'] and first['anchor']['exact_parameter']=='33/128','original exact localized anchor')
    require(not failed['passed'] and failed['gates']=={'uniform_root':False,'primitive_period':True,'anchor_identified':True},'only second root-bound failure')
    require(failed['anchor']['state_box']==first['endpoint_root_enclosures'][1] and failed['anchor']['exact_parameter']==first['exact_parameter_endpoints'][1]==failed['exact_parameter_endpoints'][0]=='17/64','full qualified predecessor join')
    for chart in (first,failed):
        require(chart['statement']['period']==18 and chart['statement']['phase_dimension']==13 and chart['settings']==io.SETTINGS,'fixed native settings')
        require(chart['normal']==inp['normal'] and chart['origin']==inp['origin'] and chart['exact_weights']==inp['weights'],'unchanged gauge and weights')
        require([r['divisor'] for r in chart['divisors'] if r['excluded']]==[1,2,3,6,9],'full primitive exclusions')
        require(not any(v['contained_in_qualified_chart'] for v in chart['known_fold_membership']['all_cyclic_alignments']),'no prior fold membership')
    return failed


def prepare(profile=False):
    root=PROFILE_PARENT if profile else PARENT;pin=PROFILE_PIN if profile else PIN
    require(entry(root/'manifest.json')['sha256']==pin,'exact saved source manifest');verify_inventory(root,read(root/'manifest.json')['files'])
    rt=read(root/'runtime-start.json');outcome=read(root/'outcome.json');inp=read(root/'inputs.json');result=read(root/'certificate.json.gz')
    expected='3ee14c839fb27c07051b84998f09a3a201c52f4b' if profile else EXECUTION
    require(rt['source_commit']==expected and rt['mode']==('profile' if profile else 'target'),'exact saved execution class')
    for name,digest in rt['source_sha256'].items():
        require(entry(root/'source'/name)['sha256']==digest and entry(ROOT/name)['sha256']==digest,'unchanged archived source '+name)
        require(sha256(subprocess.check_output(['git','show',expected+':'+name],cwd=ROOT))==digest,'committed saved source '+name)
    if profile:
        require(outcome['exit_code']==0 and outcome['status']=='synthetic_profile_completed' and len(result['charts'])==2,'completed synthetic input')
        chart=result['charts'][1]
    else:
        require(outcome['exit_code']==2 and outcome['status']=='inconclusive' and not outcome['passed'],'retained partial target')
        original=io.io.historical.original_anchor()
        require(inp['model_inputs']==original['model_inputs'] and inp['provenance']['original_anchor']==original['provenance'],'same exact original map')
        chart=target_pair(result,inp)
    return {'model_inputs':inp['model_inputs'],'chart':chart,'provenance':{'root':str(root),'manifest':pin,'execution':expected,'certificate':entry(root/'certificate.json.gz'),'chart_index':1,'original_anchor':inp['provenance'].get('original_anchor'),'scope':'retained exact predictor/Taylor witness only; entire new derivative domain must be acquired'}}


def synthetic_proposal(chart):
    # Fixed engineering expansion only; target uses the unchanged numerical
    # proposal. Both execute the identical fresh-domain derivative/proof path.
    weights=chart['exact_weights'];exponent=-20
    return {'radius_exponent':exponent,'exact_weights':weights,'exact_coordinate_radii':[str(Fraction(v)*Fraction(2)**exponent) for v in weights],'scope':'fixed synthetic resource proposal, not a scientific result'}


def save(path,obj):
    data=canonical_bytes(obj);root=path.parent
    payload={'uncompressed_bytes':len(data),'uncompressed_sha256':sha256(data)}
    require(len(data)<256*1024**2,'uncompressed artifact cap')
    if path.suffix=='.gz':data=gzip.compress(data,compresslevel=6,mtime=0)
    require(len(data)<60*1024**2,'single stored artifact cap')
    require(total(root)+len(data)<CAP,'aggregate raw cap')
    temp=path.with_name(path.name+'.tmp')
    with temp.open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
    os.replace(temp,path)
    return payload
def sources():
    require(not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip(),'clean source required')
    result={}
    for name in SOURCES:
        b=(ROOT/name).read_bytes();require(b==subprocess.check_output(['git','show','HEAD:'+name],cwd=ROOT),'committed source '+name)
        result[name]=sha256(b)
    return result

def profile_gate(path,pin):
    require(entry(path/'manifest.json')['sha256']==pin,'fixed completed profile manifest')
    verify_inventory(path,read(path/'manifest.json')['files'])
    rt=read(path/'runtime-start.json');result=read(path/'outcome.json')
    require(rt['mode']=='profile' and result['exit_code']==0 and result['status']=='synthetic_profile_completed','completed engineering profile')
    require(rt['source_sha256']==sources(),'unchanged profiled source bytes')
    for key,value in {'python':platform.python_version(),'python_flint':flint.__version__,'flint':flint.__FLINT_VERSION__,
        'numpy':numpy.__version__,'scipy':scipy.__version__,'precision':256,'map_cap':18,'unit_derivative_cap':36,'internal_seconds':20,'tick_cap':400}.items():
        require(rt.get(key)==value,'profile library/settings equality '+key)
    require(rt['threads']==dict.fromkeys(THREADS,'1') and rt['nice']>=10,'profile thread/nice')
    require(result['adapter_counts']=={'map_attempted':18,'map_returned':18,'unit_attempted':36,'unit_returned':36,'map_cap':18,'unit_cap':36},'profile all18 fresh-domain calls')
    elapsed=result['elapsed_seconds'];rss=result['max_rss_kib']
    require(type(elapsed) in (int,float) and math.isfinite(elapsed) and 0<=elapsed<20,'finite profile elapsed')
    require(type(rss) in (int,float) and math.isfinite(rss) and 0<=rss<2*1024**2,'bounded profile RSS')
    require(result['completed_reboxes']==1 and result['root_gate'],'one complete synthetic root rebox')
    require(2*elapsed+5<30,'profile projected target time')
    estimate=max(4*total(path),80*1024**2)
    require(estimate<112*1024**2,'profile projected target storage')
    return {'manifest_sha256':pin,'projected_seconds':2*elapsed+5,'projected_bytes':estimate,'scope':'engineering gate only'}


def execute(out,mode,profile=None,profile_pin=None):
    bound=sources();require(os.getpriority(os.PRIO_PROCESS,0)>=10,'nice10')
    out.mkdir(parents=True,exist_ok=False);start=time.monotonic();limits=20 if mode=='profile' else 30
    ticks=0;serial=0;model=None;stopped=False;code=2;outcome={'passed':False,'mode':mode,'status':'started','completed_reboxes':0}
    runtime={'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'source_sha256':bound,
      'mode':mode,'pid':os.getpid(),'ppid':os.getppid(),'started_utc':now(),'nice':os.getpriority(os.PRIO_PROCESS,0),
      'threads':{k:os.environ[k] for k in THREADS},'python':platform.python_version(),'python_flint':flint.__version__,'flint':flint.__FLINT_VERSION__,
      'numpy':numpy.__version__,'scipy':scipy.__version__,'precision':256,'map_cap':18,'unit_derivative_cap':36,'internal_seconds':limits,'tick_cap':400,
      'raw_cap':CAP,'rss_cap_kib':2*1024**2,'parent_manifest':PIN if mode=='target' else None,'cpu_only':True}
    save(out/'runtime-start.json',runtime)
    for name in SOURCES:
        dest=out/'source'/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes((ROOT/name).read_bytes())
    def stop(*_):
        nonlocal stopped
        stopped=True
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGALRM,stop);signal.setitimer(signal.ITIMER_REAL,limits)
    resource.setrlimit(resource.RLIMIT_FSIZE,(CAP,CAP))
    def tick():
        nonlocal ticks
        ticks+=1
        if stopped or ticks>400 or time.monotonic()-start>=limits:raise TimeoutError('fixed time/tick cap')
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss>=2*1024**2:raise MemoryError('RSS cap')
        if total(out)>112*1024**2:raise OSError('ordinary raw cap')
    def observe(kind,row):
        nonlocal serial
        save(out/f'stage-{serial:03d}-{kind}.json',row);serial+=1;tick()
    try:
        if mode=='target':save(out/'cost-gate.json',profile_gate(profile,profile_pin))
        with ctx.workprec(256):
            inp=prepare(profile=mode=='profile')
            payload=save(out/'inputs.json.gz',inp);save(out/'inputs-payload.json',payload);tick()
            proposal=synthetic_proposal(inp['chart']) if mode=='profile' else proposed_radii(inp['chart'])
            save(out/'radii-proposal.json',proposal);tick()
            model=ReboxingMeanValueHiddenRateMap(inp['model_inputs'],max_calls=18)
            result=rebox(model,inp['chart'],proposal,tick=tick,observe=observe)
            result['saved_rebox_provenance']=inp['provenance']
            if mode=='target':
                fold=io.io.known_fold(inp['model_inputs']);result['known_fold_membership']={'identity':fold['identity'],'all_cyclic_alignments':io.io.fold_alignments(result,fold['orbit'],fold['rate'])}
        payload=save(out/'certificate.json.gz',result);save(out/'certificate-payload.json',payload);tick()
        code=0 if mode=='profile' or result['passed'] else 2
        outcome.update(passed=bool(result['passed']) and mode=='target',completed_reboxes=1,root_gate=result['gates']['uniform_root'],
          status='synthetic_profile_completed' if mode=='profile' else ('saved_chart_reboxed' if result['passed'] else 'inconclusive'),scientific_claim=bool(result['passed']) and mode=='target')
    except BaseException as exc:
        outcome.update(status='failed_or_interrupted',error={'type':type(exc).__name__,'message':str(exc)});save(out/'failure.json',outcome['error'])
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        if model is not None:outcome['adapter_counts']=model.counts()
        outcome.update(elapsed_seconds=time.monotonic()-start,ticks=ticks,stages=serial,finished_utc=now(),max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,exit_code=code)
        save(out/'outcome.json',outcome);save(out/'manifest.json',{'files':inventory(out),'within_limit':total(out)<CAP})
    print(json.dumps(outcome),flush=True);return code

def launch(out,mode,profile=None,profile_pin=None):
    bound=sources();require(os.getpriority(os.PRIO_PROCESS,0)>=10,'launcher nice10 required')
    seconds=25 if mode=='profile' else 45
    out.mkdir(parents=True,exist_ok=False);start=time.monotonic();proc=None
    meta={'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
       'source_sha256':bound,'mode':mode,'started_utc':now(),'launcher_pid':os.getpid(),'load_average':os.getloadavg(),
       'meminfo':Path('/proc/meminfo').read_text(),'external_seconds':seconds,'kill_grace_seconds':5,'cpu_only':True}
    cmd=['timeout','--signal=TERM','--kill-after=5s',f'{seconds}s','nice','-n','10',sys.executable,
         str(Path(__file__).resolve()),'--execute','--mode',mode,'--out',str(out/'run')]
    for flag,value in [('--profile',profile),('--profile-manifest',profile_pin)]:
        if value is not None:cmd.extend([flag,str(value)])
    meta['argv']=cmd;save(out/'launcher-start.json',meta)
    with (out/'process.log').open('xb') as log:
        try:
            proc=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            meta['owned_group_pid']=proc.pid;save(out/'launcher-start.json',meta);code=proc.wait(timeout=seconds+7)
        except BaseException as exc:
            meta['error']={'type':type(exc).__name__,'message':str(exc)};code=124
        finally:
            if proc is not None:
                try:os.killpg(proc.pid,signal.SIGKILL)
                except ProcessLookupError:pass
                try:proc.wait(timeout=5)
                except subprocess.TimeoutExpired:code=124
    live=[]
    if proc is not None:
        for q in Path('/proc').iterdir():
            if not q.name.isdigit():continue
            try:
                fields=(q/'stat').read_text().rsplit(')',1)[1].split()
                if int(fields[2])==proc.pid and fields[0]!='Z':live.append(int(q.name))
            except (OSError,ValueError):pass
    if live:code=124
    meta.update(exit_code=code,elapsed_seconds=time.monotonic()-start,finished_utc=now(),owned_live_pids=live)
    save(out/'launcher-result.json',meta)
    if (out/'run/manifest.json').exists():verify_inventory(out/'run',read(out/'run/manifest.json')['files'])
    files=inventory(out);require(total(out)<CAP+1024**2,'total launcher plus worker cap')
    save(out/'manifest.json',{'files':files,'total_bytes':sum(v['bytes'] for v in files.values()),'within_limit':total(out)<CAP+1024**2})
    for q in sorted(out.rglob('*'),reverse=True):q.chmod(0o555 if q.is_dir() else 0o444)
    out.chmod(0o555)
    print(json.dumps({'out':str(out),'exit_code':code,'manifest':entry(out/'manifest.json'),'elapsed_seconds':meta['elapsed_seconds']}),flush=True)
    return code


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--mode',choices=['profile','target'],required=True)
    p.add_argument('--execute',action='store_true');p.add_argument('--profile',type=Path);p.add_argument('--profile-manifest')
    a=p.parse_args();require((a.mode=='target')==(a.profile is not None and a.profile_manifest is not None),'target requires pinned profile')
    return (execute if a.execute else launch)(a.out,a.mode,a.profile,a.profile_manifest)
if __name__=='__main__':raise SystemExit(main())
