"""Fresh-domain weighted retry of the retained first correlated chart."""
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
import certify_s2_period18_arclength_reboxing as io
import certify_s2_period18_endpoint as endpoint
import certify_s2_period18_reanchor as inherited
from neural_shrimp.rigorous_arclength_connection import two_arclength_charts
from neural_shrimp.rigorous_connection import restore_ball
from neural_shrimp.rigorous_connection_frames import fraction_mid
from neural_shrimp.rigorous_reboxing_gradient import ReboxingMeanValueHiddenRateMap
from neural_shrimp.rigorous_connection import require
from neural_shrimp.rigorous_orbits import exact,vector_json
from flint import arb
for name in ('entry','read','now','total','inventory','verify_inventory','sha256'):
    globals()[name]=getattr(io,name)
CARD='docs/plans/s2-period18-weighted-reanchor.md'
SOURCES=list(dict.fromkeys(inherited.SOURCES+['tools/analyze_s2_period18_correlated_radii.py','tools/launch_s2_period18_correlated_radii.py','tests/test_correlated_radii_diagnostic.py','docs/plans/s2-period18-correlated-radii-diagnostic.md','tools/certify_s2_period18_weighted_reanchor.py','tests/test_s2_period18_weighted_reanchor.py',CARD]))
CAP=128*1024**2
SETTINGS=dict(precision=256,halfwidth_exponent=-7,radius_exponent=-21,newton_steps=2)
PARENT=inherited.PARENT
PIN=inherited.PIN
PROPOSAL=Path('/data2/neural-shrimp-data/explore/certificates/s2-period18-correlated-radii-2026-10-10/attempt-001/run')
PROPOSAL_PIN='b2fcb6e72d5d7526ca2670a6288b81c49e2acb07ea5d684d231539770d0241de'
PROPOSAL_FILE='7af78e386ec3cb602716f19df694902fedb426cd6d6c876ef345f9fff98d0cb2'

def save(path,obj):
    data=io.canonical_bytes(obj);root=path.parent
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

def validate_proposal(row):
    p=row['proposal'];weights=p['exact_weights']
    require(p['radius_exponent']==SETTINGS['radius_exponent'] and len(weights)==235,'fixed235D radii proposal')
    ww=[exact(v) for v in weights]
    require(all(v.is_exact() and v.is_finite() and 0<v<=1 for v in ww),'positive exact normalized weights')
    require(len(p['exact_coordinate_radii'])==235 and all(exact(p['exact_coordinate_radii'][i])==ww[i]*arb((1,-21)) for i in range(235)),'exact proposed coordinate radii')
    return weights


def proposed_weights():
    require(entry(PROPOSAL/'manifest.json')['sha256']==PROPOSAL_PIN,'exact saved proposal manifest');verify_inventory(PROPOSAL,read(PROPOSAL/'manifest.json')['files'])
    rt=read(PROPOSAL/'runtime-start.json');out=read(PROPOSAL/'outcome.json')
    require(rt['source_commit']=='f69353f6f8d65c09ab3307b8526462b4fc047030' and rt['parent_manifest']=='a4d46f5d06e52b7c54251031baf847c4164f3f171f00faae96ace24849569885','fixed failed first-chart diagnosis')
    require(out['exit_code']==0 and out['status']=='saved_correlated_radii_proposed' and out['new_domain_evaluated'] is False,'completed proposal only')
    for name,digest in rt['source_sha256'].items():
        require(entry(ROOT/name)['sha256']==digest and sha256(subprocess.check_output(['git','show',rt['source_commit']+':'+name],cwd=ROOT))==digest,'unchanged proposal source '+name)
    require(entry(PROPOSAL/'radii-proposal.json')['sha256']==PROPOSAL_FILE,'fixed exact proposal bytes')
    return validate_proposal(read(PROPOSAL/'radii-proposal.json'))


def prepare():
    inp=inherited.prepare();inp['weights']=proposed_weights()
    inp['provenance']['radii_proposal']={'manifest':PROPOSAL_PIN,'file':PROPOSAL_FILE,'radius_exponent':-21,
      'scope':'proposal only; new entire predictor+weighted-error derivative acquired before strict root/primitive/anchor gates'}
    return inp


def profile_input():
    inp=io.synthetic();inp['model_inputs']['lr_hidden']=inp['model_inputs']['lr_output']='1/4';initial=['0']*234+['1/4']
    return {'model_inputs':inp['model_inputs'],'initial':initial,'normal':['0']*234+['1'],'origin':initial,
      'period':18,'weights':proposed_weights(),'anchor':{'state_box':vector_json([exact(v) for v in initial])},'provenance':inp['provenance']}


def profile_gate(path,pin):
    require(entry(path/'manifest.json')['sha256']==pin,'fixed completed profile manifest');verify_inventory(path,read(path/'manifest.json')['files'])
    rt=read(path/'runtime-start.json');result=read(path/'outcome.json')
    require(rt['mode']=='profile' and result['exit_code']==0 and result['status']=='synthetic_profile_completed','completed engineering profile')
    require(rt['source_sha256']==sources(),'unchanged profiled source bytes')
    for key,value in {'python':platform.python_version(),'python_flint':flint.__version__,'flint':flint.__FLINT_VERSION__,
        'numpy':numpy.__version__,'scipy':scipy.__version__,'precision':256,'map_cap':144,'unit_derivative_cap':288,'jet_cap':108,'internal_seconds':20,'tick_cap':400}.items():
        require(rt.get(key)==value,'profile library/settings equality '+key)
    require(rt['threads']==dict.fromkeys(THREADS,'1') and rt['nice']>=10,'profile thread/nice')
    require(result['adapter_counts']=={'map_attempted':144,'map_returned':144,'unit_attempted':288,'unit_returned':288,'map_cap':144,'unit_cap':288},'profile all144 native calls')
    require(result['jet_counts']=={'attempted':108,'returned':108,'cap':108},'profile all108 path jets')
    charts=read(path/'certificate.json.gz')['charts'];require(len(charts)==2 and all(c['gates']['uniform_root'] for c in charts),'both synthetic root paths')
    elapsed=result['elapsed_seconds'];rss=result['max_rss_kib']
    require(type(elapsed) in (int,float) and math.isfinite(elapsed) and 0<=elapsed<20,'finite profile elapsed')
    require(type(rss) in (int,float) and math.isfinite(rss) and 0<=rss<2*1024**2,'bounded profile RSS')
    require(3*elapsed+5<40,'profile projected target time')
    estimate=max(4*total(path),80*1024**2);require(estimate<112*1024**2,'profile projected target storage')
    return {'manifest_sha256':pin,'projected_seconds':3*elapsed+5,'projected_bytes':estimate,'scope':'engineering gate only'}


def execute(out,mode,profile=None,profile_pin=None):
    bound=sources();require(os.getpriority(os.PRIO_PROCESS,0)>=10,'nice10')
    out.mkdir(parents=True,exist_ok=False);start=time.monotonic();limits=20 if mode=='profile' else 40
    ticks=0;serial=0;model=None;jet_model=None;stopped=False;code=2;outcome={'passed':False,'mode':mode,'status':'started'}
    runtime={'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'source_sha256':bound,
      'mode':mode,'pid':os.getpid(),'ppid':os.getppid(),'started_utc':now(),'nice':os.getpriority(os.PRIO_PROCESS,0),
      'threads':{k:os.environ[k] for k in THREADS},'python':platform.python_version(),'python_flint':flint.__version__,'flint':flint.__FLINT_VERSION__,
      'numpy':numpy.__version__,'scipy':scipy.__version__,'precision':256,'map_cap':144,'unit_derivative_cap':288,'jet_cap':108,'internal_seconds':limits,'tick_cap':400,
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
    fold=None
    def observe(kind,row):
        nonlocal serial
        name=f'stage-{serial:03d}-{kind}'
        if kind=='chart_completed':
            if fold is not None:
                c=row['result'];c['known_fold_membership']={'identity':fold['identity'],'all_cyclic_alignments':io.fold_alignments(c,fold['orbit'],fold['rate'])}
            payload=save(out/(name+'.json.gz'),row);save(out/(name+'-payload.json'),payload)
        else:save(out/(name+'.json'),row)
        serial+=1;tick()
    try:
        if mode=='target':save(out/'cost-gate.json',profile_gate(profile,profile_pin))
        with ctx.workprec(256):
            inp=profile_input() if mode=='profile' else prepare()
            save(out/'inputs.json',inp);tick()
            fold=io.known_fold(inp['model_inputs']) if mode=='target' else None
            if fold is not None:save(out/'fold-identity.json',fold['identity'])
            model=ReboxingMeanValueHiddenRateMap(inp['model_inputs'],max_calls=144);jet_model=io.CountedJetMap(inp['model_inputs'],108)
            result=two_arclength_charts(model,jet_model,inp,synthetic_profile=mode=='profile',**SETTINGS,tick=tick,observe=observe)
        payload=save(out/'certificate.json.gz',result);save(out/'certificate-payload.json',payload);tick()
        code=0 if mode=='profile' or result['passed'] else 2
        outcome.update(passed=bool(result['passed']) and mode=='target',status='synthetic_profile_completed' if mode=='profile' else ('correlated_anchor_joined' if result['passed'] else 'inconclusive'),
          scientific_claim=bool(result['passed']) and mode=='target',completed_charts=len(result['charts']),
          known_fold_identified=any(v['contained_in_qualified_chart'] for c in result['charts'] for v in c.get('known_fold_membership',{}).get('all_cyclic_alignments',[])))
    except BaseException as exc:
        outcome.update(status='failed_or_interrupted',error={'type':type(exc).__name__,'message':str(exc)});save(out/'failure.json',outcome['error'])
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        if model is not None:outcome['adapter_counts']=model.counts()
        if jet_model is not None:outcome['jet_counts']=jet_model.counts()
        outcome.update(elapsed_seconds=time.monotonic()-start,ticks=ticks,stages=serial,finished_utc=now(),max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,exit_code=code)
        save(out/'outcome.json',outcome);save(out/'manifest.json',{'files':inventory(out),'within_limit':total(out)<CAP})
    print(json.dumps(outcome),flush=True);return code

def launch(out,mode,profile=None,profile_pin=None):
    bound=sources();require(os.getpriority(os.PRIO_PROCESS,0)>=10,'launcher nice10 required')
    seconds=25 if mode=='profile' else 60
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
