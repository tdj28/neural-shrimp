"""Reverse two-chart pilot anchored at the independently certified fold."""
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
import certify_s2_period18_near_fold_endpoint as endpoint
from neural_shrimp.rigorous_fold_reverse import fold_native_proposal
from neural_shrimp.rigorous_arclength_connection import two_arclength_charts
from neural_shrimp.rigorous_connection import restore_ball
from neural_shrimp.rigorous_connection_frames import fraction_mid
from neural_shrimp.rigorous_reboxing_gradient import ReboxingMeanValueHiddenRateMap
from neural_shrimp.rigorous_connection import require
from neural_shrimp.rigorous_orbits import exact,vector_json
from flint import arb,arb_mat
for name in ('entry','read','now','total','inventory','verify_inventory','sha256'):
    globals()[name]=getattr(io,name)
CARD='docs/plans/s2-period18-fold-reverse.md'
SOURCES=list(dict.fromkeys(endpoint.SOURCES+io.SOURCES+['src/neural_shrimp/rigorous_fold_reverse.py','tools/certify_s2_period18_fold_reverse.py','tests/test_s2_period18_fold_reverse.py',CARD]))
CAP=128*1024**2
SETTINGS=dict(precision=256,halfwidth_exponent=-7,radius_exponent=-16,newton_steps=2)
PARENT=Path('/data2/neural-shrimp-data/explore/certificates/s2-period18-near-fold-endpoint-2026-10-10/attempt-001-target/run')
PIN='c2361075dd34763265f8f4756dcf979361795e9ee7cf63700dcd470dd3f86150'
EXECUTION='92fdb83d76123f71a8bf47eeac9b83b177a749f8'

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


def prepare():
    require(entry(PARENT/'manifest.json')['sha256']==PIN,'exact localized156th endpoint manifest');verify_inventory(PARENT,read(PARENT/'manifest.json')['files'])
    rt=read(PARENT/'runtime-start.json');outcome=read(PARENT/'outcome.json');old=read(PARENT/'inputs.json');result=read(PARENT/'result.json')
    require(rt['source_commit']==EXECUTION and rt['parent_manifest']==endpoint.PIN and rt['mode']=='target','localized source/native-root identity')
    require(outcome['passed'] and outcome['exit_code']==0 and outcome['completed_intersections']==3,'completed inherited localization')
    for name,digest in rt['source_sha256'].items():
        require(entry(PARENT/'source'/name)['sha256']==digest and entry(ROOT/name)['sha256']==digest,'unchanged localization source '+name)
        require(sha256(subprocess.check_output(['git','show',EXECUTION+':'+name],cwd=ROOT))==digest,'committed localization source '+name)
    require(old['provenance']['parent_manifest']==endpoint.PIN and old['previous']['exact_parameter_endpoints'][1]=='45/128','same exact156th endpoint')
    original=io.original_anchor();require(old['model_inputs']==original['model_inputs'] and old['provenance']['original_anchor']==original['provenance'],'same exact map/data/anchor')
    require(result['passed'] and result['scientific_claim'] and all(result['parent_gates'].values()) and result['iterations']==3 and result['precision']==256,'proved native localization')
    require(result['statement']['period']==18 and result['statement']['phase_dimension']==13 and len(result['final_state_box'])==235,'complete primitive native target')
    fold=io.known_fold(old['model_inputs']);receipt=read(ROOT/'certificates/s2-period18-fold-receipt.json')
    full=read(Path(receipt['external_root'])/'run/certificate.json')['result']
    return {'model_inputs':old['model_inputs'],'fold':fold,'fold_right':full['right_vector'],'target_box':result['final_state_box'],
      'provenance':{'localized_manifest':PIN,'localized_execution':EXECUTION,'original_anchor':original['provenance'],
       'fold':fold['identity'],'scope':'two previously proved physical roots; identity remains unproved until full correlated containment'}}


def profile_input():
    inp=io.synthetic();inp['model_inputs']['lr_hidden']=inp['model_inputs']['lr_output']='1/4';initial=['0']*234+['1/4']
    # Full native derivative and row-scale proposal path is exercised separately;
    # the synthetic fixed-point family then uses its nonsingular rate gauge.
    return {'model_inputs':inp['model_inputs'],'initial':initial,'normal':['0']*234+['1'],'origin':initial,
      'period':18,'weights':io.prepare_weights()+['1'],'anchor':{'state_box':vector_json([exact(v) for v in initial])},'provenance':inp['provenance']}

def profile_gate(path,pin):
    require(entry(path/'manifest.json')['sha256']==pin,'fixed completed profile manifest');verify_inventory(path,read(path/'manifest.json')['files'])
    rt=read(path/'runtime-start.json');result=read(path/'outcome.json')
    require(rt['mode']=='profile' and result['exit_code']==0 and result['status']=='synthetic_profile_completed','completed engineering profile')
    require(rt['source_sha256']==sources(),'unchanged profiled source bytes')
    for key,value in {'python':platform.python_version(),'python_flint':flint.__version__,'flint':flint.__FLINT_VERSION__,
        'numpy':numpy.__version__,'scipy':scipy.__version__,'precision':256,'map_cap':180,'unit_derivative_cap':360,'jet_cap':108,'internal_seconds':20,'tick_cap':400}.items():
        require(rt.get(key)==value,'profile library/settings equality '+key)
    require(rt['threads']==dict.fromkeys(THREADS,'1') and rt['nice']>=10,'profile thread/nice')
    require(result['adapter_counts']=={'map_attempted':180,'map_returned':180,'unit_attempted':360,'unit_returned':360,'map_cap':180,'unit_cap':360},'profile all180 native calls')
    require(result['jet_counts']=={'attempted':108,'returned':108,'cap':108},'profile all108 path jets')
    charts=read(path/'certificate.json.gz')['charts'];require(len(charts)==2 and all(c['gates']['uniform_root'] for c in charts),'both synthetic root paths')
    elapsed=result['elapsed_seconds'];rss=result['max_rss_kib']
    require(type(elapsed) in (int,float) and math.isfinite(elapsed) and 0<=elapsed<20,'finite profile elapsed')
    require(type(rss) in (int,float) and math.isfinite(rss) and 0<=rss<2*1024**2,'bounded profile RSS')
    require(3*elapsed+5<45,'profile projected target time')
    estimate=max(4*total(path),80*1024**2);require(estimate<112*1024**2,'profile projected target storage')
    return {'manifest_sha256':pin,'projected_seconds':3*elapsed+5,'projected_bytes':estimate,'scope':'engineering gate only'}


def execute(out,mode,profile=None,profile_pin=None):
    bound=sources();require(os.getpriority(os.PRIO_PROCESS,0)>=10,'nice10')
    out.mkdir(parents=True,exist_ok=False);start=time.monotonic();limits=20 if mode=='profile' else 45
    ticks=0;serial=0;model=None;jet_model=None;stopped=False;code=2;outcome={'passed':False,'mode':mode,'status':'started'}
    runtime={'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'source_sha256':bound,
      'mode':mode,'pid':os.getpid(),'ppid':os.getppid(),'started_utc':now(),'nice':os.getpriority(os.PRIO_PROCESS,0),
      'threads':{k:os.environ[k] for k in THREADS},'python':platform.python_version(),'python_flint':flint.__version__,'flint':flint.__FLINT_VERSION__,
      'numpy':numpy.__version__,'scipy':scipy.__version__,'precision':256,'map_cap':180,'unit_derivative_cap':360,'jet_cap':108,'internal_seconds':limits,'tick_cap':400,
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
    target=None
    def observe(kind,row):
        nonlocal serial
        name=f'stage-{serial:03d}-{kind}'
        if kind in ('chart_completed','fold_point_border'):
            if kind=='chart_completed' and target is not None:
                c=row['result'];c['known_center_branch_membership']={'identity':{'localized_manifest':PIN},
                  'all_cyclic_alignments':io.fold_alignments(c,[target[j*13:(j+1)*13] for j in range(18)],target[-1])}
            payload=save(out/(name+'.json.gz'),row);save(out/(name+'-payload.json'),payload)
        else:save(out/(name+'.json'),row)
        serial+=1;tick()
    try:
        if mode=='target':save(out/'cost-gate.json',profile_gate(profile,profile_pin))
        with ctx.workprec(256):
            inp=profile_input() if mode=='profile' else prepare()
            model=ReboxingMeanValueHiddenRateMap(inp['model_inputs'],max_calls=180);jet_model=io.CountedJetMap(inp['model_inputs'],108)
            if mode=='target':
                target=inp['target_box'];f=inp['fold']
                proposal=fold_native_proposal(model,f['orbit'],f['rate'],inp['fold_right'],target,tick=tick,observe=observe)
                save(out/'native-fold-proposal.json.gz',proposal);save(out/'native-fold-proposal-payload.json',{'uncompressed_bytes':len(io.canonical_bytes(proposal)),'uncompressed_sha256':sha256(io.canonical_bytes(proposal))})
                inp.update({k:proposal[k] for k in ('initial','normal','origin','weights','anchor','period')})
            else:
                # Match36 actual mean-value derivative calls and one dense inverse
                # without pretending the synthetic family has a state-only fold.
                from neural_shrimp.rigorous_arclength_connection import bordered
                from neural_shrimp.rigorous_orbits import midpoint_matrix,matrix_json
                states=[exact(v) for v in inp['initial']];dummy=arb_mat([[int(i==0)] for i in range(13)])
                for j in range(18):
                    tick();_,jj,_=model.step(states[j*13:(j+1)*13],states[-1]);dummy=jj*dummy
                _,jj,_=bordered(model,states,18,[exact(v) for v in inp['normal']],states,arb(0),tick)
                tick();pp=midpoint_matrix(jj.inv());save(out/'profile-extra-proposal.json',{'calls':36,'inverse_rows':pp.nrows(),'propagated_direction':matrix_json(dummy),'scope':'engineering only, synthetic rate gauge'})
            save(out/'inputs.json',inp);tick()
            result=two_arclength_charts(model,jet_model,inp,synthetic_profile=mode=='profile',**SETTINGS,tick=tick,observe=observe)
        payload=save(out/'certificate.json.gz',result);save(out/'certificate-payload.json',payload);tick()
        code=0 if mode=='profile' or result['passed'] else 2
        outcome.update(passed=bool(result['passed']) and mode=='target',status='synthetic_profile_completed' if mode=='profile' else ('fold_anchored_charts_qualified' if result['passed'] else 'inconclusive'),
          scientific_claim=bool(result['passed']) and mode=='target',completed_charts=len(result['charts']),
          center_branch_identified=any(v['contained_in_qualified_chart'] for c in result['charts'] for v in c.get('known_center_branch_membership',{}).get('all_cyclic_alignments',[])))
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
