#!/usr/bin/env python3
"""One fixed native-event seed, synthetic cost profile, and isolated chain attempt."""
import os
THREADS=('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')
for k in THREADS:os.environ[k]='1'
import argparse,gc,gzip,hashlib,json,math,resource,signal,subprocess,sys,time
from fractions import Fraction
from pathlib import Path
import numpy as np,flint
from flint import arb,ctx
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools')]
import continue_eigenpair_neutral_bridge as B
from neural_shrimp.rigorous_eigenpair_locus import require
from neural_shrimp.rigorous_eigenpair_march import compact,decode
from neural_shrimp.rigorous_orbits import exact,vector_json,matrix_json
from neural_shrimp.rigorous_cusps import TwoRateCuspMap
from neural_shrimp.rigorous_generalized_chain import augmented_chain,chain_root_proof
from neural_shrimp.generalized_chain_proposal import from_accepted_chart
PARENT=B.T.BASE/'attempt-006-neutral-bridge'
PINS={'source':'80be53d7084ec6319670f4204dca91b95b5f1b05','outer':'5e12f84fa4e4e5443def4d46b164923dd23e906a98e57ca23bc33d92517266d9','worker':'a6e61f5cfcaaa403cee5404d148a06fca1311f4a5ed751355e2df9ce60c4b87e'}
BASE=B.T.BASE/'generalized-chain'
SOURCES=list(dict.fromkeys(B.SOURCES+['src/neural_shrimp/rigorous_generalized_chain.py','src/neural_shrimp/generalized_chain_proposal.py','tests/test_rigorous_generalized_chain.py','tests/test_generalized_chain_proposal.py','tools/probe_eigenpair_generalized_chain.py','tools/launch_eigenpair_generalized_chain.py','tests/test_generalized_chain_driver.py','docs/plans/eigenpair-generalized-chain-attempt.md','docs/reviews/eigenpair-neutral-bridge-unit006.json']))
SETTINGS={'precision':256,'period':18,'radius_exponent':-80,'newton_steps':8}
CAP=64*1024**2

def read(p):
    with(gzip.open(p,'rt')if str(p).endswith('.gz')else p.open())as f:return json.load(f)
def entry(p):
    b=p.read_bytes();return {'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
def inventory(root,pin):return B.M.Q.B.inventory(root,pin)
def sources():
    require(not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip(),'clean source')
    result={n:entry(ROOT/n)for n in SOURCES}
    for n,row in result.items():require(hashlib.sha256(subprocess.check_output(['git','show','HEAD:'+n],cwd=ROOT)).hexdigest()==row['sha256'],'committed source '+n)
    return result

def qualified_event(out,full,accepted):
    require(out['passed'] and out['status']=='qualified_event_stop' and out['ticks']==508,'fixed completed event stop')
    require(len(out['progress']['accepted'])==4 and out['progress']['active_call']is None,'four durable native accepted roots')
    rec=out['progress']['last_accepted'];require(rec==out['progress']['accepted'][-1] and rec['index']==3 and rec['call']==3 and rec['right']=='275/128','exact event root')
    c=B.accepted_check(full,'275/128')
    require(c==accepted['chart'] and accepted['index']==3 and accepted['call']==3,'full compact root identity')
    require(c['parameter_endpoints']==['273/128','275/128'] and c['parameter_center']=='137/64','fixed selected native event chart')
    flags=B.event_flags(accepted['diagnostics']);require(flags==out['event_flags']==['nontrivial_plus_one_gap_at_most_0.03','nontrivial_unit_circle_margin_at_most_0.03'],'exact retained event flags')
    return c

def bind_event():
    identities={'outer':inventory(PARENT,PINS['outer']),'worker':inventory(PARENT/'run',PINS['worker'])}
    rt=read(PARENT/'run/runtime.json');la=read(PARENT/'launcher-result.json');out=read(PARENT/'run/outcome.json');inp=read(PARENT/'run/inputs.json')
    event=read(PARENT/'run/event-stop.json');require(event['flags']==out['event_flags'] and event['accepted']==out['progress']['accepted'],'durable event matches accepted prefix')
    require(rt['commit']==la['commit']==PINS['source'] and rt['unit']==6 and rt['mode']==la['mode']=='target','event execution identity')
    require(rt['precision']==256 and rt['threads']==dict.fromkeys(THREADS,'1') and rt['nice']>=10,'native event settings')
    require(la['exit_code']==0 and not la['owned_live_pids'] and la['stop_reason']is None,'completed owned event process')
    require(set(rt['sources'])==set(B.SOURCES),'exact bridge sources')
    for n,row in rt['sources'].items():
        require(entry(ROOT/n)==entry(PARENT/'run/source'/n)==row,'unchanged bridge source '+n)
        require(hashlib.sha256(subprocess.check_output(['git','show',PINS['source']+':'+n],cwd=ROOT)).hexdigest()==row['sha256'],'historical bridge source '+n)
    rec=out['progress']['last_accepted'];full=read(B.T.C.input_path(PARENT,rec['full_chart_file']));accepted=read(B.T.C.input_path(PARENT,rec['accepted_file']))
    c=qualified_event(out,full,accepted)
    for k in ('period','sigma','dimension','normal','origin'):require(c[k]==inp['parent'][k],'unchanged native map/gauge '+k)
    reference=B.T.BASE/'attempt-001-narrower-target';inventory(reference/'run',B.FIRST_PINS['parent_run_manifest'])
    require(inp['map_inputs']==read(reference/'run/inputs.json')['map_inputs'],'exact original model/physical rates convention')
    identities.update(parent=str(PARENT),pins=PINS,full_chart=rec['full_chart_file'],full_identity=entry(B.T.C.input_path(PARENT,rec['full_chart_file'])),accepted_file=rec['accepted_file'],accepted_identity=entry(B.T.C.input_path(PARENT,rec['accepted_file'])),native_endpoint=c['parameter_endpoints'][1],native_center=c['parameter_center'])
    del full;gc.collect()
    seed,diagnostic=from_accepted_chart(c,accepted['diagnostics']['spectrum'])
    return inp['map_inputs'],seed,diagnostic,identities

def synthetic():
    inp={'schema':'neural-shrimp-c1-input-v1','network':{'activation':'tanh','d':2,'width':3,'bias':True,'in_scale':'3/2','out_scale':'2/3','loss_scale':'1/2','ridge':'1/1000'},'x':[[str(Fraction(i-12,48)),str(Fraction((i*7)%23-11,48))]for i in range(24)],'y':[str(Fraction((i*3)%17-8,64))for i in range(24)],'lr_hidden':'1/100','lr_output':'1/100'}
    z=[exact(str(Fraction(i,128)))for i in range(13)]+[exact('1/100')]*2+[exact('1/4')]*13+[exact('1/5')]*13
    return inp,z

def receipt(path,pin,outer,mode,identity):
    require(path==BASE/f'attempt001-{mode}'/'run','fixed precursor path')
    inventory(path,pin);inventory(path.parent,outer);rt=read(path/'runtime.json');out=read(path/'outcome.json');la=read(path.parent/'launcher-result.json')
    require(rt['mode']==mode and rt['sources']==identity and rt['commit']==la['commit'],'matching precursor source/mode')
    require(out['passed'] and la['mode']==mode and la['exit_code']==0 and not la['owned_live_pids'] and la['stop_reason']is None,'completed precursor')
    for name,row in identity.items():
        require(entry(path/'source'/name)==row,'archived precursor source '+name)
        require(hashlib.sha256(subprocess.check_output(['git','show',rt['commit']+':'+name],cwd=ROOT)).hexdigest()==row['sha256'],'historical precursor source '+name)
    for k,v in {'settings':SETTINGS,'python':sys.version,'numpy':np.__version__,'flint':flint.__version__,'flint_library':flint.__FLINT_VERSION__,'threads':dict.fromkeys(THREADS,'1')}.items():require(rt[k]==v,'same precursor '+k)
    require(rt['nice']>=10 and finite(out['elapsed_seconds'],30) and finite(out['peak_rss_kib'],2*1024**2),'bounded precursor resources')
    return out

def finite(x,upper):return type(x)in(int,float) and math.isfinite(x) and 0<=x<upper

def cost(profile):
    require(profile['ticks']==72 and profile['status']=='synthetic_profile_completed','two complete synthetic41D evaluations')
    times=profile['evaluation_seconds'];require(len(times)==2 and all(finite(t,10) and t>0 for t in times),'finite measured evaluation cost')
    predicted=20*max(times)+20
    require(predicted<100,'measured10-evaluation2x+20s target forecast')
    return predicted

def near(seed,box):
    require(len(seed)==len(box)==41,'whole candidate dimensions')
    scales=[arb(1)]*13+[arb(10),exact('1/10')]+[arb(1)]*13
    return all(abs(box[i]-seed[i])<scales[i]*exact('1/10')for i in range(28)) and all(abs(v)<1024 for v in box[28:])

def main(args):
    identity=sources();require(os.getpriority(os.PRIO_PROCESS,0)>=10,'nice10');require(args.out==BASE/f'attempt001-{args.mode}'/'run','fixed exclusive attempt path')
    out=args.out;out.mkdir(parents=True,exist_ok=False);start=time.monotonic();limit=100 if args.mode=='target'else 30;tickcap={'profile':72,'proposal':0,'target':360}[args.mode];ticks=[0];stop=[False];used=[0];serial=[0];seed=[None]
    signal.signal(signal.SIGTERM,lambda *_:stop.__setitem__(0,True));resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,)*2)
    def check():
        if stop[0] or time.monotonic()-start>=limit:raise TimeoutError('fixed worker limit')
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss>=2*1024**2:raise MemoryError('2GiB sampled self RSS')
    def save(name,value,final=False):
        raw=(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode();require(used[0]+len(raw)<CAP if final else used[0]+len(raw)<CAP-2*1024**2,'aggregate raw cap')
        p=out/name;p.parent.mkdir(parents=True,exist_ok=True)
        with p.open('xb')as f:f.write(raw);f.flush();os.fsync(f.fileno())
        used[0]+=len(raw)
    def tick():ticks[0]+=1;require(ticks[0]<=tickcap,'fixed native step count');check()
    def observe(kind,row):
        save(f'event-{serial[0]:03d}-{kind}.json',row);serial[0]+=1;check()
        if kind=='newton_proposal':require(near(seed[0],[decode(v)for v in row['center']]),'fixed proposal neighborhood')
    runtime={'commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'sources':identity,'mode':args.mode,'settings':SETTINGS,'python':sys.version,'numpy':np.__version__,'flint':flint.__version__,'flint_library':flint.__FLINT_VERSION__,'threads':dict.fromkeys(THREADS,'1'),'nice':os.getpriority(os.PRIO_PROCESS,0),'pid':os.getpid(),'time_cap':limit,'tick_cap':tickcap}
    save('runtime.json',runtime)
    for n in SOURCES:
        p=out/'source'/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/n).read_bytes());used[0]+=p.stat().st_size
    outcome={'passed':False,'status':'unresolved','scope':'Isolated normalized generalized-chain root preparation/proof only; no generic double+1, attraction or branch-connection claim.'}
    try:
        with ctx.workprec(256):
            if args.mode=='profile':
                inp,z=synthetic();model=TwoRateCuspMap(inp);times=[];save('inputs.json',{'map_inputs':inp,'seed':vector_json(z),'synthetic_only':True})
                for i in range(2):
                    before=time.monotonic();v,j,_=augmented_chain(model,z,18,tick);save(f'profile-{i}.json',{'value':matrix_json(v),'jacobian':matrix_json(j)});times.append(time.monotonic()-before)
                outcome.update(passed=True,status='synthetic_profile_completed',evaluation_seconds=times)
                outcome['forecast_seconds']=cost({'ticks':ticks[0],**outcome})
            else:
                inp,z,diagnostic,bindings=bind_event();seed[0]=z;save('inputs.json',{'map_inputs':inp,'seed':vector_json(z),'diagnostic':diagnostic,'bindings':bindings});check()
                if args.mode=='proposal':outcome.update(passed=True,status='saved_native_seed_prepared')
                else:
                    prof=receipt(args.profile,args.profile_manifest,args.profile_outer_manifest,'profile',identity);forecast=cost(prof)
                    prior=receipt(args.proposal,args.proposal_manifest,args.proposal_outer_manifest,'proposal',identity)
                    require(prior['ticks']==0 and prior['status']=='saved_native_seed_prepared','saved-only exact precursor')
                    require(read(args.proposal/'inputs.json')==read(out/'inputs.json'),'identical exact frozen seed/model/bindings')
                    save('precursors.json',{'profile':str(args.profile),'profile_manifest':args.profile_manifest,'profile_outer':args.profile_outer_manifest,'proposal':str(args.proposal),'proposal_manifest':args.proposal_manifest,'proposal_outer':args.proposal_outer_manifest,'forecast_seconds':forecast})
                    proof=chain_root_proof(TwoRateCuspMap(inp),z,18,radius_exponent=-80,newton_steps=8,tick=tick,observe=observe);save('proof.json',proof);check()
                    proximity=near(z,[decode(v)for v in proof['box']]);save('proximity.json',{'passed':proximity,'scope':'Fixed proposal neighborhood only; not a rigorous connection to the native locus.'})
                    outcome.update(passed=bool(proof['passed'] and proximity),status='chain_proof_completed',gates=proof['gates'],whole_box_proximity=proximity)
    except BaseException as exc:outcome.update(passed=False,status='failed_or_interrupted',failure={'type':type(exc).__name__,'message':str(exc)})
    outcome.update(elapsed_seconds=time.monotonic()-start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,ticks=ticks[0],observations=serial[0],stored_bytes_before_final=used[0])
    if not finite(outcome['elapsed_seconds'],limit) or not finite(outcome['peak_rss_kib'],2*1024**2):outcome['passed']=False
    save('outcome.json',outcome,True);save('manifest.json',{'files':{str(p.relative_to(out)):entry(p)for p in sorted(out.rglob('*'))if p.is_file()}},True)
    for p in sorted(out.rglob('*'),reverse=True):p.chmod(0o555 if p.is_dir()else 0o444)
    out.chmod(0o555);print(json.dumps(outcome));return 0 if outcome['passed']else 2

def parser():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['profile','proposal','target'],required=True);p.add_argument('--out',type=Path,required=True)
    for name in ('profile','proposal'):
        p.add_argument('--'+name,type=Path);p.add_argument('--'+name+'-manifest');p.add_argument('--'+name+'-outer-manifest')
    return p
if __name__=='__main__':raise SystemExit(main(parser().parse_args()))
