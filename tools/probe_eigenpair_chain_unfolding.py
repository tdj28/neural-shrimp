#!/usr/bin/env python3
"""Bounded saved-only coefficients of the independently audited exact Jordan root."""
import os
THREADS=('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')
for k in THREADS:os.environ[k]='1'
import argparse,hashlib,json,math,resource,signal,subprocess,sys,time
from pathlib import Path
import numpy as np,flint
from flint import arb,arb_mat,ctx
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools')]
import probe_eigenpair_generalized_chain as C
import probe_eigenpair_chain_lift as L
from neural_shrimp.rigorous_chain_unfolding import chain_coefficients
from neural_shrimp.rigorous_eigenpair_locus import require
from neural_shrimp.rigorous_eigenpair_march import decode
from neural_shrimp.rigorous_orbits import vector_json
from neural_shrimp.rigorous_cusps import TwoRateCuspMap
BASE=C.BASE/'unfolding'
AUDIT=Path('/data2/neural-shrimp-data/audits/generalized-chain-independent/attempt002')
APINS={'source':'4e8e0ec0844ed806d7a9c6452da9e7c603e63f02','outer':'ccc9147597490b78bcfc92f87a13093097694fb402c0ef1303ef7c4a52b64d91','worker':'f8eb89d3e205f9d1af994b93e27cae0a9657569aa25d35346df03dc45ba0d947'}
SOURCES=list(dict.fromkeys(L.SOURCES+['src/neural_shrimp/rigorous_chain_unfolding.py','tests/test_rigorous_chain_unfolding.py','tools/probe_eigenpair_chain_unfolding.py','tools/launch_eigenpair_chain_unfolding.py','tests/test_eigenpair_chain_unfolding_driver.py','docs/plans/eigenpair-chain-unfolding.md','docs/plans/eigenpair-chain-unfolding-attempt.md']))
SETTINGS={'precision':384,'native_dimension':13,'augmented_dimension':41,'native_steps':0}
CAP=64*1024**2
read=C.read;entry=C.entry;inventory=C.inventory

def sources():
    require(not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip(),'clean source')
    result={n:entry(ROOT/n)for n in SOURCES}
    for n,row in result.items():require(hashlib.sha256(subprocess.check_output(['git','show','HEAD:'+n],cwd=ROOT)).hexdigest()==row['sha256'],'committed source '+n)
    return result

def admit_audit(out,fresh):
    require(out['passed']is True and out['status']=='completed' and out['native_steps']==72,'completed independent audit')
    require(out['positive_actual_rates']is True and out['root_passed']is True and out['primitive_period']==18 and out['strict_inclusions']==41,'complete independent physical primitive root')
    require(out['quotient_unit_circle_excluded']is True and out['complement_attracting']is True and out['exact_algebraic_multiplicity_two']is True,'same-root qualified critical cluster')
    require(fresh['root']['passed']is True and fresh['root']['coordinate_inclusions']==[True]*41 and decode(fresh['root']['contraction'])<1,'fresh complete strict root')
    require([v['divisor']for v in fresh['divisors']]==[1,2,3,6,9]and all(v['passed']is True for v in fresh['divisors']),'fresh native proper-divisor exclusions')
    box=[decode(v)for v in fresh['box']];jac=arb_mat([[decode(v)for v in row]for row in fresh['whole_jacobian']])
    require(len(box)==41 and jac.nrows()==jac.ncols()==41 and box[13]>0 and box[14]>0,'same positive-rate whole box')
    return box,jac

def bind_audit():
    info={'outer':inventory(AUDIT,APINS['outer']),'worker':inventory(AUDIT/'run',APINS['worker'])}
    rt=read(AUDIT/'run/runtime.json');la=read(AUDIT/'launch-end.json');out=read(AUDIT/'run/outcome.json');inp=read(AUDIT/'run/input-binding.json');fresh=read(AUDIT/'run/fresh-box.json')
    require(rt['commit']==la['source_commit']==APINS['source']and rt['producer']==L.PINS['source'],'exact independent audit execution')
    require(la['exit_code']==0 and la['owned_group_empty']is True and la['child_pid_absent']is True and la['final_byte_gate']['passed']is True,'owned completed audit')
    require(rt['precision']==384 and rt['threads']==dict.fromkeys(THREADS,'1')and rt['nice']>=10,'audit scientific settings')
    require(rt['bindings']['outer']['manifest']==L.PINS['outer']and rt['bindings']['worker']['manifest']==L.PINS['worker'],'same original proved root')
    inventory(L.PARENT,L.PINS['outer']);inventory(L.PARENT/'run',L.PINS['worker'])
    original=read(L.PARENT/'run/inputs.json')
    require(inp['inputs_sha256']==entry(L.PARENT/'run/inputs.json')['sha256']and inp['proof_sha256']==entry(L.PARENT/'run/proof.json')['sha256']and inp['map_inputs']==original['map_inputs'],'exact physical map and root proof')
    for folder,source in [('audit-source',APINS['source']),('native-source',L.PINS['source'])]:
        for p in sorted((AUDIT/'run'/folder).rglob('*.py')):
            rel=str(p.relative_to(AUDIT/'run'/folder));require(p.read_bytes()==subprocess.check_output(['git','show',source+':'+rel],cwd=ROOT),'historical audit source '+rel)
    spectrum=read(AUDIT/'run/quotient-spectrum.json')
    require(spectrum['passed']is True and spectrum['complement_attracting']is True and all(v['strict_inside']is True for v in spectrum['spectral_exclusion']['discs']),'same-box attracting complement')
    require(len(spectrum['spectral_exclusion']['discs'])==11,'full11D complementary spectrum')
    box,jac=admit_audit(out,fresh)
    original_proof=read(L.PARENT/'run/proof.json');original_box=[decode(v)for v in original_proof['box']]
    require(all(box[i].contains(original_box[i])for i in range(41)),'fresh unique box contains the original proved root')
    info.update(root=str(AUDIT),pins=APINS,fresh_identity=entry(AUDIT/'run/fresh-box.json'),spectrum_identity=entry(AUDIT/'run/quotient-spectrum.json'),producer=L.PINS)
    return box,jac,info

def synthetic():
    n=13;box=[arb(0)]*13+[arb(1)]*2+[arb(1)]+[arb(0)]*12+[arb(0),arb(1)]+[arb(0)]*11
    j=arb_mat(41,41);a=arb_mat(13,13);a[0,1]=1
    for i in range(2,13):a[i,i]=arb(-1)/2
    for i in range(13):
        for k in range(13):j[i,k]=a[i,k];j[13+i,15+k]=a[i,k];j[26+i,28+k]=a[i,k]
        j[26+i,15+i]=-1
    j[1,13]=1;j[14,0]=4;j[14,1]=7;j[27,0]=7;j[27,14]=1;j[39,15]=1;j[40,16]=1;j[40,28]=1
    return box,j

def cost(out):
    require(out['passed']is True and out['status']=='synthetic_coefficients_completed' and out['ticks']==0,'two full13D synthetic coefficient evaluations')
    times=out['evaluation_seconds'];require(len(times)==2 and all(C.finite(t,10) and t>0 for t in times),'finite measured coefficient cost')
    forecast=4*max(times)+15
    require(forecast<30,'measured saved-only target with4x time and15s reserve')
    return forecast

def profile_receipt(args,identity):
    require(args.profile==BASE/'attempt001-profile/run','fixed completed coefficient profile')
    inventory(args.profile,args.profile_manifest);inventory(args.profile.parent,args.profile_outer_manifest)
    rt=read(args.profile/'runtime.json');out=read(args.profile/'outcome.json');la=read(args.profile.parent/'launcher-result.json')
    require(rt['mode']==la['mode']=='profile' and rt['commit']==la['commit'] and rt['sources']==identity,'same-source profile')
    require(la['exit_code']==0 and not la['owned_live_pids'] and la['stop_reason']is None,'completed owned profile')
    for n,row in identity.items():
        require(entry(args.profile/'source'/n)==row,'archived profile source '+n)
        require(hashlib.sha256(subprocess.check_output(['git','show',rt['commit']+':'+n],cwd=ROOT)).hexdigest()==row['sha256'],'historical profile source '+n)
    for k,v in {'settings':SETTINGS,'python':sys.version,'numpy':np.__version__,'flint':flint.__version__,'flint_library':flint.__FLINT_VERSION__,'threads':dict.fromkeys(THREADS,'1')}.items():require(rt[k]==v,'same profile '+k)
    require(rt['nice']>=10 and C.finite(out['elapsed_seconds'],30) and C.finite(out['peak_rss_kib'],2*1024**2),'bounded profile')
    return cost(out)

def main(args):
    identity=sources();require(os.getpriority(os.PRIO_PROCESS,0)>=10,'nice10');require(args.out==BASE/f'attempt001-{args.mode}'/'run','exclusive coefficient path')
    out=args.out;out.mkdir(parents=True,exist_ok=False);start=time.monotonic();ticks=[0];stop=[False];used=[0];serial=[0]
    signal.signal(signal.SIGTERM,lambda *_:stop.__setitem__(0,True));resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,)*2)
    def check():
        if stop[0] or time.monotonic()-start>=30:raise TimeoutError('30s saved coefficient worker limit')
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss>=2*1024**2:raise MemoryError('2GiB sampled self RSS')
    def save(name,value,final=False):
        raw=(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode();require(used[0]+len(raw)<CAP if final else used[0]+len(raw)<CAP-2*1024**2,'aggregate raw cap')
        p=out/name;p.parent.mkdir(parents=True,exist_ok=True)
        with p.open('xb')as f:f.write(raw);f.flush();os.fsync(f.fileno())
        used[0]+=len(raw)
    def tick():ticks[0]+=1;require(ticks[0]==0,'zero new native steps');check()
    def observe(kind,row):save(f'event-{serial[0]:03d}-{kind}.json',row);serial[0]+=1;check()
    save('runtime.json',{'commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'sources':identity,'mode':args.mode,'settings':SETTINGS,'python':sys.version,'numpy':np.__version__,'flint':flint.__version__,'flint_library':flint.__FLINT_VERSION__,'threads':dict.fromkeys(THREADS,'1'),'nice':os.getpriority(os.PRIO_PROCESS,0),'pid':os.getpid()})
    for n in SOURCES:
        p=out/'source'/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/n).read_bytes());used[0]+=p.stat().st_size
    result={'passed':False,'status':'unresolved','scope':'Saved whole-box conditional normal-form coefficients and physical-rate rank only, bound to the independent root/spectrum proof. No additional boundary topology or global shrimp claim.'}
    try:
        with ctx.workprec(384):
            if args.mode=='profile':
                box,jac=synthetic();times=[]
                save('inputs.json',{'box':vector_json(box),'jacobian':C.matrix_json(jac),'synthetic_only':True})
                for i in range(2):
                    t=time.monotonic();coefficients=chain_coefficients(box,jac);save(f'coefficients-{i}.json',coefficients);times.append(time.monotonic()-t)
                    require(coefficients['quadratic_nondegenerate']and coefficients['resonant_nondegenerate']and coefficients['unfolding_transverse'],'full synthetic coefficient operations')
                result.update(passed=True,status='synthetic_coefficients_completed',evaluation_seconds=times)
                result['forecast_seconds']=cost({**result,'ticks':ticks[0]})
            else:
                forecast=profile_receipt(args,identity);save('profile_binding.json',{'worker':str(args.profile),'worker_manifest':args.profile_manifest,'outer_manifest':args.profile_outer_manifest,'forecast_seconds':forecast})
                box,jac,bindings=bind_audit();save('inputs.json',{'box':vector_json(box),'jacobian':C.matrix_json(jac),'bindings':bindings});check()
                coefficients=chain_coefficients(box,jac);save('coefficients.json',coefficients);check()
                result.update(passed=True,status='saved_coefficients_completed',quadratic_nondegenerate=coefficients['quadratic_nondegenerate'],resonant_nondegenerate=coefficients['resonant_nondegenerate'],unfolding_transverse=coefficients['unfolding_transverse'],coefficients={k:coefficients[k]for k in ('a','b','b_minus_a','s')})
                if 'unfolding_determinant'in coefficients:result['unfolding_determinant']=coefficients['unfolding_determinant']
    except BaseException as exc:result.update(passed=False,status='failed_or_interrupted',failure={'type':type(exc).__name__,'message':str(exc)})
    result.update(elapsed_seconds=time.monotonic()-start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,ticks=ticks[0],observations=serial[0],stored_bytes_before_final=used[0])
    if not C.finite(result['elapsed_seconds'],30) or not C.finite(result['peak_rss_kib'],2*1024**2):result['passed']=False
    save('outcome.json',result,True);save('manifest.json',{'files':{str(p.relative_to(out)):entry(p)for p in sorted(out.rglob('*'))if p.is_file()}},True)
    for p in sorted(out.rglob('*'),reverse=True):p.chmod(0o555 if p.is_dir()else 0o444)
    out.chmod(0o555);print(json.dumps(result));return 0 if result['passed']else 2

def parser():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['profile','target'],required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--profile',type=Path);p.add_argument('--profile-manifest');p.add_argument('--profile-outer-manifest');return p
if __name__=='__main__':raise SystemExit(main(parser().parse_args()))
