"""One bounded post-outcome review of the selected S2 period-18 certificate.

Reuses the unchanged C1 verifier and existing independent S2 Arb-jet AD.
This is not a new proof kernel or a proof of the surrounding shrimp geometry.
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
import argparse
from datetime import datetime,timezone
from fractions import Fraction
import hashlib,json,pathlib,resource,signal,subprocess,sys,time
ROOT=pathlib.Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools')]
from flint import arb,ctx
from neural_shrimp.rigorous_orbits import BallTanhMap,ball_json,canonical_bytes,check_orbit
import numpy as np
from review_c1_evidence import objective_ad,stored,matrix
from verify_certificates import verify
OLD=pathlib.Path('/data2/neural-shrimp-data/raw/EXP-028/attempt-002/stage3/certificates.json')
OLD_HASH='414bf427491b5f2ec76d35af49e718811054986078c9638b276d07593b24a7e8'
CAP=8*1024*1024

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def require(ok,text):
    if not ok:raise ValueError(text)
def pair_fraction(value):
    m,e=value;return Fraction(m)*(Fraction(2)**e)
def ball_interval(value):
    mid,rad=pair_fraction(value['mid']),pair_fraction(value['rad']);require(rad>=0,'nonnegative ball radius');return mid-rad,mid+rad

def run(receipt_path,out):
    require(not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip(),'clean review source')
    require(os.getpriority(os.PRIO_PROCESS,0)>=10,'nice10')
    out.mkdir(parents=True,exist_ok=False);start=time.monotonic();result={'status':'incomplete','stages':[]};written=0
    def save(name,value):
        nonlocal written
        data=canonical_bytes(value);require(written+len(data)<=CAP,'aggregate review cap')
        with (out/name).open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
        written+=len(data)
    def done(name,value):save(name+'.json',value);result['stages'].append(name)
    def stop(*_):raise TimeoutError('fixed review deadline or termination')
    source_names=['tools/reviews/review_shrimp_period18_certificate.py','tools/review_c1_evidence.py','src/neural_shrimp/rigorous_orbits.py','tools/verify_certificates.py','tools/certify_c1.py','src/neural_shrimp/trainability.py','src/neural_shrimp/models.py','src/neural_shrimp/gdmaps.py','src/neural_shrimp/__init__.py','src/neural_shrimp/controls.py','docs/plans/shrimp-period18-independent-review.md']
    source={p:sha(ROOT/p) for p in source_names};metadata=dict(source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),sources=source,pid=os.getpid(),nice=os.getpriority(os.PRIO_PROCESS,0),threads=1,started_utc=datetime.now(timezone.utc).isoformat(),internal_seconds=60,raw_byte_cap=CAP,receipt=str(receipt_path),receipt_sha256=sha(receipt_path))
    save('runtime-start.json',metadata);signal.signal(signal.SIGALRM,stop);signal.signal(signal.SIGTERM,stop);signal.setitimer(signal.ITIMER_REAL,60)
    try:
        receipt=json.loads(receipt_path.read_text());raw=pathlib.Path(receipt['external_root']);inputs=json.loads((raw/'inputs.json').read_text());proof=json.loads((raw/'certificate.json').read_text())['result']
        require(sha(OLD)==OLD_HASH,'historical source file');historical=json.loads(OLD.read_text())[111];require((historical['map_id'],historical['component'],historical['period'])==('S2-28011-zoom',587,18),'fixed row111 identity')
        require(inputs['period']==18 and len(inputs['approximation'])==13,'period and dimension')
        require(proof['settings']==dict(precision=256,radius_exponent=-80,newton_steps=5,max_power=4096),'fixed producer proof settings')
        require([Fraction(x) for x in inputs['approximation']]==[Fraction(x) for x in historical['certificate']['orbit'][0]],'exact historical phase0')
        for key in ('lr_hidden','lr_output'):require(Fraction(inputs[key])==Fraction(historical[key]),'exact rate '+key)
        plan_path=ROOT/'experiments/trainability/EXP-028-plan.json';plan=json.loads(plan_path.read_text());spec=next(s for s in plan['maps'] if s['map_id']=='S2-28011');require(spec['seed']==28011 and spec['system']=='S2','consumed development seed')
        parent_path=pathlib.Path('/data2/neural-shrimp-data/explore/certificates/c1-2026-10-07/attempt-003/inputs.json');require(sha(parent_path)=='a0916d9bb32f58fb3f426bd584b92686af7515b007ec6fb3f71ac4554fb5eb81','previously certified consumed data');parent=json.loads(parent_path.read_text())
        for key,shape in [('x',(24,2)),('y',(24,))]:
            require(inputs[key]==parent[key],'same consumed exact '+key);q=np.asarray(inputs[key],dtype=object);fs=[Fraction(z) for z in q.flat];a=np.array([float(z) for z in fs],dtype=np.float64).reshape(q.shape);require(a.shape==shape and all(Fraction(float(z))==f for z,f in zip(a.flat,fs)),'exact binary64 data '+key);h=hashlib.sha256(repr(a.shape).encode()+a.tobytes()).hexdigest();require(h==spec['data_sha256'][key],'frozen train hash '+key)
        for key,value in spec['network'].items():require(Fraction(inputs['network'][key])==Fraction(value) if isinstance(value,float) else inputs['network'][key]==value,'exact model '+key)
        historical_bindings={'file_sha256':OLD_HASH,'row':111,'map_id':'S2-28011-zoom','component':587,'plan_sha256':sha(plan_path),'input_sha256':sha(raw/'inputs.json'),'data_sha256':spec['data_sha256'],'exact_rates':{k:inputs[k] for k in ('lr_hidden','lr_output')}}
        for path,key in [(ROOT/'docs/EXP-028-summary.json','gate'),(pathlib.Path('/data2/neural-shrimp-data/audits/EXP-028/validation-001/audit.json'),'audit_gate')]:
            qs=json.loads(path.read_text())[key]['qualifying'];matches=[q for q in qs if (q['map_id'],q['component'])==('S2-28011-zoom',587)];require(len(matches)==1 and matches[0]['period']==18,'historical qualifying membership');historical_bindings[str(path)]={'sha256':sha(path),'row':matches[0]}
        survey_path=pathlib.Path('/data2/neural-shrimp-data/explore/a2-survey/survey.json');rays_path=survey_path.with_name('rays.json');survey=next(r for r in json.loads(survey_path.read_text()) if r['key']=='S2-28011-zoom:587');rays=json.loads(rays_path.read_text())[survey['key']];counts={}
        for ray in rays['rays']:counts[ray['type']]=counts.get(ray['type'],0)+1
        require(survey['period']==rays['period']==18 and survey['shrimp_consistent_draft'] is True and counts==survey['metrics']['types'],'saved anatomy identity and type counts')
        historical_bindings['anatomy']={'survey_sha256':sha(survey_path),'rays_sha256':sha(rays_path),'survey_row':survey,'scope':'Saved labels/bindings/counts only; no independent boundary continuation, local-map reanalysis or geometry proof.'};done('01-input-binding',historical_bindings)
        replay=verify(receipt_path);require(replay['passed'],'authoritative replay');done('02-authoritative-replay',replay)
        require([r['divisor'] for r in proof['divisors']]==[1,2,3,6,9],'exact proper divisors')
        with ctx.workprec(384):
            box=[stored(v) for v in proof['box']];center=[stored(v) for v in proof['center']];krawczyk=[stored(v) for v in proof['krawczyk_image']];require(all(b.contains_interior(k) for b,k in zip(box,krawczyk)),'strict stored inclusion')
            defect=matrix(proof['defect']);contract=max(sum(abs(defect[i,j]).upper() for j in range(13)).upper() for i in range(13));require(contract<1,'stored contraction')
            orbit=[[stored(v) for v in row] for row in proof['orbit_enclosures']]
            for d in (1,2,3,6,9):require(any(not (orbit[d][j]-box[j]).contains(0) for j in range(13)),'independent divisor '+str(d))
            witness=proof['attraction_witness_power'];require(type(witness) is int and 1<=witness<=4096 and witness&(witness-1)==0,'power-of-two witness');power=matrix(proof['monodromy']);k=1
            while k<witness:power=power*power;k*=2
            bound=max(sum(abs(power[i,j]).upper() for j in range(13)).upper() for i in range(13));require(bound<1,'stored attraction')
            done('03-stored-witnesses',{'inclusion':True,'contraction_upper':ball_json(contract),'divisors':[1,2,3,6,9],'attraction_witness_power':witness,'attraction_norm_upper':ball_json(bound)})
            seed=[Fraction(x) for x in inputs['approximation']];norm=max([Fraction(1)]+[abs(x) for x in seed]);distances=[]
            for value,s in zip(proof['box'],seed):
                lo,hi=ball_interval(value);distances.append(max(abs(lo-s),abs(hi-s))/norm)
            distance=max(distances);require(distance<=Fraction(1,10**7),'whole-box proximity')
            done('04-whole-box-proximity',{'normalized_max_distance':str(distance),'distance_decimal_display':float(distance),'cap':'1/10000000','per_coordinate':[str(x) for x in distances],'scope':'Entire exact dyadic box proximity to saved phase0; not seed inclusion, branch identity or parameter continuation.'})
            model=BallTanhMap(inputs);enclosure,jacobian=model.step(box);ad=[]
            for sign in (0,1):
                sample=[c+sign*(-1)**i*arb((1,-82)) for i,c in enumerate(center)];require(all(b.contains(s) for b,s in zip(box,sample)),'AD sample in box');v,j=objective_ad(inputs,sample);map_ok=all(enclosure[i].contains(v[i]) for i in range(13));jac_ok=all(jacobian[i,k].contains(j[i,k]) for i in range(13) for k in range(13));require(map_ok and jac_ok,'independent objective AD');ad.append({'sample':sign,'map_enclosed':map_ok,'jacobian_enclosed':jac_ok})
            done('05-independent-AD',{'samples':ad,'source_sha256':sha(ROOT/'tools/review_c1_evidence.py'),'scope':'Existing separate Arb-jet differentiation of objective; same Arb library, sampled implementation cross-check.'})
            stronger=check_orbit(inputs,precision=384,radius_exponent=-90,newton_steps=6,max_power=4096);save('06-nested-proof-result.json',stronger);require(stronger['passed'],'fixed higher-precision proof');nested=all(b.contains_interior(stored(s)) for b,s in zip(box,stronger['box']));require(nested,'nested proof box');done('07-nested-proof-summary',{'passed':True,'settings':stronger['settings'],'strictly_inside_original_box':nested,'result_sha256':sha(out/'06-nested-proof-result.json')})
        result.update(status='passed',statement=proof['exact_statement'],scope='Exact-real primitive attracting period18 orbit near selected shrimp-consistent numerical window. No whole geometry, surrounding chaos, float-implementation or utility proof.')
    except BaseException as error:result.update(status='failed_or_incomplete',error=type(error).__name__+': '+str(error))
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);result.update(elapsed_seconds=time.monotonic()-start,max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss);save('result.json',result)
        entries={p.name:{'bytes':p.stat().st_size,'sha256':sha(p)} for p in out.iterdir() if p.is_file()};save('manifest.json',{'schema':'period18-independent-review-files-v1','status':result['status'],'files':entries})
        for p in out.iterdir():p.chmod(0o444)
        out.chmod(0o555)
    print(json.dumps(result,sort_keys=True));return 0 if result['status']=='passed' else 2

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--receipt',type=pathlib.Path,required=True);p.add_argument('--out',type=pathlib.Path,required=True);a=p.parse_args();raise SystemExit(run(a.receipt,a.out))
