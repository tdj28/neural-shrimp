"""Bounded point-only Arb-AD cross-check of a saved numerical fold candidate."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
import argparse,hashlib,json,pathlib,signal,subprocess,sys,time
from fractions import Fraction
import numpy as np
from flint import arb,arb_mat,ctx
ROOT=pathlib.Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'tools'))
from review_c1_evidence import objective_ad,rational
RAW=pathlib.Path('/data2/neural-shrimp-data/explore/s2-period18-fold/attempt-001')
PIN='3a44977630bd3d9f329bad8e6369c4d024fc4a379ee6ebe593c85d9b913cb620'
CAP=8*1024*1024

def require(ok,msg):
    if not ok:raise ValueError(msg)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def norm(x):return float(np.sqrt(np.sum(np.abs(x)**2)))
def run(out):
    require(not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip(),'clean source');out.mkdir(parents=True,exist_ok=False);start=time.monotonic();calls=0;stages=[];used=0
    def save(name,obj):
        nonlocal used
        b=(json.dumps(obj,sort_keys=True,indent=2,allow_nan=False)+'\n').encode();require(used+len(b)<=CAP,'byte cap')
        with (out/name).open('xb') as f:f.write(b);f.flush();os.fsync(f.fileno())
        used+=len(b)
    def stop(*_):raise TimeoutError('fixed sixty-second review deadline')
    signal.signal(signal.SIGALRM,stop);signal.signal(signal.SIGTERM,stop);signal.setitimer(signal.ITIMER_REAL,60)
    report={'passed':False,'scope':'Numerical point-only cross-check at two adjacent native phases of an18-cycle; no interval bifurcation proof.'}
    source_names=['tools/reviews/review_s2_period18_fold.py','tools/review_c1_evidence.py','docs/plans/s2-period18-fold-independent-review.md']
    save('runtime-start.json',dict(commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),sources={p:sha(ROOT/p) for p in source_names},pid=os.getpid(),nice=os.getpriority(os.PRIO_PROCESS,0),threads=1,precision=128,internal_seconds=60,analytic_AD_calls_cap=324,raw_manifest=PIN))
    try:
        require(sha(RAW/'manifest.json')==PIN,'fixed input manifest');m=json.loads((RAW/'manifest.json').read_text())
        for name,e in m.items():require((RAW/name).stat().st_size==e['bytes'] and sha(RAW/name)==e['sha256'],'input '+name)
        original=json.loads((RAW/'outcome.json').read_text());binding=json.loads((RAW/'inputs.json').read_text());path=json.loads((RAW/'path.json').read_text());events=[json.loads(x) for x in (RAW/'augmented-evaluations.jsonl').read_text().splitlines()]
        require(len(path)==25 and len(events)==43 and all(x['certificate']['verified'] and x['certificate']['attracting'] for x in path),'retained path/counters')
        require(max(abs(np.array(events[-1]['residual'])))==original['solver']['augmented_residual_inf'],'final augmented residual')
        for phase in ('diagnostics','next_phase_diagnostics'):
            d=original[phase];v=np.array(d['right_null']);w=np.array(d['left_null_normalized']);wu=np.array(d['left_unit_null']);require(abs(norm(v)-1)<1e-12 and abs(w@v-1)<1e-12 and norm(w-wu/(wu@v))<1e-12,'stored null normalization')
            rows=d['finite_difference_series'];require([x['exponent'] for x in rows]==[14,16,18,20,22],'fixed FD series')
            for row in rows:require(abs(row['s_second']+row['c']/row['a'])<1e-12,'stored fold ratio')
            for key in ('a','c'):require(rows[-1][key]*rows[-2][key]>0 and abs(rows[-1][key]-rows[-2][key])/max(abs(rows[-1][key]),abs(rows[-2][key]))<1e-3,'stored FD convergence')
        parent=pathlib.Path(binding['historical_sources'][0]['path']);require(sha(parent)==binding['historical_sources'][0]['sha256'],'consumed training source');inputs=json.loads(parent.read_text());center=np.array(binding['center']);direction=np.array(binding['direction']);s=original['diagnostics']['s'];require(np.allclose(center+s*direction,original['parameter_log10'],rtol=0,atol=1e-15),'controls')
        def evaluate(x,sval,label):
            nonlocal calls,used
            data=inputs.copy();logrates=center+sval*direction
            for key,z in zip(('lr_hidden','lr_output'),logrates):data[key]=str(Fraction(10.0**float(z)))
            with ctx.workprec(128):
                z=[rational(str(Fraction(float(t)))) for t in x];M=arb_mat([[int(i==j) for j in range(13)] for i in range(13)]);orbit=[x.copy()];jac=[];maxradius=0.
                for _ in range(18):
                    calls+=1;require(calls<=324,'AD operation cap');image,J=objective_ad(data,z);M=J*M;z=image;orbit.append(np.array([float(t.mid()) for t in z]));jac.append(np.array([[float(J[i,j].mid()) for j in range(13)] for i in range(13)]));maxradius=max(maxradius,max(float(t.rad().upper()) for t in z),max(float(M[i,j].rad().upper()) for i in range(13) for j in range(13)))
                MM=np.array([[float(M[i,j].mid()) for j in range(13)] for i in range(13)]);R=orbit[-1]-x;require(maxradius<1e-18,'point arithmetic resolution')
            target=out/(label+'.npz');np.savez(target,x=x,s=np.array(sval),R=R,M=MM,orbit=np.array(orbit),jacobians=np.array(jac));used+=target.stat().st_size;require(used<CAP,'byte cap');stages.append(dict(label=label,file=target.name,sha256=sha(target),max_arb_radius=maxradius,rates={k:data[k] for k in ('lr_hidden','lr_output')}));save(label+'.json',stages[-1]);return R,MM,np.array(orbit),jac
        bases=[];results=[]
        for k,key in enumerate(('diagnostics','next_phase_diagnostics')):
            d=original[key];x=np.array(d['state']);v=np.array(d['right_null']);w=np.array(d['left_null_normalized']);R,M,orb,J=evaluate(x,s,'base'+str(k));bases.append((R,M,orb,J));A=M-np.eye(13);right=A@v;left=w@A;require(max(abs(R))<1e-8 and norm(right)<1e-7 and norm(left)/max(1,norm(w))<1e-7,'point closure/null action');fd=[]
            for e in (18,20):
                dx=2.**-e*max(1,max(abs(x)));ds=2.**-e*max(1,abs(s));Rp,*_=evaluate(x,s+ds,f'p{k}-e{e}-splus');Rm,*_=evaluate(x,s-ds,f'p{k}-e{e}-sminus');_,Mp,*_=evaluate(x+dx*v,s,f'p{k}-e{e}-xplus');_,Mm,*_=evaluate(x-dx*v,s,f'p{k}-e{e}-xminus');parameter=(Rp-Rm)/(2*ds);quad=(Mp-Mm)@v/(2*dx);a=float(w@parameter);c=float(w@quad);saved=next(row for row in d['finite_difference_series'] if row['exponent']==e);require(abs(a-saved['a'])/max(1,abs(a))<1e-5 and abs(c-saved['c'])/max(1,abs(c))<1e-5,'independent coefficient comparison');fd.append(dict(exponent=e,a=a,c=c,s_second=-c/a,parameter_action=parameter.tolist(),quadratic_action=quad.tolist()))
            result=dict(phase=k,closure_inf=float(max(abs(R))),right_null_action=right.tolist(),left_null_action=left.tolist(),right_null_residual=norm(right),normalized_left_null_residual=norm(left)/max(1,norm(w)),fd=fd);results.append(result);save('phase'+str(k)+'-checks.json',result)
        x=np.array(original['diagnostics']['state']);v0=np.array(original['diagnostics']['right_null']);v1=np.array(original['next_phase_diagnostics']['right_null']);q=bases[0][3][0]@v0;k=norm(q);sign=1 if float(q@v1)>=0 else -1;transport=norm(sign*q/k-v1);require(transport<1e-7,'native kernel transport')
        a0,c0=results[0]['fd'][-1]['a'],results[0]['fd'][-1]['c'];a1,c1=results[1]['fd'][-1]['a'],results[1]['fd'][-1]['c'];preda,predc=sign*k*a0,sign*c0/k;require(abs(a1-preda)/max(1,abs(a1))<1e-5 and abs(c1-predc)/max(1,abs(c1))<1e-5,'native coefficient normalization');div={str(d):float(max(abs(bases[0][2][d]-x))/max(1,max(abs(x)))) for d in (1,2,3,6,9)};require(all(z>1e-6 for z in div.values()),'primitive numerical period')
        report.update(passed=True,phases=results,transport=dict(k=k,orientation_sign=sign,right_vector_error=transport,predicted_a1=preda,predicted_c1=predc,predicted_s_second1=(-c0/a0)/k**2),proper_divisor_separations=div,new_eigen_SVD_solve_continuation_calls=0,shared_primitives=['same consumed data and Arb library','existing independent second-order objective AD'],limits=['Only two adjacent phases checked, not all18','Point ball arithmetic and converged finite differences are not a uniform interval fold proof','No independent eigenvalue decomposition; simple +1 spectral isolation remains producer evidence','Producer runtime explicit inventory has three core files; historical Git commit binds remaining tracked source'])
    except BaseException as e:report.update(error=type(e).__name__+': '+str(e))
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);report.update(elapsed_seconds=time.monotonic()-start,AD_calls=calls,retained_evaluations=stages);save('result.json',report);entries={p.name:dict(bytes=p.stat().st_size,sha256=sha(p)) for p in out.iterdir()};save('manifest.json',dict(files=entries,passed=report['passed']))
        for p in out.iterdir():p.chmod(0o444)
        out.chmod(0o555)
    print(json.dumps(dict(passed=report['passed'],elapsed_seconds=report['elapsed_seconds'],AD_calls=calls,error=report.get('error'))));return 0 if report['passed'] else 2
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=pathlib.Path,required=True);a=p.parse_args();raise SystemExit(run(a.out))
