#!/usr/bin/env python3
"""Saved-only forcing/defect diagnosis. Float weights are proposals, never proof."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
import gzip,hashlib,json,resource,signal,subprocess,sys,time
from pathlib import Path
import numpy as np,flint
from flint import arb,arb_mat,ctx
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools/reviews'),str(ROOT/'tools')]
from review_clark_rnn_c1_math import restore
from neural_shrimp.rigorous_orbits import eye,ball_json,matrix_json,vector_json,norm_inf_upper
BASE=Path('/data2/neural-shrimp-data/explore/eigenpair-locus-prototype/attempt-001-target')
PIN='35c8aeb88105a5836d47de7b6747b74da789bc266ab6084731b1f33688798369'
RUN_PIN='6e21d614a08c336e3d7cde3127e0a6f8d6d4af461e261282df56df95106dbe26'
SOURCE='463e55775ec89d9f45c00ebf594f616a09b278f3'
FILES=['src/neural_shrimp/__init__.py','src/neural_shrimp/controls.py','src/neural_shrimp/models.py','src/neural_shrimp/rigorous_orbits.py',
 'tools/reviews/review_clark_rnn_c1_math.py','tools/diagnose_eigenpair_locus.py','tools/launch_eigenpair_diagnosis.py',
 'tests/test_eigenpair_diagnosis.py','docs/plans/eigenpair-locus-diagnosis.md']
def need(v,s):
    if not v:raise ValueError(s)
def entry(p):b=p.read_bytes();return {'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
def load(p):
    with (gzip.open(p,'rt') if str(p).endswith('.gz') else p.open()) as f:return json.load(f)
def matrix(rows):return arb_mat([[restore(v)for v in row]for row in rows])
def column(rows):return arb_mat([[restore(v)]for v in rows])
def floats(m):return np.array([[float(m[i,j])for j in range(m.ncols())]for i in range(m.nrows())])
def inventory(root,pin):
    need(entry(root/'manifest.json')['sha256']==pin,'manifest identity');inv=load(root/'manifest.json')['files']
    actual={str(p.relative_to(root)):p for p in root.rglob('*')if p.is_file()and p!=root/'manifest.json'};need(set(inv)==set(actual),'complete inventory')
    for k,p in actual.items():need(not p.is_symlink()and not p.stat().st_mode&0o222 and entry(p)==inv[k],'immutable '+k)

def proposals(b,a):
    """All calculations here are floating engineering diagnostics on |D|."""
    need(b.ndim==2 and b.shape[0]==b.shape[1] and a.shape==(b.shape[0],),'proposal dimensions')
    need(np.isfinite(b).all()and np.isfinite(a).all()and np.all(b>=0)and np.all(a>=0),'finite nonnegative inputs')
    values,vectors=np.linalg.eig(b);index=int(np.argmax(np.abs(values)));value=values[index];v=vectors[:,index]
    scale=max(float(np.max(np.abs(v))),np.finfo(float).tiny);w=np.maximum(np.abs(v)/scale,2.**-40)
    perron_ratio=(b@w)/w
    result={'spectral_radius_estimate':float(abs(value)),'selected_eigenvalue':[float(value.real),float(value.imag)],
            'eigenvalues':[[float(z.real),float(z.imag)]for z in values],
            'selected_eigenvector':[[float(z.real),float(z.imag)]for z in v],
            'eigen_residual_inf':float(np.max(np.abs(b@v-value*v))),
            'all_ones_row_bound':float(np.max(b.sum(axis=1))),
            'perron_abs_floor_weights':w.tolist(),'perron_weighted_row_ratios':perron_ratio.tolist(),
            'perron_weighted_row_bound':float(np.max(perron_ratio)),
            'linear_radii_proposal':None,'scope':'floating |D| proposal only; neither true-multiplier spectrum nor interval contraction proof'}
    try:
        e=np.linalg.solve(np.eye(len(a))-b,a)
        need(np.isfinite(e).all()and np.all(e>0),'positive resolvent radii')
        raw=e*(17/16)+max(float(np.max(e)),2.**-100)*2.**-40
        rr=(a+b@raw)/raw
        result['linear_radii_proposal']={'minimal_solve':e.tolist(),'solve_residual_inf':float(np.max(np.abs((np.eye(len(a))-b)@e-a))),
             'radii':raw.tolist(),'normalized_weights':(raw/np.max(raw)).tolist(),'forcing_plus_propagation_ratios':rr.tolist(),'maximum_ratio':float(np.max(rr)),
             'requires_reacquisition':'Any changed correction box must reacquire J on its whole domain; this saved-box proposal is not a passed gate.'}
    except (ValueError,np.linalg.LinAlgError) as exc:result['linear_radii_failure']=str(exc)
    return result

def main(out):
    need(not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip(),'clean source');need(os.getpriority(os.PRIO_PROCESS,0)>=10,'nice10')
    out.mkdir(parents=True,exist_ok=False);start=time.monotonic();stop=[False];total=[0]
    signal.signal(signal.SIGTERM,lambda *_:stop.__setitem__(0,True));resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,4*1024**3))
    def save(name,value):
        b=(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n').encode();need(total[0]+len(b)<32*1024**2,'output cap')
        with(out/name).open('xb')as f:f.write(b);f.flush();os.fsync(f.fileno())
        total[0]+=len(b)
    def guard():
        if stop[0]or time.monotonic()-start>=45:raise TimeoutError('saved-diagnosis deadline')
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss>=1536*1024:raise MemoryError('1.5GiB monitoredRSS')
    save('runtime.json',{'commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'sources':{k:entry(ROOT/k)for k in FILES},'pid':os.getpid(),'python':sys.version,'numpy':np.__version__,'flint':flint.__version__,'precision':384,'native_calls':0,'scope':'saved-only exploratory diagnosis'})
    for name in FILES:
        p=out/'source'/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/name).read_bytes());total[0]+=p.stat().st_size
    result={'passed':False,'native_calls':0}
    try:
        inventory(BASE,PIN);inventory(BASE/'run',RUN_PIN);rt=load(BASE/'run/runtime.json');need(rt['commit']==SOURCE,'fixed target source')
        for name,rec in rt['sources'].items():need(entry(BASE/'run/source'/name)==rec and hashlib.sha256(subprocess.check_output(['git','show',SOURCE+':'+name],cwd=ROOT)).hexdigest()==rec['sha256'],'archived source '+name)
        saved=load(BASE/'run/result.json.gz');need(len(saved['charts'])==1 and not saved['passed'],'one failed chart')
        r=saved['charts'][0];size=470;need(r['shooting_dimension']==size and r['settings']['precision']==256,'fixed saved chart')
        guard()
        with ctx.workprec(384):
            pre=matrix(r['preconditioner']);jp=matrix(r['predictor_jacobian']);jb=matrix(r['whole_jacobian']);rr=matrix(r['predictor_residual']);d=matrix(r['defect']);radii=[restore(v)for v in r['coordinate_radii']]
            need(all(radii[i]>0 for i in range(size)),'positive radii')
            forcing=-pre*rr
            a=[abs(forcing[i,0]).upper()for i in range(size)]
            b=arb_mat([[abs(d[i,j]).upper()for j in range(size)]for i in range(size)])
            prop=[sum((b[i,j]*radii[j]for j in range(size)),arb(0)).upper()for i in range(size)]
            rows=[]
            for i in range(size):
                rows.append({'coordinate':i,'forcing_over_radius':ball_json((a[i]/radii[i]).upper()),'nominal_radius_propagation_over_radius':ball_json((prop[i]/radii[i]).upper()),'sum_over_radius':ball_json(((a[i]+prop[i])/radii[i]).upper()),'saved_image_over_radius':ball_json((abs(restore(r['image_correction'][i][0]))/radii[i]).upper())})
            save('coordinate-splits.json',{'rows':rows,'forcing':matrix_json(forcing),'nominal_radii':vector_json(radii),'note':'Propagation split uses nominal stored radii; original Arb error-ball magnitude inflation is reflected separately by saved image ratio.'})
            save('jacobian-domains.json',{'predictor_only_defect_bound':ball_json(norm_inf_upper(eye(size)-pre*jp)),'whole_domain_defect_bound_independent384':ball_json(norm_inf_upper(eye(size)-pre*jb)),'saved_defect_bound384':ball_json(norm_inf_upper(d)),'scope':'Two enclosures on different saved domains; their subtraction is not an independently isolated correction-radius contribution.'})
            guard();bf=floats(b);af=np.array([float(v)for v in a]);proposal=proposals(bf,af);save('floating-proposals.json',proposal)
            order=sorted(range(size),key=lambda i:float((a[i]+prop[i])/radii[i]),reverse=True)
            result.update(passed=True,spectral_radius_estimate=proposal['spectral_radius_estimate'],all_ones_row_bound=proposal['all_ones_row_bound'],perron_weighted_row_bound=proposal['perron_weighted_row_bound'],positive_resolvent_proposal=proposal['linear_radii_proposal']is not None,worst_coordinate=order[0],worst_nominal_ratio=float((a[order[0]]+prop[order[0]])/radii[order[0]]),largest_forcing_ratio=max(float(a[i]/radii[i])for i in range(size)),largest_propagation_ratio=max(float(prop[i]/radii[i])for i in range(size)),positive_interpretation='No new proof. Weights/step proposals require a new source-bound whole-domain acquisition.')
    except BaseException as exc:result['failure']={'type':type(exc).__name__,'message':str(exc)}
    result.update(elapsed_seconds=time.monotonic()-start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if result['elapsed_seconds']>=45 or result['peak_rss_kib']>=1536*1024:result['passed']=False
    save('outcome.json',result);save('manifest.json',{'files':{str(p.relative_to(out)):entry(p)for p in sorted(out.rglob('*'))if p.is_file()}})
    for p in sorted(out.rglob('*'),reverse=True):p.chmod(0o555 if p.is_dir()else 0o444)
    out.chmod(0o555);print(json.dumps(result));return 0 if result['passed']else 2
if __name__=='__main__':raise SystemExit(main(Path(sys.argv[1])))
