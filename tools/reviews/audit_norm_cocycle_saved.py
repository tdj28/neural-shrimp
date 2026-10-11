"""Independent saved-only interval arithmetic; no producer numerical imports.

Post-outcome review, not a prospective experiment. Derivative enclosures and
known-root existence are authenticated premises, not independently acquired.
"""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
import argparse,gzip,hashlib,json,resource,signal,subprocess,time
from fractions import Fraction
from pathlib import Path
from flint import arb,arb_mat,ctx
ROOT=Path(__file__).resolve().parents[2]
BASE=Path('/data2/neural-shrimp-data/explore/certificates')
RAW=BASE/'s2-period18-norm-cocycle-2026-10-10/attempt-001'
OUTER='69f9bb5403d415a0c66dd9a6005f06ca41060c1f89ba19c52d8f435c72e93ed4'
PINS=[(RAW/'run','655a9d031c6eb00c4b98c69e2e88bb9ebf42dfb91c8e6608f3c92bea440034ad','38ea6ac3f5b0611f15f0965f5e7a92d6a6a24804'),
(BASE/'s2-period18-correlated-cocycle-2026-10-10/attempt-001-target/run','dafbf4da8cf43a60b4c65883aa2d6874b756a788af97925fc531a0c810ea364b','4ef38b0530ada42ec8164777d136768627989eed'),
(BASE/'s2-period18-arclength-slices-2026-10-10/attempt-001-target/run','ccb8f1c8d1cd08389fdcd9ce31d07debb077e6ca7f8d000056d35cba67f84f3a','1b3df14e08b5b1ae89a4e834f57295cc22207834'),
(BASE/'s2-period18-slice-budget-2026-10-10/attempt-002/run','301b5a433201f4349dfd8fb319f1e269c1edb7d31eb4ab21981a0207fc1e14cc','f991e22571f25f7f972564b37d7e919fefb3c01a')]

def need(x,message):
    if not x:raise ValueError(message)
def sha(b):return hashlib.sha256(b).hexdigest()
def read(p):
    b=p.read_bytes()
    if p.suffix=='.gz':b=gzip.decompress(b)
    return json.loads(b)
def put(p,x):p.write_text(json.dumps(x,indent=2,sort_keys=True,allow_nan=False)+'\n')
def enc(a):return {'mid':list(map(int,a.mid().man_exp())),'rad':list(map(int,a.rad().man_exp()))}
def clean(x):
    if isinstance(x,dict):return {k:clean(v) for k,v in x.items() if k not in ('display','scope')}
    if isinstance(x,list):return [clean(v) for v in x]
    return x
def ball(x):
    need(set(x)=={'mid','rad','display'} and x['rad'][0]>=0,'finite saved dyadic ball')
    return arb(arb(tuple(x['mid'])),arb(tuple(x['rad'])))
def mat(x):return arb_mat([[ball(v) for v in row] for row in x])
def exact(x):
    f=Fraction(x);return arb(f.numerator)/arb(f.denominator)
def eye(n):return arb_mat([[int(i==j) for j in range(n)] for i in range(n)])
def mid(a):return arb_mat([[v.mid() for v in row] for row in a.tolist()])
def serial(a):return [[enc(v) for v in row] for row in a.tolist()]
def inf(a):return max(sum((abs(v).upper() for v in row),arb(0)).upper() for row in a.tolist())
def frob(a):return sum((abs(v).upper()**2 for row in a.tolist() for v in row),arb(0)).sqrt().upper()
def finite(a):return all(v.is_finite() for row in a.tolist() for v in row)
def inventory(p,pin):
    need(sha((p/'manifest.json').read_bytes())==pin,'external manifest pin '+str(p))
    files=read(p/'manifest.json')['files'];actual={str(v.relative_to(p)) for v in p.rglob('*') if v.is_file() and v!=p/'manifest.json'}
    need(actual==set(files),'exact manifest inventory')
    for name,r in files.items():
        q=p/name;need(not q.is_symlink() and q.resolve().is_relative_to(p.resolve()),'safe regular path');b=q.read_bytes()
        need(sha(b)==r['sha256'] and len(b)==r['bytes'],'hash/size '+name);need(not(q.stat().st_mode&0o222),'read-only '+name)
    return {'path':str(p),'manifest_sha256':pin,'files':len(files),'bytes':sum(v['bytes'] for v in files.values())}
def authenticate(p,pin,commit):
    inv=inventory(p,pin);rt=read(p/'runtime-start.json');o=read(p/'outcome.json')
    need(rt['source_commit']==commit and rt['precision']==256 and o['exit_code']==0,'completed exact execution')
    need(rt['nice']>=10 and set(rt['threads'].values())=={'1'},'source runtime limits')
    for path,digest in rt['source_sha256'].items():
        need(sha(subprocess.check_output(['git','show',commit+':'+path],cwd=ROOT))==digest,'historical runtime source '+path)
    inv['execution_commit']=commit;return inv,rt,o

def pd(a,witness):
    """Use saved exact congruence proposal; no Cholesky or floating gate."""
    n=a.nrows();b=mat(witness['basis']);need(finite(a) and all(v.is_exact() for row in b.tolist() for v in row),'finite/exact congruence')
    inverse=b.inv();defect=inf(eye(n)-mid(inverse)*b);need(defect<1,'invertible congruence')
    c=inverse*a*inverse.transpose()
    margins=[c[i,i].lower()-sum((abs(c[i,j]).upper() for j in range(n) if j!=i),arb(0)) for i in range(n)]
    need(all(v.is_finite() and v>0 for v in margins),'strict independently reduced PD')
    return {'minimum_margin':enc(min(v.lower() for v in margins)),'inverse_defect':enc(defect)}

def tangent(pre,jac,weights,saved):
    n=jac.nrows();need(pre.nrows()==n==len(weights),'implicit dimensions')
    need(all(w.is_exact() and w>0 for w in weights),'positive weights')
    d=eye(n)-pre*jac
    q=max((sum((abs(d[i,j]).upper()*weights[j] for j in range(n)),arb(0))/weights[i]).upper() for i in range(n))
    need(q<1 and enc(q)==clean(saved['weighted_neumann_bound']),'same strict uniform Neumann bound')
    v=pre*arb_mat([[int(i==n-1)] for i in range(n)]);e=d*v
    radius=(max((abs(e[i,0]).upper()/weights[i]).upper() for i in range(n))/(1-q)).upper()
    y=arb_mat([[arb(v[i,0],weights[i]*radius)] for i in range(n)])
    need(serial(y)==clean(saved['initial_derivative_box']),'initial true-derivative enclosure')
    for row in saved['refinements']:
        image=v+d*y;need(serial(image)==clean(row['image']),'independent derivative image')
        y=arb_mat([[y[i,0].intersection(image[i,0])] for i in range(n)])
        need(serial(y)==clean(row['retained']),'independent derivative intersection')
    need(serial(y)==clean(saved['derivative_box']),'final derivative bound')
    return {'weighted_neumann_bound':enc(q),'derivative_coordinates':n,'refinements':3}

def polynomial(cs):
    t=arb(0,1);value=arb_mat(cs[0].nrows(),cs[0].ncols())
    for k,c in enumerate(cs):value+=c*(arb(1) if k==0 else arb(0).union(arb(1)) if k%2==0 else t)
    return value

def advance(a,ea,ga,b,eb,gb):
    n=a[0].nrows();full=[arb_mat(n,n) for _ in range(5)]
    for i in range(3):
        for j in range(3):full[i+j]+=a[i]*b[j]
    na=ga+frob(a[1])+frob(a[2]);nb=gb+frob(b[1])+frob(b[2])
    eps=(na*eb+ea*nb+ea*eb+frob(full[3])+frob(full[4])).upper()
    return full[:3],eps,(ga*gb).upper()

def case(root,role,rows,bases,delta):
    saved=read(root/(role+'-result.json.gz'));need(saved['passed'] and saved['phases']==18,'complete saved case')
    frames=[b.transpose() for b in bases];inverses=[b.inv() for b in frames];n=13
    for b,inv in zip(frames,inverses):need(inf(eye(n)-mid(inv)*b)<1,'frame invertibility')
    cs=[eye(n),arb_mat(n,n),arb_mat(n,n)];eps=arb(0);gamma=arb(1);checks=[]
    phase_records=[read(p) for p in sorted(root.glob(f'stage-*-{role}-norm_phase.json.gz'))]
    need([v['phase'] for v in phase_records]==list(range(18)), 'exact saved phase proposal inventory')
    for j,(j0,jp) in enumerate(rows):
        a0=mid(j0);d=jp*delta;a1=mid(d);error=j0-a0+(d-a1)*arb(0,1)
        left,right=frames[(j+1)%18],inverses[j]
        a=[left*x*right for x in [a0,a1,arb_mat(n,n)]];ea=frob(left*error*right)
        record=phase_records[j];g=ball(record['operator_bound']['gamma'])
        need(g.is_exact() and g>=0,'fixed gamma proposal')
        checks.append(pd(eye(n)*(g*g)-a[0].transpose()*a[0],record['operator_bound']['positive']))
        cs,eps,gamma=advance(a,ea,g,cs,eps,gamma)
    metric=saved['metric'];p=mat(metric['P']);need(all(v.is_exact() for row in p.tolist() for v in row),'fixed exact P')
    need(all(p[i,j]==p[j,i] for i in range(n) for j in range(n)),'symmetric P')
    pc=pd(p,metric['P_positive']);point=mid(cs[0]);dc=pd(p-point.transpose()*p*point,metric['point_D_positive'])
    condition=inf(p)*inf(p.inv());defect=inf(p-point.transpose()*p*point-eye(n));need(condition<2**40 and defect<arb('1/100'),'proposal guard')
    ds=[arb_mat(n,n) for _ in range(5)];ds[0]=p+arb_mat(n,n)
    for i in range(3):
        for j in range(3):ds[i+j]-=cs[i].transpose()*p*cs[j]
    norm=gamma+frob(cs[1])+frob(cs[2]);penalty=(inf(p)*(2*norm*eps+eps*eps)).upper()
    complete=polynomial(ds)-eye(n)*penalty;cc=pd(complete,metric['complete_D_positive'])
    return {'role':role,'passed':True,'exact_endpoints':saved['exact_endpoints'],'halfwidth':str(Fraction(1,4096 if role=='full' else 16384)),
      'phase_operator_gates':len(checks),'remainder_norm_upper':enc(eps),'remainder_penalty':enc(penalty),'P':pc,'point_D':dc,'complete_D':cc,
      'method':'Independent recurrence and metric arithmetic using saved fixed proposals; no new SVD/Cholesky/Lyapunov/model call.'}

def audit():
    inv=[inventory(RAW,OUTER)];runtimes=[];outcomes=[]
    for p,pin,commit in PINS:
        a,b,c=authenticate(p,pin,commit);inv.append(a);runtimes.append(b);outcomes.append(c)
    root,parent,old,budget=[v[0] for v in PINS]
    launch=read(RAW/'launcher-result.json')
    need(launch['source_commit']==PINS[0][2] and launch['exit_code']==0 and launch['owned_live_pids']==[] and launch['nice']>=10, 'producer launcher source/exit/ownership')
    need(sha((RAW/'launcher-source.py').read_bytes())==launch['launcher_source_sha256'], 'producer launcher bytes')
    need(0<launch['elapsed_seconds']<37 and 0<outcomes[0]['elapsed_seconds']<20 and outcomes[0]['max_rss_kib']<1024**2,'producer bounded completion')
    need(outcomes[1]['adapter_counts']['map_returned']==90 and outcomes[1]['mixed_counts']['returned']==18 and outcomes[1]['completed_physical_phases']==18,'complete native derivative acquisition')
    need(outcomes[2]['completed_cases']==2,'complete original metric/localization acquisition')
    need(outcomes[0]['completed_phase_attempts']==outcomes[0]['completed_phase_returns']==36,'all two-case phases')
    need(runtimes[0]['model_calls']==runtimes[0]['mixed_derivative_calls']==0,'no new derivatives')
    inp=read(parent/'inputs.json.gz');prior=read(old/'inputs.json.gz');original=read(old/'probe-result.json.gz')
    need(inp['chart']==prior['chart'] and inp['model_inputs']==prior['model_inputs'],'same model and inherited chart')
    chart=inp['chart'];need(chart['passed'] and all(chart['gates'].values()),'inherited qualified primitive root')
    need(inp['exact_endpoints']==original['localization']['exact_endpoints']==['719/2048','45/128'],'same closed parameter slice')
    start=read(old/'stage-007-slice_start.json.gz');final=read(old/'stage-010-slice_intersection.json.gz');report=read(budget/'probe-budget.json')
    need(start['normal']==chart['normal'] and start['origin']==chart['origin'] and start['preconditioner']==chart['preconditioner'] and start['inherited_image']==chart['image_correction'],'identical gauge/preconditioner/inherited root')
    need(final['retained_error']==original['localization']['final_error'] and final['iteration']==2,'last known-root intersection')
    t=inp['uniform_tangent'];need(t==report['implicit_tangent'] and t['passed'],'exact tangent premise identity')
    tv=tangent(mat(start['preconditioner']),mat(final['jacobian']),[exact(v) for v in chart['exact_weights']],t)
    need(inp['whole_box']==original['localization']['final_state_box'] and inp['whole_tangent']==[v[0] for v in t['derivative_box']],'actual root and tangent input binding')
    physical=read(parent/'stage-004-correlated_derivative_inputs.json.gz')
    # Decoded actual inputs may widen stored radii: compare exact documented decode.
    need(clean(physical['actual_whole_root_box'])==[enc(ball(v)) for v in inp['whole_box']],'actual decoded whole root')
    need(clean(physical['actual_whole_total_tangent'])==[enc(ball(v)) for v in inp['whole_tangent']],'actual decoded true tangent')
    native=read(parent/'probe-result.json.gz')
    need(clean(physical['actual_center_root_box'])==[enc(ball(v)) for v in native['center_localization']['final_state_box']], 'actual decoded center root')
    need(physical['exact_endpoints']==inp['exact_endpoints'] and physical['center_parameter']=='1439/4096', 'same center and complete slice')
    rows=[];bases=[]
    for j,phase in enumerate(original['stability']['phases']):
        need(phase['phase']==j and phase['P_positive']['passed'],'ordered frame proposal')
        bases.append(mat(phase['P_positive']['basis']))
        r=read(parent/f'stage-{5+j:03d}-correlated_phase_model.json.gz');need(r['phase']==j,'native mixed derivative phase')
        rows.append((mat(r['center_J']),mat(r['whole_total_Jprime'])))
    need(len(rows)==18,'all eighteen native phases')
    results=[case(root,role,rows,bases,exact(str(delta))) for role,delta in [('full',Fraction(1,4096)),('quarter',Fraction(1,16384))]]
    return {'passed':True,'inventories':inv,'uniform_tangent':tv,'cases':results,'scope':'Attraction on the two fixed inherited primitive-root slices, conditional on authenticated prior root and analytic derivative enclosure sources. Independent saved arithmetic; same Arb library. No new derivative evaluation or whole-branch claim.'}

def main(out):
    need(not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip(),'clean committed review')
    out.mkdir(parents=True,exist_ok=False);start=time.monotonic();status={'passed':False};code=1
    put(out/'runtime-start.json',{'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'checker_sha256':sha(Path(__file__).read_bytes()),'pid':os.getpid(),'ppid':os.getppid(),'nice':os.getpriority(os.PRIO_PROCESS,0),'precision':256,'internal_seconds':60,'raw_cap_bytes':8*1024**2,'rss_cap_kib':2*1024**2,'post_outcome':True})
    def stop(*_):raise TimeoutError('fixed saved-audit deadline')
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGALRM,stop);signal.setitimer(signal.ITIMER_REAL,60)
    try:
        need(os.getpriority(os.PRIO_PROCESS,0)>=10,'nice10')
        with ctx.workprec(256):status=audit()
        code=0
    except BaseException as e:status.update(error={'type':type(e).__name__,'message':str(e)})
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);status.update(elapsed_seconds=time.monotonic()-start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,exit_code=code)
        if status['peak_rss_kib']>=2*1024**2:status.update(passed=False,exit_code=1,error={'type':'MemoryError','message':'audit RSS exceeded'});code=1
        put(out/'result.json',status)
        files={str(p.relative_to(out)):{'sha256':sha(p.read_bytes()),'bytes':p.stat().st_size} for p in out.rglob('*') if p.is_file()}
        need(sum(v['bytes'] for v in files.values())<8*1024**2,'raw audit cap');put(out/'manifest.json',{'files':files})
    print(json.dumps(status));return code
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True,type=Path);raise SystemExit(main(p.parse_args().out))
