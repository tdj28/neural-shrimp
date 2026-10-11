"""Fixed four saved failures: compare constant inverse and correlated old family."""
import gc,gzip,json,os,resource,signal,subprocess,sys,time,traceback
from pathlib import Path
from fractions import Fraction as F
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'src')]
import probe_native705_later_complement as P
from neural_shrimp.correlated_inverse_diagnosis import polynomial,weighted_norm,exact
from neural_shrimp.rigorous_orbits import matrix_json,vector_json,ball_json,eye
from flint import arb,arb_mat,ctx
M=P.M;need=P.need;read=P.read;entry=P.O.entry
OUT=P.BASE.with_name('later001-inverse-diagnosis')/'run'
FAILED='aea25e9590f5b147d8671faf7f2c1de80acb2100d6c26cdad21f99c4f210e4b9'
FILES=sorted(set(P.FILES+['src/neural_shrimp/correlated_inverse_diagnosis.py','tools/diagnose_native705_inverse_correlation.py','tools/launch_native705_inverse_diagnosis.py','tests/test_correlated_inverse_diagnosis.py','tests/test_native705_inverse_diagnosis.py','docs/plans/native705-inverse-correlation-diagnosis.md','docs/reviews/native705-later001-bound-failures.json']))

def column_matrix(path):
    r=read(path);need(r['encoding']=='mid-rad-dyadic-pairs'and len(r['rows'])==705 and all(len(row)==705 for row in r['rows']),'complete independently saved705 coefficient')
    return arb_mat([[arb(arb(tuple(v[0])),arb(tuple(v[1])))for v in row]for row in r['rows']])

def audit(tick,save):
    P.O.verify(P.BASE,FAILED);old=read(P.BASE/'launcher-result.json');need(not old['passed']and old['launched']==4 and old['complete_charts']==0 and not old['owned_live_pids'],'exact four completed unsuccessful attempts')
    card=read(ROOT/P.CARD);row,run,ir,rt,domain,inp,proof,ib=M.metadata(card,31,tick);need(read(P.BASE/'fixed-index.json')==card,'same literal original proof card')
    coeff=[column_matrix(ir/f'target/defect_coefficient_{k}.json.gz')for k in range(4)];radius=arb((1,-10));radii=[P.U.R.exact(P.U.M.point(v))for v in proof['root']['radii']];weights=[v/radius for v in radii]
    ratios=[max(abs(P.U.M.interval(z)[0]),abs(P.U.M.interval(z)[1]))/P.U.M.point(r)for z,r in zip(proof['root']['image_correction'],proof['root']['radii'],strict=True)]
    save('known-root-image.json',dict(original_contraction=proof['root']['contraction'],maximum_uniform_image_ratio=str(max(ratios)),coordinate_ratios=[str(v)for v in ratios],scope='Already independently proved uniform root minus original predictor. Bound alone does not supply a fresh derivative domain.'))
    rows=[]
    for d in sorted(old['attempts'],key=lambda v:v['assignment']['attempt']):
        idx=d['assignment']['attempt'];p=P.BASE/f'chart{idx:03d}/run';P.O.verify(p,d['worker']['sha256']);P.D.B.archived(p,old['source_commit'],'runtime.json','source');out=read(p/'outcome.json');need(out['assignment']==d['assignment']and out['native_calls']==(360 if idx==2 else 252),'exact retained failed worker and counts')
        c=out['assignment']['cell'];left,right=[F(v)-F(row['producer']['settings']['parameter'])for v in c['endpoints']];need(F(-1,32)<=left<right<=F(1,32),'same original closed parameter subset')
        C=P.U.R.matrix(read(p/'assembly/target-preconditioner.json.gz')['C']);J=P.U.R.matrix(read(p/'assembly/target_whole-full.json.gz')['jacobian']);tick();D=eye(705)-C*J;fixed,frows=weighted_norm(D,weights);del C,J,D;gc.collect();tick()
        D=polynomial(coeff,left,right);correlated,crows=weighted_norm(D,weights);del D;gc.collect();tick()
        rec=dict(attempt=idx,parameter_endpoints=c['endpoints'],original_offset=[str(left),str(right)],fixed_inverse_bound=ball_json(fixed),correlated_family_bound=ball_json(correlated),fixed_row_bounds=vector_json(frows),correlated_row_bounds=vector_json(crows),old_status=out['status'],old_error=out.get('error'),fixed_contracts=bool(fixed<1),correlated_contracts=bool(correlated<1),source_inputs={n:entry(p/n)for n in('assembly/target-preconditioner.json.gz','assembly/target_whole-full.json.gz','restriction.json')})
        if(p/'localization.json').exists():
            loc=read(p/'localization.json');err=loc['whole_refinement']['box'];rr=[max(abs(P.U.M.interval(v[0])[0]),abs(P.U.M.interval(v[0])[1]))/P.U.M.point(r)for v,r in zip(err,proof['root']['radii'],strict=True)];rec['saved_localization_error_maximum_ratio']=str(max(rr))
        save(f'chart-{idx}.json',rec);rows.append({k:v for k,v in rec.items()if not k.endswith('row_bounds')})
    save('input-binding.json',dict(failed_root=str(P.BASE),failed_outer=FAILED,index=row,independent_defects=[entry(ir/f'target/defect_coefficient_{k}.json.gz')for k in range(4)]))
    return dict(passed=True,native_calls=0,rows=rows,maximum_uniform_image_ratio=str(max(ratios)),scope='Saved diagnostic comparing valid but different sufficient bounds: fresh fixed-C/smaller physical domain versus original affine-C polynomial family/larger original correction domain. Neither failure proves instability. No new root, derivative acquisition or complementary stability theorem.')

def main():
    resource.setrlimit(resource.RLIMIT_AS,(6*1024**3,)*2);need(os.getpriority(os.PRIO_PROCESS,0)>=10 and all(os.environ.get(k)=='1'for k in P.O.THREADS),'CPU1nice10');need(not subprocess.check_output(['git','-c','core.preloadIndex=false','-c','index.threads=1','status','--porcelain'],cwd=ROOT).strip(),'clean committed source');OUT.mkdir(parents=True,exist_ok=False);start=time.monotonic();used=0;out=dict(passed=False,native_calls=0)
    def tick():need(time.monotonic()-start<120 and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<4*1024**2,'120s4GiB owned diagnosis')
    def stop(*_):raise TimeoutError('owned diagnosis deadline')
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGALRM,stop);signal.setitimer(signal.ITIMER_REAL,120)
    def save(name,v,closing=False):
        nonlocal used
        data=P.D.canonical_bytes(v);need(used+len(data)<(24 if closing else 22)*1024**2,'24MiB with2MiB final reserve');path=OUT/name;path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('xb')as f:f.write(data);f.flush();os.fsync(f.fileno())
        used+=len(data)
    try:
        commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();save('runtime.json',dict(source_commit=commit,pid=os.getpid(),precision=384,python=P.D.platform.python_version(),numpy=P.D.np.__version__,python_flint=P.D.flint.__version__,flint=P.D.flint.__FLINT_VERSION__,sources={n:entry(ROOT/n)for n in FILES}))
        for n in FILES:
            b=(ROOT/n).read_bytes();need(b==subprocess.check_output(['git','show',commit+':'+n],cwd=ROOT),'literal committed archive');p=OUT/'source'/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b);used+=len(b)
        with ctx.workprec(384):
            before=time.monotonic();n=705;s=arb((1,-24));C=arb_mat([[arb(int(i==j))+s for j in range(n)]for i in range(n)]);J=arb_mat([[arb(int(i==j))-s for j in range(n)]for i in range(n)]);D=eye(n)-C*J;q,_=weighted_norm(D,[arb(1)]*n);dq,_=weighted_norm(polynomial([D,D/2,D/4,D/8],'-1/32','1/32'),[arb(1)]*n);seconds=time.monotonic()-before;forecast=8*seconds+50;save('profile.json',dict(passed=q<1 and dq<1 and forecast<120,seconds=seconds,forecast_seconds=forecast,target_arrays_unopened=True));need(q<1 and dq<1 and forecast<120,'full705 measured diagnosis admission');del C,J,D;gc.collect();out=audit(tick,save)
    except BaseException as e:out.update(error=type(e).__name__+': '+str(e),traceback=traceback.format_exc())
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);out.update(seconds=time.monotonic()-start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss);save('outcome.json',out,True);save('manifest.json',dict(files={str(p.relative_to(OUT)):entry(p)for p in OUT.rglob('*')if p.is_file()}),True)
    print(json.dumps({k:v for k,v in out.items()if k!='rows'}));return 0 if out['passed']else 2
if __name__=='__main__':raise SystemExit(main())
