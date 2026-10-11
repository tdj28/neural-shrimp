"""Bounded source-bound screening of six exact cubic predictor domains only.

No root/tangent/quotient replay and no native acquisition. All original
metadata/source bindings remain the same as the full independent adapter.
"""
import os
THREADS=('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')
for k in THREADS:os.environ[k]='1'
import gc,json,platform,resource,signal,subprocess,sys,time,traceback
from pathlib import Path
from fractions import Fraction as F
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'tools/reviews'),str(ROOT/'tools'),str(ROOT/'src')]
import flint
from flint import arb,ctx
import review_cubic_remaining as B
import review_cubic_center as C
from review_cubic_polynomial_domain import candidates
need,read=B.need,B.read
OUT=Path('/data2/neural-shrimp-data/audits/native705-cubic32-polynomial-domains/attempt001/run')
FILES=sorted(set(B.FILES+[f'tools/reviews/{prefix}{name}.py'for prefix in('','native704_aux/')for name in('review_joined_period18_math','review_quadratic_connection_math','review_unit_pair_math')]+['tools/reviews/diagnose_cubic_polynomial_domains.py','tools/reviews/launch_cubic_polynomial_domains.py','tests/reviews/test_cubic_polynomial_screen.py','docs/plans/native705-cubic-polynomial-screen.md']))


def screen(w,run,domains,data):
    original=data['domain']
    coeffs=[[B.R.exact(B.M.point(v))for v in original[k]]for k in('center','slope','curvature','cubic')]
    need(all(len(row)==705 for row in coeffs),'same full exact705 predictor')
    sc=F(original['parameter']);rows=[]
    for item in domains:
        d=item['domain'];left,right=map(F,d['endpoints']);lo,hi=left-sc,right-sc
        td=read(run/f"domains/{d['index']:03d}/tangent-domain-inputs.json")
        need(td['domain']==d and td['original_offset']==[str(lo),str(hi)],'literal actual scalar interval')
        pred,nested,monomial=candidates(*coeffs,lo,hi)
        row=dict(domain=d,offset=[str(lo),str(hi)],predictor=B.vector_json(pred),nested_derivative=B.vector_json(nested),monomial_derivative=B.vector_json(monomial),nested_comparison=C.coverage_diagnostic(td['predictor_tangent'],nested),monomial_comparison=C.coverage_diagnostic(td['predictor_tangent'],monomial))
        # A zero-error predictor is itself a point in the full-error segment.
        # This is only a necessary screening condition, not root localization.
        row['predictor_segment_comparison']=C.coverage_diagnostic(td['segment'],pred)
        row['passed']=row['nested_comparison']['passed']and row['predictor_segment_comparison']['passed']
        w.save(f"domains/{d['index']:03d}.json",row);rows.append(row);w.tick()
    need(len(rows)==6,'all six literal original domains screened')
    return dict(passed=all(r['passed']for r in rows),domains=[dict(domain=r['domain'],passed=r['passed'],nested_failures=len(r['nested_comparison']['failures']),monomial_failures=len(r['monomial_comparison']['failures']),predictor_failures=len(r['predictor_segment_comparison']['failures']))for r in rows],scope='Exact fixed-polynomial containment screening only. Whole known-root, true tangent, polarized domains and quotient still require the full independent replay. No native calls.')


class Work:
    def __init__(self):self.start=time.monotonic();self.used=0
    def tick(self):need(time.monotonic()-self.start<120 and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<2*1024**2,'120s2GiB saved polynomial screen')
    def save(self,name,value,closing=False):
        b=(json.dumps(value,sort_keys=True,separators=(',',':'))+'\n').encode();need(self.used+len(b)<(64 if closing else 60)*1024**2,'64MiB4MiB closing reserve');p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True)
        with p.open('xb')as f:f.write(b);f.flush();os.fsync(f.fileno())
        self.used+=len(b)


def main():
    resource.setrlimit(resource.RLIMIT_AS,(6*1024**3,)*2)
    need(not subprocess.check_output(['git','-c','core.preloadIndex=false','-c','index.threads=1','status','--porcelain'],cwd=ROOT).strip()and os.getpriority(os.PRIO_PROCESS,0)>=10,'clean committed CPU1 nice10 source')
    OUT.mkdir(parents=True,exist_ok=False);w=Work();result=dict(passed=False,native_calls=0)
    def stop(*_):raise TimeoutError('120s saved polynomial screen')
    signal.signal(signal.SIGALRM,stop);signal.signal(signal.SIGTERM,stop);signal.setitimer(signal.ITIMER_REAL,120)
    try:
        source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();w.save('runtime.json',dict(source_commit=source,pid=os.getpid(),sources={n:B.A.S.entry(ROOT/n)for n in FILES},precision=480,native_calls=0,python=platform.python_version(),python_flint=flint.__version__,flint=flint.__FLINT_VERSION__,threads=dict.fromkeys(THREADS,'1')))
        for n in FILES:
            b=(ROOT/n).read_bytes();need(b==subprocess.check_output(['git','show',source+':'+n],cwd=ROOT),'exact committed screen source');need(w.used+len(b)<60*1024**2,'bounded source retention');p=OUT/'source'/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b);w.used+=len(b)
        with ctx.workprec(480):
            t=time.monotonic();x=[arb(i-352)/1024 for i in range(705)];p,v,m=candidates(x,x,x,x,F(-1,64),F(-1,128));need(all(z.is_finite()for a in(p,v,m)for z in a),'finite full705 polynomial profile');seconds=time.monotonic()-t
            report=dict(passed=12*seconds+45<120,seconds=seconds,forecast_seconds=12*seconds+45,target_arrays_unopened=True,scope='Twelve full705 polynomial computations plus45s metadata/source allowance; raw/RSS hard caps remain active.')
            w.save('profile.json',report);need(report['passed'],'full705 screen profile before arrays');del x,p,v,m;gc.collect()
            card=read(ROOT/B.CARD);run,domains,_,_,data,_=B.metadata(w,card);result.update(screen(w,run,domains,data));w.tick()
    except BaseException as exc:result.update(passed=False,error=type(exc).__name__+': '+str(exc),traceback=traceback.format_exc())
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);result.update(elapsed_seconds=time.monotonic()-w.start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss);w.save('outcome.json',result,True);w.save('manifest.json',dict(files={str(p.relative_to(OUT)):B.A.S.entry(p)for p in OUT.rglob('*')if p.is_file()}),True)
    print(json.dumps(result));return 0 if result['passed']else 2
if __name__=='__main__':raise SystemExit(main())
