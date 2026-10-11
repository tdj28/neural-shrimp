"""All-six saved rigorous domain screen before any complementary cell replay.

Each domain independently rebuilds its known-root enclosure and, only after all
actual acquisition checks, its full tangent and saved native quotient blocks.
Failures are retained per domain; none is bypassed. Zero native/model calls.
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
from review_cubic_polynomial_domain import bounded_domain
need,read=B.need,B.read
OUT=Path('/data2/neural-shrimp-data/audits/native705-cubic32-full-domains/attempt001/run')
CARD='docs/plans/native705-cubic-full-domains-inputs.json'
FAILURE=dict(root='/data2/neural-shrimp-data/audits/native705-cubic32-remaining-independent/attempt004',source='4d0b50738f48616f69c46fea6d62aecc22cd61b4',outer='58b7d8f70fdf0268ce9a3fa8e81c1d87f74723b122f8cd8f1714dbbcb6706030',worker='e88bc1b40ca91a612e2a15004d23b45fff927bd940ab5045d10481dfb854c2f4')
FILES=sorted(set(B.FILES+[f'tools/reviews/{prefix}{name}.py'for prefix in('','native704_aux/')for name in('review_joined_period18_math','review_quadratic_connection_math','review_unit_pair_math')]+['tools/reviews/diagnose_cubic_full_domains.py','tools/reviews/launch_cubic_full_domains.py','tests/reviews/test_cubic_full_domains.py','tests/reviews/test_cubic_exact_ranges.py','docs/plans/native705-cubic-full-domains.md',CARD]))


def card_check(card):
    need(card==dict(status='reviewed',failed_replay=FAILURE,domains=list(range(1,7)),precision=480,native_calls=0,worker_seconds=360,worker_rss_gib=4,worker_raw_mib=256),'exact reviewed all-six diagnostic card')


def screen(w,run,domains,data):
    need([s['domain']for s in domains]==B.domains(),'all six canonical domains in source order')
    rows=[]
    for spec in domains:
        w.tick();row=dict(domain=spec['domain'],passed=False)
        try:
            shared=C.shared_domain(w,run,spec,data)
            need(len(shared['whole'])==len(shared['tangent'])==705 and len(shared['derivatives'])==18,'entire successful derivative-domain screen')
            row.update(passed=True,dimension=705,derivative_phases=18)
            del shared
        except ValueError as exc:
            # Mathematical/domain rejection is a recorded result, not permission
            # to evaluate later stages on an uncovered domain. Resource failures
            # use separate exception types and terminate the entire worker.
            row.update(error=str(exc),traceback=traceback.format_exc())
        w.tick();w.save(f"domains/{spec['domain']['index']:03d}/screen-result.json",row);rows.append(row);gc.collect()
    return dict(passed=all(r['passed']for r in rows),domains=rows,scope='All-six same-root/whole/segment/polarized/true-tangent/native acquisition-domain screening only. No center-cell or 11-mode metric proof, no native calls, no extension of accepted complementary coverage.')


class Work:
    def __init__(self):self.start=time.monotonic();self.used=0
    def tick(self):
        if time.monotonic()-self.start>=360:raise TimeoutError('360s saved full-domain screen')
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss>=4*1024**2:raise MemoryError('4GiB saved full-domain screen')
    def save(self,name,value,closing=False):
        b=(json.dumps(value,sort_keys=True,separators=(',',':'))+'\n').encode()
        if self.used+len(b)>=(256 if closing else 252)*1024**2:raise OSError('256MiB4MiB closing reserve')
        p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True)
        with p.open('xb')as f:f.write(b);f.flush();os.fsync(f.fileno())
        self.used+=len(b)


def main():
    resource.setrlimit(resource.RLIMIT_AS,(6*1024**3,)*2)
    need(not subprocess.check_output(['git','-c','core.preloadIndex=false','-c','index.threads=1','status','--porcelain'],cwd=ROOT).strip()and os.getpriority(os.PRIO_PROCESS,0)>=10,'clean committed CPU1 nice10 source')
    card=read(ROOT/CARD);card_check(card);OUT.mkdir(parents=True,exist_ok=False);w=Work();result=dict(passed=False,native_calls=0)
    def stop(*_):raise TimeoutError('360s saved full-domain screen')
    signal.signal(signal.SIGALRM,stop);signal.signal(signal.SIGTERM,stop);signal.setitimer(signal.ITIMER_REAL,360)
    try:
        source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();w.save('runtime.json',dict(source_commit=source,pid=os.getpid(),sources={n:B.A.S.entry(ROOT/n)for n in FILES},precision=480,native_calls=0,python=platform.python_version(),python_flint=flint.__version__,flint=flint.__FLINT_VERSION__,threads=dict.fromkeys(THREADS,'1'),authorship='Producer-agent diagnostic adapter; unchanged independently authored root/tangent/native reducers, root review and execution required.'))
        for n in FILES:
            b=(ROOT/n).read_bytes();need(b==subprocess.check_output(['git','show',source+':'+n],cwd=ROOT),'exact committed diagnostic source')
            if w.used+len(b)>=252*1024**2:raise OSError('bounded source retention')
            p=OUT/'source'/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b);w.used+=len(b)
        with ctx.workprec(480):
            old,rt,failed,inventory=B.A.S.bundle(FAILURE,w.tick,2)
            need(failed['passed']is False and failed['native_calls']==0 and len(failed['completed_cells'])==2 and 'actual segment'in failed['traceback'],'literal preserved004 domain failure')
            w.save('failed-replay-binding.json',dict(card=card,inventory=inventory,retained_outcome=failed,scope='Mandatory immutable failed replay, including its prior failures and original source archive.'))
            t=time.monotonic();trial=B.A.A.synthetic(705,w.tick);x=[arb(i-352)/1024 for i in range(705)];p,v,e=bounded_domain(x,x,x,x,F(-1,64),F(-1,128));need(all(z.is_finite()for a in(p,v)for z in a),'finite full705 exact-rational polynomial profile');seconds=time.monotonic()-t
            manifest=read(old/'run/manifest.json');raw=w.used+6*sum(v['bytes']for v in manifest['files'].values())+64*1024**2
            forecast=12*seconds+6*failed['elapsed_seconds']+90
            ok=trial['passed']and forecast<360 and raw<256*1024**2 and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss+512*1024<4*1024**2
            w.save('profile.json',dict(passed=ok,seconds=seconds,forecast_seconds=forecast,forecast_bytes=raw,target_arrays_unopened=True,reference=FAILURE,scope='Six times the complete cold004 replay plus twelve fresh full705 synthetic/polynomial units and90s allowance; hard caps unchanged.'))
            need(ok,'full705 cold-cost measured screen admission');del trial,x,p,v,e;gc.collect()
            previous=read(ROOT/B.CARD);run,domains,_,_,original,pc=B.metadata(w,previous);data=B.context(w,original,pc);result.update(screen(w,run,domains,data));w.tick()
    except BaseException as exc:result.update(passed=False,error=type(exc).__name__+': '+str(exc),traceback=traceback.format_exc())
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);result.update(elapsed_seconds=time.monotonic()-w.start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss);w.save('outcome.json',result,True);w.save('manifest.json',dict(files={str(p.relative_to(OUT)):B.A.S.entry(p)for p in OUT.rglob('*')if p.is_file()}),True)
    print(json.dumps(result));return 0 if result['passed']else 2
if __name__=='__main__':raise SystemExit(main())
