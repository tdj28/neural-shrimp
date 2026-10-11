"""Saved-only diagnosis of the first independent coefficient-enclosure failure."""
import json,os,sys,subprocess,time,signal,resource
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'src'),str(Path(__file__).resolve().parent)]
import review_s2_cusps as R
import review_cusp_math as A
from flint import ctx,arb_mat
from neural_shrimp.rigorous_orbits import canonical_bytes,ball_json,matrix_json
FILES=R.FILES+['tools/reviews/diagnose_cusp_reduction.py','tools/reviews/launch_cusp_reduction_diagnostic.py']
def main(out):
    A.need(not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip(),'clean source');A.need(os.getpriority(os.PRIO_PROCESS,0)>=10 and all(os.environ.get(k)=='1'for k in R.THREADS),'CPU1nice10');out.mkdir(parents=True,exist_ok=False);start=time.monotonic();result={'passed':False,'native_calls':0}
    def save(name,value):
        data=canonical_bytes(value);A.need(sum(p.stat().st_size for p in out.rglob('*')if p.is_file())+len(data)<8*1024**2,'8MiB cap')
        with(out/name).open('xb')as f:f.write(data);f.flush();os.fsync(f.fileno())
    def stop(*_):raise TimeoutError('15s diagnostic limit')
    signal.signal(signal.SIGALRM,stop);signal.signal(signal.SIGTERM,stop);signal.setitimer(signal.ITIMER_REAL,15)
    save('runtime-start.json',{'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'source':{n:R.entry(ROOT/n)for n in FILES},'pid':os.getpid(),'nice':os.getpriority(os.PRIO_PROCESS,0),'threads':dict.fromkeys(R.THREADS,'1')})
    for n in FILES:p=out/'source'/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/n).read_bytes())
    try:
        raw=R.BASE/'cusp-a-attempt-001';R.inventory(raw,R.PINS['cusp-a'][0]);p=raw/'run/proof.json';A.need(R.entry(p)['sha256']==R.PINS['cusp-a'][2],'fixed saved proof');proof=R.read(p)
        def observe(kind,row):
            save(kind+'.json',row)
            # An independently enclosed equivalent inverse-transpose operation.
            with ctx.workprec(384):
                inv=A.matrix(proof['border']).inv();n=13;left=inv.transpose()*arb_mat([[0]]*n+[[1]]);w=arb_mat([[left[i,0]for i in range(n)]]);q=arb_mat([[A.restore(v)]for v in proof['box'][15:28]]);h=arb_mat([[A.restore(v)]for v in proof['box'][28:]])
                g=A.matrix(proof['base_gradients']);g1=A.matrix(proof['direction_gradients']);g2=A.matrix(proof['acceleration_gradients']);third=arb_mat([[sum((g2[i,j]*q[j,0]+2*g1[i,j]*h[j,0]for j in range(n)),A.exact(0))]for i in range(n)]);U=arb_mat(2,2)
                for k in range(2):
                    par=arb_mat([[g[i,n+k]]for i in range(n)]);y=inv*arb_mat([[-par[i,0]]for i in range(n)]+[[0]]);U[0,k]=(w*par)[0,0];U[1,k]=(w*arb_mat([[g1[i,n+k]+sum((g1[i,j]*y[j,0]for j in range(n)),A.exact(0))]for i in range(n)]))[0,0]
                save('equivalent-right-border-inverse.json',{'inverse':matrix_json(inv),'left':matrix_json(left),'cubic':ball_json((w*third)[0,0]/6),'unfolding':matrix_json(U),'determinant':ball_json(U.det()),'source_original_cubic':proof['cubic_coefficient'],'source_original_unfolding':proof['unfolding'],'source_original_determinant':proof['unfolding_determinant']})
        reduction=A.arithmetic(proof,384,observe);save('reduction.json',reduction);result['passed']=True
    except BaseException as e:result['error']=type(e).__name__+': '+str(e)
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);result.update(elapsed_seconds=time.monotonic()-start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss);save('result.json',result);save('manifest.json',{'files':{str(p.relative_to(out)):R.entry(p)for p in sorted(out.rglob('*'))if p.is_file()}})
        for p in sorted(out.rglob('*'),reverse=True):p.chmod(0o555 if p.is_dir()else 0o444)
        out.chmod(0o555)
    print(json.dumps(result));return 0 if result['passed']else 2
if __name__=='__main__':raise SystemExit(main(Path(sys.argv[1])))
