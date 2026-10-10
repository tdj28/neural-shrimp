#!/usr/bin/env python3
"""Saved-array-only phase scores; no model, correction or eigensystem calls."""
import argparse,hashlib,json,os,resource,subprocess,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
RAW=Path('/data2/neural-shrimp-data/explore/clark-rnn/n12-p6-boundaries-002')
PIN='4d89c7a228458f0c4572767bdbed13609ce4c8ffc1413576fec9c7c435aeadb5'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main(out):
    start=time.monotonic();out.mkdir(parents=True,exist_ok=False);result={'status':'incomplete','pairs':[]}
    resource.setrlimit(resource.RLIMIT_AS,(1024**3,)*2)
    def save(name,obj):
        b=(json.dumps(obj,indent=2,allow_nan=False)+'\n').encode()
        if len(b)+sum(x.stat().st_size for x in out.rglob('*')if x.is_file())>2*1024**2:raise OSError('2MiB raw cap')
        (out/name).write_bytes(b)
    def check():
        if time.monotonic()-start>5:raise TimeoutError('5s saved diagnosis')
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss>448*1024:raise MemoryError('448MiB RSS')
    try:
        assert os.getpriority(os.PRIO_PROCESS,0)>=10
        assert all(os.environ[k]=='1'for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'))
        assert sha(RAW/'manifest.json')==PIN
        inv=json.loads((RAW/'manifest.json').read_text())['files'];assert set(inv)=={str(p.relative_to(RAW))for p in RAW.rglob('*')if p.is_file()and p!=RAW/'manifest.json'}
        for n,d in inv.items():
            p=RAW/n;assert not Path(n).is_absolute()and '..'not in Path(n).parts and not p.is_symlink()and p.stat().st_size==d['bytes']and sha(p)==d['sha256']and not p.stat().st_mode&0o222
        save('runtime.json',{'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'source_sha256':sha(Path(__file__)),'pid':os.getpid(),'input_manifest':PIN,'nice':os.getpriority(os.PRIO_PROCESS,0)})
        (out/'diagnostic-source.py').write_bytes(Path(__file__).read_bytes())
        for p in sorted((RAW/'run').glob('*phase_sign_match.json')):
            check();meta=json.loads(p.read_text());a=RAW/'run'/meta['array_file'];assert sha(a)==meta['array_sha256']
            with np.load(a)as f:parent=f['parent_cycle'];child=f['child_cycle']
            assert parent.shape==child.shape==(6,12)and np.isfinite(parent).all()and np.isfinite(child).all()
            scale=max(1,float(np.max(np.abs(parent))),float(np.max(np.abs(child))))
            scores=[dict(sign=s,phase=k,error=float(np.max(np.abs(parent-s*np.roll(child,k,axis=0)))/scale))for s in [1,-1]for k in range(6)]
            native=scores[0]['error'];best=min(x['error']for x in scores)
            percycle=[]
            for x in [parent,child]:
                den=max(1,float(np.max(np.abs(x))));percycle.append(dict(proper_divisors={str(d):float(np.max(np.abs(x[d]-x[0]))/den)for d in [1,2,3]},sign_reversed_half_wave_error=float(np.max(np.abs(x+np.roll(x,3,axis=0)))/den)))
            result['pairs'].append(dict(tag=meta['tag'],original_passed=meta['passed'],source_json_sha256=sha(p),source_array_sha256=sha(a),native_error=native,best_error=best,native_minus_best=native-best,scores=scores,cycles=percycle))
        result['status']='completed';check()
    except BaseException as e:result.update(status='failed_or_interrupted',error=type(e).__name__+': '+str(e))
    finally:
        result.update(elapsed_seconds=time.monotonic()-start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,scope='Saved numerical representation diagnosis only; no old stop relabeling, new trajectory or boundary claim.');save('result.json',result)
        save('manifest.json',{'files':{str(p.relative_to(out)):{'bytes':p.stat().st_size,'sha256':sha(p)}for p in out.rglob('*')if p.is_file()}})
    return 0 if result['status']=='completed'else 2
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);raise SystemExit(main(p.parse_args().out))
