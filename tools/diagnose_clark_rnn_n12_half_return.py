#!/usr/bin/env python3
"""Fixed saved endpoints: native half-wave identity and signed half-return spectrum."""
import argparse,hashlib,json,os,re,resource,subprocess,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
RAW=Path('/data2/neural-shrimp-data/explore/clark-rnn/n12-p6-boundaries-003')
PIN='b6131a7bb3cf9760cd76148ca86d750b38063c4611c2fdae03caa5df07ec09b8'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def context(cycle,phase,monodromy):
    n=cycle.shape[1]
    if cycle.shape!=(6,n)or phase.shape!=(6,n,n)or monodromy.shape!=(n,n):raise ValueError('native6 dimensions')
    if not all(np.isfinite(a).all()for a in [cycle,phase,monodromy]):raise ValueError('finite saved arrays')
    first=np.eye(n);second=np.eye(n)
    for a in phase[:3]:first=a@first
    for a in phase[3:]:second=a@second
    signed=-first;e,v=np.linalg.eig(signed)
    scale=max(1,float(np.max(np.abs(cycle))));ms=max(1,float(np.max(np.abs(monodromy))))
    return {'half_wave_relative_error':float(np.max(np.abs(cycle+np.roll(cycle,3,axis=0)))/scale),'half_derivative_relative_error':float(np.max(np.abs(first-second))/max(1,np.max(np.abs(first)),np.max(np.abs(second)))),'full_product_relative_error':float(np.max(np.abs(second@first-monodromy))/ms),'square_relative_error':float(np.max(np.abs(first@first-monodromy))/ms),'proper_divisor_separations':{str(d):float(np.max(np.abs(cycle[d]-cycle[0]))/scale)for d in [1,2,3]},'signed_half_jacobian':signed.tolist(),'signed_half_eigenvalues_real':e.real.tolist(),'signed_half_eigenvalues_imag':e.imag.tolist(),'signed_half_eigen_residual':float(np.max(np.abs(signed@v-v*e[None,:]))/max(1,np.max(np.abs(signed)))),'closest_signed_half_plus1':[float(e[np.argmin(np.abs(e-1))].real),float(e[np.argmin(np.abs(e-1))].imag)],'closest_signed_half_minus1':[float(e[np.argmin(np.abs(e+1))].real),float(e[np.argmin(np.abs(e+1))].imag)]}
def main(out):
    start=time.monotonic();out.mkdir(parents=True,exist_ok=False);result={'status':'incomplete','records':[]};resource.setrlimit(resource.RLIMIT_AS,(1024**3,)*2)
    def save(name,obj):
        b=(json.dumps(obj,indent=2,allow_nan=False)+'\n').encode()
        if len(b)+sum(x.stat().st_size for x in out.rglob('*')if x.is_file())>2*1024**2:raise OSError('2MiB raw cap')
        (out/name).write_bytes(b)
    def check():
        if time.monotonic()-start>10:raise TimeoutError('10s saved diagnosis')
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
        selected=[];last={}
        for p in sorted((RAW/'run').glob('*native_eigen.json')):
            d=json.loads(p.read_text());tag=d['tag']
            if re.fullmatch(r'g(16|25|34)-seed[01]',tag):selected.append((p,d,'seed'))
            m=re.fullmatch(r'g(16|25|34)-dir(-?1)-step(\d+)',tag)
            if m:
                key=(m[1],m[2]);step=int(m[3])
                if key not in last or step>last[key][0]:last[key]=(step,p,d)
        selected += [(p,d,'last_closed_direction')for step,p,d in last.values()]
        assert sum(k=='seed'for p,d,k in selected)==6 and len(last)==6
        for p in sorted((RAW/'run').glob('*boundary_result.json')):selected.append((p,json.loads(p.read_text()),'unqualified_boundary_proposal'))
        for p,d,kind in selected:
            check();sources=[p]
            if kind!='unqualified_boundary_proposal':
                previous=int(p.name.split('-')[0])-1;ns=list((RAW/'run').glob(f'{previous:05d}-native.json'));assert len(ns)==1;native=json.loads(ns[0].read_text());assert native['tag']==d['tag'];sources.append(ns[0]);a=RAW/'run'/native['array_file'];assert sha(a)==native['array_sha256']
                with np.load(a)as f:cycle=f['cycle'];phase=f['phase_jacobians'];mon=f['monodromy'];theta=f['theta']
                eigenpath=RAW/'run'/d['array_file'];assert sha(eigenpath)==d['array_sha256'];sources.append(eigenpath)
                with np.load(eigenpath)as f:assert np.array_equal(f['theta'],theta)and np.array_equal(f['monodromy'],mon)
            else:
                a=RAW/'run'/d['array_file'];assert sha(a)==d['array_sha256']
                with np.load(a)as f:cycle=f['cycle'];phase=f['phase_jacobians'];mon=f['monodromy']
            sources.append(a);result['records'].append(dict(tag=d['tag'],kind=kind,gain=d['gain'],delta=d['delta']if'delta'in d else native['delta'],original_status=d.get('status'),source_pins={str(x.relative_to(RAW)):sha(x)for x in sources},**context(cycle,phase,mon)))
        result['status']='completed';check()
    except BaseException as e:result.update(status='failed_or_interrupted',error=type(e).__name__+': '+str(e))
    finally:
        result.update(elapsed_seconds=time.monotonic()-start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,scope='Saved numerical identity/spectrum diagnostics, not exact root proof or fold/flip/route qualification. For exact odd F and F^3(x)=-x, M6=(DF^3)^2; signed half-return G=-F^3 maps DP=+1 to DG=-1, and DP=-1 to DG=+1.');save('result.json',result)
        save('manifest.json',{'files':{str(p.relative_to(out)):{'bytes':p.stat().st_size,'sha256':sha(p)}for p in out.rglob('*')if p.is_file()}})
    return 0 if result['status']=='completed'else 2
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);raise SystemExit(main(p.parse_args().out))
