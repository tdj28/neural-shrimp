"""Fixed, small Clark-response benchmark over existing saved C1 Jacobians."""
from __future__ import annotations
import argparse, hashlib, json, math, os, platform, resource, signal, subprocess, sys, time
from fractions import Fraction
from pathlib import Path
import numpy as np
import scipy
from neural_shrimp.response_spectrum import response, references, classify_count, matrix_free_probe
ROOT=Path(__file__).resolve().parents[1]
SOURCES=['src/neural_shrimp/response_spectrum.py','src/neural_shrimp/__init__.py','src/neural_shrimp/models.py','src/neural_shrimp/controls.py','tools/explore_response_spectrum.py','tools/launch_response_spectrum.py','tests/test_response_spectrum.py','tests/test_response_spectrum_driver.py','docs/plans/clark-response-benchmark.md']
PANEL=[('p10-row35','certificates/c1-row35-receipt.json','07f51a37e8c2304934f0ef60e6e985117cf0e76bf3cf994457374e87bdf6e1f0',10),('p18-row111','certificates/s2-period18-receipt.json','f5b8ed6bc4bded4f6c3515fed5a92d6cfc12614e4e68f0169037810b0d317277',18)]
REPEATS=(4,16,64); ETAS=(1e-2,1e-4,1e-6); SHIFTS=(-.5,-.05,-.005,0.); BOUNDARIES=('free','periodic','causal')
THREADS=('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS')

def digest(data):return hashlib.sha256(data).hexdigest()
def entry(path):b=path.read_bytes();return {'bytes':len(b),'sha256':digest(b)}
def convert(x):
    if isinstance(x,np.ndarray):return convert(x.tolist())
    if isinstance(x,dict):return {str(k):convert(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [convert(v) for v in x]
    if isinstance(x,(float,np.floating)):return float(x) if math.isfinite(x) else None
    if isinstance(x,np.integer):return int(x)
    return x

def encoded(x):return (json.dumps(convert(x),sort_keys=True,indent=2,allow_nan=False)+'\n').encode()

class Store:
    def __init__(self,out,cap=16*1024**2):self.out=Path(out);self.out.mkdir(parents=True,exist_ok=False);self.cap=cap;self.total=0
    def save(self,name,value,final=False):
        data=encoded(value)
        if self.total+len(data)>self.cap-(0 if final else 1024**2):raise RuntimeError('output byte cap')
        with (self.out/name).open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
        self.total+=len(data)
    def finish(self):
        inv={p.name:entry(p) for p in sorted(self.out.iterdir()) if p.is_file()}
        self.save('manifest.json',{'files':inv,'total_bytes':sum(v['bytes'] for v in inv.values())},True)
        for p in self.out.iterdir():p.chmod(0o444)
        self.out.chmod(0o555)

class Budget:
    def __init__(self,seconds):self.start=time.monotonic();self.seconds=seconds;self.stopped=False
    def stop(self,*args):self.stopped=True
    def tick(self):
        if self.stopped or time.monotonic()-self.start>self.seconds:raise TimeoutError('deadline/termination')
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss>480*1024:raise MemoryError('480 MiB monitored RSS')

def virtual_memory():
    return {line.split(':')[0]:line.split(':')[1].strip() for line in Path('/proc/self/status').read_text().splitlines() if line.startswith(('VmPeak:', 'VmSize:', 'VmHWM:', 'VmRSS:'))}

def environment():
    return {'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,'threads':{k:os.environ.get(k) for k in THREADS},'nice':os.getpriority(os.PRIO_PROCESS,0)}

def sources():return {name:entry(ROOT/name) for name in SOURCES}

def load_slot(slot):
    label,rel,pin,period=slot; data=(ROOT/rel).read_bytes()
    if digest(data)!=pin:raise ValueError('receipt mismatch')
    receipt=json.loads(data); external=Path(receipt['external_root']); files={}
    for name,expected in receipt['files'].items():
        p=external/name
        if p.is_symlink() or entry(p)!=expected:raise ValueError('frozen inventory mismatch')
        files[name]=entry(p)
    cert=json.loads((external/'certificate.json').read_bytes()); result=cert['result']
    if not result['passed'] or result['exact_statement']['primitive_period']!=period or result['exact_statement']['dimension']!=13:raise ValueError('wrong fixed slot')
    if cert['inputs_sha256']!=files['inputs.json']['sha256']:raise ValueError('input binding')
    def fraction(pair):return Fraction(int(pair[0]))*Fraction(2)**int(pair[1])
    def mid(ball):return float(fraction(ball['mid']))
    maps=np.array([[[mid(v) for v in row] for row in m] for m in result['jacobians']],dtype=np.float64)
    mon=np.array([[mid(v) for v in row] for row in result['monodromy']])
    width=max(float(fraction(v['rad'])) for m in result['jacobians'] for row in m for v in row)
    if maps.shape!=(period,13,13) or not np.isfinite(maps).all():raise ValueError('bad midpoint maps')
    return maps,mon,{'label':label,'receipt':rel,'receipt_sha256':pin,'external_root':str(external),'files':files,
        'jacobian_midpoints':maps,'midpoints_bytes_sha256':digest(maps.tobytes(order='C')),'dtype':str(maps.dtype),'shape':maps.shape,'order':'C',
        'max_saved_jacobian_ball_radius_float':width,'scope':'floating midpoints of saved enclosure; no new derivative acquisition or certificate claim'}

def profile_gate(path,pin,current):
    path=Path(path)
    if digest((path/'manifest.json').read_bytes())!=pin:raise ValueError('profile manifest pin')
    inv=json.loads((path/'manifest.json').read_bytes())
    if set(inv['files'])!={p.name for p in path.iterdir() if p.is_file()}-{'manifest.json'}:raise ValueError('profile inventory')
    for n,e in inv['files'].items():
        if entry(path/n)!=e:raise ValueError('profile file')
    r=json.loads((path/'result.json').read_bytes()); start=json.loads((path/'start.json').read_bytes())
    if r['status']!='complete' or start['mode']!='profile' or start['sources']!=sources() or start['environment']!=current:raise ValueError('profile scope/source/runtime')
    for key,cap in [('elapsed_seconds',15),('max_rss_kib',400*1024),('projected_panel_seconds',45)]:
        v=r[key]
        if isinstance(v,bool) or not isinstance(v,(float,int)) or not math.isfinite(v) or v<0 or v>=cap:raise ValueError('profile resource gate '+key)
    return {'path':str(path),'manifest_sha256':pin,'result':r}

def run(mode,out,profile=None,profile_pin=None):
    if subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip():raise ValueError('clean source required')
    env=environment()
    if any(v!='1' for v in env['threads'].values()) or env['nice']<10:raise ValueError('CPU1/nice10 required')
    binding=profile_gate(profile,profile_pin,env) if mode=='panel' else None
    store=Store(out); budget=Budget(15 if mode=='profile' else 60); signal.signal(signal.SIGTERM,budget.stop)
    # Virtual-memory hard cap supplements the launcher RSS monitor; no claim of an RSS RLIMIT.
    resource.setrlimit(resource.RLIMIT_AS,(1024*1024**2,1024*1024**2))
    start={'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'sources':sources(),'environment':env,'mode':mode,'pid':os.getpid(),'virtual_memory':virtual_memory(),'started_unix':time.time(),'profile_gate':binding,'settings':{'repeats':REPEATS,'etas':ETAS,'shifts':SHIFTS,'boundaries':BOUNDARIES,'internal_seconds':budget.seconds,'hard_address_space_bytes':1024*1024**2,'monitored_rss_kib':480*1024}}
    store.save('start.json',start);store.save('source-snapshot.json',{name:(ROOT/name).read_text() for name in SOURCES}); result={'status':'incomplete','rows':0,'row_failures':0}
    try:
        if mode=='profile':
            # Deterministic, synthetic 13D/18-phase problem. No empirical reads.
            m=np.diag(np.exp(np.linspace(-.6,.3,13)))+.01*np.triu(np.ones((13,13)),1)+.003*np.tril(np.ones((13,13)),-1);a=np.array([m+((t%3)-1)*.0001*np.triu(np.ones((13,13)),1) for t in range(18)]);times=[]
            grid_start=time.monotonic()
            for rep in REPEATS:
                for shift in SHIFTS:
                    for eta in ETAS:
                        for boundary in BOUNDARIES:
                            number=result['rows'];store.save(f'profile-{number:03d}-start.json',{'repeats':rep,'shift':shift,'eta':eta,'boundary':boundary})
                            r=response(a,18*rep,shift,eta,boundary,budget.tick);r['virtual_memory_after']=virtual_memory();store.save(f'profile-{number:03d}.json',r);times.append(r['elapsed_seconds']);result['rows']+=1;budget.tick()
                            if not r['solve_qualified']:raise ValueError('profile solve residual')
            grid_elapsed=time.monotonic()-grid_start
            mf=matrix_free_probe(a,18*16,-.05,.01,'free',300,budget.tick);store.save('profile-matrixfree.json',mf);budget.tick()
            result.update(status='complete',synthetic_grid_elapsed_seconds=grid_elapsed,projected_panel_seconds=4*grid_elapsed+2*mf['elapsed_seconds']+5)
        else:
            summary=[]
            for slot in PANEL:
                budget.tick();label=slot[0];a,savedmon,identity=load_slot(slot);store.save(label+'-inputs.json',identity)
                ref=references(a);ref['saved_monodromy_midpoint']=savedmon;ref['product_vs_saved_midpoint_frobenius']=float(np.linalg.norm(ref['monodromy']-savedmon));store.save(label+'-reference.json',ref);budget.tick()
                for rep in REPEATS:
                    for shift in SHIFTS:
                        for eta in ETAS:
                            for boundary in BOUNDARIES:
                                budget.tick();number=result['rows'];config={'label':label,'repeats':rep,'shift':shift,'eta':eta,'boundary':boundary}
                                store.save(f'row-{number:03d}-start.json',config)
                                try:
                                    r=response(a,len(a)*rep,shift,eta,boundary,budget.tick);r.update(config);r.update(classify_count(ref,shift,r['count']))
                                except (TimeoutError,MemoryError):raise
                                except Exception as exc:r={**config,'failure':type(exc).__name__+': '+str(exc)};result['row_failures']+=1
                                store.save(f'row-{number:03d}.json',r);result['rows']+=1
                                summary.append({k:v for k,v in r.items() if k not in ('response_block',)})
                # One deterministic impulse diagnoses matrix-free cost/convergence, not a spectral count.
                store.save(label+'-matrixfree-start.json',{'periods':16,'shift':-.05,'eta':.01,'maxiter':300,'boundary':'free','impulse':'e0'});mf=matrix_free_probe(a,len(a)*16,-.05,.01,'free',300,budget.tick);store.save(label+'-matrixfree.json',mf);budget.tick()
                direct=response(a,len(a)*16,-.05,.01,'free',budget.tick);store.save(label+'-matrixfree-direct.json',direct)
                store.save(label+'-matrixfree-comparison.json',{'readout_column_error':float(np.linalg.norm(mf['readout_column']-direct['response_block'][:,0])),'meaning':'diagnostic only; maxiter/convergence reported without relaxed tolerances'})
            store.save('summary.json',summary);result['status']='complete'
    except BaseException as exc:
        result['failure']=type(exc).__name__+': '+str(exc)
    finally:
        result.update(virtual_memory=virtual_memory(),elapsed_seconds=time.monotonic()-budget.start,max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        store.save('result.json',result,True);store.finish()
    print(json.dumps(result),flush=True);return 0 if result['status']=='complete' else 2

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=['profile','panel'],required=True);parser.add_argument('--out',required=True);parser.add_argument('--profile');parser.add_argument('--profile-manifest');args=parser.parse_args()
    if args.mode=='panel' and (not args.profile or not args.profile_manifest):parser.error('panel needs pinned profile')
    raise SystemExit(run(args.mode,args.out,args.profile,args.profile_manifest))
