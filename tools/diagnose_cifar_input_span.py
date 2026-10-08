"""One synthetic QR resource profile, then one fixed saved-array decomposition; no model evaluation."""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[name]='1'
import argparse
from datetime import datetime,timezone
from fractions import Fraction
import hashlib,io,json,math,platform,resource,signal,subprocess,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src/neural_shrimp'))
from cifar_input_span import Packing,basis_checks,decay,error_check,finite,identities,linear_identities,norm,project,require,saved_relations,thin_basis,vector_checks
DATA=Path('/data2/neural-shrimp-data/explore/cifar')
BUNDLE=DATA/'device-benchmark/reference-001';LINEAR=DATA/'linear-diagnostic/attempt-001';OUTROOT=DATA/'input-span'
PINS={'reference':'414e8f2b1491a7f28cd1e1522376f7b008d7d3ac57bd29f69f0d5f4b953add38','linear':'d47983793f58b033389fca5fc4aedc9ad78798b38cef06b2f3a4cf0f731534a3'}
LINEAR_SOURCE='a9830e714d259597a53cee6ef7be41a9f07bb205'
REFERENCE_SOURCE='a3debab72195d5ac58bee5562cb59afd8b6466e8'
SOURCES=('src/neural_shrimp/cifar_input_span.py','tools/diagnose_cifar_input_span.py','tests/test_cifar_input_span.py','src/neural_shrimp/cifar.py','docs/plans/cifar-input-span-diagnostic.md','docs/plans/cifar-input-span-execution.md')
STATE_NAMES=('original','trial-0','trial-1','trial-2','trial-3')
REFERENCE_NAMES={'reference.npz','result.json'}
LINEAR_NAMES={'applied-direction.npy','direction.npy','events.jsonl','initial-map1.npy','initial-map2.npy','initial-residual.npy','initial-theta.npy','initial.json','input-identity.json','linear.json','parity.json','result.json','returned-direction.json','runtime-start.json','true-linear-residual.npy'}|{f'trial-{i}-{s}.npy' for i in range(4) for s in ('map1','map2','residual','theta')}
SETTINGS={'d':3072,'n':512,'width':32,'k':10,'precision':'float64','period':2,'retained_columns':512,'orthogonality_limit':1e-10,'representation_limit':1e-12,'identity_atol':1e-12,'identity_rtol':1e-8,'triage_fraction':.5,'profile_seconds':5,'target_seconds':30,'raw_cap':64*1024**2,'reserve':1024**2,'external_reserve':512*1024,'synthetic_seed':28240}

def sha(b):return hashlib.sha256(b).hexdigest()
def entry(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024**2),b''):h.update(b)
    return {'bytes':p.stat().st_size,'sha256':h.hexdigest()}
def read(p):return json.loads(p.read_bytes())
def canonical(v):return (json.dumps(v,indent=2,sort_keys=True,allow_nan=False)+'\n').encode()
def safe_name(name):
    require(isinstance(name,str) and Path(name).name==name and name not in ('.','..'),'safe output name')

def array_info(a):return {'shape':list(a.shape),'dtype':a.dtype.str,'order':'C','sha256':sha(np.ascontiguousarray(a).tobytes())}

def inventory(folder,pin,expected=None):
    require(folder.is_dir() and not folder.is_symlink(),'real inventory directory')
    m=folder/'manifest.json';require(m.is_file() and not m.is_symlink(),'real manifest')
    require(entry(m)['sha256']==pin,'manifest hash')
    rows=read(m);require(isinstance(rows,dict),'inventory object')
    names=set(rows);require(expected is None or names==set(expected),'exact manifest inventory')
    require({p.name for p in folder.iterdir()}==names|{'manifest.json'},'exact directory inventory')
    for name,value in rows.items():
        p=folder/name;require(Path(name).name==name and name not in ('.','..') and p.is_file() and not p.is_symlink(),'safe artifact name')
        require(entry(p)==value,'artifact hash '+name)
    return rows

class Store:
    def __init__(self,folder,cap=SETTINGS['raw_cap'],reserve=SETTINGS['reserve'],external_reserve=SETTINGS['external_reserve']):
        self.folder=folder;self.cap=cap;self.reserve=reserve;self.external=external_reserve;folder.mkdir(parents=True,exist_ok=False)
    def used(self):return sum(p.stat().st_size for p in self.folder.iterdir() if p.is_file())
    def write(self,name,raw,final=False):
        safe_name(name)
        limit=self.cap-(self.external if final else self.reserve)
        require(self.used()+len(raw)<=limit,'retained-byte cap')
        with (self.folder/name).open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
    def json(self,name,value,final=False):self.write(name,canonical(value),final)
    def array(self,name,value):
        safe_name(name)
        a=finite(value);require(self.used()+a.nbytes+256<=self.cap-self.reserve,'retained-byte cap')
        with (self.folder/name).open('xb') as f:np.save(f,a,allow_pickle=False);f.flush();os.fsync(f.fileno())
        return entry(self.folder/name)
    def finalize(self,result):
        self.json('result.json',result,True)
        entries={p.name:entry(p) for p in sorted(self.folder.iterdir()) if p.is_file()}
        self.json('manifest.json',entries,True)


def environment():
    cpu=next((v.split(':',1)[1].strip() for v in Path('/proc/cpuinfo').read_text().splitlines() if v.startswith('model name')),'unknown')
    return dict(python=platform.python_version(),numpy=np.__version__,platform=platform.platform(),host=platform.node(),cpu=cpu,
                blas=np.show_config(mode='dicts'),thread_env={k:os.environ[k] for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')})

def source_identity():
    require(not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip(),'clean source checkout')
    hashes={}
    for name in SOURCES:
        raw=(ROOT/name).read_bytes();require(raw==subprocess.check_output(['git','show','HEAD:'+name],cwd=ROOT),'committed source '+name);hashes[name]=sha(raw)
    return {'commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'sha256':hashes}

def validate_profile_records(r,p,start,current_sources,current_environment,review):
    require(r['schema']==start['schema']=='cifar-input-span-v1' and r['mode']==start['mode']=='profile','profile action/schema')
    require(start['status']=='running' and r['status']=='profile_passed' and p['status']=='profile_passed','profile completed')
    require(r['source']==start['source'] and r['source']['sha256']==current_sources['sha256'],'profile source bytes/commit')
    require(set(r['source']['sha256'])==set(SOURCES),'complete profile source inventory')
    require(r['environment']==start['environment']==current_environment and r['review_commit']==start['review_commit']==review,'profile environment/review')
    require(r['settings']==start['settings']==SETTINGS and r['pid']==start['pid'] and type(r['pid']) is int and r['pid']>0,'profile settings/PID')
    require(r['cpu_threads']==start['cpu_threads']==1 and r['nice']==start['nice']>=10,'profile thread/nice')
    require(r['gpu'] is start['gpu'] is False and r['paid'] is start['paid'] is False and r['dynamics_evaluations']==start['dynamics_evaluations']==0,'profile resource scope')
    require(r['started_utc']==start['started_utc'] and isinstance(r['finished_utc'],str),'profile runtime endpoints')
    require(all(type(v) in (int,float) and math.isfinite(v) for v in (p['qr_seconds'],r['elapsed_seconds'],r['max_rss_kib'])),'finite profile resource')
    require(0<p['qr_seconds']<=r['elapsed_seconds']<=5 and r['max_rss_kib']>0,'profile actual time/resource')
    require(p['shape']==[3072,512] and p['seed']==28240 and p['retained_columns']==512,'profile shape')
    for name,shape in [('x',[512,3072]),('q',[3072,512]),('r',[512,512])]:
        row=p[name];require(row['shape']==shape and row['dtype']=='<f8' and row['order']=='C','profile array identity')
        require(isinstance(row['sha256'],str) and len(row['sha256'])==64 and all(c in '0123456789abcdef' for c in row['sha256']),'profile array digest')
    c=p['basis_checks'];require(c['passed'] is True and c['retained_columns']==512 and c['rank_truncation'] is False,'profile basis scope')
    for name,cap in [('orthogonality_frobenius',1e-10),('x_i_minus_p_relative',1e-12),('representation_relative',1e-12),('implicit_idempotence_on_q',1e-10),('qr_reconstruction_relative',1e-12)]:
        v=c[name];require(type(v) in (int,float) and math.isfinite(v) and 0<=v<=cap,'profile numerical gate '+name)
    require(c['x_i_minus_p_relative']==c['representation_relative'],'profile representation mirror')
    projection=2*p['qr_seconds']+5
    require(p['projected_target_seconds']==projection<=30,'profile cost projection')
    require(r['profile']==p,'profile/result mirror')
    return projection

def profile_gate(folder,current_sources,current_environment,review,approved_manifest_sha256):
    require(folder==OUTROOT/'profile-001','fixed profile path')
    require(isinstance(approved_manifest_sha256,str) and len(approved_manifest_sha256)==64 and all(c in '0123456789abcdef' for c in approved_manifest_sha256),'external profile manifest pin')
    rows=inventory(folder,approved_manifest_sha256,{'runtime-start.json','profile.json','result.json'})
    r=read(folder/'result.json');p=read(folder/'profile.json');start=read(folder/'runtime-start.json')
    projection=validate_profile_records(r,p,start,current_sources,current_environment,review)
    for name,expected in r['source']['sha256'].items():
        require(sha(subprocess.check_output(['git','show',r['source']['commit']+':'+name],cwd=ROOT))==expected,'profile committed bytes')
    require(sum(v['bytes'] for v in rows.values())+(folder/'manifest.json').stat().st_size<=SETTINGS['raw_cap']-SETTINGS['external_reserve'],'profile bytes')
    return {'path':str(folder),'manifest':entry(folder/'manifest.json'),'source_commit':r['source']['commit'],'projection_seconds':projection,
            'retained_bytes':sum(v['bytes'] for v in rows.values())+(folder/'manifest.json').stat().st_size}


def inputs():
    b_inventory=inventory(BUNDLE,PINS['reference'],REFERENCE_NAMES);l_inventory=inventory(LINEAR,PINS['linear'],LINEAR_NAMES)
    basis=read(BUNDLE/'result.json');result=read(LINEAR/'result.json');runtime=read(LINEAR/'runtime-start.json');ident=read(LINEAR/'input-identity.json')
    require(basis['status']=='complete' and basis['source_commit']==REFERENCE_SOURCE,'reference source/status')
    require(result['status']=='diagnostic_complete' and result['source_commit']==runtime['source_commit']==LINEAR_SOURCE,'linear source/status')
    for meta in (basis,runtime):
        for name,value in meta['source_sha256'].items():
            require(sha(subprocess.check_output(['git','show',meta['source_commit']+':'+name],cwd=ROOT))==value,'historical source '+name)
    require(result['source_sha256']==runtime['source_sha256'],'linear source mirror')
    require(ident==result['input_identity'] and ident['data_identity']==basis['data_identity'],'data identity')
    require(ident['bundle']==str(BUNDLE) and ident['bundle_manifest_sha256']==PINS['reference'],'reference path/manifest')
    require(basis['model']==basis['data_identity']['model']=={'d':3072,'width':32,'k':10},'model packing')
    require(basis['eta']==ident['eta']==.00827753 and basis['lam']==ident['lam']==.01,'binary64 rate/ridge identity')
    with np.load(BUNDLE/'reference.npz',allow_pickle=False) as z:
        require(set(z.files)=={'x','y','theta','directions','gradient','hvp','j2vp'},'reference members')
        # Existing derivative reference arrays are inventoried but not used or reevaluated.
        x=finite(z['x'],(512,3072));y=finite(z['y'],(512,10));theta=finite(z['theta'],(98666,))
    for name,a in [('x',x),('y',y)]:require(array_info(a)==basis['data_identity'][name],'subset-A byte identity '+name)
    vectors={};map2={}
    for label in STATE_NAMES:
        prefix='initial' if label=='original' else label
        for what,suffix in [('state','theta'),('residual','residual')]:vectors[label+'-'+what]=finite(np.load(LINEAR/(prefix+'-'+suffix+'.npy'),allow_pickle=False),(98666,))
        map2[label]=finite(np.load(LINEAR/(prefix+'-map2.npy'),allow_pickle=False),(98666,))
    for name,file in [('p','direction.npy'),('ap','applied-direction.npy'),('e','true-linear-residual.npy')]:vectors[name]=finite(np.load(LINEAR/file,allow_pickle=False),(98666,))
    saved_relations(theta,vectors,map2,result['trials'])
    return x,vectors,dict(reference_manifest=PINS['reference'],linear_manifest=PINS['linear'],reference_source=REFERENCE_SOURCE,linear_source=LINEAR_SOURCE,
      reference_inventory=b_inventory,linear_inventory=l_inventory,data_identity=basis['data_identity'],model=basis['model'],eta=basis['eta'],lam=basis['lam'],
      array_identities={n:array_info(v) for n,v in vectors.items()},x=array_info(x),y=array_info(y),ridge_normalization='lambda/2 sum(theta**2), gradient lambda*theta; data loss0.5 sum(error**2)/n, not averaged over outputs',state_order=list(STATE_NAMES))


def run(mode,out,review,profile_manifest_sha256=None):
    require(mode in ('profile','collect'),'fixed modes');require(out.resolve()==OUTROOT/('profile-001' if mode=='profile' else 'attempt-001'),'fixed exclusive output')
    require(len(review)==40 and all(c in '0123456789abcdef' for c in review),'exact review commit')
    subprocess.run(['git','cat-file','-e',review+'^{commit}'],cwd=ROOT,check=True,stdout=subprocess.DEVNULL)
    require(os.getpriority(os.PRIO_PROCESS,0)>=10,'nice10')
    require((mode=='profile' and profile_manifest_sha256 is None) or (mode=='collect' and isinstance(profile_manifest_sha256,str) and len(profile_manifest_sha256)==64 and all(c in '0123456789abcdef' for c in profile_manifest_sha256)),'collect requires approved profile manifest SHA256; profile does not accept one')
    source=source_identity();env=environment();store=Store(out);t0=time.monotonic()
    record=dict(schema='cifar-input-span-v1',mode=mode,status='running',source=source,review_commit=review,approved_profile_manifest_sha256=profile_manifest_sha256,environment=env,settings=SETTINGS,
      started_utc=datetime.now(timezone.utc).isoformat(),pid=os.getpid(),cpu_threads=1,nice=os.getpriority(os.PRIO_PROCESS,0),gpu=False,paid=False,
      dynamics_evaluations=0,scope='Saved-array numerical diagnostic; no orbit, convergence, solver-stall explanation or utility claim.')
    store.json('runtime-start.json',record)
    def stop(*_):raise TimeoutError('bounded input-span diagnostic interrupted')
    old_alarm=signal.signal(signal.SIGALRM,stop);old_term=signal.signal(signal.SIGTERM,stop);signal.setitimer(signal.ITIMER_REAL,5 if mode=='profile' else 30)
    try:
      if mode=='profile':
        x=np.random.default_rng(28240).normal(size=(512,3072)).astype(np.float64);q0=time.monotonic();q,r=thin_basis(x);elapsed=time.monotonic()-q0
        checks=basis_checks(x,q,r);projected=2*elapsed+5
        profile=dict(status='profile_passed' if checks['passed'] and projected<=30 else 'profile_failed',shape=[3072,512],seed=28240,
          x=array_info(x),q=array_info(q),r=array_info(r),qr_seconds=elapsed,basis_checks=checks,retained_columns=512,projected_target_seconds=projected)
        store.json('profile.json',profile);record.update(profile=profile,status=profile['status'])
      else:
        profile=profile_gate(OUTROOT/'profile-001',source,env,review,profile_manifest_sha256)
        store.cap-=profile['retained_bytes']  # Shared64MiB unit cap includes the completed profile.
        store.json('profile-binding.json',profile)
        x,vectors,bound=inputs();store.json('input-binding.json',bound)
        q0=time.monotonic();q,r=thin_basis(x);qr_seconds=time.monotonic()-q0
        store.array('basis-q.npy',q);store.array('basis-r.npy',r);checks=basis_checks(x,q,r)
        store.json('basis-checks.json',{**checks,'qr_seconds':qr_seconds});require(checks['passed'],'basis qualification failed')
        packing=Packing();factors=decay(bound['eta'],bound['lam']);store.json('decay-factor.json',factors)
        require(factors['elimination_valid'] and factors['omitted_directions_attracting'],'fixed elimination scope')
        results=[];parts={};all_pass=True
        for name,v in vectors.items():
            active,perp=project(v,q,packing);parts[name]=(active,perp)
            store.array(name+'-active.npy',active);store.array(name+'-complement.npy',perp)
            row={'name':name,**vector_checks(v,active,perp,x,q,packing)}
            if name.endswith('-residual'):
                base=name.removesuffix('-residual')+'-state'
                predicted,check=identities(v,perp,parts[base][1],factors['rounded_factor'])
                store.array(name+'-predicted-complement.npy',predicted);row['period2_complement_identity']=check
                row['passed']=bool(row['passed'] and check['passed'])
            store.json(name+'-checks.json',row);results.append(row);all_pass=all_pass and row['passed']
        ap_pred,e_pred,linear=linear_identities(vectors,parts,factors['rounded_factor'])
        store.array('ap-predicted-complement.npy',ap_pred);store.array('e-predicted-complement.npy',e_pred);store.json('linear-identities.json',linear)
        all_pass=all_pass and linear['passed'];fraction=next(r for r in results if r['name']=='original-residual')['complementary_squared_fraction']
        record.update(status='diagnostic_passed' if all_pass else 'identity_check_failed',all_numerical_gates_passed=bool(all_pass),dimensions=packing.dimensions(512),
         state_order=list(STATE_NAMES),vectors=results,linear_identities=linear,decay=factors,profile_binding=profile,basis_checks=checks,
         original_complementary_squared_residual_fraction=fraction,
         prioritize_reduced_solver=bool(all_pass and fraction is not None and fraction>=.5),
         interpretation='Threshold triage only: a large omitted share motivates a separately tested reduced solver; it neither guarantees convergence nor explains the GMRES stall.')
    except BaseException as e:
      record.update(status='failed_or_interrupted',error=type(e).__name__+': '+str(e));store.json('failure.json',record,True)
    finally:
      signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,old_alarm);signal.signal(signal.SIGTERM,old_term);record.update(elapsed_seconds=time.monotonic()-t0,max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,finished_utc=datetime.now(timezone.utc).isoformat())
      store.finalize(record)
    print(json.dumps({'status':record['status'],'elapsed_seconds':record['elapsed_seconds'],'manifest':entry(out/'manifest.json')}))
    return 0 if record['status'] in ('profile_passed','diagnostic_passed') else 2

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('mode',choices=['profile','collect']);parser.add_argument('--out',type=Path,required=True);parser.add_argument('--review-commit',required=True);parser.add_argument('--profile-manifest-sha256');args=parser.parse_args()
    raise SystemExit(run(args.mode,args.out,args.review_commit,args.profile_manifest_sha256))
