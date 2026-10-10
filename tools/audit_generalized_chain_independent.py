#!/usr/bin/env python3
"""Independent enclosure arithmetic; shared pinned native derivatives and Arb only."""
import argparse,hashlib,importlib,json,os,resource,signal,subprocess,sys,time
from pathlib import Path
from flint import arb,arb_mat,ctx
ROOT=Path(__file__).resolve().parents[1]
RAW=Path('/data2/neural-shrimp-data/explore/eigenpair-locus-prototype/generalized-chain/attempt001-target')
PRODUCER='b5a62752ba770154009ce694a09f1ede2826a5bd'
OUTER='bfb722e8c08dd4ad23c198f8020d8042fb804750f70bcef58994fbb0fed4d951'
WORKER='e108e6e965c8dfa70a986242bed7232674b051a2b233b2da7f106ecf3bc6e141'
SPECTRUM='99b94ac9444de9754e2ee0b8b6203e6cb70d37366a7489d6c310ad6d1b65591e'
MODULES=['__init__','rigorous_orbits','rigorous_folds','rigorous_cusps','rigorous_eigenpair_locus','rigorous_generalized_chain','rigorous_chain_spectrum']
THREADS=('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')
def require(ok,msg):
    if not ok:raise ValueError(msg)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def ball(x):
    require(isinstance(x,dict) and {'mid','rad'}<=set(x),'ball encoding')
    for k in ('mid','rad'):require(isinstance(x[k],list) and len(x[k])==2 and all(type(i)is int for i in x[k]),'dyadic pair')
    require(x['rad'][0]>=0,'nonnegative radius')
    a=arb(arb(tuple(x['mid'])),arb(tuple(x['rad'])))
    require(a.is_finite(),'finite decoded ball');return a

def enc(a):
    m,e=a.mid().man_exp();r,s=a.rad().man_exp()
    return {'mid':[int(m),int(e)],'rad':[int(r),int(s)],'display':str(a)}
def matenc(a):return [[enc(a[i,j])for j in range(a.ncols())]for i in range(a.nrows())]
def matrix(a):
    require(isinstance(a,list) and a and all(isinstance(r,list)and len(r)==len(a[0])for r in a),'matrix shape')
    return arb_mat([[ball(x)for x in r]for r in a])
def finite(a):return all(a[i,j].is_finite()for i in range(a.nrows())for j in range(a.ncols()))
def krawczyk(center,box,value,jac,pre):
    n=len(center)
    require(n==len(box) and value.nrows()==n and value.ncols()==1 and jac.nrows()==jac.ncols()==pre.nrows()==pre.ncols()==n,'Krawczyk dimensions')
    require(all(x.is_finite() and x.rad()==0 for x in center)and all(x.is_finite()for x in box)and all(finite(a)for a in (value,jac,pre)),'finite exact center and matrices')
    require(all(box[i].contains_interior(center[i])for i in range(n)),'center in interior')
    defect=arb_mat([[int(i==j)for j in range(n)]for i in range(n)])-pre*jac
    norm=max((sum((abs(defect[i,j]).upper()for j in range(n)),arb(0))).upper()for i in range(n))
    correction=-pre*value+defect*arb_mat([[box[i]-center[i]]for i in range(n)])
    inclusion=[bool(box[i].contains_interior(center[i]+correction[i,0]))for i in range(n)]
    return {'passed':bool(all(inclusion)and norm.is_finite()and norm<1),'coordinate_inclusions':inclusion,'contraction':enc(norm),'defect':matenc(defect),'image_correction':matenc(correction)}
def divisors(orbit,box,period,n):
    require(len(orbit)==period+1 and all(len(r)==n for r in orbit),'orbit dimensions')
    result=[]
    for d in range(1,period):
        if period%d:continue
        sep=[orbit[d][i]-box[i]for i in range(n)];w=[i for i,x in enumerate(sep)if x.is_finite()and not x.contains(0)]
        result.append({'divisor':d,'passed':bool(w),'witness_coordinates':w,'separations':[enc(x)for x in sep]})
    return result

def inventory(root,pin):
    require(sha(root/'manifest.json')==pin,'manifest pin')
    inv=json.loads((root/'manifest.json').read_text())['files']
    require(set(inv)=={str(p.relative_to(root))for p in root.rglob('*')if p.is_file()and p!=root/'manifest.json'},'full inventory')
    for name,row in inv.items():
        p=root/name;require(not Path(name).is_absolute()and '..'not in Path(name).parts and not p.is_symlink(),'safe path')
        require(p.stat().st_size==row['bytes']and sha(p)==row['sha256']and not p.stat().st_mode&0o222,'immutable inventory item '+name)
    return {'manifest':pin,'files':len(inv),'bytes':sum(r['bytes']for r in inv.values())}

def main(out):
    require(os.getpriority(os.PRIO_PROCESS,0)>=10 and all(os.environ.get(k)=='1'for k in THREADS),'CPU1 nice10')
    resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,)*2);out.mkdir(parents=True,exist_ok=False)
    start=time.monotonic();ticks=[0];stopped=[False];used=[0]
    signal.signal(signal.SIGTERM,lambda *_:stopped.__setitem__(0,True))
    def check():
        if stopped[0]or time.monotonic()-start>=30:raise TimeoutError('30s audit ceiling')
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss>=2*1024**2:raise MemoryError('2GiB RSS')
    def tick():
        ticks[0]+=1;require(ticks[0]<=72,'72 native steps');check()
    def save(name,data):
        raw=(json.dumps(data,indent=2,allow_nan=False)+'\n').encode();require(used[0]+len(raw)<60*1024**2,'60MiB worker raw')
        p=out/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw);used[0]+=len(raw)
    result={'passed':False,'status':'incomplete','scope':'Independent root/divisor arithmetic and fresh384-bit shared-kernel native evaluation. Conditional quotient applies only after same-box exact chain root proof. No generic unfolding or full branch membership.'}
    try:
        require(int(next(x.split()[1]for x in Path('/proc/meminfo').read_text().splitlines()if x.startswith('MemAvailable:')))*1024>=3*1024**3,'3GiB available host')
        require(not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip(),'clean audit source')
        commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
        for n in ['tools/audit_generalized_chain_independent.py','tools/launch_generalized_chain_independent.py','tests/test_generalized_chain_independent_audit.py','docs/plans/generalized-chain-independent-audit.md']:
            p=out/'audit-source'/n;p.parent.mkdir(parents=True,exist_ok=True);b=(ROOT/n).read_bytes();p.write_bytes(b);used[0]+=len(b)
        bindings={'outer':inventory(RAW,OUTER),'worker':inventory(RAW/'run',WORKER)}
        rt=json.loads((RAW/'run/runtime.json').read_text());require(rt['commit']==PRODUCER and rt['settings']=={'newton_steps':8,'period':18,'precision':256,'radius_exponent':-80},'producer identity/settings')
        for name in MODULES:
            n='src/neural_shrimp/'+name+'.py';b=subprocess.check_output(['git','show',PRODUCER+':'+n],cwd=ROOT)
            if name=='rigorous_chain_spectrum':require(hashlib.sha256(b).hexdigest()==SPECTRUM,'reviewed quotient source')
            else:require(b==(RAW/'run/source'/n).read_bytes()and hashlib.sha256(b).hexdigest()==rt['sources'][n]['sha256'],'exact native source '+n)
            p=out/'native-source'/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b);used[0]+=len(b)
        sys.path.insert(0,str(out/'native-source/src'))
        from neural_shrimp.rigorous_cusps import TwoRateCuspMap
        from neural_shrimp.rigorous_generalized_chain import augmented_chain
        from neural_shrimp.rigorous_chain_spectrum import chain_complement
        import flint,numpy
        save('runtime.json',{'commit':commit,'producer':PRODUCER,'bindings':bindings,'precision':384,'python':sys.version,'numpy':numpy.__version__,'flint':flint.__version__,'flint_library':flint.__FLINT_VERSION__,'threads':{k:os.environ[k]for k in THREADS},'nice':os.getpriority(os.PRIO_PROCESS,0),'pid':os.getpid(),'native_limit':72,'seconds_limit':30,'memory_limit_bytes':2*1024**3})
        inp=json.loads((RAW/'run/inputs.json').read_text());proof=json.loads((RAW/'run/proof.json').read_text())
        require(proof['dimension']==13 and proof['augmented_dimension']==41 and proof['period']==18,'fixed13D period18')
        save('input-binding.json',{'inputs_sha256':sha(RAW/'run/inputs.json'),'proof_sha256':sha(RAW/'run/proof.json'),'map_inputs':inp['map_inputs'],'precision_amendment':'384 bits; outward decoding of saved dyadic radius may enlarge the original ball conservatively; all root and spectrum checks use this same decoded box.'})
        with ctx.workprec(384):
            c=[ball(x)for x in proof['center']];box=[ball(x)for x in proof['box']]
            saved=krawczyk(c,box,matrix(proof['value']),matrix(proof['whole_jacobian']),matrix(proof['preconditioner']))
            saved_div=divisors([[ball(x)for x in row]for row in proof['orbit_enclosures']],box,18,13)
            save('saved-arithmetic.json',{'root':saved,'divisors':saved_div})
            require(saved['passed']and all(d['passed']for d in saved_div),'saved strict root/divisor audit')
            model=TwoRateCuspMap(inp['map_inputs']);require(model.n==13,'native model dimension')
            before=time.monotonic();value,jac,_=augmented_chain(model,c,18,tick)
            inverse=jac.inv();require(finite(inverse),'finite center inverse');pre=arb_mat([[inverse[i,j].mid()for j in range(41)]for i in range(41)])
            save('fresh-center.json',{'center':[enc(x)for x in c],'value':matenc(value),'jacobian':matenc(jac),'preconditioner':matenc(pre),'seconds':time.monotonic()-before});check()
            before=time.monotonic();_,whole,data=augmented_chain(model,box,18,tick)
            fresh=krawczyk(c,box,value,whole,pre);div=divisors(data['orbit'],box,18,13)
            save('fresh-box.json',{'box':[enc(x)for x in box],'whole_jacobian':matenc(whole),'monodromy':matenc(data['monodromy']),'orbit_enclosures':[[enc(x)for x in row]for row in data['orbit']],'root':fresh,'divisors':div,'seconds':time.monotonic()-before})
            require(fresh['passed']and all(d['passed']for d in div),'fresh strict root/divisor audit');check()
            spectrum=chain_complement(data['monodromy'],box[15:28],box[28:41]);save('quotient-spectrum.json',spectrum);check()
            result.update(passed=True,status='completed',root_passed=True,primitive_period=18,strict_inclusions=sum(fresh['coordinate_inclusions']),fresh_contraction=fresh['contraction'],quotient_unit_circle_excluded=spectrum['passed'],complement_attracting=spectrum['complement_attracting'],exact_algebraic_multiplicity_two=spectrum['passed'])
    except BaseException as e:result.update(status='failed_or_interrupted',error=type(e).__name__+': '+str(e))
    finally:
        result.update(native_steps=ticks[0],elapsed_seconds=time.monotonic()-start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss);save('outcome.json',result)
        files={str(p.relative_to(out)):{'bytes':p.stat().st_size,'sha256':sha(p)}for p in out.rglob('*')if p.is_file()};save('manifest.json',{'files':files})
    return 0 if result['passed']else 2
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);sys.exit(main(p.parse_args().out))
