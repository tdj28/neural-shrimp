"""Bounded two-cusp replay/nested proof and independent saved arithmetic review."""
import hashlib,json,math,os,platform,resource,signal,subprocess,sys,time,gzip
from pathlib import Path
from fractions import Fraction
import numpy as np
import flint
from flint import ctx
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'src'),str(Path(__file__).resolve().parent)]
from neural_shrimp.rigorous_cusps import TwoRateCuspMap,cusp_proof
from neural_shrimp.rigorous_orbits import canonical_bytes,exact,dyadic
import review_cusp_math as A
BASE=Path('/data2/neural-shrimp-data/explore/certificates/s2-cusps-2026-10-10')
EXEC='c24b47882ff05e32ee141518629f699871a054f6'
PINS={'cusp-a':('5410873af62a19d940c3608cb4b3b03b62c1ca2a71a8c2c6a180872da5aad36a','bc7031bd1650476e21a083610d5643b97ef65dfe98a490c349e0f638203c8563','bc8519ad4463eb0e4beb2df0f4c8d876c3861999d09c0c22276c7e9818aca2ef'), 'cusp-b':('8a4aa8ea7b056e13ad171f825e2df2ba06867fe97680297a2e07cbe4944cfd57','69a4ea158baa8d9e7d4141f12ac8a772b3ac08e92da0eb282d2d40dab877d8d8','3975dbf689ae7c5f2f48603f97b5782d9c8b16492d608c1b08e534d134d34ae7')}
PROFILE=('b3e3675487c89e8460ac164893d71c43a3d0842677d8ef07883937f2eac9c4cd','98f4f9c3fe20fd52e45d041d2f1ec536fc4dd95f8228b65e0442316f9b5d8655')
PARENTS={'cusp-a':('attempt-003-target','1a1b58e678f11db81ca7adb2821ca2ff84056768734073a11e70e004813a71f9','node-+1-031.json'),'cusp-b':('attempt-004-fold-plus','231b1d448fd06b09a180861d4e67569d3ef0f42967557ba4e24dd7b4780284f6','node-+1-027.json')}
SOURCES=['src/neural_shrimp/numerical_boundary.py','src/neural_shrimp/rigorous_folds.py','src/neural_shrimp/rigorous_orbits.py','src/neural_shrimp/__init__.py','src/neural_shrimp/models.py','src/neural_shrimp/controls.py','tools/explore_s2_boundary_curves.py','tests/test_numerical_boundary.py','docs/plans/s2-period18-boundary-curves.md','src/neural_shrimp/rigorous_connection.py','src/neural_shrimp/rigorous_cusps.py','tests/test_rigorous_cusps.py','tools/certify_s2_cusps.py','docs/plans/s2-period18-cusp-pilot.md']
FILES=SOURCES+['tools/reviews/review_s2_cusps.py','tools/reviews/review_cusp_math.py','tools/reviews/review_clark_rnn_c1_math.py','tools/reviews/launch_s2_cusp_review.py','tests/reviews/test_cusp_outcome_review.py','tests/reviews/test_rigorous_cusps_independent.py','docs/plans/s2-cusp-independent-review.md']
THREADS=('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')
SETTINGS={'radius_exponent':-80,'newton_steps':8}
def entry(p):return {'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
def read(p):return json.loads(p.read_bytes())
def inventory(root,pin):
    A.need(root.is_dir()and not root.is_symlink()and entry(root/'manifest.json')['sha256']==pin,'pinned manifest');m=read(root/'manifest.json')['files'];actual={str(p.relative_to(root)):p for p in root.rglob('*')if p.is_file()and p!=root/'manifest.json'};A.need(set(actual)==set(m),'complete inventory')
    for name,p in actual.items():A.need(not p.is_symlink()and not(p.stat().st_mode&0o222)and entry(p)==m[name],'immutable file '+name)
    return {'sha256':pin,'files':len(m),'bytes':sum(v['bytes']for v in m.values())}
def finite(v,cap):return type(v)in(int,float)and math.isfinite(v)and 0<=v<cap

def producer(root,pins,mode):
    summary=inventory(root,pins[0]);inventory(root/'run',pins[1]);run=root/'run';rt=read(run/'runtime-start.json');launch=read(root/'launcher-result.json');outcome=read(run/'outcome.json')
    A.need(rt['source']==launch['source']and rt['mode']==mode and launch['exit_code']==0 and not launch['owned_group_live_pids'],'completed producer identity')
    commit=rt['source']['commit'];A.need(set(rt['source']['files'])==set(SOURCES),'complete transitive source set')
    for name,h in rt['source']['files'].items():
        A.need(entry(ROOT/name)['sha256']==h==entry(run/'source'/name)['sha256']==hashlib.sha256(subprocess.check_output(['git','show',commit+':'+name],cwd=ROOT)).hexdigest(),'source '+name)
    A.need(rt['python']==sys.version and rt['numpy']==np.__version__ and rt['flint']==flint.__version__ and rt['flint_library']==flint.__FLINT_VERSION__,'replay library identity')
    A.need(rt['threads']==dict.fromkeys(THREADS,'1')and rt['nice']>=10,'producer resource configuration')
    for pid in (rt['pid'],launch['launcher_pid'],launch['owned_group_pid']):A.need(not Path('/proc',str(pid)).exists(),'producer PID absent')
    A.need(finite(outcome['elapsed_seconds'],15 if mode=='profile'else 90)and finite(outcome['peak_rss_kib'],512*1024)and finite(launch['elapsed_seconds'],25 if mode=='profile'else 205),'producer bounded resources')
    A.need(rt['settings']==dict(precision=256,period=18,**SETTINGS),'fixed scientific settings')
    if mode=='profile':
        times=outcome['evaluation_seconds'];A.need(outcome['status']=='synthetic_profile_completed'and outcome['resource_pass']is True and outcome['native_steps']==36 and len(times)==2 and all(type(x)in(int,float)and math.isfinite(x)and x>0 for x in times),'profile success')
        A.need(20*max(times)+5==outcome['projected_seconds']<85,'profile measured projection')
    else:A.need(commit==EXEC and outcome['passed']and outcome['exit_code']==0 and outcome['native_steps']==180 and outcome['status']=='cusp_proof_completed','target success/source')
    return rt,outcome,summary

def input_binding(panel,run,inp):
    name,pin,nodefile=PARENTS[panel];parent=Path('/data2/neural-shrimp-data/explore/s2-period18-boundary-curves')/name;inventory(parent,pin)
    node=read(parent/'run'/nodefile);model=read(parent/'run/inputs.json');origin=read(run/'origin.json')
    A.need(origin['manifest_sha256']==pin and origin['parent']==str(parent)and origin['node']==nodefile and origin['node_sha256']==entry(parent/'run'/nodefile)['sha256'],'seed provenance')
    A.need(node['qualification']['qualified']and model['sigma']==1 and inp['period']==18 and inp['map_inputs']==model['map_inputs'],'qualified exact-model input')
    seed=list(map(Fraction,node['point_exact_dyadics']));A.need(len(seed)==28 and list(map(Fraction,inp['approximation'][:28]))==seed,'exact native state/rates/q seed')
    found=None
    with gzip.open(parent/'run/evaluations.jsonl.gz','rt')as f:
        for line in f:
            row=json.loads(line)
            if row['kind']=='evaluation_return'and row['z']==node['point']:found=row
    A.need(found is not None and found['id']==origin['saved_evaluation_id'],'retained seed derivative return')
    q=np.array([float(v)for v in seed[15:]]);M=np.asarray(found['data']['monodromy']);J=np.asarray(found['data']['jacobian']);B=np.block([[M-np.eye(13),q[:,None]],[q[None,:],np.zeros((1,1))]])
    h=np.linalg.solve(B,np.r_[-J[13:26,:13]@q,0])[:13]
    A.need([Fraction(float(x))for x in h]==list(map(Fraction,inp['approximation'][28:])),'exact proposed curvature reconstruction')
    A.need(len(inp['approximation'])==41,'full augmented seed')
    return {'parent_manifest':pin,'native_seed_exact':True,'curvature_proposal_reproduced':True,'model_exact':True,'scope':'Seed provenance and finite proximity only; no branch connection inferred.'}

def main(out):
    A.need(not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip(),'clean reviewer source');A.need(os.getpriority(os.PRIO_PROCESS,0)>=10 and all(os.environ.get(k)=='1'for k in THREADS),'CPU1 nice10')
    out.mkdir(parents=True,exist_ok=False);start=time.monotonic();stop=[False];ticks=[0];serial=[0];result={'passed':False,'panels':{}}
    def save(name,obj,final=False):
        p=out/name;p.parent.mkdir(parents=True,exist_ok=True);b=canonical_bytes(obj);A.need(sum(q.stat().st_size for q in out.rglob('*')if q.is_file())+len(b)<(64 if final else 60)*1024**2,'aggregate output cap')
        with p.open('xb')as f:f.write(b);f.flush();os.fsync(f.fileno())
    def tick():
        ticks[0]+=1
        if stop[0]or time.monotonic()-start>=120 or ticks[0]>740:raise TimeoutError('fixed audit time/step budget')
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss>=1536*1024:raise MemoryError('1.5GiB monitored RSS cap')
    signal.signal(signal.SIGTERM,lambda *_:stop.__setitem__(0,True));signal.signal(signal.SIGALRM,lambda *_:stop.__setitem__(0,True));signal.setitimer(signal.ITIMER_REAL,120)
    save('runtime-start.json',{'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'sources':{p:entry(ROOT/p)for p in FILES},'pid':os.getpid(),'python':sys.version,'numpy':np.__version__,'flint':flint.__version__,'flint_library':flint.__FLINT_VERSION__,'threads':dict.fromkeys(THREADS,'1'),'nice':os.getpriority(os.PRIO_PROCESS,0),'internal_seconds':120,'native_steps_cap':740,'rss_cap_kib':1536*1024})
    for name in FILES:p=out/'source'/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/name).read_bytes())
    try:
        pr,po,ps=producer(BASE/'attempt-002-profile',PROFILE,'profile');save('profile-binding.json',{'runtime':pr,'outcome':po,'inventory':ps})
        for panel,pins in PINS.items():
            base=BASE/(panel+'-attempt-001');run=base/'run';rt,ro,rs=producer(base,pins,'target');A.need(rt['source']['files']==pr['source']['files'],'same measured source')
            A.need(entry(run/'proof.json')['sha256']==pins[2],'proof hash');inp=read(run/'inputs.json');original=read(run/'proof.json');A.need(original['settings']==SETTINGS,'original settings')
            pb=read(run/'profile-binding.json');A.need(pb=={'path':str(BASE/'attempt-002-profile/run'),'manifest_sha256':PROFILE[1]},'measured profile input pin')
            save(panel+'/binding.json',{'inventory':rs,'runtime':rt,'outcome':ro,'proof_sha256':pins[2],'input':input_binding(panel,run,inp)});tick()
            def observe(kind,row):save(panel+f'/observation-{serial[0]:03d}-{kind}.json',row);serial[0]+=1
            if panel=='cusp-a':
                old=Path('/data2/neural-shrimp-data/audits/s2-cusps/attempt-001');inventory(old,'ea2b4066222edc2e2fc1b671b22ea8b64b8003a5a1f9503ad1c2f333805e4e58');inventory(old/'run','84966316809db2173388655f21f449a08d18f0e7189bd6344a8f98f2ca947d93')
                prior=read(old/'run/runtime-start.json');A.need(prior['source_commit']=='f75bf98a69c71b38eaacaf1e324e42a4196e8e76','retained replay source')
                for name in SOURCES:A.need(prior['sources'][name]==entry(ROOT/name),'retained replay same analytic sources')
                saved=old/'run/cusp-a/replay.json';A.need(entry(saved)['sha256']=='5bbc441e1e5fce4855ae0bea6ec5d5f68192eb84f9e0115deb161bc45a9d6a3b','retained replay hash');replay=read(saved)
                save(panel+'/resumed-replay-binding.json',{'root':str(old),'file':str(saved),'sha256':entry(saved)['sha256'],'meaning':'successful original byte replay retained before the later interval-reduction failure; no derivative rerun'})
            else:
                with ctx.workprec(256):
                    z=[exact(v)for v in inp['approximation']];replay=cusp_proof(TwoRateCuspMap(inp['map_inputs']),z,18,**SETTINGS,tick=tick,observe=observe)
            save(panel+'/replay.json',replay);A.need(canonical_bytes(replay)==canonical_bytes(original),'byte-identical original replay');tick()
            reduced=A.arithmetic(original,384);save(panel+'/independent-saved-math.json',reduced);tick()
            with ctx.workprec(384):
                z=[exact(v)for v in inp['approximation']];nested=cusp_proof(TwoRateCuspMap(inp['map_inputs']),z,18,radius_exponent=-100,newton_steps=8,tick=tick,observe=observe)
            save(panel+'/nested-proof.json',nested);A.need(nested['passed']and A.strict_nested(original['box'],nested['box']),'passing strictly nested stronger proof');tick()
            scales=[Fraction(1)]*13+[Fraction(10),Fraction.from_float(.1)]+[Fraction(1)]*13;seed=list(map(Fraction,inp['approximation']));distance=max(max(abs(lo-s),abs(hi-s))/d for (lo,hi),s,d in zip(map(A.endpoints,original['box'][:28]),seed,scales));A.need(distance<Fraction(1,10),'independent whole-box finite proximity')
            result['panels'][panel]={'passed':True,'replay_identical':True,'nested_strict':True,'independent_saved_math':True,'whole_box_scaled_proximity':str(distance),'actual_hidden_rate':original['box'][13],'actual_output_rate':original['box'][14],'cubic_coefficient':reduced['cubic_coefficient'],'unfolding_determinant':reduced['unfolding_determinant'],'contraction':reduced['contraction'],'scope':'Local generic primitive18 cusp; no attracting orbit/window or proved body/boundary connection.'};save(panel+'/summary.json',result['panels'][panel])
        result['passed']=True
    except BaseException as e:result['error']=type(e).__name__+': '+str(e)
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);result.update(elapsed_seconds=time.monotonic()-start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,native_and_guard_ticks=ticks[0],observations=serial[0]);save('result.json',result,True);save('manifest.json',{'files':{str(p.relative_to(out)):entry(p)for p in sorted(out.rglob('*'))if p.is_file()}},True)
        for p in sorted(out.rglob('*'),reverse=True):p.chmod(0o555 if p.is_dir()else 0o444)
        out.chmod(0o555)
    print(json.dumps(result),flush=True);return 0 if result['passed']else 2
if __name__=='__main__':raise SystemExit(main(Path(sys.argv[1])))
