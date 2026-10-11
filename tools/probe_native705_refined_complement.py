#!/usr/bin/env python3
"""Fixed improved705 complement;36 profile+18 target calls,18 retained centers."""
import gc,json,os,resource,signal,subprocess,sys,time,traceback
from pathlib import Path
from fractions import Fraction
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'src')]
import localize_native_unit_pair_saved as L
import localize_native705_saved as P
import probe_native705_correlated as Q
import refine_native705_tangent as T
from flint import arb,arb_mat,ctx
from neural_shrimp.rigorous_cusps import TwoRateCuspMap
from neural_shrimp.rigorous_correlated_complement import total_two_rate_jacobian,correlated_complement
from neural_shrimp.rigorous_orbits import eye,exact,vector_json,matrix_json
D=L.D;U=L.U;O=L.O;need=L.need;read=L.read
OUT=Path('/data2/neural-shrimp-data/explore/eigenpair-locus-prototype/generalized-chain/unit-pair/pal/complement/attempt004-correlated/run')
CARD='docs/plans/native705-refined-complement-inputs.json'
NATIVE=['src/neural_shrimp/rigorous_cusps.py','src/neural_shrimp/rigorous_folds.py','src/neural_shrimp/rigorous_orbits.py','src/neural_shrimp/controls.py','src/neural_shrimp/models.py']
FILES=sorted(set(T.FILES+['tools/probe_native705_refined_complement.py','tools/launch_native705_refined_complement.py','tests/test_native705_refined_complement.py','docs/plans/native705-refined-complement.md',CARD]))

def save_bound(root,name,value,used,*,closing=False):
    raw=D.canonical_bytes(value);limit=64*1024**2-(0 if closing else 2*1024**2)
    need(used[0]+len(raw)<limit,'64MiB raw cap with final reserve');p=root/name;p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('xb')as f:f.write(raw);f.flush();os.fsync(f.fileno())
    used[0]+=len(raw)


def gd_fixture():
    f=lambda v:str(Fraction(v))
    inp={'schema':'neural-shrimp-c1-input-v1','network':{'activation':'tanh','d':2,'width':3,'bias':True,'in_scale':'3/2','out_scale':'2/3','loss_scale':'1/2','ridge':'1/1000'},'x':[[f(Fraction(i-12,48)),f(Fraction((i*7)%23-11,48))]for i in range(24)],'y':[f(Fraction((i*3)%17-8,64))for i in range(24)],'lr_hidden':'1/100','lr_output':'1/100'}
    x=[arb(exact(f(Fraction(i-6,64))),arb((1,-20)))for i in range(13)];v=[arb(exact(f(Fraction(i+1,32))),arb((1,-20)))for i in range(13)]
    rates=[arb(exact('1/100'),arb((1,-20)))]*2;vp=[arb('1/8'),arb('-1/16')]
    return inp,x,rates,v,vp

def synthetic_cocycle(tick):
    n=13;p=18;F=[]
    for _ in range(p):F+=[arb(0)]*n+[arb(int(i==0))for i in range(n)]+[arb(int(i==1))for i in range(n)]
    F+=[arb(1),arb(1)];A=eye(n)
    for i in range(2,n):A[i,i]=arb('9/10')
    rows=[{'phase':j,'center_A':A,'whole_A':A,'whole_Aprime':arb_mat(n,n)}for j in range(p)]
    E=arb_mat([[arb(int(i==k+2))for k in range(11)]for i in range(13)])
    return correlated_complement(rows,F,F,[arb(0)]*704,[E]*18,[eye(11)]*18,arb((1,-16)),eye(2),tick=tick)

def native_domains(proof):
    need(proof['passed']and proof['dimension']==705 and U.R.original_ball(proof['weighted_neumann_bound'])<1,'all705 localized root and true tangent')
    center=[U.R.original_ball(v)for v in proof['center_box']];whole=[U.R.original_ball(v)for v in proof['whole_box']];tangent=[U.R.original_ball(v[0])for v in proof['tangent_box']]
    need(len(center)==len(whole)==len(tangent)==705 and all(v.is_finite()for v in center+whole+tangent),'all705 finite state/mode/rate/angle domains')
    need(all(v>0 for v in center[-3:]+whole[-3:])and abs(whole[-1]).upper()<=arb('1/4'),'both actual rates and entire positive bounded angle')
    T=U.terminal(2,whole[-1]);terminal=arb_mat([[T[2,2],T[4,2]],[T[2,4],T[4,4]]])
    return center[:-1],whole[:-1],tangent[:-1],terminal,{'center':vector_json(center),'whole':vector_json(whole),'tangent':vector_json(tangent)}


def retained_center(a,v,j,center):
    need(a['kind']==v['kind']=='center'and a['phase']==v['phase']==j,'same literal saved center derivative phase')
    off=39*j;need(a['state']==vector_json(center[off:off+13])and a['rates']==vector_json(center[-2:]),'identical actual center state and both rates')
    need(len(a['state_tangent'])==13 and len(a['rate_tangent'])==2 and all(U.R.original_ball(x).is_zero()for x in a['state_tangent']+a['rate_tangent']),'exact zero historical center tangent')
    A=U.R.matrix(v['A']);Ap=U.R.matrix(v['Aprime']);need(A.nrows()==A.ncols()==Ap.nrows()==Ap.ncols()==13 and all(x.is_finite()for row in A.tolist()for x in row)and all(x.is_zero()for row in Ap.tolist()for x in row),'full native center matrix and zero total derivative');return A


def bind_localization(card,tick):
    model,center,whole,old_tangent,E,B,terminal,full,parent=Q.bind_localization(card['original'],tick)
    rec=card['tangent'];root=Path(rec['root']);O.verify(root,rec['outer']);O.verify(root/'run',rec['worker']);D.B.archived(root/'run',rec['source'],'runtime.json','source')
    out=read(root/'run/outcome.json');launch=read(root/'launcher-result.json');binding=read(root/'run/bindings.json')
    need(out['passed']and out['status']=='conditional_correlated705_tangent_completed'and out['native_calls']==144 and not out['complementary_stability_evaluated']and launch['exit_code']==0 and not launch['owned_live_pids'],'complete original705 correlated tangent')
    need(binding['card']['curve']==card['original']['curve']and binding['card']['localization']==card['original']['localization']and binding['card']['failed_complement']==card['failed'],'literal same705 family and root localization for tangent')
    need(O.entry(root/'run/tangent.json')['sha256']==rec['proof'],'full pinned705 tangent proof');proof=read(root/'run/tangent.json')
    need(proof['passed']and proof['dimension']==705 and U.R.original_ball(proof['weighted_neumann_bound'])<1 and out['parameter_endpoints']==['-1/4096','63/4096'],'all705 true tangents on complete original interval')
    newfull=[U.R.original_ball(v[0])for v in proof['tangent_box']];need(len(newfull)==705 and all(v.is_finite()for v in newfull),'full finite angle/mode/rate/state tangent')
    tangent=newfull[:-1];full['tangent']=vector_json(newfull)
    failed=card['failed'];fr=Path(failed['root']);O.verify(fr,failed['outer']);O.verify(fr/'run',failed['worker']);D.B.archived(fr/'run',failed['source'],'runtime.json','source')
    fo=read(fr/'run/outcome.json');fl=read(fr/'launcher-result.json');fb=read(fr/'run/bindings.json');frt=read(fr/'run/runtime.json')
    need(not fo['passed']and fo['status']=='completed_correlated_bound_unresolved'and fo['native_calls']==72 and fl['exit_code']==2 and not fl['owned_live_pids']and fb['card']==card['original'],'immutable original full705 failed bound')
    for name in NATIVE:need(O.entry(ROOT/name)==frt['sources'][name]==O.entry(fr/'run/source'/name),'unchanged exact center acquisition source '+name)
    need(fb['map_inputs']==parent['map_inputs'],'same native map/data for retained center derivatives')
    centers=[];selected={}
    for j in range(18):
        k=37+2*j;ip=fr/f'run/native/{k:03d}-input.json';vp=fr/f'run/native/{k:03d}-return.json';centers.append(retained_center(read(ip),read(vp),j,center));selected[str(ip.relative_to(fr/'run'))]=O.entry(ip);selected[str(vp.relative_to(fr/'run'))]=O.entry(vp)
    tick();return model,center,whole,tangent,E,B,terminal,full,centers,{'card':card,'unchanged_root':parent,'tangent':rec,'retained_center_derivatives':selected,'new_target_calls':18,'shared_target_center_calls':18,'physical_scope':'Same full705 known root. Only true tangent replaced by correlated proof; center/whole domains and exact E/B unchanged. All705 tangent retained before dropping angle.'}

def main():
    resource.setrlimit(resource.RLIMIT_AS,(6*1024**3,)*2)
    need(not subprocess.check_output(['git','-c','core.preloadIndex=false','-c','index.threads=1','status','--porcelain'],cwd=ROOT).strip(),'clean committed source')
    need(os.getpriority(os.PRIO_PROCESS,0)>=10 and all(os.environ.get(k)=='1'for k in O.THREADS),'CPU1nice10')
    OUT.mkdir(parents=True,exist_ok=False);start=time.monotonic();used=[0];stop=[False];calls=[0];result={'passed':False,'status':'incomplete'}
    def save(name,value,closing=False):save_bound(OUT,name,value,used,closing=closing)
    def tick():need(not stop[0]and time.monotonic()-start<240 and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<4*1024**2,'240s/4GiB cap')
    def native(model,x,rates,v,vp,kind,phase):
        tick();need(calls[0]<54,'fixed54 native-call total cap');calls[0]+=1
        save(f'native/{calls[0]:03d}-input.json',{'kind':kind,'phase':phase,'state':vector_json(x),'rates':vector_json(rates),'state_tangent':vector_json(v),'rate_tangent':vector_json(vp)})
        a,ap=total_two_rate_jacobian(model,x,rates,v,vp);save(f'native/{calls[0]:03d}-return.json',{'kind':kind,'phase':phase,'A':matrix_json(a),'Aprime':matrix_json(ap)});tick();return a,ap
    signal.signal(signal.SIGALRM,lambda *_:stop.__setitem__(0,True));signal.signal(signal.SIGTERM,lambda *_:stop.__setitem__(0,True));signal.setitimer(signal.ITIMER_REAL,240)
    source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();save('runtime.json',{'source_commit':source,'pid':os.getpid(),'nice':os.getpriority(os.PRIO_PROCESS,0),'threads':dict.fromkeys(O.THREADS,'1'),'sources':{n:O.entry(ROOT/n)for n in FILES},'precision':384,'python':D.platform.python_version(),'numpy':D.np.__version__,'scipy':D.scipy.__version__,'python_flint':D.flint.__version__,'flint':D.flint.__FLINT_VERSION__,'native_call_cap':54})
    for n in FILES:p=OUT/'source'/n;p.parent.mkdir(parents=True,exist_ok=True);b=(ROOT/n).read_bytes();need(used[0]+len(b)<62*1024**2,'source cap');p.write_bytes(b);used[0]+=len(b)
    try:
        with ctx.workprec(384):
            before=time.monotonic();inp,x,rates,v,vp=gd_fixture();profile_model=TwoRateCuspMap(inp);save('profile-inputs.json',{'map_inputs':inp})
            for j in range(36):native(profile_model,x,rates,v,vp,'synthetic',j)
            synthetic=synthetic_cocycle(tick);need(synthetic['passed'],'full18phase11mode correlated profile');seconds=time.monotonic()-before;forecast=4*seconds+4*21.120+20
            save('profile.json',{'passed':True,'seconds':seconds,'forecast_seconds':forecast,'native_calls':calls[0],'target_arrays_unopened':True,'full_correlated_kernel':synthetic});need(calls[0]==36 and forecast<240,'full measured profile admission')
            card=read(ROOT/CARD);model,center,whole,tangent,E,B,terminal,full,centers,binding=bind_localization(card,tick);save('bindings.json',binding)
            save('actual-domain-inputs.json',{'center':vector_json(center),'whole':vector_json(whole),'tangent':vector_json(tangent),'exact_endpoints':['-1/4096','63/4096'],'halfwidth':'1/128','full705_localized_domains':full,'terminal':matrix_json(terminal),'complements':[matrix_json(e)for e in E],'metrics':[matrix_json(b)for b in B]})
            rows=[]
            for j in range(18):
                off=39*j;A0=centers[j]
                Aw,Ap=native(model,whole[off:off+13],whole[-2:],tangent[off:off+13],tangent[-2:],'whole_tangent',j)
                rows.append({'phase':j,'center_A':A0,'whole_A':Aw,'whole_Aprime':Ap})
            need(calls[0]==54,'complete18 fresh whole-tangent acquisitions');half=arb((1,-7));seq=[0]
            def observe(kind,row):seq[0]+=1;save(f'events/{seq[0]:03d}-{kind}.json',row);tick()
            proof=correlated_complement(rows,center,whole,tangent,E,B,half,terminal,tick=tick,observe=observe);save('complement.json',proof)
            result.update(passed=proof['passed'],status='first705_refined_complement_attracting'if proof['passed']else'completed_correlated_bound_unresolved',complement_dimension=11,parameter_endpoints=['-1/4096','63/4096'],remainder=proof['remainder_norm_upper'],metric_passed=proof['metric']['passed'],scope=proof['scope'])
    except BaseException as e:result.update(status='failed_or_interrupted',error=type(e).__name__+': '+str(e),traceback=traceback.format_exc())
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);result.update(native_calls=calls[0],elapsed_seconds=time.monotonic()-start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss);save('outcome.json',result,closing=True);save('manifest.json',{'files':{str(p.relative_to(OUT)):O.entry(p)for p in sorted(OUT.rglob('*'))if p.is_file()}},closing=True)
    print(json.dumps(result));return 0 if result['passed']else 2
if __name__=='__main__':raise SystemExit(main())
