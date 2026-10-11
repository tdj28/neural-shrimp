#!/usr/bin/env python3
"""One fixed full705 correlated tangent refinement;72 profile+72 target calls."""
import gc,json,os,resource,signal,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'src')]
import localize_native705_saved as L
import probe_native705_correlated as B
from flint import arb,arb_mat,ctx
from neural_shrimp.rigorous_unit_pair_pal import path_coefficients
from neural_shrimp.rigorous_quadratic_tangent_forcing import path_residual_derivative,balanced_directions,mixed_hessian_from_second_coefficients,refine_tangent
from neural_shrimp.rigorous_orbits import eye,exact,vector_json,matrix_json
D=L.D;U=L.U;O=L.O;need=L.need;read=L.read
OUT=Path('/data2/neural-shrimp-data/explore/eigenpair-locus-prototype/generalized-chain/unit-pair/pal/complement/attempt003-tangent/run')
CARD='docs/plans/native705-tangent-inputs.json';SCALE=arb('1/256')
PATH_FILES=['src/neural_shrimp/'+n+'.py'for n in ['rigorous_eigenpair_locus','rigorous_eigenpair_path','rigorous_unit_pair','rigorous_unit_pair_pal']]
FILES=sorted(set(B.FILES+PATH_FILES+[CARD,'docs/plans/native705-tangent-target.md','docs/plans/native705-correlated-tangent.md','tools/refine_native705_tangent.py','tools/launch_native705_tangent.py','src/neural_shrimp/rigorous_quadratic_tangent_forcing.py','tests/test_rigorous_quadratic_tangent_forcing.py','tests/test_native705_tangent_driver.py']))


def segment_inputs(args,proof,original):
    n=705;need(proof['passed']and proof['dimension']==n and U.R.original_ball(proof['weighted_neumann_bound'])<1,'strict all705 known-root localization')
    error=[U.R.original_ball(v[0])for v in proof['whole_refinement']['box']]
    c,s,a=args['center'],args['slope'],args['curvature'];need(len(error)==len(c)==len(s)==len(a)==n,'all705 root correction/predictor coordinates')
    d=arb(0,args['halfwidth']);predictor=[c[i]+s[i]*d+a[i]*d*d/2 for i in range(n)];pprime=[s[i]+a[i]*d for i in range(n)]
    segment=[predictor[i]+arb(0,abs(error[i]).upper())for i in range(n)]
    need(len(original)==n and all(v.is_finite()for v in error+predictor+pprime+segment),'full finite predictor-root segment')
    for old,new in zip(original,segment,strict=True):U.contains(old,D.ball_json(new),'exact original domain covers every predictor-root segment')
    need(all(v>0 for v in segment[-3:])and abs(segment[-1]).upper()<=arb('1/4'),'entire segment positive actual rates and bounded positive angle')
    plus,minus=balanced_directions(error,pprime,SCALE)
    return segment,error,pprime,plus,minus


def bind(card,tick):
    args,parent=L.bind(card['curve'],tick)
    rec=card['localization'];r=Path(rec['root']);O.verify(r,rec['outer']);O.verify(r/'run',rec['worker']);D.B.archived(r/'run',rec['source'],'runtime.json','source');out=read(r/'run/outcome.json');launch=read(r/'launcher-result.json')
    need(out['passed']and out['status']=='conditional_full705_localization_completed'and out['native_calls']==0 and launch['exit_code']==0 and not launch['owned_live_pids'],'completed literal705 localization')
    need(read(r/'run/bindings.json')['card']==card['curve']and O.entry(r/'run/localization.json')['sha256']==rec['proof'],'same authenticated705 root correction')
    failed=card['failed_complement'];fr=Path(failed['root']);O.verify(fr,failed['outer']);O.verify(fr/'run',failed['worker']);D.B.archived(fr/'run',failed['source'],'runtime.json','source');fo=read(fr/'run/outcome.json');fl=read(fr/'launcher-result.json')
    need(not fo['passed']and fo['status']=='completed_correlated_bound_unresolved'and fo['native_calls']==72 and fl['exit_code']==2 and not fl['owned_live_pids'],'retained full705 insufficient complement trial')
    fb=read(fr/'run/bindings.json');need(fb['card']['curve']==card['curve']and fb['localization']==rec,'same original705 curve/localization in failed bound')
    origin=Path(card['curve']['producer']['root']);chart=read(origin/'run/certificate.json.gz');segment,error,pprime,plus,minus=segment_inputs(args,read(r/'run/localization.json'),chart['whole_box'])
    cc=U.R.matrix(chart['quadratic_center_coefficients']);wc=U.R.matrix(chart['quadratic_whole_coefficients']);rp=path_residual_derivative(cc,wc,args['halfwidth'])
    normal=[U.R.exact(U.F(v))for v in chart['normal']];gaugeorigin=[U.R.exact(U.F(v))for v in chart['origin']];sc=U.R.exact(U.F(chart['parameter_center']))
    need(len(normal)==len(gaugeorigin)==705 and sc==arb('31/4096'),'same full705 fixed affine gauge')
    rt=read(origin/'run/runtime.json')
    for name in B.NATIVE+PATH_FILES:need(O.entry(ROOT/name)==rt['sources'][name]==O.entry(origin/'run/source'/name),'literal same original native/path source '+name)
    inp=read(origin/'run/inputs.json')['map_inputs'];need(inp['network']['d']==2 and inp['network']['width']==3 and inp['network']['bias']is True and 1<=len(inp['x'])<=24,'same bounded original13 model')
    model=B.TwoRateCuspMap(inp);need(model.n==13,'same13 native dimension');tick()
    return model,args,segment,error,pprime,plus,minus,rp,normal,gaugeorigin,sc,{'card':card,'curve':parent,'map_inputs':inp,'native_source_matches':B.NATIVE+PATH_FILES,'same_localization':O.entry(r/'run/localization.json'),'same_original_certificate':O.entry(origin/'run/certificate.json.gz')}


def synthetic_inputs():
    inp,x,rates,_,_=B.gd_fixture();flat=[]
    for _ in range(18):flat+=x+[arb(int(i==0))for i in range(13)]+[arb(int(i==1))for i in range(13)]
    flat+=rates+[arb('1/1024')];error=[arb(0,arb((1,-24)))]*705;pprime=[arb('1/8')]*705;plus,minus=balanced_directions(error,pprime,SCALE)
    normal=[arb(0)]*704+[arb(1)];return B.TwoRateCuspMap(inp),flat,plus,minus,pprime,normal,[arb(0)]*705,arb(0),inp


def main():
    resource.setrlimit(resource.RLIMIT_AS,(6*1024**3,)*2)
    need(not subprocess.check_output(['git','-c','core.preloadIndex=false','-c','index.threads=1','status','--porcelain'],cwd=ROOT).strip(),'clean committed source')
    need(os.getpriority(os.PRIO_PROCESS,0)>=10 and all(os.environ.get(k)=='1'for k in O.THREADS),'CPU1nice10')
    OUT.mkdir(parents=True,exist_ok=False);start=time.monotonic();used=[0];stop=[False];calls=[0];result={'passed':False,'status':'incomplete','complementary_stability_evaluated':False}
    def save(name,value,closing=False):B.save_bound(OUT,name,value,used,closing=closing)
    def tick():need(not stop[0]and time.monotonic()-start<240 and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<4*1024**2,'240s/4GiB cap')
    def acquisition(model,flat,v,normal,origin,sc,stage):
        begin=calls[0];need(begin in(0,36,72,108),'fixed four36-phase acquisition groups')
        class Recorded:
            n=13
            def cusp_step(self,x,a,b):
                tick();need(calls[0]<144 and calls[0]-begin<36,'fixed144 total/36 group call cap');calls[0]+=1
                save(f'native/{calls[0]:03d}-input.json',{'stage':stage,'phase':(calls[0]-begin-1)//2,'direction':'q'if(calls[0]-begin)%2 else'r','inputs':[v.serialized()for v in x+[a,b]]})
                values=model.cusp_step(x,a,b);save(f'native/{calls[0]:03d}-return.json',{'stage':stage,'returns':[v.serialized()for v in values]});tick();return values
        def observe(kind,row):
            if kind=='pal_path_assembly_return':save('coefficients/'+stage+'.json',row)
        answer=path_coefficients(Recorded(),flat,v,[arb(0)]*705,arb(0),18,normal,origin,sc,tick,observe)
        need(calls[0]-begin==36,'complete36 retained native mixed acquisition');return answer
    signal.signal(signal.SIGALRM,lambda *_:stop.__setitem__(0,True));signal.signal(signal.SIGTERM,lambda *_:stop.__setitem__(0,True));signal.setitimer(signal.ITIMER_REAL,240)
    source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();save('runtime.json',{'source_commit':source,'pid':os.getpid(),'nice':os.getpriority(os.PRIO_PROCESS,0),'threads':dict.fromkeys(O.THREADS,'1'),'sources':{n:O.entry(ROOT/n)for n in FILES},'precision':384,'python':D.platform.python_version(),'numpy':D.np.__version__,'scipy':D.scipy.__version__,'python_flint':D.flint.__version__,'flint':D.flint.__FLINT_VERSION__,'native_call_cap':144,'polarization_scale':'1/256'})
    for n in FILES:p=OUT/'source'/n;p.parent.mkdir(parents=True,exist_ok=True);b=(ROOT/n).read_bytes();need(used[0]+len(b)<62*1024**2,'source cap');p.write_bytes(b);used[0]+=len(b)
    try:
        with ctx.workprec(384):
            before=time.monotonic();model,flat,plus,minus,pprime,normal,origin,sc,inp=synthetic_inputs();save('profile-inputs.json',{'map_inputs':inp,'segment':vector_json(flat),'plus':vector_json(plus),'minus':vector_json(minus),'normal':vector_json(normal),'origin':vector_json(origin),'parameter':D.ball_json(sc)})
            cp=acquisition(model,flat,plus,normal,origin,sc,'profile_plus');cm=acquisition(model,flat,minus,normal,origin,sc,'profile_minus');mixed=mixed_hessian_from_second_coefficients(cp,cm)
            small=arb((1,-16));n=705;C=arb_mat([[arb(int(i==j))+small for j in range(n)]for i in range(n)]);J=arb_mat([[arb(int(i==j))-small for j in range(n)]for i in range(n)])
            trial=refine_tangent(C,J,arb_mat(n,1),mixed,pprime,[arb(1)]*n,tick=tick);seconds=time.monotonic()-before;forecast=4*seconds+4*15.419+20
            save('profile.json',{'passed':trial['passed'],'seconds':seconds,'forecast_seconds':forecast,'native_calls':calls[0],'dimension':705,'target_arrays_unopened':True});need(trial['passed']and calls[0]==72 and forecast<240,'measured full705144-call admission');del C,J,cp,cm,mixed,trial;gc.collect()
            card=read(ROOT/CARD);need(card['settings']=={'precision':384,'scale':'1/256','native_calls':144,'synthetic_calls':72,'target_calls':72,'seconds':240,'rss_bytes':4*1024**3,'output_bytes':64*1024**2},'literal fixed tangent settings')
            model,args,segment,error,pprime,plus,minus,rp,normal,origin,sc,binding=bind(card,tick);save('bindings.json',binding)
            save('actual-domain-inputs.json',{'segment':vector_json(segment),'error':vector_json(error),'predictor_tangent':vector_json(pprime),'plus':vector_json(plus),'minus':vector_json(minus),'normal':vector_json(normal),'origin':vector_json(origin),'parameter':D.ball_json(sc),'scale':'1/256','exact_parameter_endpoints':['-1/4096','63/4096'],'residual_derivative':matrix_json(rp)})
            cp=acquisition(model,segment,plus,normal,origin,sc,'target_plus');cm=acquisition(model,segment,minus,normal,origin,sc,'target_minus');mixed=mixed_hessian_from_second_coefficients(cp,cm);need(calls[0]==144,'all72 actual augmented mixed acquisitions')
            proof=refine_tangent(args['C'],args['J'],rp,mixed,pprime,args['weights'],tick=tick);save('tangent.json',proof);tick()
            result.update(passed=True,status='conditional_correlated705_tangent_completed',parameter_endpoints=['-1/4096','63/4096'],weighted_neumann_bound=proof['weighted_neumann_bound'],tangent_correction_radius=proof['tangent_error']['weighted_uniform_radius'],scope=proof['scope'])
    except BaseException as e:result.update(status='failed_or_interrupted',error=type(e).__name__+': '+str(e),traceback=traceback.format_exc())
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);result.update(native_calls=calls[0],elapsed_seconds=time.monotonic()-start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss);save('outcome.json',result,closing=True);save('manifest.json',{'files':{str(p.relative_to(OUT)):O.entry(p)for p in sorted(OUT.rglob('*'))if p.is_file()}},closing=True)
    print(json.dumps(result));return 0 if result['passed']else 2
if __name__=='__main__':raise SystemExit(main())
