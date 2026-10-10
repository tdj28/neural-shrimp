#!/usr/bin/env python3
"""One source-bound saved704 complement repair; no fresh native acquisitions."""
import gc,json,os,resource,signal,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools/reviews')]
from flint import arb,arb_mat,ctx
import numpy as np
import review_unit_pair_driver as B
import review_unit_pair_math as U
from neural_shrimp.rigorous_floquet_complement import uniform_complement
from neural_shrimp.rigorous_orbits import canonical_bytes,ball_json
from neural_shrimp.rigorous_periodic_lyapunov import exact_float_matrix
O=B.O;need=U.need;read=B.read
OUT=Path('/data2/neural-shrimp-data/explore/eigenpair-locus-prototype/generalized-chain/unit-pair/attempt002-spectrum/run')
CARD='docs/plans/floquet-complement-inputs.json'
FILES=[CARD,'docs/plans/floquet-complement.md','tools/probe_floquet_complement.py','tools/launch_floquet_complement.py','src/neural_shrimp/rigorous_floquet_complement.py','tests/test_floquet_complement_driver.py','tests/test_rigorous_floquet_complement.py']
FILES=sorted(set(FILES+[str(Path(m.__file__).resolve().relative_to(ROOT))for m in list(sys.modules.values())if getattr(m,'__file__',None)and Path(m.__file__).resolve().is_relative_to(ROOT)]))

def admitted_native_blocks(proof,rows,old):
    n=13;p=18;need((proof['dimension'],proof['period'],proof['shooting_dimension'])==(n,p,704),'fixed full704 dimensions')
    need(proof['parameter_endpoints']==old['exact_parameter_endpoints']==['0','1/32768'],'same complete finite interval')
    flat=proof['whole_box'];need(len(flat)==704 and old['whole_box']==flat and len(rows)==36,'entire original native box')
    J=proof['whole_jacobian'];need(len(J)==704 and all(len(r)==704 for r in J),'full native704 Jacobian')
    blocks=[]
    for j in range(p):
        q,r=rows[2*j:2*j+2];off=39*j
        need(q['phase']==r['phase']==j and q['direction']=='q'and r['direction']=='r','complete q/r native phase order')
        need(q['input']==flat[off:off+26]+flat[-2:]and r['input']==flat[off:off+13]+flat[off+26:off+39]+flat[-2:],'exact original whole-domain acquisition')
        for e in(q,r):need(len(e['block'])==26 and all(len(row)==26 for row in e['block']),'full native block')
        A=[row[:13]for row in q['block'][:13]]
        need(A==[row[:13]for row in r['block'][:13]]==[row[off:off+13]for row in J[off:off+13]],'actual whole704 state Jacobian agrees both acquisitions')
        blocks.append(A)
    need(old['native_blocks']==blocks,'literal old failed quotient input domain');return flat,blocks

def profile(tick):
    # Full unrelated13D/18-phase similarity with mixed stable complex/real modes.
    p=18;n=13;A=np.eye(n);A[2,2]=-.3
    for i in range(3,13,2):
        rho=.2+.015*i;a=.1*i;c,s=rho*np.cos(a),rho*np.sin(a);A[i:i+2,i:i+2]=[[c,s],[-s,c]]
    rng=np.random.default_rng(701304);T=np.eye(n)+.01*rng.standard_normal((n,n));inv=np.linalg.inv(T)
    blocks=[exact_float_matrix(np.eye(n))for _ in range(p-1)]+[exact_float_matrix(T@A@inv)]
    # Exact critical columns/derivative coherence uses coordinate critical axes.
    # Similarity restricted to complementary block avoids floating zero errors.
    T[:2,:]=np.eye(n)[:2,:];T[:,:2]=np.eye(n)[:,:2];inv=np.linalg.inv(T);blocks[-1]=exact_float_matrix(T@A@inv)
    flat=[]
    for _ in range(p):flat.extend([arb(0)]*n+[arb(int(i==0))for i in range(n)]+[arb(int(i==1))for i in range(n)])
    flat.extend([arb(1),arb(1)]);started=time.monotonic();result=uniform_complement(blocks,flat,p,n,arb_mat([[1,0],[0,1]]),tick=tick)
    need(result['passed'],'full-size unrelated mathematical profile');return {'passed':True,'seconds':time.monotonic()-started,'dimension':n,'period':p,'complement_dimension':11,'native_calls':0}

def bind(card,tick):
    parent=card['producer'];pr=Path(parent['root']);O.verify(pr,parent['outer']);O.verify(pr/'run',parent['worker']);rt=B.archived(pr/'run',parent['source'],'runtime.json','source')
    done=read(pr/'run/outcome.json');B.check_native_contract(done,rt,read(pr/'launcher-result.json'),card)
    independent=card['independent'];ir=Path(independent['root']);O.verify(ir,independent['outer']);O.verify(ir/'run',independent['worker']);B.archived(ir/'run',independent['source'],'runtime.json','source')
    io=read(ir/'run/outcome.json');il=read(ir/'launcher-result.json');ib=read(ir/'run/input-bindings.json')
    need(io['passed']and io['strict_inclusions']==io['anchor_coordinates']==704 and io['primitive_period']==18 and io['same_independently_connected_root']and io['parameter_endpoints']==['0','1/32768'],'independently proved finite704 root tube and anchor')
    need(il['source_commit']==independent['source']and il['exit_code']==0 and not il['owned_live_pids'],'completed independent audit')
    need(ib['card']['producer']==parent and ib['certificate']==O.entry(pr/'run/certificate.json.gz'),'same exact independently audited original proof')
    old=card['failed'];fr=Path(old['root']);O.verify(fr,old['outer']);O.verify(fr/'run',old['worker']);B.archived(fr/'run',old['source'],'runtime.json','source')
    fo=read(fr/'run/outcome.json');fl=read(fr/'launcher-result.json')
    need(fo['passed']is False and fo['status']=='completed_complement_unresolved'and fo['native_calls']==0 and fl['exit_code']==2 and not fl['owned_live_pids'],'preserved completed failed sufficient bound')
    inputs=read(fr/'run/inputs.json');need(U.M.interval(inputs['parameter_interval'])==(U.F(0),U.F(1,32768)),'exact old parameter interval')
    cert=pr/'run/certificate.json.gz';need(O.entry(cert)['sha256']==parent['certificate'],'literal complete original proof')
    files=sorted((pr/'run/events').glob('*-direction_return.json.gz'));need(len(files)==216,'all216 shared acquired native directions')
    rows=[read(p)for p in files[-36:]];proof=read(cert);flat,blocks=admitted_native_blocks(proof,rows,inputs)
    del proof;gc.collect();tick()
    return flat,blocks,{'card':card,'old_inputs':O.entry(fr/'run/inputs.json'),'certificate':O.entry(cert),'whole_direction_files':{str(p.relative_to(pr/'run')):O.entry(p)for p in files[-36:]},'independent_outcome':O.entry(ir/'run/outcome.json'),'shared_native_acquisitions':36,'new_native_calls':0}

def main():
    resource.setrlimit(resource.RLIMIT_AS,(6*1024**3,)*2)
    need(not subprocess.check_output(['git','-c','core.preloadIndex=false','-c','index.threads=1','status','--porcelain'],cwd=ROOT).strip(),'clean committed source')
    need(os.getpriority(os.PRIO_PROCESS,0)>=10 and all(os.environ.get(k)=='1'for k in O.THREADS),'CPU1nice10')
    card=read(ROOT/CARD);OUT.mkdir(parents=True,exist_ok=False);start=time.monotonic();stop=[False];result={'passed':False,'status':'incomplete','native_calls':0};used=[0]
    def save(name,value):
        data=canonical_bytes(value);need(used[0]+len(data)<32*1024**2,'32MiB output cap');p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True)
        with p.open('xb')as f:f.write(data);f.flush();os.fsync(f.fileno())
        used[0]+=len(data)
    def tick():need(not stop[0]and time.monotonic()-start<120 and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<4*1024**2,'120s/4GiB cap')
    signal.signal(signal.SIGTERM,lambda *_:stop.__setitem__(0,True));signal.signal(signal.SIGALRM,lambda *_:stop.__setitem__(0,True));signal.setitimer(signal.ITIMER_REAL,120)
    source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();save('runtime.json',{'source_commit':source,'pid':os.getpid(),'nice':os.getpriority(os.PRIO_PROCESS,0),'threads':dict.fromkeys(O.THREADS,'1'),'sources':{n:O.entry(ROOT/n)for n in FILES},'precision':384,'native_calls':0})
    for n in FILES:p=OUT/'source'/n;p.parent.mkdir(parents=True,exist_ok=True);data=(ROOT/n).read_bytes();need(used[0]+len(data)<32*1024**2,'archive cap');p.write_bytes(data);used[0]+=len(data)
    try:
        with ctx.workprec(384):
            timing=profile(tick);timing['forecast_seconds']=4*timing['seconds']+4*15.713+10;save('profile.json',timing);need(timing['forecast_seconds']<120,'measured full-size/source before any target arrays')
            flat,blocks,binding=bind(card,tick);save('bindings.json',binding)
            box=[U.R.original_ball(v)for v in flat];A=[U.R.matrix(v)for v in blocks];t=arb('1/65536',arb('1/65536'));S=U.terminal(2,t);terminal=arb_mat([[S[2,2],S[4,2]],[S[2,4],S[4,4]]])
            save('inputs.json',{'whole_box':flat,'native_blocks':blocks,'parameter_interval':ball_json(t),'terminal_critical_block':[[ball_json(terminal[i,j])for j in range(2)]for i in range(2)]})
            seq=[0]
            def observe(kind,row):seq[0]+=1;save(f'events/{seq[0]:03d}-{kind}.json',row);tick()
            proof=uniform_complement(A,box,18,13,terminal,tick=tick,observe=observe);save('complement.json',proof);tick()
            result.update(passed=proof['passed'],status='finite_complement_attracting'if proof['passed']else'completed_sufficient_bound_unresolved',complement_dimension=11,parameter_endpoints=['0','1/32768'],remainder=proof['product_error_norm_upper'],metric_passed=proof['metric']['passed'],scope=proof['scope'])
    except BaseException as exc:result.update(status='failed_or_interrupted',error=type(exc).__name__+': '+str(exc),traceback=traceback.format_exc())
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);result.update(elapsed_seconds=time.monotonic()-start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss);save('outcome.json',result);save('manifest.json',{'files':{str(p.relative_to(OUT)):O.entry(p)for p in sorted(OUT.rglob('*'))if p.is_file()}})
    print(json.dumps(result));return 0 if result['passed']else 2
if __name__=='__main__':raise SystemExit(main())
