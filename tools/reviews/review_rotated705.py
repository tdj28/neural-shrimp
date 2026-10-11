"""One independent480 full705 rotated root and same-known-endpoint connection."""
import gc,gzip,hashlib,json,os,platform,resource,signal,subprocess,sys,time,traceback
from fractions import Fraction as F
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'src'),str(Path(__file__).parent)]
import flint
from flint import arb,arb_mat,ctx
import review_cubic705_prefix as B
import review_rotated705_math as A
H=B.H;OLD=B.OLD;R=B.R;M=B.M;U=B.U;need=B.need;read=B.read;entry=B.entry;inventory=B.inventory;historical=B.historical;THREADS=B.THREADS
from neural_shrimp.rigorous_orbits import ball_json,vector_json,canonical_bytes
OUT=Path('/data2/neural-shrimp-data/audits/native705-rotated-independent/attempt001/run');CARD='docs/plans/native705-rotated-independent-inputs.json'
FILES=[CARD,'docs/plans/native705-rotated-independent.md','tools/reviews/review_rotated705.py','tools/launch_rotated705_review.py','tests/test_rotated705_driver.py','tests/test_rotated705_review.py']
FILES=sorted(set(FILES+[str(Path(m.__file__).resolve().relative_to(ROOT))for m in list(sys.modules.values())if getattr(m,'__file__',None)and Path(m.__file__).resolve().is_relative_to(ROOT)]))

def matrix_file(path,cols):
    z=read(path);need(z['encoding']=='mid-rad-dyadic-pairs'and len(z['rows'])==705 and all(len(row)==cols for row in z['rows']),'full independently stored matrix dimensions')
    result=[]
    for row in z['rows']:
        line=[]
        for v in row:
            need(type(v)is list and len(v)==2 and all(type(x)is list and len(x)==2 and all(type(k)is int for k in x)for x in v)and v[1][0]>=0 and all(abs(x[1])<10000 for x in v),'finite exact dyadic-pair encoding')
            line.append(arb(tuple(v[0]),arb(tuple(v[1]))))
        result.append(line)
    out=arb_mat(result);need(H.finite(out),'finite entire independently saved matrix');return out

def bind(card,w):
    ir,irt,io,rows=OLD.bundle(card['old_independent'],w.tick);ep=read(ir/'run/connected-endpoint.json');pr=read(ir/'run/chart003/independent-proof.json');ib=read(ir/'run/input-binding.json')
    need(io['passed']and io['connected']and io['connected_count']==3 and io['connected_right']=='115/512'and io['native_calls']==0 and irt['precision']==480,'complete independently proved old prefix')
    need(all(x['passed']and x['strict_inclusions']==x['anchor_coordinates']==705 for x in io['charts'])and pr['passed']and all(pr['gates'].values()),'every old full root and ordered join')
    need(ep['chart']==pr['chart']and ep['endpoint']==pr['endpoints'][1]and ep['chart']['parameter_endpoints']==['83/512','115/512']and entry(ir/'run/connected-endpoint.json')['sha256']==card['old_independent']['endpoint'],'literal independently known endpoint')
    old=Path(card['old_producer']['root']);rows.append(inventory(old,card['old_producer']['outer'],w.tick));pc=read(old/'input-card.json');ort=historical(old/'chart003/run',card['old_producer']['source'])
    need(all(ib['card']['producer'][k]==v for k,v in card['old_producer'].items())and pc['normal']==ep['chart']['normal']and pc['origin']==ep['chart']['origin'],'same independently audited original native geometry')
    new=Path(card['new_producer']['root']);rows.append(inventory(new,card['new_producer']['outer'],w.tick));run=new/'attempt002/run';rows.append(inventory(run,card['new_producer']['worker'],w.tick));nrt=historical(run,card['new_producer']['source']);nc=read(new/'attempt002/input-card.json');no=read(new/'outcome.json');out=read(run/'outcome.json');ni=read(run/'inputs.json')
    need(no['source']==card['new_producer']['source']and no['status']=='mathematical_stop'and no['connected_count']==0 and len(no['attempts'])==2 and not no['owned_live_pids'],'preserved adaptive failure and unjoined successful root')
    for attempt in ('attempt001','attempt002'):
        la=read(new/attempt/'launcher-result.json');owned=read(new/attempt/'owned-process.json');need(owned['source']==card['new_producer']['source']and not la['owned_live_pids']and not Path('/proc',str(owned['group'])).exists(),'all old owned groups terminated')
    need(nc['source']==card['new_producer']['source']and nc['settings']==nrt['settings']==ni['settings']and ni['card']==entry(new/'attempt002/input-card.json')and nc['map_inputs']==ni['map_inputs']==pc['map_inputs'],'literal new actual input source and model')
    need(out['passed']and not out['connected']and out['status']=='uniform705_unjoined'and out['ticks']==331 and out['strict_inclusions']==705 and out['events']==B.EXPECTED,'complete retained uniform root acquisitions')
    events=sorted((run/'events').glob('*.json.gz'));need(len(events)==sum(B.EXPECTED.values())and[int(p.name.split('-')[0])for p in events]==list(range(len(events))),'entire new native event stream')
    for kind,count in B.EXPECTED.items():need(len(list((run/'events').glob('*-'+kind+'.json.gz')))==count,'exact actual native event count')
    point,prt,po,rr=OLD.bundle(card['point_producer'],w.tick);rows+=rr
    need(po['passed']and po['status']=='same_known_endpoint_joined'and po['native_calls']==72 and po['events']=={'direction_return':72}and po['localized_join_inclusions']==705 and prt['precision']==480,'full fixed producer point residual run')
    pin=read(point/'run/point-input.json');need(pin['model']==pc['map_inputs']and pin['normal']==pc['normal']and pin['origin']==pc['origin']and pin['parameter']=='115/512','same old exact point family and gauge')
    for rt in (ort,nrt,prt):
        need(rt['threads']==dict.fromkeys(THREADS,'1'),'single-thread historical runtime')
        for n in OLD.NATIVE:need(rt['sources'][n]==ib['source_native'][n],'identical source-authenticated native model '+n)
    ev=sorted((point/'run/events').glob('*direction_return.json.gz'));need(len(ev)==72 and[int(p.name.split('-')[0])for p in ev]==list(range(72)),'36profile36target actual point phases')
    w.save('input-binding.json',dict(card=card,inventories=rows,source_native=ib['source_native'],new_uniform_certificate_unparsed=True,producer_localization_unparsed=True,point_input=entry(point/'run/point-input.json')))
    return ir,old,ep,pc,run,nc,point,pin,ev

def unique(run,kind):
    rows=list((run/'events').glob('*-'+kind+'.json.gz'));need(len(rows)==1,'unique actual '+kind);return rows[0]

def endpoint(ir,old,ep,pc,point,pin,events,w):
    domain=read(unique(old/'chart003/run','cubic_correlated_domain'));proposal=read(unique(old/'chart003/run','cubic_predictor_return'));cfg=read(old/'chart003/run/runtime.json')['settings'];c,s,a,b,normal,origin,weights,h,rho,d,radii,error,whole=B.geometry(domain,proposal,cfg,pc)
    need(cfg['parameter']=='99/512'and h==arb(1)/32,'literal old endpoint offset');predictor=[c[i]+s[i]*h+a[i]*h*h/2+b[i]*h**3 for i in range(705)];known=[R.original_ball(v)for v in ep['endpoint']]
    defects=[matrix_file(ir/f'run/chart003/defect_coefficient_{k}.json.gz',705)for k in range(4)];forcing=[matrix_file(ir/f'run/chart003/forcing_coefficient_{k}.json.gz',1)for k in range(6)]
    D=sum((v*h**k for k,v in enumerate(defects)),arb_mat(705,705));v=sum((x*h**k for k,x in enumerate(forcing)),arb_mat(705,1));w.tick()
    initial=[(known[i]-predictor[i]).intersection(error[i])for i in range(705)];e,q,rad=A.localize(D,v,weights,initial);known=[known[i].intersection(predictor[i]+e[i])for i in range(705)]
    w.save('old-endpoint-localization.json',dict(known=vector_json(known),weighted_bound=ball_json(q),radius=ball_json(rad)))
    x=[R.exact(M.point(v))for v in pin['point']];need(len(x)==705,'full exact recorded point');A.segment_in_domain(predictor,domain['error'],x,known)
    fresh=A.residual(x,18,normal,origin,R.exact(F(115,512)),[read(p)for p in events[36:]]);w.matrix('fresh-point-residual.json.gz',fresh)
    C0=B.exact_inverse(proposal['preconditioner_exact']);C1=matrix_file(ir/'run/chart003/inverse_linear_proposal.json.gz',705);need(all(x.is_exact()for row in C1.tolist()for x in row),'same literal exact independent C1')
    rhs=-(C0+C1*h)*fresh;e,q,rad=A.localize(D,rhs,weights,[known[i]-x[i]for i in range(705)]);answer=[known[i].intersection(x[i]+e[i])for i in range(705)];w.tick()
    w.save('fresh-endpoint-localization.json',dict(passed=True,point=vector_json(x),known=vector_json(answer),weighted_bound=ball_json(q),radius=ball_json(rad),old_parameter='115/512',old_domain=entry(unique(old/'chart003/run','cubic_correlated_domain')),scope='Identical independently inherited endpoint; no new existence claim.'))
    return vector_json(answer)

def new_root(run,nc,known,w):
    cfg=nc['settings'];need(cfg['parameter']=='31/2048','literal rotated root center');domain=read(unique(run,'cubic_correlated_domain'));proposal=read(unique(run,'cubic_predictor_return'))
    c,s,a,b,normal,origin,weights,h,rho,d,radii,error,whole=B.geometry(domain,proposal,cfg,nc,allowed_widths=(-6,));C0=B.exact_inverse(proposal['preconditioner_exact']);del proposal;gc.collect();sc=R.exact(F(cfg['parameter']));w.stage='new-root'
    membership=H.cubic_membership(known,c,s,a,b,normal,origin,sc,h,radii);w.save('new-root/projected-membership.json',membership);need(membership['passed']and membership['coordinates']==705,'all705 actual projected new-gauge membership')
    J0=H.jacobian_coefficients(c,s,a,b,error,arb(0),B.acquisitions(run,'gradient_path',0),13,18,normal,w.tick);Jw=H.jacobian_coefficients(c,s,a,b,error,d,B.acquisitions(run,'gradient_path',1),13,18,normal,w.tick);J=[J0[0],J0[1],Jw[2]];del J0,Jw;gc.collect()
    for k,j in enumerate(J):w.matrix(f'new-root/jacobian_coefficient_{k}.json.gz',j)
    rc=H.residual_coefficients(c,s,a,b,arb(0),B.acquisitions(run,'cubic_path',0),13,18,normal,origin,sc);rw=H.residual_coefficients(c,s,a,b,d,B.acquisitions(run,'cubic_path',1),13,18,normal,origin,sc);res=[arb_mat([[rc[i,k]if k<4 else rw[i,4]]for i in range(705)])for k in range(5)]
    for k,v in enumerate(res):w.matrix(f'new-root/residual_coefficient_{k}.json.gz',v)
    proof=H.polynomial_root(C0,J,res,weights,rho,h,error,w.tick,w.emit);need(proof['passed']and all(proof['coordinate_inclusions']),'full705 independent uniform root')
    divisors={str(k):[i for i in range(13)if not(whole[39*k+i]-whole[i]).contains(0)]for k in(1,2,3,6,9)};frame=[i for i in range(13)if not whole[26+i].contains(0)];need(all(divisors.values())and frame,'primitive18 and independent spectral frame')
    ends=[]
    for sign,corr in zip((-1,1),proof['endpoint_corrections'],strict=True):
        t=sign*h;ends.append(vector_json([c[i]+s[i]*t+a[i]*t*t/2+b[i]*t**3+R.original_ball(corr[i])for i in range(705)]))
    chart=dict(shooting_dimension=705,normal=nc['normal'],origin=nc['origin'],parameter_center=cfg['parameter'],parameter_endpoints=['-1/2048','63/2048'],settings=dict(halfwidth_exponent=-6,radius_exponent=-10));w.save('new-root/independent-proof.json',dict(passed=True,chart=chart,endpoints=ends,root=proof,membership=membership,proper_divisors=divisors,frame_nonzero=frame));w.save('connected-endpoint.json',dict(chart=chart,endpoint=ends[1]))
    return dict(passed=True,connected=True,strict_inclusions=705,anchor_coordinates=705,primitive_period=18,contraction=proof['contraction'],physical_right=ends[1][-3:])

class Work:
 def __init__(self):self.start=time.monotonic();self.used=0;self.stopped=False;self.stage='profile'
 def tick(self):need(not self.stopped and time.monotonic()-self.start<600 and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<6*1024**2,'600s/6GiB cap')
 def save(self,name,row,closing=False):
  raw=canonical_bytes(row);need(self.used+len(raw)<(1024 if closing else 1020)*1024**2,'1GiB with4MiB closing reserve');p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True)
  with p.open('xb')as f:f.write(raw);f.flush();os.fsync(f.fileno())
  self.used+=len(raw)
 def matrix(self,name,a):
  # Stream exact dyadic pairs row-by-row: no full JSON string/matrix copy.
  need(H.finite(a),'finite derived matrix');p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True);start=self.used
  try:
   with p.open('xb')as raw:
    with gzip.GzipFile(fileobj=raw,mode='wb',mtime=0)as f:
     f.write(b'{"encoding":"mid-rad-dyadic-pairs","rows":[')
     for i in range(a.nrows()):
      row=[[[int(t)for t in a[i,j].mid().man_exp()],[int(t)for t in a[i,j].rad().man_exp()]]for j in range(a.ncols())]
      f.write(((','if i else'')+json.dumps(row,separators=(',',':'))).encode());need(start+raw.tell()<1020*1024**2,'derived matrix cap preserves final reserve');self.tick()
     f.write(b']}')
    raw.flush();os.fsync(raw.fileno())
  finally:self.used=start+(p.stat().st_size if p.exists()else 0)
  need(self.used<1020*1024**2,'full derived matrix byte cap')

 def emit(self,tag,value):
  if isinstance(value,arb_mat):self.matrix(self.stage+'/'+tag+'.json.gz',value)
  else:self.save(self.stage+'/'+tag+'.json',value)
  self.tick()

def synthetic(w):
 n=705;C,J,res,h,rho=B.synthetic_fixture(n);before=time.monotonic();used=w.used
 proof=H.polynomial_root(C,J,res,[arb(1)]*n,rho,h,[arb(0,rho)]*n,w.tick,w.emit);need(proof['passed'],'fresh full705 degree5 polynomial control');elapsed=time.monotonic()-before;stored=w.used-used;forecast=8*elapsed+120;raw=4*stored+64*1024**2;peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024+2*1024**3
 report=dict(passed=bool(forecast<600 and raw<1024*1024**2 and peak<6*1024**3),seconds=elapsed,stored_bytes=stored,forecast_seconds=forecast,forecast_bytes=raw,forecast_peak_bytes=peak,dimension=705,target_arrays_unopened=True);w.save('profile.json',report);need(report['passed'],'complete measured600s/6GiB/1GiB source/serialization admission');del C,J,res;gc.collect()


def main():
    resource.setrlimit(resource.RLIMIT_AS,(10*1024**3,)*2);need(not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT).strip(),'clean exact source');need(os.getpriority(os.PRIO_PROCESS,0)>=10 and all(os.environ.get(k)=='1'for k in THREADS),'CPU1nice10');OUT.mkdir(parents=True,exist_ok=False);w=Work();signal.signal(signal.SIGTERM,lambda *_:setattr(w,'stopped',True));signal.signal(signal.SIGALRM,lambda *_:setattr(w,'stopped',True));signal.setitimer(signal.ITIMER_REAL,600);result=dict(passed=False,connected=False,native_calls=0,status='incomplete')
    try:
        source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();w.save('runtime.json',dict(source_commit=source,pid=os.getpid(),sources={n:entry(ROOT/n)for n in FILES},precision=480,native_calls=0,python=platform.python_version(),python_flint=flint.__version__,flint=flint.__FLINT_VERSION__,threads=dict.fromkeys(THREADS,'1'),nice=os.getpriority(os.PRIO_PROCESS,0)))
        for n in FILES:
            raw=(ROOT/n).read_bytes();need(hashlib.sha256(subprocess.check_output(['git','show',source+':'+n],cwd=ROOT)).hexdigest()==hashlib.sha256(raw).hexdigest(),'own exact source archive');p=OUT/'source'/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw);w.used+=len(raw)
        with ctx.workprec(480):
            card=read(ROOT/CARD);need(card['audit']==dict(output=str(OUT),seconds=600,rss_bytes=6*1024**3,as_bytes=10*1024**3,raw_bytes=1024*1024**2,precision=480,new_roots=1,native_calls=0),'literal fixed audit resource card');synthetic(w);ir,old,ep,pc,run,nc,point,pin,ev=bind(card,w);known=endpoint(ir,old,ep,pc,point,pin,ev,w);result.update(new_root(run,nc,known,w));result.update(status='independently_connected_rotated_cubic705',scope='Independent old-endpoint localization, fresh recorded native residual assembly, all actual derivative-domain and rotated-gauge membership gates, full705 polynomial root/primitive18/frame proof. Shared native acquisitions and Arb. No complementary-spectrum or global-window assertion.')
    except BaseException as e:result.update(status='failed_or_interrupted',error=type(e).__name__+': '+str(e),traceback=traceback.format_exc())
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);result.update(elapsed_seconds=time.monotonic()-w.start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss);w.save('outcome.json',result,True);w.save('manifest.json',dict(files={str(p.relative_to(OUT)):entry(p)for p in sorted(OUT.rglob('*'))if p.is_file()}),True)
    print(json.dumps(result));return 0 if result['passed']else 2
if __name__=='__main__':raise SystemExit(main())
