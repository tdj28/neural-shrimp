"""Independent saved native assembly/localization and correlated gauge audit."""
import gc,hashlib,json,os,resource,signal,subprocess,sys,time
from pathlib import Path
from fractions import Fraction
from flint import arb,arb_mat,ctx
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'src'),str(Path(__file__).parent)]
import review_eigenpair_march as B
E=B.E;O=B.O;need=E.need;read=B.read;compact=B.compact
from neural_shrimp.rigorous_orbits import canonical_bytes,vector_json,ball_json
RAW=B.BASE/'attempt-001-reanchor-target'
OUTER='5a344e0c0b891d6051c62f54e3138360e9a385f2c7f9133f40d8f133386ea7c8'
WORKER='2147813200b2c5cc90132baccafcee13fd4bce0bb9bd6a93d972d036d0af2171'
EXEC='2cd9e2dc43c48a920d8156d56f9e74a38fc65f04'
PRIOR=Path('/data2/neural-shrimp-data/audits/eigenpair-quadratic-march/attempt-001')
PRIORPIN='16950add27cdb24a9514a9fec7d9d37423c9d05d07e4bc3a84202bd4a6ca9d6d'
FILES=list(dict.fromkeys(B.FILES+['tools/reviews/review_eigenpair_reanchor.py','tests/reviews/test_eigenpair_reanchor_review.py','docs/plans/eigenpair-reanchor-independent-audit.md']))

def residual(chart,box,events,parameter):
 p,n=chart['period'],chart['dimension'];m=2*n;N=p*m+2;rows=[]
 need(len(events)==p,'all native residual returns')
 for k,e in enumerate(events):
  need(e['phase']==k and e['input']==box[k*m:(k+1)*m]+box[-2:],'native residual whole input')
  v=E.R.matrix(e['value']);need(v.nrows()==m and v.ncols()==1,'native residual shape')
  for i in range(m):rows.append(v[i,0]-(chart['sigma'] if k==p-1 and i>=n else 1)*E.R.original_ball(box[((k+1)%p)*m+i]))
 q=E.vector(box[n:m]);rows.append((sum((v*v for v in q),arb(0))-1)/2)
 rows.append(sum((E.R.exact(chart['normal'][i])*(E.R.original_ball(box[i])-E.R.exact(chart['origin'][i])) for i in range(N)),arb(0))-E.R.exact(parameter))
 return arb_mat([[v] for v in rows])

def local_image(center,g,A,C,box):
 N=len(center);D=E.M.ident(N)-C*A
 v=arb_mat([[center[i]] for i in range(N)])-C*g+D*arb_mat([[box[i]-center[i]] for i in range(N)])
 return v,D

def worker(out):
 need(not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT).strip(),'clean auditor')
 need(os.getpriority(os.PRIO_PROCESS,0)>=10 and all(os.environ.get(k)=='1' for k in O.THREADS),'CPU1nice10')
 resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,)*2);out.mkdir(parents=True,exist_ok=False);start=time.monotonic();stop=[False];result={'passed':False,'charts':[]}
 def save(n,v):
  b=canonical_bytes(v);need(len(b)+sum(p.stat().st_size for p in out.rglob('*') if p.is_file())<32*1024**2,'32MiBraw');p=out/n;p.parent.mkdir(parents=True,exist_ok=True)
  with p.open('xb') as f:f.write(b);f.flush();os.fsync(f.fileno())
 def tick():need(not stop[0] and time.monotonic()-start<150 and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<3*1024**2,'150s/3GiB')
 signal.signal(signal.SIGTERM,lambda *_:stop.__setitem__(0,True));signal.signal(signal.SIGALRM,lambda *_:stop.__setitem__(0,True));signal.setitimer(signal.ITIMER_REAL,150)
 save('runtime.json',{'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'sources':{n:O.entry(ROOT/n) for n in FILES},'pid':os.getpid(),'nice':os.getpriority(os.PRIO_PROCESS,0),'threads':dict.fromkeys(O.THREADS,'1')})
 for n in FILES:p=out/'source'/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/n).read_bytes())
 try:
  O.verify(PRIOR,PRIORPIN);need(read(PRIOR/'run/outcome.json')['passed'],'independent parent proof');previous=read(PRIOR/'run/unit2-endpoint.json')['original']
  O.verify(RAW,OUTER);O.verify(RAW/'run',WORKER);run=RAW/'run';rt=read(run/'runtime.json');la=read(RAW/'launcher-result.json');outcome=read(run/'outcome.json');inp=read(run/'inputs.json')
  need(rt['commit']==la['commit']==EXEC and outcome['passed'] and la['exit_code']==0 and not la['owned_live_pids'] and la['stop_reason'] is None,'successful source-bound target')
  need(not Path('/proc',str(rt['pid'])).exists(),'worker absent')
  for name,v in rt['sources'].items():need(O.entry(run/'source'/name)==v and hashlib.sha256(subprocess.check_output(['git','show',EXEC+':'+name],cwd=ROOT)).hexdigest()==v['sha256'],'producer committed source '+name)
  need(inp['map_inputs']==read(B.B.RAW/'run/exact-map-inputs.json'),'same native map')
  phase=[];events=[];iterations=0;index=0;known=None;fresh=None
  with ctx.workprec(256):
   for path in sorted((run/'events').glob('*.json.gz')):
    kind=path.name.split('-',1)[1].removesuffix('.json.gz')
    if kind not in ('localization_start','phase_return','localization_center_return','localization_image_return','localization_intersection','localization_return','reanchor_proposal','path_phase_return','path_assembly_return','chart_return'):continue
    v=read(path)
    if kind=='localization_start':
     need(v['parent_right']==previous['parameter_endpoints'][1]=='19/32768' and v['normal']==previous['normal'] and v['origin']==previous['origin'],'entire independently audited endpoint gauge')
     need(len(v['initial_box'])==470 and all(E.M.interval(a)[0]<=E.M.interval(b)[0]<=E.M.interval(b)[1]<=E.M.interval(a)[1] for a,b in zip(v['initial_box'],previous['endpoint_root_boxes'][1],strict=True)),'decoded box covers complete audited endpoint')
     original_full=read(B.BASE/'attempt-002-march-target/run/events/0548-chart_return.json.gz')
     need(v['preconditioner']==original_full['preconditioner'],'fixed exact parent C');del original_full;gc.collect()
     center=E.vector(v['center']);box=E.vector(v['initial_box']);need(all(z.is_exact() for z in center),'fixed exact center');C=E.R.matrix(v['preconditioner']);need(all(C[i,j].is_exact() for i in range(470) for j in range(470)),'exact C');savedstart=v;tick()
    elif kind=='phase_return':
     if iterations<3:phase.append(v)
     else:events.append((kind,v))
    elif kind=='localization_center_return':
     need(v['center']==savedstart['center'],'fixed center');g=residual(previous,v['center'],phase,savedstart['parent_right']);E.R.same(g,v['residual'],'independent center residual');phase=[];tick()
    elif kind=='localization_image_return':
     need(v['iteration']==iterations and v['prior_box']==vector_json(box),'same entire current root box')
     domain=[center[i].union(box[i]) for i in range(470)];E.identity(domain,v['domain'],'hull(center,known box)')
     A=E.rebuild_whole(dict(previous,whole_box=v['domain']),phase);E.R.same(A,v['jacobian'],'native localization matrix');image,D=local_image(center,g,A,C,box);E.R.same(image,v['image'],'localization identity');E.R.same(D,v['defect'],'localization defect');phase=[];tick()
    elif kind=='localization_intersection':
     need(v['iteration']==iterations,'intersection order');new=[box[i].intersection(image[i,0]) for i in range(470)];E.identity(new,v['box'],'outward root intersection');box=new;iterations+=1;save(f'localization-{iterations}.json',{'passed':True,'box':vector_json(box)});del A,D,image;gc.collect();tick()
    elif kind=='localization_return':
     need(iterations==3 and not phase and v['box']==vector_json(box) and v['passed'],'three completed root localizations');known={'box':vector_json(box)};fresh=known
    elif kind=='reanchor_proposal':
     proposal=v;need(v['parameter']=='0' and v['halfwidth_exponent']==-7 and v['radius_exponent']==-17,'fixed reanchor proposals')
     need(all(Fraction(o)==E.M.point(ball_json(b.mid())) for o,b in zip(v['origin'],box,strict=True)),'localized midpoint gauge origin only')
     need(v['initial']==v['origin'] and len(v['normal'])==470 and all(E.R.exact(z).is_exact() for z in v['normal']),'exact full native gauge');del C,g;gc.collect()
    elif kind in ('path_phase_return','path_assembly_return'):events.append((kind,v))
    elif kind=='chart_return':
     c=v;settings={'precision':256,'newton_steps':2,'halfwidth_exponent':-7,'radius_exponent':-17}
     need(c['normal']==proposal['normal'] and c['origin']==proposal['origin'] and c['weights']==proposal['weights'] and Fraction(c['parameter_center'])==Fraction(index,64),'same proposed gauge/radii/order')
     r256=E.check_chart(c,known,events,256,settings);save(f'chart-{index}-256.json',r256);tick();r384=E.check_chart(c,fresh,events,384,settings);save(f'chart-{index}-384.json',r384);tick()
     need(r256['passed'] and r384['passed'],'fresh full470D root primitive correlated join')
     result['charts'].append({'index':index,'contraction384':r384['contraction'],'anchor_coordinates':r384['anchor_coordinates']});previous=compact(c);known={'box':c['endpoint_root_boxes'][1],'exact_parameter':c['parameter_endpoints'][1]};fresh={'box':r384['recomputed_endpoints'][1],'exact_parameter':c['parameter_endpoints'][1]};events=[];index+=1;del c,v,r256,r384;gc.collect()
  need(index==2 and not events and iterations==3,'complete localization/reanchor');save('endpoint.json',{'original':previous,'independent384':fresh})
  result.update(passed=True,localizations=3,total_connected_charts=12,new_gauge=['-1/128','3/128'],scope='Independent native residual/J assembly and exact256 known-root localization; exact256 plus fresh384 strict470D root/primitive/correlated native join. Shared local analytic blocks/pathjets and Arb. No genericity, attraction, cusp or complete shrimp claim.')
 except BaseException as exc:result['error']=type(exc).__name__+': '+str(exc)
 finally:
  signal.setitimer(signal.ITIMER_REAL,0);result.update(elapsed_seconds=time.monotonic()-start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss);save('outcome.json',result);save('manifest.json',{'files':{str(p.relative_to(out)):O.entry(p) for p in sorted(out.rglob('*')) if p.is_file()}})
 print(json.dumps(result));return 0 if result['passed'] else 2
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();raise SystemExit(worker(a.out))
