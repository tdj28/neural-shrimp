"""Exact-rational interval replay of the conditional55-to705 junction lift.

Native derivative returns and known-root authority are shared. No producer
refinement/lift reducer and no native map is imported or executed.
"""
import os
THREADS=('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')
for k in THREADS:os.environ[k]='1'
import hashlib,json,platform,resource,signal,subprocess,sys,time,traceback
from fractions import Fraction as F
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(Path(__file__).parent))
import review_cifar_stationary_merit as M
need=M.need;read=M.read;entry=M.entry
OUT=Path('/data2/neural-shrimp-data/audits/junction-native-lift-exact/attempt001/run')
PARENT=Path('/data2/neural-shrimp-data/explore/eigenpair-locus-prototype/generalized-chain/unit-pair/pal/junction-lift001/attempt003')
EXEC='0ba9605de2eb98862bda5c09f436b9d4a15f93ee'
PINS=['ebfb53346cfc47589ecd24dc14ef8e29c25e5ba62cbb616ceda97176cc4eadb4','d2c869aa01524a394efe65ba033c62b656c3f206e56bd331db799be7293f7411']
KNOWN=Path('/data2/neural-shrimp-data/audits/native-unit-pair-flip-independent/attempt002')
KSOURCE='1a991e0205cf6ddf00adb5b678a14d0f44857a49'
KPINS=['d14c91bdfc3c754a6654a8c30ffbd4b69ac59150aeea717d1426fc2b1480e952','7f0e39d8b206859914b5aabb4da063427ad81c0b9f26f836bbb6b9899d47db49']
FILES=['tools/reviews/review_junction_lift_exact.py','tools/reviews/launch_junction_lift_exact.py','tools/reviews/review_cifar_stationary_merit.py','tests/reviews/test_junction_lift_exact.py','docs/plans/junction-lift-exact-independent.md']


def bounds(v):
 need(set(v)>={'mid','rad'},'serialized finite interval');m,e=v['mid'];a,b=v['rad'];need(all(type(x)is int for x in(m,e,a,b))and a>=0,'integer dyadic endpoints');x=F(m)*F(2)**e;y=F(a)*F(2)**b;return x-y,x+y

def encloses(outer,inner):return outer[0]<=inner[0]<=inner[1]<=outer[1]
def add(x,y):return x[0]+y[0],x[1]+y[1]
def neg(x):return -x[1],-x[0]
def mul(x,y):
 z=[a*b for a in x for b in y];return min(z),max(z)
def dot(x,y):
 z=(F(0),F(0))
 for a,b in zip(x,y,strict=True):z=add(z,mul(a,b))
 return z

def mat(rows):return [[bounds(v)for v in row]for row in rows]
def vec(rows):return [bounds(v)for v in rows]
def exact(v):
 x=bounds(v);need(x[0]==x[1],'exact point/inverse proposal');return x


def refinement(root,record,events,tick=lambda:None):
 n=len(root['center']);need(n==55 and len(root['box'])==n,'full known55 root')
 c=[exact(x)for x in root['center']];whole=vec(root['box']);C=[[exact(x)for x in row]for row in root['preconditioner']];J=mat(root['whole_jacobian']);value=[bounds(row[0])for row in root['value']]
 need(len(C)==len(J)==n and all(len(x)==n for x in C+J)and len(value)==n,'complete independent root arrays')
 need(all(encloses(b,x)for b,x in zip(whole,c))and len(events)==4,'center in inherited original domain and exactly3 refinements')
 first=events[0];need(first['center']==root['center']and first['whole']==root['box']and first['preconditioner']==root['preconditioner']and first['value']==root['value']and first['whole_jacobian']==root['whole_jacobian'],'literal old acquired derivative and center association')
 D=mat(first['defect']);need(len(D)==n and all(len(row)==n for row in D),'full defect')
 for i in range(n):
  for j in range(n):
   v=add((F(i==j),F(i==j)),neg(dot(C[i],[J[k][j]for k in range(n)])))
   need(encloses(D[i][j],v),'exact rational full defect enclosure')
  tick()
 q=max(sum(max(abs(x[0]),abs(x[1]))for x in row)for row in D);need(q<1 and bounds(first['contraction'])[0]>=q,'exact complete row-norm contraction')
 rhs=[neg(dot(row,value))for row in C];error=[add(b,neg(x))for b,x in zip(whole,c)]
 for k,event in enumerate(events[1:]):
  need(event['iteration']==k,'ordered refinement');image=[bounds(row[0])for row in event['image']];retained=[bounds(row[0])for row in event['retained']];need(len(image)==len(retained)==n,'all55 refinement coordinates')
  for i in range(n):
   target=add(rhs[i],dot(D[i],error));need(encloses(image[i],target),'exact rational fixed-point image')
   intersection=(max(error[i][0],image[i][0]),min(error[i][1],image[i][1]));need(encloses(retained[i],intersection),'conditional known-root intersection retained')
  error=retained;tick()
 box=vec(record['box']);need(record['passed']and record['dimension']==n and len(box)==n,'complete final refined root')
 need(all(encloses(z,add(x,e))and encloses(b,z)for z,x,e,b in zip(box,c,error,whole,strict=True)),'same root in literal inherited derivative box')
 return dict(passed=True,dimension=n,iterations=3,row_norm_upper=str(q),method='Exact Fraction endpoint interval sums/products; no Arb matrix refinement routine.'),record['box']


def check_seed(inp,state,mode,rates,k,label):
 need(inp['phase']==k and inp['direction']==label and inp['state']==state and inp['mode']==mode and inp['rates']==rates,'literal propagated acquisition domain')
 g=mat(inp['base_gradients']);m=mat(inp['mixed']);need(len(g)==len(m)==15 and all(len(x)==15 for x in g+m),'full actual mixed-jet seed arrays')
 need(all(g[i][j]==(F(i==j),F(i==j))and m[i][j]==(F(0),F(0))for i in range(15)for j in range(15)),'exact held-direction native seeds')


def lift(box,events,reported,tick=lambda:None):
 need(len(box)==55 and len(events)==72,'full55 source and36 input/return pairs');state=box[:13];rates=box[13:15];q=box[15:28];r=box[28:41];angle=box[41];flat=[]
 need(all(bounds(v)[0]>0 for v in rates+[angle])and bounds(angle)[1]<=F(1,4),'same positive physical rates/angle')
 for k in range(18):
  flat+=state+q+r;outputs=[]
  for offset,label,mode in [(0,'q',q),(2,'r',r)]:
   inp,ret=events[4*k+offset:4*k+offset+2];check_seed(inp,state,mode,rates,k,label)
   need(ret['phase']==k and ret['direction']==label and ret['input']==state+mode+rates,'exact native return/input association')
   value=mat(ret['value']);block=mat(ret['block']);rate=mat(ret['rates']);need(len(value)==len(block)==len(rate)==26 and all(len(a)==1 for a in value)and all(len(a)==26 for a in block)and all(len(a)==2 for a in rate),'full finite native return dimensions');outputs.append(ret)
  left,right=outputs;need(left['value'][:13]==right['value'][:13],'same state advancement independent of direction')
  for i in range(13):
   need(left['block'][i][:13]==right['block'][i][:13]and left['block'][13+i][13:]==left['block'][i][:13]and right['block'][13+i][13:]==right['block'][i][:13],'literal complete repeated DF blocks')
  state=[x[0]for x in left['value'][:13]];q=[x[0]for x in left['value'][13:]];r=[x[0]for x in right['value'][13:]];tick()
 flat+=rates+[angle];need(reported['dimension']==13 and reported['period']==18 and reported['shooting_dimension']==705 and reported['native_calls']==36 and reported['box']==flat,'literal full705 normalized native lift')
 return dict(passed=True,phases=18,shared_native_returns=36,coordinates=705,no_new_native_calls=True,box=flat,scope='Same independently proved normalized55 junction mapped into705 native coordinates; full-state periodicity and normalization inherited from known root. No continuation membership follows here.')


def authority(rt,owner,source,known):
 key='commit'if known else'source_commit'
 need(rt[key]==owner[key]==source and owner['exit_code']==0 and not owner['owned_live_pids'],'literal completed source and owner')
 pids=[rt['pid']]
 if not known:
  need(owner['worker_pid_absent'],'explicit modern worker absence');pids += [owner['launcher_pid'],owner['owned_group_pid']]
 for pid in pids:need(type(pid)is int and not Path('/proc',str(pid)).exists(),'actual predecessor absent')


def archive(root,source,tick):
 rt=read(root/'run/runtime.json');authority(rt,read(root/'launcher-result.json'),source,root==KNOWN)
 for n,v in rt['sources'].items():
  tick();need(entry(root/'run/source'/n)==v and hashlib.sha256(subprocess.check_output(['git','show',source+':'+n],cwd=ROOT)).hexdigest()==v['sha256'],'complete committed source archive')
 return rt


def bind(tick):
 for p,pins in[(PARENT,PINS),(KNOWN,KPINS)]:
  for path,pin in zip((p,p/'run'),pins,strict=True):M.inventory(path,pin,tick)
 rt=archive(PARENT,EXEC,tick);kr=archive(KNOWN,KSOURCE,tick);b=read(PARENT/'run/input-binding.json');ko=read(KNOWN/'run/outcome.json');o=read(PARENT/'run/outcome.json')
 need(ko['passed']and o['passed']and o['native_calls']==108 and o['target_acquired']and rt['settings']['precision']==512,'known independent junction and complete lifted108-call packet')
 need(b['independent_root']==str(KNOWN)and b['source']==KSOURCE and b['outer']==KPINS[0]and b['worker']==KPINS[1]and b['root']==entry(KNOWN/'run/fresh-root.json'),'literal full independent known-root authority')
 for n,v in b['native_source'].items():need(v==kr['sources'][n]==rt['sources'][n],'unchanged native derivative source')
 need(b['map_inputs']==read(KNOWN/'run/binding.json')['map_inputs'],'same exact network/objective')
 return dict(known=str(KNOWN),known_source=KSOURCE,known_pins=KPINS,producer=str(PARENT),producer_source=EXEC,producer_pins=PINS,native_source=b['native_source'])


def dyadic(v):
 v=F(v);d=v.denominator;need(d&(d-1)==0,'dyadic synthetic');return [v.numerator,-(d.bit_length()-1)]
def ball(a,b=None):
 a=F(a);b=a if b is None else F(b);return dict(mid=dyadic((a+b)/2),rad=dyadic((b-a)/2))
def fixture(n=55):
 C=[[(F(i==j)+F(1,16384),)*2 for j in range(n)]for i in range(n)];J=[[(F(i==j)-F(1,32768),F(i==j)+F(1,32768))for j in range(n)]for i in range(n)]
 D=[[add((F(i==j),)*2,neg(dot(C[i],[J[k][j]for k in range(n)])))for j in range(n)]for i in range(n)];enc=lambda a:[[ball(*v)for v in row]for row in a];root=dict(center=[ball(0)]*n,box=[ball(-F(1,1024),F(1,1024))]*n,preconditioner=enc(C),whole_jacobian=enc(J),value=[[ball(0)]for _ in range(n)])
 q=max(sum(max(abs(v[0]),abs(v[1]))for v in row)for row in D);ev=[dict(center=root['center'],whole=root['box'],preconditioner=root['preconditioner'],value=root['value'],whole_jacobian=root['whole_jacobian'],defect=enc(D),contraction=ball(q))];err=vec(root['box'])
 for k in range(3):
  image=[dot(row,err)for row in D];err=[(max(a[0],b[0]),min(a[1],b[1]))for a,b in zip(err,image)];ev.append(dict(iteration=k,image=[[ball(*v)]for v in image],retained=[[ball(*v)]for v in err]))
 record=dict(passed=True,dimension=n,box=[ball(*v)for v in err]);return root,record,ev


def main():
 resource.setrlimit(resource.RLIMIT_AS,(2*1024**3,)*2);need(not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT).strip()and os.getpriority(os.PRIO_PROCESS,0)>=10,'clean nice10 source');OUT.mkdir(parents=True,exist_ok=False);start=time.monotonic();used=0;result=dict(passed=False,native_calls=0)
 def tick():need(time.monotonic()-start<90 and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<1024**2,'90s1GiB cap')
 def stop(*_):raise TimeoutError('owned exact interval audit deadline')
 signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGALRM,stop);signal.setitimer(signal.ITIMER_REAL,90)
 def save(name,value,closing=False):
  nonlocal used
  b=(json.dumps(value,sort_keys=True,indent=2)+'\n').encode();need(used+len(b)<(32 if closing else 28)*1024**2,'32MiB4MiB closing reserve');p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True)
  with p.open('xb')as f:f.write(b);f.flush();os.fsync(f.fileno())
  used+=len(b)
 try:
  source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();save('runtime.json',dict(source_commit=source,python=platform.python_version(),pid=os.getpid(),files={n:entry(ROOT/n)for n in FILES},native_calls=0,interval_backend='exact fractions.Fraction endpoints'))
  for n in FILES:
   data=(ROOT/n).read_bytes();need(data==subprocess.check_output(['git','show',source+':'+n],cwd=ROOT),'exact own committed source');q=OUT/'source'/n;q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes(data);used+=len(data)
  t=time.monotonic();root,record,events=fixture();refinement(root,record,events,tick);seconds=time.monotonic()-t;profile=dict(passed=4*seconds+20<90,seconds=seconds,forecast_seconds=4*seconds+20,dimension=55,target_arrays_unopened=True);save('profile.json',profile);need(profile['passed'],'full55 exact-rational measured admission');del root,record,events
  binding=bind(tick);save('input-binding.json',binding);root=read(KNOWN/'run/fresh-root.json');record=read(PARENT/'run/known-root-refinement.json');paths=sorted((PARENT/'run/events').glob('*target*'));need(len(paths)==76 and[int(p.name.split('-')[0])for p in paths]==list(range(152,228)),'complete ordered target events')
  events=[read(p)for p in paths];answer,box=refinement(root,record,events[:4],tick);save('refinement.json',answer);lifted=lift(box,events[4:],read(PARENT/'run/native705-lift.json'),tick);save('independent705.json',lifted);save('target-events.json',{p.name:entry(p)for p in paths});result.update(passed=True,refinement=answer,coordinates=705,shared_native_returns=36,scope=lifted['scope'])
 except BaseException as e:result.update(error=type(e).__name__+': '+str(e),traceback=traceback.format_exc())
 finally:
  signal.setitimer(signal.ITIMER_REAL,0);result.update(elapsed_seconds=time.monotonic()-start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss);save('outcome.json',result,True);save('manifest.json',dict(files={str(p.relative_to(OUT)):entry(p)for p in OUT.rglob('*')if p.is_file()}),True)
 print(json.dumps(result));return 0 if result['passed']else 2
if __name__=='__main__':raise SystemExit(main())
