"""Fixed saved exact-rational domain mismatch diagnosis; no model or Arb calls."""
import hashlib,json,os,resource,signal,sys,time
from fractions import Fraction as F
from pathlib import Path
ROOT=Path('/data2/neural-shrimp-data/audits/native704-complement-independent/attempt002')
PARENT=Path('/data2/neural-shrimp-data/explore/eigenpair-locus-prototype/generalized-chain/unit-pair/attempt007-correlated')
PIN='63693fa6403218a6e4c7723f5a599ab311b96ff233d806595f62c3f5214eea93'
PPIN='66ac6a538185a53f016d5a6f9eeb7c0ab9102158287bea11e621389207c3b155'
def pair(v):return F(v[0])*F(2)**v[1]
def interval(v):m=pair(v['mid']);r=pair(v['rad']);return m-r,m+r

def compare(old,new):
 assert len(old)==len(new)==704;fail=[]
 for i,(a,b)in enumerate(zip(old,new,strict=True)):
  lo,hi=interval(a);x,y=interval(b);excess=max(F(0),lo-x,y-hi)
  if excess:
   width=hi-lo;fail.append(dict(index=i,phase=i//39 if i<702 else None,kind=('state'if i%39<13 else'q'if i%39<26 else'r')if i<702 else'rate',left_excess=str(max(F(0),lo-x)),right_excess=str(max(F(0),y-hi)),relative_excess=float(excess/width)if width else None,new_to_old_width=float((y-x)/width)if width else None,midpoint_difference=float((x+y-lo-hi)/2)))
 return dict(contained=not fail,failed_coordinates=len(fail),max_relative_excess=max((r['relative_excess']for r in fail),default=0),rows=fail)

def read_bound(root,pin,name):
 m=root/'manifest.json';assert hashlib.sha256(m.read_bytes()).hexdigest()==pin
 row=json.loads(m.read_text())['files'][name];p=root/name;data=p.read_bytes();assert len(data)==row['bytes']and hashlib.sha256(data).hexdigest()==row['sha256']and not p.stat().st_mode&0o222
 return json.loads(data),{'root':str(root),'manifest':pin,'file':name,**row}

def main():
 resource.setrlimit(resource.RLIMIT_AS,(1024**3,)*2);signal.signal(signal.SIGALRM,lambda *_:(_ for _ in()).throw(TimeoutError('30s')));signal.alarm(30);start=time.monotonic()
 assert os.getpriority(os.PRIO_PROCESS,0)>=10
 old,a=read_bound(PARENT,PPIN,'run/actual-domain-inputs.json');new,b=read_bound(ROOT,PIN,'run/localization-480.json');result={'status':'saved_exact_endpoint_diagnosis','bindings':[a,b],'native_calls':0,'pid':os.getpid(),'domains':{k:compare(old[k],new[k])for k in('center','whole','tangent')}}
 signal.alarm(0);result.update(seconds=time.monotonic()-start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss);assert result['peak_rss_kib']<448*1024
 raw=json.dumps(result,indent=2,allow_nan=False)+'\n';assert len(raw.encode())<2*1024**2;print(raw)
if __name__=='__main__':main()
