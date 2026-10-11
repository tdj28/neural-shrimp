"""Independent fsum/action-association replay of adjacent-pair block ordering."""
import hashlib,json,math,os,resource,signal,subprocess,sys,time,traceback
from pathlib import Path
import numpy as np
import review_tiny_lm_parity_order as P
M=P.M;B=P.B
need,read,entry,exact,array,dot,norm,close=P.need,P.read,P.entry,P.exact,P.array,P.dot,P.norm,P.close
ROOT=Path(__file__).resolve().parents[1];DATA=P.DATA;THREADS=P.THREADS
OUT=DATA/'audits/tiny-lm-pair-blocks-independent/attempt001/run'
TARGET=DATA/'explore/tiny-lm-pair-blocks/attempt001-target'
SPEC=dict(root=str(TARGET),source='1c91e14fb01276a5d764e8facc65ab6585f38606',outer='03022d864990b299468c026a1b1bf9210a61a028bfae91f02a604fe7fb3cc5e6',worker='615376d85ffcc62f6d2a24a0799544dab16c3e4a61b18cb4fae9f2e22fc19b58')
AUTH=dict(root=str(DATA/'audits/tiny-lm-parity-order-independent/attempt001'),source='b481b47224d41ccc17f73b5a3b6502e8b4e1bddc',outer='e114103e8fde5e925a028e9f9d34f4eec269fb8f01c7a6f16333fc6a4c19c7c2',worker='03e33ee5f82550563639b0d08dfcb49a45948446dbdd637444232e50f33ac586')
ORDERS=dict(U=[i for j in range(14,-1,-2)for i in(j,j+1)]);ORDERS['V']=ORDERS['U'][::-1]
GATES=dict(transpose=1e-8,decomposition=1e-12,material=.2,attenuation=.1)
FILES=sorted(set(P.FILES+['tools/review_tiny_lm_pair_blocks.py','tools/launch_tiny_lm_pair_blocks_review.py','tests/test_tiny_lm_pair_blocks_review.py','docs/plans/tiny-lm-pair-blocks-independent.md']))

def reduce(u,p,r,x,y,f,g,prior):
 row,vec=P.reduce(u,p,r,x,y,f,g);den=norm(prior);need(den>0,'nonzero independently qualified parity contrast')
 row['products']={dict(P='P',R='R',Q='U',T='V')[k]:v for k,v in row['products'].items()}
 row['pair_block_odd_norm']=row.pop('grouped_odd_norm');row['pair_block_minus_ordered_squared']=row.pop('grouped_minus_ordered_squared')
 row.update(even_relative_parity_even=row['even_norm']/den,prior_parity_even_norm=den,gates=GATES)
 return row,{dict(parity_even='pair_block_even',grouped_odd='pair_block_odd',chronological_odd='chronological_odd')[k]:v for k,v in vec.items()}

def gates(cases):
 need([(r['seed'],r['rate_role'])for r in cases]==[(s,k)for s in(28161,28162)for k in('original','quarter_step')],'literal four-case complete order')
 qualified=all(r['qualified']for r in cases);originals=[r['comparison']for r in cases if r['rate_role']=='original']
 return dict(all_four_qualified=qualified,material_in_both_originals=qualified and all(r['even_relative_ordered']>=.2 for r in originals),tenfold_attenuation_in_both_originals=qualified and all(r['even_relative_parity_even']<=.1 for r in originals))

def metadata(tick):
 inv,parity,mean=P.metadata(tick)
 ar,art,ao,iv=P.bundle(AUTH,tick);inv+=iv
 close(ao['cases'],parity['cases']);need(ao['passed']and ao['native_calls']==0 and ao['retained_hvp_checked']==224 and ao['old_reverse_hvp_checked']==32 and all(r['qualified']for r in ao['cases']),'entire qualified independent parity authority')
 need(read(ar/'run/input-binding.json')['producer']==P.SPEC,'literal independent parity source')
 target,rt,out,iv=P.bundle(SPEC,tick);inv+=iv
 card=read(target/'run/run-card.json');need(card==read(target/'run/source/docs/plans/tiny-lm-pair-blocks-inputs.json')and card['parity_authority']==AUTH and card['parity_producer']==P.SPEC and card['status']=='target_admitted','literal committed pair design')
 settings=dict(as_bytes=8*1024**3,gates=GATES,orders=ORDERS,profile_counts=dict(hvp=2),raw_bytes=192*1024**2,roles=['original','quarter_step'],rss_bytes=4*1024**3,seeds=[28161,28162],target_counts=dict(hvp=128),worker_seconds=dict(profile=60,target=240))
 need(card['settings']==settings,'all fixed orders/gates/resources')
 ib=read(target/'run/input-binding.json');pb=read(target/'run/parity-binding.json')
 need(ib['producer']==M.PRODUCER and ib['independent']==P.MEAN and ib['event_pairs']==1308 and pb['producer']==P.SPEC and pb['independent']==AUTH,'complete inherited source/response chain')
 pref=read(target/'run/profile-reference.json');need(pref['profile']==card['profile'],'same fresh profile authority')
 pr,prt,po,iv=P.bundle(card['profile'],tick);inv+=iv
 for n,v in rt['sources'].items():
  if n!='docs/plans/tiny-lm-pair-blocks-inputs.json':need(prt['sources'][n]==v,'exact same profiled source '+n)
 oldrt=read(P.TARGET/'run/runtime.json')
 need(rt['sources']['src/neural_shrimp/tiny_lm.py']==oldrt['sources']['src/neural_shrimp/tiny_lm.py'],'unchanged full native model')
 for k in('torch','numpy'):need(rt[k]==prt[k]==oldrt[k],'unchanged native library '+k)
 need(po['passed']and po['mode']=='profile'and po['attempted']==po['returned']==dict(hvp=2)and po['event_pairs']==2 and po['seconds']<60 and po['peak_rss_kib']<4*1024**2,'complete bounded fresh profile')
 old=max(x for row in pb['all_upstream_hvp_costs'].values()for x in row);need(old==pb['effective_max_hvp_seconds']and all(math.isfinite(x)and x>0 for row in pb['all_upstream_hvp_costs'].values()for x in row),'entire retained cold/actual maxima')
 largest=max(old,*po['times']['hvp']);forecast=dict(seconds=2*128*largest+30,bytes=2*356*max(po['return_bytes']['hvp'])+64*1024**2+po['preforecast_bytes'],effective_max_hvp_seconds=largest,old_max_hvp_seconds=old,new_max_hvp_seconds=max(po['times']['hvp']),vectors=356)
 close(forecast,po['forecast']);close(forecast,pref['original_forecast']);close(forecast['seconds'],pref['current_forecast']['seconds']);need(forecast['seconds']<240 and forecast['bytes']<192*1024**2 and pref['current_forecast']['bytes']<192*1024**2,'complete measured fixed target admission')
 need(out['passed']and out['mode']=='target'and out['attempted']==out['returned']==dict(hvp=128)and out['event_pairs']==128 and len(out['cases'])==4 and out['seconds']<240 and out['peak_rss_kib']<4*1024**2,'complete bounded pair acquisition')
 return inv,out,ao,mean

def audit(tick,save):
 inv,out,authority,mean=metadata(tick);ev=M.Events(TARGET/'run',tick);pe=M.Events(P.TARGET/'run',tick);me=M.Events(M.TARGET/'run',tick);cases=[];rounding=0
 for seed in(28161,28162):
  folder=M.RESPONSE/'run'/f'seed-{seed}-arm-256';eta=read(folder/'inputs.json')['eta'];u=array(folder/'directions.npy',(2,15776))[0]
  for role in('original','quarter_step'):
   label=f'seed-{seed}-{role}';rate=eta/(1 if role=='original'else 4)
   if role=='original':
    ref=array(folder/'reference.npy',(17,15776));grad=array(folder/'reference-gradients.npy',(17,15776));frozen=array(folder/'direction-0-frozen.npy',(17,15776))[16];cg=array(folder/'readout-gradients.npy',(3,15776))[2]
    reverse=array(P.B.TARGET/'run'/f'seed-{seed}-arm-256/direction-0-horizon-16/reverse-trace.npy',(17,15776))[-1]
   else:
    ref=me.alias(label,'reference',(17,15776));grad=me.alias(label,'reference-gradients',(17,15776));frozen=me.alias(label,'frozen',(17,15776))[16];cg=me.alias(label,'readout-gradients',(3,15776))[2];reverse=pe.alias(label,'R',(17,15776))[-1]
   exact(ref[1:],ref[:-1]-rate*grad[:-1],'complete inherited GD states');exact(cg,grad[16]-.001*ref[16],'literal CE gradient')
   bind=read(TARGET/'run'/label/'binding.json');need(bind==read(P.TARGET/'run'/label/'binding.json')and bind['eta']==rate and bind['full_state']and bind['original_qualification'],'complete exact case/token/state/response genealogy')
   qualified=authority['cases'][len(cases)];need((qualified['seed'],qualified['rate_role'])==(seed,role)and qualified['qualified'],'independent prior case')
   baseline=dict(P=pe.alias(label,'P',(17,15776))[-1],R=reverse,frozen=frozen,ce_gradient=cg,parity_even=pe.alias(label,'parity_even',(15776,)))
   for key,value in baseline.items():exact(ev.alias(label,'baseline-'+key,(15776,)),value,'literal independently qualified baseline '+key)
   values={}
   for key,order in ORDERS.items():
    tr=ev.alias(label,key,(17,15776));exact(tr[0],u,'same full-state initial direction');need(read(TARGET/'run'/label/(key+'-order.json'))==dict(indices=order),'literal adjacent-pair permutation')
    for j,state in enumerate(order):exact(tr[j+1],tr[j]-rate*ev.call(label+'/'+key,'hvp',ref[state],tr[j]),'every actual same-state/vector recurrence')
    values[key]=tr[-1]
   row,vec=reduce(u,baseline['P'],baseline['R'],values['U'],values['V'],frozen,cg,baseline['parity_even'])
   terms=dict(pair_block_even=[baseline['P'],baseline['R'],values['U'],values['V']],chronological_odd=[baseline['P'],baseline['R']],pair_block_odd=[values['U'],values['V']])
   for key,value in vec.items():rounding=max(rounding,P.derived(ev.alias(label,key,(15776,)),value,terms[key]))
   result=dict(seed=seed,rate_role=role,qualified=row['qualified']and qualified['qualified'],comparison=row,inherited_response_qualified=True)
   saved=read(TARGET/'run'/label/'result.json');need(saved==out['cases'][len(cases)],'entire ordered producer case');close(result,saved);cases.append(result);save(label+'.json',result)
 need(ev.n==128 and ev.counts==dict(hvp=128,gradient=0,ce=0)and len(list((ev.run/'events').glob('*-input.json')))==len(list((ev.run/'events').glob('*-return.json')))==128,'all128 actual inputs/returns exhausted')
 gate=gates(cases);need(gate==out['descriptive'],'separate qualified material and attenuation gates');save('input-binding.json',dict(producer=SPEC,parity=P.SPEC,parity_independent=AUTH,inventories=inv))
 return dict(passed=True,native_calls=0,retained_hvp_checked=128,cases=cases,descriptive=gate,maximum_derived_rounding_ratio=rounding,scope='Independent fsum, vector/scalar identities and all128 saved native action associations. Exact byte-equal P/R/parity and response baselines inherit prior independent audit. Shared native actions; consumed development directional mechanism, no changed training or demonstrated utility.')

def main():
 resource.setrlimit(resource.RLIMIT_AS,(2*1024**3,)*2);need(os.getpriority(os.PRIO_PROCESS,0)>=10 and all(os.environ.get(k)=='1'for k in THREADS),'CPU1nice10');need(not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT).strip(),'clean exact source');OUT.mkdir(parents=True,exist_ok=False);start=time.monotonic();used=0;result=dict(passed=False,native_calls=0)
 def tick():need(time.monotonic()-start<120 and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<1024**2,'120s1GiB audit cap')
 def stop(*_):raise TimeoutError('owned120s deadline')
 signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGALRM,stop);signal.setitimer(signal.ITIMER_REAL,120)
 def write(name,raw,closing=False):
  nonlocal used
  need(used+len(raw)<(8 if closing else 6)*1024**2,'8MiB2MiB closing reserve');p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True)
  with p.open('xb')as f:f.write(raw);f.flush();os.fsync(f.fileno())
  used+=len(raw)
 def save(name,value,closing=False):write(name,(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n').encode(),closing)
 try:
  commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();save('runtime.json',dict(source_commit=commit,pid=os.getpid(),numpy=np.__version__,threads=dict.fromkeys(THREADS,'1'),sources={n:entry(ROOT/n)for n in FILES}))
  for n in FILES:
   raw=(ROOT/n).read_bytes();need(raw==subprocess.check_output(['git','show',commit+':'+n],cwd=ROOT),'complete committed own source');write('source/'+n,raw)
  t=time.monotonic();u=np.linspace(.1,1,15776);reduce(u,u,2*u,3*u,4*u,5*u,u,u);seconds=time.monotonic()-t;forecast=16*seconds+4*2.661671+30;save('profile.json',dict(passed=forecast<120,seconds=seconds,forecast_seconds=forecast,target_arrays_unopened=True));need(forecast<120,'full-sized fsum/vector measured admission');result=audit(tick,save);tick()
 except BaseException as e:result.update(passed=False,error=type(e).__name__+': '+str(e),traceback=traceback.format_exc())
 finally:
  signal.setitimer(signal.ITIMER_REAL,0);result.update(seconds=time.monotonic()-start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss);save('result.json',result,True);save('manifest.json',dict(files={str(p.relative_to(OUT)):entry(p)for p in OUT.rglob('*')if p.is_file()}),True)
 print(json.dumps({k:result.get(k)for k in ('passed','seconds','error','descriptive')}));return 0 if result['passed']else 2
if __name__=='__main__':raise SystemExit(main())
