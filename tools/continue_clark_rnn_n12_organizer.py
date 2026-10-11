#!/usr/bin/env python3
"""Fixed near-organizer fold approach and explicit literal native phase bridge."""
import argparse,hashlib,importlib.util,json,os,resource,signal,subprocess,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('joint_curves',ROOT/'tools/continue_clark_rnn_n12_daughter_fold.py');J=importlib.util.module_from_spec(spec);spec.loader.exec_module(J)
D,B,S,R,M=J.D,J.B,J.S,J.R,J.M
need=J.need
STEPS=16
END_GAIN=4.0756
EXTRA=J.EXTRA+['tools/continue_clark_rnn_n12_organizer.py','tools/launch_clark_rnn_n12_organizer.py','tests/test_clark_rnn_n12_organizer.py','docs/plans/clark-rnn-n12-organizer-connection.md','docs/reviews/clark-rnn-n12-fold-continuation-002.json','docs/reviews/clark-rnn-n12-flip-cubic-outcome.json']
PARENTS={
 'joint':('n12-p6-fold-continuation-002','3ed78bc034921adb9c469384114cb2f52784bd7b0d834477f9bb1fe268ad78eb','a5cc8471a9e0d28cb51e2f7d054f835403daaab7ab33049fc99277c8b3425f6f','ef913ec9e0aaa7ba93853089fa56b2c16d883be2'),
 'cubic':('n12-p6-flip-cubic-001','b00e52917b1fad0267c94ada164311fc7fb86d979302fa65d34d144cc12b1c36','1bb762e16d379a57c88708995564a8860ea7444f4ff0284579f0bdc218d61547','c7c0ff4ec4e6a7587705f8c1a07562ec1b46e396')}
comparison=J.comparison

def bound_parent(kind):
 name,outer,worker,execution=PARENTS[kind];parent=R.RAW/name
 for folder,pin in [(parent,outer),(parent/'run',worker)]:
  need(S.sha(folder/'manifest.json')==pin,'literal '+kind+' inventory pin');inv=json.loads((folder/'manifest.json').read_text())['files']
  need(set(inv)=={str(p.relative_to(folder))for p in folder.rglob('*')if p.is_file()and p!=folder/'manifest.json'},'complete '+kind+' inventory')
  for name,v in inv.items():
   p=folder/name;need(not Path(name).is_absolute()and '..'not in Path(name).parts and not p.is_symlink()and not p.stat().st_mode&0o222 and p.stat().st_size==v['bytes']and S.sha(p)==v['sha256'],'immutable '+kind+' bytes')
 launch=json.loads((parent/'launch-end.json').read_text());rt=json.loads((parent/'run/runtime-start.json').read_text())
 need(launch['source_commit']==rt['source']['commit']==execution and launch['exit_code']==0 and launch['owned_group_empty'],'completed '+kind+' owner')
 for n,pin in rt['source']['sha256'].items():need(hashlib.sha256(subprocess.check_output(['git','show',execution+':'+n],cwd=ROOT)).hexdigest()==pin,'historical '+kind+' source '+n)
 return parent,json.loads((parent/'run/result.json').read_text())

def record(parent,kind,tag):
 paths=[p for p in(parent/'run').glob('*'+kind+'.json')if json.loads(p.read_text()).get('tag')==tag];need(len(paths)==1,'unique fixed '+tag+' '+kind)
 p=paths[0];md=json.loads(p.read_text());a=p.with_name(md['array_file']);need(S.sha(a)==md['array_sha256'],'exact '+tag+' array')
 with np.load(a)as f:return md,{k:f[k].copy()for k in f.files}

def verify_input():
 j,_,_,_,projection,binding=J.verify_input();parent,old=bound_parent('joint')
 need(old['status']=='completed_fixed_joint_native_curves'and len(old['points'])==49 and len(old['associations'])==48,'complete prior joint curves')
 need(all(p['fold']['passed']and p['signed']['passed']for p in old['points'])and all(a['forward_fold']['passed']and a['forward_signed']['passed']and a['all_reverse_native_passed']and all(v['passed']for v in a['reverse'])and max(a['reverse_q_errors'])<=1e-7 for a in old['associations']),'prior joint qualification and joins')
 audit=json.loads((ROOT/'docs/reviews/clark-rnn-n12-fold-continuation-002.json').read_text())
 need(audit['independent_saved_audit']['passed']and audit['raw']['manifest_sha256']==PARENTS['joint'][1],'prior independent saved audit')
 point=old['points'][-1];need(point['gain']==4.08,'fixed joint endpoint')
 fm,fa=record(parent,'fold_curve_point','fold48');sm,sa=record(parent,'branch_point','signed48')
 need(all(fm[k]==v for k,v in point['fold'].items())and all(sm[k]==v for k,v in point['signed'].items()),'literal completed endpoint reports')
 need(np.array_equal(fa['projection'],projection),'unchanged original daughter projection')
 cubic,cr=bound_parent('cubic');need(cr['status']=='completed_numerical_cubic_bracket'and len(cr['midpoints'])==24 and len(cr['associations'])==24 and cr['gain_width']<=3e-9,'complete old local cubic bracket')
 need(all(a['forward']['passed']and a['reverse']['passed']and a['reverse_native_passed']and a['reverse_q_error']<=1e-7 for pair in cr['associations']for a in pair),'all original cubic bracket joins')
 cb=json.loads((cubic/'run/input-binding.json').read_text());need(cb['matrix_input_sha256']==binding['matrix_input_sha256']and cb['parent_endpoint_failure_retained'],'same map and retained failed global phase join')
 seeds=[]
 for row in cr['bracket']:
  md,a=record(cubic,'branch_point',row['tag']);need(all(md[k]==v for k,v in row.items())and md['passed']and all(md['gates'].values()),'exact qualified local bracket row');seeds.append((md,a['physical'],a['cycle']))
 need(seeds[0][0]['cubic']>0>seeds[1][0]['cubic']and seeds[0][0]['gain']<seeds[1][0]['gain']<END_GAIN,'strict inherited bracket below fold endpoint')
 return j,fa['physical'],[fa['cycle_x'],fa['cycle_y']],(sm,sa['physical'],sa['cycle']),projection,seeds,dict(prior=binding,parents=PARENTS,joint_saved_audit=S.sha(ROOT/'docs/reviews/clark-rnn-n12-fold-continuation-002.json'),literal_native_phase_transport=True,old_failed_phase_zero_join_retained=True)

def phase_transport(cycle,jacobians,q,k):
 need(cycle.shape==(6,12)and jacobians.shape==(6,12,12)and q.shape==(12,)and type(k)is int and 0<=k<6,'complete native phase transport shapes')
 need(all(np.isfinite(v).all()for v in [cycle,jacobians,q]),'finite native phase transport')
 product=np.eye(12)
 for a in jacobians[:k]:product=a@product
 raw=product@q;length=float(np.linalg.norm(raw));need(np.isfinite(length)and length>1e-8,'nonzero literal transported mode')
 return cycle[k].copy(),raw/length,product,length

def mode_join(q,other):
 # A normalized real eigenvector has two orientations; states are never signed.
 dot=float(q@other);sign=1 if dot>=0 else -1;error=float(np.linalg.norm(q-sign*other))
 return dict(orientation=sign,dot=dot,error=error,passed=bool(np.isfinite(error)and error<=1e-7))

class Work(J.Work):
 def __init__(self,out):super().__init__(out);self.native_returns={}
 def it(self,x,j,g,d,tag):
  ret=super().it(x,j,g,d,tag);self.native_returns[tag]=(ret[2],ret[3]);return ret

def run_curves(w,j,z,oldcycles,signed,projection,result):
    g0=signed[0]['gain'];gains=np.linspace(g0,END_GAIN,STEPS+1);need(g0>END_GAIN,'fixed decreasing gains');fz,fh,fb,fa=w.correct(z,z,j,g0,'fold00',True)
    fr,fp,fc=w.fold_point(fz,fh,fa,j,g0,projection,'fold00');sz,sh,sb,sa=w.correct(signed[1],signed[1],j,g0,'signed00',False);sr,sn=w.signed_point(sz,sh,sb,sa,j,g0,'signed00')
    joins=[B.native_join(oldcycles[k],fc[k],z[12],fz[12])for k in range(2)]+[B.native_join(signed[2],sn['cycle'],signed[1][12],sz[12])]
    need(all(v['passed']for v in joins)and np.linalg.norm(fz[13:]-z[13:])<1e-7 and np.linalg.norm(sz[13:]-signed[1][13:])<1e-7,'exact two saved critical seeds/modes')
    result.update(gains=gains.tolist(),start_joins=joins,points=[dict(index=0,gain=g0,fold=fr,signed=sr,comparison=comparison(fp,sz,projection))],associations=[]);S.save(w.out/'progress.json',result)
    for i,g in enumerate(gains[1:],1):
        oldg=gains[i-1];nfz,nfh,nfb,nfa=w.correct(w.predict_gain(fz,fb,fa,g-oldg,f'fold{i}-predict'),fz,j,g,f'fold{i:02d}',True)
        nfr,nfp,nfc=w.fold_point(nfz,nfh,nfa,j,g,projection,f'fold{i:02d}')
        nsz,nsh,nsb,nsa=w.correct(w.predict_gain(sz,sb,sa,g-oldg,f'signed{i}-predict'),sz,j,g,f'signed{i:02d}',False);nsr,nsn=w.signed_point(nsz,nsh,nsb,nsa,j,g,f'signed{i:02d}')
        forward_fold=D.paired_join(fc,nfc);forward_signed=M.phase_match(sn['cycle'],nsn['cycle']);point=dict(index=i,gain=float(g),fold=nfr,signed=nsr,comparison=comparison(nfp,nsz,projection));result['pending']=dict(point=point,forward_fold=forward_fold,forward_signed=forward_signed);S.save(w.out/'progress.json',result)
        need(forward_fold['passed']and forward_signed['passed'],'all unchanged forward native phases')
        rfz,rfh,rfb,rfa=w.correct(w.predict_gain(nfz,nfb,nfa,oldg-g,f'fold{i}-reverse-predict'),nfz,j,oldg,f'fold{i}-reverse',True);rfn=w.reverse_fold(rfz,rfa,j,oldg,f'fold{i}-reverse')
        rsz,rsh,rsb,rsa=w.correct(w.predict_gain(nsz,nsb,nsa,oldg-g,f'signed{i}-reverse-predict'),nsz,j,oldg,f'signed{i}-reverse',False);rsn=w.native(rsz[:12],j,oldg,float(rsz[12]),f'signed{i}-reverse-native')
        reverse=[B.native_join(fc[k],rfn[k]['cycle'],fz[12],rfz[12])for k in range(2)]+[B.native_join(sn['cycle'],rsn['cycle'],sz[12],rsz[12])];qe=[float(np.linalg.norm(rfz[13:]-fz[13:])),float(np.linalg.norm(rsz[13:]-sz[13:]))]
        assoc=dict(index=i,forward_fold=forward_fold,forward_signed=forward_signed,reverse=reverse,reverse_q_errors=qe,all_reverse_native_passed=all(v['passed']for v in rfn+[rsn]))
        w.emit('joint_curve_association',dict(assoc,previous_fold=fz,current_fold=nfz,reverse_fold=rfz,previous_signed=sz,current_signed=nsz,reverse_signed=rsz,previous_cycle_x=fc[0],previous_cycle_y=fc[1],previous_signed_cycle=sn['cycle'],current_cycle_x=nfc[0],current_cycle_y=nfc[1],current_signed_cycle=nsn['cycle'],reverse_cycle_x=rfn[0]['cycle'],reverse_cycle_y=rfn[1]['cycle'],reverse_signed_cycle=rsn['cycle']))
        need(assoc['all_reverse_native_passed']and all(v['passed']for v in reverse)and max(qe)<=1e-7,'three full native cycle/leak and both mode reverse joins')
        result['points'].append(point);result['associations'].append(assoc);result.pop('pending');fz,fb,fa,fc=nfz,nfb,nfa,nfc;sz,sb,sa,sn=nsz,nsb,nsa,nsn;S.save(w.out/'progress.json',result)
    need(len(result['points'])==17 and len(result['associations'])==16,'fixed complete joint unit')
    return sz,sb,sa,sn

def signed_step(w,previous,g,j,tag,result):
 z,b,a,native,oldg=previous
 nz,h,nb,na=w.correct(w.predict_gain(z,b,a,g-oldg,tag+'-predict'),z,j,g,tag,False)
 row,nn=w.signed_point(nz,h,nb,na,j,g,tag);forward=M.phase_match(native['cycle'],nn['cycle'])
 pending=dict(point=row,forward=forward);result['pending_signed']=pending;S.save(w.out/'progress.json',result);need(forward['passed'],'continued same native signed phase')
 rz,rh,rb,ra=w.correct(w.predict_gain(nz,nb,na,oldg-g,tag+'-reverse-predict'),nz,j,oldg,tag+'-reverse',False)
 rn=w.native(rz[:12],j,oldg,float(rz[12]),tag+'-reverse-native');reverse=B.native_join(native['cycle'],rn['cycle'],z[12],rz[12]);qe=float(np.linalg.norm(z[13:]-rz[13:]))
 association=dict(forward=forward,reverse=reverse,reverse_native_passed=rn['passed'],reverse_q_error=qe)
 w.emit('organizer_signed_association',dict(association,tag=tag,previous_physical=z,current_physical=nz,reverse_physical=rz,previous_cycle=native['cycle'],current_cycle=nn['cycle'],reverse_cycle=rn['cycle']))
 need(rn['passed']and reverse['passed']and qe<=1e-7,'same native signed reverse state/leak/mode')
 result['signed_extension'].append(dict(row,association=association));result.pop('pending_signed');S.save(w.out/'progress.json',result)
 return (nz,nb,na,nn,g)

def bridge_endpoint(w,j,continued,seed,tag,result):
 target,b,a,tn,g=continued;md,z,cycle=seed;need(g==md['gain'],'identical actual bracket gain')
 original,h,ob,oa=w.correct(z,z,j,g,tag+'-original',False);report,on=w.signed_point(original,h,ob,oa,j,g,tag+'-original')
 oldjoin=B.native_join(cycle,on['cycle'],z[12],original[12]);need(oldjoin['passed']and np.linalg.norm(original[13:]-z[13:])<=1e-7,'recorrected exact old cubic root and mode')
 need(report['cubic']*md['cubic']>0,'strict original cubic sign survives fresh qualification')
 fresh_cycle,phases=w.native_returns[tag+'-original-native'];rows=[];matches=[]
 for k in range(6):
  x,q,transport,length=phase_transport(fresh_cycle,phases,original[13:],k);pz=np.r_[x,original[12],q]
  ph,pb,pa,_=w.half(pz,j,g,-1,tag+f'-phase{k}');need(max(abs(ph))<1e-11,'literal uncorrected phase augmented root')
  pr,pn=w.signed_point(pz,ph,pb,pa,j,g,tag+f'-phase{k}')
  cyclic=B.native_join(np.roll(fresh_cycle,-k,axis=0),pn['cycle'],original[12],pz[12]);join=B.native_join(tn['cycle'],pn['cycle'],target[12],pz[12]);mj=mode_join(target[13:],q)
  reverse=np.eye(12)
  for d in phases[k:]:reverse=d@reverse
  back=reverse@q;need(np.isfinite(back).all()and np.linalg.norm(back)>1e-8,'nonzero reverse transported mode');back/=np.linalg.norm(back);reverse_mode=mode_join(original[13:],back)
  row=dict(phase=k,literal_cyclic_identity=cyclic,continued_native_join=join,continued_mode_join=mj,reverse_mode=reverse_mode,qualification=pr,original_cubic=report['cubic'])
  w.emit('literal_phase_bridge',dict(row,tag=tag,original_physical=original,original_cycle=fresh_cycle,phase_jacobians=phases,physical=pz,cycle=pn['cycle'],continued_physical=target,continued_cycle=tn['cycle'],transport=transport,transport_norm=length,reverse_transport=reverse,reverse_q=back))
  rows.append(row);result['pending_bridge']=dict(tag=tag,gain=g,rows=rows);S.save(w.out/'progress.json',result)
  need(cyclic['passed']and reverse_mode['passed']and pr['cubic']*report['cubic']>0,'every literal phase primitive identity, reverse mode and strict cubic sign')
  if join['passed']:
   need(mj['passed'],'strict actual phase joins also preserve critical mode');matches.append(k)
 need(len(matches)==1,'exactly one actual native phase joins continued root')
 result['phase_bridges'].append(dict(tag=tag,gain=g,old_saved_join=oldjoin,unique_matching_phase=matches[0],rows=rows));result.pop('pending_bridge');S.save(w.out/'progress.json',result)

def run(w,j,z,cycles,signed,projection,seeds,result):
 sz,sb,sa,sn=run_curves(w,j,z,cycles,signed,projection,result)
 result.update(signed_extension=[],phase_bridges=[]);current=(sz,sb,sa,sn,END_GAIN)
 for label,seed in [('right',seeds[1]),('left',seeds[0])]:
  current=signed_step(w,current,seed[0]['gain'],j,'organizer-'+label,result)
  bridge_endpoint(w,j,current,seed,'old-cubic-'+label,result)
 need(result['signed_extension'][0]['cubic']<0<result['signed_extension'][1]['cubic'],'continued native root has strict same bracket signs')
 need(len(result['phase_bridges'])==2,'both original local bracket endpoints explicitly joined')

def main(out):
    identity=S.source();identity['sha256'].update({p:S.sha(ROOT/p)for p in EXTRA});need(os.getpriority(os.PRIO_PROCESS,0)>=10 and all(os.environ.get(k)=='1'for k in R.M.THREADS),'CPU1 nice10')
    resource.setrlimit(resource.RLIMIT_AS,(2*1024**3,)*2);out.mkdir(parents=True,exist_ok=False);S.ACTIVE_ROOT,S.RAW_CAP=out,256*1024**2;w,booked=Work(out),[0];signal.signal(signal.SIGTERM,lambda *_:setattr(w,'interrupted',True))
    def reserve(path,size,final=False):
        old=Path(path).stat().st_size if Path(path).exists()else 0;new=booked[0]-old+size
        if new>(S.RAW_CAP if final else S.RAW_CAP-S.FINAL_RESERVE):raise OSError('prewrite raw cap')
        booked[0]=new
    S.reserve=reserve;S.save(out/'runtime-start.json',dict(source=identity,pid=os.getpid(),nice=os.getpriority(os.PRIO_PROCESS,0),threads={k:os.environ[k]for k in R.M.THREADS},started_unix=time.time(),python=sys.version,numpy=np.__version__));result={'status':'incomplete','points':[],'associations':[]}
    try:
        need(int(next(x.split()[1]for x in Path('/proc/meminfo').read_text().splitlines()if x.startswith('MemAvailable:')))*1024>=2*1024**3,'2GiB current shared headroom');J.profile(w,booked,out)
        j,z,cycles,signed,projection,seeds,binding=verify_input();S.save(out/'input-binding.json',dict(binding,steps=STEPS,end_gain=END_GAIN));run(w,j,z,cycles,signed,projection,seeds,result);w.check();result['status']='completed_native_organizer_phase_connection'
    except BaseException as e:result.update(status='failed_or_interrupted',error=type(e).__name__+': '+str(e))
    finally:
        result.update(elapsed_seconds=time.monotonic()-w.start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,native_calls=w.native_calls,derivative_half_units=w.augmented_calls,native_augmented_calls=w.native_aug_calls,third_calls=w.third_calls,linear_solve_calls=w.solve_calls,total_calls=w.calls,scope='Numerical native6 fold approach and explicit literal-phase identity of its parent signed branch with both old cubic-bracket endpoints. Original phase0 failure retained. No exact cubic-zero root, quintic nondegeneracy, interval global connection or whole-shrimp proof.');S.save(out/'result.json',result,final=True);S.manifest(out)
    return 0 if result['status']=='completed_native_organizer_phase_connection'else 2
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);sys.exit(main(p.parse_args().out))
