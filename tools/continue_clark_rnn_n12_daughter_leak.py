#!/usr/bin/env python3
"""Fixed-leak continuation of the two exact attracting native6 daughter states."""
import argparse,hashlib,importlib.util,json,os,resource,signal,subprocess,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('extension',ROOT/'tools/extend_clark_rnn_n12_daughters.py');E=importlib.util.module_from_spec(spec);spec.loader.exec_module(E)
D,B,S,R,M=E.D,E.B,E.S,E.R,E.M
need=D.need
PIN='acce75e78d6fd3b10737dabc710ef56a223d8ea8036641fc09e14a098067030c'
EXECUTION='f663fe128f56699c7d5ad250c6dd85815c582983'
STEP=-.0002;STEPS=32
EXTRA=E.EXTRA+['tools/continue_clark_rnn_n12_daughter_leak.py','tools/launch_clark_rnn_n12_daughter_leak.py','tests/test_clark_rnn_n12_daughter_leak.py','docs/plans/clark-rnn-n12-daughter-leak.md']

def delta_tangent(b):
    need(b.shape==(25,25),'full paired Jacobian');v=np.linalg.solve(b[:24,:24],-b[:24,-1]);need(np.isfinite(v).all(),'finite actual-leak tangent');return v

def verify_input():
    j,_,original=E.verify_input();parent=R.RAW/'n12-p6-daughter-extension-001';need(S.sha(parent/'manifest.json')==PIN,'exact extension outer')
    inv=json.loads((parent/'manifest.json').read_text())['files'];need(set(inv)=={str(p.relative_to(parent))for p in parent.rglob('*')if p.is_file()and p!=parent/'manifest.json'},'complete extension inventory')
    for name,item in inv.items():
        p=parent/name;need(not Path(name).is_absolute()and '..'not in Path(name).parts and not p.is_symlink()and p.stat().st_size==item['bytes']and S.sha(p)==item['sha256']and not p.stat().st_mode&0o222,'immutable extension hash')
    launch=json.loads((parent/'launch-end.json').read_text());rt=json.loads((parent/'run/runtime-start.json').read_text());need(launch['source_commit']==EXECUTION and launch['exit_code']==0 and launch['owned_group_empty'],'exact extension execution')
    for name,value in rt['source']['sha256'].items():need(hashlib.sha256(subprocess.check_output(['git','show',EXECUTION+':'+name],cwd=ROOT)).hexdigest()==value,'historical source')
    result=json.loads((parent/'run/result.json').read_text());need(result['status']=='bounded_daughter_unit_completed'and len(result['branches'])==2,'fixed extension result');seeds=[]
    for i,(tag,branch)in enumerate(zip(B.TAGS,result['branches'])):
        need(branch['source_tag']==tag and len(branch['points'])==[5,9][i]and len(branch['associations'])==[4,8][i],'exact branch length and source')
        if i==0:need(branch['status']=='unresolved_retained'and branch['error']=='ValueError: paired native state/leak guard','retain amplitude guard failure')
        else:need(branch['status']=='stopped_at_native_stability_crossing'and branch['candidate_boundary']['kind']=='actual_native6_plus_bracket','retain native plus crossing')
        need(all(p['passed']for p in branch['points'])and all(v['passed']for v in branch['saved_endpoint_joins'])and all(a['forward']['passed']and a['reverse_native_passed']and all(v['passed']for v in a['reverse'])for a in branch['associations']),'all prior native joins')
        last=branch['points'][-1];need(last['native_stability']=='attracting','fixed attracting endpoints')
        paths=[p for p in(parent/'run').glob('*daughter_point.json')if json.loads(p.read_text())['tag']==last['tag']];need(len(paths)==1,'unique saved endpoint');p=paths[0];meta=json.loads(p.read_text());need(all(meta[k]==v for k,v in last.items()),'exact result association')
        with np.load(p.with_name(meta['array_file']))as f:z=f['physical'].copy();cycles=[f['cycle_x'].copy(),f['cycle_y'].copy()];q=f['projection'].copy()
        need(z[-1]==last['delta']and np.array_equal(z[:12],cycles[0][0])and np.array_equal(z[12:24],cycles[1][0]),'exact native partner state')
        meta['source_tag']=tag;meta['projection']=q;seeds.append((meta,z,cycles))
    return j,seeds,dict(parent_manifest=PIN,execution=EXECUTION,prior=original,delta_step=STEP,steps=STEPS,targets=B.TAGS,coordinate_change='fixed actual leak, retain dynamic signed projection amplitude and all original association guards')

class Work(D.Work):
    def correct_delta(self,guess,anchor,j,g,delta,q,tag):
        z=guess.copy();z[-1]=delta;scale=max(1.,float(max(abs(anchor[:24]))))
        for k in range(12):
            need(np.isfinite(z).all()and .6<delta<1 and abs(delta-anchor[-1])<=.02 and max(abs(z[:24]-anchor[:24]))<=.1*scale,'paired native state/leak guard')
            amplitude=float(q@(z[:12]-z[12:24])/2);need(amplitude>0,'same signed pair orientation');h,b,a,_=self.pair(z,j,g,amplitude,q,tag);need(np.isfinite(h).all()and np.isfinite(b).all(),'finite paired equations')
            if max(abs(h))<1e-13:return z,h,b,a,amplitude
            need(k<11,'fixed-leak correction iteration cap');v=np.linalg.solve(b[:24,:24],-h[:24]);self.emit('fixed_leak_solve',{'tag':tag,'matrix':b[:24,:24],'rhs':-h[:24],'direction':v,'delta':delta});z[:24]+=v
        raise AssertionError('unreachable')
    def predict_delta(self,z,b,delta,tag):
        v=delta_tangent(b);self.emit('leak_predictor',{'tag':tag,'matrix':b[:24,:24],'rhs':-b[:24,-1],'tangent':v,'physical':z,'delta_step':delta-z[-1]});return np.r_[z[:24]+(delta-z[-1])*v,delta]
    def branch(self,meta,parent,oldcycles,j,tag):
        g=meta['gain'];q=meta['projection'];result={'source_tag':tag,'gain':g,'points':[],'associations':[],'status':'incomplete'}
        try:
            z,h,b,a,amp=self.correct_delta(parent,parent,j,g,parent[-1],q,tag+'-leak-anchor');row,cycles=self.qualify(z,h,j,g,amp,q,tag+'-leak-anchor');joins=[B.native_join(x,y,parent[-1],z[-1])for x,y in zip(oldcycles,cycles)];need(all(v['passed']for v in joins)and row['native_stability']==meta['native_stability'],'saved native endpoint requalification');result['saved_endpoint_joins']=joins;result['points'].append(row)
            for i in range(1,STEPS+1):
                delta=float(parent[-1]+i*STEP);guess=self.predict_delta(z,b,delta,tag+f'-leak-step{i}');nz,nh,nb,na,namp=self.correct_delta(guess,z,j,g,delta,q,tag+f'-leak-point{i}');new,cy=self.qualify(nz,nh,j,g,namp,q,tag+f'-leak-point{i}')
                forward=D.paired_join(cycles,cy);result['pending']={'point':new,'forward':forward};S.save(self.out/(tag+'-progress.json'),result);need(forward['passed'],'untransformed paired forward phases')
                rg=self.predict_delta(nz,nb,float(z[-1]),tag+f'-leak-reverse{i}');rz,rh,rb,ra,ramp=self.correct_delta(rg,nz,j,g,float(z[-1]),q,tag+f'-leak-reverse{i}')
                returned=[self.native(rz[k:k+12],j,g,float(rz[-1]),tag+f'-leak-reverse{i}-{k}')for k in [0,12]];reverse=[B.native_join(cycles[k],returned[k]['cycle'],z[-1],rz[-1])for k in range(2)]
                assoc={'index':i,'forward':forward,'reverse':reverse,'reverse_native_passed':all(v['passed']for v in returned)};self.emit('daughter_association',dict(assoc,previous=z,current=nz,reverse_state=rz,previous_cycle_x=cycles[0],previous_cycle_y=cycles[1],current_cycle_x=cy[0],current_cycle_y=cy[1]));need(assoc['reverse_native_passed']and all(v['passed']for v in reverse),'paired reverse full native join')
                previous=result['points'][-1];result['points'].append(new);result['associations'].append(assoc);result.pop('pending');z,b,cycles=nz,nb,cy;boundary=E.crossing(previous,new)
                if boundary is not None:result['candidate_boundary']=boundary;result['status']='stopped_at_native_stability_crossing';break
                S.save(self.out/(tag+'-progress.json'),result)
            else:result['status']='fixed_leak_steps_completed'
        except(np.linalg.LinAlgError,ValueError,FloatingPointError)as e:result.update(status='unresolved_retained',error=type(e).__name__+': '+str(e))
        S.save(self.out/(tag+'-progress.json'),result);return result

def main(out):
    identity=S.source();identity['sha256'].update({p:S.sha(ROOT/p)for p in EXTRA});need(os.getpriority(os.PRIO_PROCESS,0)>=10 and all(os.environ.get(k)=='1'for k in R.M.THREADS),'CPU1 nice10')
    resource.setrlimit(resource.RLIMIT_AS,(2*1024**3,)*2);out.mkdir(parents=True,exist_ok=False);S.ACTIVE_ROOT,S.RAW_CAP=out,256*1024**2;w,booked=Work(out),[0];signal.signal(signal.SIGTERM,lambda *_:setattr(w,'interrupted',True))
    def reserve(path,size,final=False):
        old=Path(path).stat().st_size if Path(path).exists()else 0;new=booked[0]-old+size
        if new>(S.RAW_CAP if final else S.RAW_CAP-S.FINAL_RESERVE):raise OSError('prewrite raw cap')
        booked[0]=new
    S.reserve=reserve;S.save(out/'runtime-start.json',{'source':identity,'pid':os.getpid(),'nice':os.getpriority(os.PRIO_PROCESS,0),'threads':{k:os.environ[k]for k in R.M.THREADS},'started_unix':time.time(),'python':sys.version,'numpy':np.__version__});result={'status':'incomplete','branches':[]}
    try:
        need(int(next(x.split()[1]for x in Path('/proc/meminfo').read_text().splitlines()if x.startswith('MemAvailable:')))*1024>=2*1024**3,'2GiB shared headroom')
        j=np.eye(12)*.2+np.ones((12,12))*.025;x=np.arange(1,13)/20;q=np.eye(12)[0];times={k:[]for k in ['native','pair','eigen','solve']};sizes={k:[]for k in times}
        def measure(kind,f):
            before,t=booked[0],time.monotonic();v=f();times[kind].append(time.monotonic()-t);sizes[kind].append(booked[0]-before);return v
        for k in range(10):
            native=measure('native',lambda:w.it(x,j,2.,.8,'synthetic_profile'));pair=measure('pair',lambda:w.pair(np.r_[x,-x,.8],j,2.,.02,q,'synthetic_profile'))
            def eigen():
                e,v=np.linalg.eig(native[1]);w.emit('profile_eigen',{'matrix':native[1],'eigenvalues':e,'eigenvectors':v})
            def solve():
                matrix,rhs=pair[1][:24,:24],-pair[0][:24];v=np.linalg.solve(matrix,rhs);w.emit('profile_solve',{'matrix':matrix,'rhs':rhs,'solution':v})
            measure('eigen',eigen);measure('solve',solve)
        t,b=({k:max(v)for k,v in a.items()}for a in[times,sizes]);seconds=2*(1280*(t['pair']+t['solve'])+512*(t['native']+4*t['eigen']))+20
        raw=1280*(b['pair']+b['solve']+4096)+512*(2*b['native']+4*b['eigen']+4096)+16*1024**2;peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024+256*1024**2
        gate=all(np.isfinite(v)and v>0 for a in times.values()for v in a)and seconds<=180 and raw<=256*1024**2 and peak<=768*1024**2;S.save(out/'profile.json',{'times':times,'bytes':sizes,'projected_seconds':seconds,'projected_raw_bytes':raw,'projected_peak_bytes':peak,'gate':bool(gate),'target_unopened':True});need(gate,'measured paired resource admission')
        j,seeds,binding=verify_input();S.save(out/'input-binding.json',binding)
        for meta,z,cycle in seeds:
            row=w.branch(meta,z,cycle,j,meta['source_tag']);result['branches'].append(row);S.save(out/'progress.json',result)
        w.check();result['status']='bounded_daughter_unit_completed'
    except BaseException as e:result.update(status='failed_or_interrupted',error=type(e).__name__+': '+str(e))
    finally:
        result.update(elapsed_seconds=time.monotonic()-w.start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,native_calls=w.native_calls,half_calls=w.augmented_calls,total_calls=w.calls,scope='Numerical Gperiod2 daughters on full native primitive6 orbits only. Native stability crossings are proposals, not qualified boundaries or native12/24 joins. No interval, generalizedflip, shrimp or utility claim.');S.save(out/'result.json',result,final=True);S.manifest(out)
    return 0 if result['status']=='bounded_daughter_unit_completed'else 2
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);sys.exit(main(p.parse_args().out))
