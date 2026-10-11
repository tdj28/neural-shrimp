#!/usr/bin/env python3
"""Direct native F6 flip from the actual negative leading mode of a saved daughter."""
import argparse,hashlib,importlib.util,json,os,resource,signal,subprocess,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('leak',ROOT/'tools/continue_clark_rnn_n12_daughter_leak.py');L=importlib.util.module_from_spec(spec);spec.loader.exec_module(L)
D,B,S,R,M=L.D,L.B,L.S,L.R,L.M
from neural_shrimp.clark_rnn_boundaries import third_directional
need=D.need
PIN='e8d538b14ad4da8f26d5177a0a10dfde96b28eabc614e3c621c4a506cb17fcf7'
EXECUTION='b99a5b57a434dfa187ce3526bc294f76710d3537'
EXTRA=L.EXTRA+['tools/qualify_clark_rnn_n12_daughter_flip.py','tools/launch_clark_rnn_n12_daughter_flip.py','tests/test_clark_rnn_n12_daughter_flip.py','docs/plans/clark-rnn-n12-daughter-flip.md']

def flip_coefficients(a,q,third):
    m=a['monodromy'];values,vectors=np.linalg.eig(m.T);idx=int(np.argmin(abs(values+1)));w=vectors[:,idx]
    need(abs(values[idx].imag)<1e-8 and max(abs(w.imag))<1e-8,'real native critical left vector');w=w.real
    need(abs(w@q)>1e-8,'nondegenerate native left-right overlap');w=w/(w@q)
    alpha,cubic,rhs,sol,bqq=B.flip_coefficients(m,a['second'],a['first'][:,13],q,w,third)
    residual=float(max(abs(w@m+w)));report=dict(parameter=alpha,cubic=cubic,left_residual=residual,passed=abs(alpha)>1e-5 and abs(cubic)>1e-5 and residual<1e-7)
    return report,w,rhs,sol,bqq


def verify_input():
    j,_,original=L.verify_input();parent=R.RAW/'n12-p6-daughter-leak-001';need(S.sha(parent/'manifest.json')==PIN,'exact fixed-leak outer')
    inv=json.loads((parent/'manifest.json').read_text())['files'];need(set(inv)=={str(p.relative_to(parent))for p in parent.rglob('*')if p.is_file()and p!=parent/'manifest.json'},'complete fixed-leak inventory')
    for name,item in inv.items():
        p=parent/name;need(not Path(name).is_absolute()and '..'not in Path(name).parts and not p.is_symlink()and p.stat().st_size==item['bytes']and S.sha(p)==item['sha256']and not p.stat().st_mode&0o222,'immutable fixed-leak hash')
    launch=json.loads((parent/'launch-end.json').read_text());runtime=json.loads((parent/'run/runtime-start.json').read_text());need(launch['source_commit']==EXECUTION and launch['exit_code']==0 and launch['owned_group_empty'],'exact completed fixed-leak execution')
    for name,value in runtime['source']['sha256'].items():need(hashlib.sha256(subprocess.check_output(['git','show',EXECUTION+':'+name],cwd=ROOT)).hexdigest()==value,'historical fixed-leak source')
    result=json.loads((parent/'run/result.json').read_text());need(result['status']=='bounded_daughter_unit_completed'and len(result['branches'])==2,'fixed leak complete')
    first,branch=result['branches'];need(first['status']=='fixed_leak_steps_completed'and len(first['points'])==33,'retained lower-gain trajectory')
    need(branch['source_tag']==B.TAGS[1]and branch['status']=='stopped_at_native_stability_crossing'and len(branch['points'])==2 and len(branch['associations'])==1,'fixed higher-gain branch')
    crossing=branch['candidate_boundary'];need(crossing['kind']=='other_native6_stability_crossing','retain original unclassified leading-sign change')
    need(all(v['passed']for v in branch['points'])and all(a['forward']['passed']and a['reverse_native_passed']and all(v['passed']for v in a['reverse'])for a in branch['associations']),'prior native points and all joins')
    need(branch['points'][0]['native_stability']=='attracting'and branch['points'][1]['native_stability']=='unstable','exact stability change')
    endpoints=[]
    for last in [crossing['previous'],crossing['current']]:
        matches=[p for p in(parent/'run').glob('*daughter_point.json')if json.loads(p.read_text())['tag']==last['tag']];need(len(matches)==1,'unique bracket endpoint');p=matches[0];meta=json.loads(p.read_text());need(all(meta[k]==v for k,v in last.items()),'exact bracket result')
        with np.load(p.with_name(meta['array_file']))as f:z=f['physical'].copy();cycles=[f['cycle_x'].copy(),f['cycle_y'].copy()];projection=f['projection'].copy()
        need(z[-1]==last['delta']and np.array_equal(z[:12],cycles[0][0])and np.array_equal(z[12:24],cycles[1][0]),'exact saved native state')
        endpoints.append(dict(meta=last,physical=z,cycles=cycles,projection=projection))
    need(np.array_equal(endpoints[0]['projection'],endpoints[1]['projection']),'same fixed amplitude projection')
    current=endpoints[1];matches=[p for p in(parent/'run').glob('*native_context.json')if json.loads(p.read_text())['tag']==current['meta']['tag']+'-x'];need(len(matches)==1,'exact current native eigenrecord');md=json.loads(matches[0].read_text())
    with np.load(matches[0].with_name(md['array_file']))as f:values=f['eigenvalues'];vectors=f['eigenvectors'];theta=f['theta']
    index=int(np.argmax(abs(values)));mode=current['meta']['spectra'][0]['leading_real_mode']
    need(np.array_equal(theta,current['physical'][:12])and abs(values[index].imag)<1e-8 and values[index].real<-1 and max(abs(vectors[:,index].imag))<1e-8,'actual leading native minus mode')
    need(mode is not None and mode['sign']==-1 and abs(mode['phase0_real']-values[index].real)<1e-10,'qualified adjacent-phase leading mode')
    q=vectors[:,index].real;q/=np.linalg.norm(q);meta=dict(gain=branch['gain'],source_tag=branch['source_tag'],endpoints=endpoints,right_vector=q)
    return j,[(meta,current['physical'],current['cycles'])],dict(parent_manifest=PIN,execution=EXECUTION,prior=original,selection='actual unstable negative leading native mode; original crossing remains unclassified',leading_index=index,leading_value=float(values[index].real))

class Work(L.Work):
    def __init__(self,out):super().__init__(out);self.native_aug_calls=0;self.third_calls=0;self.contexts={}
    def native(self,x,j,g,d,tag):
        value=super().native(x,j,g,d,tag);self.contexts[tag]=value;return value
    def aug(self,z,j,g,tag):
        self.check();need(self.native_aug_calls<32,'32 full-native augmented calls');need(self.augmented_calls+2<=2560,'1280 derivative units')
        self.native_aug_calls+=1;self.calls+=2;self.augmented_calls+=2;t=time.monotonic();h,b,a=M.augmented(z,j,g,6,-1);self.emit('native_flip_augmented',dict(a,h=h,dh=b,tag=tag));return h,b,a,time.monotonic()-t
    def third(self,x,q,j,g,d,tag):
        self.check();need(self.third_calls<16,'16 full-native third calls');need(self.augmented_calls+2<=2560,'1280 derivative units')
        self.third_calls+=1;self.calls+=2;self.augmented_calls+=2;t=time.monotonic();c,stages=third_directional(x,q,j,g,d,6)
        self.emit('native_flip_third',dict(tag=tag,theta=x,direction=q,gain=g,delta=d,third=c,stages=stages));return c,time.monotonic()-t
    def branch(self,meta,parent,oldcycles,j,tag):
        g=meta['gain'];q=meta['right_vector'];z=np.r_[parent[:12],parent[-1],q];anchor=z.copy();result={'source_tag':tag,'gain':g,'status':'unresolved'}
        for k in range(12):
            need(np.isfinite(z).all()and .6<z[12]<1 and abs(z[12]-anchor[12])<.01 and max(abs(z[:12]-anchor[:12]))<.1*max(1,max(abs(anchor[:12])))and z[13:]@anchor[13:]>0 and np.linalg.norm(z[13:]-anchor[13:])<.5,'native flip state/leak/orientation guard')
            h,b,a,_=self.aug(z,j,g,tag+'-native-root');need(np.isfinite(h).all()and np.isfinite(b).all(),'finite native augmented')
            if max(abs(h))<1e-13:break
            need(k<11,'native augmented iteration cap');v=np.linalg.solve(b,-h);self.emit('native_flip_solve',{'tag':tag,'matrix':b,'rhs':-h,'direction':v});z+=v
        x,delta,q=z[:12],float(z[12]),z[13:];projection=meta['endpoints'][0]['projection']
        _,_,p,_=self.pair(np.r_[x,parent[12:24],delta],j,g,meta['endpoints'][1]['meta']['amplitude'],projection,tag+'-partner-proposal')
        y=p['Gx'];amplitude=float(projection@(x-y)/2);left,right=sorted(ep['meta']['delta']for ep in meta['endpoints']);need(left<delta<right and amplitude>0,'native root inside exact leak bracket and signed pair orientation')
        physical=np.r_[x,y,delta];ph,pb,pa,_=self.pair(physical,j,g,amplitude,projection,tag+'-paired-root');row,cycles=self.qualify(physical,ph,j,g,amplitude,projection,tag+'-flip')
        critical=[]
        for partner in ['x','y']:
            for suffix in ['', '-adjacent']:
                context=self.contexts[tag+'-flip-'+partner+suffix];e=context['eigenvalues'];idx=int(np.argmax(abs(e)));rest=np.delete(e,idx)
                need(abs(e[idx]+1)<1e-7 and min(abs(rest+1))>1e-4 and max(abs(rest))<1-1e-5,'both partners/phase native simple-1 and complement')
                critical.append(dict(partner=partner,phase=suffix,critical_error=float(abs(e[idx]+1)),complement_radius=float(max(abs(rest)))))
        coefficients=[]
        qy=pa['first_x'][:,:12]@q;qy/=np.linalg.norm(qy);zy=np.r_[y,delta,qy];hy,by,ay,_=self.aug(zy,j,g,tag+'-partner-root');need(max(abs(hy))<1e-11,'literal partner augmented root')
        for partner,az,aa in [('x',z,a),('y',zy,ay)]:
            c,_=self.third(az[:12],az[13:],j,g,delta,tag+'-'+partner)
            report,w,rhs,sol,bqq=flip_coefficients(aa,az[13:],c);need(report['passed'],'native flip nondegeneracy')
            self.emit('native_flip_criticality',dict(report,tag=tag+'-'+partner,physical=az,left=w,bqq=bqq,parameter_derivative=aa['first'][:,13],third=c,solve_matrix=aa['monodromy']-np.eye(12),solve_rhs=rhs,solve_solution=sol));coefficients.append(dict(partner=partner,**report))
        need(coefficients[0]['parameter']*coefficients[1]['parameter']>0 and coefficients[0]['cubic']*coefficients[1]['cubic']>0,'same partner crossing and cubic signs')
        joins=[]
        for i,endpoint in enumerate(meta['endpoints']):
            forward=D.paired_join(endpoint['cycles'],cycles);need(forward['passed'],'both saved native bracket phases')
            olddelta=endpoint['meta']['delta'];guess=self.predict_delta(physical,pb,olddelta,tag+f'-reverse{i}');rz,rh,rb,ra,ra_amp=self.correct_delta(guess,physical,j,g,olddelta,projection,tag+f'-reverse{i}')
            returned=[self.native(rz[k:k+12],j,g,float(rz[-1]),tag+f'-reverse{i}-{k}')for k in [0,12]];reverse=[B.native_join(endpoint['cycles'][k],returned[k]['cycle'],endpoint['physical'][-1],rz[-1])for k in range(2)]
            need(all(v['passed']for v in returned)and all(v['passed']for v in reverse),'both bracket full-native reverse joins');join=dict(endpoint=i,forward=forward,reverse=reverse);joins.append(join);self.emit('flip_bracket_join',dict(join,bracket_physical=endpoint['physical'],flip_physical=physical,reverse_physical=rz,bracket_cycle_x=endpoint['cycles'][0],bracket_cycle_y=endpoint['cycles'][1],flip_cycle_x=cycles[0],flip_cycle_y=cycles[1]))
        result.update(status='numerically_qualified_native6_daughter_flip',point=row,critical_spectra=critical,coefficients=coefficients,bracket_joins=joins,leak_bracket=[left,right]);self.emit('native_daughter_flip_result',dict(result,physical=physical,cycle_x=cycles[0],cycle_y=cycles[1]));return result

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
                matrix,rhs=pair[1],-pair[0];v=np.linalg.solve(matrix,rhs);w.emit('profile_solve',{'matrix':matrix,'rhs':rhs,'solution':v})
            measure('eigen',eigen);measure('solve',solve)
            aug=measure('pair',lambda:w.aug(np.r_[x,.8,q],j,2.,'synthetic_profile'))
            def solve_aug():
                matrix,rhs=aug[1],-aug[0];v=np.linalg.solve(matrix,rhs);w.emit('profile_native_augmented_solve',{'matrix':matrix,'rhs':rhs,'solution':v})
            measure('solve',solve_aug)
            c=measure('pair',lambda:w.third(x,q,j,2.,.8,'synthetic_profile'))
            def solve_coefficients():
                report,left,rhs,sol,bqq=flip_coefficients(aug[2],q,c[0]);w.emit('profile_coefficients',dict(report,left=left,rhs=rhs,solution=sol,bqq=bqq))
            measure('solve',solve_coefficients)
        t,b=({k:max(v)for k,v in a.items()}for a in[times,sizes]);seconds=2*(1280*(t['pair']+t['solve'])+512*(t['native']+4*t['eigen']))+20
        raw=1280*(b['pair']+b['solve']+4096)+512*(2*b['native']+4*b['eigen']+4096)+16*1024**2;peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024+256*1024**2
        gate=all(np.isfinite(v)and v>0 for a in times.values()for v in a)and seconds<=180 and raw<=256*1024**2 and peak<=768*1024**2;S.save(out/'profile.json',{'times':times,'bytes':sizes,'projected_seconds':seconds,'projected_raw_bytes':raw,'projected_peak_bytes':peak,'gate':bool(gate),'target_unopened':True});need(gate,'measured paired resource admission')
        j,seeds,binding=verify_input();S.save(out/'input-binding.json',binding)
        for meta,z,cycle in seeds:
            row=w.branch(meta,z,cycle,j,meta['source_tag']);result['branches'].append(row);S.save(out/'progress.json',result)
        w.check();result['status']='bounded_daughter_unit_completed'
    except BaseException as e:result.update(status='failed_or_interrupted',error=type(e).__name__+': '+str(e))
    finally:
        result.update(elapsed_seconds=time.monotonic()-w.start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,native_calls=w.native_calls,half_and_native_augmented_units=w.augmented_calls,native_augmented_calls=w.native_aug_calls,third_calls=w.third_calls,total_calls=w.calls,scope='Numerical native primitive6 daughter flip with both native partners/phases, nonzero total multiplier derivative/cubic and exact leak bracket joins. No interval, native12/24, generalizedflip, whole shrimp or utility claim.');S.save(out/'result.json',result,final=True);S.manifest(out)
    return 0 if result['status']=='bounded_daughter_unit_completed'else 2
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);sys.exit(main(p.parse_args().out))
