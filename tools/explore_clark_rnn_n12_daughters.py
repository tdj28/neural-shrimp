#!/usr/bin/env python3
"""Two fixed signed-half period-two daughters, retaining all native F6 states."""
import argparse,importlib.util,json,os,resource,signal,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('flip_branch',ROOT/'tools/explore_clark_rnn_n12_flip_branch.py');B=importlib.util.module_from_spec(spec);spec.loader.exec_module(B)
S,R,M=B.S,B.R,B.M
from neural_shrimp.clark_rnn_symmetry import pair_fixed_amplitude
EXTRA=B.EXTRA+['tools/explore_clark_rnn_n12_daughters.py','tools/launch_clark_rnn_n12_daughters.py','tests/test_clark_rnn_n12_daughters.py','docs/plans/clark-rnn-n12-daughters.md']
AMPLITUDES=[.001]+[.002*1.3**k for k in range(24)]
need=B.need

def paired_join(old,new):
    result=[M.phase_match(a,b)for a,b in zip(old,new)]
    return {'partners':result,'passed':all(v['passed']for v in result)}

def birth_check(z,parent,q,a,d):
    scale=max(1.,float(max(abs(parent))))
    midpoint=float(max(abs((z[:12]+z[12:24])/2-parent))/scale)
    mode=float(np.linalg.norm((z[:12]-z[12:24])/(2*a)-q))
    delta=abs(float(z[-1]-d))
    return {'midpoint_error':midpoint,'mode_error':mode,'leak_error':delta,'passed':midpoint<=20*a*a and mode<=.05 and delta<=20*a*a}

class Work(B.Work):
    def check(self):
        if self.interrupted:raise InterruptedError('SIGTERM')
        if time.monotonic()-self.start>180:raise TimeoutError('180s ceiling')
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss>768*1024:raise MemoryError('768MiB RSS')
        if self.calls>3072:raise RuntimeError('3072 half/native cap')
    def it(self,x,j,g,d,tag):
        self.check()
        if self.native_calls>=512:raise RuntimeError('512 native cap')
        # Call the original native implementation, avoiding the smaller branch-unit cap.
        return M.Worker.it(self,x,j,g,d,tag)
    def pair(self,z,j,g,amp,q,tag):
        self.check()
        if self.augmented_calls+2>2560:raise RuntimeError('1280 paired/2560half cap')
        self.calls+=2;self.augmented_calls+=2;t=time.monotonic();h,b,a=pair_fixed_amplitude(z,j,g,amp,q,3)
        self.emit('paired_half',dict(a,h=h,dh=b,tag=tag));return h,b,a,time.monotonic()-t
    def correct_pair(self,guess,anchor,j,g,amp,q,tag):
        z=guess.copy();scale=max(1.,float(max(abs(anchor[:24]))))
        for k in range(12):
            need(np.isfinite(z).all()and .6<z[-1]<1 and abs(z[-1]-anchor[-1])<=.02 and max(abs(z[:24]-anchor[:24]))<=.1*scale,'paired native state/leak guard')
            h,b,a,_=self.pair(z,j,g,amp,q,tag);need(np.isfinite(h).all()and np.isfinite(b).all(),'finite pair equations')
            if max(abs(h))<1e-13:return z,h,b,a
            need(k<11,'paired correction iteration limit');direction=np.linalg.solve(b,-h)
            self.emit('paired_solve',{'tag':tag,'iteration':k,'matrix':b,'rhs':-h,'direction':direction});z+=direction
        raise AssertionError('unreachable')
    def predict(self,z,b,da,tag):
        rhs=np.zeros(25);rhs[-1]=2;direction=np.linalg.solve(b,rhs)
        need(np.isfinite(direction).all(),'finite amplitude tangent')
        self.emit('amplitude_predictor',{'tag':tag,'matrix':b,'rhs':rhs,'tangent':direction,'amplitude_step':da,'physical':z})
        return z+da*direction
    def qualify(self,z,h,j,g,amp,q,tag):
        all_native=[];cycles=[];spectra=[]
        for partner,x in [('x',z[:12]),('y',z[12:24])]:
            root=self.native(x,j,g,float(z[-1]),tag+'-'+partner)
            adjacent=self.native(root['cycle'][1],j,g,float(z[-1]),tag+'-'+partner+'-adjacent')
            all_native.extend([root,adjacent]);cycles.append(root['cycle'])
            context={'eigenvalues':root['eigenvalues'],'eigenvalues_adjacent':adjacent['eigenvalues']};mode=M.leading_boundary_mode(context)
            spectra.append({'partner':partner,'phase0_radius':root['radius'],'phase1_radius':adjacent['radius'],'leading_real_mode':mode})
        scale=max(1.,float(np.max(abs(cycles))));cross=float(np.max(abs(cycles[0]+np.roll(cycles[1],3,axis=0)))/scale)
        separation=float(max(abs(z[:12]-z[12:24]))/scale);radius=max(v['radius']for v in all_native)
        gates={'four_native_primitive6':all(v['passed']for v in all_native),'paired_return':bool(max(abs(h))<1e-13),'G_primitive2':separation>1e-6,'literal_native_partner_identity':cross<1e-8}
        row={'tag':tag,'gain':g,'amplitude':amp,'delta':float(z[-1]),'gates':gates,'passed':all(gates.values()),'residual':float(max(abs(h))),'G_pair_separation':separation,'cross_identity_error':cross,'spectra':spectra,'maximum_native_radius':radius,'native_stability':'attracting'if radius<1-1e-6 else'unstable'if radius>1+1e-6 else'near_neutral'}
        self.emit('daughter_point',dict(row,physical=z,cycle_x=cycles[0],cycle_y=cycles[1],projection=q))
        need(row['passed'],'paired/native6 qualification');return row,cycles
    def branch(self,meta,parent,cycle,j,tag):
        g,d=meta['gain'],float(parent[12]);q=parent[13:].copy();amp=AMPLITUDES[0];initial=np.r_[parent[:12]+amp*q,parent[:12]-amp*q,d]
        result={'source_tag':tag,'gain':g,'points':[],'associations':[],'status':'incomplete'}
        try:
            originals=[self.native(parent[:12],j,g,d,tag+'-original')]
            need(originals[0]['passed'],'original native root')
            z,h,b,a=self.correct_pair(initial,initial,j,g,amp,q,tag+'-anchor');row,cycles=self.qualify(z,h,j,g,amp,q,tag+'-anchor')
            birth=birth_check(z,parent[:12],q,amp,d);association=paired_join([cycle,cycle],cycles);result['birth']=birth;result['original_association']=association
            need(birth['passed']and association['passed'],'original critical state/mode and native phase birth')
            result['points'].append(row);S.save(self.out/(tag+'-progress.json'),result)
            for i,amp in enumerate(AMPLITUDES[1:],1):
                oldamp=result['points'][-1]['amplitude'];guess=self.predict(z,b,amp-oldamp,tag+f'-step{i}')
                nz,nh,nb,na=self.correct_pair(guess,z,j,g,amp,q,tag+f'-point{i}');new,cy=self.qualify(nz,nh,j,g,amp,q,tag+f'-point{i}')
                forward=paired_join(cycles,cy);result['pending']={'point':new,'forward':forward};S.save(self.out/(tag+'-progress.json'),result);need(forward['passed'],'untransformed paired forward phases')
                rg=self.predict(nz,nb,oldamp-amp,tag+f'-reverse{i}');rz,rh,rb,ra=self.correct_pair(rg,nz,j,g,oldamp,q,tag+f'-reverse{i}')
                reverse_native=[self.native(rz[k:k+12],j,g,float(rz[-1]),tag+f'-reverse{i}-{k}')for k in [0,12]]
                reverse=[B.native_join(cycles[k],reverse_native[k]['cycle'],z[-1],rz[-1])for k in range(2)]
                assoc={'index':i,'forward':forward,'reverse':reverse,'reverse_native_passed':all(v['passed']for v in reverse_native)}
                self.emit('daughter_association',dict(assoc,previous=z,current=nz,reverse_state=rz,previous_cycle_x=cycles[0],previous_cycle_y=cycles[1],current_cycle_x=cy[0],current_cycle_y=cy[1]))
                need(assoc['reverse_native_passed']and all(v['passed']for v in reverse),'paired reverse full native join')
                previous=result['points'][-1];result['points'].append(new);result['associations'].append(assoc);result.pop('pending');z,b,cycles=nz,nb,cy
                if previous['native_stability']=='attracting'and new['maximum_native_radius']>=1:
                    modes=[v['leading_real_mode']for v in new['spectra']]
                    flip=all(v is not None and v['sign']==-1 and v['phase0_real']<=-1 for v in modes)
                    result['candidate_boundary']={'previous':previous,'current':new,'kind':'actual_native6_flip_bracket'if flip else'other_native6_stability_crossing'};result['status']='stopped_at_native_stability_crossing';break
                S.save(self.out/(tag+'-progress.json'),result)
            else:result['status']='fixed_amplitudes_completed'
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
                matrix,rhs=pair[1],-pair[0];v=np.linalg.solve(matrix,rhs);w.emit('profile_solve',{'matrix':matrix,'rhs':rhs,'solution':v})
            measure('eigen',eigen);measure('solve',solve)
        t,b=({k:max(v)for k,v in a.items()}for a in[times,sizes]);seconds=2*(1280*(t['pair']+t['solve'])+512*(t['native']+4*t['eigen']))+20
        raw=1280*(b['pair']+b['solve']+4096)+512*(2*b['native']+4*b['eigen']+4096)+16*1024**2;peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024+256*1024**2
        gate=all(np.isfinite(v)and v>0 for a in times.values()for v in a)and seconds<=180 and raw<=256*1024**2 and peak<=768*1024**2;S.save(out/'profile.json',{'times':times,'bytes':sizes,'projected_seconds':seconds,'projected_raw_bytes':raw,'projected_peak_bytes':peak,'gate':bool(gate),'target_unopened':True});need(gate,'measured paired resource admission')
        j,seeds,binding=B.verify_input(R.RAW/'n12-p6-signed-critical-001');S.save(out/'input-binding.json',{'parent_manifest':B.PIN,'matrix_input_sha256':binding['matrix_input_sha256'],'targets':B.TAGS,'amplitudes':AMPLITUDES,'G2_equals_F6':True})
        for meta,z,cycle in seeds:
            row=w.branch(meta,z,cycle,j,meta['tag']);result['branches'].append(row);S.save(out/'progress.json',result)
        w.check();result['status']='bounded_daughter_unit_completed'
    except BaseException as e:result.update(status='failed_or_interrupted',error=type(e).__name__+': '+str(e))
    finally:
        result.update(elapsed_seconds=time.monotonic()-w.start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,native_calls=w.native_calls,half_calls=w.augmented_calls,total_calls=w.calls,scope='Numerical Gperiod2 daughters on full native primitive6 orbits only. Native stability crossings are proposals, not qualified boundaries or native12/24 joins. No interval, generalizedflip, shrimp or utility claim.');S.save(out/'result.json',result,final=True);S.manifest(out)
    return 0 if result['status']=='bounded_daughter_unit_completed'else 2
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);sys.exit(main(p.parse_args().out))
