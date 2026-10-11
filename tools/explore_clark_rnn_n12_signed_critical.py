#!/usr/bin/env python3
"""Four fixed signed-half critical roots; preserve full unquotiented native6 gates."""
import argparse,importlib.util,json,os,resource,signal,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('native_boundary',ROOT/'tools/explore_clark_rnn_n12_p6_boundaries.py');M=importlib.util.module_from_spec(spec);spec.loader.exec_module(M);S=M.S;R=M.R
from neural_shrimp.clark_rnn_symmetry import critical_fixed_gain
from neural_shrimp.clark_rnn_boundaries import third_directional
PIN='b6131a7bb3cf9760cd76148ca86d750b38063c4611c2fdae03caa5df07ec09b8'
TARGETS=[('g16-dir-1-step80',-1),('g25-dir-1-step100',-1),('g34-dir-1-step112',-1),('g25-dir1-step86',1)]
EXTRA=M.EXTRA+['src/neural_shrimp/clark_rnn_symmetry.py','tools/explore_clark_rnn_n12_signed_critical.py','tools/launch_clark_rnn_n12_signed_critical.py','tests/test_clark_rnn_n12_signed_critical.py','docs/plans/clark-rnn-n12-signed-critical.md']
class Work(M.Worker):
    def __init__(self,out):super().__init__(out);self.third_calls=0
    def check(self):
        if self.interrupted:raise InterruptedError('SIGTERM')
        if time.monotonic()-self.start>60:raise TimeoutError('60s ceiling')
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss>448*1024:raise MemoryError('448MiB RSS')
        if self.calls>=512:raise RuntimeError('512 combined calls')
    def half(self,z,j,g,sign,tag):
        self.check()
        if self.augmented_calls>=128:raise RuntimeError('128 half-return calls')
        self.calls+=1;self.augmented_calls+=1;t=time.monotonic();h,b,a=critical_fixed_gain(z,j,g,sign,3);self.emit('signed_half',dict(a,h=h,dh=b,sign=sign,tag=tag));return h,b,a,time.monotonic()-t
    def third(self,x,q,j,g,d,tag):
        self.check()
        if self.third_calls>=14:raise RuntimeError('14 third returns')
        self.calls+=1;self.third_calls+=1;t=time.monotonic();c,stages=third_directional(x,q,j,g,d,3);self.emit('signed_third',{'tag':tag,'theta':x,'direction':q,'gain':g,'delta':d,'third':-c,'stages':stages});return -c,time.monotonic()-t
    def fixed(self,x,j,g,d,tag):
        x=x.copy();initial=x.copy();scale=max(1,np.max(np.abs(x)))
        for k in range(12):
            h,b,a,_=self.half(np.r_[x,d,np.eye(12)[0]],j,g,1,tag)
            if not all(np.isfinite(v).all()for v in [h,b]):return None,'nonfinite'
            if np.max(np.abs(h[:12]))/scale<1e-13:return x,'closed'
            if k==11:return None,'iteration_limit'
            try:dx=np.linalg.solve(a['monodromy']-np.eye(12),-h[:12])
            except np.linalg.LinAlgError:return None,'linear_failure'
            self.emit('fixed_solve',{'tag':tag,'iteration':k,'matrix':a['monodromy']-np.eye(12),'rhs':-h[:12],'direction':dx})
            if not np.isfinite(dx).all()or np.max(np.abs(x+dx-initial))>.1*scale:return None,'step_guard'
            x+=dx
    def native(self,x,j,g,d,tag):
        end,m,c,p,_=self.it(x,j,g,d,tag);scale=max(1,np.max(np.abs(c)));sep={str(k):float(np.max(np.abs(c[k]-c[0]))/scale)for k in [1,2,3]};e,v=np.linalg.eig(m);res=float(np.max(np.abs(m@v-v*e[None,:]))/max(1,np.max(np.abs(m))));closure=float(np.max(np.abs(end-x))/scale)
        self.emit('native_context',{'tag':tag,'theta':x,'gain':g,'delta':d,'cycle':c,'monodromy':m,'eigenvalues':e,'eigenvectors':v,'proper_divisor_separations':sep,'closure':closure,'eigen_residual':res})
        return {'passed':bool(all(np.isfinite(a).all()for a in [end,m,c,p,e,v])and closure<1e-10 and min(sep.values())>1e-6 and res<1e-10),'cycle':c,'monodromy':m,'closure':closure,'proper_divisors':sep,'radius':float(max(abs(e))),'eigenvalues':e,'eigen_residual':res}
    def target(self,x,j,g,d,sign,tag):
        report={'tag':tag,'sign':sign,'gain':g,'source_delta':d,'status':'unresolved'}
        original=self.native(x,j,g,d,tag+'-original');seed,status=self.fixed(x,j,g,d,tag+'-fixed-seed');report['seed_correction_status']=status
        if seed is None or not original['passed']:return report
        context=self.native(seed,j,g,d,tag+'-corrected-seed');distance=float(np.max(np.abs(context['cycle']-original['cycle']))/max(1,np.max(np.abs(original['cycle']))));report['source_native_cycle_distance']=distance
        if not context['passed']or distance>1e-7:return report
        _,_,a,_=self.half(np.r_[seed,d,np.eye(12)[0]],j,g,sign,tag+'-mode');e,v=np.linalg.eig(a['monodromy']);index=int(np.argmin(abs(e-sign)));q=v[:,index];self.emit('signed_mode',{'tag':tag,'matrix':a['monodromy'],'eigenvalues':e,'eigenvectors':v,'selected_index':index,'sign':sign})
        if abs(e[index].imag)>1e-8 or abs(e[index]-sign)>.3 or max(abs(q.imag))>1e-8:report['status']='initial_mode_unqualified';return report
        q=q.real/np.linalg.norm(q.real);z=np.r_[seed,d,q];center=z.copy()
        for k in range(12):
            h,b,a,_=self.half(z,j,g,sign,tag+'-critical')
            if not all(np.isfinite(v).all()for v in [h,b]):report['status']='nonfinite';return report
            if np.max(np.abs(h))<1e-12:break
            if k==11:report['status']='critical_iteration_limit';return report
            try:dx=np.linalg.solve(b,-h)
            except np.linalg.LinAlgError:report['status']='critical_linear_failure';return report
            self.emit('critical_solve',{'tag':tag,'iteration':k,'matrix':b,'rhs':-h,'direction':dx});new=z+dx
            if not np.isfinite(new).all()or abs(new[12]-d)>.01 or not .6<new[12]<1 or np.max(np.abs(new[:12]-center[:12]))>.2*max(1,np.max(np.abs(center[:12]))):report['status']='critical_guard';return report
            z=new
        root=self.native(z[:12],j,g,float(z[12]),tag+'-critical-native');adjacent=self.native(root['cycle'][1],j,g,float(z[12]),tag+'-critical-adjacent');pm=M.phase_match(context['cycle'],root['cycle']);reverse,rs=self.fixed(z[:12],j,g,d,tag+'-reverse');reverse_error=None if reverse is None else float(np.max(np.abs(reverse-seed))/max(1,np.max(np.abs(seed))))
        self.emit('association',dict(pm,tag=tag,source_cycle=context['cycle'],critical_cycle=root['cycle'],reverse=np.full(12,np.nan)if reverse is None else reverse,seed=seed,reverse_status=rs,reverse_error=reverse_error))
        gm=a['monodromy'];q=z[13:];e,v=np.linalg.eig(gm);le,lv=np.linalg.eig(gm.T);idx=int(np.argmin(abs(e-sign)));w=lv[:,int(np.argmin(abs(le-sign)))];self.emit('critical_eigensystems',{'tag':tag,'matrix':gm,'eigenvalues':e,'eigenvectors':v,'left_eigenvalues':le,'left_eigenvectors':lv})
        if max(abs(w.imag))>1e-9 or abs(w.real@q)<1e-8:report['status']='left_vector_unqualified';return report
        w=w.real/(w.real@q);other=np.delete(e,idx);B=a['second'][:,:12,:12];bqq=np.einsum('ijk,j,k->i',B,q,q);param=a['first'][:,13];fold_a=float(w@param);fold_c=float(w@bqq);alpha=cubic=None
        if sign==-1:
            rhs=np.column_stack([-param,bqq]);sol=np.linalg.solve(gm-np.eye(12),rhs);third,_=self.third(z[:12],q,j,g,float(z[12]),tag+'-cubic');alpha=float(w@(np.einsum('ijk,j,k->i',B,q,sol[:,0])+a['second'][:,:12,13]@q));cubic=float(w@third/6-w@np.einsum('ijk,j,k->i',B,q,sol[:,1])/2);self.emit('criticality_solves',{'tag':tag,'matrix':gm-np.eye(12),'rhs':rhs,'solution':sol,'left':w,'bqq':bqq,'third':third})
        eigen_res=float(np.max(abs(gm@v-v*e[None,:]))/max(1,np.max(abs(gm))));left_res=float(max(abs(w@gm-sign*w)));half_error=float(np.max(abs(root['cycle']+np.roll(root['cycle'],3,axis=0)))/max(1,np.max(abs(root['cycle']))));square=float(np.max(abs(root['monodromy']-gm@gm))/max(1,np.max(abs(root['monodromy']))))
        native_spectra=[]
        for native in [root,adjacent]:
            values=native['eigenvalues'];critical=int(np.argmin(abs(values-1)));rest=np.delete(values,critical);native_spectra.append({'critical_error':float(abs(values[critical]-1)),'other_plus_distance':float(min(abs(rest-1))),'complement_radius':float(max(abs(rest)))})
        native_critical=all(v['critical_error']<1e-7 and v['other_plus_distance']>1e-4 and v['complement_radius']<1-1e-5 for v in native_spectra)
        gates={'native6':root['passed'],'native6_adjacent':adjacent['passed'],'both_native_critical_spectra':native_critical,'G_fixed':bool(max(abs(a['endpoint']-z[:12]))<1e-11),'G_critical_simple':bool(abs(e[idx]-sign)<1e-7 and min(abs(other-sign))>1e-4),'G_complement_attracting':bool(max(abs(other))<1-1e-5),'G_eigensystems':bool(eigen_res<1e-10 and left_res<1e-7),'half_wave':bool(half_error<1e-8),'native_square':bool(square<1e-7),'native_association':pm['passed'],'reverse':reverse_error is not None and reverse_error<=1e-7,'criticality':bool(abs(fold_a)>1e-5 and abs(fold_c)>1e-5)if sign==1 else bool(abs(alpha)>1e-5 and abs(cubic)>1e-5)}
        report['native_phase_spectra']=native_spectra
        report.update(status='numerically_qualified_signed_half_critical'if all(gates.values())else'unqualified',delta=float(z[12]),gates=gates,residual=float(max(abs(h))),native_identity=pm,reverse_status=rs,reverse_error=reverse_error,half_wave_error=half_error,square_error=square,G_complement_radius=float(max(abs(other))),G_critical_error=float(abs(e[idx]-sign)),G_eigen_residual=eigen_res,G_left_residual=left_res,G_fold_parameter=fold_a if sign==1 else None,G_fold_quadratic=fold_c if sign==1 else None,G_flip_total_alpha=alpha,G_flip_cubic=cubic)
        self.emit('signed_critical_result',dict(report,physical=z,cycle=root['cycle'],native_monodromy=root['monodromy'],G_monodromy=gm,left_vector=w));return report

def main(out):
    identity=S.source();identity['sha256'].update({p:S.sha(ROOT/p)for p in EXTRA})
    if os.getpriority(os.PRIO_PROCESS,0)<10 or any(os.environ.get(k)!='1'for k in R.M.THREADS):raise ValueError('CPU1nice10')
    resource.setrlimit(resource.RLIMIT_AS,(1024**3,)*2);out.mkdir(parents=True,exist_ok=False);S.ACTIVE_ROOT=out;S.RAW_CAP=64*1024**2;w=Work(out);signal.signal(signal.SIGTERM,lambda *_:setattr(w,'interrupted',True));booked=[0]
    def reserve(path,size,final=False):
        old=Path(path).stat().st_size if Path(path).exists()else 0;new=booked[0]-old+size
        if new>(S.RAW_CAP if final else S.RAW_CAP-S.FINAL_RESERVE):raise OSError('prewrite raw cap')
        booked[0]=new
    S.reserve=reserve;S.save(out/'runtime-start.json',{'source':identity,'pid':os.getpid(),'nice':os.getpriority(os.PRIO_PROCESS,0),'threads':{k:os.environ[k]for k in R.M.THREADS},'started_unix':time.time(),'python':sys.version,'numpy':np.__version__});result={'status':'incomplete','targets':[]}
    try:
        if int(next(x.split()[1]for x in Path('/proc/meminfo').read_text().splitlines()if x.startswith('MemAvailable:')))*1024<1024**3:raise MemoryError('1GiB headroom')
        j=np.eye(12)*.2+np.ones((12,12))*.025;x=np.arange(1,13)/20;q=np.eye(12)[0];times={k:[]for k in ['native','half','eigen','solve','third']};sizes={k:[]for k in times}
        def measured(kind,f):
            before=booked[0];t=time.monotonic();v=f();times[kind].append(time.monotonic()-t);sizes[kind].append(booked[0]-before);return v
        for i in range(10):
            ret=measured('native',lambda:w.it(x,j,2.,.8,'synthetic_profile'));measured('half',lambda:w.half(np.r_[x,.8,q],j,2.,-1,'synthetic_profile'))
            def eig():
                e,v=np.linalg.eig(ret[1]);w.emit('profile_eigen',{'eigenvalues':e,'eigenvectors':v,'matrix':ret[1]})
            def solve():
                a=ret[1]-np.eye(12);b=-ret[0];v=np.linalg.solve(a,b);w.emit('profile_solve',{'matrix':a,'rhs':b,'solution':v})
            measured('eigen',eig);measured('solve',solve);measured('third',lambda:w.third(x,q,j,2.,.8,'synthetic_profile'))
        t={k:max(v)for k,v in times.items()};b={k:max(v)for k,v in sizes.items()};projected=2*(512*(t['native']+t['eigen']+t['solve'])+128*(t['half']+4*t['eigen']+t['solve'])+14*t['third'])+15;raw=512*(b['native']+b['eigen']+b['solve']+1024)+128*(b['half']+4*b['eigen']+b['solve']+4096)+14*(b['third']+1024)+8*1024**2;peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024+128*1024**2
        gate=bool(all(np.isfinite(v)and v>0 for values in times.values()for v in values)and projected<=60 and raw<=64*1024**2 and peak<=448*1024**2);S.save(out/'profile.json',{'times':times,'bytes':sizes,'projected_seconds':projected,'projected_raw_bytes':raw,'projected_peak_bytes':peak,'gate':gate,'target_unopened':True})
        if not gate:raise RuntimeError('resource profile failed')
        parent=R.RAW/'n12-p6-boundaries-003'
        if S.sha(parent/'manifest.json')!=PIN:raise ValueError('boundary003 pin')
        inv=json.loads((parent/'manifest.json').read_text())['files']
        if set(inv)!={str(p.relative_to(parent))for p in parent.rglob('*')if p.is_file()and p!=parent/'manifest.json'}:raise ValueError('complete inventory')
        for n,d in inv.items():
            p=parent/n
            if Path(n).is_absolute()or '..'in Path(n).parts or p.is_symlink()or p.stat().st_size!=d['bytes']or S.sha(p)!=d['sha256']:raise ValueError('input hash')
        binding=json.loads((parent/'run/input-binding.json').read_text());inp=R.RAW/'n12-p6-refinement-001/run/inputs.npz'
        if S.sha(inp)!=binding['source_input_sha256']:raise ValueError('same native matrix input')
        with np.load(inp)as f:j=f['j'].copy()
        S.save(out/'input-binding.json',{'boundary_manifest':PIN,'matrix_input_sha256':S.sha(inp),'targets':TARGETS,'same_original_native_state':True,'G':'-F^3; G^2=F^6 by exact oddness; candidate identities still numerically checked'})
        for tag,sign in TARGETS:
            matches=[p for p in (parent/'run').glob('*native_eigen.json')if json.loads(p.read_text())['tag']==tag]
            if len(matches)!=1:raise ValueError('unique fixed native seed')
            p=matches[0];meta=json.loads(p.read_text());a=p.with_name(meta['array_file'])
            if S.sha(a)!=meta['array_sha256']:raise ValueError('seed array')
            with np.load(a)as f:x=f['theta'].copy()
            try:row=w.target(x,j,meta['gain'],meta['delta'],sign,tag)
            except (np.linalg.LinAlgError,ValueError,FloatingPointError) as e:row={'tag':tag,'sign':sign,'status':'unqualified_arithmetic_failure','error':type(e).__name__+': '+str(e)}
            row.update(source_json_sha256=S.sha(p),source_array_sha256=S.sha(a));result['targets'].append(row);S.save(out/'progress.json',result)
        w.check();result['status']='completed'
    except BaseException as e:result.update(status='failed_or_interrupted',error=type(e).__name__+': '+str(e))
    finally:
        result.update(elapsed_seconds=time.monotonic()-w.start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,native_calls=w.native_calls,half_calls=w.augmented_calls,third_calls=w.third_calls,total_calls=w.calls,scope='Numerical signed-half G=-F3 critical roots with native primitive6 context. No interval proof, native6 flip, native12/24 join, generic shrimp or utility claim.');S.save(out/'result.json',result,final=True);S.manifest(out)
    return 0 if result['status']=='completed'else 2
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);sys.exit(main(p.parse_args().out))
