#!/usr/bin/env python3
"""One forward endpoint-only G=-F5 fold continuation and possible +/-1 localization."""
import argparse,importlib.util,json,os,resource,signal,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('native_p10pal',ROOT/'tools/continue_clark_rnn_new_p10.py');P=importlib.util.module_from_spec(spec);spec.loader.exec_module(P);S=P.S;R=P.R;B=P.B
from neural_shrimp.clark_rnn_symmetry import fold_family,fold_flip
from neural_shrimp.clark_rnn import iterate
PIN='7b0dd0f7924850552da37bb8f7d81a62105f8b8beede79b876fd1e14c874f741'
COL=P.COL;ROW=P.ROW;CC=np.r_[COL,np.ones(4)];RR=np.r_[np.ones(4)*.05,np.ones(10)]
EXTRA=P.EXTRA+['src/neural_shrimp/clark_rnn_symmetry.py','tools/continue_clark_rnn_p10_symmetric_fold.py','tools/launch_clark_rnn_p10_symmetric_fold.py','docs/plans/clark-rnn-p10-symmetric-fold.md','tests/test_clark_rnn_symmetry.py']
def domain(y):return 15<y[4]*10<18 and .43<y[5]*.1<.48
class Work(P.Work):
    def check(self):
        if self.interrupted:raise InterruptedError('SIGTERM')
        if time.monotonic()-self.start>60:raise TimeoutError('60s cap')
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss>448*1024:raise MemoryError('448MiB worker')
        if self.calls>=4500:raise RuntimeError('4500 paired evaluations')
    def returned(self,z,j,tag,interaction=False):
        self.check();self.calls+=1;start=time.monotonic();h,b,a=(fold_flip if interaction else fold_family)(z,j)
        self.emit('g_return',dict(a,h=h,b=b,tag=tag,interaction=interaction))
        end,m,cycle,js=iterate(z[:4],j,z[4],z[5],10)
        self.emit('native10_return',{'tag':tag,'theta':z[:4].copy(),'gain':z[4],'delta':z[5],'endpoint':end,'monodromy':m,'cycle':cycle,'phase_jacobians':js})
        a.update(cycle=cycle,native_endpoint=end,native_monodromy=m,native_phase_jacobians=js)
        return h,b,a,time.monotonic()-start
    def full(self,y,j,sign,tag):
        h,b,a,t=self.returned(y*COL,j,tag);return ROW*h,ROW[:,None]*b*COL[None,:],a,t
    def correct(self,start,plane,tan,j,sign,tag):
        y=start.copy()
        for k in range(8):
            h,b,a,_=self.full(y,j,1,tag);r=np.r_[h,tan@(y-plane)]
            if np.max(np.abs(r))<1e-10:
                if np.max(np.abs(a['endpoint']-a['theta']))>1e-9 or np.max(np.abs(a['native_endpoint']-a['theta']))>1e-9:return None,'physical_closure'
                return (y,b,a),'closed'
            if k==7:return None,'iteration_limit'
            mat=np.vstack([b,tan])
            try:dx=np.linalg.solve(mat,-r)
            except np.linalg.LinAlgError:return None,'linear_failure'
            self.emit('pal_solve',{'tag':tag,'iteration':k,'matrix':mat,'rhs':-r,'solution':dx})
            if not np.all(np.isfinite(dx)) or np.max(np.abs(dx))>.1:return None,'correction_guard'
            y+=dx
            if not domain(y):return None,'domain_edge'
    def node(self,y,b,a,tan,sign,tag):
        self.check();m=a['monodromy'];e,v=np.linalg.eig(m);self.emit('g_eigen',{'tag':tag,'matrix':m,'eigenvalues':e,'eigenvectors':v})
        ne,nv=np.linalg.eig(a['native_monodromy']);self.emit('native10_eigen',{'tag':tag,'matrix':a['native_monodromy'],'eigenvalues':ne,'eigenvectors':nv})
        le,lv=np.linalg.eig(m.T);self.emit('g_left_eigen',{'tag':tag,'matrix':m.T,'eigenvalues':le,'eigenvectors':lv})
        index=int(np.argmin(abs(e-1)));other=np.delete(e,index);q=y[6:];left=lv[:,int(np.argmin(abs(le-1)))];seps=B.B.primitive_separations(a['cycle'])
        if np.max(abs(left.imag))>1e-8 or abs(left.real@q)<1e-8:raise ValueError('G left/right normalization')
        left=left.real/(left.real@q);c=float(left@np.einsum('ijk,j,k->i',a['second'][:,:4,:4],q,q));coupling=left@a['first'][:,4:6]
        residual=float(np.max(abs(m@v-v*e[None,:]))/max(1,np.max(abs(m))));nr=float(np.max(abs(a['native_monodromy']@nv-nv*ne[None,:]))/max(1,np.max(abs(a['native_monodromy']))))
        sym=float(np.max(abs(a['cycle']+np.roll(a['cycle'],5,axis=0)))/max(1,np.max(abs(a['cycle']))));near=int(np.argmin(abs(other+1)));cross=other[near]
        gclosure=float(max(abs(a['endpoint']-a['theta'])));square=float(np.max(abs(a['native_monodromy']-m@m))/max(1,np.max(abs(a['native_monodromy']))))
        good=gclosure<1e-9 and square<1e-7 and abs(e[index]-1)<1e-7 and min(abs(other-1))>1e-4 and all(x>1e-6 for x in seps.values()) and residual<1e-10 and nr<1e-10 and sym<1e-8 and max(abs(a['native_endpoint']-a['theta']))<1e-9
        out={'tag':tag,'gain':float(y[4]*10),'delta':float(y[5]*.1),'g_critical_error':float(abs(e[index]-1)),'g_other_minus_distance':float(abs(cross+1)),'g_nearest_minus_real':float(cross.real),'g_nearest_minus_imag':float(cross.imag),'g_complement_radius':float(max(abs(other))),'native_plus_one_count_1e6':int(sum(abs(ne-1)<1e-6)),'native_radius':float(max(abs(ne))),'native_other_radius':float(max(abs(np.delete(ne,int(np.argmin(abs(ne-1))))))),'antipodal_defect':sym,'proper_divisor_separations':seps,'g_fold_c':c,'g_fold_parameter_coupling':coupling.tolist(),'g_generic_fold':bool(abs(c)>1e-6 and max(abs(coupling))>1e-6),'qualified_G_fold_native_primitive10':bool(good),'g_eigen_residual':residual,'native_eigen_residual':nr,'g_closure':gclosure,'native_square_relative_defect':square}
        self.emit('symmetric_node',dict(out,y=y.copy(),physical=y*COL,tangent=tan.copy(),left_vector=left,g_monodromy=m,native_monodromy=a['native_monodromy']))
        if not good:raise ValueError('G/native qualification')
        return out
    def localize(self,y,j,tag):
        _,_,a,_=self.full(y,j,1,tag+'-initial');e,v=np.linalg.eig(a['monodromy']);self.emit('interaction_initial_eigen',{'tag':tag,'matrix':a['monodromy'],'eigenvalues':e,'eigenvectors':v});rv=v[:,int(np.argmin(abs(e+1)))]
        if np.max(abs(rv.imag))>1e-8:return {'status':'complex_initial_mode'}
        z=np.r_[y*COL,rv.real/np.linalg.norm(rv.real)];w=z/CC
        for k in range(15):
            h,b,a,_=self.returned(w*CC,j,tag,True);rr=RR*h;bb=RR[:,None]*b*CC[None,:]
            if max(abs(rr))<1e-10:
                e,v=np.linalg.eig(a['monodromy']);ne,nv=np.linalg.eig(a['native_monodromy']);u,s,vh=np.linalg.svd(bb);self.emit('interaction_decompositions',{'tag':tag,'g_matrix':a['monodromy'],'g_eigenvalues':e,'g_eigenvectors':v,'native_matrix':a['native_monodromy'],'native_eigenvalues':ne,'native_eigenvectors':nv,'augmented_matrix':bb,'svd_u':u,'svd_s':s,'svd_vh':vh})
                plus=int(np.argmin(abs(e-1)));minus=int(np.argmin(abs(e+1)));remaining=np.delete(e,[plus,minus]);seps=B.B.primitive_separations(a['cycle']);sym=float(max(abs((a['cycle']+np.roll(a['cycle'],5,axis=0))).ravel())/max(1,np.max(abs(a['cycle']))));native=float(max(abs(a['native_endpoint']-a['theta'])))
                er=float(max(abs((a['monodromy']@v-v*e[None,:])).ravel())/max(1,np.max(abs(a['monodromy']))));ner=float(max(abs((a['native_monodromy']@nv-nv*ne[None,:])).ravel())/max(1,np.max(abs(a['native_monodromy']))))
                gclosure=float(max(abs(a['endpoint']-a['theta'])));square=float(np.max(abs(a['native_monodromy']-a['monodromy']@a['monodromy']))/max(1,np.max(abs(a['native_monodromy']))));native_plus=int(sum(abs(ne-1)<1e-6))
                passed=gclosure<1e-9 and square<1e-7 and native_plus==2 and plus!=minus and abs(e[plus]-1)<1e-7 and abs(e[minus]+1)<1e-7 and min(abs(np.delete(e,plus)-1))>1e-4 and min(abs(np.delete(e,minus)+1))>1e-4 and s[-1]>1e-8 and er<1e-10 and ner<1e-10 and native<1e-9 and sym<1e-8 and all(x>1e-6 for x in seps.values())
                out={'status':'numerical_G_plus_minus_interaction' if passed else 'unqualified_interaction','gain':float(w[4]*10),'delta':float(w[5]*.1),'scaled_residual':float(max(abs(rr))),'g_plus_error':float(abs(e[plus]-1)),'g_minus_error':float(abs(e[minus]+1)),'g_other_radius':float(max(abs(remaining))),'native_closure':native,'g_closure':gclosure,'native_square_relative_defect':square,'native_plus_one_count_1e6':native_plus,'antipodal_defect':sym,'proper_divisor_separations':seps,'augmented_min_singular':float(s[-1]),'g_eigen_residual':er,'native_eigen_residual':ner,'scope':'G=-F5 +/-1 interaction only; nativeF10 has two +1 modes. No generic codim2 normal-form or original-map fold-flip theorem.'}
                self.emit('interaction_result',dict(out,physical=w*CC,cycle=a['cycle'],g_monodromy=a['monodromy'],native_monodromy=a['native_monodromy']));return out
            if k==14:return {'status':'iteration_limit'}
            try:dx=np.linalg.solve(bb,-rr)
            except np.linalg.LinAlgError:return {'status':'linear_failure'}
            self.emit('interaction_solve',{'tag':tag,'iteration':k,'matrix':bb,'rhs':-rr,'solution':dx})
            if max(abs(dx))>.1 or not np.all(np.isfinite(dx)):return {'status':'correction_guard'}
            w+=dx
            if not domain(w):return {'status':'domain_edge'}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();identity=S.source();identity['sha256'].update({p:S.sha(ROOT/p) for p in EXTRA})
    if os.getpriority(os.PRIO_PROCESS,0)<10 or any(os.environ.get(k)!='1' for k in R.M.THREADS):raise ValueError('CPU1 nice10')
    a.out.mkdir(parents=True,exist_ok=False);S.ACTIVE_ROOT=a.out;S.RAW_CAP=128*1024**2;w=Work(a.out);signal.signal(signal.SIGTERM,lambda *_:setattr(w,'interrupted',True));booked=[0]
    def reserve(path,size,final=False):
        old=Path(path).stat().st_size if Path(path).exists() else 0;new=booked[0]-old+size
        if new>(S.RAW_CAP if final else S.RAW_CAP-S.FINAL_RESERVE):raise OSError('prewrite raw cap')
        booked[0]=new
    S.reserve=reserve
    available=int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))*1024
    S.save(a.out/'runtime-start.json',{'source':identity,'pid':os.getpid(),'started_unix':time.time(),'python':sys.version,'numpy':np.__version__,'nice':os.getpriority(os.PRIO_PROCESS,0),'threads':{k:os.environ[k] for k in R.M.THREADS},'mem_available_bytes':available});result={'status':'incomplete','nodes':[],'stop':'node_cap','interaction':None}
    try:
        if available<1024**3:raise MemoryError('headroom')
        tj=np.eye(4)*.2+np.ones((4,4))*.1;ty=np.r_[np.arange(1,5)/10,2.,.5,[1.,0,0,0]]/COL;times=[]
        for k in range(10):times.append(w.full(ty,tj,1,'synthetic_profile')[3])
        projected=4400*max(times)+15;S.save(a.out/'profile.json',{'times':times,'projected_seconds':projected,'gate':np.isfinite(projected) and projected<=60,'target_unopened':True})
        if not np.isfinite(projected) or projected>60:raise RuntimeError('resource screen')
        raw=R.RAW/'new-p10-boundary-curves-001'
        if S.sha(raw/'manifest.json')!=PIN:raise ValueError('pin')
        inv=json.loads((raw/'manifest.json').read_text())['files']
        if set(inv)!={str(p.relative_to(raw)) for p in raw.rglob('*') if p.is_file() and p!=raw/'manifest.json'}:raise ValueError('inventory')
        for name,d in inv.items():
            p=raw/name
            if Path(name).is_absolute() or '..' in Path(name).parts or p.is_symlink() or S.sha(p)!=d['sha256'] or p.stat().st_size!=d['bytes']:raise ValueError('bytes')
        inp=R.RAW/'new-p10-refinement-001/run/inputs.npz';bind=json.loads((raw/'run/input-binding.json').read_text())
        if S.sha(inp)!=bind['source_input_sha256']:raise ValueError('J input')
        with np.load(inp) as f:j=f['j']
        matches=[p for p in (raw/'run').glob('*-node.json') if json.loads(p.read_text())['tag']=='fold:1:53']
        if len(matches)!=1:raise ValueError('unique endpoint')
        p=matches[0];meta=json.loads(p.read_text());arr=p.with_name(meta['array_file'])
        if S.sha(arr)!=meta['array_sha256']:raise ValueError('array')
        with np.load(arr) as f:y=f['y'].copy();ref=f['tangent'].copy()
        S.save(a.out/'input-binding.json',{'manifest_sha256':PIN,'node':str(p),'node_sha256':S.sha(p),'array_sha256':S.sha(arr),'matrix_input':str(inp),'matrix_input_sha256':S.sha(inp),'same_native_state':True,'G_definition':'-F^5; G^2=F^10 by odd symmetry'})
        h,b,na,_=w.full(y,j,1,'seed');tan=w.tangent(b,ref,'seed')
        if max(abs(h))>1e-9:raise ValueError('seed residual')
        old=w.node(y,b,na,tan,1,'seed');result['nodes'].append(old);oldcycle=na['cycle'];cur=y.copy()
        for k in range(1,193):
            pred=cur+.0025*tan;tag=f'symmetric:{k}'
            if not domain(pred):result['stop']='domain_edge';break
            c,status=w.correct(pred,pred,tan,j,1,tag)
            if c is None:result['stop']='forward_'+status;break
            nxt,nb,na=c;nt=w.tangent(nb,tan,tag);pm=B.phase_match(oldcycle,na['cycle']);w.emit('phase_sign_match',dict(pm,tag=tag,parent_cycle=oldcycle,child_cycle=na['cycle']))
            if not pm['passed']:result['stop']='native_identity';break
            reverse,rs=w.correct(nxt-.0025*nt,cur,tan,j,1,tag+'-reverse')
            if reverse is None:result['stop']='reverse_'+rs;break
            mismatch=float(max(abs(reverse[0]-cur)));w.emit('reverse_check',{'tag':tag,'old':cur,'reverse':reverse[0],'mismatch':mismatch})
            if mismatch>1e-7:result['stop']='reverse_mismatch';break
            node=w.node(nxt,nb,na,nt,1,tag);node['reverse_mismatch']=mismatch;result['nodes'].append(node)
            if abs(old['g_nearest_minus_imag'])<1e-8 and abs(node['g_nearest_minus_imag'])<1e-8 and (old['g_nearest_minus_real']+1)*(node['g_nearest_minus_real']+1)<=0:
                result['stop']='first_G_minus_crossing';result['interaction']=w.localize(nxt,j,'interaction');break
            cur=nxt;tan=nt;oldcycle=na['cycle'];old=node
            if k%8==0:S.save(a.out/'progress.json',result)
        w.check();result['status']='completed'
    except BaseException as e:result.update(status='failed_or_interrupted',error=type(e).__name__+': '+str(e))
    finally:
        result.update(elapsed_seconds=time.monotonic()-w.start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,paired_evaluations=w.calls,scope='Numerical symmetry-restricted G=-F5 fold and possible +/-1 interaction; full native10 evidence retained. No original-map generic fold-flip, whole shrimp, interval or ODE claim.');S.save(a.out/'result.json',result,final=True);S.manifest(a.out)
    return 0 if result['status']=='completed' else 2
if __name__=='__main__':sys.exit(main())
