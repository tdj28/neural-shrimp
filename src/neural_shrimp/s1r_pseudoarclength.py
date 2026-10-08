"""Bounded local PAL pilot. Existing continuation/certificate helpers stay frozen."""
import time
import numpy as np
from scipy.linalg import eig
from scipy.optimize import linear_sum_assignment
from .period2_branch import Options, PilotStop, phase_match, normalized, distance
from .s1r_gap_diagnostic import rank_summary

LIMITS=dict(analytic=256,map=256,solve=64,svd=102,eig=34,correctors=16,qualifiers=17,
    common=8,edges=4,assemblies=256,accepted_edges=2,iterations=8,trials=13,internal_seconds=60.,
    external_seconds=75,kill_grace_seconds=5,store_bytes=127*1024**2,launcher_bytes=1024**2)
TOL=dict(linear_forcing=1e-6,backward=1e-12,nearzero=1e-12,armijo=1e-4,
    parameter_correction=1e-10,plane=1e-10,svd=1e-10,null=1e-10,orientation=1e-8,
    transport=1e-7,initial_tangent=2e-11,initial_h_min=1e-4,initial_arc_max=.005,
    initial_hidden_offset=.000125,common_h=1e-10,local_h=.001,local_theta=.02)
STOP=False

def request_stop(*_):
    global STOP
    STOP=True

def require(ok,reason):
    if not ok:raise ValueError(reason)

def finite(a):
    a=np.asarray(a,dtype=np.float64)
    if not np.isfinite(a).all():raise Reject('nonfinite array')
    return a

def norm(a):return float(np.linalg.norm(a))

def unit(v):
    v=finite(v);length=norm(v)
    if length==0:raise PilotStop('ambiguous_zero_direction')
    return v/length

class Reject(RuntimeError):pass

class Operations:
    def __init__(self,emit,seconds=60.,clock=time.monotonic):
        self.emit,self.clock=emit,clock;self.deadline=clock()+seconds
        self.attempted={k:0 for k in ('analytic','map','solve','svd','eig')};self.returned=self.attempted.copy();self.serial=0
    def check(self):
        if STOP:raise PilotStop('cooperative_signal')
        if self.clock()>=self.deadline:raise PilotStop('internal_deadline')
    def call(self,kind,inputs,function):
        self.check()
        if self.attempted[kind]>=LIMITS[kind]:raise PilotStop(kind+'_operation_cap')
        ident=self.serial;self.serial+=1;self.attempted[kind]+=1
        self.emit('operation_start',dict(id=ident,kind=kind,inputs=inputs,attempted=self.attempted.copy()))
        try:value=function()
        except BaseException as error:
            self.emit('operation_exception',dict(id=ident,kind=kind,error=type(error).__name__+': '+str(error)));raise
        self.returned[kind]+=1
        self.emit('operation_return',dict(id=ident,kind=kind,value=value,returned=self.returned.copy()))
        return ident,value

class Engine:
    def __init__(self,derivative,hidden,output,emit,seconds=60.,clock=time.monotonic):
        self.derivative,self.hidden,self.output,self.emit=derivative,hidden,float(output),emit
        self.ops=Operations(emit,seconds,clock);self.o=Options();self.derivatives={};self.phases={}
        self.counts=dict(correctors=0,qualifiers=0,common=0,edges=0,assemblies=0);self.nodes={};self.last_witnessed=None;self.durable_edges=[]
        self.anchor=None
    def count(self,kind):
        self.ops.check()
        if self.counts[kind]>=LIMITS[kind]:raise PilotStop(kind+'_cap')
        ident=self.counts[kind];self.counts[kind]+=1;return ident
    def phase(self,theta,h):
        theta=finite(theta);h=float(h);require(np.isfinite(h),'finite hidden parameter')
        key=theta.tobytes();pkey=(key,np.float64(h).tobytes())
        if pkey in self.phases:
            value=self.phases[pkey];self.emit('cache_hit',dict(kind='map',id=value['id'],theta=theta,h=h));return value
        if key in self.derivatives:
            did,raw=self.derivatives[key];self.emit('cache_hit',dict(kind='analytic',id=did,theta=theta))
        else:
            def compute():
                loss,g,H=self.derivative(theta)
                return dict(loss=float(loss) if np.isfinite(loss) else None,gradient=np.asarray(g),hessian=np.asarray(H))
            did,raw=self.ops.call('analytic',dict(theta=theta),compute);self.derivatives[key]=(did,raw)
        require(raw['loss'] is not None,'finite analytic loss')
        g,H=finite(raw['gradient']),finite(raw['hessian']);n=len(theta)
        require(g.shape==(n,) and H.shape==(n,n),'analytic shapes')
        rates=np.r_[np.full(self.hidden,10.**h),np.full(n-self.hidden,10.**self.output)]
        def compute_map():
            with np.errstate(over='ignore',invalid='ignore'):
                return dict(image=theta-rates*g,J=np.eye(n)-rates[:,None]*H,c=-np.r_[np.log(10.)*rates[:self.hidden],np.zeros(n-self.hidden)]*g)
        ident,rawmap=self.ops.call('map',dict(theta=theta,h=h,rates=rates,analytic_id=did),compute_map)
        value=dict(id=ident,theta=theta,h=h,rates=rates,**rawmap);self.phases[pkey]=value
        for k in ('image','J','c'):finite(value[k])
        return value
    def assembly(self,theta,h,purpose):
        ident=self.count('assemblies');self.emit('assembly_start',dict(id=ident,purpose=purpose,theta=theta,h=float(h)))
        first=self.phase(theta,h);second=self.phase(first['image'],h);n=len(theta)
        with np.errstate(over='ignore',invalid='ignore'):
            M=second['J']@first['J'];value=dict(id=ident,theta=theta,h=float(h),map1=first['image'],map2=second['image'],R=second['image']-theta,M=M,A=M-np.eye(n),b=second['J']@first['c']+second['c'],first=first['id'],second=second['id'])
        self.emit('assembly',dict(purpose=purpose,**value))
        for k in ('R','M','A','b'):finite(value[k])
        return value
    def svd(self,matrix,purpose):
        matrix=finite(matrix)
        def compute():
            U,s,Vh=np.linalg.svd(matrix,full_matrices=True);return dict(U=U,s=s,Vh=Vh)
        ident,value=self.ops.call('svd',dict(matrix=matrix,purpose=purpose),compute)
        U,s,Vh=(finite(value[k]) for k in ('U','s','Vh'));m,n=matrix.shape
        require(U.shape==(m,m) and s.shape==(m,) and Vh.shape==(n,n),'full SVD shape')
        require(np.all(s>=0) and np.all(s[:-1]>=s[1:]),'ordered nonnegative singular values')
        scale=max(1.,norm(matrix));V=Vh[:m].T
        checks=dict(right=norm(matrix@V-U*s)/scale,left=norm(matrix.T@U-V*s)/scale,
            reconstruction=norm(matrix-(U*s)@Vh[:m])/scale,orthogonality=max(norm(U.T@U-np.eye(m)),norm(Vh@Vh.T-np.eye(n))),rank=rank_summary(s))
        self.emit('svd_checks',dict(id=ident,purpose=purpose,**checks))
        if max(checks[k] for k in ('right','left','reconstruction','orthogonality'))>TOL['svd']:raise PilotStop('SVD_residual_gate')
        return ident,value,checks
    def tangent(self,a,reference,purpose):
        B=np.column_stack((a['A'],a['b']));sid,v,checks=self.svd(B,purpose)
        if checks['rank']['status']!='full_row_rank_resolved':raise PilotStop('augmented_rank_ambiguity')
        t=v['Vh'][-1].copy();cosine=float(t@unit(reference))
        if abs(cosine)<=TOL['orientation']:raise PilotStop('tangent_orientation_ambiguity')
        if cosine<0:t=-t;cosine=-cosine
        residual=norm(B@t)/max(1.,norm(B));self.emit('tangent',dict(purpose=purpose,svd_id=sid,reference=reference,tangent=t,cosine=cosine,residual=residual))
        if residual>TOL['null']:raise PilotStop('null_residual_gate')
        return t
    def spectrum(self,M,purpose):
        norm_id,nv,_=self.svd(M,purpose+':norm');cond_id,cv,_=self.svd(M-np.eye(len(M)),purpose+':condition')
        def compute():
            w,left,right=eig(M,left=True,right=True,check_finite=True);return dict(values=w,left=left,right=right)
        eid,v=self.ops.call('eig',dict(matrix=M,purpose=purpose),compute)
        w,left,right=(v[k] for k in ('values','left','right'));n=len(M)
        require(w.shape==(n,) and left.shape==right.shape==(n,n) and all(np.isfinite(x).all() for x in (w,left,right)),'full finite eigensystem')
        matrix_norm=float(nv['s'][0]);denom=np.maximum(matrix_norm+np.abs(w),np.finfo(float).tiny)
        ln,rn=np.linalg.norm(left,axis=0),np.linalg.norm(right,axis=0)
        require(np.all(ln>0) and np.all(rn>0),'nonzero eigenvectors')
        rr=np.linalg.norm(M@right-right*w,axis=0)/(denom*rn);lr=np.linalg.norm(M.conj().T@left-left*np.conj(w),axis=0)/(denom*ln)
        rho=float(np.max(np.abs(w)));condition=float(cv['s'][0]/cv['s'][-1]) if cv['s'][-1]>0 else None
        if condition is not None and not np.isfinite(condition):condition=None
        row=dict(eig_id=eid,norm_svd_id=norm_id,condition_svd_id=cond_id,matrix_norm=matrix_norm,condition=condition,right_residual=rr,left_residual=lr,radius=rho,stability='attracting' if rho<1-self.o.stability_margin else 'unstable' if rho>1+self.o.stability_margin else 'neutral_or_unresolved')
        self.emit('spectrum_checks',dict(purpose=purpose,**row))
        finite(rr);finite(lr)
        if max(float(rr.max()),float(lr.max()))>self.o.eigen_residual:raise PilotStop('eigen_residual_gate')
        return row,w
    def states(self,pair,h,purpose):
        pair=finite(pair);n=pair.shape[1];require(pair.shape==(2,n),'full pair')
        aa=[self.assembly(pair[i],h,purpose+':'+str(i)) for i in range(2)]
        images=np.stack([self.phase(pair[i],h)['image'] for i in range(2)])
        closures=[normalized(aa[i]['map2'],pair[i],self.o) for i in range(2)]
        transitions=[normalized(images[i],pair[1-i],self.o) for i in range(2)]
        separation=float(np.max(np.abs(pair[0]-pair[1])))/max(1.,float(np.max(np.abs(pair))))
        checks=dict(closure=closures,transitions=transitions,separation=separation,passed=bool(max(closures+transitions)<=1 and separation>=self.o.divisor_separation))
        self.emit('state_checks',dict(purpose=purpose,pair=pair,h=h,**checks))
        if not checks['passed']:raise Reject('physical_state_gate')
        return aa,checks
    def qualify(self,pair,h,reference,carried,purpose):
        ident=self.count('qualifiers');name=f'node-{ident:04d}';out=dict(id=name,pair=pair,h=float(h),carried_phase=int(carried),accepted=False,purpose=purpose)
        self.emit('qualifier_start',out)
        try:
            aa,checks=self.states(pair,h,name);spec=[self.spectrum(aa[i]['M'],name+':phase'+str(i)) for i in (0,1)]
            cost=np.abs(spec[0][1][:,None]-spec[1][1][None,:]);rows,cols=linear_sum_assignment(cost);error=float(np.max(cost[rows,cols]))
            if error>self.o.phase_spectrum_agreement:raise PilotStop('native_spectrum_disagreement')
            primary=self.tangent(aa[carried],reference,name+':tangent'+str(carried));first=self.phase(pair[carried],h)
            propagated=np.r_[first['J']@primary[:-1]+first['c']*primary[-1],primary[-1]]
            other=self.tangent(aa[1-carried],propagated,name+':tangent'+str(1-carried))
            transport_residual=norm(np.column_stack((aa[1-carried]['A'],aa[1-carried]['b']))@unit(propagated))/max(1.,norm(np.column_stack((aa[1-carried]['A'],aa[1-carried]['b']))))
            cosine=float(unit(propagated)@other)
            self.emit('phase_transport',dict(node=name,carried_phase=carried,primary=primary,other=other,propagated=propagated,residual=transport_residual,cosine=cosine))
            if transport_residual>TOL['null'] or abs(1-cosine)>TOL['transport']:raise PilotStop('phase_tangent_disagreement')
            out.update(accepted=True,state_checks=checks,spectra=[v[0] for v in spec],spectrum_error=error,spectrum_assignment_rows=rows,spectrum_assignment_cols=cols,tangent=primary,z=np.r_[pair[carried],h])
        except BaseException as e:out['error']=type(e).__name__+': '+str(e);raise
        finally:self.emit('qualifier_end',out)
        self.nodes[name]=out;return out
    def local(self,z):
        if self.anchor is not None and (abs(z[-1]-self.anchor[-1])>TOL['local_h'] or norm(z[:-1]-self.anchor[:-1])>TOL['local_theta']):raise Reject('local_box')
    def correct(self,seed,predictor,plane_origin,plane_tangent,offset,purpose,reference=None):
        reference=plane_tangent if reference is None else reference
        ident=self.count('correctors');ell=norm(predictor-seed);require(ell>0,'positive predictor length')
        z=finite(predictor).copy();out=dict(id=ident,purpose=purpose,seed=seed,predictor=predictor,plane_origin=plane_origin,plane_tangent=plane_tangent,orientation_reference=reference,offset=offset,ell=ell,state=z.copy(),converged=False)
        self.emit('corrector_start',out)
        try:
            for iteration in range(LIMITS['iterations']):
                self.local(z);a=self.assembly(z[:-1],z[-1],f'corrector{ident}:iteration{iteration}')
                G=np.r_[a['R'],plane_tangent@(z-plane_origin)-offset];J=np.vstack((np.column_stack((a['A'],a['b'])),plane_tangent))
                sid,returned=self.ops.call('solve',dict(corrector=ident,iteration=iteration,z=z,G=G,J=J),lambda:dict(step=np.linalg.solve(J,-G)))
                step=finite(returned['step']);error=J@step+G;gn=norm(G);en=norm(error);denom=norm(J)*norm(step)+gn
                backward=en/denom if denom>0 else 0. if en==0 else None
                near=gn<=TOL['nearzero']*max(1.,norm(z));forcing=None if near else en/gn
                good=(en<=TOL['nearzero'] if near else forcing<=TOL['linear_forcing']) and backward is not None and backward<=TOL['backward']
                slope=float(G@(J@step));relative=float(np.max(np.abs(step[:-1])))/max(1.,float(np.max(np.abs(z[:-1]))))
                self.emit('linear_checks',dict(corrector=ident,iteration=iteration,solve_id=sid,error=error,forcing=forcing,nearzero=near,backward=backward,passed=bool(good),slope=slope,theta_relative_correction=relative,h_correction=float(abs(step[-1]))))
                if not good:raise Reject('linear_equation_gate')
                if relative<=self.o.newton_relative_correction and abs(step[-1])<=TOL['parameter_correction'] and abs(G[-1])<=TOL['plane'] and normalized(a['map2'],z[:-1],self.o)<=1:
                    # Keep the evaluated state; no unevaluated final micro-step.
                    if norm(z-predictor)>ell/2 or distance(z[:-1],predictor[:-1])>self.o.candidate_distance:raise Reject('predictor_locality')
                    out.update(converged=True,state=z.copy(),iterations=iteration+1);break
                if not np.isfinite(slope) or slope>=0:raise Reject('non_descent')
                merit=.5*float(G@G);accepted=False
                for half in range(LIMITS['trials']):
                    alpha=2.**-half;trial=z+alpha*step
                    try:self.local(trial)
                    except Reject as error:
                        self.emit('armijo_trial',dict(corrector=ident,iteration=iteration,halving=half,alpha=alpha,state=trial,accepted=False,error=str(error)))
                        raise
                    try:
                        ta=self.assembly(trial[:-1],trial[-1],f'corrector{ident}:trial{iteration}:{half}')
                        TG=np.r_[ta['R'],plane_tangent@(trial-plane_origin)-offset];value=.5*float(TG@TG);passed=bool(np.isfinite(value) and value<=merit+TOL['armijo']*alpha*slope)
                    except Reject as e:
                        self.emit('armijo_trial',dict(corrector=ident,iteration=iteration,halving=half,alpha=alpha,state=trial,accepted=False,error=str(e)));continue
                    self.emit('armijo_trial',dict(corrector=ident,iteration=iteration,halving=half,alpha=alpha,state=trial,G=TG,merit=value,old_merit=merit,slope=slope,accepted=passed))
                    if passed:z=trial;out['state']=z.copy();accepted=True;break
                if not accepted:raise Reject('line_search_stalled')
            else:raise Reject('newton_iteration_limit')
        except BaseException as e:out['error']=type(e).__name__+': '+str(e);raise
        finally:self.emit('corrector_end',out)
        pair=np.stack([z[:-1],self.phase(z[:-1],z[-1])['image']])
        return self.qualify(pair,z[-1],reference,0,purpose)
    def match(self,parent,child,replay=False):
        value=phase_match(parent['pair'],child['pair'],self.o,replay=replay)
        self.emit('pair_match',dict(parent=parent['id'],child=child['id'],**value))
        if value['cost_gap']<self.o.phase_cost_gap_fraction_of_separation*value['separation']:raise PilotStop('cyclic_phase_ambiguity')
        if not value['passed']:raise Reject('pair_match_gate')
        if (parent['carried_phase']^value['phase'])!=child['carried_phase']:raise PilotStop('carried_phase_mismatch')
        return value
    def common(self,reference,candidate):
        ident=self.count('common');out=dict(id=ident,reference=reference['id'],candidate=candidate['id'],canonical_h=reference['h'],candidate_original_h=candidate['h'],h_difference=abs(reference['h']-candidate['h']),passed=False)
        self.emit('common_start',out)
        try:
            if out['h_difference']>TOL['common_h']:raise Reject('common_h_disagreement')
            _,checks=self.states(candidate['pair'],reference['h'],'common'+str(ident))
            match=self.match(reference,candidate,True)
            out.update(state_checks=checks,match=match,passed=True)
        except BaseException as error:out['error']=type(error).__name__+': '+str(error);raise
        finally:self.emit('common_end',out)
    def edge(self,parent,s):
        ident=self.count('edges');z,t=parent['z'],parent['tangent'];out=dict(id=ident,parent=parent['id'],s=s,nodes=[],accepted=False)
        self.emit('edge_start',out)
        try:
            direct=self.correct(z,z+s*t,z,t,s,'direct');out['nodes'].append(direct['id']);self.match(parent,direct)
            half=self.correct(z,z+s/2*t,z,t,s/2,'half');out['nodes'].append(half['id']);self.match(parent,half)
            def intersection(node,offset):
                denominator=float(t@node['tangent'])
                if denominator<=TOL['orientation']:raise PilotStop('plane_predictor_ambiguity')
                length=(offset-float(t@(node['z']-z)))/denominator
                return node['z']+length*node['tangent']
            endpoint=self.correct(half['z'],intersection(half,s),z,t,s,'half_endpoint',reference=half['tangent']);out['nodes'].append(endpoint['id']);self.match(half,endpoint);self.common(direct,endpoint)
            reverse=self.correct(direct['z'],intersection(direct,0.),z,t,0.,'reverse',reference=direct['tangent']);out['nodes'].append(reverse['id']);self.common(parent,reverse)
            out.update(accepted=True,direct=direct['id'])
        except Reject as error:out['error']=str(error)
        except BaseException as error:out['error']=type(error).__name__+': '+str(error);raise
        finally:self.emit('edge_end',out)
        if out['accepted']:
            # Only after the durable end record advances the witnessed state.
            self.last_witnessed=direct['id'];self.durable_edges.append(ident)
            return direct
        return None
    def run(self,pair,h,carried,secant,saved_h_component):
        self.anchor=np.r_[pair[carried],h].copy();anchor=self.qualify(pair,h,secant,carried,'anchor')
        if abs(anchor['tangent'][-1]-saved_h_component)>TOL['initial_tangent'] or abs(saved_h_component)<TOL['initial_h_min']:raise PilotStop('initial_tangent_binding')
        self.last_witnessed=anchor['id'];s=min(TOL['initial_arc_max'],TOL['initial_hidden_offset']/abs(saved_h_component));self.emit('initial_step',dict(anchor=anchor['id'],saved_h_component=saved_h_component,s=s))
        current=anchor
        for macro in range(2):
            for attempt in range(2):
                child=self.edge(current,s)
                if child is not None:current=child;break
                if attempt==1:return dict(status='unresolved_gap',last_witnessed=self.last_witnessed,accepted_edges=self.durable_edges)
                s/=2;self.emit('halve',dict(macro=macro,new_s=s,parent=current['id']))
        self.ops.check()
        return dict(status='two_edges_complete',last_witnessed=self.last_witnessed,accepted_edges=self.durable_edges)
