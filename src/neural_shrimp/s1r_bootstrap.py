"""Bounded S1R bootstrap bookkeeping and witnessed common-control association."""
import time
import numpy as np
from .gdmaps import run_grid
from .period2_branch import Budget,Continuation,Options,PilotStop,phase_match

WARM_STEPS=40000
OUTPUT_LOG=0.6271428571428572
STARTS=(1.10,.95)
CENTRAL_HIDDEN=1.123569870288833
WARM_OPTIONS=dict(tail_steps=384,ftle_steps=1000,ftle_align=200,sd_horizons=())


class InterruptibleBudget(Budget):
    def __init__(self,seconds,corrections=160,stop_requested=lambda:False,clock=time.monotonic):
        super().__init__(seconds,corrections,clock);self.stop_requested=stop_requested
    def check(self):
        if self.stop_requested():raise PilotStop('external_signal')
        super().check()


class WarmObserver:
    """Copy snapshots only at cadence/finalization, and never alter live arrays."""
    def __init__(self,budget,emit,label,cadence=1000):
        self.budget,self.emit,self.label=budget,emit,label
        self.cadence=cadence;self.loops=0;self.terminal=None
    def __call__(self,kind,snapshot):
        if kind=='terminal':
            self.terminal=snapshot()
            self.emit('bootstrap_terminal',dict(label=self.label,**self.terminal))
            return
        self.budget.check()
        if kind=='after_step':self.loops+=1
        if kind=='start' or (kind=='after_step' and self.loops%self.cadence==0):
            self.emit('bootstrap_checkpoint',dict(label=self.label,**snapshot()))


def run_bootstrap(grid,theta0,budget,emit,label):
    observer=WarmObserver(budget,emit,label)
    result=run_grid(grid,theta0,WARM_STEPS,observer=observer,**WARM_OPTIONS)
    terminal=observer.terminal
    full,seeds=bootstrap_seeds(result,terminal)
    emit('bootstrap_screen',dict(label=label,full_horizon=full,result=result,
         qualification_seed_indices=[40000,39999] if seeds is not None else None,
         status='period_two_candidate' if seeds is not None else 'bootstrap_not_period_two'))
    return result,seeds


def bootstrap_seeds(result,terminal):
    full=bool(terminal['accepted_updates'][0]==WARM_STEPS and terminal['attempted_updates'][0]==WARM_STEPS
              and terminal['state'].shape==(1,72) and terminal['alive'][0])
    seeds=None
    if full and int(result['period'][0])==2:
        if terminal['tail'].shape!=(1,384,72) or not np.array_equal(terminal['tail_state_updates'][-2:],[39999,40000]):
            raise ValueError('historical tail indexing mismatch')
        seeds=terminal['tail'][0,[-1,-2]].copy()
        if not np.array_equal(seeds[0],result['final_state'][0]):raise ValueError('terminal seed mismatch')
    return full,seeds


class HiddenBridge(Continuation):
    def __init__(self,family,o,budget,emit=lambda kind,payload:None,max_links=19):
        super().__init__(family,o,budget,emit)
        if type(max_links) is not int or max_links<1:raise ValueError('positive integer link limit required')
        self.max_links=max_links
    def link(self,*args,**kwargs):
        self.budget.check()
        if len(self.edges)>=self.max_links:raise PilotStop('global_link_attempt_limit')
        return super().link(*args,**kwargs)
    def hidden_segment(self,parent,target,previous=None,min_step=.000125,max_halvings=4):
        """Only hidden coordinate varies; direct/half/reverse core is unchanged."""
        target=np.asarray(target,dtype=np.float64)
        if target.shape!=(2,) or not np.isfinite(target).all() or target[1]!=parent['params'][1]:raise ValueError('fixed output required')
        width=abs(float(target[0]-parent['params'][0]))
        if width<=0 or not np.isfinite(min_step) or min_step<=0 or type(max_halvings) is not int or max_halvings<0:
            raise ValueError('positive hidden width and valid subdivision controls required')
        current=parent;trial_width=width;retries=0;path=[]
        while current['params'][0]!=target[0]:
            self.budget.check();remaining=target[0]-current['params'][0]
            delta=np.copysign(min(trial_width,abs(remaining)),remaining)
            next_params=target.copy() if abs(remaining)<=trial_width else np.array([current['params'][0]+delta,target[1]])
            predictor=current['pair']
            if previous is not None:
                denominator=current['params'][0]-previous['params'][0]
                if denominator!=0:predictor=current['pair']+(next_params[0]-current['params'][0])/denominator*(current['pair']-previous['pair'])
            edge=self.link(current,next_params,predictor,purpose='hidden_bridge')
            if edge['status']!='accepted':
                retries+=1;trial_width/=2
                if retries>max_halvings or trial_width<min_step*(1-1e-10):
                    return dict(status='unresolved_gap',parent=parent['id'],last_accepted=current['id'],target=target,path=path,
                                failed_edge=edge['id'],reason=edge['reason'],halvings=retries),current,previous
                continue
            child=self.nodes[edge['direct']].copy();child['carried_phase']=edge['parent_match']['phase']
            if child['carried_phase']:child['pair']=child['pair'][::-1]
            previous,current=current,child;path.append(edge['id'])
        return dict(status='connected',parent=parent['id'],target=target,path=path,endpoint=current['id'],halvings=retries),current,previous


def compare_common_controls(a,b,o=Options()):
    """Numerical comparison only after qualification at identical controls."""
    if not a['accepted'] or not b['accepted']:raise ValueError('two qualified nodes required')
    for key in ('params','rates'):
        x,y=np.asarray(a[key]),np.asarray(b[key])
        if x.dtype!=np.float64 or y.dtype!=np.float64 or x.shape!=(2,) or y.shape!=(2,) or x.tobytes()!=y.tobytes():
            raise ValueError('identical binary64 controls required')
    match=phase_match(a['pair'],b['pair'],o,replay=True)
    nearest=min(match['costs'])
    status='same_pair_within_tolerance' if match['passed'] else 'numerically_distinct_pairs' if nearest>1e-5 else 'unresolved_match'
    both_attracting=all(p['stability']=='attracting' for n in (a,b) for p in n['spectra']['phases'])
    return dict(status=status,match=match,distinct_threshold=1e-5,both_numerically_attracting=both_attracting,
        numerically_coexisting_attractors=bool(status=='numerically_distinct_pairs' and both_attracting),
        scope='new bootstrap versus saved lineage at common controls; no historical-state ancestry or rigorous uniqueness')


def fixed_state_profile(grid,state,budget,emit,map_calls=2000,jvp_calls=128):
    """Historical one-cell gradient-with-loss/JVP path; reset input every call."""
    theta=np.asarray(state,dtype=np.float64).reshape(1,-1).copy();original=theta.copy()
    if theta.shape!=(1,grid.net.n_parameters) or not np.isfinite(theta).all():raise ValueError('finite fixed profile state required')
    if (type(map_calls) is not int or type(jvp_calls) is not int or map_calls<1 or jvp_calls<1):raise ValueError('positive integer profile counts required')
    direction=np.zeros_like(theta);direction[0,0]=1.;counts=dict(map_calls=0,jvp_calls=0)
    try:
        start=time.monotonic()
        for _ in range(map_calls):
            budget.check();grad,loss=grid.net.gradient(theta,grid.x,grid.y,return_loss=True)
            result=theta-grid.rates*grad
            if not np.isfinite(result).all() or not np.isfinite(loss).all():raise FloatingPointError('profile map')
            counts['map_calls']+=1
        map_seconds=time.monotonic()-start;emit('profile_map_timing',dict(**counts,seconds=map_seconds))
        start=time.monotonic()
        for _ in range(jvp_calls):
            budget.check();result=direction-grid.rates*grid.net.hvp(theta,grid.x,grid.y,direction)
            if not np.isfinite(result).all():raise FloatingPointError('profile JVP')
            counts['jvp_calls']+=1
        jvp_seconds=time.monotonic()-start
        if not np.array_equal(theta,original):raise AssertionError('profile input state changed')
        estimate=2*(80000*map_seconds/map_calls+2400*jvp_seconds/jvp_calls)+30
        return dict(**counts,map_seconds=map_seconds,jvp_seconds=jvp_seconds,projected_collection_seconds=estimate,
                    fits_internal_100_seconds=bool(np.isfinite(estimate) and estimate<=100),
                    formula='2*(80000*map_seconds/2000 + 2400*jvp_seconds/128) + 30',
                    limitation='timing only; safety factor covers unmeasured screening/checkpoints heuristically, corrector cost may vary')
    finally:emit('profile_counts',counts)
