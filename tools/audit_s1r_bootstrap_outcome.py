"""Prospective saved-data audit; no integration or Newton corrections are called."""
import os
THREADS=('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')
for name in THREADS: os.environ[name]='1'
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import signal
import subprocess
import sys
import time
import numpy as np
from scipy.optimize import linear_sum_assignment
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools')]
from audit_s1r_branch_outcome import (require,digest,entry,read_json,inventory,record,
                                     loss_gradient_hessian,pair_match)
from neural_shrimp.period2_branch_io import identity,canonical,Store
from neural_shrimp.gdmaps import GridMap
from neural_shrimp.trainability import SYSTEMS
BASIS='e0fccd73df025f81130d07dfbe60dfe170ef4840'
REVIEW='ff2ec8dc133d4fabf766ec5d776e19af1143f2f7'
OLD_SOURCE='db10bd1e1b25b0e24b1f0f8f97cb84035742c776'
OLD_MANIFEST='45c35991b7ec56f7d9888b6dcf28434396ff94f65de7bf90ba9822408c5f3c8b'
HIDDEN=1.123569870288833
OUTPUT=.6271428571428572
STARTS=(1.1,.95)
SOURCES=('docs/plans/s1r-historical-family-association-next.md','src/neural_shrimp/__init__.py',
 'src/neural_shrimp/gdmaps.py','src/neural_shrimp/trainability.py','src/neural_shrimp/models.py',
 'src/neural_shrimp/period2_branch.py','src/neural_shrimp/period2_branch_io.py',
 'src/neural_shrimp/s1r_bootstrap.py','tools/explore_s1r_bootstrap_association.py','tools/explore_a3_ns_curve.py')
AUDIT_FILES=('tools/audit_s1r_bootstrap_outcome.py','tools/audit_s1r_branch_outcome.py',
 'tests/test_s1r_bootstrap_outcome_audit.py','docs/reviews/s1r-bootstrap-outcome-card.md')
COMPLETE={'completed_common_control_comparison','upper_bootstrap_unqualified','saved_central_unqualified','bridge_gap'}
STRUCTURAL={'bootstrap_start','bootstrap_terminal','bootstrap_screen','bootstrap_result','node_pending','node',
 'edge_pending','edge','bridge_segment','association'}


def exact(a,b,message):
    if isinstance(a,np.ndarray) or isinstance(b,np.ndarray):
        a,b=np.asarray(a),np.asarray(b)
        require(a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes(),message)
    else:require(canonical(a)==canonical(b),message)


def close(a,b,message,atol=1e-13,rtol=1e-12):
    require(np.allclose(a,b,atol=atol,rtol=rtol,equal_nan=True),message)


def aligned(pair,phase):
    require(type(phase) is int and phase in (0,1),'binary phase')
    return pair[::-1] if phase else pair


def match(a,b,replay=False):
    if not b['accepted']:return dict(passed=False,reason='child_not_qualified')
    return pair_match(a['pair'],b['pair'],replay)


def common(a,b):
    require(a['accepted'] and b['accepted'],'qualified common-control pair')
    for k in ('params','rates'):
        require(a[k].dtype==b[k].dtype==np.float64 and a[k].shape==b[k].shape==(2,),'control layout')
        exact(a[k],b[k],'identical controls before matching')
    m=pair_match(a['pair'],b['pair'],True)
    status='same_pair_within_tolerance' if m['passed'] else 'numerically_distinct_pairs' if min(m['costs'])>1e-5 else 'unresolved_match'
    attracting=all(s['stability']=='attracting' for n in (a,b) for s in n['spectra']['phases'])
    return dict(status=status,match=m,distinct_threshold=1e-5,both_numerically_attracting=attracting,
      numerically_coexisting_attractors=bool(status=='numerically_distinct_pairs' and attracting),
      scope='new bootstrap versus saved lineage at common controls; no historical-state ancestry or rigorous uniqueness')


def events(folder,*,historical=False):
    journal=[json.loads(line) for line in (folder/'journal.jsonl').read_text().splitlines()]
    seq=[e['sequence'] for e in journal]
    require(seq==sorted(set(seq)),'ordered journal')
    # Store can skip a sequence when quota rejects an emission before writing.
    out=[];used=set()
    for e in journal:
        name=e['file'];require(Path(name).name==name and name not in used,'journal filename')
        obj=read_json(folder/name);kind,p=record(folder,name)
        require(obj['sequence']==e['sequence'] and kind==e['kind'],'event binding')
        require(name==f"{e['sequence']:05d}-{kind}.json",'event canonical filename')
        used.add(name)
        if 'arrays_file' in obj:used.add(obj['arrays_file'])
        shared={'node_pending','node','edge_pending','edge','corrector_start','corrector_iteration','line_search','corrector_end','pair_spectrum_pending'}
        allowed=shared|({'anchor_feasibility','hidden_link','inputs','rate_identity','segment'} if historical else STRUCTURAL|{'bound_inputs','bootstrap_checkpoint','profile_map_timing','profile_counts'})
        require(kind in allowed,'unknown event kind')
        out.append((kind,p))
    event_files={p.name for p in folder.iterdir() if p.name[:5].isdigit()}
    require(used==event_files,'all event files journaled')
    return out,journal


def bound_artifacts(folder,manifest_sha,execution,mode):
    m=inventory(folder,manifest_sha);r=read_json(folder/'result.json');rt=read_json(folder/'runtime-start.json')
    require(m['schema']=='s1r-branch-manifest-v1' and m['status']==r['status'],'manifest status')
    require(rt['source_commit']==execution and rt['source_review_commit']==REVIEW,'producer source/review')
    require(rt['mode']==r['mode']==mode and r['schema']=='s1r-bootstrap-association-v1','run schema/mode')
    require(set(rt['sources'])==set(SOURCES),'complete source inventory')
    for name in SOURCES:
        data=subprocess.check_output(['git','show',BASIS+':'+name],cwd=ROOT)
        expected=dict(bytes=len(data),sha256=digest(data))
        require(rt['sources'][name]==expected and entry(ROOT/name)==expected,'reviewed source bytes '+name)
        data2=subprocess.check_output(['git','show',execution+':'+name],cwd=ROOT)
        require(data2==data,'execution source changed '+name)
    require(rt['device']=='cpu' and not rt['gpu'] and not rt['paid_compute'] and rt['nice']>=10,'CPU resource scope')
    exact(rt['thread_env'],dict.fromkeys(THREADS,'1'),'thread limits')
    exact(rt['limits'],dict(internal_seconds=8 if mode=='profile' else 100,external_seconds=9 if mode=='profile' else 115,
      kill_grace_seconds=1 if mode=='profile' else 5,phase_correctors=160,link_attempts=19,raw_bytes=64*1024**2),'producer budgets')
    require(rt['input_manifest_sha256']==OLD_MANIFEST,'original input manifest')
    require(sum(p.stat().st_size for p in folder.iterdir())<=64*1024**2,'producer raw cap')
    require(all(not p.stat().st_mode&0o222 for p in folder.iterdir()) and not folder.stat().st_mode&0o222,'immutable producer data')
    ev,journal=events(folder)
    exact(read_json(folder/'checkpoint.json'),dict(status=r['status'],latest=journal[-1] if journal else None),'terminal checkpoint')
    return r,rt,ev,m


def bind_inputs(p,old):
    inventory(old,OLD_MANIFEST)
    require(read_json(old/'runtime-start.json')['source_commit']==OLD_SOURCE,'old execution source')
    _,original=record(old,'00000-inputs.json');old_events,_=events(old,historical=True)
    central=[n for k,n in old_events if k=='node' and n['id']=='node-0004']
    require(len(central)==1 and central[0]['accepted'],'saved central qualification')
    net,x,y,theta=SYSTEMS['S1R'](28002)
    exact(p['model'],asdict(net),'model constructor')
    exact(p['identity'],original['identity'],'original complete identities')
    for key,value in dict(x=x,y=y,theta0=theta).items():
        exact(value,p[key],'constructor '+key);exact(value,original[key],'original '+key)
        exact(identity(value),p['identity'][key],'identity '+key)
    exact(p['central_pair'],central[0]['pair'],'saved central phases')
    exact(p['central_params'],np.array([HIDDEN,OUTPUT]),'central parameters')
    require(p['input_source']==OLD_SOURCE and p['input_manifest_sha256']==OLD_MANIFEST,'input lineage')
    return net


def snapshot(p,initial):
    n=p['completed_loops'];require(type(n) is int and 0<=n<=40000 and p['planned_steps']==40000,'warm loop limits')
    for key in ('attempted_updates','accepted_updates','jvp_successful_returns','jvp_attempts'):
        a=p[key];require(a.shape==(1,) and a.dtype==np.int64 and 0<=a[0]<=40000,'counter '+key)
    a,b,j,k=[int(p[key][0]) for key in ('attempted_updates','accepted_updates','jvp_successful_returns','jvp_attempts')]
    require(b<=a and j<=k and a<=n+1 and b<=n,'attempt/return/update order')
    require(k<=max(0,min(a-38800,1200)) and k-j<=1,'JVP attempt/return bounds')
    for key in ('state','last_valid_state','tangent'):require(p[key].shape==(1,72),'snapshot shape')
    require(p['alive'].shape==(1,) and p['alive'].dtype==bool,'alive layout')
    if p['alive'][0]:
        require(b==n and a in (n,n+1),'live update counters')
        exact(p['state'],p['last_valid_state'],'latest accepted live state')
        require(j>=max(0,n-38800),'completed JVP count')
    else:require(p['diverged_at'][0]>=0 and b+1==a==n,'divergent counters')
    if n==0 and a==0:exact(p['state'],initial[None,:],'warm initial state')
    filled=max(0,n-39616)
    require(p['tail'].shape==(1,filled,72) and p['tail_losses'].shape==(1,filled),'tail layout')
    exact(p['tail_state_updates'],np.arange(39617,39617+filled,dtype=np.int64),'post-update tail indices')
    exact(p['tail_loss_updates'],np.arange(39616,39616+filled,dtype=np.int64),'pre-update loss indices')
    if filled and p['alive'][0]:exact(p['tail'][0,-1],p['state'][0],'terminal tail binding')
    return dict(completed=n,attempted=a,accepted=b,jvp_attempts=k,jvp_returns=j)


def screen_tail(tail):
    if not np.isfinite(tail).all():return -1,float('inf')
    best=float('inf')
    for period in range(1,65):
        if period*6>len(tail):break
        a,b=tail[period:],tail[:-period]
        residual=float(np.max(np.abs(a-b)/(1e-8+1e-6*np.maximum(np.abs(a),np.abs(b)))))
        best=min(best,residual)
        if residual<=1:return period,residual
    return 0,best


def warm_audit(ev,inputs):
    output=[]
    require(all(p['label'] in ('start_1.1','start_0.95') for k,p in ev if k.startswith('bootstrap_')),'unknown bootstrap family')
    for start in STARTS:
        label='start_'+str(start)
        snaps=[p for k,p in ev if k in ('bootstrap_checkpoint','bootstrap_terminal') and p['label']==label]
        terminals=[p for k,p in ev if k=='bootstrap_terminal' and p['label']==label]
        started=any(k=='bootstrap_start' and p['label']==label for k,p in ev)
        if not snaps:
            output.append(dict(label=label,status='no_saved_snapshot',started=started));continue
        require(started and len(terminals)<=1,'snapshot lifecycle')
        counts=[snapshot(p,inputs['theta0']) for p in snaps]
        require(all(a['completed']<=b['completed'] and a['attempted']<=b['attempted'] and a['accepted']<=b['accepted'] for a,b in zip(counts,counts[1:])),'monotone warm counts')
        checkpoints=[p['completed_loops'] for k,p in ev if k=='bootstrap_checkpoint' and p['label']==label]
        require(checkpoints==list(range(0,max(checkpoints,default=-1)+1,1000)),'checkpoint cadence')
        terminal=terminals[0] if terminals else snaps[-1]
        n=terminal['completed_loops'];full=n==40000 and bool(terminal['alive'][0]) and terminal['accepted_updates'][0]==terminal['attempted_updates'][0]==40000
        if full:require(terminal['jvp_attempts'][0]==terminal['jvp_successful_returns'][0]==1200,'full 1200 JVP path')
        screens=[p for k,p in ev if k=='bootstrap_screen' and p['label']==label]
        require(len(screens)<=1,'one screening')
        if screens:
            s=screens[0];require(bool(s['full_horizon'])==bool(full),'full-horizon screen gate')
            exact(s['result']['final_state'],terminal['state'],'screen final state')
            if not terminal['alive'][0]:
                require(s['result']['period'][0]==-1 and s['result']['status'][0]==0,'divergent screen')
            if full:
                close(s['result']['last_evaluated_loss'],terminal['last_evaluated_loss'],'screen last loss')
                close(s['result']['tail_mean_loss'][0],np.mean(terminal['tail_losses'][0]),'screen tail mean loss')
                period,residual=screen_tail(terminal['tail'][0])
                require(s['result']['period'][0]==period,'independent recurrence period')
                close(s['result']['recurrence_residual'][0],residual,'recurrence residual')
                if period>0:require(s['result']['status'][0]==(1 if period==1 else 2),'recurrence status')
                candidate=period==2
                require(s['qualification_seed_indices']==([40000,39999] if candidate else None),'native seed indices')
                require(s['status']==('period_two_candidate' if candidate else 'bootstrap_not_period_two'),'candidate status')
            else:require(s['qualification_seed_indices'] is None,'incomplete warmup cannot qualify')
        # Evaluate every available saved adjacent tail state independently; never roll forward.
        tail=terminal['tail'][0];rates=np.r_[np.full(64,10.**start),np.full(8,10.**OUTPUT)]
        max_state=max_loss=0.;evaluated=0
        for i in range(max(0,len(tail)-1)):
            if not np.isfinite(tail[i:i+2]).all():continue
            loss,gradient,_=loss_gradient_hessian(tail[i],inputs['x'],inputs['y'],inputs['model'])
            error=float(np.max(np.abs(tail[i]-rates*gradient-tail[i+1])))/max(1.,float(np.max(np.abs(tail[i+1]))))
            loss_error=abs(loss-terminal['tail_losses'][0,i+1])/max(1.,abs(loss))
            require(error<=2e-11 and loss_error<=2e-12,'independent tail transition/loss')
            max_state=max(max_state,error);max_loss=max(max_loss,loss_error);evaluated+=1
        output.append(dict(label=label,status='full_recorded_horizon' if full else 'incomplete_or_divergent',**counts[-1],
          checkpoints=checkpoints,independent_tail_transitions=evaluated,max_tail_state_error=max_state,max_tail_loss_error=max_loss))
    return output


class Truncated(Exception):pass


class Structure:
    """Independent replay of only the saved control flow, never a numerical solver."""
    def __init__(self,ev,result):
        self.queue=[(k,p) for k,p in ev if k in STRUCTURAL];self.i=0;self.result=result
        self.nodes={};self.pending={};self.edges=[];self.segments=[];self.cases=[];self.association=None
    def take(self,kind):
        if self.i==len(self.queue):raise Truncated()
        k,p=self.queue[self.i];require(k==kind,'event order: expected '+kind+' got '+k);self.i+=1;return p
    def node(self,params,seeds,purpose):
        p=self.take('node_pending');key=f'node-{len(self.pending):04d}'
        exact(p['params'],params,'node controls');exact(p['rates'],np.array([10.**float(v) for v in params]),'node rates')
        require(p['id']==key and p['purpose']==purpose and not p['accepted'] and p['reason']=='pending','node pending')
        exact(np.stack(p['seeds']),seeds,'full-pair predictor seeds');self.pending[key]=p
        n=self.take('node');require(n['id']==key,'node order')
        for field in ('params','rates','purpose'):exact(n[field],p[field],'node final '+field)
        self.nodes[key]=n;return n
    def edge(self,parent,target,predictor):
        p=self.take('edge_pending');key=f'edge-{len(self.edges):04d}'
        expected=dict(id=key,parent=parent['id'],target=target,purpose='hidden_bridge',status='pending',nodes=[],parent_phase=parent.get('carried_phase',0))
        exact(p,expected,'edge pending');self.edges.append(p)
        out=dict(expected);out['nodes']=[]
        def finish(reason):
            out.update(status='accepted' if reason=='accepted' else 'rejected',reason=reason)
            final=self.take('edge');exact(final,out,'all edge witnesses/rejection reason');self.edges[-1]=out;return out
        def child(name,params,seeds):
            n=self.node(params,seeds,'hidden_bridge:'+name);out['nodes'].append(n['id']);out[{'half_endpoint':'endpoint'}.get(name,name)]=n['id'];return n
        direct=child('direct',target,predictor);out['parent_match']=match(parent,direct)
        if not out['parent_match']['passed']:return finish('direct_or_parent_match')
        half=child('half',(parent['params']+target)/2,parent['pair']);out['half_parent_match']=match(parent,half)
        if not out['half_parent_match']['passed']:return finish('half_or_parent_match')
        endpoint=child('half_endpoint',target,aligned(half['pair'],out['half_parent_match']['phase']))
        out['half_child_match']=match(half,endpoint);out['direct_half_match']=match(direct,endpoint,True)
        if not out['half_child_match']['passed'] or not out['direct_half_match']['passed']:return finish('direct_half_mismatch')
        reverse=child('reverse',parent['params'],aligned(direct['pair'],out['parent_match']['phase']))
        out['reverse_match']=match(parent,reverse,True)
        return finish('accepted' if out['reverse_match']['passed'] else 'reverse_mismatch')
    def segment(self,parent,target,previous,index):
        current=parent;width=abs(target[0]-current['params'][0]);path=[];halves=0
        while current['params'][0]!=target[0]:
            remaining=target[0]-current['params'][0]
            proposal=target.copy() if abs(remaining)<=width else np.array([current['params'][0]+np.copysign(width,remaining),OUTPUT])
            prediction=current['pair']
            if previous is not None:
                denominator=current['params'][0]-previous['params'][0]
                if denominator!=0:prediction=current['pair']+(proposal[0]-current['params'][0])/denominator*(current['pair']-previous['pair'])
            require(len(self.edges)<=19,'link limit')
            if len(self.edges)==19:raise Truncated()
            edge=self.edge(current,proposal,prediction)
            if edge['status']=='rejected':
                halves+=1;width/=2
                if halves>4 or width<.000125*(1-1e-10):
                    expected=dict(status='unresolved_gap',parent=parent['id'],last_accepted=current['id'],target=target,path=path,failed_edge=edge['id'],reason=edge['reason'],halvings=halves)
                    break
            else:
                child=self.nodes[edge['direct']].copy();child['carried_phase']=edge['parent_match']['phase'];child['pair']=aligned(child['pair'],child['carried_phase'])
                previous,current=current,child;path.append(edge['id'])
        else:expected=dict(status='connected',parent=parent['id'],target=target,path=path,endpoint=current['id'],halvings=halves)
        expected['macro_index']=index;saved=self.take('bridge_segment');exact(saved,expected,'segment path/retries/gap')
        self.segments.append(expected);return current,previous,expected['status']
    def run(self,inputs):
        expected_status=None;qualified={};central=None
        try:
            for start in STARTS:
                label='start_'+str(start);case=dict(label=label,params=[start,OUTPUT],status='running',node=None)
                exact(self.take('bootstrap_start'),case,'bootstrap order/controls');self.cases.append(case)
                terminal=self.take('bootstrap_terminal');require(terminal['label']==label,'terminal family')
                screen=self.take('bootstrap_screen');require(screen['label']==label,'screen family')
                case.update(period=int(screen['result']['period'][0]),screen_status=int(screen['result']['status'][0]),status='bootstrap_not_period_two')
                if screen['qualification_seed_indices'] is not None:
                    n=self.node(np.array([start,OUTPUT]),terminal['tail'][0,[-1,-2]],'bootstrap:'+label)
                    case.update(node=n['id'],status='qualified_pair' if n['accepted'] else 'bootstrap_qualification_failed')
                    if n['accepted']:qualified[start]=n
                exact(self.take('bootstrap_result'),case,'bootstrap result')
            if 1.1 not in qualified:expected_status='upper_bootstrap_unqualified'
            else:
                parent=self.node(inputs['central_params'],inputs['central_pair'],'saved_central_requalification');central=parent['id']
                if not parent['accepted']:expected_status='saved_central_unqualified'
                else:
                    previous=None
                    for i in range(1,13):
                        target=np.array([1.1 if i==12 else HIDDEN+(1.1-HIDDEN)*i/12,OUTPUT])
                        parent,previous,status=self.segment(parent,target,previous,i)
                        if status!='connected':expected_status='bridge_gap';break
                    else:
                        self.association=common(parent,qualified[1.1]);exact(self.take('association'),self.association,'common-control decision')
                        expected_status='completed_common_control_comparison'
        except Truncated:
            require(self.result['status'] in ('resource_stop','failed_or_interrupted'),'truncated successful result')
            for case in self.cases:
                if case['status']=='running':case['status']='interrupted_before_completion'
        require(self.i==len(self.queue),'unconsumed witness events/bridging gap')
        if expected_status is not None:require(self.result['status']==expected_status,'terminal branch status')
        exact(self.result['bootstraps'],self.cases,'result warmup cases');exact(self.result['bridge'],self.segments,'result segments')
        exact(self.result['association'],self.association,'result association')
        require(self.result.get('central')==central,'central result binding')
        require(self.result['link_attempts']==len(self.edges)<=19,'total link attempts')
        return dict(final_status=self.result['status'],completed_nodes=len(self.nodes),pending_nodes=len(self.pending)-len(self.nodes),
         links=len(self.edges),accepted_links=sum(e['status']=='accepted' for e in self.edges),segments=len(self.segments),association=self.association)


def corrector_history(ev,result):
    calls={};active=None;iterations={};line_last={};current_node=None;node_calls={};spectrum_pending={}
    for kind,p in ev:
        if kind=='node_pending':current_node=p['id'];node_calls[current_node]=[]
        if kind=='corrector_start':
            require(active is None and p['call']==len(calls)+1 and current_node is not None,'corrector lifecycle')
            active=p['call'];calls[active]=dict(start=p,end=None);node_calls[current_node].append(active)
            exact(p['state'],p['seed'],'initial corrector state')
        elif kind=='corrector_iteration':
            call=p['call'];require(call==active,'iteration active call')
            key=(call,p['iteration']);require(key not in iterations and 1<=p['iteration']<=20,'iteration identity')
            require(p['iteration']==1+sum(c==call for c,i in iterations),'consecutive correction iterations')
            previous=iterations.get((call,p['iteration']-1))
            expected=calls[call]['start']['seed'] if previous is None else line_last.get((call,p['iteration']-1),{}).get('state')
            require(expected is not None,'iteration follows accepted line step');exact(p['state'],expected,'corrector iteration state')
            require(p['step'].shape==p['state'].shape==p['residual'].shape==(72,) and all(np.isfinite(p[k]).all() for k in ('step','state','residual')),'iteration arrays')
            correction=float(np.max(np.abs(p['step'])))/max(1.,float(np.max(np.abs(p['state']))))
            require(correction==p['final_relative_correction'],'relative correction arithmetic');iterations[key]=p
        elif kind=='line_search':
            key=(p['call'],p['iteration']);require(p['call']==active and key in iterations,'line-search call')
            it=iterations[key];require(it['final_relative_correction']>1e-10,'line search only before convergence');prior=line_last.get(key)
            require(p['halving']==(0 if prior is None else prior['halving']+1) and p['halving']<=12,'line-search sequence')
            require(prior is None or not prior['accepted'],'line search after acceptance')
            exact(p['state'],it['state']+2.**-p['halving']*it['step'],'saved line-search step')
            require(p['accepted']==bool(np.max(np.abs(p['residual']))<np.max(np.abs(it['residual']))),'saved line-search descent')
            line_last[key]=p
        elif kind=='corrector_end':
            require(p['call']==active,'corrector end lifecycle');c=calls[active];exact(p['seed'],c['start']['seed'],'corrector seed')
            if p['converged']:
                it=iterations[(active,p['iterations'])]
                require(p['reason']=='converged' and p['final_relative_correction']==it['final_relative_correction']<=1e-10,'convergence gate')
                exact(p['state'],it['state']+it['step'],'last converged state')
            if p['candidate_distance'] is not None:
                distance=float(np.max(np.abs(p['seed']-p['state'])))/max(1.,float(np.max(np.abs(p['seed']))),float(np.max(np.abs(p['state']))))
                require(distance==p['candidate_distance'],'candidate distance')
                if p['converged']:require(distance<=.02,'candidate gate')
            c['end']=p;active=None
        elif kind=='pair_spectrum_pending':
            require(current_node is not None and current_node not in spectrum_pending and active is None,'pending spectrum lifecycle')
            spectrum_pending[current_node]=p
        elif kind=='node':
            if 'spectra' in p:
                require(p['id'] in spectrum_pending,'spectrum preceded by saved pair')
                q=spectrum_pending[p['id']]
                for field in ('pair','checks','params','corrections'):exact(q[field],p[field],'durable pair '+field)
            if p.get('corrections'):
                require([c['call'] for c in p['corrections']]==node_calls[p['id']],'all native corrector ownership')
            for c in p.get('corrections',[]):
                require(c['call'] in node_calls[p['id']] and calls[c['call']]['end'] is not None,'node correction ownership')
                exact(c,calls[c['call']]['end'],'node corrected-state record')
            current_node=None
    count=result['phase_corrector_calls'];require(type(count) is int and len(calls)<=count<=160,'phase call budget')
    # A quota failure can happen after incrementing the counter, before the start event.
    require(count-len(calls)<=1 and (count==len(calls) or result['status']=='resource_stop'),'unrecorded phase attempt')
    if result['status'] in COMPLETE:require(active is None and count==len(calls),'completed corrector lifecycle')
    return dict(recorded_starts=len(calls),recorded_ends=sum(c['end'] is not None for c in calls.values()),
      reported_attempts=count,stored_iterations=len(iterations),stored_line_search_trials=sum(k=='line_search' for k,p in ev),
      limitation='Saved-step arithmetic and descent replayed; intermediate residuals and Newton linear systems not recomputed.')


def qualification(n,pending,family,independent=False):
    corrections=n.get('corrections',[])
    if n['reason'].startswith('qualification_error:'):
        require(not n['accepted'],'exception cannot qualify');return dict(node=n['id'],accepted=False,gate=n['reason'])
    for i,c in enumerate(corrections):
        exact(c['seed'],pending['seeds'][i],'native qualification seed')
    if len(corrections)<2 or not all(c['converged'] for c in corrections):
        require(not n['accepted'] and n['reason'] in ('first_phase_corrector','second_phase_corrector') or n['reason'].startswith('qualification_error:'),'failed correction status')
        return dict(node=n['id'],accepted=False,gate=n['reason'])
    pair=np.stack([c['state'] for c in corrections]);exact(n['pair'],pair,'two native corrected states')
    require(pair.shape==(2,72) and np.isfinite(pair).all(),'complete finite native pair')
    f,jac=family(n['params']);images=np.stack([f(t) for t in pair]);second=np.stack([f(t) for t in images])
    normalized=lambda a,b:float(np.max(np.abs(a-b)/(1e-9+1e-8*np.maximum(np.abs(a),np.abs(b)))))
    closures=[normalized(second[i],pair[i]) for i in range(2)];transitions=[normalized(images[i],pair[1-i]) for i in range(2)]
    sep=float(np.max(np.abs(pair[0]-pair[1])))/max(1.,float(np.max(np.abs(pair))))
    good=max(closures+transitions)<=1 and sep>=1e-6
    saved=n['checks'];require(bool(saved['passed'])==good,'full-state gate decision')
    if not independent:
        close(closures,saved['closure'],'phase closures');close(transitions,saved['transitions'],'phase transitions');close(sep,saved['separation'],'divisor separation')
    im=pair_match(np.stack([pair[0],images[0]]),np.stack([images[1],pair[1]]),True)
    if independent:
        require(im['passed']==n['independent_phase_match']['passed'] and im['phase']==n['independent_phase_match']['phase'],'independent native match')
    else:exact(im,n['independent_phase_match'],'native independent phase match')
    good=good and im['passed'] and im['phase']==0
    if not good:
        require(not n['accepted'] and n['reason']=='state_gate','rejected state gate');return dict(node=n['id'],accepted=False,gate='state_gate')
    spectra=n['spectra'];require(len(spectra['phases'])==2,'both native spectra')
    j0,j1=jac(pair[0]),jac(pair[1]);diagnostics=[];passing=True
    for i,matrix in enumerate((j1@j0,j0@j1)):
        s=spectra['phases'][i];v,left,right=s['values'],s['left'],s['right']
        require(v.shape==(72,) and left.shape==right.shape==matrix.shape==(72,72),'complete spectrum')
        require(all(np.isfinite(a).all() for a in (v,left,right,matrix)),'finite spectrum')
        error=float(np.max(np.abs(matrix-s['matrix'])));require(error<=(2e-11 if independent else 1e-12),'monodromy binding')
        norm=np.linalg.norm(matrix,2);rn=np.linalg.norm(right,axis=0);ln=np.linalg.norm(left,axis=0)
        require(np.all(rn>0) and np.all(ln>0),'nonzero eigenvectors')
        rr=np.linalg.norm(matrix@right-right*v,axis=0)/((norm+np.abs(v))*rn)
        lr=np.linalg.norm(matrix.T.conj()@left-left*v.conj(),axis=0)/((norm+np.abs(v))*ln)
        residual=float(max(np.max(rr),np.max(lr)));passing=passing and residual<=1e-10
        close(rr,s['right_residual'],'saved right residual',atol=2e-14);close(lr,s['left_residual'],'saved adjoint residual',atol=2e-14)
        radius=float(np.max(np.abs(v)));label='attracting' if radius<1-1e-7 else 'unstable' if radius>1+1e-7 else 'neutral_or_unresolved'
        require(s['radius']==radius and s['stability']==label,'spectral stability label')
        item=dict(phase=i,radius=radius,stability=label,matrix_max_error=error,max_eigenpair_residual=residual)
        if independent:
            fresh=np.linalg.eigvals(matrix);cost=np.abs(fresh[:,None]-v[None,:]);rows,cols=linear_sum_assignment(cost)
            fresh_error=float(np.max(cost[rows,cols]));require(fresh_error<=1e-8,'independent spectrum')
            item['fresh_spectrum_error']=fresh_error
        diagnostics.append(item)
    v0,v1=[s['values'] for s in spectra['phases']];cost=np.abs(v0[:,None]-v1[None,:]);rows,cols=linear_sum_assignment(cost)
    error=float(np.max(cost[rows,cols]));passing=passing and error<=1e-7
    exact(cols,spectra['assignment'],'phase spectrum assignment');require(error==spectra['multiset_error'],'phase spectrum error')
    require(bool(spectra['passed'])==bool(n['accepted'])==passing,'spectrum acceptance gate')
    require(n['reason']==('qualified' if passing else 'spectrum_gate'),'qualification terminal reason')
    return dict(node=n['id'],accepted=passing,closure=closures,transition=transitions,separation=sep,phases=diagnostics)


LAUNCHERS={
 'profile':dict(manifest='c93f1ec8adedcb623ff5ef6d32f288cd6442069c4bb9de3e690712fdbd5600b5',source='5c1db56d92d419844ee3b40d114988df17f0f31cc952ea809eb419dd09738fd5'),
 'collect':dict(manifest='adca7ab0dcbb858cc89c30bdfea14e3b33713bcfe686ceeaeddc6642a5483d30',source='1f52e5e487a11b7b34e0b8d728bc6b5466f884895332a0159443cc742ec0faf6')}


def launcher_records(start,receipt,result,runtime,out,profile):
    mode=runtime['mode'];ext,grace=(9,1) if mode=='profile' else (115,5)
    command=['timeout','--signal=TERM',f'--kill-after={grace}s',f'{ext}s','nice','-n','10',
      '/data2/neural-shrimp/.venv/bin/python','tools/explore_s1r_bootstrap_association.py',mode,'--review-commit',REVIEW]
    if mode=='collect':command+=['--profile',str(profile)]
    command+=['--out',str(out)]
    for name,value in start.items():exact(receipt[name],value,'launcher immutable start '+name)
    require(receipt['source_commit']==runtime['source_commit'] and receipt['source_review_commit']==REVIEW,'launcher source/review')
    require(receipt['cwd']=='/home/d7082791602/.codex/worktrees/tiny-lm-prediction-review/neural-shrimp' and receipt['out']==str(out),'launcher owned paths')
    exact(receipt['command'],command,'launcher exact command');exact(receipt['thread_env'],dict.fromkeys(THREADS,'1'),'launcher threads')
    require(receipt['mode']==('profile_only' if mode=='profile' else 'one_collection'),'launcher mode')
    require(receipt['target_collection_authorized']==(mode=='collect'),'launcher recorded launch scope')
    # The trusted root authorization is external to the receipt; metadata is not authorization.
    require(receipt['owned_pids_remaining']==[] and runtime['pid'] in receipt['observed_owned_pids'] and receipt['timeout_pid'] in receipt['observed_owned_pids'],'recorded owned termination')
    require(all(type(p) is int and p>0 for p in receipt['observed_owned_pids']),'recorded PIDs')
    require('launcher_failure' not in receipt,'launcher failure retained, cannot qualify')
    expected_exit=0 if result['status'] in COMPLETE|{'profile_complete'} else 2 if result['status']=='resource_stop' else None
    require(receipt['exit_code']==expected_exit if expected_exit is not None else receipt['exit_code']!=0,'launcher exit/status')
    elapsed=receipt['elapsed_seconds'];child=result['elapsed_seconds']
    require(np.isfinite(elapsed) and np.isfinite(child) and 0<=child<=ext+grace and elapsed>=child,'child wall bounds')
    wall=(datetime.fromisoformat(receipt['finished_utc'])-datetime.fromisoformat(receipt['started_utc'])).total_seconds()
    require(abs(elapsed-wall)<=5,'launcher monotonic/wall-clock consistency')
    for key in ('uptime','memory','gpu','gpu_processes'):
        item=receipt[key];require(set(item)=={'command','exit_code','stdout','stderr'} and isinstance(item['command'],list),'resource observation '+key)
    return dict(exit_code=receipt['exit_code'],elapsed_seconds=elapsed,child_recorded_seconds=child,
      observed_owned_pids=receipt['observed_owned_pids'],recorded_owned_pids_remaining=[],
      limitation='Termination is checked from immutable launcher evidence; historical PID numbers are not queried for current ownership.')


def audit_launcher(folder,mode,result,runtime,out,profile):
    pin=LAUNCHERS[mode];m=inventory(folder,pin['manifest'])
    require(set(m['files'])=={'launcher.py','launcher-start.json','launcher-receipt.json','process.log'},'complete launcher inventory')
    require(entry(folder/'launcher.py')['sha256']==pin['source'],'reviewed launcher source')
    require(all(not p.stat().st_mode&0o222 for p in folder.iterdir()) and not folder.stat().st_mode&0o222,'immutable launcher')
    start=read_json(folder/'launcher-start.json');receipt=read_json(folder/'launcher-receipt.json')
    summary=launcher_records(start,receipt,result,runtime,out,profile)
    require(receipt['output_manifest_verified'] and receipt['output_manifest']==entry(out/'manifest.json'),'launcher raw-manifest binding')
    require(receipt['output_files']==len(list(out.iterdir())) and receipt['output_total_bytes']==sum(p.stat().st_size for p in out.iterdir()),'launcher output inventory counts')
    if result['status']!='failed_or_interrupted':exact(read_json(folder/'process.log'),result,'stdout result binding')
    return dict(**summary,path=str(folder),manifest_sha256=pin['manifest'],files=m['files'])


def audit(args,emit):
    result,runtime,ev,manifest=bound_artifacts(args.collection,args.manifest_sha256,args.producer_commit,'collect')
    profile,profile_rt,pev,pm=bound_artifacts(args.profile,args.profile_manifest_sha256,args.profile_commit,'profile')
    launchers=[]
    for mode,folder,res,rt,out in [('profile',args.profile_launcher,profile,profile_rt,args.profile),('collect',args.collection_launcher,result,runtime,args.collection)]:
        launchers.append(audit_launcher(folder,mode,res,rt,out,args.profile))
    emit('launcher_bindings',launchers)
    require(profile['status']=='profile_complete' and profile_rt['profile_binding'] is None,'completed resource profile')
    timing=profile['timing'];require(timing['map_calls']==2000 and timing['jvp_calls']==128,'profile counts')
    require(all(np.isfinite(timing[k]) and timing[k]>0 for k in ('map_seconds','jvp_seconds')),'profile finite timings')
    estimate=2*(80000*timing['map_seconds']/2000+2400*timing['jvp_seconds']/128)+30
    require(estimate==timing['projected_collection_seconds']<=100 and timing['fits_internal_100_seconds'],'profile resource gate')
    exact(runtime['profile_binding'],dict(path=str(args.profile),manifest_sha256=args.profile_manifest_sha256,projected_collection_seconds=estimate),'collection profile binding')
    counts=[p for k,p in pev if k=='profile_counts'];require(counts==[dict(map_calls=2000,jvp_calls=128)],'profile saved successful returns')
    require(result['historical_family_association']=='unresolved' and result['lower_start_association']=='not_attempted','historical scope')
    require(result['status'] in COMPLETE|{'resource_stop','failed_or_interrupted'},'recognized completion/resource status')
    inputs=[p for k,p in ev if k=='bound_inputs'];profile_inputs=[p for k,p in pev if k=='bound_inputs']
    require(len(inputs)<=1 and len(profile_inputs)==1,'input record cardinality')
    if not inputs:
        require(result['status'] not in COMPLETE and not result['bootstraps'] and not result['bridge'] and result['association'] is None,'incomplete before input binding')
        return dict(status='valid_incomplete_before_inputs',producer_status=result['status'])
    inputs=inputs[0];exact(inputs,profile_inputs[0],'profile and collection data')
    require(Path(runtime['input_root'])==Path(profile_rt['input_root'])==args.original,'original raw path')
    net=bind_inputs(inputs,args.original);emit('bindings',dict(collection_manifest=args.manifest_sha256,profile_manifest=args.profile_manifest_sha256,producer_commit=args.producer_commit,source_basis=BASIS))
    warm=warm_audit(ev,inputs);emit('warmup_audit',warm)
    history=corrector_history(ev,result);emit('corrector_history',history)
    structure=Structure(ev,result);graph=structure.run(inputs);emit('witness_graph',graph)
    def shared_family(params):return GridMap(net,inputs['x'],inputs['y'],[10.**float(params[0])],[10.**float(params[1])]).cell_map(0)
    nodes=[]
    for key,n in structure.nodes.items():nodes.append(qualification(n,structure.pending[key],shared_family))
    emit('all_native_qualifications',nodes)
    # Predeclared positions: qualified bootstrap(s), saved central, first/middle/last completed direct nodes.
    names=[k for k,n in structure.nodes.items() if n['accepted'] and (n['purpose'].startswith('bootstrap:') or n['purpose']=='saved_central_requalification')]
    direct=[k for k,n in structure.nodes.items() if n['accepted'] and n['purpose']=='hidden_bridge:direct']
    if direct:names.extend(direct[i] for i in sorted({0,len(direct)//2,len(direct)-1}))
    representatives=[]
    def independent_family(params):
        rates=np.r_[np.full(64,10.**float(params[0])),np.full(8,10.**float(params[1]))]
        def f(t):return t-rates*loss_gradient_hessian(t,inputs['x'],inputs['y'],inputs['model'])[1]
        def jac(t):return np.eye(72)-rates[:,None]*loss_gradient_hessian(t,inputs['x'],inputs['y'],inputs['model'])[2]
        return f,jac
    for key in dict.fromkeys(names):representatives.append(qualification(structure.nodes[key],structure.pending[key],independent_family,True))
    emit('analytic_representatives',representatives)
    return dict(status='passed_saved_data_consistency' if result['status'] in COMPLETE else 'valid_incomplete_attempt',
      producer_status=result['status'],source_basis=BASIS,execution_source=args.producer_commit,source_review=REVIEW,
      collection_manifest_sha256=args.manifest_sha256,profile_manifest_sha256=args.profile_manifest_sha256,
      launchers=launchers,raw_files=len(manifest['files'])+1,raw_bytes=sum(p.stat().st_size for p in args.collection.iterdir()),
      warmups=warm,corrector_history=history,witness_graph=graph,qualified_native_pairs=sum(n['accepted'] for n in nodes),
      independent_representatives=representatives,historical_family_association='unresolved',lower_start_association='not_attempted',
      limitations=['Counters/checkpoints bind the recorded warmup; unsaved intermediate updates are not independently replayed.',
       'FTLE growth sequence and chaotic/drift status are not independently recomputed; period-two candidates use independent tail recurrence.',
       'All completed pairs use the shared reviewed model derivative; selected pairs and saved tail transitions use independent sample-Jacobian formulas.',
       'No Newton solve, correction iteration, new trajectory, rigorous existence/uniqueness proof or historical full-state ancestry inference.'])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('collection','profile','original','out','collection-launcher','profile-launcher'):p.add_argument('--'+key,type=Path,required=True)
    for key in ('manifest-sha256','profile-manifest-sha256','producer-commit','profile-commit'):p.add_argument('--'+key,required=True)
    args=p.parse_args();require(os.getpriority(os.PRIO_PROCESS,0)>=10,'nice10 required')
    source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    bindings={}
    for name in (*SOURCES,*AUDIT_FILES):
        data=(ROOT/name).read_bytes();require(data==subprocess.check_output(['git','show','HEAD:'+name],cwd=ROOT),'uncommitted audit dependency '+name)
        bindings[name]=entry(ROOT/name)
    store=Store(args.out,16*1024**2);started=time.monotonic()
    meta=dict(status='started',audit_source_commit=source,sources=bindings,started_utc=datetime.now(timezone.utc).isoformat(),
      numpy=np.__version__,load_average=os.getloadavg(),pid=os.getpid(),nice=os.getpriority(os.PRIO_PROCESS,0),thread_env={k:os.environ[k] for k in THREADS},
      inputs={k:str(v) for k,v in vars(args).items() if k!='out'},limits=dict(internal_seconds=45,external_seconds=60,kill_grace_seconds=5,raw_bytes=16*1024**2),gpu=False,paid=False)
    store.json('runtime-start.json',meta)
    def stop(signum,frame):raise TimeoutError('bounded audit stop')
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGALRM,stop);signal.alarm(45)
    report=dict(status='failed',audit_source_commit=source)
    try:report.update(audit(args,store.emit))
    except BaseException as error:report.update(error=type(error).__name__+': '+str(error));raise
    finally:
        signal.alarm(0);report['elapsed_seconds']=time.monotonic()-started;store.finish(report)
        for path in args.out.iterdir():path.chmod(0o444)
        args.out.chmod(0o555)
    print(json.dumps(dict(status=report['status'],elapsed_seconds=report['elapsed_seconds'])))


if __name__=='__main__':main()
