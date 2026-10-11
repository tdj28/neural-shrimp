"""Independent saved first-order block-variation replay; no producer imports.

Caller authenticates complete target, original nonlinear paths, and the prior
independent512-action block-mean authority before invoking this reduction.
"""
import math
import numpy as np
import review_tiny_lm_block_mean_core as C
need,read,array,exact,close,norm,dot=C.need,C.read,C.array,C.exact,C.close,C.norm,C.dot
GATES=dict(qualification=.01,vector_error=.20,improvement_factor=2.)

def correction_sum(ends,forcing,saved):
    ends=np.asarray(ends);forcing=np.asarray(forcing);saved=np.asarray(saved)
    need(ends.shape==(8,len(saved))and forcing.shape==saved.shape and all(np.isfinite(v).all()for v in(ends,forcing,saved)),'eight full correction endpoints and forcing')
    avg=np.asarray([math.fsum(map(float,col))/8 for col in ends.T]);magnitude=np.asarray([math.fsum(abs(float(x))for x in col)/8 for col in ends.T])
    want=avg+forcing;eps=np.finfo(float).eps
    bound=16*eps*magnitude+4*eps*(np.abs(avg)+np.abs(forcing))+np.finfo(float).tiny
    ratio=float(np.max(np.abs(saved-want)/bound));need(ratio<=1,'independent correction mean plus forcing rounding enclosure');return ratio

def replay_corrections(states,powers,diagonal,forcing,corrections,prediction,eta,ends_for,action,label,tick):
    n=powers.shape[1];need(states.shape==(17,n)and powers.shape==(9,n)and diagonal.shape==forcing.shape==(8,n)and corrections.shape==(9,n)and prediction.shape==(n,)and math.isfinite(eta)and eta>0,'complete first-order full-state arrays')
    need(all(np.isfinite(v).all()for v in(states,powers,diagonal,forcing,corrections,prediction)),'finite complete first-order arrays');exact(forcing,diagonal-powers[1:],'literal same-power diagonal variation');exact(corrections[0],np.zeros(n),'zero initial correction');exact(corrections[1],forcing[0],'exact linear action on zero');maximum=0.;calls=0
    for j in range(1,8):
        ends=ends_for(j);need(ends.shape==(8,n),'all eight current-correction endpoints')
        for k in range(8):
            tick();current=corrections[j];a=action(label+f'/correction-{j}/block-{k}/phase-0','hvp',states[2*k],current);calls+=1;mid=current-eta*a
            b=action(label+f'/correction-{j}/block-{k}/phase-1','hvp',states[2*k+1],mid);calls+=1;exact(ends[k],mid-eta*b,'literal two-phase block on same correction vector')
        maximum=max(maximum,correction_sum(ends,forcing[j],corrections[j+1]))
    exact(prediction,powers[-1]+corrections[-1],'entire first-order prediction');need(calls==112,'all112 exact saved correction actions');return maximum

def gates(cases):
    need([(r['seed'],r['rate_role'])for r in cases]==[(s,k)for s in(28161,28162)for k in('original','quarter_step')],'literal four consumed cases')
    q=all(r['qualified']for r in cases);original=[r for r in cases if r['rate_role']=='original']
    return dict(all_four_qualified=bool(q),accurate_in_both_originals=bool(q and all(r['comparison']['comparisons']['first_order']['vector_error']<.2 for r in original)),twice_better_than_block_mean_in_both=bool(q and all(2*r['comparison']['comparisons']['first_order']['vector_error']<r['block_mean_comparison']['comparisons']['block_mean']['vector_error']for r in original)))

def replay(target,out,mean_target,mean_cases,tick,save):
    ev=C.M.Events(target/'run',tick);me=C.M.Events(mean_target/'run',tick);te=C.M.Events(C.M.TARGET/'run',tick);cases=[];maximum=0.
    for seed in(28161,28162):
        folder=C.M.RESPONSE/'run'/f'seed-{seed}-arm-256';eta=read(folder/'inputs.json')['eta'];u=array(folder/'directions.npy',(2,15776))[0]
        for role in('original','quarter_step'):
            i=len(cases);label=f'seed-{seed}-{role}';rate=eta/(1 if role=='original'else 4);prior=mean_cases[i]
            need((prior['seed'],prior['rate_role'])==(seed,role)and prior['qualified'],'literal independently audited block-mean case');binding=read(target/'run'/label/'binding.json');need(binding==read(mean_target/'run'/label/'binding.json')and binding['eta']==rate and binding['full_state']and binding['original_qualification'],'entire unchanged case/checkpoint/token/rate identity')
            if role=='original':ref=array(folder/'reference.npy',(17,15776));grad=array(folder/'reference-gradients.npy',(17,15776))
            else:ref=te.alias(label,'reference',(17,15776));grad=te.alias(label,'reference-gradients',(17,15776))
            exact(ref[1:],ref[:-1]-rate*grad[:-1],'all16 inherited actual GD updates')
            old={k:me.alias(label,'baseline-'+k,(15776,))for k in('ordered','mean','frozen','direction','ce_gradient')}
            exact(old['direction'],u,'same original full-state direction');exact(old['ce_gradient'],grad[16]-.001*ref[16],'same actual CE gradient')
            for k,value in old.items():exact(ev.alias(label,'baseline-'+k,(15776,)),value,'literal independently qualified baseline '+k)
            ro=read(mean_target/'run'/label/'nonlinear-readout.json');eps=ro['epsilons'];responses=[];slopes=[]
            need(len(eps)==2 and all(math.isfinite(e)and e>0 for e in eps),'two unchanged positive nonlinear amplitudes')
            for k,e in enumerate(eps):
                branch={};ce={}
                for side,sign in(('plus',1),('minus',-1)):
                    states=array(folder/f'direction-0-epsilon-{k}-{side}.npy',(17,15776))if role=='original'else te.alias(label,f'epsilon-{k}-{side}',(17,15776))
                    ces=array(folder/f'direction-0-epsilon-{k}-{side}-ce.npy',(3,2))if role=='original'else te.alias(label,f'epsilon-{k}-{side}-ce',(3,2));exact(ces[:,0],np.asarray([1.,4.,16.]),'literal nonlinear horizons');exact(states[0],ref[0]+sign*e*u,'same perturbed initial state');branch[side]=states[16];ce[side]=ces[2,1]
                responses.append((branch['plus']-branch['minus'])/(2*e));slopes.append(float((ce['plus']-ce['minus'])/(2*e)))
            exact(me.alias(label,'responses',(2,15776)),np.asarray(responses),'prior independently qualified nonlinear response');exact(ev.alias(label,'responses',(2,15776)),np.asarray(responses),'current unchanged nonlinear response');close(read(target/'run'/label/'nonlinear-readout.json'),dict(epsilons=eps,slopes=slopes));close(ro,dict(epsilons=eps,slopes=slopes))
            powers=me.alias(label,'block-mean-powers',(9,15776));exact(powers[0],u,'same independently replayed powers');exact(ev.alias(label,'block-mean-powers',(9,15776)),powers,'entire literal512-action powers')
            diagonal=np.asarray([me.alias(label,f'power-{j}-block-ends',(8,15776))[j]for j in range(8)]);exact(ev.alias(label,'block-diagonal-actions',(8,15776)),diagonal,'literal row-j/power-j diagonal forcing source')
            forcing=ev.alias(label,'first-order-forcing',(8,15776));corrections=ev.alias(label,'first-order-corrections',(9,15776));pred=ev.alias(label,'first-order-prediction',(15776,))
            maximum=max(maximum,replay_corrections(ref,powers,diagonal,forcing,corrections,pred,rate,lambda j:ev.alias(label,f'correction-{j}-block-ends',(8,15776)),ev.call,label,tick))
            baseline=C.comparison(powers[-1],old,responses,slopes);close(baseline,prior['comparison']);row=C.comparison(pred,old,responses,slopes);row['comparisons']['first_order']=row['comparisons'].pop('block_mean');row['gates']=GATES
            result=dict(seed=seed,rate_role=role,qualified=bool(row['qualified']and prior['qualified']),comparison=row,block_mean_comparison=baseline,inherited_response_qualified=True);saved=read(target/'run'/label/'result.json');need(saved==out['cases'][i],'whole original case order');close(result,saved);cases.append(result);save(label+'.json',result)
    need(ev.n==448 and ev.counts==dict(hvp=448,gradient=0,ce=0)and len(list((ev.run/'events').glob('*-input.json')))==len(list((ev.run/'events').glob('*-return.json')))==448,'all448 actual correction inputs/returns exhausted');descriptive=gates(cases);need(descriptive==out['descriptive'],'separate qualification/accuracy/twofold gates')
    return dict(passed=True,native_calls=0,retained_hvp_checked=448,correction_means_checked=28,cases=cases,descriptive=descriptive,maximum_mean_rounding_ratio=maximum,scope='Independent fsum means, exact current/mid/state action chains and whole-vector first-order recurrence. Literal prior independently audited512-action powers/diagonal forcing and nonlinear baselines. Retrospective consumed cases; no online utility, isolated ordering or geometry claim.')
