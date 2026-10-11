"""Independent fsum and exact retained-action replay; metadata is caller-owned.

No native model or producer block-mean helper imports. The caller must first
bind complete original/pair/mean independent authorities and all raw hashes.
"""
import math
from pathlib import Path
import numpy as np
import review_tiny_lm_pair_blocks as B
P=B.P;M=B.M
need,read,array,exact,close,norm,dot=B.need,B.read,B.array,B.exact,B.close,B.norm,B.dot
GATES=dict(qualification=.01,vector_error=.20,improvement_factor=10.)

def mean_of_eight(ends,saved):
    ends=np.asarray(ends);saved=np.asarray(saved);need(ends.shape==(8,len(saved))and np.isfinite(ends).all()and np.isfinite(saved).all(),'eight complete finite endpoints')
    independent=np.asarray([math.fsum(map(float,col))/8 for col in ends.T])
    magnitude=np.asarray([math.fsum(abs(float(x))for x in col)/8 for col in ends.T])
    bound=16*np.finfo(float).eps*magnitude+np.finfo(float).tiny
    ratio=float(np.max(np.abs(saved-independent)/bound));need(ratio<=1,'independent eight-block mean rounding enclosure');return ratio

def replay_powers(states,u,eta,powers,ends_for,action,label,tick):
    n=len(u);need(states.shape==(17,n)and powers.shape==(9,n)and math.isfinite(eta)and eta>0,'entire full-state path and powers');exact(powers[0],u,'literal unchanged initial direction');maximum=0.;calls=0
    for j in range(8):
        ends=ends_for(j);need(ends.shape==(8,n),'all eight actual block endpoints')
        for k in range(8):
            tick();current=powers[j]
            a=action(label+f'/power-{j}/block-{k}/phase-0','hvp',states[2*k],current);calls+=1
            mid=current-eta*a
            b=action(label+f'/power-{j}/block-{k}/phase-1','hvp',states[2*k+1],mid);calls+=1
            exact(ends[k],mid-eta*b,'literal two-step endpoint from the same power input')
        maximum=max(maximum,mean_of_eight(ends,powers[j+1]))
    need(calls==128,'all128 ordered block actions per case');return maximum

def comparison(pred,old,responses,slopes):
    need(set(old)=={'ordered','mean','frozen','direction','ce_gradient'}and len(responses)==len(slopes)==2,'entire independent comparison inputs')
    fine=responses[1];gain=norm(fine);need(gain>0,'nonzero nonlinear response');scale=max(gain,1e-12);cescale=max(abs(slopes[1]),1e-8);metrics={}
    for name,value in dict(block_mean=pred,**{k:old[k]for k in ('ordered','mean','frozen','direction')}).items():
        vn=norm(value);slope=dot(old['ce_gradient'],value)
        metrics[name]=dict(vector_error=norm(value-fine)/scale,gain=vn,signed_alignment=None if vn==0 else dot(value,fine)/(vn*gain),training_ce_slope=slope,training_ce_slope_error=abs(slope-slopes[1])/cescale)
    fd=norm(responses[0]-fine)/scale;ce=abs(slopes[0]-slopes[1])/cescale
    qualified=fd<=.01 and metrics['ordered']['vector_error']<=.01 and ce<=.01 and metrics['ordered']['training_ce_slope_error']<=.01
    return dict(qualified=bool(qualified),response_gain=gain,finite_difference_disagreement=fd,ce_slope_disagreement=ce,ce_slopes=list(slopes),comparisons=metrics,gates=GATES)

def gates(cases):
    need([(c['seed'],c['rate_role'])for c in cases]==[(s,r)for s in(28161,28162)for r in('original','quarter_step')],'four literal consumed cases')
    q=all(c['qualified']for c in cases);original=[c['comparison']['comparisons']for c in cases if c['rate_role']=='original']
    return dict(all_four_qualified=q,accurate_in_both_originals=bool(q and all(v['block_mean']['vector_error']<.2 for v in original)),tenfold_better_than_single_step_mean_in_both=bool(q and all(10*v['block_mean']['vector_error']<v['mean']['vector_error']for v in original)))

def replay(target,out,prior_mean_cases,prior_pair_cases,tick,save):
    """Caller has already authenticated the exact target and both authorities.

    prior_mean_cases are the independently recomputed temporal-mean rows;
    prior_pair_cases supply four qualified literal consumed-case associations.
    """
    ev=M.Events(target/'run',tick);me=M.Events(M.TARGET/'run',tick);cases=[];maximum=0.
    for seed in(28161,28162):
        folder=M.RESPONSE/'run'/f'seed-{seed}-arm-256';eta=read(folder/'inputs.json')['eta'];u=array(folder/'directions.npy',(2,15776))[0]
        for role in('original','quarter_step'):
            index=len(cases);label=f'seed-{seed}-{role}';rate=eta/(1 if role=='original'else 4);authority=prior_pair_cases[index];mean_authority=prior_mean_cases[index]
            need((authority['seed'],authority['rate_role'])==(seed,role)and authority['qualified']and(mean_authority['seed'],mean_authority['rate_role'])==(seed,role),'literal independently qualified case')
            binding=read(target/'run'/label/'binding.json');need(binding==read(P.TARGET/'run'/label/'binding.json')and binding['eta']==rate and binding['full_state']and binding['original_qualification'],'entire prior checkpoint/token/direction/rate binding')
            if role=='original':
                ref=array(folder/'reference.npy',(17,15776));grad=array(folder/'reference-gradients.npy',(17,15776));ordered=array(folder/'direction-0-transported.npy',(17,15776))[16];frozen=array(folder/'direction-0-frozen.npy',(17,15776))[16];cg=array(folder/'readout-gradients.npy',(3,15776))[2];eps=read(folder/'outcome.json')['epsilon_values']
            else:
                ref=me.alias(label,'reference',(17,15776));grad=me.alias(label,'reference-gradients',(17,15776));ordered=me.alias(label,'ordered',(17,15776))[16];frozen=me.alias(label,'frozen',(17,15776))[16];cg=me.alias(label,'readout-gradients',(3,15776))[2];eps=read(target/'run'/label/'nonlinear-readout.json')['epsilons'];close(eps,[x*max(1.,norm(ref[0]))for x in(1e-4,5e-5)])
            exact(ref[1:],ref[:-1]-rate*grad[:-1],'all actual inherited training states');exact(cg,grad[16]-.001*ref[16],'same full-state CE gradient')
            old=dict(ordered=ordered,mean=me.alias(label,'mean-powers',(17,15776))[16],frozen=frozen,direction=u,ce_gradient=cg)
            for name,value in old.items():exact(ev.alias(label,'baseline-'+name,(15776,)),value,'entire independently qualified baseline '+name)
            responses=[];slopes=[]
            for k,e in enumerate(eps):
                need(math.isfinite(e)and e>0,'positive unchanged nonlinear epsilon');branch={};ce={}
                for side,sign in(('plus',1),('minus',-1)):
                    states=array(folder/f'direction-0-epsilon-{k}-{side}.npy',(17,15776))if role=='original'else me.alias(label,f'epsilon-{k}-{side}',(17,15776))
                    ces=array(folder/f'direction-0-epsilon-{k}-{side}-ce.npy',(3,2))if role=='original'else me.alias(label,f'epsilon-{k}-{side}-ce',(3,2));exact(ces[:,0],np.asarray([1,4,16.]),'literal nonlinear CE horizons');exact(states[0],ref[0]+sign*e*u,'same initial nonlinear direction and amplitude');branch[side]=states[16];ce[side]=ces[2,1]
                responses.append((branch['plus']-branch['minus'])/(2*e));slopes.append(float((ce['plus']-ce['minus'])/(2*e)))
            exact(ev.alias(label,'responses',(2,15776)),np.asarray(responses),'literal two nonlinear full-state responses');close(read(target/'run'/label/'nonlinear-readout.json'),dict(epsilons=eps,slopes=slopes))
            powers=ev.alias(label,'block-mean-powers',(9,15776));maximum=max(maximum,replay_powers(ref,u,rate,powers,lambda j:ev.alias(label,f'power-{j}-block-ends',(8,15776)),ev.call,label,tick))
            row=comparison(powers[-1],old,responses,slopes);base=mean_authority['rows'][-1];need(base['horizon']==16 and base['mechanism_qualified']and base['training_ce_slope_resolved'],'independent inherited h16 qualification')
            for name in('ordered','mean','frozen'):
                for key in('vector_error','gain','signed_alignment','training_ce_slope'):close(row['comparisons'][name][key],base['comparisons'][name][key])
            result=dict(seed=seed,rate_role=role,qualified=bool(row['qualified']and authority['qualified']),comparison=row,inherited_response_qualified=True);saved=read(target/'run'/label/'result.json');need(saved==out['cases'][index],'entire producer case order');close(result,saved);cases.append(result);save(label+'.json',result)
    need(ev.n==512 and ev.counts==dict(hvp=512,gradient=0,ce=0)and len(list((ev.run/'events').glob('*-input.json')))==len(list((ev.run/'events').glob('*-return.json')))==512,'all512 exact saved actions exhausted')
    descriptive=gates(cases);need(descriptive==out['descriptive'],'accuracy and improvement gates retained separately')
    return dict(passed=True,native_calls=0,retained_hvp_checked=512,mean_powers_checked=32,cases=cases,descriptive=descriptive,maximum_mean_rounding_ratio=maximum,scope='Independent fsum/byte-exact action and nonlinear response replay on four consumed retrospective cases. Shared retained native actions; no training intervention, online forecast, or isolation of ordering.')
