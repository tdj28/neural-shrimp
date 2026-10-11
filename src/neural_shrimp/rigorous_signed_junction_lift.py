"""Conditional saved native lift of the same known simple-minus-one mode.

Caller binds the independently proved 4n+3 junction root, its conditional
refinement, the complete native (x,q,r) lift and exact source/input/return
archive. No derivative or new root is acquired by this helper.
"""
from flint import arb,arb_mat
from neural_shrimp.rigorous_eigenpair_locus import require,finite,column
from neural_shrimp.rigorous_orbits import vector_json,ball_json,matrix_json
from neural_shrimp.rigorous_saved_cubic_endpoint import original_ball


def identical(a,b):
    """Literal interval identity, ignoring human display formatting."""
    if isinstance(a,dict)and {'mid','rad'}<=set(a):
        return isinstance(b,dict)and a['mid']==b.get('mid')and a['rad']==b.get('rad')
    if isinstance(a,list):return isinstance(b,list)and len(a)==len(b)and all(identical(x,y)for x,y in zip(a,b))
    return a==b


def saved_flip_lift(root_box,unit_box,returns,*,n,period,tick=lambda:None,observe=lambda k,v:None):
    """Pack all (x_i,w_i), rates using DF from literal saved q/r passes.

    MixedJet base gradients DF do not depend on the held directional seed.
    Both saved direction passes must have identical state inputs and base DF.
    Use the q-pass value for state transport, as in the original native lift.
    Existing same-root Mw=-w and norm(w)=1, not zero-containment below, prove
    the signed closure. The sanity gates detect corruption and convention bugs.
    """
    require(type(n)is int and 2<=n<=13 and type(period)is int and 1<=period<=18,'bounded native lift')
    require(len(root_box)==4*n+3 and len(unit_box)==3*n*period+3 and len(returns)==2*period,'entire same-root arrays')
    require(identical(unit_box[:3*n],root_box[:n]+root_box[n+2:3*n+2])and identical(unit_box[-3:],root_box[n:n+2]+[root_box[3*n+2]]),'same refined phase-zero root and actual rates/angle')
    rates=root_box[n:n+2];w=[original_ball(v)for v in root_box[3*n+3:]];w0=list(w);flat=[];proof=[];last=None
    require(len(w)==n and all(v.is_finite()for v in w),'same whole normalized minus-one mode')
    for phase in range(period):
        block=unit_box[3*n*phase:3*n*(phase+1)];state,q,r=block[:n],block[n:2*n],block[2*n:]
        qr,rr=returns[2*phase:2*phase+2]
        for name,direction,record in [('q',q,qr),('r',r,rr)]:
            require(record['phase']==phase and record['direction']==name and identical(record['input'],state+direction+rates),'literal saved phase/direction/state/rate input')
            require(len(record['value'])==len(record['block'])==2*n and all(len(row)==1 for row in record['value'])and all(len(row)==2*n for row in record['block']),'complete native return shape')
        Aq=[row[:n]for row in qr['block'][:n]];Ar=[row[:n]for row in rr['block'][:n]]
        require(identical(Aq,Ar)and identical(qr['value'][:n],rr['value'][:n]),'direction-independent same state value and DF')
        nextstate=[row[0]for row in qr['value'][:n]]
        if phase+1<period:require(identical(nextstate,unit_box[3*n*(phase+1):3*n*(phase+1)+n]),'literal retained complete next-state domain')
        A=arb_mat([[original_ball(v)for v in row]for row in Aq]);require(finite(A),'finite actual whole-state derivative')
        point=[original_ball(v)for v in state];flat.extend(point+w)
        observe('saved_signed_lift_input',dict(phase=phase,state=state,mode=vector_json(w),rates=rates,derivative=Aq));tick()
        image=A*column(w);require(finite(image),'finite full-mode propagation');w=[image[i,0]for i in range(n)];last=nextstate
        row=dict(phase=phase,mode_return=vector_json(w));observe('saved_signed_lift_return',row);proof.append(row)
    flat.extend(original_ball(v)for v in rates)
    mode_close=[w[i]+w0[i]for i in range(n)];state_close=[original_ball(last[i])-original_ball(unit_box[i])for i in range(n)];norm=sum((v*v for v in w0),arb(0))-1
    return flat,dict(dimension=n,period=period,sigma=-1,shooting_dimension=len(flat),box=vector_json(flat),state_closure=vector_json(state_close),mode_closure=vector_json(mode_close),normalization=ball_json(norm),sanity=dict(state_closed=all(v.contains(0)for v in state_close),mode_closed=all(v.contains(0)for v in mode_close),normalized=norm.contains(0)),saved_phase_blocks=period,native_calls=0,scope='Conditional exact signed native lift of an existing proved junction root. Same-root Mw=-w and normalization are inherited from authenticated55 proof; displayed zero-containment alone proves neither existence nor boundary connection.')
