"""Source-bound replay and point operator for a known cubic705 endpoint.

This is a producer reduction, not an independent audit. The same reviewed
native assembly is fed only authenticated saved returns after exact rational
input-domain coverage checks. The caller binds source, map, original proof
and known endpoint; no derivative or new existence is acquired here.
"""
from fractions import Fraction as F
from flint import arb,arb_mat
from .rigorous_cubic_gradient_path import cubic_jacobian_coefficients
from .rigorous_gradient_path import GradientPathJet,PathJet
from .rigorous_eigenpair_locus import require,finite
from .rigorous_orbits import ball_json,eye,midpoint_matrix


def interval(v):
    require(type(v)is dict and {'mid','rad'}<=set(v),'serialized exact dyadic ball')
    m,e=v['mid'];r,f=v['rad'];require(all(type(k)is int for k in(m,e,r,f))and r>=0 and abs(e)<10000 and abs(f)<10000,'finite dyadic encoding')
    c=F(m)*F(2)**e;radius=F(r)*F(2)**f;return c-radius,c+radius


def original_ball(v):
    # Same exact magnitude-restoration recipe as the reviewed original_ball
    # in tools/reviews/review_quadratic_connection_math.py. Equality is checked.
    interval(v);mid=arb(tuple(v['mid']));m,e=v['rad']
    if m:
        require(m.bit_length()<=30,'saved magnitude precision')
        out=arb(mid,arb((m,e))-arb((1,e+m.bit_length()-32)))
    else:out=mid
    got=ball_json(out);require(got['mid']==v['mid']and got['rad']==v['rad'],'literal original ball restored');return out


def covered(old,new):
    if type(old)is dict and {'mid','rad'}<=set(old):
        a,b=interval(old);c,d=interval(new);require(a<=c<=d<=b,'every actual saved native input covers replay');return
    if type(old)is dict:
        require(type(new)is dict and set(old)==set(new),'same native input fields')
        for k in old:covered(old[k],new[k])
    elif type(old)is list:
        require(type(new)is list and len(old)==len(new),'same complete native input shape')
        for a,b in zip(old,new,strict=True):covered(a,b)
    else:require(old==new,'same native input metadata')


def as_path(v):
    require(type(v)is list and len(v)==2 and all(type(row)is list and len(row)==4 for row in v),'complete returned path')
    return PathJet([original_ball(x)for x in v[0]],[original_ball(x)for x in v[1]])


class SavedGradientModel:
    def __init__(self,n,period,records):
        require(2<=n<=13 and 1<=period<=18 and len(records)==2*period,'complete bounded gradient acquisition');self.n=n;self.period=period;self.records=records;self.index=0;self.input_seen=False
    def observe(self,kind,v):
        if kind=='gradient_path_input':
            require(not self.input_seen and self.index<len(self.records),'one input before each saved return');old=self.records[self.index]['input'];covered(old,v);self.input_seen=True
        elif kind=='gradient_path_return':
            require(self.index>0,'return follows exact input');old=self.records[self.index-1]['return'];covered(old,v)
        else:require(kind=='gradient_path_assembly_return'and self.index==2*self.period,'only complete final assembly')
    def cusp_step(self,x,*rates):
        require(self.input_seen and self.index<len(self.records),'replay cannot call an unrecorded phase');row=self.records[self.index];phase=self.index//2;name='q'if self.index%2==0 else'r'
        require(row['input']['phase']==row['return']['phase']==phase and row['input']['direction']==row['return']['direction']==name,'literal native phase and mode order')
        covered(row['input']['inputs'],[v.serialized()for v in list(x)+list(rates)]);ret=row['return']['returns'];require(len(ret)==self.n,'all native returned coordinates');answer=[]
        for v in ret:
            require(set(v)=={'value','gradient'}and len(v['gradient'])==self.n+2,'full held-mode base gradients')
            answer.append(GradientPathJet(as_path(v['value']),[as_path(g)for g in v['gradient']]))
        self.index+=1;self.input_seen=False;return answer


def replay_jacobian(center,slope,curvature,cubic,error,d,period,normal,records,*,tick=lambda:None):
    n=(len(center)-3)//(3*period);model=SavedGradientModel(n,period,records)
    out=cubic_jacobian_coefficients(model,center,slope,curvature,cubic,error,d,period,normal,tick=tick,observe=model.observe,retain_assembly=False)
    require(model.index==2*period and not model.input_seen,'all retained derivatives used exactly once');return out


def endpoint_operator(J,C0,C1,d,weights,*,tick=lambda:None):
    """Enclose I-C*H_U(p(d)+E,s+d) at an exact endpoint.

    J=(J0(E),J1(E),J2(I,E)) must cover the whole ORIGINAL correction E.
    The exact midpoint of C0+C1*d is only a finite preconditioner proposal;
    the fresh full interval product establishes its actual defect. This
    operator also supports a saved-only true tangent Neumann enclosure.
    """
    n=len(weights);require(1<=n<=705 and len(J)==3 and all(v.nrows()==v.ncols()==n and finite(v)for v in J+[C0,C1]),'full finite endpoint matrix polynomial')
    require(d.is_finite()and d.is_exact()and all(w.is_finite()and w.is_exact()and w>0 for w in weights),'exact endpoint and positive weights')
    C=midpoint_matrix(C0+C1*d);pointJ=J[0]+J[1]*d+J[2]*(d*d);D=eye(n)-C*pointJ;tick()
    q=max((sum((abs(D[i,j]).upper()*weights[j]for j in range(n)),arb(0))/weights[i]).upper()for i in range(n));require(q.is_finite()and q<1,'strict saved full-error endpoint contraction')
    return C,D,q
