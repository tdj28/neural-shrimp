"""Independent exponential-form native RNN map and complete local proof reduction."""
from fractions import Fraction
from flint import arb,arb_mat,ctx
from neural_shrimp.rigorous_orbits import ball_json,matrix_json,vector_json

def need(ok,why):
    if not ok:raise ValueError(why)
def pair(v):
    need(isinstance(v,list)and len(v)==2 and all(type(x)is int for x in v),'exact dyadic pair')
    return Fraction(v[0])*Fraction(2)**v[1]
def endpoints(v):
    m,r=pair(v['mid']),pair(v['rad']);need(r>=0,'radius');return m-r,m+r
def exact(v):
    q=Fraction(v);return arb(q.numerator)/q.denominator
def restore(v):
    endpoints(v);m=arb(tuple(v['mid']));r,e=v['rad']
    need(r.bit_length()<=30,'supported saved magnitude')
    a=arb(m,arb((r,e))-arb((1,e+r.bit_length()-32)))if r else m
    b=ball_json(a);need(b['mid']==v['mid']and b['rad']==v['rad'],'exact saved ball reconstruction');return a
def eye(n):return arb_mat([[int(i==j)for j in range(n)]for i in range(n)])
def norm(a):return max(sum((abs(a[i,j]).upper()for j in range(a.ncols())),arb(0)).upper()for i in range(a.nrows()))

class ExpRNN:
    """No producer step/iterate/derivative function; tanh via exp(2x)."""
    def __init__(self,inputs):
        self.n=len(inputs['j0']);self.j=arb_mat([[exact(v)for v in row]for row in inputs['j0']]);self.g=exact(inputs['gain']);self.d=exact(inputs['delta'])
        need(self.j.ncols()==self.n and self.g>0 and 0<self.d<=1,'native model')
    def step(self,x):
        need(len(x)==self.n,'native state dimension');e=[(2*v).exp()for v in x];t=[(v-1)/(v+1)for v in e];s=[4*v/((v+1)*(v+1))for v in e]
        y=arb_mat([[v]for v in x])*(1-self.d)+self.j*arb_mat([[v]for v in t])*(self.g*self.d)
        A=arb_mat([[self.j[i,k]*s[k]*self.d*self.g+int(i==k)*(1-self.d)for k in range(self.n)]for i in range(self.n)])
        return [y[i,0]for i in range(self.n)],A
    def iterate(self,x,p):
        M=eye(self.n);orbit=[x];blocks=[]
        for _ in range(p):
            x,A=self.step(x);orbit.append(x);blocks.append(A);M=A*M
        return x,M,orbit,blocks

def strict_nested(outer,inner):
    need(len(outer)==len(inner),'nested dimensions')
    return all(a<c and d<b for a,b,c,d in [(*endpoints(o),*endpoints(i))for o,i in zip(outer,inner)])

def independent_proof(inputs,proposal,precision=384):
    """Prove on the supplied exact nested box, with independently evaluated map."""
    with ctx.workprec(precision):
        model=ExpRNN(inputs);n=model.n;p=inputs['period'];need(type(p)is int and 1<=p<=64,'period')
        center=[restore(v)for v in proposal['center']];box=[restore(v)for v in proposal['box']]
        need(all(v.is_exact()for v in center),'exact center');need(all(c==b.mid()for c,b in zip(center,box)),'box centered at exact center')
        C=arb_mat([[restore(v)for v in row]for row in proposal['preconditioner']]);need(all(C[i,j].is_exact()for i in range(n)for j in range(n)),'exact preconditioner')
        y,_,_,_=model.iterate(center,p);R=arb_mat([[y[i]-center[i]]for i in range(n)])
        _,M,orbit,js=model.iterate(box,p);D=eye(n)-C*(M-eye(n));E=arb_mat([[restore({'mid':[0,0],'rad':v['rad']})]for v in proposal['box']])
        K=[center[i]+(-C*R+D*E)[i,0]for i in range(n)];contraction=norm(D)
        inclusion=[box[i].contains_interior(K[i])for i in range(n)];existence=all(inclusion)and contraction<1
        divisors=[]
        for d in range(1,p):
            if p%d==0:divisors.append({'divisor':d,'separating_coordinates':[i for i in range(n)if not(orbit[d][i]-box[i]).contains(0)]})
        odd=[{'phase':k,'separating_coordinates':[i for i in range(n)if not(orbit[k][i]+box[i]).contains(0)]}for k in range(p)]
        power=M;rows=[];k=1;attracting=False
        while k<=512:
            b=norm(power);rows.append({'power':k,'norm_upper':ball_json(b)})
            if b<1:attracting=True;break
            power=power*power;k*=2
        gates={'existence_and_local_uniqueness':bool(existence),'primitive':all(row['separating_coordinates']for row in divisors),'attraction':bool(attracting),'odd_companion_all_phase_distinct':all(row['separating_coordinates']for row in odd)}
        return {'passed':all(gates.values()),'gates':gates,'precision':precision,'box':proposal['box'],'contraction':ball_json(contraction),'krawczyk':vector_json(K),'inclusions':list(map(bool,inclusion)),'divisors':divisors,'odd_checks':odd,'power_bounds':rows,'attraction_power':k if attracting else None,'monodromy':matrix_json(M),'native_jacobians':[matrix_json(x)for x in js],'native_orbit':[vector_json(x)for x in orbit],'residual':matrix_json(R),'defect':matrix_json(D),'scope':'Independent exp-form native enclosure and proof on a fixed supplied box; exact map only, no window/boundary theorem'}
