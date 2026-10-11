"""Exact-real flip certificates for tanh GD on an exact two-rate log ray.

Third directional AD of the explicit gradient supplies fourth loss derivatives.
All derivative and proof operations use Arb; no floating derivative enters a gate.
The earlier C1 and fold kernels are unchanged.
"""
from __future__ import annotations
from flint import arb, arb_mat, ctx
from .rigorous_orbits import (BallTanhMap, exact, eye, midpoint_matrix,
                             norm_inf_upper, ball_json, vector_json, matrix_json)


class FlipJet:
    """f, base gradient, q-direction derivative/mixed gradient, Dq²f,Dq³f.

    The base variables are (theta,s). q is held constant for all derivatives,
    including when its coefficients are interval enclosures in a proof box.
    No factorials are absorbed into any derivative field.
    """
    __slots__=('f','g','d','m','d2','d3')
    def __init__(self,f,g,d,m,d2=0,d3=0):
        self.f,self.g,self.d,self.m=f,tuple(g),d,tuple(m)
        self.d2,self.d3=arb(d2),arb(d3)
    @classmethod
    def variable(cls,value,size,index,direction=0):
        return cls(arb(value),[arb(int(i==index)) for i in range(size)],
                   arb(direction),[arb(0) for _ in range(size)])
    def __add__(self,other):
        if not isinstance(other,FlipJet):
            return FlipJet(self.f+other,self.g,self.d,self.m,self.d2,self.d3)
        if len(self.g)!=len(other.g):raise ValueError('jet dimensions differ')
        return FlipJet(self.f+other.f,[a+b for a,b in zip(self.g,other.g)],
                       self.d+other.d,[a+b for a,b in zip(self.m,other.m)],
                       self.d2+other.d2,self.d3+other.d3)
    __radd__=__add__
    def __neg__(self):return self*-1
    def __sub__(self,other):return self+(-other)
    def __rsub__(self,other):return -self+other
    def __mul__(self,other):
        if not isinstance(other,FlipJet):
            return FlipJet(self.f*other,[x*other for x in self.g],self.d*other,
                           [x*other for x in self.m],self.d2*other,self.d3*other)
        if len(self.g)!=len(other.g):raise ValueError('jet dimensions differ')
        return FlipJet(self.f*other.f,
            [a*other.f+self.f*b for a,b in zip(self.g,other.g)],
            self.d*other.f+self.f*other.d,
            [a*other.f+b*other.d+self.d*c+self.f*d
             for a,b,c,d in zip(self.m,self.g,other.g,other.m)],
            self.d2*other.f+2*self.d*other.d+self.f*other.d2,
            self.d3*other.f+3*self.d2*other.d+3*self.d*other.d2+self.f*other.d3)
    __rmul__=__mul__
    def unary(self,value,first,second,third):
        return FlipJet(value,[first*x for x in self.g],first*self.d,
            [first*m+second*g*self.d for m,g in zip(self.m,self.g)],
            first*self.d2+second*self.d*self.d,
            first*self.d3+3*second*self.d*self.d2+third*self.d*self.d*self.d)
    def tanh(self):
        t=self.f.tanh();s=1-t*t
        return self.unary(t,s,-2*t*s,s*(6*t*t-2))
    def exp(self):
        e=self.f.exp();return self.unary(e,e,e,e)


class FlipTanhMap(BallTanhMap):
    """Both layer rates are exp(log(10)*(ell_i+d_i*s)).

    ell_i,d_i are exact supplied rationals. The historical adapter must bind
    their binary64 origin; exp/log are real Arb enclosures, not binary64 powers.
    The base map's positive placeholder rates are unused by this jet update.
    """
    def __init__(self,inputs,family):
        super().__init__(inputs)
        if family.get('schema')!='neural-shrimp-exact-log-ray-v1':
            raise ValueError('unknown exact rate-family schema')
        if len(family['ell'])!=2 or len(family['direction'])!=2:
            raise ValueError('rate family dimension')
        self.ell=[exact(x) for x in family['ell']]
        self.direction=[exact(x) for x in family['direction']]
        if not all(x.is_finite() for x in self.ell+self.direction):
            raise ValueError('finite rate family required')
        if not any(not x.contains(0) for x in self.direction):
            raise ValueError('nonconstant ray required')
        self.log_ten=arb(10).log()
    def rates_at(self,s):
        return [((ell+d*s)*self.log_ten).exp() for ell,d in zip(self.ell,self.direction)]
    def jet_step(self,theta,s):
        rates=self.rates_at(s)
        grad=[t*self.ridge for t in theta]
        for x,y in zip(self.x,self.y):
            acts,features=[],[]
            for h in range(self.w):
                feature=[(d*self.w+h,self.ins*x[d]) for d in range(self.d)]
                if self.bias:feature.append((self.a+h,arb(1)))
                acts.append(sum((theta[i]*v for i,v in feature),0).tanh())
                features.append(feature)
            prediction=sum((theta[self.hidden+h]*acts[h] for h in range(self.w)),0)*self.out
            if self.bias:prediction+=theta[-1]
            residual=prediction-y
            for h in range(self.w):
                oi=self.hidden+h
                grad[oi]+=residual*acts[h]*(self.k*self.out)
                hidden=residual*theta[oi]*(1-acts[h]*acts[h])*(self.k*self.out)
                for i,feature in features[h]:grad[i]+=hidden*feature
            if self.bias:grad[-1]+=residual*self.k
        return [t-grad[i]*rates[int(i>=self.hidden)] for i,t in enumerate(theta)]


def augmented(model,z,period,tick=lambda:None):
    """H=(G-x,(M+I)q,(q^Tq-1)/2), z=(x,s,q); G=F_s^period."""
    n=model.n
    if len(z)!=2*n+1:raise ValueError('augmented dimension mismatch')
    theta,s,q=z[:n],z[n],z[n+1:]
    current=[FlipJet.variable(theta[i],n+1,i,q[i]) for i in range(n)]
    parameter=FlipJet.variable(s,n+1,n)
    initial=current;orbit=[[v.f for v in current]]
    for _ in range(period):
        tick();current=model.jet_step(current,parameter);orbit.append([v.f for v in current])
    residual=[a-b for a,b in zip(current,initial)]
    m=arb_mat([[v.g[j] for j in range(n)] for v in current]);a=m-eye(n);k=m+eye(n)
    h=[v.f for v in residual]+[v.d+q[i] for i,v in enumerate(current)]+[(sum((v*v for v in q),arb(0))-1)/2]
    jac=arb_mat([list(v.g)+[arb(0)]*n for v in residual]
        +[list(v.m)+[k[i,j] for j in range(n)] for i,v in enumerate(current)]
        +[[arb(0)]*(n+1)+list(q)])
    return h,jac,{'a':a,'k':k,'monodromy':m,'q':q,'orbit':orbit,
        'parameter_derivative':[v.g[n] for v in current],
        'mixed_directional':arb_mat([list(v.m) for v in current]),
        'second_directional':[v.d2 for v in current],
        'third_directional':[v.d3 for v in current]}


def validated_solve(matrix,rhs):
    """Arb solve plus an explicit whole-matrix Neumann inverse witness."""
    pre=midpoint_matrix(matrix.inv());defect=eye(matrix.nrows())-pre*matrix
    bound=norm_inf_upper(defect)
    answer=matrix.solve(rhs)
    return answer,{'preconditioner':matrix_json(pre),'defect':matrix_json(defect),
        'norm_inf_upper':ball_json(bound),'passed':bool(bound.is_finite() and bound<1)}


def _proof(model,approximation,period,radius_exponent,newton_steps,max_power,
           tick=lambda:None,observe=lambda kind,record:None):
    center=[x.mid() for x in approximation];n=model.n
    for iteration in range(newton_steps):
        h,j,_=augmented(model,center,period,tick)
        observe('newton_evaluation',{'iteration':iteration,'center':vector_json(center),
                                     'h':vector_json(h),'jacobian':matrix_json(j)})
        delta=j.solve(arb_mat([[x] for x in h]));center=[(x-delta[i,0]).mid() for i,x in enumerate(center)]
    h0,j0,_=augmented(model,center,period,tick);pre=midpoint_matrix(j0.inv())
    radius=arb((1,radius_exponent));box=[arb(x,radius) for x in center]
    hb,jb,data=augmented(model,box,period,tick)
    observe('whole_box_evaluation',{'box':vector_json(box),'h':vector_json(hb),'jacobian':matrix_json(jb)})
    defect=eye(2*n+1)-pre*jb;contraction=norm_inf_upper(defect)
    image_delta=-pre*arb_mat([[x] for x in h0])+defect*arb_mat([[box[i]-center[i]] for i in range(2*n+1)])
    image=[center[i]+image_delta[i,0] for i in range(2*n+1)]
    inclusion=[bool(b.contains_interior(k)) for b,k in zip(box,image)]
    root=bool(all(inclusion) and contraction.is_finite() and contraction<1)
    divisors=[]
    for d in range(1,period):
        if period%d==0:
            diffs=[data['orbit'][d][i]-box[i] for i in range(n)]
            coords=[i for i,v in enumerate(diffs) if v.is_finite() and not v.contains(0)]
            divisors.append({'divisor':d,'excluded':bool(coords),'separating_coordinates':coords,'differences':vector_json(diffs)})
    result={'schema':'neural-shrimp-flip-result-v1','passed':False,'evidence_rung':0,
        'status':'augmented_root_only' if root else 'root_unproved',
        'settings':{'radius_exponent':radius_exponent,'newton_steps':newton_steps,'max_power':max_power},
        'gates':{'augmented_root_unique':root,'primitive_period':all(d['excluded'] for d in divisors),
                 'simple_minus_one_multiplier':False,'regular_orbit_branch':False,
                 'total_multiplier_crossing':False,'cubic_nondegeneracy':False,'other_modes_attracting':False},
        'center':vector_json(center),'box':vector_json(box),'h_center':vector_json(h0),'h_box':vector_json(hb),
        'jacobian_box':matrix_json(jb),'preconditioner':matrix_json(pre),'defect':matrix_json(defect),
        'contraction_norm_upper':ball_json(contraction),'krawczyk_image':vector_json(image),
        'coordinate_inclusions':inclusion,'divisors':divisors,
        'orbit_enclosures':[vector_json(x) for x in data['orbit']],
        'monodromy':matrix_json(data['monodromy']),'residual_jacobian':matrix_json(data['a']),
        'flip_null_matrix':matrix_json(data['k']),'right_vector':vector_json(data['q']),
        'parameter_derivative':vector_json(data['parameter_derivative']),
        'mixed_directional':matrix_json(data['mixed_directional']),
        'second_directional':vector_json(data['second_directional']),
        'third_directional':vector_json(data['third_directional']),'failures':[]}
    observe('root_gate',result)
    try:
        tick();q=data['q'];k=data['k'];a=data['a']
        border=arb_mat([[k[j,i] for j in range(n)]+[q[i]] for i in range(n)]+[list(q)+[arb(0)]])
        left,left_check=validated_solve(border,arb_mat([[arb(0)]]*n+[[arb(1)]]))
        p=[left[i,0] for i in range(n)]
        rhs=arb_mat([[-data['parameter_derivative'][i],data['second_directional'][i]] for i in range(n)])
        solved,branch_check=validated_solve(a,rhs)
        theta_s=[solved[i,0] for i in range(n)];u=[solved[i,1] for i in range(n)]
        mixed=data['mixed_directional']
        total_action=[mixed[i,n]+sum((mixed[i,j]*theta_s[j] for j in range(n)),arb(0)) for i in range(n)]
        bqu=[sum((mixed[i,j]*u[j] for j in range(n)),arb(0)) for i in range(n)]
        alpha=sum((p[i]*total_action[i] for i in range(n)),arb(0))
        cubic=sum((p[i]*data['third_directional'][i] for i in range(n)),arb(0))/6-sum((p[i]*bqu[i] for i in range(n)),arb(0))/2
        crossing=bool(alpha.is_finite() and not alpha.contains(0));nondeg=bool(cubic.is_finite() and not cubic.contains(0))
        deflated=data['monodromy']+arb_mat([[q[i]*p[j] for j in range(n)] for i in range(n)])
        power,exponent,powers,attracting=deflated,1,[],False
        while exponent<=max_power:
            tick();bound=norm_inf_upper(power);powers.append({'power':exponent,'norm_inf_upper':ball_json(bound)})
            if bound.is_finite() and bound<1:attracting=True;break
            if not bound.is_finite():break
            power=power*power;exponent*=2
        simple=bool(root and left_check['passed']);regular=bool(root and branch_check['passed'])
        result.update({'left_border':matrix_json(border),'left_inverse_check':left_check,
            'left_vector':vector_json(p),'left_border_multiplier':ball_json(left[n,0]),
            'orbit_branch_inverse_check':branch_check,'orbit_branch_derivative':vector_json(theta_s),
            'quadratic_resolvent':vector_json(u),'total_multiplier_action':vector_json(total_action),
            'b_q_resolvent':vector_json(bqu),'alpha':ball_json(alpha),'cubic_b':ball_json(cubic),
            'deflated_monodromy':matrix_json(deflated),'power_bounds':powers,
            'complement_witness_power':exponent if attracting else None,
            'complement_witness':matrix_json(power) if attracting else None})
        result['gates'].update({'simple_minus_one_multiplier':simple,'regular_orbit_branch':regular,
            'total_multiplier_crossing':bool(simple and regular and crossing),
            'cubic_nondegeneracy':bool(simple and regular and nondeg),
            'other_modes_attracting':bool(simple and attracting)})
        result['cubic_sign']='positive' if nondeg and cubic>0 else 'negative' if nondeg and cubic<0 else 'unresolved'
        result['criticality']='supercritical' if nondeg and cubic>0 else 'subcritical' if nondeg and cubic<0 else 'unresolved'
        result['doubled_side_in_s']='greater' if crossing and nondeg and -alpha/cubic>0 else 'less' if crossing and nondeg and -alpha/cubic<0 else 'unresolved'
    except (ValueError,ZeroDivisionError) as exc:
        result['failures'].append({'stage':'flip_gates','type':type(exc).__name__,'message':str(exc)})
    result['passed']=all(result['gates'].values());result['evidence_rung']=3 if result['passed'] else 0
    if result['passed']:result['status']='primitive_nondegenerate_flip_with_attracting_complement'
    else:
        result['criticality']='unproved'
        result['doubled_side_in_s']='unproved'
    result['exact_statement']={'period':period,'dimension':n,'augmented_dimension':2*n+1,
        'parameter':'exact-real historical two-rate log-ray coordinate s; both rates differentiated',
        'normalization':'q^Tq=1, p^Tq=1 and Mq=-q at enclosed root',
        'alpha':'total derivative of the simple multiplier along the local period-p orbit branch',
        'cubic_b':'p^T C(q,q,q)/6 - p^T B(q,(M-I)^(-1)B(q,q))/2',
        'local_consequence':'If all gates pass and b>0: attracting primitive2p cycles for sufficiently small detuning with -alpha*(s-s*)>0. If b<0: subcritical unstable doubled cycles on the opposite side.',
        'not_claimed':['attraction at the neutral flip','a stated finite-detuning period2p orbit','connection to the separately certified dyadic center','global branch identity','whole-window geometry','training utility']}
    return result


def check_flip(inputs,*,precision=256,radius_exponent=-100,newton_steps=5,max_power=4096,
               tick=lambda:None,observe=lambda kind,record:None):
    if inputs.get('schema')!='neural-shrimp-flip-input-v1':raise ValueError('unknown flip input schema')
    for value,lo,hi in ((precision,128,2048),(radius_exponent,-1024,-1),(newton_steps,0,12),(max_power,1,4096),(inputs['period'],1,64)):
        if type(value)is not int or not lo<=value<=hi:raise ValueError('invalid bounded flip settings')
    with ctx.workprec(precision):
        model=FlipTanhMap(inputs['map_inputs'],inputs['rate_family'])
        proposal=[exact(x) for x in inputs['approximation']]
        if len(proposal)!=2*model.n+1 or not all(x.is_finite() for x in proposal):raise ValueError('invalid full flip proposal')
        result=_proof(model,proposal,inputs['period'],radius_exponent,newton_steps,max_power,tick,observe)
        result['settings']['precision']=precision
        rates=model.rates_at(stored_parameter(result['box'][model.n]))
        result['rate_enclosures']=vector_json(rates)
        result['gates']['positive_finite_rates']=all(x.is_finite() and x>0 for x in rates)
        result['passed']=all(result['gates'].values());result['evidence_rung']=3 if result['passed'] else 0
        if not result['passed'] and result['status']=='primitive_nondegenerate_flip_with_attracting_complement':
            result['status']='rate_family_unproved'
            result['criticality']='unproved'
            result['doubled_side_in_s']='unproved'
        return result


def stored_parameter(value):
    return arb(arb(tuple(value['mid'])),arb(tuple(value['rad'])))
