"""Mean-value gradient enclosure for the unchanged exact-real GD objective."""
from flint import arb,arb_mat
from .rigorous_orbits import eye
from .rigorous_connection import HiddenRateMap,require


def mean_value_gradient(theta,evaluate_unit):
    """g(X) subset g(z)+H(X)(X-z), z=mid(X), H=I-DF_unit(X)."""
    require(bool(theta) and all(v.is_finite() for v in theta),'finite state box')
    _,unit_jac=evaluate_unit(theta)
    center=[v.mid() for v in theta];image,_=evaluate_unit(center)
    hessian=eye(len(theta))-unit_jac
    offset=arb_mat([[theta[i]-center[i]] for i in range(len(theta))])
    variation=hessian*offset
    gradient=[center[i]-image[i]+variation[i,0] for i in range(len(theta))]
    return gradient,hessian


class MeanValueHiddenRateMap(HiddenRateMap):
    def __init__(self,inputs,max_calls=180):
        super().__init__(inputs)
        require(type(max_calls) is int and 1<=max_calls<=180,'bounded map calls')
        self.max_calls=max_calls;self.map_attempted=0;self.map_returned=0
        self.unit_attempted=0;self.unit_returned=0

    def evaluate_unit(self,theta):
        require(self.unit_attempted<2*self.max_calls,'bounded unit derivatives')
        self.unit_attempted+=1
        result=self.unit.step(theta);self.unit_returned+=1
        return result

    def step(self,theta,hidden_rate):
        require(hidden_rate.is_finite() and hidden_rate>0,'positive rate interval')
        require(len(theta)==self.n and all(v.is_finite() for v in theta),'finite native state')
        require(self.map_attempted<self.max_calls,'bounded map evaluations')
        self.map_attempted+=1
        gradient,hessian=mean_value_gradient(theta,self.evaluate_unit)
        rates=[hidden_rate]*self.hidden+[self.output]*(self.n-self.hidden)
        image=[theta[i]-rates[i]*gradient[i] for i in range(self.n)]
        jac=arb_mat([[int(i==j)-rates[i]*hessian[i,j] for j in range(self.n)] for i in range(self.n)])
        partial=[-gradient[i] if i<self.hidden else arb(0) for i in range(self.n)]
        self.map_returned+=1
        return image,jac,partial

    def counts(self):
        return {'map_attempted':self.map_attempted,'map_returned':self.map_returned,
                'unit_attempted':self.unit_attempted,'unit_returned':self.unit_returned,'map_cap':self.max_calls,'unit_cap':2*self.max_calls}
