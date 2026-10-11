"""Known adjoint containment at actual locus roots; no new inverse or model.

alpha=0 is a theorem at those roots, not at all independent points of B(X).
A valid initial solution enclosure and the saved uniform border proof are
external premises. All iterations use the same valid whole-domain defect.
"""
from flint import arb,arb_mat
from .rigorous_shooting_adjoint import need,finite
from .rigorous_orbits import vector_json,matrix_json,ball_json


def refine(solution,proposal,forcing,defect,observe=lambda kind,value:None,iterations=5):
    k=len(solution)
    need(type(iterations)is int and iterations==5,'fixed five intersections')
    need(2<=k<=235 and proposal.nrows()==forcing.nrows()==k and proposal.ncols()==forcing.ncols()==1 and defect.nrows()==defect.ncols()==k,'refinement dimensions')
    need(all(v.is_finite() for v in solution) and finite(proposal) and finite(forcing) and finite(defect),'finite saved inputs')
    need(all(proposal[i,0].is_exact() for i in range(k)),'fixed exact solve proposal')
    need(solution[-1].contains(0),'initial known-alpha containment')
    error=arb_mat([[solution[i]-proposal[i,0]] for i in range(k)])
    error[k-1,0]=-proposal[k-1,0]
    observe('conditional_start',{'initial_solution':vector_json(solution),'proposal':matrix_json(proposal),'forcing':matrix_json(forcing),'defect':matrix_json(defect),'error':matrix_json(error),'scope':'alpha=0 restricted only at the authenticated exact locus roots'})
    for it in range(iterations):
        image=forcing+defect*error
        observe('image_return',{'iteration':it,'prior_error':matrix_json(error),'image':matrix_json(image)})
        need(finite(image),'finite returned image')
        updated=arb_mat([[error[i,0].intersection(image[i,0])] for i in range(k)])
        need(finite(updated),'finite nonempty intersections')
        need(updated[k-1,0].contains(-proposal[k-1,0]),'exact alpha remains compatible')
        updated[k-1,0]=-proposal[k-1,0]
        observe('intersection_return',{'iteration':it,'error':matrix_json(updated)})
        error=updated
    result=[proposal[i,0]+error[i,0] for i in range(k)];result[-1]=arb(0)
    need(all(v.is_finite() for v in result),'finite refined solution')
    observe('refined_solution_return',{'solution':vector_json(result)})
    return result


def coefficients(solution,rates,second,n):
    k=len(solution)-1
    need(type(n)is int and 1<=n<=13 and k%n==0 and rates.nrows()==k and rates.ncols()==2 and len(second)==k,'coefficient dimensions')
    need(all(v.is_finite() for v in solution+second) and finite(rates),'finite coefficients input')
    need(solution[-1].is_exact() and solution[-1]==0,'exact-root alpha restriction')
    w=solution[:-1];aa=[sum((w[i]*rates[i,j] for i in range(k)),arb(0)) for j in range(2)];cc=sum((w[i]*second[i] for i in range(k)),arb(0))
    need(all(v.is_finite() for v in aa+[cc]),'finite coefficient output')
    gates={'nonzero_rate_transversality':any(not v.contains(0) for v in aa),'nonzero_quadratic':not cc.contains(0)}
    return {'passed':all(gates.values()),'gates':gates,'actual_rate_coefficients':vector_json(aa),'second_derivative_contraction':ball_json(cc),'quadratic_coefficient':ball_json(cc/2),'left_native':vector_json(w[-n:]),'left_shooting':vector_json(w),
     'scope':'Root-conditioned adjoint bounds. Inherits authenticated parent root and prior uniform border invertibility/simple+1. No other unit-circle exclusion, attraction, complete dynamical fold, cusp or geometry claim.'}
