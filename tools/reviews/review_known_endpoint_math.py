"""Independent weighted-Neumann enclosure of an already existing root."""
from flint import arb,arb_mat
import review_unit_pair_pal_math as P
from neural_shrimp.rigorous_orbits import ball_json,vector_json
need=P.need;M=P.M;R=P.R

def localize(C,J,G,center,known,whole,weights):
    n=len(center);need(n>0 and len(known)==len(whole)==len(weights)==n,'full existing root')
    need(C.nrows()==C.ncols()==J.nrows()==J.ncols()==G.nrows()==n and G.ncols()==1,'full derivative system')
    need(all(v.is_finite()and v.is_exact()for row in C.tolist()for v in row)and all(v.is_finite()and v.is_exact()for v in center+weights)and all(v>0 for v in weights),'exact finite proposal')
    for i in range(n):
        P.U.contains(whole[i],known[i],'same known root in acquired derivative domain');P.U.contains(whole[i],ball_json(center[i]),'center in acquired mean-value domain')
    defect=M.ident(n)-C*J
    rows=[(sum((abs(defect[i,j]).upper()*weights[j]for j in range(n)),arb(0))/weights[i]).upper()for i in range(n)]
    k=max(rows);need(k.is_finite()and 0<=k<1,'strict weighted Neumann contraction');rhs=-C*G
    size=max((abs(rhs[i,0]).upper()/weights[i]).upper()for i in range(n));radius=(size/(1-k)).upper()
    error=arb_mat([[arb(0,(radius*weights[i]).upper()).intersection(R.original_ball(known[i])-center[i])]for i in range(n)])
    for _ in range(3):
        image=rhs+defect*error
        error=arb_mat([[error[i,0].intersection(image[i,0])]for i in range(n)])
    box=[(center[i]+error[i,0]).intersection(R.original_ball(known[i]))for i in range(n)]
    need(all(v.is_finite()for v in box),'finite retained known root')
    return dict(passed=True,dimension=n,weighted_bound=ball_json(k),box=vector_json(box),scope='Conditional enclosure of the same independently existing root; exact mean-value-domain checks, separately reconstructed matrix, weighted Neumann inequality and three component intersections. No new root existence inferred.')
