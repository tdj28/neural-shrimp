"""Conditional localization of one already proved full-state endpoint."""
from flint import arb,arb_mat
from .rigorous_parameter_localization import fixed_point_box
from .rigorous_orbits import eye,ball_json,vector_json,matrix_json
from .rigorous_eigenpair_locus import require

def localize(C,J,G,center,known,whole,weights):
 n=len(center);require(n>0 and len(known)==len(whole)==len(weights)==n,'complete known root')
 require(C.nrows()==C.ncols()==J.nrows()==J.ncols()==G.nrows()==n and G.ncols()==1,'square same-root derivative blocks')
 require(all(v.is_finite()and v.is_exact()for row in C.tolist()for v in row)and all(v.is_finite()and v.is_exact()for v in center+weights)and all(w>0 for w in weights),'exact finite preconditioner/center/weights')
 for i in range(n):require(whole[i].contains(center[i])and whole[i].contains(known[i]),'same known-root mean-value segment domain')
 D=eye(n)-C*J;bounds=[(sum((abs(D[i,j]).upper()*weights[j]for j in range(n)),arb(0))/weights[i]).upper()for i in range(n)];k=max(bounds)
 require(k.is_finite()and 0<=k<1,'full weighted known-root Neumann bound')
 v=-C*G;initial=arb_mat([[known[i]-center[i]]for i in range(n)])
 error,proof=fixed_point_box(D,v,weights,k,initial=initial,iterations=3)
 box=[center[i]+error[i,0]for i in range(n)]
 # The intersection remains the same known root, never an existence assertion.
 box=[box[i].intersection(known[i])for i in range(n)]
 return {'passed':True,'dimension':n,'weighted_bound':ball_json(k),'row_bounds':vector_json(bounds),'rhs':matrix_json(v),'refinement':proof,'center':vector_json(center),'box':vector_json(box),'scope':'Conditional localization of exactly the previously proved root in its authenticated domain; no new root or interval is introduced.'}
