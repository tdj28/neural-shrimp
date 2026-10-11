"""Independent endpoint tightening and qualitative interior rate minimum.

Caller supplies the existing root and authenticated full-error native Jacobian
and exact-point residual. No producer endpoint reduction is imported.
"""
from fractions import Fraction as F
from flint import arb,arb_mat
import review_cubic705_math as H
from neural_shrimp.rigorous_orbits import ball_json,vector_json
need,R,M,U=H.need,H.R,H.M,H.U


def point_operator(C0,J,d,weights,tick=lambda:None):
 n=len(weights)
 need(1<=n<=705 and len(J)==3 and all(a.nrows()==a.ncols()==n and H.finite(a)for a in J+[C0]),'complete point operator')
 need(all(v.is_exact()and v.is_finite()for row in C0.tolist()for v in row)and d.is_exact()and d.is_finite(),'exact finite operator proposal and endpoint')
 need(all(w.is_exact()and w.is_finite()and w>0 for w in weights),'positive exact weights')
 # Independently propose a finite C1; no producer inverse polynomial authority.
 C1=H.mid(-C0*H.mid(J[1])*C0);C=H.mid(C0+C1*d)
 A=J[0]+J[1]*d+J[2]*d*d;D=M.ident(n)-C*A;tick()
 q=max((sum((abs(D[i,j]).upper()*weights[j]for j in range(n)),arb(0))/weights[i]).upper()for i in range(n))
 need(q.is_finite()and 0<=q<1,'strict actual full-error point contraction')
 return C,D,q


def translated_domain(error,predictor,points):
 need(len(error)==len(predictor)and all(len(v)==len(error)for v in points),'every native endpoint domain coordinate')
 for j,old in enumerate(error):
  elo,ehi=M.interval(old);center=M.point(ball_json(predictor[j]))
  for point in points:
   lo,hi=M.interval(point[j]);need(elo<=lo-center<=hi-center<=ehi,'exact original correction covers entire mean-value segment')
 return True


def localize(C,D,G,point,known,weights,tick=lambda:None):
 n=len(point);need(len(known)==len(weights)==n and C.nrows()==C.ncols()==D.nrows()==D.ncols()==G.nrows()==n and G.ncols()==1,'full same-root localization')
 need(all(v.is_exact()and v.is_finite()for v in point+weights)and all(v>0 for v in weights)and all(v.is_exact()and v.is_finite()for row in C.tolist()for v in row),'finite exact point/proposal/scales')
 need(H.finite(D)and H.finite(G),'finite retained residual/derivative')
 q=max((sum((abs(D[i,j]).upper()*weights[j]for j in range(n)),arb(0))/weights[i]).upper()for i in range(n));need(q.is_finite()and 0<=q<1,'strict own Neumann bound')
 rhs=-C*G;size=max((abs(rhs[i,0]).upper()/weights[i]).upper()for i in range(n));radius=(size/(1-q)).upper()
 error=arb_mat([[arb(0,(radius*weights[i]).upper()).intersection(R.original_ball(known[i])-point[i])]for i in range(n)])
 for _ in range(3):
  image=rhs+D*error;error=arb_mat([[error[i,0].intersection(image[i,0])]for i in range(n)]);tick()
 box=[(point[i]+error[i,0]).intersection(R.original_ball(known[i]))for i in range(n)];need(all(v.is_finite()for v in box),'finite same independently existing root')
 return dict(passed=True,dimension=n,weighted_bound=ball_json(q),box=vector_json(box),scope='Conditional same-root tightening from independent full-error point operator and authenticated fresh point residual; three own intersections. Existence and actual derivative coverage are caller prerequisites.')


def interior_minimum(endpoints,joined):
 need(joined is True and len(endpoints)==3 and all(len(row)==705 for row in endpoints),'two independently joined full705 charts')
 values=[M.interval(row[-3])for row in endpoints]
 left=values[0][0]-values[1][1];right=values[2][0]-values[1][1]
 need(left>0 and right>0,'strict exact endpoint rate ordering')
 return dict(passed=True,coordinate='hidden_learning_rate',left_gap=str(left),right_gap=str(right),endpoint_intervals=[[str(v)for v in row]for row in values],scope='Continuity on the independently joined compact path proves at least one interior minimum of hidden learning rate. No unique, nondegenerate or located minimum, complementary stability or complete window claim.')
