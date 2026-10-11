"""Independent known-root localization and moving-frame complementary arithmetic.

No producer localization, quotient, cocycle, norm, spectral or model routine is
called. The authenticated native analytic acquisitions and Arb are shared.
"""
from pathlib import Path
import sys
from fractions import Fraction as F
from flint import arb,arb_mat,ctx
sys.path.insert(0,str(Path(__file__).with_name('native704_aux')))
import review_unit_pair_math as U
M,R=U.M,U.R
need=M.need
from neural_shrimp.rigorous_orbits import ball_json,vector_json,matrix_json

def finite(a):return all(v.is_finite()for row in a.tolist()for v in row)
def infnorm(a):return max(sum((abs(v).upper()for v in row),arb(0)).upper()for row in a.tolist())
def frob(a):return sum((abs(v).upper()**2 for row in a.tolist()for v in row),arb(0)).sqrt().upper()
def mid(a):return arb_mat([[v.mid()for v in row]for row in a.tolist()])
def contains(outer,inner):U.contains(outer,ball_json(inner),'fresh independent domain within actual acquired domain')

def enclosure(D,v,w,kappa,initial=None):
 n=len(w);need(0<=kappa and kappa<1 and finite(D)and finite(v),'finite strict known-root Neumann premise')
 rho=max((abs(v[i,0]).upper()/w[i]).upper()for i in range(n))/(1-kappa)
 box=arb_mat([[arb(0,(rho*w[i]).upper())]for i in range(n)])
 if initial is not None:box=arb_mat([[box[i,0].intersection(initial[i,0])]for i in range(n)])
 for _ in range(3):
  image=v+D*box;need(finite(image),'finite known-root image');box=arb_mat([[box[i,0].intersection(image[i,0])]for i in range(n)])
 return box

def localization(chart,groups,tick=lambda:None):
 need(chart['schema']=='neural-shrimp-native-unit-pair-tube-v1'and(chart['dimension'],chart['period'],chart['shooting_dimension'])==(13,18,704),'native704 theorem convention')
 need(chart['settings']=={'precision':256,'newton_steps':2,'halfwidth_exponent':-16,'radius_exponent':-11}and chart['parameter_center']=='1/65536'and chart['parameter_endpoints']==['0','1/32768'],'literal original tube')
 c=[R.exact(M.point(v))for v in chart['center']];s=arb_mat([[R.exact(M.point(v[0]))]for v in chart['slope']]);w=[R.exact(F(v))for v in chart['weights']];n=len(c)
 need(n==len(w)==s.nrows()==704 and all(0<v<=1 for v in w),'all exact704 weights/predictor coordinates')
 h=R.exact(F(1,65536));rho=R.exact(F(1,2048));d=arb(0,h);t=h+d
 need([M.point(v)for v in chart['coordinate_radii']]==[F(1,2048)*F(v)for v in chart['weights']],'exact original correction radii')
 pred=[c[i]+s[i,0]*d for i in range(n)];whole=[pred[i]+arb(0,rho*w[i])for i in range(n)]
 for old,v in zip(chart['whole_box'],whole,strict=True):contains(old,v)
 need(len(groups)==3 and all(len(g)==36 for g in groups),'three final complete direction groups')
 g0,_,_=U.rebuild(c,groups[0],13,18,h);tick();_,jp,hp=U.rebuild(pred,groups[1],13,18,t);tick();_,j,hw=U.rebuild(whole,groups[2],13,18,t);tick()
 C=R.matrix(chart['preconditioner']);need(C.nrows()==C.ncols()==n and all(v.is_exact()for row in C.tolist()for v in row),'exact complete preconditioner')
 D=M.ident(n)-C*j;kappa=max((sum((abs(D[i,k]).upper()*w[k]for k in range(n)),arb(0))/w[i]).upper()for i in range(n));need(kappa.is_finite()and kappa<1,'independently strict full Neumann contraction')
 initial=arb_mat([[arb(0,rho*v)]for v in w]);center_error=enclosure(D,-C*g0,w,kappa,initial);tick()
 whole_error=enclosure(D,-C*(g0+(jp*s+hp)*d),w,kappa,initial);tick()
 tangent_error=enclosure(D,-C*(j*s+hw),w,kappa);tick()
 center=[c[i]+center_error[i,0]for i in range(n)];domain=[pred[i]+whole_error[i,0]for i in range(n)];tangent=[s[i,0]+tangent_error[i,0]for i in range(n)]
 need(all(v.is_finite()for v in center+domain+tangent)and all(v>0 for v in center[-2:]+domain[-2:]),'finite full localized same root and both positive rates')
 return center,domain,tangent,dict(passed=True,dimension=704,weighted_neumann_bound=ball_json(kappa),center=vector_json(center),whole=vector_json(domain),tangent=vector_json(tangent),scope='Independent conditional known-root center/whole/tangent enclosure from full native shooting blocks; existence and branch identity inherited from authenticated prior independent704 proof.')

def positive(a,basis):
 n=a.nrows();need(a.ncols()==basis.nrows()==basis.ncols()==n and all(v.is_exact()and v.is_finite()for row in basis.tolist()for v in row),'exact square congruence proposal')
 C=mid(basis.inv());defect=infnorm(M.ident(n)-C*basis);need(defect<1,'validated nonsingular exact congruence')
 transformed=C*a*C.transpose();margins=[transformed[i,i].lower()-sum((abs(transformed[i,k]).upper()for k in range(n)if k!=i),arb(0))for i in range(n)]
 good=all(v.is_finite()and v>0 for v in margins)
 return dict(passed=good,minimum_margin=ball_json(min(margins)),inverse_defect=ball_json(defect),basis=matrix_json(C)),min(margins),C

def convolution(a,b):
 n=a[0].nrows();out=[arb_mat(n,n)for _ in range(len(a)+len(b)-1)]
 for d in range(len(out)):
  for i in range(len(a)):
   if 0<=d-i<len(b):out[d]+=a[i]*b[d-i]
 return out

def polynomial(coeff):
 out=coeff[0]+arb_mat(coeff[0].nrows(),coeff[0].ncols())
 for k in range(1,len(coeff)):out+=coeff[k]*(arb(0,1)if k%2 else arb('0.5','0.5'))
 return out

def frame(flat,tangent,E):
 Fm=[];Fp=[];inv=[]
 for j,ee in enumerate(E):
  need(ee.nrows()==13 and ee.ncols()==11 and all(v.is_exact()and v.is_finite()for row in ee.tolist()for v in row),'exact full complementary frame proposal')
  off=39*j;f=arb_mat([[flat[off+13+i],flat[off+26+i]]+[ee[i,k]for k in range(11)]for i in range(13)]);fp=arb_mat([[tangent[off+13+i],tangent[off+26+i]]+[arb(0)]*11 for i in range(13)]);fi=f.inv();need(finite(fi),'validated entire moving frame inverse');Fm.append(f);Fp.append(fp);inv.append(fi)
 return Fm,Fp,inv

def reduction(center,whole,tangent,E,B,native,phase_records,metric,tick=lambda:None):
 need(len(center)==len(whole)==len(tangent)==704 and len(E)==len(B)==len(native)==len(phase_records)==18,'full fixed native/complement inventory')
 fc,_,ic=frame(center,[arb(0)]*704,E);fw,fp,iw=frame(whole,tangent,E);ib=[]
 for v in B:need(v.nrows()==v.ncols()==11 and all(x.is_exact()and x.is_finite()for row in v.tolist()for x in row),'exact fixed11D metric factor');ib.append(v.inv());need(finite(ib[-1]),'validated cyclic metric inverse')
 poly=[M.ident(11),arb_mat(11,11),arb_mat(11,11)];eps=arb(0);gamma_product=arb(1);reports=[];h=R.exact(F(1,65536));term=U.terminal(2,h+arb(0,h));terminal=arb_mat([[term[2,2],term[4,2]],[term[2,4],term[4,4]]])
 for j,(A0,Aw,Ap)in enumerate(native):
  tick();need(all(v.nrows()==v.ncols()==13 and finite(v)for v in [A0,Aw,Ap]),'complete native derivatives');after=(j+1)%18
  T0=ic[after]*A0*fc[j];Tw=iw[after]*Aw*fw[j];Tp=iw[after]*(Ap*fw[j]+Aw*fp[j]-fp[after]*Tw)
  need(all(Tw[i,k].contains(0)for i in range(2,13)for k in range(2)),'conditional exact invariant plane consistency')
  expected=terminal if j==17 else M.ident(2);need(all(Tw[i,k].overlaps(expected[i,k])for i in range(2)for k in range(2)),'exact native terminal consistency')
  cut=lambda v:arb_mat([[v[i,k]for k in range(2,13)]for i in range(2,13)])
  Q0=B[after]*cut(T0)*ib[j];Qp=B[after]*cut(Tp)*ib[j]
  record=phase_records[j];need(record['phase']==j,'literal retained phase order');phase=[R.matrix(v)for v in record['coefficients']]
  need(len(phase)==3 and all(v.nrows()==v.ncols()==11 and all(x.is_exact()and x.is_finite()for row in v.tolist()for x in row)for v in phase)and all(x.is_zero()for row in phase[2].tolist()for x in row),'fixed exact affine phase proposals')
  error=Q0-phase[0]+(h*Qp-phase[1])*arb(0,1);local_eps=frob(error);op=record['operator_bound'];gamma=R.original_ball(op['gamma']);need(gamma.is_exact()and gamma>0,'exact positive operator proposal')
  check,_,_=positive(M.ident(11)*(gamma*gamma)-phase[0].transpose()*phase[0],R.matrix(op['positive']['basis']));need(check['passed'],'strict independent phase operator norm')
  full=convolution(phase,poly);a_bound=gamma+frob(phase[1])+frob(phase[2]);b_bound=gamma_product+frob(poly[1])+frob(poly[2]);eps=(a_bound*eps+local_eps*b_bound+local_eps*eps+frob(full[3])+frob(full[4])).upper();gamma_product=(gamma*gamma_product).upper();poly=full[:3]
  reports.append(dict(phase=j,operator=check,phase_error=ball_json(local_eps),accumulated_error=ball_json(eps)))
 P=R.matrix(metric['P']);need(P.nrows()==P.ncols()==11 and all(P[i,k].is_exact()and P[i,k]==P[k,i]for i in range(11)for k in range(11)),'exact symmetric common metric')
 pc,_,_=positive(P,R.matrix(metric['P_positive']['basis']));need(pc['passed'],'strict independent common metric positivity')
 allterms=convolution([v.transpose()*P for v in poly],poly);dc=[P-allterms[0]]+[-v for v in allterms[1:]];qbound=gamma_product+frob(poly[1])+frob(poly[2]);pbound=infnorm(P);penalty=(pbound*(2*qbound*eps+eps*eps)).upper();decrease=polynomial(dc)-M.ident(11)*penalty
 dc,margin,C=positive(decrease,R.matrix(metric['complete_D_positive']['basis']));need(dc['passed'],'strict independent complete common-parameter Lyapunov decrease')
 # C D C^T >= margin I implies D >= margin/||C||2^2 I.
 csq=infnorm(C)*infnorm(C.transpose());gap=(margin/(csq*pbound)).lower();need(0<gap and gap<1,'strict conservative metric contraction gap');modulus=(1-gap).sqrt().upper();need(modulus<1,'all11 strictly interior moduli')
 return dict(passed=True,precision=ctx.prec,phases=18,complement_dimension=11,phase_checks=reports,P_positive=pc,complete_D_positive=dc,remainder_norm_upper=ball_json(eps),polynomial_norm_upper=ball_json(qbound),remainder_penalty=ball_json(penalty),squared_metric_contraction_gap=ball_json(gap),complementary_modulus_upper=ball_json(modulus),scope='Independent moving-frame quotient and common-scalar polynomial plus complete operator-error reduction. All11 complementary multipliers strictly inside unit circle on inherited primitive18 unit-pair curve; critical pair remains neutral. Shared native analytic derivatives and Arb; no signed NS criticality, stable side or whole-window geometry.')
