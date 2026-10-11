"""Independent first705 correlated tangent and common-parameter complement.

Only authenticated acquired arrays are shared. Full705 root/terminal/path
assembly and weighted Neumann arithmetic are independently reconstructed.
"""
from flint import arb,arb_mat,ctx
from fractions import Fraction as F
import review_native704_complement_math as H
import review_unit_pair_pal_math as P
M,R,U,need=H.M,H.R,H.U,H.need
from neural_shrimp.rigorous_orbits import ball_json,matrix_json,vector_json
finite,infnorm,frob,positive,convolution,polynomial,frame=H.finite,H.infnorm,H.frob,H.positive,H.convolution,H.polynomial,H.frame


def geometry(unit):
 need(type(unit)is int and 0<=unit<=4,'only first and four fixed successor charts')
 return str(F(31+64*unit,4096)),[str(F(-1+64*unit,4096)),str(F(63+64*unit,4096))]


def localize(chart,directions,paths,tick=lambda:None,unit=0):
 need(chart['schema']=='neural-shrimp-native-unit-pair-pal-v1' and chart['shooting_dimension']==705 and chart['dimension']==13 and chart['period']==18,'actual full705 chart')
 sc_text,endpoints=geometry(unit)
 need(chart['parameter_center']==sc_text and chart['parameter_endpoints']==endpoints and chart['settings']=={'precision':256,'newton_steps':2,'halfwidth_exponent':-7,'radius_exponent':-10},'fixed first705 chart')
 n=705;c,s,a=[[R.exact(M.point(v))for v in chart[k]]for k in ('center','slope','curvature')];w=[R.exact(F(v))for v in chart['weights']];nn=[R.exact(F(v))for v in chart['normal']];oo=[R.exact(F(v))for v in chart['origin']]
 need(all(len(v)==n for v in (c,s,a,w,nn,oo))and all(v>0 for v in w),'all exact705 coefficients')
 sc=R.exact(F(sc_text));h=R.exact(F(1,128));d=arb(0,h);rho=R.exact(F(1,1024));pred=[c[i]+s[i]*d+a[i]*d*d/2 for i in range(n)];whole=[pred[i]+arb(0,rho*w[i])for i in range(n)]
 need([M.point(v)for v in chart['coordinate_radii']]==[F(1,1024)*F(v)for v in chart['weights']],'literal native weighted radius')
 for old,new in zip(chart['whole_box'],whole,strict=True):H.contains(old,new)
 need(len(directions)==144 and len(paths)==108,'all original native acquisition groups')
 g0,_=P.rebuild(c,directions[72:108],13,18,nn,oo,sc);tick();_,J=P.rebuild(whole,directions[108:],13,18,nn,oo,sc+d);tick()
 coeff=P.path_coefficients(c,s,a,arb(0),paths[36:72],13,18,nn,oo,sc);tick();wc=P.path_coefficients(c,s,a,d,paths[72:],13,18,nn,oo,sc);tick()
 C=R.matrix(chart['preconditioner']);need(C.nrows()==C.ncols()==n and all(v.is_exact()for row in C.tolist()for v in row),'fixed full exact preconditioner')
 D=M.ident(n)-C*J;k=max((sum((abs(D[i,j]).upper()*w[j]for j in range(n)),arb(0))/w[i]).upper()for i in range(n));need(k.is_finite()and 0<=k<1,'strict independent Neumann bound')
 initial=arb_mat([[arb(0,rho*v)]for v in w]);residual=arb_mat([[coeff[i,0]+coeff[i,1]*d+coeff[i,2]*d*d+wc[i,3]*d*d*d]for i in range(n)])
 ce=H.enclosure(D,-C*g0,w,k,initial,iterations=6);tick();we=H.enclosure(D,-C*residual,w,k,initial,iterations=6);tick()
 center=[c[i]+ce[i,0]for i in range(n)];domain=[pred[i]+we[i,0]for i in range(n)];error=[we[i,0]for i in range(n)];pprime=[s[i]+a[i]*d for i in range(n)];segment=[pred[i]+arb(0,abs(error[i]).upper())for i in range(n)]
 for old,new in zip(chart['whole_box'],segment,strict=True):H.contains(old,new)
 need(all(v.is_finite()for v in center+domain+segment)and all(v>0 for v in center[-3:]+domain[-3:]+segment[-3:]),'same known root, both rates and angle strictly positive')
 rp=arb_mat([[coeff[i,1]+2*coeff[i,2]*d+3*wc[i,3]*d*d]for i in range(n)])
 proof=dict(passed=True,precision=ctx.prec,dimension=705,own_intersections=6,weighted_neumann_bound=ball_json(k),center=vector_json(center),whole=vector_json(domain),error=vector_json(error),segment=vector_json(segment),predictor_tangent=vector_json(pprime),residual_derivative=matrix_json(rp))
 return dict(C=C,D=D,kappa=k,weights=w,center=center,whole=domain,error=error,segment=segment,pprime=pprime,rp=rp,normal=nn,origin=oo,parameter=sc,halfwidth=h),proof


def polarized_paths(segment,plus,minus,events,normal,origin,parameter,n,p,tick=lambda:None):
 N=3*n*p+3;need(1<=N<=705 and len(segment)==len(plus)==len(minus)==N and len(events)==4*p,'full bounded polarized paths')
 zero=[arb(0)]*N;cp=P.path_coefficients(segment,plus,zero,arb(0),events[:2*p],n,p,normal,origin,parameter);tick();cm=P.path_coefficients(segment,minus,zero,arb(0),events[2*p:],n,p,normal,origin,parameter);tick()
 return arb_mat([[(cp[i,2]-cm[i,2])/2]for i in range(N)])


def correlated_tangent(data,events,tick=lambda:None):
 need(len(events)==72,'complete actual plus/minus native paths');n=705;error,pprime=data['error'],data['pprime'];b=R.exact(F(1,256));plus=[error[i]/b+b*pprime[i]for i in range(n)];minus=[error[i]/b-b*pprime[i]for i in range(n)]
 mixed=polarized_paths(data['segment'],plus,minus,events,data['normal'],data['origin'],data['parameter'],13,18,tick);rhs=-data['C']*(data['rp']+mixed);correction=H.enclosure(data['D'],rhs,data['weights'],data['kappa'],iterations=6);tick();tangent=[pprime[i]+correction[i,0]for i in range(n)]
 need(all(v.is_finite()for v in tangent),'finite independent full705 true tangent')
 return tangent,dict(passed=True,dimension=705,precision=ctx.prec,own_intersections=6,plus=vector_json(plus),minus=vector_json(minus),mixed=matrix_json(mixed),rhs=matrix_json(rhs),correction=matrix_json(correction),tangent=vector_json(tangent),scope='Independent whole705 common-parameter residual derivative and polarized augmented Hessian forcing; shared actual analytic PathJet native returns. Known root existence and exact same-gauge branch identity inherited.')

def reduction(center,whole,tangent,E,B,native,phase_records,metric,halfwidth,angle,tick=lambda:None):
 need(len(center)==len(whole)==len(tangent)==704 and len(E)==len(B)==len(native)==len(phase_records)==18,'full fixed native/complement inventory')
 fc,_,ic=frame(center,[arb(0)]*704,E);fw,fp,iw=frame(whole,tangent,E);ib=[]
 for v in B:need(v.nrows()==v.ncols()==11 and all(x.is_exact()and x.is_finite()for row in v.tolist()for x in row),'exact fixed11D metric factor');ib.append(v.inv());need(finite(ib[-1]),'validated cyclic metric inverse')
 poly=[M.ident(11),arb_mat(11,11),arb_mat(11,11)];eps=arb(0);gamma_product=arb(1);reports=[];h=halfwidth;need(h.is_exact()and h>0 and angle>0 and abs(angle).upper()<=R.exact(F(1,4)),'exact positive gauge halfwidth and whole actual angle');term=U.terminal(2,angle);terminal=arb_mat([[term[2,2],term[4,2]],[term[2,4],term[4,4]]])
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
