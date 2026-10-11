"""Fresh output-gauge replay using unchanged independent signed polynomial reductions."""
from fractions import Fraction as F
import gc
from flint import arb,arb_mat
import review_signed_march_math as T
D,H,R,M,U,need=T.D,T.H,T.R,T.M,T.U,T.need
read,acquisitions=T.read,T.acquisitions
from neural_shrimp.rigorous_orbits import ball_json,vector_json

def replay_curve(p,known,w):
 inp=read(p/'target-inputs.json');domain,proposal,grad,paths=acquisitions(p);chart=read(p/'target-certificate.json.gz');need(chart['schema']=='neural-shrimp-correlated-signed-eigenpair-v1'and chart['sigma']==-1 and chart['shooting_dimension']==470 and chart['settings']==dict(halfwidth_exponent=-18,radius_exponent=-12)and chart['parameter_endpoints']==[str(F(inp['parameter'])-F(1,2**18)),str(F(inp['parameter'])+F(1,2**18))],'literal fixed original correlated claim')
 need(chart['center']==domain['center']==proposal['center']and chart['slope']==domain['slope']==proposal['slope']and chart['curvature']==domain['curvature']==proposal['curvature']and chart['normal']==inp['normal']and chart['origin']==inp['origin']and chart['weights']==inp['weights'],'entire predictor and phase/rate/gauge proposal')
 need(inp['settings']==dict(period=18,sigma=-1,precision=256,newton_steps=2,halfwidth_exponent=-18,radius_exponent=-12)and inp['normal']==['0']*469+['1']and not inp['anchor_exact_parameter_used'],'literal output gauge and projected join')
 need(known['dimension']==13 and known['period']==18 and known['sigma']==-1 and len(known['box'])==470,'full independently sharpened signed root')
 c,s,a=[[R.exact(M.point(v))for v in chart[k]]for k in('center','slope','curvature')];normal=[R.exact(F(v))for v in chart['normal']];origin=[R.exact(F(v))for v in chart['origin']];weights=[R.exact(F(v))for v in chart['weights']];h=R.exact(F(2)**inp['settings']['halfwidth_exponent']);rho=arb(1)/4096;d=arb(0,h);error=[R.original_ball(v)for v in domain['error']];need(all(len(v)==470 for v in(c,s,a,normal,origin,weights,error))and all(w>0 for w in weights),'all470 actual coordinates')
 for old,weight in zip(domain['error'],weights,strict=True):
  lo,hi=M.interval(old);rr=M.point(ball_json(rho*weight));need(lo==-hi and lo<=-rr and rr<=hi,'actual full correction E covers each requested radius')
 whole=[c[i]+s[i]*d+a[i]*d*d/2+error[i]for i in range(470)]
 for outer,inner in zip(domain['whole_box'],whole,strict=True):U.contains(outer,ball_json(inner),'actual fullE curved tube')
 need(domain['whole_box']==chart['whole_box']and domain['parameter']==chart['parameter_center']==inp['parameter']and domain['sigma']==-1 and whole[-2]>0 and whole[-1]>0,'same physical domains/gauge')
 zero=D.jacobian(c,s,a,error,arb(0),grad[:18],13,18,-1,normal,w.tick);wide=D.jacobian(c,s,a,error,d,grad[18:],13,18,-1,normal,w.tick);J=[zero[0],zero[1],wide[2]];del zero,wide;gc.collect()
 coeff=D.residual(c,s,a,arb(0),paths[:18],13,18,-1,normal,origin,R.exact(F(inp['parameter'])),w.tick);wholecoeff=D.residual(c,s,a,d,paths[18:],13,18,-1,normal,origin,R.exact(F(inp['parameter'])),w.tick);res=[arb_mat([[coeff[i,k]if k<3 else wholecoeff[i,3]]for i in range(470)])for k in range(4)]
 raw=chart['inverse_polynomial_exact'][0];need(chart['inverse_polynomial_encoding']=='exact-dyadic-pairs'and raw==proposal['preconditioner_exact']and len(raw)==470 and all(len(row)==470 for row in raw),'exact original C0 finite proposal');C0=arb_mat([[arb(tuple(v))for v in row]for row in raw]);need(all(v.is_exact()for row in C0.tolist()for v in row),'all original inverse entries exact');w.emit('acquired_J0',J[0]);w.emit('acquired_J1',J[1]);w.emit('acquired_J2',J[2]);proof=H.polynomial_root(C0,J,res,weights,rho,h,error,w.tick,w.emit)
 ac=D.anchor(known['box'],c,s,a,normal,origin,R.exact(F(inp['parameter'])),h,[rho*v for v in weights],exact_parameter=None);proper=[dict(divisor=k,coordinates=[i for i in range(13)if not(whole[26*k+i]-whole[i]).contains(0)])for k in(1,2,3,6,9)];gates=dict(root=proof['passed'],known_root=ac['passed']and ac['coordinates']==470,primitive=all(v['coordinates']for v in proper));ends=[]
 for sign,corr in zip((-1,1),proof['endpoint_corrections'],strict=True):
  dd=sign*h;ends.append(vector_json([c[i]+s[i]*dd+a[i]*dd*dd/2+R.original_ball(corr[i])for i in range(470)]))
 w.save('independent512.json',dict(passed=all(gates.values()),gates=gates,root=proof,anchor=ac,primitive=proper,recomputed_endpoints=ends));w.save('connected-right-endpoint.json'if all(gates.values())else'unqualified-endpoint.json',dict(box=ends[1],normal=chart['normal'],origin=chart['origin'],exact_parameter=chart['parameter_endpoints'][1],dimension=13,period=18,sigma=-1));return dict(box=ends[1],normal=chart['normal'],origin=chart['origin'],exact_parameter=chart['parameter_endpoints'][1],dimension=13,period=18,sigma=-1),dict(passed=all(gates.values()),root_coordinates=sum(proof['coordinate_inclusions']),anchor_coordinates=ac['coordinates'],contraction=proof['contraction'],native_calls=0,scope='Independent saved512 fullE native Jacobian/path assembly and polynomial proof plus projected changed-output-gauge full470 known-root join. Shared native derivatives/Arb; no complement/genericity/global closure.')
