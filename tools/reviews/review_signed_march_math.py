"""One parameterized saved signed470 reduction; all original independent mathematics retained."""
from pathlib import Path
from fractions import Fraction as F
import gc
from flint import arb,arb_mat
import review_signed_correlated as S
D,H,R,M,U,need=S.D,S.H,S.R,S.M,S.U,S.need
read=S.read
from neural_shrimp.rigorous_orbits import ball_json,vector_json

def check_geometry(inp,known,chart):
 width=inp['settings']['halfwidth_exponent'];need(type(width)is int and -8<=width<=-1 and inp['settings']['radius_exponent']==-12,'fixed bounded acquired geometry')
 h=F(2)**width;p=F(inp['parameter']);ends=[str(p-h),str(p+h)]
 need(inp['parameter_endpoints']==chart['parameter_endpoints']==ends and str(p+h)==known['exact_parameter'],'exact next-right/known-left scalar')
 need(inp['normal']==known['normal']==chart['normal']==['0']*468+['1','0']and inp['origin']==known['origin']==chart['origin'],'same actual native hidden gauge')
 need(known['dimension']==13 and known['period']==18 and known['sigma']==-1 and len(known['box'])==470,'whole independently known signed root')
 need(inp['known']['normal']==known['normal']and inp['known']['origin']==known['origin']and inp['known']['exact_parameter']==known['exact_parameter'],'historical and independently rebuilt root identity in same gauge')

def acquisitions(p):
 events=[(f.name.split('-target-')[1].removesuffix('.json.gz'),read(f))for f in sorted((p/'events').glob('*-target-*.json.gz'))]
 by={}
 for k,v in events:by.setdefault(k,[]).append(v)
 expected=dict(native_input=144,native_return=144,phase_return=54,path_phase_return=54,signed_gradient_path_return=36,signed_gradient_path_input=36,signed_proposal_input=3,path_assembly_return=3,signed_proposal_return=3,signed_gradient_path_assembly_return=2,signed_affine_proposal_input=1,signed_predictor_return=1,signed_correlated_domain=1)
 need({k:len(v)for k,v in by.items()}==expected,'every target event retained')
 need([v['index']for v in by['native_input']]==[v['index']for v in by['native_return']]==list(range(144)),'all literal144 native input/return indices')
 gradients=[];paths=[]
 for i,(kind,v)in enumerate(events):
  if kind=='signed_gradient_path_input':
   seq=events[i+1:i+4];need([k for k,_ in seq]==['native_input','native_return','signed_gradient_path_return'],'literal nested gradient acquisition order');ni,no,ret=[r for _,r in seq];need(ni['index']==no['index']and v['phase']==ret['phase'],'gradient phase/index')
   need([{k:x for k,x in z.items()if k!='kind'}for z in ni['inputs']]==v['inputs']and[{k:x for k,x in z.items()if k!='kind'}for z in no['returns']]==ret['returns']and all(z['kind']=='gradient_path'for z in ni['inputs']+no['returns']),'outer gradient tied to actual native payload');gradients.append(dict(**v,returns=ret['returns']))
  if kind=='path_phase_return':
   need(i>=2 and [k for k,_ in events[i-2:i]]==['native_input','native_return'],'literal native path acquisition order');ni,no=[r for _,r in events[i-2:i]];need(ni['index']==no['index']and all(z['kind']=='path'for z in ni['inputs']+no['returns'])and[[z['a'],z['b']]for z in ni['inputs']]==v['inputs']and[[z['a'],z['b']]for z in no['returns']]==v['returns'],'path acquisition exact payload');paths.append(v)
 return by['signed_correlated_domain'][0],by['signed_predictor_return'][0],gradients,paths[-36:]

def replay_one(p,known,w):
 inp=read(p/'inputs.json');domain,proposal,grad,paths=acquisitions(p);chart=read(p/'target-certificate.json.gz');need(chart['schema']=='neural-shrimp-correlated-signed-eigenpair-v1'and chart['sigma']==-1 and chart['shooting_dimension']==470 and chart['settings']==inp['settings']and chart['parameter_endpoints']==inp['parameter_endpoints'],'literal fixed original correlated claim')
 need(chart['center']==domain['center']==proposal['center']and chart['slope']==domain['slope']==proposal['slope']and chart['curvature']==domain['curvature']==proposal['curvature']and chart['normal']==inp['normal']and chart['origin']==inp['origin']and chart['weights']==inp['weights'],'entire predictor and phase/rate/gauge proposal')
 check_geometry(inp,known,chart)
 c,s,a=[[R.exact(M.point(v))for v in chart[k]]for k in('center','slope','curvature')];normal=[R.exact(F(v))for v in chart['normal']];origin=[R.exact(F(v))for v in chart['origin']];weights=[R.exact(F(v))for v in chart['weights']];h=R.exact(F(2)**inp['settings']['halfwidth_exponent']);rho=arb(1)/4096;d=arb(0,h);error=[R.original_ball(v)for v in domain['error']];need(all(len(v)==470 for v in(c,s,a,normal,origin,weights,error))and all(w>0 for w in weights),'all470 actual coordinates')
 for old,weight in zip(domain['error'],weights,strict=True):
  lo,hi=M.interval(old);rr=M.point(ball_json(rho*weight));need(lo==-hi and lo<=-rr and rr<=hi,'actual full correction E covers each requested radius')
 whole=[c[i]+s[i]*d+a[i]*d*d/2+error[i]for i in range(470)]
 for outer,inner in zip(domain['whole_box'],whole,strict=True):U.contains(outer,ball_json(inner),'actual fullE curved tube')
 need(domain['whole_box']==chart['whole_box']and domain['parameter']==chart['parameter_center']==inp['parameter']and domain['sigma']==-1 and whole[-2]>0 and whole[-1]>0,'same physical domains/gauge')
 zero=D.jacobian(c,s,a,error,arb(0),grad[:18],13,18,-1,normal,w.tick);wide=D.jacobian(c,s,a,error,d,grad[18:],13,18,-1,normal,w.tick);J=[zero[0],zero[1],wide[2]];del zero,wide;gc.collect()
 coeff=D.residual(c,s,a,arb(0),paths[:18],13,18,-1,normal,origin,R.exact(F(inp['parameter'])),w.tick);wholecoeff=D.residual(c,s,a,d,paths[18:],13,18,-1,normal,origin,R.exact(F(inp['parameter'])),w.tick);res=[arb_mat([[coeff[i,k]if k<3 else wholecoeff[i,3]]for i in range(470)])for k in range(4)]
 raw=chart['inverse_polynomial_exact'][0];need(chart['inverse_polynomial_encoding']=='exact-dyadic-pairs'and raw==proposal['preconditioner_exact']and len(raw)==470 and all(len(row)==470 for row in raw),'exact original C0 finite proposal');C0=arb_mat([[arb(tuple(v))for v in row]for row in raw]);need(all(v.is_exact()for row in C0.tolist()for v in row),'all original inverse entries exact');w.emit('acquired_J0',J[0]);w.emit('acquired_J1',J[1]);w.emit('acquired_J2',J[2]);proof=H.polynomial_root(C0,J,res,weights,rho,h,error,w.tick,w.emit)
 ac=D.anchor(known['box'],c,s,a,normal,origin,R.exact(F(inp['parameter'])),h,[rho*v for v in weights],exact_parameter=known['exact_parameter']);proper=[dict(divisor=k,coordinates=[i for i in range(13)if not(whole[26*k+i]-whole[i]).contains(0)])for k in(1,2,3,6,9)];gates=dict(root=proof['passed'],known_root=ac['passed']and ac['coordinates']==470,primitive=all(v['coordinates']for v in proper));ends=[]
 for sign,corr in zip((-1,1),proof['endpoint_corrections'],strict=True):
  dd=sign*h;ends.append(vector_json([c[i]+s[i]*dd+a[i]*dd*dd/2+R.original_ball(corr[i])for i in range(470)]))
 w.save('independent512.json',dict(passed=all(gates.values()),gates=gates,root=proof,anchor=ac,primitive=proper,recomputed_endpoints=ends));w.save('connected-left-endpoint.json'if all(gates.values())else'unqualified-endpoint.json',dict(box=ends[0],normal=chart['normal'],origin=chart['origin'],exact_parameter=inp['parameter_endpoints'][0],dimension=13,period=18,sigma=-1));return dict(box=ends[0],normal=chart['normal'],origin=chart['origin'],exact_parameter=inp['parameter_endpoints'][0],dimension=13,period=18,sigma=-1),dict(passed=all(gates.values()),root_coordinates=sum(proof['coordinate_inclusions']),anchor_coordinates=ac['coordinates'],contraction=proof['contraction'],native_calls=0,scope='Independent saved512 fullE native Jacobian/path assembly and polynomial proof plus exact-gauge full470 known-root join. Shared native derivatives/Arb; no complement/genericity/global closure.')
