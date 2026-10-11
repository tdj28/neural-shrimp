"""Cubic polynomial adapter over independently authored saved proof reducers.

Prepared by producer agent; requires root/peer review before scientific replay.
Only p/pprime polynomial order differs from the validated second-quarter adapter.
Native acquisitions remain shared; all root/tangent/point/quotient reductions are
unchanged separately authored independent routines. No target execution here.
"""
from fractions import Fraction as F
from flint import arb
import review_second_quarter as S
from review_cubic_polynomial_domain import bounded_domain
A,Q,H,R,M,need,read=S.A,S.Q,S.H,S.R,S.M,S.need,S.read
ball_json,vector_json,matrix_json=S.ball_json,S.vector_json,S.matrix_json

def coverage_diagnostic(old,new):
 need(len(old)==len(new),'complete diagnostic domain')
 flags=[];misses=[]
 for i,(a,z)in enumerate(zip(old,new,strict=True)):
  lo,hi=M.interval(a);left,right=M.interval(ball_json(z));ok=lo<=left<=right<=hi;flags.append(ok)
  if not ok:misses.append(dict(coordinate=i,actual_endpoints=[str(lo),str(hi)],independent_endpoints=[str(left),str(right)],lower_excess=str(max(F(0),lo-left)),upper_excess=str(max(F(0),right-hi))))
 return dict(passed=all(flags),dimension=len(flags),coordinate_containment=flags,failures=misses,scope='Exact rational actual-domain containment diagnostic; never an alternative acceptance gate.')


def shared_domain(w,run,spec,data):
 folder=run/f"domains/{spec['domain']['index']:03d}";td=read(folder/'tangent-domain-inputs.json');qd=read(folder/'quotient-domain-inputs.json');left,right=map(F,spec['domain']['endpoints']);center=(left+right)/2;half=(right-left)/2;shift=center-M.point(ball_json(data['sc']));lo,hi=shift-half,shift+half;normal,origin=data['normal'],data['origin'];c,s,a,b=[data[k]for k in('c','s','a','b')]
 need(td['domain']==qd['domain']==spec['domain']and td['scale']=='1/256'and td['original_offset']==[str(lo),str(hi)]and td['normal']==vector_json(normal)and td['origin']==vector_json(origin)and M.point(td['parameter'])==center,'actual exact tangent domain and native gauge');need(qd['complements']==[matrix_json(e)for e in data['E']]and qd['metrics']==[matrix_json(b)for b in data['metrics']],'fixed exact frames')
 err,proof=Q.root(data['ds'],data['ff'],data['weights'],data['original'],data['image'],lo,hi,w.tick);pred,pp,polynomial_evidence=bounded_domain(c,s,a,b,lo,hi);whole=[pred[i]+err[i,0]for i in range(705)];segment=[pred[i]+arb(0,abs(err[i,0]).upper())for i in range(705)]
 plus=[err[i,0]*256+pp[i]/256 for i in range(705)];minus=[err[i,0]*256-pp[i]/256 for i in range(705)]
 # Every range is derived from fixed exact coefficients, never acquisition
 # bounds. Save all available prerequisite comparisons before the first gate.
 checks=[('original full root',data['domain']['whole_box'],whole),('original full segment',data['domain']['whole_box'],segment),('actual segment',td['segment'],segment),('actual correction',[v[0]for v in td['error']],[err[i,0]for i in range(705)]),('actual predictor tangent',td['predictor_tangent'],pp),('actual plus',td['plus'],plus),('actual minus',td['minus'],minus),('actual705 whole',qd['full705']['whole'],whole)]
 w.save(f"domains/{spec['domain']['index']:03d}/root.json",proof)
 w.save(f"domains/{spec['domain']['index']:03d}/polynomial-bounds.json",dict(parameter_offset=[str(lo),str(hi)],coordinates=polynomial_evidence,scope='Exact rational extrema or eight Bernstein convex hulls, outwardly enclosed. Actual acquired bounds are not construction inputs.'))
 w.save(f"domains/{spec['domain']['index']:03d}/whole-domain.json",dict(parameter_offset=[str(lo),str(hi)],predictor=vector_json(pred),predictor_tangent=vector_json(pp),error=matrix_json(err),whole=vector_json(whole),segment=vector_json(segment),plus=vector_json(plus),minus=vector_json(minus)))
 w.save(f"domains/{spec['domain']['index']:03d}/pre-tangent-comparisons.json",dict(checks={label:coverage_diagnostic(old,new)for label,old,new in checks},scope='All currently available exact endpoint comparisons, saved before any coverage gate. No failing gate is bypassed.'))
 for label,old,new in checks:Q.cover(old,new,label)
 paths=[A.A.path_event(read(folder/f'native/{spec["start"]+j:04d}-input.json'),read(folder/f'native/{spec["start"]+j:04d}-return.json'),72+j)for j in range(1,73)]
 mixed=H.polarized_paths(segment,plus,minus,paths,normal,origin,R.exact(center),13,18,w.tick);tangent,tp=Q.tangent(data['C'],data['ds'],data['rp'],mixed,pp,data['weights'],lo,hi,w.tick)
 w.save(f"domains/{spec['domain']['index']:03d}/tangent.json",tp)
 w.save(f"domains/{spec['domain']['index']:03d}/tangent-domain-comparison.json",coverage_diagnostic(qd['full705']['tangent'],tangent))
 Q.cover(qd['full705']['tangent'],tangent,'actual705 tangent')
 derivatives=[]
 for j in range(18):
  k=spec['start']+73+j;inp=read(folder/f'native/{k:04d}-input.json');out=read(folder/f'native/{k:04d}-return.json');aw,ap=A.native(inp,out,whole,whole,tangent,'whole_tangent',j);derivatives.append((aw,ap))
 return dict(whole=whole,tangent=tangent,derivatives=derivatives,spec=spec)


def one(w,run,item,data,shared):
 spec=item['cell'];folder=run/f"cells/{item['ordinal']:03d}";shift=F(spec['center'])-M.point(ball_json(data['sc']));d=R.exact(shift);c,s,a,b=[data[k]for k in('c','s','a','b')];err,_=Q.root(data['ds'],data['ff'],data['weights'],data['original'],data['image'],shift,shift,w.tick);pc=[c[i]+s[i]*d+a[i]*d*d/2+b[i]*d*d*d for i in range(705)];known=[pc[i]+err[i,0]for i in range(705)];ca=read(folder/'center-authority.json');need(ca['cell']==spec and ca['offset']==str(shift)and ca['predictor']==vector_json(pc)and ca['original_error']==data['domain']['error'],'literal same predictor/root');Q.cover(ca['known'],known,'known center');Q.original(data['domain']['error'],pc,known)
 ri=read(folder/'residual-inputs/target2.json');point=[R.exact(M.point(v))for v in ri['point']];need(ri['normal']==vector_json(data['normal'])and ri['origin']==vector_json(data['origin'])and M.point(ri['parameter'])==F(spec['center']),'same actual point gauge');Q.original(data['domain']['error'],pc,point);events=[]
 for j in range(36):
  k=item['start']+73+j;aa=read(folder/f'native/{k:04d}-input.json');bb=read(folder/f'native/{k:04d}-return.json');need(aa['kind']==bb['kind']=='residual'and aa['stage']==bb['stage']=='target2','literal final residual group');events.append(A.J.mixed_direction(dict(aa,kind='full_jacobian',phase=j//2,direction='q'if j%2==0 else'r'),dict(bb,kind='full_jacobian')))
 value,jac=H.P.rebuild(point,events,13,18,data['normal'],data['origin'],R.exact(F(spec['center'])));del jac;center,proof=Q.point(data['C'],data['ds'],value,point,known,data['weights'],shift,w.tick);Q.original(data['domain']['error'],pc,center);w.save(f"cells/{item['ordinal']:03d}/center.json",proof);qr=read(folder/'quotient-domain-inputs.json');need(qr['cell']==spec and qr['halfwidth']==spec['halfwidth']and qr['parameter_endpoints']==spec['endpoints']and qr['complements']==[matrix_json(e)for e in data['E']]and qr['metrics']==[matrix_json(b)for b in data['metrics']],'literal cell width/frame proposal');whole,tangent=shared['whole'],shared['tangent']
 for key,v in [('center',center),('whole',whole),('tangent',tangent)]:Q.cover(qr['full705'][key],v,'actual705 '+key)
 native=[]
 for j,(aw,ap)in enumerate(shared['derivatives']):
  k=item['start']+109+j;ac,_=A.native(read(folder/f'native/{k:04d}-input.json'),read(folder/f'native/{k:04d}-return.json'),center,whole,tangent,'center',j);native.append((ac,aw,ap))
 retained=read(folder/'retained-derivatives.json');need(retained['domain']==shared['spec']['domain']and retained['calls']==list(range(shared['spec']['start']+73,shared['spec']['start']+91)),'exact retained derivative identities')
 records=[read(folder/f'events/{j+2:03d}-phase.json')for j in range(18)];metric=read(folder/'events/020-metric.json');proof=H.reduction(center[:-1],whole[:-1],tangent[:-1],data['E'],data['metrics'],native,records,metric,R.exact(F(spec['halfwidth'])),whole[-1],w.tick);w.save(f"cells/{item['ordinal']:03d}/complement.json",proof);return dict(passed=True,cell=spec,ordinal=item['ordinal'],complement_dimension=11,modulus=proof['complementary_modulus_upper'],penalty=proof['remainder_penalty'])
