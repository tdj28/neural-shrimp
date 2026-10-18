"""1024-bit analytic covering controls; no Qwen loading."""
import json,time,resource,argparse
from pathlib import Path
from fractions import Fraction
from flint import ctx
from covering import q,pack,check,rational_householder

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);a=ap.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=False);ctx.prec=1024;ctx.threads=1;start=time.monotonic()
 u,s=rational_householder([Fraction(1,32)]*896);rho=q(1,16);branches=[(q(-1),q(-2,5)),(q(2,5),q(1))];rows=[]
 for name,factor,quadratic in [('positive',q(1,10),True),('transverse_negative',q(3,2),True),('contracting_negative',q(1,10),False)]:
  image=lambda t:4*t*t-2 if quadratic else t/2
  transitions=[]
  for i,(lo,hi) in enumerate(branches):
   whole=check(image(lo),image(hi),q(-1),q(1),factor*rho,rho)
   for j,(tl,th) in enumerate(branches):transitions.append({'source':i,'target':j,**check(image(lo),image(hi),tl,th,factor*rho,rho)})
   if name=='positive':
    assert whole['covered'];assert image(q(1))-1==q(1);assert (-1-image(q(2,5))).overlaps(q(9,25))
  expected=name=='positive';assert all(v['covered']==expected for v in transitions)
  if name=='transverse_negative':assert all(v['exit_pass'] and not v['stable_pass'] for v in transitions)
  if name=='contracting_negative':assert all(not v['exit_pass'] and v['stable_pass'] for v in transitions)
  rows.append({'name':name,'expected_covered':expected,'transverse_operator_bound':pack(factor),'transitions':transitions})
 result={'passed':True,'precision':1024,'dimension':896,'chart':'Q=I-2vv^T/(v^Tv), all v_i=1/32; exact rational orthogonality and inverse Q^T=Q','v_norm_squared':str(s),'u':[str(x) for x in u],'radius':pack(rho),'positive_union_exit_margins':[pack(q(9,25)),pack(q(1))],'positive_transverse_contraction':pack(q(1,10)),'positive_adjacency':[[1,1],[1,1]],'rows':rows,'seconds':time.monotonic()-start,'rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'scope':'Analytic conjugated product-map covering control only. No Qwen covering relation or entropy certificate.'}
 (out/'report.json').write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main()
