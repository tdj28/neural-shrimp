"""Bounded saved-evidence inner-entry feasibility, never a standalone release."""
import argparse,json,hashlib,sys
from pathlib import Path
from fractions import Fraction as F
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
from inner_entry import construct

def sha(p):
 h=hashlib.sha256()
 with open(p,'rb')as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def verify(e):
 p=Path(e['path']);assert p.stat().st_size==e['bytes'] and sha(p)==e['sha256'];return p
def exact(v):
 assert v['rad']==[0,0];a,b=v['mid'];return F(a)*F(2)**b
def first_root(path):
 # Parse real top-level keys, never substring-match a nested/string key.
 # Fixed64MiB input cap bounds memory and refuses an unexpected layout.
 with open(path)as f:t=f.read(64*1024**2)
 d=json.JSONDecoder();i=0
 def ws(i):
  while i<len(t) and t[i].isspace():i+=1
  return i
 i=ws(i);assert t[i]=='{';i+=1;seen=set();prefix={}
 while True:
  i=ws(i);k,i=d.raw_decode(t,i);assert isinstance(k,str) and k not in seen;seen.add(k);i=ws(i);assert t[i]==':';i=ws(i+1);v,i=d.raw_decode(t,i)
  if k=='root':return prefix,v
  prefix[k]=v;i=ws(i);assert t[i]==',';i+=1

def encode(v):
 if isinstance(v,F):return [str(v.numerator),str(v.denominator)]
 if isinstance(v,dict):return {k:encode(x)for k,x in v.items()}
 if isinstance(v,list):return [encode(x)for x in v]
 return v

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--card',required=True);ap.add_argument('--out',required=True);a=ap.parse_args();card=json.loads(Path(a.card).read_text());out=Path(a.out);out.mkdir(exist_ok=False);results=[]
 for case in card['cases']:
  for e in case['pins'].values():verify(e)
  pref,root=first_root(case['pins']['assembly_report']['path']);p=case['period'];assert pref['period']==root['period']==p and root['dimension']==896 and root['conditional_gates_passed']
  assembly_card=json.loads(Path(case['pins']['assembly_card']['path']).read_text());proposal=assembly_card['preconditioner'];assert proposal['path']==case['pins']['proposal']['path'] and proposal['sha256']==case['pins']['proposal']['sha256']
  power=json.loads(Path(case['pins']['power_summary']['path']).read_text());assert power['binding']['original_report_sha256']==case['pins']['assembly_report']['sha256'] and power['binding']['contract']==pref['contract'] and power['independent_saved_witness']['passed']
  power4=[x for x in power['powers']if x['power']==4];assert len(power4)==1 and power4[0]['passed'];q=exact(power4[0]['induced_infinity_upper'])
  z=np.load(case['pins']['proposal']['path'],allow_pickle=False);A=z['A'];assert A.shape==(p,896,896) and np.isfinite(A).all();r=[exact(v['radius'])for v in root['phases']];e=[exact(v['image_delta_inf_upper'])for v in root['phases']]
  L=[];row_norms=[];deltas=[]
  for j in range(p):
   ds=[exact(b['factors']['jacobian_difference_norm_upper'])for b in root['block_bounds']if b['phase_column']==j];assert len(ds)==p and len(set(ds))==1
   norms=[sum((F(float(x)) for x in np.abs(row)),F(0))for row in A[j]];row_norms.append(norms);deltas.append(ds[0]);L.append(max(norms)+ds[0])
  answer=construct(r,e,L,4,q);witness=dict(period=p,pins=case['pins'],domain_sha256=root['domain_sha256'],radii=r,root_errors=e,proposal_row_norms=row_norms,jacobian_difference_bounds=deltas,lipschitz=L,result=answer,scope='Saved scalar feasibility; full release authority and independent recombination still required. No model evaluation or fixed-start reachability.')
  (out/f'p{p}.json').write_text(json.dumps(encode(witness),indent=2)+'\n');results.append(dict(period=p,positive_entry=answer['explicit_centered_entry_nonempty'],entry_radius=encode(answer['centered_entry_radius']),certificate_passed=False))
  print(json.dumps(dict(period=p,positive_entry=answer['explicit_centered_entry_nonempty'])),flush=True)
 (out/'summary.json').write_text(json.dumps(results,indent=2)+'\n')
if __name__=='__main__':main()
