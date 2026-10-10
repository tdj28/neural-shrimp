"""Exact rational projection of already independently proved connected attraction."""
import hashlib,json,os,resource,subprocess,sys,time
from pathlib import Path
from fractions import Fraction as F
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'tools/reviews'))
import review_arclength_connection as O
B=Path('/data2/neural-shrimp-data/audits')
SPECS=[('s2-refined-stability/attempt-001','a431de462bdd4256281ffa9e468f2388709dbb21e56fd0572336eb98d9cc34e7'),('s2-refined-stability/chart147-attempt001','7cfb9af04ccc5cc704237b3a20cd4d0e14b8aeb3dd213bfa0000415167df28d4'),('s2-refined-stability/chart148-attempt001','4da1c6eb1926d3deab532f67c877a92deaa6c59ac22a0db0fac34a71f87a99aa'),('s2-refined-stability/chart149-attempt001','4ed7bdb7eff7cee0dce5861fbac312054897aab042d5d3c1981c3b276323fd65'),('s2-refined-stability/chart150-attempt001','e4f0bb1d3da5a142d66083d8b18edc24723fe854faf2dd136d733d30b575b775'),('s2-refined-stability/chart151-attempt001','cee9f119c90be9793be9835b2265da844580decb4ac565d8cfeafc18e1760442'),('s2-correlated-stability/attempt-002','00b9d18ecb197cdbb2f684edef385d8c1b59a9fa27248122a3a032bb7a111245'),('s2-correlated-stability/unit002-attempt001','d5e510582823bf7130566d79f4b16d8b43dbc6e1e16ea23b713872d078dc9cee'),('s2-correlated-stability/unit003-attempt001','a87a7ac617d70a15218b3d9c5dc58f54ba4392ba33f547ada1a181a24f73a8be')]

def interval(v):
 m,r=[F(x[0])*F(2)**x[1]for x in (v['mid'],v['rad'])];assert r>=0;return m-r,m+r

def inner_decimal(left,right,digits=6):
 # Upper bound at left and lower bound at right yield an IVT-guaranteed range.
 low=interval(left)[1];high=interval(right)[0];assert low<high
 scale=10**digits;lo=F(-((-low*scale)//1),scale);hi=F((high*scale)//1,scale)
 assert low<=lo<hi<=high;return lo,hi

def main(out):
 assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT).strip()
 assert os.getpriority(os.PRIO_PROCESS,0)>=10
 out.mkdir(parents=True,exist_ok=False);start=time.monotonic();files={};segments=[];slices=[]
 for name,pin in SPECS:
  p=B/name;inv=O.verify(p,pin);a=O.read(p/'run/outcome.json');launch=O.read(p/'launcher-result.json')
  assert a['passed'] and launch['exit_code']==0 and launch['owned_live_pids']==[] and a['primitive_period']==18 and a['native_parameters']==13
  assert all(x['passed'] for x in a['slices']);intervals=sorted([tuple(map(F,x['exact_endpoints']))for x in a['slices']]);end=tuple(map(F,a['coverage']))
  assert intervals[0][0]==end[0] and intervals[-1][1]==end[1] and all(intervals[i][1]==intervals[i+1][0] for i in range(len(intervals)-1));slices+=intervals;segments.append(end)
  files[name]={'manifest_sha256':pin,'outcome_sha256':O.entry(p/'run/outcome.json')['sha256'],'source_commit':O.read(p/'run/runtime.json')['source_commit'],'slice_count':len(intervals),'coverage':a['coverage'],'inventory':inv}
 segments.sort();slices.sort();assert len(slices)==68 and segments[0][0]==F(17,64) and segments[-1][1]==F(43,128) and all(segments[i][1]==segments[i+1][0] for i in range(8))
 parent=B/'s2-period18-small-march/attempt-001';O.verify(parent,'d6888862e0cc416e7aa7beb5301b76ec8188735d0cf2de09a1870a6e6009cd16')
 left=O.read(parent/'run/saved-repair-384.json');right=O.read(parent/'run/march-7-384.json');assert left['passed'] and right['passed'];lb=left['fresh_endpoints'][0][-1];rb=right['fresh_endpoints'][1][-1]
 lo,hi=inner_decimal(lb,rb);original=O.read(ROOT/'certificates/s2-period18-receipt.json');model=O.read(Path(original['external_root'])/'inputs.json')
 result={'passed':True,'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'slices':68,'charts':9,'native_parameters':13,'primitive_period':18,'exact_full_state_gauge':['17/64','43/128'],'independent_units':files,'same_branch_parent_manifest':'d6888862e0cc416e7aa7beb5301b76ec8188735d0cf2de09a1870a6e6009cd16','endpoint_proof_files':['run/saved-repair-384.json','run/march-7-384.json'],'left_hidden_rate_ball':lb,'right_hidden_rate_ball':rb,'rigorous_hidden_rate_interval':[str(lo),str(hi)],'display_hidden_rate_interval':[f'{float(lo):.6f}',f'{float(hi):.6f}'],'exact_output_rate':model['lr_output'],'theorem':'The independently joined native branch is continuous in its exact affine gauge and every point of the stated closed segment is a locally attracting primitive18 cycle. For each hidden learning rate in the inward-rounded projected interval, continuity and the endpoint inequalities give at least one such cycle at the fixed exact output rate. No monotonicity or global uniqueness assumption is used.','limits':'One attracting interior cross-section, not complete two-parameter boundary topology. Analytic derivative kernels and Arb are shared by the underlying independent arithmetic audits; chart151 original launcher history remains missing as disclosed in its new audit.','seconds':time.monotonic()-start,'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'address_space_cap_bytes':1024**3,'model_calls':0}
 (out/'result.json').write_text(json.dumps(result,indent=2)+'\n');(out/'source.py').write_bytes(Path(__file__).read_bytes());(out/'manifest.json').write_text(json.dumps({'files':{p.name:O.entry(p)for p in out.iterdir()if p.is_file()}},indent=2)+'\n')
 for p in out.iterdir():p.chmod(0o444)
 out.chmod(0o555);print(json.dumps({'passed':True,'slices':68,'hidden_interval':result['display_hidden_rate_interval'],'output':model['lr_output'],'manifest':O.entry(out/'manifest.json')}))
if __name__=='__main__':main(Path(sys.argv[1]))
