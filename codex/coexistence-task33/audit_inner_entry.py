"""Independent exact scalar recombination of the002 feasibility witness."""
from pathlib import Path
from fractions import Fraction as F
import sys,json,subprocess,hashlib
R=Path('/home/d7082791602/.codex/worktrees/llm-hostile-fold-crosscheck/neural-shrimp');D=R/'codex/coexistence-task33';raw=Path('/data2/neural-shrimp-data/codex/coexistence/inner-entry-002');sup=raw.with_name(raw.name+'-supervision');sys.path.insert(0,str(R/'codex/certify-chaos'));import supervise as s
Q=lambda x:F(int(x[0]),int(x[1]))
def main():
 rc=json.loads((sup/'receipt.json').read_text());ad=json.loads((sup/'admission.json').read_text());assert rc['source_commit']==ad['source_commit'] and rc['returncode']==0 and rc['failure']is None and rc['worker_reaped'] and rc['worker_group_session_absent']
 for e in rc['output_files']:s.verify(e)
 assert {str(p)for p in raw.iterdir()if p.is_file()}=={e['path']for e in rc['output_files']}
 for e in ad['source_blobs']:
  b=subprocess.check_output(['git','show',ad['source_commit']+':'+e['relative_path']],cwd=R);assert hashlib.sha256(b).hexdigest()==e['sha256']
 for pid in [ad['supervisor']['pid'],rc['worker_pid']]:assert s.ident(pid)is None and not s.members(pid)
 rows=[]
 for p in [2,3]:
  x=json.loads((raw/f'p{p}.json').read_text());r=list(map(Q,x['radii']));e=list(map(Q,x['root_errors']));L=list(map(Q,x['lipschitz']));Y=list(map(Q,x['preconditioned_residual_bounds']));q=Q(x['weighted_root_contraction']);assert 0<=q<1
  t=max(Y[i]/r[i]for i in range(p))/(1-q);assert t==Q(x['normalized_root_error'])
  assert e==[min(Q(x['original_root_errors'][i]),r[i]*t)for i in range(p)]
  a=x['result'];eta=Q(a['root_centered_radius']);growth=F(1);caps=[]
  for k,row in enumerate(a['intermediate_rows']):
   assert row['step']==k and row['phase']==k%p and Q(row['growth'])==growth
   clear=r[k%p]-e[k%p];assert Q(row['clearance'])==clear and Q(row['strict_domain_margin'])==clear-growth*eta>0
   if growth:caps.append(clear/(2*growth))
   growth*=L[k%p]
  assert len(a['intermediate_rows'])==4*p+1 and eta==min(caps)
  for j in range(p):assert L[j]==max(map(Q,x['proposal_row_norms'][j]))+Q(x['jacobian_difference_bounds'][j])
  rho=eta-e[0];assert a['explicit_centered_entry_nonempty']==(rho>0) and Q(a['centered_entry_radius'])==max(F(0),rho)
  assert 0<=Q(a['return_upper'])<1 and Q(a['return_margin'])==(1-Q(a['return_upper']))*eta
  rows.append(dict(period=p,positive_centered_entry=rho>0,entry_radius=str(max(F(0),rho)),witness=s.pin(raw/f'p{p}.json')))
  print(p,'entry radius indicative',float(rho))
 result=dict(source_commit=ad['source_commit'],receipt=s.pin(sup/'receipt.json'),outputs_checked=len(rc['output_files']),source_blobs_checked=len(ad['source_blobs']),host_closure=True,elapsed_seconds=rc['external_elapsed_seconds'],independent_scalar_recombination=True,rows=rows,certificate_passed=False,scope='Positive scalar feasibility is not release; original authority crossbinding and theorem release remain. No fixed-start claim.')
 (D/'inner-entry-002-completion.json').write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main()
