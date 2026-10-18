"""Independent saved-state finite-screen audit; never evaluates the model."""
from pathlib import Path
import json,hashlib,sys,subprocess
import numpy as np
R=Path('/home/d7082791602/.codex/worktrees/llm-hostile-fold-crosscheck/neural-shrimp');D=R/'codex/certificate-ladder-task34';RAW=Path('/data2/neural-shrimp-data/codex/certificate-ladder/settle-001');SUP=RAW.with_name(RAW.name+'-supervision')
sys.path.insert(0,str(R/'codex/certify-chaos'));import supervise as s

def main():
 rc=json.loads((SUP/'receipt.json').read_text());ad=json.loads((SUP/'admission.json').read_text())
 assert rc['source_commit']==ad['source_commit']=='df679dc21bf9ed76f6bfbd37f3541f09af6b5ae9'
 assert rc['returncode']==0 and rc['failure']is None and rc['worker_reaped'] and rc['worker_group_session_absent']
 for e in rc['output_files']:s.verify(e)
 assert {str(p)for p in RAW.iterdir()if p.is_file()}=={e['path']for e in rc['output_files']}
 for e in ad['source_blobs']:
  b=subprocess.check_output(['git','show',ad['source_commit']+':'+e['relative_path']],cwd=R);assert hashlib.sha256(b).hexdigest()==e['sha256']
 for pid in [ad['supervisor']['pid'],rc['worker_pid']]:assert s.ident(pid)is None and not s.members(pid)
 card=json.loads((D/'settle-001-card.json').read_text());aggregate=json.loads((RAW/'report.json').read_text());assert aggregate['completed'] and len(aggregate['cells'])==5;rows=[]
 for c,saved in zip(card['cells'],aggregate['cells']):
  p=c['period'];a=np.load(RAW/(c['name']+'-states.npy'),allow_pickle=False);assert a.shape==(513,896)and a.dtype==np.float64 and np.isfinite(a).all()
  assert np.array_equal(a[0],np.load(c['seed_path'],allow_pickle=False).astype(np.float64))
  assert saved==json.loads((RAW/(c['name']+'-report.json')).read_text()) and saved['cell']==c
  close=[float(np.max(np.abs(a[-p+k]-a[-2*p+k]))/max(1.,float(np.max(np.abs(a[-p+k])))))for k in range(p)]
  divisors=[k for k in range(1,p)if p%k==0];sep={str(k):float(np.max(np.abs(a[-p]-a[-p+k])))for k in divisors};v=saved['diagnostics']
  assert v['relative_inf_closure_each_phase']==close and v['proper_divisor_phase0_separation']==sep
  assert v['finite_closure_pass']==all(x<1e-12 for x in close) and v['finite_primitive_screen_pass']==all(x>1e-8 for x in sep.values())
  rows.append(dict(cell=c,diagnostics=v,states=s.pin(RAW/(c['name']+'-states.npy')),report=s.pin(RAW/(c['name']+'-report.json'))))
 report=dict(receipt=s.pin(SUP/'receipt.json'),source_commit=ad['source_commit'],outputs_checked=len(rc['output_files']),source_blobs_checked=len(ad['source_blobs']),host_closure=True,rows=rows,elapsed_seconds=rc['external_elapsed_seconds'],certificate_passed=False,scope='Every finite phase/divisor diagnostic recomputed from original saved states; no model or stability computation.')
 (D/'settle-001-completion.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps([dict(name=x['cell']['name'],diagnostics=x['diagnostics'])for x in rows]))
if __name__=='__main__':main()
