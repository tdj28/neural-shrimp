"""Join original released authority with independently checked inner-entry witnesses."""
from pathlib import Path
from fractions import Fraction as F
import json,hashlib,subprocess,sys,datetime
R=Path('/home/d7082791602/.codex/worktrees/llm-hostile-fold-crosscheck/neural-shrimp');D=R/'codex/coexistence-task33';RAW=Path('/data2/neural-shrimp-data/codex/coexistence');sys.path.insert(0,str(D))
from inner_entry_evidence import first_root,exact,sha

def read(p):return json.loads(Path(p).read_text())
def pin(p):return dict(path=str(p),bytes=Path(p).stat().st_size,sha256=sha(p))
def check(e):
 assert sha(e['path'])==e['sha256']
 if 'bytes'in e:assert Path(e['path']).stat().st_size==e['bytes']
 return read(e['path'])
def Q(x):return F(int(x[0]),int(x[1]))
def dyadic_floor(x):
 assert x>0;k=x.numerator.bit_length()-x.denominator.bit_length()
 while F(2)**k>x:k-=1
 while F(2)**(k+1)<=x:k+=1
 return k

def main():
 pairpath=D/'pair-certificate-release.json';pair=read(pairpath)
 published=subprocess.check_output(['git','show','4aa46e7d:codex/coexistence-task33/pair-certificate-release.json'],cwd=R)
 assert published==pairpath.read_bytes() and pair['certificate_passed'] and pair['status']=='released'
 completion=read(D/'inner-entry-002-completion.json');assert completion['independent_scalar_recombination'] and completion['host_closure']
 rc=check(completion['receipt']);assert rc['returncode']==0 and rc['failure']is None and rc['worker_reaped'] and rc['worker_group_session_absent']
 inv={e['path']:e for e in rc['output_files']};orbits=[]
 for original,audit in zip(pair['orbits'],completion['rows']):
  p=original['period'];assert audit['period']==p and audit['positive_centered_entry']
  w=check(audit['witness']);assert inv[audit['witness']['path']]==audit['witness']
  for name,e in w['pins'].items():assert sha(e['path'])==e['sha256']
  assert w['pins']['assembly_report']==original['assembly_report'] and w['pins']['power_summary']==original['higher_power_summary']
  ar=check(original['assembly_receipt']);assert ar['returncode']==0 and ar['worker_reaped'] and ar['worker_group_session_absent']
  cardpath=Path(w['pins']['assembly_card']['path']);rel=str(cardpath.relative_to(R));cardbytes=subprocess.check_output(['git','show',ar['source_commit']+':'+rel],cwd=R);assert cardbytes==cardpath.read_bytes();card=read(cardpath)
  assert card['map']==pair['map'] and card['definition']==original['definition'] and card['preconditioner']['path']==w['pins']['proposal']['path'] and card['preconditioner']['sha256']==w['pins']['proposal']['sha256']
  definition=check(original['definition']);review=check(original['review_binding']);assert review['map']==pair['map']
  for e in list(review['reviews'].values())+[review['historical_evaluator_review']]:assert sha(e['path'])==e['sha256']
  prefix,root=first_root(original['assembly_report']['path']);assert prefix['contract']==root['contract'] and w['domain_sha256']==root['domain_sha256']==original['domain_bindings']['domain_sha256']
  for k,v in original['domain_bindings'].items():assert root[k]==v
  power=check(original['higher_power_summary']);assert power['binding']['contract']==prefix['contract'] and power['binding']['original_report_sha256']==original['assembly_report']['sha256']
  q=exact(next(x for x in power['powers']if x['power']==4)['induced_infinity_upper']);assert Q(w['result']['return_upper'])==q<1
  assert [Q(v)for v in w['radii']]==[exact(v['radius'])for v in root['phases']]
  assert [Q(v)for v in w['original_root_errors']]==[exact(v['image_delta_inf_upper'])for v in root['phases']]
  assert [Q(v)for v in w['preconditioned_residual_bounds']]==[exact(v['preconditioned_residual_inf_upper'])for v in root['phases']]
  assert Q(w['weighted_root_contraction'])==max(exact(v['weighted_contraction_upper'])for v in root['phases'])
  for j in range(p):
   ds=[exact(b['factors']['jacobian_difference_norm_upper'])for b in root['block_bounds']if b['phase_column']==j];assert len(ds)==p and all(d==Q(w['jacobian_difference_bounds'][j])for d in ds)
  rho=Q(w['result']['centered_entry_radius']);assert rho==F(audit['entry_radius'])>0;k=dyadic_floor(rho);assert 0<F(2)**k<=rho
  orbits.append(dict(period=p,definition=original['definition'],center_phase=0,radius_exact_dyadic=[1,k],witness=audit['witness'],original_assembly=original['assembly_report'],power_summary=original['higher_power_summary'],domain_bindings=original['domain_bindings'],return_steps=4*p,certificate_passed=True))
 result=dict(schema='codex-task33-explicit-inner-basin-v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),certificate_passed=True,status='released',original_pair=pin(pairpath),scalar_completion=pin(D/'inner-entry-002-completion.json'),theorem_review=pin(D/'inner-entry-release-review.md'),checker=pin(Path(__file__)),map=pair['map'],orbits=orbits,statement='Every full-dimensional real state in each stated closed infinity-norm ball around phase0 of the saved exact center tuple converges to the corresponding certified periodic orbit.',scope='Trusted original Python/FLINT/Arb enclosure execution plus exact rational scalar witness audit. Intermediate phase containment and root-centered return trapping proved; the entire original shooting box need not be invariant. No fixed-start reachability, large basin volume, global geometry or ordinary generation claim.',preserved_attempts=['inner-entry001 conservative root-location bound gave no centered entry','inner-entry002 preflight checkpoint refusal before scientific worker'])
 (D/'inner-basin-release.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps([dict(period=x['period'],radius=x['radius_exact_dyadic'])for x in orbits]))
if __name__=='__main__':main()
