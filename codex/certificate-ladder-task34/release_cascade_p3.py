"""Finite Task34 release from already closed, independently checked witnesses."""
from pathlib import Path
from fractions import Fraction
import json,hashlib,subprocess,datetime,sys
R=Path('/home/d7082791602/.codex/worktrees/llm-hostile-fold-crosscheck/neural-shrimp');D=R/'codex/certificate-ladder-task34';B=Path('/data2/neural-shrimp-data/codex/certificate-ladder');A=B/'assembly-cascade-p3-001';W=B/'assembly-cascade-p3-witness-001'
sys.path[:0]=[str(D),str(R/'codex/certify-chaos')]
import supervise as s
import witness_audit as w

def read(p):return json.loads(Path(p).read_text())
def original(sup,source,root,entry,arguments,required):
 rc=read(sup/'receipt.json');ad=read(sup/'admission.json')
 assert rc['source_commit']==ad['source_commit']==source
 assert rc['returncode']==0 and rc['failure']is None
 assert all(rc[k]is True for k in ['worker_reaped','worker_identity_absent','worker_group_session_absent'])
 assert rc['argv']==ad['argv']==['/data2/neural-shrimp/.venv/bin/python','-I','-B',str(entry),*arguments]
 assert sup==root.with_name(root.name+'-supervision')
 for e in rc['output_files']:s.verify(e)
 assert Path(rc['argv'][-1])==root
 assert {str(p)for p in root.rglob('*')if p.is_file()}=={e['path']for e in rc['output_files']}
 for e in ad['source_blobs']:
  blob=subprocess.check_output(['git','show',source+':'+e['relative_path']],cwd=R)
  assert len(blob)==e['bytes'] and hashlib.sha256(blob).hexdigest()==e['sha256']
 blobs={e['relative_path']:e for e in ad['source_blobs']}
 assert len(blobs)==len(ad['source_blobs'])
 for path,expected in required.items():
  rel=str(path.relative_to(R));assert rel in blobs and blobs[rel]['sha256']==expected and s.pin(path)['sha256']==expected
 for pid in [rc['worker_pid'],ad['supervisor']['pid']]:assert s.ident(pid)is None and not s.members(pid)
 return dict(source_commit=source,receipt=s.pin(sup/'receipt.json'),admission=s.pin(sup/'admission.json'),source_blobs_checked=len(ad['source_blobs']),outputs=rc['output_files'],host_original_groups_absent=True,elapsed_seconds=rc['external_elapsed_seconds']),rc,ad

def upper(record,digits=9):
 q=w.point(record,nonnegative=True);scale=10**digits;v=(q.numerator*scale+q.denominator-1)//q.denominator
 return str(v//scale)+'.'+str(v%scale).zfill(digits)
def pinned(e):
 assert s.pin(e['path'])['sha256']==e['sha256'];return read(e['path'])
def save(p,v):
 with p.open('x')as f:json.dump(v,f,indent=2);f.write('\n')

def main(audit_source):
 assert len(audit_source)==40
 execution,ar,aa=original(A.with_name(A.name+'-supervision'),'fadb2e6d8b5ac5a5be05a68c38b03cc96f30ae3a',A,D/'assembly_core.py',['--card',str(D/'cascade-p3-assembly-001-card.json'),'--out',str(A)],{D/'assembly_core.py':'832dcb9a86c57f8141e4eb3ad308a289f2a7a13af88443600d102776f351a4fc',D/'cascade-p3-assembly-001-card.json':'1a807f1e7630897c20546c707c4fc47e841b8dbc3d413bc5c71be01a7419669c'})
 independent,wr,wa=original(W.with_name(W.name+'-supervision'),audit_source,W,D/'saved_witness.py',['--source',execution['source_commit'],'--card',str(D/'cascade-p3-assembly-001-card.json'),'--out',str(W)],{D/'saved_witness.py':'c851a2ae7f78f692ce68da012b56b5a3ab3265c39d1f2a9ec5c02d3584a54f73',D/'witness_audit.py':'400dafad27e1d99bb3eb3478057dfd6058a6533ef6c56cba2fe82d2f5d4f238b',D/'cascade-p3-witness-001-card.json':s.pin(D/'cascade-p3-witness-001-card.json')['sha256']})
 assert s.pin(W/'report.json') in wr['output_files']
 assert s.pin(A/'summary.json') in ar['output_files'] and s.pin(A/'admission.json') in ar['output_files']
 report=read(W/'report.json');summary=read(A/'summary.json');adm=read(A/'admission.json');card=read(D/'cascade-p3-assembly-001-card.json');wcard=read(D/'cascade-p3-witness-001-card.json')
 assert report['source_commit']==execution['source_commit'];assert report['receipt_sha256']==execution['receipt']['sha256']
 assert report['report_sha256']==s.pin(A/'conditional-gates.json')['sha256']
 assert report['card_sha256']==s.pin(D/'cascade-p3-assembly-001-card.json')['sha256']
 assert report['checker_sha256']==s.pin(D/'witness_audit.py')['sha256']
 assert wcard['original_assembly_receipt']==execution['receipt'] and wcard['original_outputs']==ar['output_files']
 assert report['certificate_passed']is False and summary['certificate_passed']is False
 check=report['independent_saved_witness'];assert check['witness_arithmetic_consistent']is True
 assert check['conditional_gates']==summary['conditional_gates']
 assert check['phase_rows_checked']==3 and check['scalar_blocks_checked']==9 and check['proper_divisors_checked']==[1] and check['matrix_row_norm_entries_checked']==4*896**2
 assert report['declared_powers']==card['powers']==[1,2,4,8]
 flags=check['conditional_gates'];assert set(flags)=={'root_existence_and_local_uniqueness','primitive_period','local_attraction'}
 assert all(v is True for v in flags.values()) and check['conditional_gates_passed']is True,'no positive release from inconclusive gates'
 powers=report['details']['attraction'];assert [p['power']for p in powers]==[1,2,4,8]
 for row in powers:
  q=w.point(row['induced_infinity_upper'],nonnegative=True);margin=w.point(row['margin']);assert margin<=1-q and row['passed']is (margin>0)
 witness=next(p['power']for p in powers if p['passed'])
 definition=pinned(card['definition']);assert definition['map']==card['map'] and definition['period']==card['period']==3
 assert card['map']['alpha_ratio']==[1170935903116329,18014398509481984] and card['map']['log10_temperature_ratio']==[-333221336429143,2251799813685248]
 binding=pinned(card['review_binding']);assert binding['map']==card['map'] and binding['assembly_source_sha256']==card['assembly_source_sha256']
 for ref in list(binding['reviews'].values())+[binding['historical_evaluator_review']]:assert s.pin(ref['path'])['sha256']==ref['sha256']
 pcomp=read(D/'prefix-cascade-p3-001-completion.json');rcomp=read(D/'refine-cascade-p3-001-completion.json')
 assert pcomp['source']=='f63d015412df97724dc1fcb3576a6f469c005137' and pcomp['prefix_balls_checked']==227328 and pcomp['original_groups_absent']is True and pcomp['session_reaped_exit']==0
 assert rcomp['source']=='ae588a9380302bb0e1fb2d22c346324b5809069c' and rcomp['goal_met']is True and rcomp['all_point_witnesses_checked']is True and rcomp['host_original_groups_absent']is True
 assert s.pin(B/'prefix-cascade-p3-001-supervision/receipt.json')['sha256']==pcomp['receipt_sha256']
 assert s.pin(B/'refine-cascade-p3-001-supervision/receipt.json')['sha256']==rcomp['receipt_sha256']
 operational=pinned(card['operational_return']);endpoint=read(D/'return-cascade-p3-endpoints.json')
 assert operational['unique_phase_column_assignments']==2688 and len(operational['attempts'])==42
 assert s.pin(D/'return-cascade-p3-endpoints.json')['sha256']=='6d9be05ee72b7f78d380042804cd7422f8a8945db0d586ac5f7cefb1983ffa02'
 authority=[D/n for n in ['cascade-p3-release-review.md','assembly-source-review.md','witness-source-review.md','prefix-source-review.md','prefix-cascade-p3-001-completion.json','refine-cascade-p3-001-completion.json','return-cascade-p3-operational.json','return-cascade-p3-endpoints.json','cascade-p3-assembly-001-review-binding.json']]+[R/'codex/certify-llm'/n for n in ['prefix-asset-authority-review.md','final-evaluator-obligations-review.md','shooting-gates-review.md','final-certificate-acceptance.md']]
 assert s.pin(R/'codex/certify-llm/prefix-asset-authority-review.md')['sha256']=='189a4eb6bdf85fd96acfb478778d69604bb116efb1723d0cb1cb9c1542e4e49f'
 release=dict(schema='codex-task34-cascade-p3-finite-certificate-v1',status='released',certificate_passed=True,created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),cell='cascade-p3',period=3,map=card['map'],definition=card['definition'],assembly=execution,independent_witness=independent,independent_report=s.pin(W/'report.json'),authorities=[s.pin(p)for p in authority],domain_bindings=adm['bindings'],conditional_gates=flags,declared_powers=[1,2,4,8],witness_power=witness,attraction_upper_bounds_rounded_up={str(p['power']):upper(p['induced_infinity_upper'])for p in powers},power_results=[dict(power=p['power'],passed=p['passed'],exact_upper=p['induced_infinity_upper'])for p in powers],primitive_divisors_checked=[1],shards=42,unique_phase_columns=2688,statement='The specified exact-real frozen expected-embedding feedback map at the exact binary64 parameters alpha=.0650 and log10(T)=-.14798 has a primitive period-3 orbit, unique as an ordered shooting root in its frozen three-phase product box and locally attracting.',scope='Trusted Python/FLINT/Arb computer-assisted proof with authenticated reviewed evaluator execution and independent exact-rational scalar/primitive/all saved power-row witnesses. No independent transformer or dense-product replay. No token labels, fixed-start reachability, full-box trapping, larger basin, parameter-interval stability, complete cascade/ladder, global shrimp geometry, training or ordinary generation claim.',preserved_failures=['Receiver001 public/full Torch version schema refusal','Original H upload/SSH failure before successful H assignment','All original preparation/refinement decode refusals and inconclusive finite power rows retained unchanged'])
 save(D/'cascade-p3-certificate-release.json',release)
 save(D/'cascade-p3-witness-001-completion.json',dict(**independent,session=59470,reaped_exit_code=0,checker_report=s.pin(W/'report.json'),checks=check))
 print(json.dumps({k:release[k]for k in ['certificate_passed','witness_power','attraction_upper_bounds_rounded_up','conditional_gates']}))
if __name__=='__main__':main(sys.argv[1])
