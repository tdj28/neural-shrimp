from pathlib import Path
import json,hashlib,subprocess,sys,datetime
from fractions import Fraction
R=Path('/home/d7082791602/.codex/worktrees/llm-hostile-fold-crosscheck/neural-shrimp');D=R/'codex/coexistence-task33';RAW=Path('/data2/neural-shrimp-data/codex/coexistence');sys.path[:0]=[str(R/'codex/certify-chaos'),str(R/'codex/certify-llm')];import supervise as s;import witness_audit as w

def read(p):return json.loads(Path(p).read_text())
def authenticate_receipt(sup):
 rc=read(sup/'receipt.json');ad=read(sup/'admission.json')
 assert rc['returncode']==0 and rc['failure'] is None and rc['worker_reaped'] and rc['worker_group_session_absent']
 assert rc['source_commit']==ad['source_commit']
 for e in rc['output_files']:s.verify(e)
 for e in ad['source_blobs']:
  blob=subprocess.check_output(['git','show',ad['source_commit']+':'+e['relative_path']],cwd=R);assert hashlib.sha256(blob).hexdigest()==e['sha256']
 for pid in [rc['worker_pid'],ad['supervisor']['pid']]:assert s.ident(pid) is None and not s.members(pid)
 return rc,ad

def upper_text(record,digits=9):
 q=w.point(record,nonnegative=True);scale=10**digits;up=(q.numerator*scale+q.denominator-1)//q.denominator
 return str(up//scale)+'.'+str(up%scale).zfill(digits)

sup=RAW/'power-pair-001-supervision';rc,ad=authenticate_receipt(sup);summary=read(RAW/'power-pair-001/summary.json');card=read(D/'power-pair-001-card.json');assert len(summary)==2
release=dict(schema='codex-task33-coexisting-p2-p3-finite-certificate-v1',status='reviewed',certificate_passed=False,created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),power_execution=dict(source_commit=rc['source_commit'],receipt=s.pin(sup/'receipt.json'),admission=s.pin(sup/'admission.json'),outputs=rc['output_files'],host_groups_absent=True),orbits=[],scope='Trusted Python/FLINT/Arb computer-assisted proof. Separate exact rational scalar, separation and complete power row-norm review; not independent transformer or dense-product replay. Token labels, fixed-start basin reachability, entire-box trapping, global shrimp organization, training utility and ordinary generation excluded.',preserved_failures=['Original P2/P3 attraction powers1/2 insufficient','P3assembly001 refused wrong preparation source before mathematical computation; corrected002 source bound to originals','Original setup/upload/refinement decode refusals retained in corresponding records'])
maps=[]
for period,case,row,attempt in zip((2,3),card['cases'],summary,('assembly-p2-001','assembly-p3-002')):
 assert case['period']==row['binding']['period']==period
 for k in ('assembly_receipt','assembly_report','audit_receipt','audit_report'):s.verify(case[k])
 audit=read(case['audit_report']['path']);flags=audit['independent_saved_witness']['conditional_gates'];assert flags['root_existence_and_local_uniqueness'] and flags['primitive_period']
 assert audit['report_sha256']==row['binding']['original_report_sha256']==case['assembly_report']['sha256']
 assert row['binding']['original_audit_sha256']==case['audit_report']['sha256']
 assert row['report_sha256']==s.pin(RAW/'power-pair-001'/('p%d-powers.json'%period))['sha256']
 assert row['independent_saved_witness']['matrix_entries_checked']==2*896**2
 assert [x['power']for x in row['powers']]==[4,8]
 for x in row['powers']:assert x['passed']==(w.point(x['margin'])>0) and w.point(x['margin'])<=1-w.point(x['induced_infinity_upper'])
 assert row['independent_saved_witness']['passed']==any(x['passed']for x in row['powers'])
 acard=read(D/('p%d-assembly-%s-card.json'%(period,'001'if period==2 else'002')));maps.append(acard['map']);s.verify(dict(acard['definition'],bytes=Path(acard['definition']['path']).stat().st_size))
 binding=read(acard['review_binding']['path']);assert hashlib.sha256(Path(acard['review_binding']['path']).read_bytes()).hexdigest()==acard['review_binding']['sha256'];assert binding['map']==acard['map']
 for review in list(binding['reviews'].values())+[binding['historical_evaluator_review']]:assert hashlib.sha256(Path(review['path']).read_bytes()).hexdigest()==review['sha256']
 adm=read(RAW/attempt/'admission.json');assert row['binding']['contract']['evidence_sha256']==s.pin(RAW/attempt/'admission.json')['sha256']
 release['orbits'].append(dict(period=period,certificate_passed=bool(row['independent_saved_witness']['passed']),definition=acard['definition'],assembly_receipt=case['assembly_receipt'],assembly_report=case['assembly_report'],original_witness_audit=case['audit_report'],review_binding=acard['review_binding'],domain_bindings=adm['bindings'],higher_power_summary=s.pin(RAW/'power-pair-001'/('p%d-summary.json'%period)),attraction_upper_bounds_rounded_up={str(x['power']):upper_text(x['induced_infinity_upper'])for x in row['powers']},witness_power=row['independent_saved_witness']['witness_power']))
assert maps[0]==maps[1];release['map']=maps[0];release['certificate_passed']=all(x['certificate_passed']for x in release['orbits'])
release['status']='released'if release['certificate_passed']else'inconclusive_attraction'
release['statement']='The specified exact-real frozen expected-embedding feedback map has coexisting primitive period-2 and period-3 orbits, each unique as an ordered shooting root in its stated product box and locally attracting.'if release['certificate_passed']else'Both primitive periodic roots verified; full attracting-pair claim not released.'
(D/'power-pair-001-completion.json').write_text(json.dumps(dict(receipt=s.pin(sup/'receipt.json'),source_blobs_checked=len(ad['source_blobs']),outputs_checked=len(rc['output_files']),elapsed_seconds=rc['external_elapsed_seconds'],host_closure=True,orbits=[dict(period=x['period'],attraction=x['certificate_passed'],bounds=x['attraction_upper_bounds_rounded_up'])for x in release['orbits']]),indent=2)+'\n')
(D/'pair-certificate-release.json').write_text(json.dumps(release,indent=2)+'\n');print(json.dumps(dict(certificate_passed=release['certificate_passed'],orbits=[dict(period=x['period'],bounds=x['attraction_upper_bounds_rounded_up'])for x in release['orbits']])))
