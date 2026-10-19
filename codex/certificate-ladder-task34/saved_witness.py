"""Task34 cascade-P3 original receipt and declared-power saved witness audit.

No model, dense-product replay, target launch or certificate finalization.
The source commit and exact original card are required external CLI inputs.
"""
import sys,json,hashlib,subprocess,argparse,time,importlib.util
from pathlib import Path
R=Path('/home/d7082791602/.codex/worktrees/llm-hostile-fold-crosscheck/neural-shrimp');sys.path.insert(0,str(R/'codex/certify-llm'))
HERE=Path(__file__).resolve().parent
# Select the Task34 declared-power adapter explicitly, not the generic 1/2 module.
_spec=importlib.util.spec_from_file_location('task34_witness_audit',HERE/'witness_audit.py')
w=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(w)
from flint import ctx
from cache_io import unpack
from phase_domain import pack_ball,canonical_hash
BASE=Path('/data2/neural-shrimp-data/codex/certificate-ladder/assembly-cascade-p3-001');SUP=BASE.with_name(BASE.name+'-supervision')
CARD=R/'codex/certificate-ladder-task34/cascade-p3-assembly-001-card.json'
POWERS=[1,2,4,8]
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for b in iter(lambda:f.read(1024**2),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def equal(a,b,label):
 if a!=b:raise ValueError(label)
def expected_input(source,card_path):
 if type(source)is not str or len(source)!=40 or any(c not in '0123456789abcdef'for c in source):raise ValueError('explicit full lowercase source commit required')
 path=Path(card_path)
 if path!=CARD or path.is_symlink()or path.resolve()!=path:raise ValueError('exact original Task34 assembly card path required')
 return path

def main(out,source,card_path):
 start=time.monotonic();card_path=expected_input(source,card_path);out=Path(out);out.mkdir(exist_ok=False);ctx.prec=1024;ctx.threads=1
 receipt=read(SUP/'receipt.json');ad=read(SUP/'admission.json')
 equal(receipt['source_commit'],source,'receipt source');equal(ad['source_commit'],source,'admission source')
 if not(type(receipt['returncode'])is int and receipt['returncode']==0 and receipt['failure']is None and receipt['worker_reaped']is True and receipt['worker_group_session_absent']is True and receipt['worker_identity_absent']is True):raise ValueError('successful original assembly closure required')
 equal(receipt['argv'],ad['argv'],'original assembly command')
 equal(ad['argv'][-4:],['--card',str(card_path),'--out',str(BASE)],'exact original card/output command')
 equal(ad['argv'][-5],str(R/'codex/certificate-ladder-task34/assembly_core.py'),'exact Task34 assembly entry')
 for e in receipt['output_files']:
  assert Path(e['path']).stat().st_size==e['bytes'];equal(sha(e['path']),e['sha256'],'output')
 equal({str(p)for p in BASE.iterdir()if p.is_file()},{e['path']for e in receipt['output_files']},'whole output set')
 for e in ad['source_blobs']:
  b=subprocess.check_output(['git','show',source+':'+e['relative_path']],cwd=R);equal(hashlib.sha256(b).hexdigest(),e['sha256'],'source blob')
 card_blob=subprocess.check_output(['git','show',source+':'+str(card_path.relative_to(R))],cwd=R)
 equal(hashlib.sha256(card_blob).hexdigest(),sha(card_path),'expected source-bound original card')
 matches=[e for e in ad['source_blobs']if e['relative_path']==str(card_path.relative_to(R))]
 equal(len(matches),1,'exact card in original admission sources');equal(matches[0]['sha256'],sha(card_path),'original admitted card')
 card=read(card_path);equal(card['period'],3,'card period');equal(card['cell'],'cascade-p3','card cell');equal(card['powers'],POWERS,'prospectively declared powers')
 definition=read(card['definition']['path']);points=read(BASE/'point-admission.json');derivative=read(BASE/'derivative-admission.json');admission=read(BASE/'admission.json')
 equal(sha(card['definition']['path']),card['definition']['sha256'],'definition');equal(points['definition_sha256'],card['definition']['sha256'],'point definition')
 equal(definition['period'],3,'definition period');equal(definition['map'],card['map'],'definition exact map')
 equal(admission['plan_sha256'],canonical_hash(card),'original assembly plan')
 closures=read(BASE/'closure-admission.json');equal(closures['certificate_passed'],False,'conditional original closures');equal(closures['operational_return'],card['operational_return'],'bound original receiver')
 equal(len(closures['original_closures']),42,'full original closure coverage')
 equal([(v['attempt'],v['source_commit'],v['receipt_sha256'])for v in closures['original_closures']],[(Path(v['attempt']).name,v['source_commit'],v['receipt_sha256'])for v in card['shards']],'all original closures bound to assigned shards')
 if not all(v['original_native_receipt_bytes_joined']is True and v['producer_termination_reported']is True and v['certificate_passed']is False for v in closures['original_closures']):raise ValueError('original native receipt and termination joins required')
 equal(w.domain_digest(definition['centers'],definition['radii'],3,896),admission['bindings']['domain_sha256'],'exact domain')
 equal(len(derivative['authenticated_attempts']),42,'full42')
 equal([(r['phase_index'],r['direction_indices'])for r in derivative['authenticated_attempts']],[(i,list(range(j,j+64)))for i in range(3)for j in range(0,896,64)],'complete original coverage')
 equal(admission['point_report_sha256'],sha(BASE/'point-admission.json'),'bound points');equal(admission['derivative_report_sha256'],sha(BASE/'derivative-admission.json'),'bound derivative')
 images=[[pack_ball(unpack(v))for v in row]for row in points['original_images']]
 for old,new in zip(sum(points['original_images'],[]),sum(images,[])):
  lo,hi=w.endpoints(old);nl,nh=w.endpoints(new);assert nl<=lo and nh>=hi
 print('original receipt/source/output/native point authority passed; loading full saved witnesses',flush=True)
 result=read(BASE/'conditional-gates.json')
 contract=result['contract'];equal(contract['period'],3,'period');equal(contract['evidence_sha256'],sha(BASE/'admission.json'),'contract admission')
 equal(contract['map_sha256'],canonical_hash(card['map']),'contract map')
 equal(contract['evaluator_source_sha256'],canonical_hash({'point_plan':card['point_plan'],'derivative_source':card['derivative_source_sha256'],'review_binding':card['review_binding']}),'contract evaluator')
 for key,val in admission['bindings'].items():equal(contract[key],val,'contract '+key)
 checked=w.audit(result,centers=definition['centers'],radii=definition['radii'],images=images,contract=contract,jacobian_hashes=derivative['coverage']['phase_matrix_sha256'])
 equal(checked['conditional_gates'],read(BASE/'summary.json')['conditional_gates'],'summary')
 detail=dict(root_contraction=result['root']['contraction_upper'],phase_margins=[dict(phase=x['phase'],inclusion_margin=x['inclusion_margin'],contraction_margin=x['contraction_margin'])for x in result['root']['phases']],attraction=[{k:x[k]for k in ('power','induced_infinity_upper','margin','passed')}for x in result['attraction']['powers']])
 report=dict(certificate_passed=False,receipt_sha256=sha(SUP/'receipt.json'),report_sha256=sha(BASE/'conditional-gates.json'),checker_sha256=sha(w.__file__),source_commit=source,card_path=str(card_path),card_sha256=sha(card_path),closure_admission_sha256=sha(BASE/'closure-admission.json'),declared_powers=POWERS,source_blobs_checked=len(ad['source_blobs']),output_files_checked=len(receipt['output_files']),original_image_coordinates_checked=2688,independent_saved_witness=checked,details=detail,seconds=time.monotonic()-start,dense_products_recomputed=False,model_calls=0)
 (out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(checked),flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);ap.add_argument('--source',required=True);ap.add_argument('--card',required=True);args=ap.parse_args();main(args.out,args.source,args.card)
