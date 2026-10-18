"""Task33 original receipt and independent saved scalar witness audit; no model."""
import sys,json,hashlib,subprocess,argparse,time
from pathlib import Path
R=Path('/home/d7082791602/.codex/worktrees/llm-hostile-fold-crosscheck/neural-shrimp');sys.path.insert(0,str(R/'codex/certify-llm'))
import witness_audit as w
from flint import ctx
from cache_io import unpack
from phase_domain import pack_ball,canonical_hash
BASE=Path('/data2/neural-shrimp-data/codex/coexistence/assembly-p2-001');SUP=BASE.with_name(BASE.name+'-supervision')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for b in iter(lambda:f.read(1024**2),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def equal(a,b,label):
 if a!=b:raise ValueError(label)
def main(out):
 start=time.monotonic();out=Path(out);out.mkdir(exist_ok=False);ctx.prec=1024;ctx.threads=1
 receipt=read(SUP/'receipt.json');ad=read(SUP/'admission.json');source='88ed989e67c22582f1e360a1886a1535dc325655'
 equal(receipt['source_commit'],source,'receipt source');equal(ad['source_commit'],source,'admission source')
 assert receipt['returncode']==0 and receipt['failure']is None and receipt['worker_reaped'] and receipt['worker_group_session_absent']
 for e in receipt['output_files']:
  assert Path(e['path']).stat().st_size==e['bytes'];equal(sha(e['path']),e['sha256'],'output')
 equal({str(p)for p in BASE.iterdir()if p.is_file()},{e['path']for e in receipt['output_files']},'whole output set')
 for e in ad['source_blobs']:
  b=subprocess.check_output(['git','show',source+':'+e['relative_path']],cwd=R);equal(hashlib.sha256(b).hexdigest(),e['sha256'],'source blob')
 card=read(R/'codex/coexistence-task33/p2-assembly-001-card.json');definition=read(card['definition']['path']);points=read(BASE/'point-admission.json');derivative=read(BASE/'derivative-admission.json');admission=read(BASE/'admission.json')
 equal(sha(card['definition']['path']),card['definition']['sha256'],'definition');equal(points['definition_sha256'],card['definition']['sha256'],'point definition')
 equal(w.domain_digest(definition['centers'],definition['radii'],2,896),admission['bindings']['domain_sha256'],'exact domain')
 equal(len(derivative['authenticated_attempts']),28,'full28')
 equal([(r['phase_index'],r['direction_indices'])for r in derivative['authenticated_attempts']],[(i,list(range(j,j+64)))for i in range(2)for j in range(0,896,64)],'complete original coverage')
 equal(admission['point_report_sha256'],sha(BASE/'point-admission.json'),'bound points');equal(admission['derivative_report_sha256'],sha(BASE/'derivative-admission.json'),'bound derivative')
 images=[[pack_ball(unpack(v))for v in row]for row in points['original_images']]
 for old,new in zip(sum(points['original_images'],[]),sum(images,[])):
  lo,hi=w.endpoints(old);nl,nh=w.endpoints(new);assert nl<=lo and nh>=hi
 print('original receipt/source/output/native point authority passed; loading full saved witnesses',flush=True)
 result=read(BASE/'conditional-gates.json')
 contract=result['contract'];equal(contract['period'],2,'period');equal(contract['evidence_sha256'],sha(BASE/'admission.json'),'contract admission')
 equal(contract['map_sha256'],canonical_hash(card['map']),'contract map')
 equal(contract['evaluator_source_sha256'],canonical_hash({'point_plan':card['point_plan'],'derivative_source':card['derivative_source_sha256'],'review_binding':card['review_binding']}),'contract evaluator')
 for key,val in admission['bindings'].items():equal(contract[key],val,'contract '+key)
 checked=w.audit(result,centers=definition['centers'],radii=definition['radii'],images=images,contract=contract,jacobian_hashes=derivative['coverage']['phase_matrix_sha256'])
 equal(checked['conditional_gates'],read(BASE/'summary.json')['conditional_gates'],'summary')
 detail=dict(root_contraction=result['root']['contraction_upper'],phase_margins=[dict(phase=x['phase'],inclusion_margin=x['inclusion_margin'],contraction_margin=x['contraction_margin'])for x in result['root']['phases']],attraction=[{k:x[k]for k in ('power','induced_infinity_upper','margin','passed')}for x in result['attraction']['powers']])
 report=dict(certificate_passed=False,receipt_sha256=sha(SUP/'receipt.json'),report_sha256=sha(BASE/'conditional-gates.json'),checker_sha256=sha(w.__file__),source_commit=source,source_blobs_checked=len(ad['source_blobs']),output_files_checked=len(receipt['output_files']),original_image_coordinates_checked=1792,independent_saved_witness=checked,details=detail,seconds=time.monotonic()-start,dense_products_recomputed=False,model_calls=0)
 (out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(checked),flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);args=ap.parse_args();main(args.out)
