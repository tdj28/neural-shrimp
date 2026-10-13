"""One bounded saved-evidence audit; no Arb/model/dense product execution."""
import time
START=time.monotonic()
import argparse,hashlib,importlib.util,json,os,resource,signal,sys,traceback
from pathlib import Path
from fractions import Fraction
OWNED=Path('/data2/neural-shrimp-data/codex/llm-p8-certificate')
ATTEMPT=OWNED/'final-assembly-002'
WORKTREE=Path('/home/d7082791602/.codex/worktrees/llm-p8-certificate/neural-shrimp')
CHECKER=WORKTREE/'codex/certify-llm/witness_audit.py'
CHECKER_SHA='48098fec18005f5d521dfe6685cf311532814263ceaa982c763aae864b505d90'
PINS={
'receipt':'697911f2e3d20184b1afc90e8c95efc70c0f3c1cc5c1bb671024cb210d324097',
'source':'f2fc9ec1dd1e674cdb9b136bb005d43e613604c5',
'plan':'83cae9fa60a07f74403a29289f1789e440a078ce7ef40937f61c91a70787de09',
'report':'c1f27845aecc77a48cf7afca194953826588570949db0bb99a319413edb6ec19',
'admission':'760d8033e330816865c7b5435debe51791f99aaf7c8bebe068678e3d88e47c20',
'authority':'f612bd036cf12affcbbe5f7225ddf931c2ffb415fef4f444ee2e5236a3b1b855',
'authority_manifest':'75213999135c3cc74af1c9263532aa4dd9d97c9e3909d58ef42ae860e8feb984',
'closure':'94d181cf5b3e67a0da70705451890a1979af92542d3f27819aa8043d959a3803'}

REVIEW_PINS={
'final-evaluator-obligations-review.md':'92cf2b93036b92d18c09944da7e3073d2b358756a93ab43eb41a6e50fdc8961e',
'prefix-asset-authority-review.md':'189a4eb6bdf85fd96acfb478778d69604bb116efb1723d0cb1cb9c1542e4e49f',
'shooting-gates-review.md':'e60e08a8e1a8d12a42d20f3f08ff17492133d04577f06529bfe96c2fe5a67d1a',
'witness-audit-review.md':'20fd5578e7f71226d5854b9152918c532af1a3fe93661504f1ea366ac2a7ede9',
'final-certificate-acceptance.md':'f2f77da1bfcea217b76723f8cab7269fa6d862986983e42619aadba0ae74c974'}

def require(b,msg):
 if not b:raise ValueError(msg)
def equal(a,b,msg):require(type(a)is type(b) and a==b,msg)
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(1024**2),b''):h.update(block)
 return h.hexdigest()
def canonical(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def read(path,pin=None):
 if pin is not None:equal(sha(path),pin,'original file digest: '+str(path))
 return json.loads(Path(path).read_text(),parse_constant=lambda s:(_ for _ in ()).throw(ValueError('nonfinite JSON '+s)))
def write(path,v):
 data=json.dumps(v,indent=2,allow_nan=False)+'\n';require(len(data.encode())<9*1024**2,'compact output limit')
 tmp=path.with_suffix(path.suffix+'.partial');tmp.write_text(data);tmp.replace(path)
def pair(f):
 f=Fraction(f);d=f.denominator;require(d&(d-1)==0,'dyadic only');m,e=f.numerator,-(d.bit_length()-1)
 if m==0:return [0,0]
 while m%2==0:m//=2;e+=1
 return [m,e]
def inventory(receipt):
 rows=receipt['artifacts'];require(type(rows)is list,'artifact inventory');d={r['path']:r for r in rows};require(len(d)==len(rows),'unique artifact paths')
 for k in d:require(not Path(k).is_absolute() and '..'not in Path(k).parts,'relative artifact path')
 return d

def audit(out,event):
 event('authority_admission_started')
 authority_dir=OWNED/'final-assembly-002-authority-001'
 authority=read(authority_dir/'report.json',PINS['authority'])
 read(authority_dir/'manifest.json',PINS['authority_manifest'])
 closure=read(authority_dir/'closure.json',PINS['closure'])
 for k,v in [('schema','codex-completed-attempt-authority-v1'),('passed',True),('receipt_sha256',PINS['receipt']),('source_commit',PINS['source']),('exit_code',0),('timed_out',False),('all_original_groups_absent',True),('certificate_passed',False)]:equal(authority[k],v,'external authority '+k)
 equal(closure['receipt_sha256'],PINS['receipt'],'closure receipt');require(closure['all_original_pids_groups_sessions_absent']is True,'complete original closure')
 receipt=read(ATTEMPT/'receipt.json',PINS['receipt']);files=inventory(receipt)
 for k,v in [('source_commit',PINS['source']),('exit_code',0),('timed_out',False)]:equal(receipt[k],v,'assembly receipt '+k)
 def original(name,pin=None):
  row=files[name];path=ATTEMPT/name
  require(path.is_file() and not path.is_symlink() and path.stat().st_size==row['bytes'],'original file shape '+name)
  if pin is not None:equal(row['sha256'],pin,'external/receipt pin '+name)
  return read(path,row['sha256'])
 plan=original('source/codex/certify-llm/final-assembly-002-plan.json',PINS['plan'])
 admission=original('run/admission-manifest.json',PINS['admission'])
 point=original('run/point-admission.json',admission['point_report_sha256'])
 derivative=original('run/derivative-admission.json',admission['derivative_report_sha256'])
 proposal=original('run/proposal-admission.json',admission['proposal_report_sha256'])
 contract=original('run/contract.json');definition=original('run/candidate-definition.json',plan['definition']['sha256'])
 summary=original('run/gate-summary.json');status=original('run/status.json');point_plan=original('run/point-admission-plan.json',plan['point_admission']['sha256'])
 equal(admission['original_plan_sha256'],PINS['plan'],'admission plan');equal(admission['canonical_plan_sha256'],canonical(plan),'canonical plan')
 equal(admission['source_sha256'],plan['source_sha256'],'complete source pins')
 for name,pin in plan['source_sha256'].items():equal(files['source/'+name]['sha256'],pin,'operationally authenticated source '+name)
 for r in plan['reviews']:equal(files['source/'+r['path']]['sha256'],r['sha256'],'external review '+r['scope'])
 for name,pin in REVIEW_PINS.items():original('source/codex/certify-llm/'+name,pin) if name.endswith('.json') else equal(sha(ATTEMPT/'source/codex/certify-llm'/name),pin,'accepted source/math/asset-prefix authority '+name)
 for k,v in [('map_sha256',plan['map_sha256']),('original_seed',plan['seed']),('shards',plan['shards']),('point_backend',plan['point_backend']),('derivative_backend',plan['derivative_backend']),('assembly_backend',plan['assembly_backend'])]:equal(admission[k],v,'admission '+k)
 equal(canonical(plan['map']),plan['map_sha256'],'exact map recipe hash');equal(plan['gate'],dict(bound_method='factored',powers=[1,2],stability_weights=None),'frozen gates')
 equal(admission['evaluator_bundle']['derivative_map'],plan['map'],'combined exact derivative recipe');equal(point['derivative_map'],plan['map'],'point bridge exact derivative recipe');equal(point['point_map'],admission['evaluator_bundle']['point_map'],'independent point recipe')
 for key in ('point_backend','derivative_backend'):equal(point[key],plan[key],'point/derivative precision bridge '+key)
 equal(derivative['map_sha256'],plan['map_sha256'],'original derivative map hash');equal(derivative['backend_sha256'],canonical(plan['derivative_backend']),'derivative backend hash')
 evaluator_paths=['codex/certify-llm/'+n+'.py'for n in ('derivative_shard','streamed_head','ball_jet','ball_qwen','ball_linear','cache_io','phase_domain','derivative_coverage','shooting_gates','proof_gates')]+['codex/crosscheck/qwen_map.py']
 evaluator=dict(schema='codex-derivative-evaluator-source-bundle-v1',source_sha256={n:plan['source_sha256'][n]for n in evaluator_paths})
 equal(derivative['evaluator_sha256'],canonical(evaluator),'complete immutable evaluator bundle')
 for key in ('attempt','receipt_sha256','map_source_sha256','cache_sha256'):
  equal(derivative['trusted_prefix'][key],plan['derivative_prefix'][key],'derivative prefix '+key);equal(point['derivative_prefix'][key],plan['derivative_prefix'][key],'point bridge derivative prefix '+key)
 for key,bkey in [('attempt','prefix_attempt'),('receipt_sha256','prefix_receipt_sha256'),('map_source_sha256','prefix_map_source_sha256')]:equal(point['point_prefix'][key],point_plan['base_plan'][bkey],'separate2048 prefix '+key)
 equal(point['point_prefix']['precision'],2048,'point prefix precision');equal(derivative['trusted_prefix']['precision'],1024,'derivative prefix precision')
 equal(point['plan_sha256'],plan['point_admission']['sha256'],'original point admission');equal(point['definition_sha256'],plan['definition']['sha256'],'point exact final domain');equal(derivative['design_sha256'],plan['definition']['sha256'],'derivative exact final domain')
 for key,value in [('status','conditional_assembly_completed'),('period',8),('dimension',896),('precision',1024),('threads',1),('model_evaluations',0),('optimizer_actions',0),('report_sha256',PINS['report']),('admission_manifest_sha256',PINS['admission'])]:equal(status[key],value,'complete original assembly '+key)
 equal(status['report_bytes'],850507874,'complete report bytes');equal(files['run/conditional-gates.json']['bytes'],850507874,'original inventory report bytes')
 for d in [admission,point,derivative,proposal,summary,status]:require(d['certificate_passed']is False,'conditional records only')
 require(admission['artifact_authentication_completed']is True and point['artifact_authentication_completed']is True and derivative['artifact_authentication_completed']is True,'completed reviewed readers')
 checker_bytes=CHECKER.read_bytes();equal(hashlib.sha256(checker_bytes).hexdigest(),CHECKER_SHA,'independently reviewed checker bytes');checker_copy=out/'executed-checker.py';checker_copy.write_bytes(checker_bytes);spec=importlib.util.spec_from_file_location('independent_saved_witness_checker',checker_copy);checker=importlib.util.module_from_spec(spec);spec.loader.exec_module(checker)
 c=admission['original_centers'];r=admission['nominal_radii'];images=admission['native_images'];binding=admission['bindings'];coverage=derivative['coverage']
 equal(c,definition['centers'],'original definition centers');equal(r,definition['radii'],'original nominal radii')
 equal(point['original_centers'],c,'original point centers');equal(point['nominal_radii'],r,'point nominal radii')
 equal(binding['original_centers_sha256'],canonical(c),'native original centers hash');equal(binding['original_radii_sha256'],canonical(r),'native radii hash')
 equal(point['original_centers_sha256'],canonical(c),'point original centers hash');equal(point['original_images_sha256'],canonical(point['original_images']),'original image digest')
 equal(point['actual_native_images_sha256'],canonical(images),'native image digest');equal(binding['actual_native_images_sha256'],canonical(images),'actual images downstream')
 dh=checker.domain_digest(c,r,8,896);mh=canonical(dict(period=8,dimension=896,centers=c,radii=r))
 for d in [binding,point,coverage]:equal(d['shooting_domain_sha256'],dh,'exact shooting domain');equal(d['phase_domain_sha256'],mh,'exact phase domain')
 for k,v in [('map_sha256',plan['map_sha256']),('evaluator_source_sha256',canonical(admission['evaluator_bundle'])),('evidence_sha256',PINS['admission']),('period',8),('domain_sha256',dh),('enclosures_sha256',binding['enclosures_sha256']),('proposals_sha256',binding['proposals_sha256'])]:equal(contract[k],v,'external gate contract '+k)
 equal(admission['evaluator_source_sha256'],canonical(admission['evaluator_bundle']),'evaluator bundle')
 equal(proposal['original_seed'],plan['seed'],'original proposal seed');equal(proposal['native_proposals_sha256'],binding['proposals_sha256'],'actual proposals hash')
 require(proposal['conversion']['native_conversion_exact']is True,'executed exact conversion');equal(proposal['conversion']['original_binary64_values_compared'],9*896*896,'every A/B entry converted')
 event('independent_original_point_endpoints_started')
 require(len(point['original_phase_provenance'])==8 and len(images)==8,'eight complete images')
 original_point_count=0
 for i,prov in enumerate(point['original_phase_provenance']):
  equal(prov['phase_index'],i,'original point order');parent=Path(point['parent_attempt'])
  raw=read(parent/prov['point_path'],prov['point_sha256']);inp=read(parent/prov['input_path'],prov['input_sha256'])
  live=read((parent/prov['input_path']).with_name('live-input.json'),prov['live_input_sha256'])
  equal(inp['center'],c[i],'original point input center');equal(live['center'],c[i],'actual original live point center');equal(raw['value'],point['original_images'][i],'original image payload')
  require(len(raw['value'])==len(images[i])==len(live['tangent'])==896,'full point coordinates')
  for j,(old,new) in enumerate(zip(raw['value'],images[i])):
   lo,hi=checker.endpoints(old);nl,nu=checker.endpoints(new);require(nl<=lo and nu>=hi,'native point image narrowed original endpoints')
   equal(checker.point(live['tangent'][j]),Fraction(0),'zero original live tangent');checker.point(c[i][j]);original_point_count+=1
 event('independent_original_receiver_endpoints_started')
 refs=plan['shards'];rows=derivative['authenticated_attempts'];cov=coverage['shards']
 require(len(refs)==len(rows)==len(cov)==112,'all112 retained blocks');equal([(v['phase_index'],v['start_column'])for v in refs],[(i,j)for i in range(8)for j in range(0,896,64)],'canonical full coverage')
 require(coverage['coverage_complete']is True,'completed native collector');equal(coverage['phase_column_counts'],[896]*8,'all896 directions');equal(coverage['total_columns'],7168,'all8x896 directions')
 equal(binding['phase_matrix_sha256'],coverage['phase_matrix_sha256'],'eight actual native Jacobian hashes')
 receivers=0
 for index,(ref,row,cr) in enumerate(zip(refs,rows,cov)):
  phase=ref['phase_index'];columns=list(range(ref['start_column'],ref['start_column']+64))
  for k in ['attempt','receipt_sha256','phase_index']:equal(row[k],ref[k],'assigned receipt '+k)
  equal(row['direction_indices'],columns,'original assigned columns');equal(cr['direction_indices'],columns,'collector assigned columns');equal(cr['phase_index'],phase,'collector phase')
  original_receipt=read(Path(ref['attempt'])/'receipt.json',ref['receipt_sha256']);inv=inventory(original_receipt)
  equal(inv['run/derivative-shard.json']['sha256'],row['original_shard_file_sha256'],'executed original full derivative payload')
  receiver=read(Path(ref['attempt'])/'run/actual-inputs.json',inv['run/actual-inputs.json']['sha256'])
  equal(receiver['phase_index'],phase,'original receiver phase');actual=receiver['actual_inputs'];require(len(actual)==896,'whole original receiver')
  for v in [receiver['actual_inputs_sha256'],row['original_live_inputs_sha256'],cr['original_actual_inputs_sha256']]:equal(v,canonical(actual),'original live receiver digest')
  witnesses=[];full=[];radius=checker.point(r[phase]);require(len(cr['original_endpoint_witnesses'])==896,'all saved receiver witnesses')
  for j,x in enumerate(actual):
   lo,hi=checker.endpoints(x);center=checker.point(c[phase][j]);low=center-radius-lo;high=hi-center-radius
   require(low>=0 and high>=0,'original receiver fails exact nominal containment')
   witnesses.append(dict(coordinate=j,lower_margin=pair(low),upper_margin=pair(high)))
   mid=checker.dyadic(x['mid']);rad=checker.dyadic(x['rad']);err=abs(center-mid)
   full.append(dict(phase=0,coordinate=j,nominal_lower=pair(center-radius),nominal_upper=pair(center+radius),actual_lower=pair(lo),actual_upper=pair(hi),center_rounding_error=pair(err),nominal_radius=pair(radius),actual_radius=pair(rad),lower_margin=pair(low),upper_margin=pair(high),radius_margin=pair(rad-radius-err)))
   receivers+=1
  equal(witnesses,cr['original_endpoint_witnesses'],'every original collector containment witness')
  equal(full,receiver['selected_phase_endpoint_witnesses'],'every original receiver containment witness');equal(canonical(full),row['original_endpoint_witnesses_sha256'],'original endpoint witness hash')
 equal(status['bindings'],binding,'final actual native bindings');equal(summary['bindings'],binding,'summary actual native bindings');equal(summary['admission_manifest_sha256'],PINS['admission'],'summary admission')
 event('full_saved_report_loading_started',bytes=files['run/conditional-gates.json']['bytes'])
 full=original('run/conditional-gates.json',PINS['report']);require(full['certificate_passed']is False,'conditional wrapper only');result=full['conditional_result']
 event('exact_fraction_witness_audit_started')
 checked=checker.audit(result,centers=c,radii=r,images=images,contract=contract,jacobian_hashes=binding['phase_matrix_sha256'])
 equal(checked['conditional_gates'],summary['conditional_gates'],'summary gates independently checked');equal(checked['conditional_gates_passed'],summary['conditional_gates_passed'],'summary aggregate checked')
 equal(status['conditional_gates'],checked['conditional_gates'],'final status gates');equal(status['conditional_gates_passed'],checked['conditional_gates_passed'],'final status aggregate')
 event('exact_fraction_witness_audit_completed')
 return dict(schema='codex-final-assembly-independent-witness-review-v1',certificate_passed=False,audit_complete=True,passed=checked['conditional_gates_passed'],pins=PINS,checker_sha256=CHECKER_SHA,inherited_accepted_review_pins=REVIEW_PINS,script_sha256=sha(__file__),original_point_coordinates_checked=original_point_count,original_receiver_coordinates_checked=receivers,original_shards_bound=112,native_jacobian_hashes=binding['phase_matrix_sha256'],exact_domain_hashes=dict(shooting=dh,phase=mh),witness_audit=checked,complete_operational_authority_inherited=dict(path=str(authority_dir/'report.json'),sha256=PINS['authority']),finite_acceptance_obligations=dict(actual_assets='Inherited actual finite exact-asset authority189a4eb6; no checkpoint replay.',historical_prefixes='Inherited same-map construction authority189a4eb6; both separately pinned successful-prefix stages cross-bound.',complete_actual_evidence='All112 original receipt/coverage/source authority and eight original point records bound; executed reviewed readers validate every finite signed derivative.',non_narrowing='Independently checked original point and receiver endpoints plus exact domain hashes; native-J assignment inherited from authenticated reviewed collector with eight downstream hashes.',mathematical_conclusions='All saved scalar root/primitive/attraction witnesses recomputed with exact Fraction; matrix products remain reviewed authenticated execution.'),native_jacobian_assignment_authority='Inherited exact reviewed successful collector execution: every original derivative endpoint through native assignment and all finite signed columns were checked there. This audit independently binds original receipt inventories and eight native hashes through coverage/admission/contract/monodromy inputs; it does not reconstruct the6,422,528 native Jacobian entries.',proposal_conversion_authority='Inherited successful exact native A/B conversion against all7,225,344 original binary64 values, source/receipt-bound; no new proposal arithmetic.',dense_products_recomputed=False,model_evaluations=0,optimizer_actions=0,scope='Exact Fraction scalar/norm witness and original point/receiver endpoint audit under previously accepted evaluator, actual asset/prefix and arithmetic execution authorities. Separate reviewed release required to issue a certificate.')

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--script-sha256',required=True);a=p.parse_args()
 require(sha(__file__)==a.script_sha256,'externally frozen audit script')
 out=a.output.resolve();require(out==OWNED/'final-assembly-002-witness-audit-001','exclusive fixed audit output')
 nice=os.getpriority(os.PRIO_PROCESS,0);require(nice<=10,'cannot raise priority');os.nice(10-nice)
 cap=min([18*1024**3]+[v for v in resource.getrlimit(resource.RLIMIT_AS)if v!=resource.RLIM_INFINITY]);resource.setrlimit(resource.RLIMIT_AS,(cap,cap))
 out.mkdir(exist_ok=False);sys.dont_write_bytecode=True;stage='started'
 def event(name,**kw):
  nonlocal stage
  stage=name;require(time.monotonic()-START<1800,'audit wall envelope')
  row=dict(event=name,elapsed_seconds=time.monotonic()-START,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,**kw)
  with(out/'events.jsonl').open('a')as f:f.write(json.dumps(row)+'\n');f.flush()
  require(sum(f.stat().st_size for f in out.iterdir()if f.is_file())<=10*1024**2,'audit output ceiling')
 def stopped(signum,frame):raise TimeoutError('owned audit supervisor sent termination')
 old=signal.signal(signal.SIGTERM,stopped)
 try:
  (out/'executed-audit.py').write_bytes(Path(__file__).read_bytes());write(out/'inputs.json',dict(pins=PINS,checker_path=str(CHECKER),checker_sha256=CHECKER_SHA,script_sha256=a.script_sha256,certificate_passed=False,scope='One CPU1/nice10/18GiB/1800s saved-only audit; cost unmeasured, original failures preserved.'))
  report=audit(out,event);report.update(elapsed_seconds=time.monotonic()-START,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss);write(out/'report.json',report);event('completed')
 except BaseException as exc:
  write(out/'failure.json',dict(certificate_passed=False,audit_complete=False,stage=stage,error=repr(exc),traceback=traceback.format_exc(),elapsed_seconds=time.monotonic()-START));raise
 finally:signal.signal(signal.SIGTERM,old)
 if not report['passed']:return 2
 return 0
if __name__=='__main__':sys.exit(main())
