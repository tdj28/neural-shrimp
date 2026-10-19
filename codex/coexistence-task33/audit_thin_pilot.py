"""Independent saved thin-pilot endpoints and initial BF16 row; no model replay.

Execution source and original receipt/admission/plan pins are explicit CLI inputs.
The fixed reviewed thin science card and caller/loop hashes remain literal.
"""
import argparse,ast,hashlib,json,math,re,struct,subprocess,sys
from fractions import Fraction as F
from pathlib import Path
R=Path('/home/d7082791602/.codex/worktrees/llm-hostile-fold-crosscheck/neural-shrimp')
D=R/'codex/coexistence-task33';B=Path('/data2/neural-shrimp-data/codex/coexistence')
RAW=B/'thin-pilot-001';SUP=B/'thin-pilot-001-supervision'
AUDIT=B/'thin-pilot-001-audit-001';RESULT=AUDIT/'result.json'
CARD=D/'thin-pilot-001-card.json';PLAN=D/'thin-pilot-001-plan.json'
CARD_SHA='33922427cb39bb72662122ae06c461acdbfc9314344e4e9f8a468d2b47164213'
DRIVER_SHA='a12fd306fb49392c0d4922b6c4d7aabec1dd56e0b0b835373ecf74287ff05cbc'
LOOP_SHA='1e9974f2bd5cddfc0ddfd6b1038c7d046df7f8abdb6f061b67fb764a6189369e'
QWEN_SHA='493a3d1d57fec227f5ba2e3f8f9cfcfdd02d22495d1d44e6449ede901310ae6e'
ASSET=Path('/data2/neural-shrimp-data/models/hf/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775/model.safetensors')
ASSET_SHA='fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe'
DIM=896;STEPS=20
THREADS=('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')

def need(ok,message):
 if not ok:raise ValueError(message)

def pairs(items):
 d={}
 for k,v in items:
  need(k not in d,'duplicate JSON key');d[k]=v
 return d

def read(path):return json.loads(Path(path).read_text(),object_pairs_hook=pairs)
def qrecord(q):return dict(numerator=q.numerator,denominator=q.denominator)
def canonical(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def pin(path):
 path=Path(path);need(path.is_file() and not path.is_symlink(),'regular immutable file required')
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return dict(path=str(path),bytes=path.stat().st_size,sha256=h.hexdigest())
def verify(e):
 actual=pin(e['path']);need(actual['bytes']==e['bytes'] and actual['sha256']==e['sha256'],'original file pin mismatch');return actual

def dyadic(pair):
 need(type(pair)is list and len(pair)==2 and all(type(v)is int for v in pair),'native dyadic integer pair required')
 return F(pair[0])*F(2)**pair[1]

def bounds(record):
 need(type(record)is dict and set(record)=={'mid','rad'},'native midpoint/radius record required')
 m,r=dyadic(record['mid']),dyadic(record['rad']);need(r>=0,'negative native radius')
 return m,r,m-r,m+r

def point(record):
 m,r,_,_=bounds(record);need(r==0,'saved centre must be exact');return m

def flags(record):
 need(record.get('trajectory_enclosure')is False and record.get('certificate_passed')is False,'false trajectory/certificate flags required')

def vector(records,n,exact=False):
 need(type(records)is list and len(records)==n,'complete component vector required')
 return [point(v) if exact else bounds(v) for v in records]

def checked_initial(record,expected,n=DIM):
 flags(record);need(record['step']==0,'initial step differs')
 actual=vector(record['centre'],n,True);need(actual==expected,'initial centre differs from exact checkpoint embedding198')
 return actual

def checked_step(record,previous,step,n=DIM):
 flags(record);need(type(record['step'])is int and record['step']==step,'saved step sequence differs')
 current=vector(record['input_centre'],n,True);need(current==previous,'exact input-centre chain broken')
 images=vector(record['full_image'],n);following=vector(record['next_centre'],n,True);defects=vector(record['defects'],n)
 for (m,r,lo,hi),c,(dm,dr,dlo,dhi) in zip(images,following,defects):
  need(c==m,'next centre is not exact original image midpoint')
  need(dlo<=lo-c and dhi>=hi-c,'signed defect does not enclose full image minus next centre')
 exactmax=max(max(abs(dlo),abs(dhi))for _,_,dlo,dhi in defects)
 _,_,lower,upper=bounds(record['defect_inf_upper'])
 need(lower>=0 and upper>=exactmax,'saved defect bound understates original signed defect endpoints')
 for key in ('point_seconds','elapsed_seconds'):
  need(type(record[key])in (int,float) and math.isfinite(record[key]) and record[key]>=0,'invalid point timing')
 return following,dict(step=step,component_count=n,checks=dict(exact_input_chain=True,full_image_endpoints=True,exact_image_midpoints=True,signed_defect_contains_image_minus_next=True,saved_upper_contains_all_defect_endpoints=True),computed_defect_endpoint_max=qrecord(exactmax),saved_defect_inf_upper_exact=qrecord(upper),next_centre_canonical_sha256=canonical(record['next_centre']),point_seconds=record['point_seconds'],elapsed_seconds=record['elapsed_seconds'])

def bf16(word):
 need(type(word)is int and 0<=word<65536,'BF16 word required')
 sign=-1 if word>>15 else 1;exponent=(word>>7)&255;fraction=word&127
 need(exponent!=255,'nonfinite checkpoint BF16 coefficient')
 return sign*F(fraction if exponent==0 else 128+fraction)*F(2)**(-133 if exponent==0 else exponent-134)

def read_bf16_row(path,expected_sha,shape,row,tensor='model.embed_tokens.weight'):
 """Stream the file hash, then read only the selected stored row; no tensors."""
 asset=pin(path);need(asset['sha256']==expected_sha,'checkpoint hash differs')
 need(len(shape)==2 and all(type(v)is int and v>0 for v in shape) and type(row)is int and 0<=row<shape[0],'exact row/shape required')
 with Path(path).open('rb')as f:
  size=f.read(8);need(len(size)==8,'short safetensors header length');length=struct.unpack('<Q',size)[0]
  need(2<=length<=16*1024**2 and 8+length<=asset['bytes'],'invalid bounded safetensors header length')
  header=json.loads(f.read(length),object_pairs_hook=pairs);entry=header[tensor]
  need(entry['dtype']=='BF16' and entry['shape']==list(shape),'wrong stored embedding dtype/shape')
  offsets=entry['data_offsets'];need(type(offsets)is list and len(offsets)==2 and all(type(v)is int for v in offsets),'integer tensor offsets required')
  start,end=offsets;need(0<=start<end and end-start==shape[0]*shape[1]*2 and 8+length+end<=asset['bytes'],'embedding tensor byte extent differs')
  position=8+length+start+row*shape[1]*2;f.seek(position);raw=f.read(shape[1]*2);need(len(raw)==shape[1]*2,'short embedding row')
 values=[bf16(v[0])for v in struct.iter_unpack('<H',raw)]
 return values,dict(asset=asset,tensor=tensor,stored_dtype='BF16',shape=list(shape),row=row,row_byte_offset=position,row_bytes=len(raw),row_raw_sha256=hashlib.sha256(raw).hexdigest(),exact_rational_values_sha256=canonical([qrecord(v)for v in values]),conversion='independent little-endian BF16 integer decoding to Fraction; no Torch/NumPy/Arb/model loading')

def original_allocation(rc,ad,plan,card,source,plan_pin):
 lane=plan['lanes']['thin'];cmd=['/data2/neural-shrimp/.venv/bin/python','-I','-B',str(D/'thin_pilot.py'),'--card',str(CARD),'--out',str(RAW)]
 need(plan['status']=='reviewed_for_execution' and plan['repository']==str(R) and plan['raw_root']==str(B),'original plan not admitted')
 need([plan['python'],'-I','-B',lane['entry'],*lane['arguments']]==cmd and lane['output']==str(RAW) and lane['supervision']==str(SUP),'original thin allocation differs')
 need(rc['source_commit']==ad['source_commit']==source and rc['argv']==ad['argv']==cmd and rc['lane']==ad['lane']=='thin','original source/command/lane differs')
 need(type(rc['returncode'])is int and all(rc[k]is True for k in ('worker_reaped','worker_identity_absent','worker_group_session_absent')),'original process not closed')
 need(rc['failure']is None or type(rc['failure'])is str,'invalid original failure status')
 pp=Path(ad['plan']['path']);pp=pp if pp.is_absolute()else R/pp
 need(pp==PLAN and all(ad['plan'][k]==plan_pin[k]for k in ('bytes','sha256')),'original plan receipt differs')
 need(ad['limits']==lane['limits'] and plan['cpu_threads']==1 and all(ad['threads'][k]=='1'for k in THREADS),'original limits/threads differ')
 expected={str(Path(e['path']).relative_to(R)):{k:e[k]for k in ('bytes','sha256')}for e in plan['source_files']}
 actual={e['relative_path']:{k:e[k]for k in ('bytes','sha256')}for e in ad['source_blobs']}
 need(len(expected)==len(plan['source_files']) and len(actual)==len(ad['source_blobs']) and expected==actual,'complete original source inventory differs')
 for p,h in [(D/'thin_pilot.py',DRIVER_SHA),(D/'thin_pseudo_orbit.py',LOOP_SHA),(R/'codex/crosscheck/qwen_map.py',QWEN_SHA),(CARD,CARD_SHA)]:need(actual[str(p.relative_to(R))]['sha256']==h,'reviewed original source/card differs')
 inputs={e['path']:e for e in card['inputs']};pre={e['path']:e for e in lane['preflight_pins']}
 need(len(inputs)==len(card['inputs']) and all(pre.get(k)==v for k,v in inputs.items()),'card inputs differ from original preflight pins')
 need(card['steps']==20 and card['precision']==2048 and card['alpha']==.06468 and card['log10T']==-.148 and card['initial_token_id']==198,'thin science definition differs');flags(card)
 return rc['returncode']==0 and rc['failure']is None

def check_admission(record,card,card_pin):
 flags(record)
 need(record['card_path']==str(CARD) and record['card_sha256']==card_pin['sha256'] and record['driver_sha256']==DRIVER_SHA and record['loop_sha256']==LOOP_SHA,'thin internal source/card binding differs')
 need(record['dimension']==896 and record['layers']==24 and record['vocabulary']==151936 and record['precision']==2048 and record['initial_token_id']==198 and record['initial_error_exact_zero']is True,'initial state/full map admission differs')
 need(record['alpha_ratio']==list(float(card['alpha']).as_integer_ratio()) and record['log10_temperature_ratio']==list(float(card['log10T']).as_integer_ratio()),'exact parameter ratios differ')
 need(record['producer_initial_state_authority']==card['producer_initial_state_authority'],'producer initial-state authority differs')
 need(record['prefix_authority']=={k:card[k]for k in record['prefix_authority']} and set(record['prefix_authority'])=={'prefix_receipt','prefix_receipt_sha256','prefix_source_commit','prefix_output','prefix_admission_path','prefix_admission_sha256','prefix_map_source_sha256'},'original prefix authority differs')

def final_status(success,report,failure,rows,final,steps=STEPS,n=DIM):
 count=len(rows)
 if report is not None:
  flags(report);need(report['status']=='thin_point_defects_completed' and report['steps']==steps==count and report['precision']==2048,'incomplete completion report')
  need(vector(report['final_centre'],n,True)==final,'final centre differs from saved chain')
  need(report['local_image_and_defect_radii_preserved']is True and report['midpoint_selection_is_explicit']is True and all(report[k]is False for k in ('trajectory_error_propagated','whole_tube_derivatives_evaluated','reachability_test_performed')),'false final scientific scope')
  need(report['point_seconds']==[v['point_seconds']for v in rows],'point timing coverage differs')
 if failure is not None:
  flags(failure);need(type(failure['type'])is str and type(failure['message'])is str and failure['completed_steps']==count,'failure does not identify the saved partial prefix')
  need(len(failure['point_seconds'])in (count,count+1) and failure['point_seconds'][:count]==[v['point_seconds']for v in rows],'failure timing prefix differs')
 need(not(report is not None and failure is not None),'simultaneous worker success and failure records')
 if success:need(report is not None and failure is None and count==steps,'successful original lacks complete saved chain')
 return 'complete_success' if success else 'closed_partial_failure'

def git_blob(source,path):return subprocess.check_output(['git','show',source+':'+str(path.relative_to(R))],cwd=R)
def file_at_source(source,path,expected):
 e=pin(path);need(e['sha256']==expected and git_blob(source,path)==path.read_bytes(),'exact original source-file binding differs');return e

def argument_admission(a):
 """Pure argument validation before any filesystem or evidence access."""
 need(re.fullmatch('[0-9a-f]{40}',a.source)is not None,'full exact source commit required')
 for value in (a.card_sha256,a.plan_sha256,a.receipt_sha256,a.admission_sha256):need(re.fullmatch('[0-9a-f]{64}',value)is not None,'exact SHA256 required')
 need(Path(a.card)==CARD and a.card_sha256==CARD_SHA and Path(a.plan)==PLAN,'reviewed original card/plan paths required')
 need(Path(a.output)==RESULT,'exact owned thin-pilot-001-audit-001/result.json required')
 return RESULT

def allocate_output(output):
 need(output==RESULT,'exact owned audit result required')
 need(not output.exists() and not output.is_symlink() and not output.parent.exists() and not output.parent.is_symlink(),'fresh absent audit result and parent required')
 output.parent.mkdir(exist_ok=False)

def refusal(a,error):
 record=dict(schema='codex-task33-thin-pilot-saved-audit-refusal-v1',status='audit_refused',error_type=type(error).__name__,message=str(error),requested_source=a.source,requested_card=dict(path=a.card,sha256=a.card_sha256),requested_plan=dict(path=a.plan,sha256=a.plan_sha256),requested_receipt_sha256=a.receipt_sha256,requested_admission_sha256=a.admission_sha256,trajectory_enclosure=False,certificate_passed=False,reachability_proved=False,scope='Audit refusal preserved; no saved evidence is promoted by this record.')
 with (AUDIT/'failure.json').open('x')as f:json.dump(record,f,indent=2,allow_nan=False);f.write('\n')

def run(a):
 output=RESULT
 card_pin=file_at_source(a.source,CARD,a.card_sha256);plan_pin=file_at_source(a.source,PLAN,a.plan_sha256)
 receipt_pin=pin(SUP/'receipt.json');admission_pin=pin(SUP/'admission.json')
 need(receipt_pin['sha256']==a.receipt_sha256 and admission_pin['sha256']==a.admission_sha256,'explicit original receipt/admission binding differs')
 rc=read(receipt_pin['path']);ad=read(admission_pin['path']);card=read(CARD);plan=read(PLAN)
 success=original_allocation(rc,ad,plan,card,a.source,plan_pin)
 for e in ad['source_blobs']:
  data=git_blob(a.source,R/e['relative_path']);need(len(data)==e['bytes'] and hashlib.sha256(data).hexdigest()==e['sha256'],'original source blob differs')
 for e in card['inputs']:verify(e)
 # Source definition only: neither current nor original producer code executes.
 authority=card['producer_initial_state_authority'];producer=Path(authority['source']['path']);current=producer.read_text();original=git_blob(authority['original_source_commit'],producer).decode()
 need(hashlib.sha256(original.encode()).hexdigest()==authority['original_source_sha256'],'original fixed-start authority blob differs')
 for text in (original,current):
  nodes=[n for n in ast.parse(text).body if isinstance(n,ast.ClassDef)and n.name=='SoftLoop'];need(len(nodes)==1 and hashlib.sha256(ast.get_source_segment(text,nodes[0]).encode()).hexdigest()==authority['fixed_start_class_sha256'],'original SoftLoop fixed-start class differs')
 inventory={e['path']:e for e in rc['output_files']};need(len(inventory)==len(rc['output_files']),'duplicate original output paths')
 for text,e in inventory.items():
  p=Path(text);need(p.parent==RAW and p.name in {'admission.json','centre-000.json','report.json','failure.json',*(f'step-{i:03d}.json'for i in range(1,21))},'unexpected original output allocation');verify(e)
 actual={str(p)for p in RAW.rglob('*')if p.is_file()} if RAW.exists()else set();need(actual==set(inventory),'whole original output inventory differs')
 sys.path.insert(0,str(R/'codex/certify-chaos'));import supervise as s
 for pid in (rc['worker_pid'],ad['supervisor']['pid']):need(s.ident(pid)is None and not s.members(pid),'original host closure not established')
 def saved(name):return read(RAW/name)if str(RAW/name)in inventory else None
 internal=saved('admission.json');initial=saved('centre-000.json')
 if internal is not None:check_admission(internal,card,card_pin)
 rows=[];expected=[];initial_result=None;previous=[]
 if initial is not None:
  need(internal is not None,'initial record lacks original internal admission')
  expected,row_authority=read_bf16_row(ASSET,ASSET_SHA,(151936,896),198)
  previous=checked_initial(initial,expected)
  initial_result=dict(file=inventory[str(RAW/'centre-000.json')],component_count=896,exact_checkpoint_row_match=True,embedding=row_authority,source_authority=authority)
 names=[i for i in range(1,21)if str(RAW/f'step-{i:03d}.json')in inventory]
 need(names==list(range(1,len(names)+1)),'saved steps are not a complete initial prefix')
 if names:need(initial_result is not None,'saved steps lack independently bound initial state')
 for i in names:
  file=inventory[str(RAW/f'step-{i:03d}.json')];previous,row=checked_step(read(file['path']),previous,i)
  if rows:need(row['elapsed_seconds']>=rows[-1]['elapsed_seconds'],'elapsed timing order differs')
  row['file']=file;rows.append(row)
 report=saved('report.json');failure=saved('failure.json');status=final_status(success,report,failure,rows,previous)
 candidate=None
 if rows:
  first=rows[0];candidate=dict(step=1,file=first['file'],next_centre_canonical_sha256=first['next_centre_canonical_sha256'],delta0=first['saved_defect_inf_upper_exact'],e1_upper=first['saved_defect_inf_upper_exact'],reason='Initial centre independently equals exact embedding198; first full image and signed local defect enclose F(c0)-c1. This establishes only e1<=delta0, not a multi-step trajectory tube.')
 result=dict(schema='codex-task33-thin-pilot-saved-audit-v1',status=status,original_execution=dict(source_commit=a.source,receipt=receipt_pin,admission=admission_pin,card=card_pin,plan=plan_pin,raw_root=str(RAW),output_files=rc['output_files'],source_blobs=ad['source_blobs'],returncode=rc['returncode'],failure=rc['failure'],host_original_groups_absent=True,elapsed_seconds=rc['external_elapsed_seconds']),initial_state=initial_result,steps_checked=len(rows),steps=rows,entry_candidate=candidate,worker_failure=failure,worker_completion_present=report is not None,original_partial_evidence_preserved=True,trajectory_enclosure=False,certificate_passed=False,reachability_proved=False,scope='Independent exact BF16 initial row and native Fraction centre/image/midpoint/defect/upper chain. Point-image validity inherits reviewed authenticated original evaluator execution. No model replay, propagated trajectory-error proof or basin claim.')
 with output.open('x')as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
 print(json.dumps(dict(status=status,steps_checked=len(rows),initial_row_bound=initial_result is not None,trajectory_enclosure=False)))
def main():
 ap=argparse.ArgumentParser()
 for name in ('source','card','card-sha256','plan','plan-sha256','receipt-sha256','admission-sha256','output'):ap.add_argument('--'+name,required=True)
 a=ap.parse_args();output=argument_admission(a)
 allocate_output(output)
 try:run(a)
 except BaseException as error:
  refusal(a,error)
  raise

if __name__=='__main__':main()
