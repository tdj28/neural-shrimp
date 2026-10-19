#!/usr/bin/env python3
"""Independent finite saved-data Task38 audit. No model or producer-module import."""
import ast, hashlib, json, os, resource, signal, tarfile, time
from pathlib import Path
resource.setrlimit(resource.RLIMIT_AS,(2*1024**3,2*1024**3))
resource.setrlimit(resource.RLIMIT_CPU,(60,60)); signal.alarm(90); os.nice(10)
import numpy as np
from tokenizers import Tokenizer
R=Path('/home/d7082791602/.codex/worktrees/llm-hostile-fold-crosscheck/neural-shrimp')
D=Path('/data2/neural-shrimp-data'); E=D/'explore'; B=E/'llm-soft-loop/pod64/bands'
OUT=Path('/tmp/task38-root/pv-steering-result-001.json')
assert not OUT.exists()
t0=time.monotonic(); pins=[]
def pin(p):
 p=Path(p); b=p.read_bytes(); h=hashlib.sha256(b).hexdigest(); pins.append({'path':str(p),'bytes':len(b),'sha256':h}); return h
def read(p):
 pin(p); return json.loads(Path(p).read_text())
def array(p):
 pin(p); return np.load(p,allow_pickle=False)
tokenizer_path=D/'models/hf/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775/tokenizer.json'
pin(tokenizer_path); tok=Tokenizer.from_file(str(tokenizer_path))
base=array(B/'pv-p00/scan.npz'); aa=base['alpha']; tt=base['log10_temperature']; bc=base['lyapunov']>.001; bp=base['period']; bargs=read(B/'pv-p00/scan.json')['args']
rows=[]; cms=[]; shapes={}; conditions=[f'p{k:02d}' for k in range(19)]+['s1','s2','s3']
for c in conditions:
 d=B/('pv-p14b' if c=='p14' else 'pv-'+c); z=array(d/'scan.npz'); rec=read(d/'scan.json'); args=rec['args']; lam=z['lyapunov']; per=z['period']; chaos=lam>.001
 gates={'grid':bool(np.array_equal(aa,z['alpha']) and np.array_equal(tt,z['log10_temperature'])),'shape':lam.shape==(128,128) and per.shape==(128,128),'finite_exponents':bool(np.isfinite(lam).all()),'args_common':{k:[bargs[k],v] for k,v in args.items() if k not in ['prefix','start_vector','out','chunk'] and v!=bargs[k]}}
 prefix_info={}
 if c.startswith('p'):
  doc=next((R/'docs/prompt-variants').glob(c+'-*.json')); desc=read(doc); actual=E/args['prefix']; gates['prefix_actual_equals_document']=pin(actual)==hashlib.sha256(doc.read_bytes()).hexdigest()
  if 'prefix_text' in desc:
   ids=tok.encode(desc['prefix_text'],add_special_tokens=False).ids
   gates['token_ids']=ids==desc['prefix_token_ids']; gates['token_count']=len(ids)==desc['prefix_tokens']==rec['prefix_tokens']; gates['executed_text']=rec['prefix_text']==desc['prefix_text']
   prefix_info={'tokens':len(ids),'text':desc['prefix_text'],'token_ids':ids}
  else:
   v=array(E/desc['embeds']); gates['vector_hash']=pins[-1]['sha256']==desc['embeds_sha256']; gates['one_vector']=v.size==896 and rec['prefix_tokens']==1
   prefix_info={'shape':list(v.shape),'norm':float(np.linalg.norm(v)),'embeds':desc['embeds']}
 else:
  v=array(E/args['start_vector']); gates['base_prefix']=args['prefix'] is None and rec['prefix_text']==json.loads((B/'pv-p00/scan.json').read_text())['prefix_text']
  gates['start_vector_document']=pins[-1]['sha256']==pin(R/'docs/prompt-variants'/Path(args['start_vector']).name)
  prefix_info={'shape':list(v.shape),'norm':float(np.linalg.norm(v)),'start_vector':args['start_vector']}
 inter=int((chaos&bc).sum()); union=int((chaos|bc).sum())
 row={'condition':c,'raw_dir':str(d),'gates':gates,'prefix':prefix_info,'chunk':args['chunk'],'chaotic_count':int(chaos.sum()),'chaotic_share':float(chaos.mean()),'min_lambda':float(lam.min()),'max_lambda':float(lam.max()),'detected_period_count':int((per>0).sum()),'periodic_and_positive_exponent_count':int(((per>0)&chaos).sum()),'chaos_without_detected_period_count':int(((per==0)&chaos).sum()),'unclassified_nonchaotic_count':int(((per==0)&~chaos).sum()),'chaos_binary_agreement':float((chaos==bc).mean()),'period_label_agreement':float((per==bp).mean()),'chaos_jaccard':inter/union,'band_shares':{n:float(chaos[m].mean()) for n,m in [('cold',tt<-.5),('middle',(tt>=-.5)&(tt<0)),('hot',tt>=0)]}}
 rows.append(row)
 if c.startswith('p'):cms.append(chaos)
stack=np.stack(cms); freq=stack.mean(0); profiles={}
for name,x in [('all',stack),('text',stack[:16]),('vector',stack[16:])]:
 q=x.mean(0).mean(1); k=int(q.argmax()); profiles[name]={'peak_row':k,'logT':float(tt[k]),'mean_frequency':float(q[k])}
frequency={'at_least_one':float((freq>0).mean()),'half_or_more':float((freq>=.5).mean()),'all':float((freq==1).mean()),'max_cell':float(freq.max()),'profiles':profiles,'cold':float(freq[tt<-.5].mean()),'middle':float(freq[(tt>=-.5)&(tt<0)].mean()),'hot':float(freq[tt>=0].mean())}
# Read original receipts using a whitelist, never copy billing fields.
receipts=[]
for name in ['PV-P14','PV-P14B']:
 p=D/'runpod'/f'{name}-orchestrator.log'; pin(p); parsed=[]
 for line in p.read_text().splitlines():
  try:o=json.loads(line)
  except ValueError:continue
  if isinstance(o,dict):parsed.append(o)
 last=parsed[-1]; keep={k:last[k] for k in ['commit','jobset','runner_returncode','remote_sha256','local_sha256','hash_verified','wall_seconds'] if k in last}; keep['name']=name; keep['upload']={k:last.get('upload',{}).get(k) for k in ['code_sha256','data_sha256','jobset_args']}
 run_dir=Path(last['out']); arc=run_dir/'results.tgz'; pin(arc); members=[]
 with tarfile.open(arc,'r:gz') as tf:
  for m in tf.getmembers():
   if m.isfile() and (m.name.endswith('.log') or m.name.endswith('.json')) and m.size<2_000_000:
    data=tf.extractfile(m).read(); members.append({'name':m.name,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),'oom':b'out of memory' in data.lower(),'cuda_oom':b'CUDA out of memory' in data})
 keep['small_metadata_members']=members; receipts.append(keep)
source=R/'tools/explore_llm_phase_map.py'; pin(source); tree=ast.parse(source.read_text()); vals={}
for n in tree.body:
 if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name) and n.targets[0].id in ['CONCEPTS','NEUTRAL']: vals[n.targets[0].id]=ast.literal_eval(n.value)
vec_rec=read(E/'llm-phase-map/vectors-l12/vectors.json'); vs=array(E/'llm-phase-map/vectors-l12/vectors.npz'); raw=vec_rec['info']['philosophy']['raw_difference_norm']; scale=vec_rec['info']['philosophy']['mean_residual_norm']
steering={'philosophy_texts':vals['CONCEPTS']['philosophy'],'neutral_texts':vals['NEUTRAL'],'text_count':len(vals['CONCEPTS']['philosophy'])+len(vals['NEUTRAL']),'raw_difference_norm':raw,'normalization_norm':scale,'saved_vector_norm':float(np.linalg.norm(vs['philosophy'])),'raw_difference_equivalent_alpha':raw/scale,'quoted_alpha_range':[.0625,.0775],'raw_difference_factors':[.0625*scale/raw,.0775*scale/raw],'normalization_semantics':'Source averages per-text non-padding-token norms, then equally averages eight concept and eight neutral texts; difference uses final-token states after decoder layer12 (zero-based). This audit does not rerun model activations.'}
result={'schema':'task38-independent-pv-steering-saved-audit-v1','scope':'saved arrays, exact prefix identity/tokenization, finite statistics, source recipe; not a model or trajectory certification','rows':rows,'frequency':frequency,'p14_originals':receipts,'steering':steering,'pins':pins,'elapsed_seconds':time.monotonic()-t0,'all_condition_gates_pass':all(all(v is True or v=={} for v in r['gates'].values()) for r in rows),'target_model_calls':0}
OUT.write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps({'output':str(OUT),'sha256':hashlib.sha256(OUT.read_bytes()).hexdigest(),'gates':result['all_condition_gates_pass'],'elapsed_seconds':result['elapsed_seconds'],'frequency':frequency,'steering':{k:v for k,v in steering.items() if k not in ['philosophy_texts','neutral_texts']}}))
