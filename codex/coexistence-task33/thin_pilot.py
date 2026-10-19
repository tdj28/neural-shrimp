"""Bounded thin-point pseudo-orbit and full local defects; no trajectory enclosure.

This is a separate proposal experiment, not a restart of transient-pilot-001.
Every point-image ball and local defect is saved before selecting another point.
Only a separately reviewed whole-tube derivative/remainder bound can turn those
local defects into a true-trajectory error bound. Main requires a new explicit
card, externally bounded supervision and a fresh output directory.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

R=Path('/home/d7082791602/.codex/worktrees/llm-hostile-fold-crosscheck/neural-shrimp')
RAW=Path('/data2/neural-shrimp-data/codex/coexistence')
PREFIX=dict(
 prefix_receipt=str(RAW/'prefix-001-supervision/receipt.json'),
 prefix_receipt_sha256='0e787d0ebd6fb505a202c96fb3cc2429803aa8cc50c5271f728dc3da59af6f12',
 prefix_source_commit='8c5170bce23a3cf054de8638ba047dcabc893518',
 prefix_output=str(RAW/'prefix-001'),
 prefix_admission_path=str(RAW/'prefix-001-supervision/admission.json'),
 prefix_admission_sha256='84be8d159949dc60aa0f3909c6a16ee5a967dab0e40cd717dda2738331bcfa28',
 prefix_map_source_sha256='fcc90849601984506adf26f651067734f9faef408c8f42dffe64bf5c68e98a80')
SCOPE='Point-centre image enclosures and local defects only; not a true-trajectory tube or reachability proof.'

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb')as f:
  for block in iter(lambda:f.read(1024**2),b''):h.update(block)
 return h.hexdigest()

def validate_card(card):
 if (card.get('experiment')!='thin-pseudo-orbit-point-defects-v1'
     or card.get('trajectory_enclosure')is not False
     or card.get('certificate_passed')is not False
     or card.get('repository')!=str(R)
     or type(card.get('precision'))is not int or card['precision']!=2048
     or type(card.get('steps'))is not int or card['steps']!=20
     or card.get('alpha')!=.06468 or card.get('log10T')!=-.148
     or type(card.get('initial_token_id'))is not int or card['initial_token_id']!=198
     or card.get('initial_state')!='exact checkpoint embedding of last prefix token'):
  raise ValueError('explicit new thin-pilot definition required')
 if any(card.get(key)!=value for key,value in PREFIX.items()):
  raise ValueError('original Task33 same-cell2048 prefix authority required')
 inputs=card['inputs'];by_path={item['path']:item for item in inputs}
 if len(by_path)!=len(inputs):raise ValueError('unique input pins required')
 authority=card['producer_initial_state_authority']
 if (authority.get('method')!='SoftLoop.prepare: embedding of last prefix token, K=1'
     or authority.get('scope')!='exact stored checkpoint constants; not bitwise identity with floating execution'
     or authority['source']['path']!=str(R/'tools/explore_llm_soft_loop.py')
     or by_path.get(authority['source']['path'])!=authority['source']):
  raise ValueError('original fixed-start source authority required')
 required={str(Path(__file__).resolve()),str(Path(__file__).resolve().with_name('thin_pseudo_orbit.py')),
  PREFIX['prefix_receipt'],PREFIX['prefix_admission_path'],str(Path(PREFIX['prefix_output'])/'inputs.json'),
  str(Path(PREFIX['prefix_output'])/'prefix-cache.json.gz')}
 if not required<=set(by_path):raise ValueError('thin caller/helper and original prefix input pins required')


def main():
 ap=argparse.ArgumentParser(description=__doc__)
 ap.add_argument('--card',required=True);ap.add_argument('--out',required=True)
 args=ap.parse_args();card_path=Path(args.card);card=json.loads(card_path.read_text());validate_card(card)
 out=Path(args.out)
 if (out.parent!=RAW or not out.name.startswith('thin-pilot-')or out.resolve()!=out
     or out.exists()or out.is_symlink()):raise ValueError('fresh separate owned thin-pilot output required')
 sys.path[:0]=[str(Path(__file__).resolve().parent),str(R/'codex/certify-llm'),str(R/'codex/crosscheck')]
 import torch
 from flint import ctx
 from ball_qwen import BallQwen,exact
 from ball_linear import DyadicWeightCache
 from cache_io import digest,unpack
 from prefix_import import install_prefix
 from qwen_map import ManualQwenSoftMap,PREFIX_IDS,PINNED_SHA256,SNAPSHOT,VECTOR_FILE
 from thin_pseudo_orbit import generate,pack
 def verify():
  for item in card['inputs']:
   path=Path(item['path'])
   if (path.is_symlink()or not path.is_file()or path.resolve()!=path
       or path.stat().st_size!=item['bytes']or digest(path)!=item['sha256']):
    raise ValueError('pinned input mismatch')
  for name,expected in PINNED_SHA256.items():
   if digest(VECTOR_FILE if name=='vectors.npz'else SNAPSHOT/name)!=expected:raise ValueError('asset mismatch')
 verify();torch.set_num_threads(1);torch.set_num_interop_threads(1);ctx.prec=2048;ctx.threads=1
 out.mkdir(exist_ok=False);started=time.monotonic();completed=0;point_times=[]
 def save(name,value):
  nonlocal completed
  value=dict(value,trajectory_enclosure=False,certificate_passed=False)
  if name.startswith('step-'):
   value['point_seconds']=point_times[-1];value['elapsed_seconds']=time.monotonic()-started
  with (out/name).open('x')as stream:json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')
  if name.startswith('step-'):completed=value['step']
 try:
  model=ManualQwenSoftMap.from_pinned_files()
  weights={key:tensor.numpy()for key,tensor in model.weights.items()}
  for value in weights.values():value.flags.writeable=False
  vector=model.vector.numpy();vector.flags.writeable=False
  ball=BallQwen(model.config,weights,PREFIX_IDS,vector,alpha=card['alpha'],log10_temperature=card['log10T'],
   linear_options={'chunk_rows':64,'chunk_cols':896,'cache':DyadicWeightCache(max_elements=262144)})
  if PREFIX_IDS[-1]!=card['initial_token_id']or ball.d!=896 or ball.vocab!=151936 or ball.layers!=24:
   raise ValueError('different exact start or full map dimensions')
  install_prefix(card,ball,PINNED_SHA256,digest,unpack)
  initial=[exact(value)for value in weights['model.embed_tokens.weight'][PREFIX_IDS[-1]]]
  if len(initial)!=896 or any(not value.is_finite()or not value.is_exact()for value in initial):
   raise ValueError('exact full checkpoint initial state required')
  save('admission.json',dict(card_path=str(card_path),card_sha256=sha(card_path),driver_sha256=sha(__file__),
   loop_sha256=sha(Path(__file__).resolve().with_name('thin_pseudo_orbit.py')),
   prefix_authority=PREFIX,producer_initial_state_authority=card['producer_initial_state_authority'],
   alpha_ratio=list(float(card['alpha']).as_integer_ratio()),
   log10_temperature_ratio=list(float(card['log10T']).as_integer_ratio()),
   initial_token_id=198,dimension=896,layers=24,vocabulary=151936,precision=ctx.prec,
   initial_error_exact_zero=True,scope=SCOPE))
  def forward(centre):
   if (ctx.prec,ctx.threads)!=(2048,1):raise ValueError('arithmetic context changed')
   tick=time.monotonic();answer=ball.forward(centre);point_times.append(time.monotonic()-tick);return answer
  final=generate(initial,forward,card['steps'],save)
  verify()
  if (ctx.prec,ctx.threads)!=(2048,1):raise ValueError('arithmetic context changed')
  save('report.json',dict(status='thin_point_defects_completed',steps=completed,precision=ctx.prec,
   seconds=time.monotonic()-started,point_seconds=point_times,final_centre=[pack(value)for value in final],
   local_image_and_defect_radii_preserved=True,midpoint_selection_is_explicit=True,
   trajectory_error_propagated=False,whole_tube_derivatives_evaluated=False,
   reachability_test_performed=False,scope=SCOPE))
 except Exception as error:
  save('failure.json',dict(type=type(error).__name__,message=str(error),completed_steps=completed,
   seconds=time.monotonic()-started,point_seconds=point_times,scope=SCOPE))
  raise

if __name__=='__main__':main()
