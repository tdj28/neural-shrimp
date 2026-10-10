from pathlib import Path
import importlib.util,json,tempfile,os
import numpy as np
import torch
root=Path(os.environ['INFERENCE_REVIEW_ROOT'])
s=importlib.util.spec_from_file_location('inference_review',root/'tools/explore_saved_lm_inference.py');M=importlib.util.module_from_spec(s);s.loader.exec_module(M)
M.bound_sources=lambda:{};M.FILES=[]
class Fake:
 n_parameters=15776
 def __init__(self,*a):self.calls=0
 def initialize(self,*a):return torch.zeros(15776)
 def forward(self,theta,tokens):
  self.calls+=1
  out=torch.zeros((8,32,66),dtype=theta.dtype)
  if self.calls==2:out[:]=float('nan')
  return out
M.TinyTransformer=Fake
M.inputs=lambda:(_ for _ in ()).throw(AssertionError('target inputs opened'))
base=Path(tempfile.mkdtemp(prefix='inference-synthetic-review-'));out=base/'nonfinite'
assert M.run(out)==2
r=json.loads((out/'result.json').read_text());assert r['attempted_forward_calls']==r['returned_forward_calls']==2
assert np.isnan(np.load(out/'forward-0002-return.npy')).all()
assert len(list(out.glob('*-attempt.json')))==2
class RepeatFake(Fake):
 def forward(self,theta,tokens):
  self.calls+=1
  out=torch.zeros((8,32,66),dtype=theta.dtype)
  if self.calls==14:out[:,:,0]=1
  return out
M.TinyTransformer=RepeatFake
M.inputs=lambda:({'vocabulary':list(range(66)),'training_character_offsets':[4096*j for j in range(8)]},np.zeros((8,32),np.int64),[(28151,.8,np.zeros(15776,np.float32),{'synthetic':True})])
out=base/'repeat';assert M.run(out)==2;r=json.loads((out/'result.json').read_text())
assert r['attempted_forward_calls']==r['returned_forward_calls']==16 and 'repeat differs' in r['error']
assert len(list(out.glob('*-return.npy')))==16 and len(list(out.glob('*-attempt.json')))==16
assert r['cases']==[]
print('2 independent synthetic durability controls passed; no real checkpoints or target model used',base)
