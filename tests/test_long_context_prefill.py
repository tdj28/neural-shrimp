import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from transformers import Qwen2Config,Qwen2ForCausalLM
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from explore_llm_soft_loop import SoftLoop

def test_chunked_prefill_matches_full_and_resets():
 torch.set_num_threads(1);torch.manual_seed(7)
 m=Qwen2ForCausalLM(Qwen2Config(vocab_size=32,hidden_size=32,intermediate_size=48,num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2,max_position_embeddings=128)).eval()
 l=SoftLoop.__new__(SoftLoop);l.torch=torch;l.model=m;l.dtype=l.state_dtype=torch.float32;l.device='cpu';l.sm=SimpleNamespace(steer=None,steer_extra={});l.extra=None;l.hybrid=False;l.prefix_batch=None;l.prefix_embeds=None;l.prefix_ids=torch.randint(0,32,(1,37));l.L0=37;l.K=1;l.E=m.get_input_embeddings().weight.detach()
 def hook(mod,inp,out):return out if l.sm.steer is None else out+l.sm.steer
 m.model.layers[0].register_forward_hook(hook)
 v=np.ones((2,32),dtype=np.float32)*.01
 x=l.prepare(v);full=l.logits(x);assert l.cache.get_seq_length()==37
 l.prefill_chunk_size=8;x2=l.prepare(v);chunk=l.logits(x2)
 torch.testing.assert_close(full,chunk,rtol=2e-5,atol=2e-6)
 torch.testing.assert_close(chunk,l.logits(x2),rtol=0,atol=0)
 assert l.cache.get_seq_length()==37

def test_pilot_launch_scope():
 from runpod_scan_group import validate_orbit_job
 import pytest
 source='a'*40
 validate_orbit_job(f'lc 35 long_context_pilot.py --source-commit {source} --out llm-soft-loop/pod64/lc',source_commit=source)
 for text in [f'lc 36 long_context_pilot.py --source-commit {source} --out llm-soft-loop/pod64/lc',f'lc 35 long_context_pilot.py --source-commit {source} --out llm-soft-loop/pod64/../bad']:
  with pytest.raises(ValueError):validate_orbit_job(text,source_commit=source)
