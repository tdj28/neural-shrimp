import hashlib,json
import numpy as np
import pytest
from neural_shrimp.cifar_restart_diagnostic import Store
from neural_shrimp.response_operator_storage import emitter,DurabilityError

def test_lossless_dtype_shape_signedzero_and_return_journal(tmp_path):
 store=Store(tmp_path/'raw',limit=2*1024**2);seq=[0];emit=emitter(store,seq,'toy')
 arrays={'zero':np.array([0.,-0.,np.nan,np.inf]),'matrix':np.arange(12,dtype=np.int16).reshape(3,4)}
 emit('return',dict(arrays,iteration=2));row=json.loads((store.root/'events.jsonl').read_text());entry=row['array_bundle'];p=store.root/entry['file']
 assert hashlib.sha256(p.read_bytes()).hexdigest()==entry['sha256'] and p.stat().st_size==entry['bytes']
 with np.load(p,allow_pickle=False)as f:
  for key,value in arrays.items():
   assert f[key].dtype==value.dtype and f[key].shape==value.shape and f[key].tobytes()==value.tobytes()
 assert seq==[1]and row['iteration']==2

def test_no_array_start_still_durable(tmp_path):
 store=Store(tmp_path/'raw',limit=2*1024**2);emitter(store,[0],'toy')('started',{'phase_actions_upper':32})
 row=json.loads((store.root/'events.jsonl').read_text());assert row['phase_actions_upper']==32 and 'array_bundle'not in row

def test_object_and_storage_error_are_not_valueerror_abstentions(tmp_path,monkeypatch):
 store=Store(tmp_path/'raw',limit=2*1024**2);emit=emitter(store,[0],'toy')
 with pytest.raises(DurabilityError):emit('bad',{'x':np.array([object()])})
 monkeypatch.setattr(store,'write',lambda *a,**k:(_ for _ in ()).throw(ValueError('full')))
 with pytest.raises(DurabilityError):emit('return',{'x':np.ones(2)})
