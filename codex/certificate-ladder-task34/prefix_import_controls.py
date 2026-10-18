import json,gzip,hashlib,importlib.util,sys,tempfile,copy
from pathlib import Path
from types import SimpleNamespace
R=Path('/home/d7082791602/.codex/worktrees/llm-hostile-fold-crosscheck/neural-shrimp');sys.path.insert(0,str(R/'codex/certify-llm'))
from flint import arb,ctx
from cache_io import digest,unpack
s=importlib.util.spec_from_file_location('center',R/'codex/certificate-ladder-task34/prefix_import.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
import flint
ctx.prec=1024
with tempfile.TemporaryDirectory() as d:
 p=Path(d);val={'mid':[3,-2],'rad':[1,-100]}; data=[[[[[val]]],[[[val]]]]]
 meta=dict(cell='cascade-p3',source_commit='a'*40,pinned_assets={},prefix_ids=[1],steering_layer=12,precision=2048,alpha_ratio=list((.0650).as_integer_ratio()),log10_temperature_ratio=list((-.14798).as_integer_ratio()),versions={'python_flint':flint.__version__,'flint':flint.__FLINT_VERSION__},temperature='exp(log(10)*exact_dyadic_log10T)',rotary='real base^(-2j/head_dim), real sin/cos')
 def prepare(md,da):
  (p/'inputs.json').write_text(json.dumps(md))
  with gzip.open(p/'prefix-cache.json.gz','wt') as f:json.dump(da,f)
  r=dict(source_commit='a'*40,returncode=0,failure=None,worker_reaped=True,worker_group_session_absent=True,output_files=[dict(path=str(q),bytes=q.stat().st_size,sha256=digest(q)) for q in [p/'inputs.json',p/'prefix-cache.json.gz']]);(p/'receipt.json').write_text(json.dumps(r))
  (p/'admission.json').write_text(json.dumps(dict(source_commit='a'*40,source_blobs=[dict(relative_path='codex/certify-llm/ball_qwen.py',sha256='b'*64)])))
  return dict(cell='cascade-p3',prefix_admission_path=str(p/'admission.json'),prefix_admission_sha256=digest(p/'admission.json'),prefix_map_source_sha256='b'*64,prefix_receipt=str(p/'receipt.json'),prefix_receipt_sha256=digest(p/'receipt.json'),prefix_source_commit='a'*40,prefix_output=str(p))
 def ball():return SimpleNamespace(prefix_ids=(1,),steering_layer=12,precision=1024,alpha=arb((.0650).as_integer_ratio()[0])/arb((.0650).as_integer_ratio()[1]),log10_temperature=arb((-.14798).as_integer_ratio()[0])/arb((-.14798).as_integer_ratio()[1]),layers=1,kv_heads=1,hdim=1,scalar=lambda v:v)
 c=prepare(meta,data);b=ball();m.install_prefix(c,b,{},digest,unpack);assert b.cache[0][0][0][0][0].contains(unpack(val))
 n=1
 for key,value in [('cell','cascade-p6'),('precision',1024),('steering_layer',11),('prefix_ids',[2]),('alpha_ratio',[0,1]),('log10_temperature_ratio',[0,1]),('versions',{}),('temperature','wrong'),('rotary','wrong'),('pinned_assets',{'x':'y'})]:
  md=copy.deepcopy(meta);md[key]=value;c=prepare(md,data)
  try:m.install_prefix(c,ball(),{},digest,unpack)
  except ValueError:n+=1
  else:raise AssertionError(key)
 c=prepare(meta,[])
 try:m.install_prefix(c,ball(),{},digest,unpack)
 except ValueError:n+=1
 else:raise AssertionError('shape')
 c=prepare(meta,data);c['prefix_receipt_sha256']='0'*64
 try:m.install_prefix(c,ball(),{},digest,unpack)
 except ValueError:n+=1
 else:raise AssertionError('hash')
print(n,'manufactured importer controls passed; no model run')
