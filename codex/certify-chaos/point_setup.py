"""Task32 A1: one shared weight load, two rigorous prefixes and point steps."""
import argparse,json,gzip,time,sys,resource,gc,hashlib
from pathlib import Path
import numpy as np

def pack(v):
 if not v.is_finite():raise ValueError('nonfinite enclosure')
 return {'mid':list(map(int,v.mid().man_exp())),'rad':list(map(int,v.rad().man_exp()))}
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--card',required=True);ap.add_argument('--out',required=True);a=ap.parse_args();c=json.loads(Path(a.card).read_text());out=Path(a.out);out.mkdir(parents=True,exist_ok=False);tic=time.monotonic();cpu=time.process_time()
 for p in c['inputs']:assert sha(p['path'])==p['sha256']
 root=Path(c['repository']);sys.path.insert(0,str(root/'codex/crosscheck'));sys.path.insert(0,str(root/'codex/certify-llm'))
 import torch,flint
 from flint import ctx
 from qwen_map import ManualQwenSoftMap,PREFIX_IDS,PINNED_SHA256
 from ball_qwen import BallQwen
 from ball_linear import DyadicWeightCache
 torch.set_num_threads(1);torch.set_num_interop_threads(1);ctx.prec=1024;ctx.threads=1
 model=ManualQwenSoftMap.from_pinned_files();weights={k:t.numpy() for k,t in model.weights.items()}
 for w in weights.values():w.flags.writeable=False
 vector=model.vector.numpy();vector.flags.writeable=False;x=weights['model.embed_tokens.weight'][198].copy();np.save(out/'point-input.npy',x)
 inputs={'source_assets':PINNED_SHA256,'prefix_ids':list(PREFIX_IDS),'precision':1024,'threads':1,'python_flint':flint.__version__,'flint':flint.__FLINT_VERSION__,'point':'exact stored newline embedding token198, setup point only','cells':c['cells']};(out/'inputs.json').write_text(json.dumps(inputs,indent=2)+'\n');results=[]
 for cell in c['cells']:
  target=out/cell['name'];target.mkdir();start=time.monotonic();cpu0=time.process_time();events=(target/'layers.jsonl').open('x');rows=[]
  def progress(kind,index,hidden):
   if not kind.endswith('_end'):return
   values=[q.value for row in hidden for q in row];row={'kind':kind,'layer':index,'max_radius':pack(max(v.rad() for v in values)),'max_abs':pack(max(v.abs_upper() for v in values)),'seconds':time.monotonic()-start};rows.append(row);events.write(json.dumps(row)+'\n');events.flush()
  ball=BallQwen(model.config,weights,PREFIX_IDS,vector,alpha=cell['alpha'],log10_temperature=cell['log10T'],linear_options={'chunk_rows':64,'chunk_cols':896,'cache':DyadicWeightCache(max_elements=262144)},progress=progress)
  prefix=None
  try:
   t=time.monotonic();prefix=ball.prefix_cache();prefix_seconds=time.monotonic()-t
   with gzip.open(target/'prefix-cache.json.gz','wt') as f:json.dump([[[[[pack(q.value) for q in head] for head in token] for token in field] for field in layer] for layer in prefix],f)
   t=time.monotonic();value,tangent=ball.forward(x);step_seconds=time.monotonic()-t;assert len(value)==896 and all(v.is_exact() and v==0 for v in tangent)
   (target/'point-step.json').write_text(json.dumps({'value':[pack(v) for v in value],'tangent':[pack(v) for v in tangent]},indent=2)+'\n')
   assert [(r['kind'],r['layer']) for r in rows]==[(kind,i) for kind in ['prefix_end','step_end'] for i in range(24)]
   result={'cell':cell,'completed':True,'alpha_ratio':list(float(cell['alpha']).as_integer_ratio()),'log10T_ratio':list(float(cell['log10T']).as_integer_ratio()),'prefix_seconds':prefix_seconds,'step_seconds':step_seconds,'seconds':time.monotonic()-start,'cpu_seconds':time.process_time()-cpu0,'max_output_radius':pack(max(v.rad() for v in value)),'rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
  except Exception as e:
   result={'cell':cell,'completed':False,'error':repr(e),'seconds':time.monotonic()-start,'cpu_seconds':time.process_time()-cpu0};(target/'failure.json').write_text(json.dumps(result,indent=2)+'\n')
  finally:events.close()
  (target/'report.json').write_text(json.dumps(result,indent=2)+'\n');results.append(result);del ball,prefix;gc.collect()
 for p in c['inputs']:assert sha(p['path'])==p['sha256']
 (out/'report.json').write_text(json.dumps({'completed':all(v['completed'] for v in results),'cells':results,'seconds':time.monotonic()-tic,'cpu_seconds':time.process_time()-cpu,'scope':'Two exact-real prefix and point enclosures at1024bits; not segment or chaos certification. Single timing per cell, not a timing median.'},indent=2)+'\n')
if __name__=='__main__':main()
