"""Exploratory paired maximum/short context map with retrieval barriers."""
import argparse,json,time,hashlib
from pathlib import Path
import numpy as np
from explore_llm_soft_loop import SoftLoop,DATA
from explore_llm_phase_map import load_vectors

def wait_for_ack(path,digest,source,guard,timeout=300):
 end=time.monotonic()+timeout
 while True:
  guard()
  if path.exists():
   ack=json.loads(path.read_text())
   if ack.get('source')==source and ack.get('manifest_sha256')==digest:return
  if time.monotonic()>=end:raise TimeoutError('Verified retrieval acknowledgment missing; refusing next cell')
  time.sleep(1)

def grid():
 return [(float(t),float(a)) for a in np.linspace(0,.1,16) for t in np.linspace(-.3,.3,16)]

def main():
 p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--source-commit',required=True);a=p.parse_args()
 out=DATA/'explore'/a.out;out.mkdir(parents=True,exist_ok=False);start=time.monotonic();deadline=start+60*690
 def guard():
  if time.monotonic()>deadline:raise TimeoutError('Map numerical deadline reached')
 def emit(phase,**kw):
  print(json.dumps(dict(phase=phase,chunk_start=0,**kw)),flush=True)
 def save(name,**kw):
  tmp=out/(name+'.tmp')
  with tmp.open('wb') as f:np.savez(f,**kw)
  tmp.replace(out/name)
 def record(name,d):
  tmp=out/(name+'.tmp');tmp.write_text(json.dumps(d,indent=2)+'\n');tmp.replace(out/name)
 def checkpoint(final=False):
  hashes={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in out.iterdir() if f.is_file() and f.suffix in ('.npz','.json') and f.name not in ('checkpoint.json','ack.json','manifest.json')}
  record('manifest.json',hashes)
  digest=hashlib.sha256((out/'manifest.json').read_bytes()).hexdigest()
  record('checkpoint.json',dict(source=a.source_commit,manifest_sha256=digest,final=final))
  wait_for_ack(out/'ack.json',digest,a.source_commit,guard)

 try:
  loop=SoftLoop('qwen0.5b',12,2,'open','Hello!',1,'float32','cuda');t=loop.torch;t.backends.cuda.matmul.allow_tf32=False;t.backends.cudnn.allow_tf32=False
  v=load_vectors('llm-phase-map/vectors-l12')['philosophy'];save('direction.npz',direction=v)
  short=loop.prefix_ids.clone();rows=np.zeros((1,896),dtype=np.float32)
  x=loop.prepare(rows);z=loop.logits(x);loop.prefill_chunk_size=32;x=loop.prepare(rows);z2=loop.logits(x)
  err=float((z-z2).norm()/z.norm());repeat=float((z2-loop.logits(x)).abs().max());assert err<3e-5 and repeat==0
  record('gate.json',dict(prefill_relative_error=err,repeat_max_error=repeat));emit('prefill_gate',error=err)
  # Explicit token-level repeated prose filler, followed by the unchanged short chat prefix.
  filler='A field notebook records observations of clouds, rivers, plants and stones. Each entry separates what was observed from what remains uncertain. '
  ids=loop.sm.tok(filler,add_special_tokens=False)['input_ids'];length=32767-len(short[0]);long_ids=(ids*((length+len(ids)-1)//len(ids)))[:length]+short[0].tolist()
  assert len(long_ids)==32767 and loop.model.config.max_position_embeddings==32768
  record('prompt.json',dict(token_ids=long_ids,text=loop.sm.tok.decode(long_ids),short_token_ids=short[0].tolist(),construction='Repeated prose token IDs, cut to exact length, followed by unchanged original chat prefix; filler/content confound disclosed.'))
  record('runtime.json',dict(source=a.source_commit,torch=t.__version__,gpu=t.cuda.get_device_name(),dtype=str(loop.dtype),prefix_tokens=32767,feedback_tokens=1,model_revision=loop.sm.engine.revision))
  loop.prefill_chunk_size=256;timings=[]
  for label,prefix in [('short',short),('maximum',t.tensor([long_ids],device=loop.device))]:
   guard();loop.prefix_ids=prefix;loop.L0=prefix.shape[1];t.cuda.reset_peak_memory_stats();tic=time.monotonic();x=loop.prepare(rows);t.cuda.synchronize();prefill=time.monotonic()-tic;tic=time.monotonic()
   for k in range(128):guard();x,_=loop.step(x,[1.]);assert loop.cache.get_seq_length()==loop.L0
   t.cuda.synchronize();timings.append(dict(label=label,prefill_seconds=prefill,step_seconds=(time.monotonic()-tic)/128,peak_bytes=t.cuda.max_memory_allocated()));record('timing.json',timings);emit('timing',**timings[-1])
  checkpoint()
  for label,prefix in [('maximum',t.tensor([long_ids],device=loop.device)),('short',short)]:
   loop.prefix_ids=prefix;loop.L0=prefix.shape[1]
   for i,(logT,alpha) in enumerate(grid()):
    guard();tic=time.monotonic();x=loop.prepare(np.repeat((alpha*v)[None],3,axis=0));temps=[10**logT]*3
    for k in range(256):guard();x,_=loop.step(x,temps)
    t.manual_seed(7+i);direction=t.randn_like(x[0]);direction/=direction.norm();radii=[float(x[0].norm())*e for e in [.001,.0003]]
    x[1]=x[0]+radii[0]*direction;x[2]=x[0]+radii[1]*direction;states=[];growth=[]
    for k in range(512):
     guard();x,_=loop.step(x,temps);states.append(x[0].cpu().numpy());g=[]
     for j,r in enumerate(radii,1):
      d=x[j]-x[0];n=d.norm();assert t.isfinite(n) and n>0;g.append(float(t.log(n/r)));x[j]=x[0]+d*(r/n)
     growth.append(g)
    save(f'{label}-cell-{i:03d}.npz',states=np.array(states),growth=np.array(growth),log_temperature=logT,alpha=alpha,seconds=time.monotonic()-tic);emit('cell_complete',cell_index=i,step=512,seconds=time.monotonic()-tic)
    checkpoint()
  record('result.json',dict(success=True,seconds=time.monotonic()-start,timings=timings,cells_per_arm=256,qualification='Exploratory finite-window map; repeated filler confounds length and content.'))
  checkpoint(final=True)
 except BaseException as e:
  record('failure.json',dict(error=repr(e),seconds=time.monotonic()-start));raise
if __name__=='__main__':main()
