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

def measure_cell(loop,x,temps,radii,guard,steps=512):
 """Keep base history when a twin collapses; never invent a finite exponent.

 Codes: 0 measured, 1 zero-distance collapse, 2 nonfinite twin/distance,
 3 disabled after an earlier invalid measurement. NaNs are explicitly masked.
 """
 t=loop.torch
 if any(not np.isfinite(r) or r<=0 for r in radii):
  raise ValueError('Initial perturbation radius must be finite and positive')
 active=[True]*len(radii);states=[];growth=[];codes=[]
 for k in range(steps):
  guard();x,_=loop.step(x,temps)
  if not bool(t.isfinite(x[0]).all()):
   return dict(states=np.array(states),growth=np.array(growth).reshape(-1,len(radii)),growth_status=np.array(codes,dtype=np.uint8).reshape(-1,len(radii)),cell_status='nonfinite_base',failed_step=k)
  states.append(x[0].detach().cpu().numpy().copy());g=[];c=[]
  for j,r in enumerate(radii,1):
   if not active[j-1]:
    g.append(float('nan'));c.append(3);x[j]=x[0];continue
   d=x[j]-x[0];n=d.norm()
   code=2 if not bool(t.isfinite(n)) else (1 if float(n)==0 else 0)
   if code:
    active[j-1]=False;g.append(float('nan'));c.append(code);x[j]=x[0]
   else:
    g.append(float(t.log(n/r)));c.append(0);x[j]=x[0]+d*(r/n)
  growth.append(g);codes.append(c)
 return dict(states=np.array(states),growth=np.array(growth),growth_status=np.array(codes,dtype=np.uint8),cell_status='complete' if all(active) else 'complete_base_censored_growth',failed_step=-1)

def main():
 p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--source-commit',required=True);p.add_argument('--short-indices',type=int,nargs='+');p.add_argument('--short-recovery',action='store_true');p.add_argument('--device',choices=['cpu','cuda'],default='cuda');p.add_argument('--local-no-ack',action='store_true');a=p.parse_args()
 if a.short_recovery:
  if a.short_indices is not None:p.error('Choose short-recovery or explicit indices')
  a.short_indices=list(range(48,256))
 if a.local_no_ack and a.device!='cpu':p.error('No-ack mode is restricted to local CPU validation')
 if a.short_indices is not None and (not a.short_indices or len(set(a.short_indices))!=len(a.short_indices) or any(i<0 or i>=256 for i in a.short_indices)):p.error('Unique indices in 0..255 required')
 out=DATA/'explore'/a.out;out.mkdir(parents=True,exist_ok=False);start=time.monotonic();deadline=start+60*(85 if a.short_recovery else 690)
 def guard():
  if time.monotonic()>deadline:raise TimeoutError('Map numerical deadline reached')
 def emit(phase,**kw):
  line=json.dumps(dict(phase=phase,chunk_start=0,**kw))
  with (out/'progress.jsonl').open('a') as f:f.write(line+'\n')
  print(line,flush=True)
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
  if not a.local_no_ack:wait_for_ack(out/'ack.json',digest,a.source_commit,guard)

 try:
  loop=SoftLoop('qwen0.5b',12,2,'open','Hello!',1,'float32',a.device);t=loop.torch;t.backends.cuda.matmul.allow_tf32=False;t.backends.cudnn.allow_tf32=False
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
  record('runtime.json',dict(source=a.source_commit,torch=t.__version__,gpu=t.cuda.get_device_name() if a.device=='cuda' else None,dtype=str(loop.dtype),prefix_tokens=32767,feedback_tokens=1,model_revision=loop.sm.engine.revision))
  loop.prefill_chunk_size=256;timings=[];cell_statuses=[]
  record('selection.json',dict(short_indices=a.short_indices,local_no_ack=a.local_no_ack,device=a.device,growth_status_codes={'0':'measured','1':'zero_distance','2':'nonfinite_twin_or_distance','3':'disabled'}))
  for label,prefix in ([] if a.short_indices is not None else [('short',short),('maximum',t.tensor([long_ids],device=loop.device))]):
   guard();loop.prefix_ids=prefix;loop.L0=prefix.shape[1];t.cuda.reset_peak_memory_stats();tic=time.monotonic();x=loop.prepare(rows);t.cuda.synchronize();prefill=time.monotonic()-tic;tic=time.monotonic()
   for k in range(128):guard();x,_=loop.step(x,[1.]);assert loop.cache.get_seq_length()==loop.L0
   t.cuda.synchronize();timings.append(dict(label=label,prefill_seconds=prefill,step_seconds=(time.monotonic()-tic)/128,peak_bytes=t.cuda.max_memory_allocated()));record('timing.json',timings);emit('timing',**timings[-1])
  checkpoint()
  for label,prefix in ([('short',short)] if a.short_indices is not None else [('maximum',t.tensor([long_ids],device=loop.device)),('short',short)]):
   loop.prefix_ids=prefix;loop.L0=prefix.shape[1]
   for i,(logT,alpha) in enumerate(grid()):
    if a.short_indices is not None and i not in a.short_indices:continue
    guard();tic=time.monotonic();x=loop.prepare(np.repeat((alpha*v)[None],3,axis=0));temps=[10**logT]*3
    for k in range(256):guard();x,_=loop.step(x,temps)
    t.manual_seed(7+i);direction=t.randn_like(x[0]);direction/=direction.norm();radii=[float(x[0].norm())*e for e in [.001,.0003]]
    x[1]=x[0]+radii[0]*direction;x[2]=x[0]+radii[1]*direction;measurement=measure_cell(loop,x,temps,radii,guard)
    save(f'{label}-cell-{i:03d}.npz',**measurement,log_temperature=logT,alpha=alpha,seconds=time.monotonic()-tic);emit('cell_complete',cell_index=i,step=len(measurement['states']),status=measurement['cell_status'],seconds=time.monotonic()-tic)
    cell_statuses.append(measurement['cell_status'])
    checkpoint()
  record('result.json',dict(success=all(s!='nonfinite_base' for s in cell_statuses),cell_statuses=cell_statuses,seconds=time.monotonic()-start,timings=timings,requested_short_indices=a.short_indices,cells_per_arm=256 if a.short_indices is None else None,qualification='Exploratory finite-window map; repeated filler confounds length and content.'))
  checkpoint(final=True)
 except BaseException as e:
  record('failure.json',dict(error=repr(e),seconds=time.monotonic()-start));raise
if __name__=='__main__':main()
