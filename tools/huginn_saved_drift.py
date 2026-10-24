"""CPU-only exploratory closure/drift audit of immutable HPC3 saved states."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import argparse,hashlib,json
from pathlib import Path
import numpy as np

def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--raw',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();a.out.mkdir(exist_ok=False)
 manifest=json.loads((a.raw/'manifest.json').read_text());results=[]
 for index in [1,2,19]:
  carry=None;error_parts=[];offset=0;final=None
  for end in range(1024,8193,1024):
   p=a.raw/f'states-{index:04d}-m-{end:05d}.npz';assert sha(p)==manifest[p.name]
   with np.load(p) as z:states=z['states']
   assert states.shape==(1024,2,12,5280) and np.isfinite(states).all()
   x=np.concatenate([carry,states]) if carry is not None else states
   ncarry=len(x)-len(states);errors=np.full((len(states),3,2),np.nan)
   # Float64 subtraction and accumulation, bounded to 64 state rows at a time.
   for j,q in enumerate([2,48,96]):
    start=max(q,ncarry)
    for k in range(start,len(x),64):
     target=x[k:k+64].astype(np.float64);past=x[k-q:min(k+64,len(x))-q].astype(np.float64)
     errors[k-ncarry:k-ncarry+len(target),j]=np.linalg.norm((target-past).reshape(len(target),2,-1),axis=2)/np.linalg.norm(target.reshape(len(target),2,-1),axis=2)
   error_parts.append(errors);carry=x[-96:].copy();final=states[-256:].copy();offset+=1024
   print(index,end,flush=True)
  errors=np.concatenate(error_parts);spectrum=[]
  for q in range(1,65):
   u=final[-128:].astype(np.float64);v=final[-128-q:-q].astype(np.float64)
   spectrum.append((np.linalg.norm((u-v).reshape(128,2,-1),axis=2)/np.linalg.norm(u.reshape(128,2,-1),axis=2)).max(0))
  spectrum=np.array(spectrum)
  windows={str(length):np.nanmax(errors[-length:],axis=0).tolist() for length in [128,256,1024,4096,8192]}
  blocks=np.array([np.nanmax(b,axis=0) for b in np.split(errors,64)])
  np.savez(a.out/f'drift-{index:04d}.npz',errors=errors,blocks=blocks,spectrum=spectrum)
  results.append(dict(index=index,window_maxima=windows,final_spectrum=spectrum.tolist(),threshold=3e-5))
 (a.out/'summary.json').write_text(json.dumps(dict(results=results,raw=str(a.raw),manifest_sha256=sha(a.raw/'manifest.json'),qualification='Saved float32 trajectories; closure tolerance sensitivity and temporal persistence, not orbit stability or proof of chaos.'),indent=2)+'\n')
if __name__=='__main__':main()
