"""One-CPU cellC cost profile only; no fate/period conclusions."""
import argparse,json,sys,importlib.util,time,resource
from pathlib import Path
import numpy as np

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--card',required=True);ap.add_argument('--out',required=True);a=ap.parse_args();c=json.loads(Path(a.card).read_text());out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
 import torch
 torch.set_num_threads(1);torch.set_num_interop_threads(1);spec=importlib.util.spec_from_file_location('own_map',c['map_source']);q=importlib.util.module_from_spec(spec);sys.modules['own_map']=q;spec.loader.exec_module(q)
 for p in c['inputs']:assert q.sha256(p['path'])==p['sha256']
 model=q.ManualQwenSoftMap.from_pinned_files();tail=np.load(c['references'],allow_pickle=False)['tail'];u=np.linspace(0,1,17);x=torch.from_numpy((1-u[:,None])*tail[-1,1]+u[:,None]*tail[-1,2]);cache=model.prefix_cache(.0693);times=[];cpu=time.process_time();start=time.monotonic()
 with torch.no_grad():
  for _ in range(16):
   t=time.monotonic();x=model(x,.0693,-.142,cache);times.append(time.monotonic()-t)
 np.save(out/'last-state.npy',x.numpy());report={'completed':True,'threads':1,'batch':17,'steps':16,'step_seconds':times,'elapsed':time.monotonic()-start,'cpu_seconds':time.process_time()-cpu,'rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'scope':'Measured cost only; no fate/period/edge result.'};(out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
if __name__=='__main__':main()
