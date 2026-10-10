"""Source-bound saved numerical boundary geometry; no target evaluation."""
import os
os.environ['MPLBACKEND']='Agg'
import argparse,hashlib,json,time
from pathlib import Path
import matplotlib.pyplot as plt
BASE=Path('/data2/neural-shrimp-data/explore/clark-rnn')
PINS={'n6-p7-continuation-001':'e5f7f2ada28cb763f1489270cc7c01df74e4a9854ab0d2943ad6256a6c15dc81','n6-p7-completion-001':'22e6b075b8986a61d967ddd490e8ce949eed7c65b55c98af0110c6e884abac55'}
def main(out):
 start=time.monotonic();out.mkdir(parents=True,exist_ok=False);traces={};counts={}
 for name,pin in PINS.items():
  root=BASE/name;assert hashlib.sha256((root/'manifest.json').read_bytes()).hexdigest()==pin
  m=json.loads((root/'manifest.json').read_text())['files'];p=root/'run/result.json';b=p.read_bytes();assert hashlib.sha256(b).hexdigest()==m['run/result.json']['sha256']
  result=json.loads(b);assert result['status']=='completed';counts[name]=sum(len(t['nodes'])-1 for t in result['traces'])
  for t in result['traces']:
   key=(t['branch'],t['direction']);old=traces.get(key,[]);traces[key]=old+t['nodes'][1:] if old else t['nodes']
 colors={'lower-fold':'#135e8a','upper-fold':'#368c67','lower-flip':'#c64b36'};labels=set();fig,ax=plt.subplots(figsize=(8,5))
 for (name,direction),ns in traces.items():
  assert all(n['qualified_primitive_simple_boundary'] for n in ns)
  label=name.replace('-',' ') if name not in labels else None;labels.add(name)
  ax.plot([n['gain'] for n in ns],[n['delta'] for n in ns],lw=1.8,color=colors[name],label=label)
  ax.scatter(ns[-1]['gain'],ns[-1]['delta'],s=26,marker='s',color=colors[name])
 ax.set(xlabel='Gain g',ylabel='Discrete leak / Euler parameter δ',title='Clark iid fixed matrix: N=6, seed28183, primitive period7')
 ax.text(.02,.03,'613 accepted numerical edges; squares: declared domain/node caps\nNo validated arcs, fold cusp or common-orbit crossing established',transform=ax.transAxes,fontsize=9)
 ax.set_xlim(5.45,12.1);ax.set_ylim(.475,.645);ax.grid(alpha=.2);ax.legend(loc='upper right',frameon=False);fig.tight_layout();fig.savefig(out/'geometry.png',dpi=160);fig.savefig(out/'geometry.pdf');plt.close(fig)
 manifest={'inputs':PINS,'accepted_edges':counts,'scope':'Saved qualified numerical polylines only; no proof of every interpolated point. Independent outcome audit pending.','elapsed_seconds':time.monotonic()-start,'files':{p.name:{'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size} for p in out.iterdir()}}
 assert manifest['elapsed_seconds']<10 and sum(v['bytes'] for v in manifest['files'].values())<2*1024**2
 (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
 for p in out.iterdir():p.chmod(0o444)
 out.chmod(0o555);print(json.dumps(manifest))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);main(p.parse_args().out)
