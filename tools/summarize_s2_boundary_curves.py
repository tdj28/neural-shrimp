#!/usr/bin/env python3
"""Saved-only numerical fold normal form and parameter-curve diagnostics."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
import argparse,gzip,hashlib,json,subprocess,time,resource
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True);a=parser.parse_args();start=time.monotonic()
 if subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip():raise ValueError('clean source')
 raw=Path('/data2/neural-shrimp-data/explore/s2-period18-boundary-curves/attempt-003-target');manifest=json.loads((raw/'manifest.json').read_text())['files']
 for p,d in manifest.items():
  if sha(raw/p)!=d['sha256']:raise ValueError('input hash')
 result=json.loads((raw/'run/outcome.json').read_text());end=json.loads((raw/'launcher-result.json').read_text())
 if end['exit_code']!=0 or end['owned_group_live_pids'] or [d['accepted'] for d in result['directions']]!=[32,32]:raise ValueError('completed fixed panel')
 nodes=[]
 for f in sorted((raw/'run').glob('node-*.json')):
  d=json.loads(f.read_text());d['file']=f.name;nodes.append(d)
 lookup={tuple(d['point']):d for d in nodes};found={}
 with gzip.open(raw/'run/evaluations.jsonl.gz','rt') as f:
  for line in f:
   row=json.loads(line)
   if row['kind']=='evaluation_return' and tuple(row['z']) in lookup:found[tuple(row['z'])]=row
 if len(found)!=64:raise ValueError('all accepted points joined to returned analytic matrices')
 rows=[]
 for node in nodes:
  if time.monotonic()-start>15:raise TimeoutError('saved analysis cap')
  row=found[tuple(node['point'])];data=row['data'];point=np.asarray(node['point'],dtype=float);m=np.asarray(data['monodromy']);j=np.asarray(data['jacobian']);q=point[15:]
  eig,w=np.linalg.eig(m.T);index=int(np.argmin(abs(eig-1)));left=w[:,index]
  if abs(eig[index]-1)>1e-7 or max(abs(left.imag))>1e-8:raise ValueError('simple real unit multiplier unresolved')
  left=left.real;pair=float(left@q)
  if abs(pair)<1e-8:raise ValueError('left/right pairing unresolved')
  left=left/pair;c=float(left@j[13:26,:13]@q);unfold=left@j[:13,13:15]
  residual=float(np.linalg.norm(m.T@left-left)/max(1,np.linalg.norm(m)*np.linalg.norm(left)))
  tangent=np.asarray(node['tangent']);parameter_tangent=tangent[13:15]*[10,.1]
  rows.append({'node':node['file'],'sign':node['sign'],'index':node['index'],'hidden':point[13],'output':point[14],'c':c,'quadratic_coefficient_c_over_2':c/2,'unfolding_hidden':float(unfold[0]),'unfolding_output':float(unfold[1]),'unfolding_tangent_dot':float(unfold@parameter_tangent),'parameter_tangent_norm':float(np.linalg.norm(parameter_tangent)),'left_vector':left.tolist(),'left_residual':residual,'closure':node['qualification']['closure'],'complement_radius':node['qualification']['complement_radius'],'evaluation_id':row['id']})
 crossings=[]
 for sign in (-1,1):
  branch=sorted([x for x in rows if x['sign']==sign],key=lambda x:x['index'])
  for prev,nxt in zip(branch,branch[1:]):
   if prev['c']*nxt['c']<0:crossings.append({'sign':sign,'from_node':prev['node'],'to_node':nxt['node'],'kind':'numerical_cusp_candidate_sign_change_not_proof'})
 a.out.mkdir(parents=True,exist_ok=False);save(a.out/'diagnostics.json',{'rows':rows,'c_sign_changes':crossings,'scope':'numerical fold branch diagnostics only; no certified cusp, branch connectivity or shrimp','all_input_hashes_verified':True})
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 fig,axs=plt.subplots(1,2,figsize=(10,4),layout='constrained')
 for sign,label in [(-1,'decreasing output direction'),(1,'increasing output direction')]:
  branch=sorted([x for x in rows if x['sign']==sign],key=lambda x:x['index']);h=[x['hidden'] for x in branch];o=[x['output'] for x in branch]
  axs[0].plot(h,o,'.-',label=label);axs[1].plot([x['index']+1 for x in branch],[x['c'] for x in branch],'.-',label=label)
 axs[0].plot(75.76429203111452,.8210596690584651,'ks',label='certified seed fold');axs[0].set(xlabel='Actual hidden learning rate',ylabel='Actual output learning rate',title='Numerical fold continuation, same native phase');axs[0].legend(fontsize=7)
 axs[1].axhline(0,color='gray',lw=.7);axs[1].set(xlabel='Accepted continuation point',ylabel='c = left · D²(F¹⁸)[q,q]',title='Fold quadratic coefficient (numerical)');axs[1].legend(fontsize=7)
 fig.savefig(a.out/'fold-curve.png',dpi=160);fig.savefig(a.out/'fold-curve.pdf');plt.close(fig)
 meta={'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'script_sha256':sha(__file__),'input_root':str(raw),'input_manifest_sha256':sha(raw/'manifest.json'),'node_count':64,'c_sign_changes':crossings,'elapsed_seconds':time.monotonic()-start,'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'fresh_model_or_derivative_acquisitions':0,'not_claimed':['rigorous cusp','closed shrimp anatomy','connection to interior']};save(a.out/'receipt.json',meta)
 save(a.out/'manifest.json',{'files':{f.name:{'sha256':sha(f),'bytes':f.stat().st_size} for f in sorted(a.out.iterdir()) if f.is_file()}})
 for f in a.out.iterdir():f.chmod(0o444)
 a.out.chmod(0o555);print(json.dumps(meta));return 0
if __name__=='__main__':raise SystemExit(main())
