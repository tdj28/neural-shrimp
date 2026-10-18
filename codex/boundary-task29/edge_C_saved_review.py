import numpy as np,json,pathlib
R=pathlib.Path('/home/d7082791602/.codex/worktrees/llm-hostile-fold-crosscheck/neural-shrimp');c=json.loads((R/'codex/boundary-task29/edge-C-card.json').read_text());o=pathlib.Path('/data2/neural-shrimp-data/explore/sheets/codex-task29/edge-C-001');r=json.loads((o/'report.json').read_text());z=np.load(c['references']);ps=c['periods'];ph=[z['tail'][-p:,j] for p,j in zip(ps,c['reference_columns'])]
def label(t,j,n):
 f=[]
 for i,p in enumerate(ps):
  w=max(64,4*p);x=t[n-w+1:n+1,j];old=t[n-w+1-p:n+1-p,j]
  recurrence=np.max(np.linalg.norm(x-old,axis=1)/np.linalg.norm(x,axis=1));d=np.min(np.linalg.norm(ph[i]-t[n,j],axis=1))/np.linalg.norm(t[n,j])
  if recurrence<1e-8 and d<1e-6:f.append(i)
 return f[0] if len(f)==1 else -1
count=0
for row in r['ledger']:
 t=np.load(o/row['trajectory']);assert len(t)==row['steps']+1 and np.isfinite(t).all()
 for j,l in enumerate(row['labels']):
  assert l>=0 and not row['drift'][j]
  n=row['stops'][j];assert label(t,j,n)==label(t,j,n-32)==label(t,j,len(t)-1)==l
  count+=1
for b in r['results']:
 path=np.load(o/f"b{b['index']}-edge-0.npy");mid=np.load(o/f"b{b['index']}-edge-midpoints.npy");assert np.array_equal(mid,path.mean(axis=1)) and len(mid)==b['edge_points']
p=R/'codex/boundary-task29/edge-C-completion.json';d=json.loads(p.read_text());d['saved_fate_review']={'ledger_rows':len(r['ledger']),'trajectories_classified':count,'all_stopping_previous_and_final_gates_match':True,'both_midpoint_arrays_exact':True,'scope':'Independent saved-state recurrence/reference-distance checks; no model replay, saddle or escape-law verification.'};p.write_text(json.dumps(d,indent=2)+'\n');print(d['saved_fate_review'])
