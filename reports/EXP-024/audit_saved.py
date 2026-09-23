from pathlib import Path
import sys,json,hashlib
import numpy as np
sys.path.insert(0,'/data2/neural-shrimp/tools')
import run_phase_memory_recovery as codec
root=Path(sys.argv[1]); rows=json.loads((root/'analysis.json').read_text())
manifest=json.loads((root/'manifest.json').read_text())
reference_root=Path(manifest.get('reference_root',str(root)))
seals=codec.load_packet(reference_root/'observer-seal.json')
provenance=json.loads((root/'finite-provenance.json').read_text()) if (root/'finite-provenance.json').exists() else None
stage_name='heldout' if provenance is not None else manifest['plan']['contract']['stage']
benchmark_indices=range(8,12) if stage_name=='heldout' else range(4)
assert len(rows)==(768 if stage_name=='heldout' else 192)
signs=[(1,1),(1,-1),(-1,1),(-1,-1)];parity=np.array([1.,-1.,-1.,1.]);sigma=1e-4
max_score=max_relative=max_oracle=max_contrast=0.;checks=0
for row in rows:
    cid=row['context'];dose=[1e-4,1e-3,1e-2].index(row['amplitude'])
    stage='benchmark' if row['phase_index'] in benchmark_indices and dose==2 else 'finite'
    vectors=[]
    for a,b in signs:
        name=f'{cid}--finite--primary--dose-{dose}--sign1-{a}--sign2-{b}'
        path=Path(provenance[name]['path']) if provenance is not None else root/stage/'payloads'/name/'packet.json'
        if provenance is not None:assert hashlib.sha256(path.read_bytes()).hexdigest()==provenance[name]['sha256']
        packet=codec.load_packet(path)
        sample=packet['samples'][:32] if row['window']=='early' else packet['samples'][32:]
        vectors.append((np.tanh(sample) if row['observation']=='activation' else sample).ravel())
    y=np.stack(vectors);center=y.sum(axis=0)/4
    A=(y[0]+y[1]-y[2]-y[3])/4;B=(y[0]-y[1]+y[2]-y[3])/4;C=(y[0]-y[1]-y[2]+y[3])/4
    for order,value in row['scores'].items():
        c=seals[cid][f"{dose}--{row['observation']}--{row['window']}--{order}"];w=c['w']
        risk=sum((s1*s2-sum(w*(v-c['m'])))**2 for (s1,s2),v in zip(signs,y))/4+sigma**2*sum(w*w)
        error=abs(value-(1-risk));max_score=max(max_score,error);max_relative=max(max_relative,error/max(1.,abs(value)));checks+=1
    cov=(y-center).T@(y-center)/4+sigma**2*np.eye(y.shape[1])
    w=np.linalg.solve(cov,C)
    risk=sum((p-w@(v-center))**2 for p,v in zip(parity,y))/4+sigma**2*(w@w)
    max_oracle=max(max_oracle,abs(row['oracle_r_squared']-(1-risk)))
    max_contrast=max(max_contrast,abs(row['direct_contrast_norm']-float(np.linalg.norm(C))))
assert max_relative<1e-12 and max_oracle<1e-8 and max_contrast<1e-12
out=dict(status='independent_saved_arithmetic_pass',rows=len(rows),reference_scores=checks,max_score_error=max_score,max_scaled_score_error=max_relative,score_scale='max(1,abs(reported R-squared)); auxiliary arithmetic only, frozen scientific solver gates unchanged',max_oracle_error=max_oracle,max_contrast_error=max_contrast,raw_receipt_sha256=hashlib.sha256((root/'receipt.json').read_bytes()).hexdigest(),scope='Independent arithmetic on saved trajectories and sealed weights; not independent data acquisition or scientific replication')
print(json.dumps(out,indent=2))
