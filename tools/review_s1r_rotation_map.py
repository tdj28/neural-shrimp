"""Independent saved-data review; does not import the producer analysis module."""
import argparse
from collections import Counter
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
from scipy.spatial.distance import cdist

ROOT=Path(__file__).resolve().parents[1]
PRODUCER_TIP='3c66e666d42b6a15922b8f752e0c079780d2b772'


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1<<20),b''):
            h.update(block)
    return h.hexdigest()


def require(condition,message):
    if not condition:
        raise ValueError(message)


def close(a,b,message):
    require(np.allclose(a,b,rtol=1e-12,atol=1e-10),message)


def review():
    receipt_path=ROOT/'docs/explore-s1r-rotation-audit-receipt.json'
    receipt=json.loads(receipt_path.read_text())
    require(subprocess.check_output(['git','show',PRODUCER_TIP+':docs/explore-s1r-rotation-audit-receipt.json'],cwd=ROOT)==receipt_path.read_bytes(),'producer-tip receipt mismatch')
    for name,expected in receipt['sources'].items():
        blob=subprocess.check_output(['git','show',receipt['source_commit']+':'+name],cwd=ROOT)
        require(hashlib.sha256(blob).hexdigest()==expected,'execution-commit source hash mismatch')
        require(digest(ROOT/name)==expected,'checked-out producer source mismatch')
    for item in receipt['inputs'].values():
        require(digest(Path(item['path']))==item['sha256'],'input hash mismatch')
    output=Path(receipt['output_directory'])
    for name,expected in receipt['outputs'].items():
        require(digest(output/name)==expected,'output hash mismatch')
    claimed=json.loads((output/'analysis.json').read_text())
    cfg=receipt['config']
    meta=json.loads(Path(receipt['inputs']['metadata']['path']).read_text())
    with np.load(receipt['inputs']['map']['path']) as data:
        h,o=data['log10_hidden'],data['log10_output']
        status,period,rho,ftle,planar=[data[k] for k in ('status','period','rho','ftle','planar_share')]
    require(status.shape==(512,512),'unexpected map shape')
    scale=np.array([h[1]-h[0],o[1]-o[0]])
    results={}
    for family in ('start_1.10','start_0.95'):
        trace=json.loads(Path(receipt['inputs'][family]['path']).read_text())
        rows=trace['rows']
        saved=claimed['families'][family]
        require(dict(Counter(x['type'] for x in rows))==saved['row_types'],'row type mismatch')
        comparisons=[]
        for index,offset in enumerate(cfg['offsets']):
            differences=[]
            statuses=[]
            shares=[]
            exclusions=Counter()
            for row in rows:
                if row['type']!='neimark_sacker':
                    continue
                target=[row['log10_hidden_boundary']+offset,row['log10_output']]
                require(h[0]<=target[0]<=h[-1] and o[0]<=target[1]<=o[-1],'unexpected out-of-map comparison')
                i=min(range(len(h)),key=lambda k:abs(h[k]-target[0]))
                j=min(range(len(o)),key=lambda k:abs(o[k]-target[1]))
                s,p=int(status[i,j]),int(period[i,j])
                reason={0:'divergent_screen',1:'fixed_screen',3:'chaotic_screen',5:'drift_screen'}.get(s)
                if s==2 and p<=2:
                    reason='period_at_most_two_screen'
                if reason is None:
                    if not np.isfinite([rho[i,j],ftle[i,j],planar[i,j]]).all():reason='nonfinite_diagnostic'
                    elif not 0<=rho[i,j]<=.5:reason='rotation_out_of_range'
                    elif abs(ftle[i,j])>=cfg['chaos']:reason='ftle_outside_band'
                    elif planar[i,j]<cfg['min_planarity']:reason='low_planarity'
                    else:reason='eligible'
                exclusions[reason]+=1
                if reason=='eligible':
                    differences.append(abs(float(rho[i,j])-row['theta_over_2pi']))
                    statuses.append(s)
                    shares.append(float(planar[i,j]))
            actual={'offset':offset,'retained':len(differences),'median':float(np.median(differences)),
                    'maximum':float(max(differences)),'status_counts':dict(Counter(map(str,statuses))),
                    'planarity_range':[min(shares),max(shares)],'exclusions':dict(exclusions)}
            expected=saved['comparisons'][index]
            require(actual['retained']==expected['eligible'],'eligible count mismatch')
            require(actual['status_counts']==expected['eligible_status_counts'],'eligible status mismatch')
            require(actual['status_counts']=={'4':actual['retained']},'retained status is not all unresolved')
            require(actual['exclusions']==expected['exclusions'],'exclusion mismatch')
            close([actual['median'],actual['maximum']],[expected['median_abs_difference'],expected['max_abs_difference']],'rotation difference mismatch')
            comparisons.append(actual)
        edges=[]
        step=np.median(np.diff([row['log10_output'] for row in rows]))
        for index in range(len(rows)-1):
            a,b=rows[index:index+2]
            if a['type']!='neimark_sacker' or b['type']!='neimark_sacker':reason='non_ns_endpoint'
            elif b['log10_output']-a['log10_output']>1.5*step:reason='output_gap'
            elif abs(b['log10_hidden_boundary']-a['log10_hidden_boundary'])>cfg['max_hidden_jump']:reason='hidden_jump'
            elif abs(b['theta_over_2pi']-a['theta_over_2pi'])>cfg['max_birth_jump']:reason='angle_jump'
            else:reason='geometrically_admissible'
            edges.append(reason)
        require(dict(Counter(edges))==saved['edge_counts'],'geometric edge mismatch')
        require(edges==[x['reason'] for x in saved['edges']],'edge order mismatch')
        rationals=sorted({Fraction(p,q) for q in range(1,cfg['max_den']+1) for p in range(1,q)
                          if cfg['min_den']<=Fraction(p,q).denominator<=cfg['max_den'] and Fraction(p,q)<Fraction(1,2)})
        roots=[]
        for index,reason in enumerate(edges):
            if reason!='geometrically_admissible':continue
            a,b=rows[index:index+2]
            ra,rb=a['theta_over_2pi'],b['theta_over_2pi']
            for rational in rationals:
                rotation=float(rational)
                if not min(ra,rb)<rotation<=max(ra,rb):continue
                av=np.array([a['log10_hidden_boundary'],a['log10_output']])/scale
                bv=np.array([b['log10_hidden_boundary'],b['log10_output']])/scale
                point=av+(rotation-ra)/(rb-ra)*(bv-av)
                matches=(status==2)&(period==2*rational.denominator)&np.isfinite(rho)&(np.abs(rho-rotation)<=1/meta['args']['samples']+1e-12)
                ii,jj=np.where(matches)
                result={'rows':[index,index+1],'rotation':str(rational),'candidates':len(ii)}
                if len(ii):
                    cells=np.c_[h[ii]/scale[0],o[jj]/scale[1]]
                    distances=cdist(point[None,:],cells)[0]
                    v=bv-av
                    dot=(cells-av)@v
                    perpendicular=np.abs(v[0]*(cells[:,1]-av[1])-v[1]*(cells[:,0]-av[0]))/np.linalg.norm(v)
                    endpoint_distances=cdist(np.stack([av,bv]),cells)
                    segment_distances=np.where((dot>=0)&(dot<=v@v),perpendicular,endpoint_distances.min(axis=0))
                    envelope=[float(segment_distances.min()),float(endpoint_distances.max(axis=0).min())]
                    probe=av+np.linspace(0,1,1001)[:,None]*v
                    nearest=cdist(probe,cells).min(axis=1)
                    require(float(nearest.min())>=envelope[0]-1e-9 and float(nearest.max())<=envelope[1]+1e-9,'sampled conditional distance violates envelope')
                    result.update(distance=float(distances.min()),envelope=envelope,
                                  nearest_grid_index=[int(ii[np.argmin(distances)]),int(jj[np.argmin(distances)])])
                roots.append(result)
        require(len(roots)==len(saved['roots']),'root count mismatch')
        for root in roots:
            matches=[x for x in saved['roots'] if x['rows']==root['rows'] and x['rotation']==root['rotation']]
            require(len(matches)==1,'root identity mismatch')
            expected=matches[0]
            require(root['candidates']==expected['candidate_cells'],'candidate count mismatch')
            if root['candidates']:
                close(root['distance'],expected['interpolated_distance_grid_cells'],'root distance mismatch')
                close(root['envelope'],expected['segment_distance_envelope_grid_cells'],'envelope mismatch')
                require(root['nearest_grid_index']==expected['nearest_candidate_grid_index'],'nearest index mismatch')
            else:
                require(expected['interpolated_distance_grid_cells'] is None,'empty candidate was dropped or given distance')
        results[family]={'row_types':dict(Counter(x['type'] for x in rows)),'edge_counts':dict(Counter(edges)),
                         'comparisons':comparisons,'roots':roots}
    return {'schema':'s1r-independent-review-v1','producer_tip':PRODUCER_TIP,'producer_execution_commit':receipt['source_commit'],
            'producer_receipt_sha256':digest(receipt_path),'reviewer_source_sha256':digest(Path(__file__)),
            'reviewer_source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            'numerical_verdict':'all independently recomputed key quantities match',
            'families':results,'independence':'No producer analysis imports; separate fractions enumeration, cdist geometry and cross-product segment distances.',
            'scientific_scope':'Exploratory saved-map association only; all retained samples are unresolved; no branch identity or ancestry established.',
            'documentation_note':'qualifying_run=false marks a capability in QUALIFIED_CAPABILITIES; it is not evidence the CUDA backend is unqualified.'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    require(not args.output.exists(),'review output already exists')
    report=review()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+'\n')
    print(json.dumps({'numerical_verdict':report['numerical_verdict'],'producer_tip':PRODUCER_TIP,
                      'root_counts':{k:len(v['roots']) for k,v in report['families'].items()}}))


if __name__=='__main__':main()
