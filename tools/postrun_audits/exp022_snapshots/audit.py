"""Root's independent literal audit; no production selection/evaluation import."""
from pathlib import Path
from fractions import Fraction
import hashlib,json,datetime,math

HERE=Path(__file__).resolve().parent
REPO=Path('/data2/neural-shrimp')
PREP=Path('/data2/neural-shrimp-data/audits/EXP-022/selection-001')
checks=[]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def check(ok,what):
    if not ok:raise AssertionError(what)
    checks.append(what)
check(sha(PREP/'selection.json')=='b8bcf7f11fc82623993e07af54cc3d6a42a338d18561cc148f4316329952416b','pinned selected output')
check(sha(PREP/'input-bindings.json')=='1edad139614b5431c809201ad6ece4216536d94c70e157f6f5624b0d2383bb36','pinned full inputs')
bindings=json.loads((PREP/'input-bindings.json').read_text())
for row in bindings:
    p=Path(row['root'])/row['path']
    check(p.stat().st_size==row['bytes'] and sha(p)==row['sha256'],'source bytes '+str(p))
selected=json.loads((PREP/'selection.json').read_text())
by_id={r['id']:r for r in selected['slots']}
check(list(by_id)==['anchor-core','anchor-failed-near-edge','core-rho025','core-rho050','failed-near-edge-rho025','failed-near-edge-rho050'],'literal six-slot order')
branches={};diagnostics=[]
for setting,bid in [('core','core-direction-+1'),('failed-near-edge','failed-near-edge-direction--1')]:
    p=REPO/'evidence/EXP-006/attempt-001/continuation'/(bid+'.json')
    branch=json.loads(p.read_text());nodes=branch['nodes'];eligible=[];stopped=False;prefix=0
    for i,n in enumerate(nodes):
        check(n['index']==i,'original node index '+bid+str(i))
        stopped=stopped or not n['attracting'] or n['unstable'] or bool(n['bracket_flags'])
        if not stopped:prefix+=1
        modes=n['transverse_multipliers'];order=sorted(modes,key=lambda m:m['modulus'],reverse=True)
        dom=order[0]
        if not stopped and all(n['gates'].values()) and dom['modulus']>order[1]['modulus'] and dom['imag']==0 and dom['real']>0:
            eligible.append((i,n,dom['real']))
    used=set()
    for t,suffix in [(.25,'025'),(.50,'050')]:
        candidates=[]
        for i,n,rho in eligible:
            distance=abs(Fraction(rho)-Fraction(t))
            if distance<=Fraction(1,40) and i not in used:candidates.append((distance,i,n,rho))
        s=by_id[setting+'-rho'+suffix]
        if not candidates:
            check(s['available'] is False and s['status']=='no_historical_match' and s['node_id'] is None,'literal absence '+s['id'])
        else:
            distance,i,n,rho=min(candidates,key=lambda c:(c[0],c[1]));used.add(i)
            check(s['node_id']==bid+'--node-'+str(i) and s['node_index']==i,'literal nearest node '+s['id'])
            for key in ['alpha','gamma','period','first_maximum_state']:check(s[key]==n[key],'literal selected '+s['id']+' '+key)
            check(s['dominant_rho']==rho,'literal selected signed multiplier '+s['id'])
            check(s['available']==(n.get('first_maximum_state') is not None),'literal no-fallback availability '+s['id'])
        nearest=min(eligible,key=lambda v:(abs(Fraction(v[2])-Fraction(t)),v[0]))
        brackets=[{'left_index':a[0],'right_index':b[0],'left_rho':a[2],'right_rho':b[2],'left_gamma':a[1]['gamma'],'right_gamma':b[1]['gamma']} for a,b in zip(eligible,eligible[1:]) if b[0]==a[0]+1 and min(a[2],b[2])<t<max(a[2],b[2])]
        diagnostics.append(dict(slot=s['id'],original_nodes=len(nodes),prefix_nodes=prefix,eligible_nodes=len(eligible),nearest_index=nearest[0],nearest_rho=nearest[2],nearest_distance=float(abs(Fraction(nearest[2])-Fraction(t))),brackets=brackets))
check(sum(r['available'] for r in by_id.values())==4,'four available of six')
check(selected['new_neural_computations']==0 and selected['scientific_qualification'] is False,'metadata-only scope')
out=dict(status='pass',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),authors='Root independent literal implementation; selector not imported. Shared historical inputs and libraries are not independent replication.',checks_pass=len(checks),checks=checks,diagnostics=diagnostics,all_slots_available=False,new_neural_integrations=0,script_sha256=sha(Path(__file__)),selection_sha256=sha(PREP/'selection.json'),input_bindings_sha256=sha(PREP/'input-bindings.json'))
with (HERE/'report.json').open('x') as f:json.dump(out,f,indent=2,allow_nan=False);f.write('\n')
print(json.dumps({'status':out['status'],'checks_pass':len(checks),'diagnostics':diagnostics},indent=2))
