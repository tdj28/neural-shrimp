"""Assemble hash-bound lookup index from completed independent proofs; no model calls."""
import hashlib,json
from pathlib import Path
from fractions import Fraction
ROOT=Path(__file__).resolve().parents[2]
RAW=Path('/data2/neural-shrimp-data')
def read(p):return json.loads(p.read_bytes())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def artifact(root,name):
 f=read(root/'manifest.json')['files'][name]
 return {'path':str(root/name),**f}
def audit_binding(root,pin):
 assert sha(root/'manifest.json')==pin
 outer=read(root/'manifest.json')['files'];assert sha(root/'run/manifest.json')==outer['run/manifest.json']['sha256']
 worker=read(root/'run/manifest.json')['files']
 for n in ['outcome.json','runtime.json']:assert sha(root/'run'/n)==worker[n]['sha256']
 o=read(root/'run/outcome.json');assert o['passed']
 return {'root':str(root),'outer_manifest':pin,'worker_manifest':sha(root/'run/manifest.json'),'source':read(root/'run/runtime.json')['source_commit']}
def producer_binding(root):
 pins={
  'attempt-003-quadratic-target':'cd181db39a626724dc6307974f97a99efdb311358523101778c58ba3e5a0059c',
  'attempt-001-march-target':'422a2948f044bb0beffe6697a813e26c684d401f1303502ab1add24528cc6c14',
  'attempt-002-march-target':'6984dff87e855a4fa43d036358688ebf536d9feac88bbe1a0fcdc2d0fc9a2715',
  'attempt-001-reanchor-target':'5a344e0c0b891d6051c62f54e3138360e9a385f2c7f9133f40d8f133386ea7c8',
  'attempt-001-tangent-march-target':'2ced48162d5960f4ce542d88cfd0377a92eda8a8dd12e78922530f3638f16e13',
  'attempt-002-tangent-march-target':'60c2b2d52d6a19a9e68d11d62978d9e148e63866818a5f6d00ea1f827740f7ed'}
 assert sha(root/'manifest.json')==pins[root.name]
 assert sha(root/'run/manifest.json')==read(root/'manifest.json')['files']['run/manifest.json']['sha256']
 rt=read(root/'run/runtime.json')
 return {'root':str(root),'outer_manifest':sha(root/'manifest.json'),'worker_manifest':sha(root/'run/manifest.json'),'source':rt['commit']}
def build():
 rows=[]
 def add(producer,audit,full,proofbase,ends,compact=None,selector=None,join=None):
  k=len(rows)+1;pr=Path(producer['root'])/'run';ar=Path(audit['root'])/'run'
  p256=artifact(ar,proofbase+'256.json');p384=artifact(ar,proofbase+'384.json')
  for proof in [p256,p384]:
   p=Path(proof['path']);assert sha(p)==proof['sha256'];v=read(p);assert v['passed']and v['inclusions']==470 and v['anchor_coordinates']==470
  item={'global_chart':k,'producer':producer,'full_chart':artifact(pr,full),'audit':audit,'independent256':p256,'independent384':p384,'exact_gauge_endpoints':ends,'native_dimensions':{'state':13,'period':18,'shooting':470,'sigma':1},'join':join or {'kind':'full470_endpoint_containment','previous_global_chart':k-1,'proof':p384['path']},'gauge_family':'original_output_rate'if k<=10 else 'reanchored_native_tangent'}
  endpoint_rates=read(Path(p384['path']))['recomputed_endpoints']
  assert len(endpoint_rates)==2 and all(len(v)==470 for v in endpoint_rates)
  item['independently_recomputed_native_rate_endpoints']={'native_coordinate_indices':[468,469],'left':endpoint_rates[0][-2:],'right':endpoint_rates[1][-2:],'source':p384,'json_pointer':'/recomputed_endpoints/{0,1}/{468,469}','exact_serialization':'mid/rad are integer*2**exponent; enclosure is [mid-rad,mid+rad]. Display text is illustrative only.'}
  if compact:item['compact_chart']={**artifact(pr,compact),'selector':selector}
  rows.append(item)
 # Original two charts carry full native lift of independently proved fold.
 p=RAW/'explore/eigenpair-locus-prototype/attempt-003-quadratic-target';pb=producer_binding(p)
 a=RAW/'audits/eigenpair-quadratic-path/attempt-003';ab=audit_binding(a,'ff9f38c0e3e769ef9a45e96d761cbc1313a0e3b1bb15711fbd57e679f7efbbae')
 for i,f in enumerate(['events/0151-chart_return.json.gz','events/0285-chart_return.json.gz']):
  ends=[str(Fraction(2*i-1,32768)),str(Fraction(2*i+1,32768))]
  join={'kind':'source_authenticated_full470_fold_lift','parent_certificate':str(RAW/'explore/certificates/s2-period18-fold-2026-10-09/attempt-002-target/run/certificate.json'),'known_gauge_parameter':'0','proof':str(a/f'run/chart-{i}-384.json')}if i==0 else None
  add(pb,ab,f,f'chart-{i}-',ends,join=join)
 # Eight further old-gauge charts.
 a=RAW/'audits/eigenpair-quadratic-march/attempt-001';ab=audit_binding(a,'16950add27cdb24a9514a9fec7d9d37423c9d05d07e4bc3a84202bd4a6ca9d6d')
 for u in [1,2]:
  p=RAW/f'explore/eigenpair-locus-prototype/attempt-{u:03d}-march-target';pb=producer_binding(p)
  accepted=read(p/'run/outcome.json')['progress']['accepted'];assert len(accepted)==4
  for i,r in enumerate(accepted):
   right=Fraction(r['right']);add(pb,ab,r['full_chart_file'],f'unit{u}-chart{i}-',[str(right-Fraction(1,16384)),str(right)],r['accepted_file'],'chart')
 # Two new-gauge charts with a separately proved correlated reanchor.
 p=RAW/'explore/eigenpair-locus-prototype/attempt-001-reanchor-target';pb=producer_binding(p)
 a=RAW/'audits/eigenpair-tangent-reanchor/attempt-002';ab=audit_binding(a,'f3dbc2784e909a318148c9de6ecf6f479c9ac6b078d9d1ddc5fc6c0ad5217658')
 for i,f in enumerate(['events/0215-chart_return.json.gz','events/0349-chart_return.json.gz']):
  join={'kind':'three_intersection_known_root_localization_and_regauge','previous_global_chart':10,'old_endpoint':'19/32768','new_parameter_is_interval_not_old_endpoint_equality':True,'localization_proofs':[artifact(a/'run',f'localization-{j}.json')for j in [1,2,3]],'proof':str(a/'run/chart-0-384.json')}if i==0 else None
  add(pb,ab,f,f'chart-{i}-',[str(Fraction(2*i-1,128)),str(Fraction(2*i+1,128))],join=join)
 # Eight tangent charts.
 a=RAW/'audits/eigenpair-tangent-march/attempt-001';ab=audit_binding(a,'3ce722c12345e55f185006acea7e40e94c111cd595e758ae0f0065a0e6d0c656')
 for u in [1,2]:
  p=RAW/f'explore/eigenpair-locus-prototype/attempt-{u:03d}-tangent-march-target';pb=producer_binding(p)
  accepted=read(p/'run/outcome.json')['progress']['accepted'];assert len(accepted)==4
  for i,r in enumerate(accepted):
   right=Fraction(r['right']);add(pb,ab,r['full_chart_file'],f'unit{u}-chart{i}-',[str(right-Fraction(1,64)),str(right)],r['accepted_file'],'chart')
 # All87 newly audited charts, each from its actual immutable next-unit card.
 for u in range(25):
  card=read(ROOT/f'docs/plans/eigenpair-remaining-unit{u:03d}.json');rec=read(ROOT/f'docs/reviews/eigenpair-remaining-unit{u:03d}.json')
  a=Path(rec['root']);ab=audit_binding(a,rec['outer_manifest']['sha256']);p=Path(card['root']);pb={'root':str(p),'source':card['execution'],'outer_manifest':card['outer'],'worker_manifest':card['worker']}
  assert sha(p/'manifest.json')==card['outer']and sha(p/'run/manifest.json')==card['worker']
  assert card['first_global_chart']==len(rows)+1
  for i,r in enumerate(card['selected']):
   compact=r.get('accepted_path');selector='chart'if compact else None
   if u in [14,15]:compact='result.json';selector=f'charts/{i}'
   add(pb,ab,r['path'],f'chart{i}-',r['expected_endpoints'],compact,selector)
 assert len(rows)==107
 membership=read(ROOT/'docs/reviews/eigenpair-chain-membership-independent.json');assert membership['outcome']['passed']
 result={'schema':'neural-shrimp-independent-native-locus-index-v1','total_connected_charts':107,'period':18,'same_native_component':True,'charts':rows,'membership':membership,'figure_guidance':'Use actual final two coordinates of each authenticated chart center/slope/curvature with its exact gauge interval and coordinate radii. Each entire chart is independently proved. Chart10-to11 uses known-root localization/regauge, not a line between gauge coordinates. No interpolation outside proved tubes. Original fold identity and final1:1 root membership are distinct referenced endpoint proofs. Shared analytic acquisitions and Arb in chart proofs; exact rational independent final membership arithmetic.','claim_limit':'No generic unfolding or complete shrimp morphology follows from this index.'}
 out=ROOT/'docs/reviews/eigenpair-connected107-proof-index.json';assert not out.exists();out.write_text(json.dumps(result,indent=2)+'\n')
 print('wrote',out,'bytes',out.stat().st_size)
if __name__=='__main__':build()
