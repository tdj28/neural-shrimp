"""Map completed chart32 certificates to the existing physical-boundary lemma.

No new interval arithmetic or model calls; exact saved theorem hypotheses only.
"""
import hashlib,json,subprocess
from pathlib import Path
from fractions import Fraction as F
import check_native705_chart31_boundary_hypotheses as P
ROOT=P.ROOT;need,read,entry,bounds=P.need,P.read,P.entry,P.bounds
SPECS=dict(
 root=dict(root='/data2/neural-shrimp-data/audits/native705-cubic-prefix-independent/attempt001',source='f2f0d768542d3f2bafc93f445a1c928f3d019f81',outer='21c4c93586dfe0f71386eafb1dc39fa8de6925baf4c22837c40784125cdc27d2',worker='1623ae469c37d35ffc3957aa48b5961bd317317400735be9f413401282928b21'),
 first=dict(root='/data2/neural-shrimp-data/audits/native705-cubic32-first-independent/attempt001',source='742eb90dda79986a1d1e24181528d6debd3d102b',outer='bbd406d5c64973f4a6551d2aace89f79dda74d98ba299a01c53cf9ae1133ce40',worker='969393821fcb3b1fc8b3ae015e5f4542cb0e9c649dea7a15e4db935bd1534967'),
 last=dict(root='/data2/neural-shrimp-data/audits/native705-cubic32-remaining-independent/attempt005',source='4bec292a67423a7f88f0124fb261157b383db287',outer='d52b683b9bc3a41846f27259550b5270af2f74d73185f6fb96844a75e49e2c35',worker='721f56772b5dc67f1d9183bdbfa95dc15870c6aaf3fa48d355a186adab31f866'))

def check():
 previous=P.check();need(previous==read(ROOT/'docs/reviews/native705-chart31-boundary-hypotheses.json'),'entire previous physical-boundary authority')
 rows={k:P.bundle(v)for k,v in SPECS.items()};p,rt,out,get,_=rows['root'];proof=get('chart001/independent-proof.json');member=get('chart001/known-root-membership.json');ib=get('input-binding.json');chart=proof['chart']
 need(out['connected']and out['connected_count']==3 and proof['passed']and set(proof['gates'])=={'uniform_root','known_root','primitive18','independent_frame','positive_rates_angle'}and all(proof['gates'].values()),'same independently joined full705 critical root')
 need(chart['shooting_dimension']==705 and chart['parameter_endpoints']==['39/1024','103/1024']and len(chart['normal'])==len(chart['origin'])==705 and any(F(x)for x in chart['normal']),'full affine gauge')
 root=proof['root'];need(root['passed']and root['dimension']==705 and len(root['coordinate_inclusions'])==705 and all(root['coordinate_inclusions'])and 0<=bounds(root['contraction'])[0]<=bounds(root['contraction'])[1]<1,'full augmented Jacobian uniformly nonsingular')
 need([v['divisor']for v in proof['proper_divisors']]==[1,2,3,6,9]and all(v['coordinates']for v in proof['proper_divisors'])and proof['frame_nonzero_coordinates'],'primitive full-state18 and nondegenerate complex plane')
 need(member==proof['anchor']and member['passed']and member['parameter_inside']and member['known_parameter']=='79/2048'and member['coordinates']==len(member['coordinate_inclusions'])==705 and all(member['coordinate_inclusions']),'entire exact previous endpoint in current uniqueness tube')
 first,last=[rows[k][2]for k in('first','last')];a,b=[rows[k][3]('authority-binding.json')for k in('first','last')]
 need(a['original_root']==b['original_root']and a['original_root']['ordinal']==32 and {k:a['original_root']['independent'][k]for k in('root','source','outer','worker')}==SPECS['root'],'all spectra on identical full root')
 need(a['prior_stable']==b['prior_stable']==previous and b['first_independent']==SPECS['first'],'exact stable predecessor and independent first-quarter genealogy')
 need(first['parameter_endpoints']==['39/1024','55/1024']and last['parameter_endpoints']==['55/1024','103/1024']and last['whole_chart_endpoints']==chart['parameter_endpoints'],'closed whole-chart union')
 cells=first['completed_cells']+last['completed_cells'];need(len(cells)==22 and first['total_cells']==4 and last['total_cells']==18 and first['complement_dimension']==last['complement_dimension']==11,'all22 independently proved11-mode cells')
 spans=sorted(tuple(map(F,c['cell']['endpoints']))for c in cells)
 need(spans[0][0]==F(39,1024)and spans[-1][1]==F(103,1024)and all(x[1]==y[0]for x,y in zip(spans,spans[1:]))and all(c['passed']and c['complement_dimension']==11 and bounds(c['modulus'])[1]<1 for c in cells),'all complementary multipliers and exact cover')
 projection=list(map(str,bounds(member['projection'])));need(first['prior_stable_projection']==last['prior_stable_projection']==a['projection']==b['projection']==projection and F(39,1024)<=F(projection[0])<=F(projection[1])<=F(55,1024),'entire known-root projection in stable firstquarter')
 source={}
 for n in('tools/reviews/review_cubic705_prefix.py','tools/reviews/review_cubic705_math.py'):
  e=rt['sources'][n];need(entry(p/'run/source'/n)==e and hashlib.sha256(subprocess.check_output(['git','show',SPECS['root']['source']+':'+n],cwd=ROOT)).hexdigest()==e['sha256'],'literal original independent normalization/angle/root proof');source[n]=e
 for n in('src/neural_shrimp/rigorous_cusps.py','src/neural_shrimp/rigorous_folds.py','src/neural_shrimp/rigorous_unit_pair.py','src/neural_shrimp/rigorous_unit_pair_pal.py'):
  e=ib['source_native'][n];need(entry(ROOT/n)==e==previous['source_equations'][n],'same analytic training/normalization/gauge equations');source[n]=e
 lemma=previous['analytic_lemma'];tex=(ROOT/'paper/gd-route/main.tex').read_text();start=tex.index(r'\begin{lemma}[Physical stability boundary]');end=tex.index('For our analytic training map',start);need(hashlib.sha256(tex[start:end].encode()).hexdigest()==lemma['lemma_and_compact_argument_sha256'],'unchanged separately reviewed lemma and compact-family argument')
 return dict(passed=True,native_calls=0,new_interval_arithmetic=False,chart=32,parameter_endpoints=chart['parameter_endpoints'],independent_complement_cells=22,authorities=SPECS,artifacts={k:v[4]for k,v in rows.items()},source_equations=source,analytic_lemma=lemma,prior_boundary_receipt=entry(ROOT/'docs/reviews/native705-chart31-boundary-hypotheses.json'),hypotheses=['Same C3 two-physical-rate analytic training map and normalized full shooting equations.','Primitive18, independent real critical plane and original source-checked0<t<=1/4.','All705 uniform contraction inclusions imply nonsingular full augmented Jacobian.','Exact22-cell cover proves all11 complementary multipliers inside unit disk.','Entire previous stable endpoint lies inside this same full uniqueness tube; no projected-only join.'],analytic_conclusion='The separately reviewed physical-stability-boundary lemma and compact-family argument extend the regular boundary and connected adjacent attracting-side family from31 to32positive-angle charts.',limitations='No quantified collar width, physical axis orientation, Lyapunov coefficient, global embedded-boundary closure or whole-window topology follows from this mapping.')

if __name__=='__main__':
 result=check();p=ROOT/'docs/reviews/native705-chart32-boundary-hypotheses.json';p.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n');print(json.dumps({k:result[k]for k in('passed','chart','independent_complement_cells')}))
