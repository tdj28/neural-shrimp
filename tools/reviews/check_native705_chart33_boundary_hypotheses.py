"""Map whole chart33 saved certificates to the unchanged physical-boundary lemma.

No new interval arithmetic/model calls. Exact theorem hypotheses and genealogy.
"""
import hashlib,json,subprocess
from fractions import Fraction as F
from pathlib import Path
import check_native705_chart32_boundary_hypotheses as P
ROOT=P.ROOT;B=P.P;need,read,entry,bounds=P.need,P.read,P.entry,P.bounds
SPECS=dict(root=P.SPECS['root'],complement=dict(root='/data2/neural-shrimp-data/audits/native705-cubic-panel-independent/attempt001',source='5f6e80313d7dc7f02eef5a0e62b3243fe1396708',outer='a0a9789cce6df7cef235401a9341257bc57e34d9546421811b96196e3564263b',worker='0acda2263797d2b8b8901ccebdc85ac7e4c70f6f68516fe978da6413795c8801'))

def check():
 previous=P.check();need(previous==read(ROOT/'docs/reviews/native705-chart32-boundary-hypotheses.json'),'entire previous physical-boundary authority')
 rows={k:B.bundle(v)for k,v in SPECS.items()};p,rt,out,get,_=rows['root'];proof=get('chart002/independent-proof.json');member=get('chart002/known-root-membership.json');prior=get('chart001/independent-proof.json');ib=get('input-binding.json');chart=proof['chart']
 need(out['connected']and out['connected_count']==3 and proof['passed']and set(proof['gates'])=={'uniform_root','known_root','primitive18','independent_frame','positive_rates_angle'}and all(proof['gates'].values()),'same independently joined full705 critical root')
 need(chart['shooting_dimension']==705 and chart['parameter_endpoints']==['205/2048','333/2048']and len(chart['normal'])==len(chart['origin'])==705 and any(F(x)for x in chart['normal']),'full affine gauge')
 root=proof['root'];need(root['passed']and root['dimension']==705 and len(root['coordinate_inclusions'])==705 and all(root['coordinate_inclusions'])and 0<=bounds(root['contraction'])[0]<=bounds(root['contraction'])[1]<1,'full augmented Jacobian uniformly nonsingular')
 need([v['divisor']for v in proof['proper_divisors']]==[1,2,3,6,9]and all(v['coordinates']for v in proof['proper_divisors'])and proof['frame_nonzero_coordinates'],'primitive full-state18 and nondegenerate complex plane')
 need(member==proof['anchor']and member['passed']and member['parameter_inside']and member['known_parameter']=='103/1024'and member['coordinates']==len(member['coordinate_inclusions'])==705 and all(member['coordinate_inclusions']),'entire previous endpoint in current uniqueness tube')
 need(prior['passed']and prior['chart']['parameter_endpoints']==previous['parameter_endpoints']==['39/1024','103/1024']and all(prior['gates'].values())and chart['normal']==prior['chart']['normal']and chart['origin']==prior['chart']['origin'],'exact original same-gauge endpoint authority')
 audit=rows['complement'][2];ag=rows['complement'][3];a=ag('authority-binding.json');initial=ag('initial-authority.json');need(a['initial_independent']==P.SPECS['last']and initial['first_independent']==P.SPECS['first']and initial['previous']['independent']==P.SPECS['last'],'full stable32 authority retained')
 oldlast=B.bundle(P.SPECS['last']);oldfirst=B.bundle(P.SPECS['first']);need(initial['independent_outcome']==oldlast[2]and initial['first_outcome']==oldfirst[2]and initial['previous']['complete_complement_endpoints']==previous['parameter_endpoints']and initial['previous']['index']['ordinal']==32,'entire independently proved predecessor interval')
 need(len(a['charts'])==len(audit['completed_charts'])==1 and a['charts'][0]['ordinal']==33 and a['retained_unqualified_chart']['ordinal']==34 and a['retained_failed_attempts']==[71,72,73,74],'only whole33 promoted; all34 failures retained')
 spec=a['charts'][0];idx=spec['root_index'];need(idx['ordinal']==33 and {k:idx['independent'][k]for k in('root','source','outer','worker')}==SPECS['root']and idx['independent']['proof']==dict(path='chart002/independent-proof.json',identity=entry(p/'run/chart002/independent-proof.json')),'literal same full705 original root')
 complete=audit['completed_charts'][0];union=ag('charts/033/union.json');need(complete['chart']==33 and complete['original_root']==idx and union==complete['union']==dict(passed=True,endpoints=chart['parameter_endpoints'],total_cells=39,complement_dimension=11),'closed whole33 cover')
 cells=audit['completed_cells'];need(len(cells)==len(a['cells'])==39 and audit['complement_dimension']==11,'all39 independently proved11-mode cells')
 need(all({k:c[k]for k in('chart','ordinal','cell')}=={k:s[k]for k in('chart','ordinal','cell')}for c,s in zip(cells,a['cells'],strict=True)),'literal retained accepted-cell identities')
 spans=sorted(tuple(map(F,c['cell']['endpoints']))for c in cells);need(spans[0][0]==F(205,2048)and spans[-1][1]==F(333,2048)and all(x[1]==y[0]for x,y in zip(spans,spans[1:]))and all(c['chart']==33 and c['passed']and c['complement_dimension']==11 and bounds(c['modulus'])[1]<1 for c in cells),'all complementary multipliers and exact closed union')
 projection=list(map(str,bounds(member['projection'])));need(F(205,2048)<=F(projection[0])<=F(projection[1])<=F(333,2048),'entire known-root projection inside independently stable whole33')
 source={}
 for n in('tools/reviews/review_cubic705_prefix.py','tools/reviews/review_cubic705_math.py'):
  e=rt['sources'][n];need(entry(p/'run/source'/n)==e and hashlib.sha256(subprocess.check_output(['git','show',SPECS['root']['source']+':'+n],cwd=ROOT)).hexdigest()==e['sha256'],'literal original independent normalization/angle/root proof');source[n]=e
 for n in('src/neural_shrimp/rigorous_cusps.py','src/neural_shrimp/rigorous_folds.py','src/neural_shrimp/rigorous_unit_pair.py','src/neural_shrimp/rigorous_unit_pair_pal.py'):
  e=ib['source_native'][n];need(entry(ROOT/n)==e==previous['source_equations'][n],'same analytic training/normalization/gauge equations');source[n]=e
 lemma=previous['analytic_lemma'];tex=(ROOT/'paper/gd-route/main.tex').read_text();start=tex.index(r'\begin{lemma}[Physical stability boundary]');end=tex.index('For our analytic training map',start);need(hashlib.sha256(tex[start:end].encode()).hexdigest()==lemma['lemma_and_compact_argument_sha256'],'unchanged separately reviewed lemma and compact-family argument')
 return dict(passed=True,native_calls=0,new_interval_arithmetic=False,chart=33,parameter_endpoints=chart['parameter_endpoints'],independent_complement_cells=39,authorities=SPECS,artifacts={k:v[4]for k,v in rows.items()},source_equations=source,analytic_lemma=lemma,prior_boundary_receipt=entry(ROOT/'docs/reviews/native705-chart32-boundary-hypotheses.json'),prior_stable_projection=projection,hypotheses=['Same C3 two-physical-rate analytic training map and normalized full shooting equations.','Primitive18, independent real critical plane and original source-checked0<t<=1/4.','All705 uniform contraction inclusions imply nonsingular full augmented Jacobian.','Exact39-cell cover proves all11 complementary multipliers inside unit disk.','Entire previous stable endpoint lies inside this same full uniqueness tube with identical affine gauge; no projected-only join.'],analytic_conclusion='The separately reviewed physical-stability-boundary lemma and compact-family argument extend the regular boundary and connected adjacent attracting-side family from32 to33positive-angle charts.',limitations='Chart34 remains unqualified. No quantified collar width, physical axis orientation, Lyapunov coefficient, global embedded-boundary closure or whole-window topology follows from this mapping.')

if __name__=='__main__':
 result=check();p=ROOT/'docs/reviews/native705-chart33-boundary-hypotheses.json';p.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n');print(json.dumps({k:result[k]for k in('passed','chart','independent_complement_cells')}))
