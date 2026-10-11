"""Hash-bound mapping of completed chart31 certificates to analytic boundary lemma.

No native calls or new interval arithmetic. Existing certified scalar gates,
exact interval unions, source equations and same-root authority only.
"""
import hashlib,json,subprocess
from pathlib import Path
from fractions import Fraction as F
ROOT=Path(__file__).resolve().parents[2]
BASE=Path('/data2/neural-shrimp-data/audits')
SPECS={
 'root':dict(root=str(BASE/'native705-correlated-independent/attempt002'),source='d70eaaf9abd34ea65f1946c7727912b89a57ce55',outer='db0c695ee52bd019239f683a7158701ad82be9728fcbd0af99359af8c9b66b79',worker='12157fcd7d23f7aff29c2a8a317ec2a7999348f6162b3347f5fcf8af1537332c'),
 'first':dict(root=str(BASE/'native705-fresh-quarter-independent/attempt001'),source='9b218f7cfe5c8c6227d4fc20c78400a86daed50f',outer='acefd8944c1cc88f38f60532f1142672d19e6ff11a325eebd1eca059108b4809',worker='1efe71c70410fc3ab1436cc6777edcc2e112ea179af05f98c50fa99384879f13'),
 'second':dict(root=str(BASE/'native705-second-quarter-independent/attempt001'),source='c59e0957054eded4fd38dfef27463b5daac94cb8',outer='5a815a2533d67a1fd2b51b162357f6e79f6abb87f5475c4a95a41b29cc38d849',worker='7eaf5976b6a9caa362755b1d64c5a047daf37cdb5492f5bf1659e5b44c7579b4'),
 'last':dict(root=str(BASE/'native705-remaining-quarters-independent/attempt001'),source='3d94d468c01b20cd7adb6e4f01ed9be7e5b2ab6c',outer='23bd1bff8f6ba5be3527af54eccf4ec6bfcb1de065fe4638cfb98d2b24ad28e0',worker='6ad82ba05ec43b8a8bffdc85e1033635659d9a911d273c7f1258e97ff32ee5ac')}

def need(x,s):
 if not x:raise ValueError(s)

def read(p):return json.loads(p.read_text())
def entry(p):return dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def bounds(v):
 m=F(v['mid'][0])*F(2)**v['mid'][1];r=F(v['rad'][0])*F(2)**v['rad'][1];return m-r,m+r

def bundle(spec):
 p=Path(spec['root']);need(entry(p/'manifest.json')['sha256']==spec['outer']and entry(p/'run/manifest.json')['sha256']==spec['worker'],'literal completed manifests');manifest=read(p/'run/manifest.json')['files'];outer=read(p/'manifest.json')['files'];need(entry(p/'launcher-result.json')==outer['launcher-result.json'],'literal owned result');la=read(p/'launcher-result.json');need(la['source_commit']==spec['source']and la['exit_code']==0 and la['failure']is None and not la['owned_live_pids'],'completed source/owned exit');selected={}
 def get(name):
  need(entry(p/'run'/name)==manifest[name],'literal saved proof artifact');selected[name]=manifest[name];return read(p/'run'/name)
 rt=get('runtime.json');need(rt['source_commit']==spec['source'],'literal execution');out=get('outcome.json');need(out['passed']and out['native_calls']==0,'completed independent certificate');return p,rt,out,get,selected

def check():
 rows={k:bundle(v)for k,v in SPECS.items()};p,rt,out,get,art=rows['root'];proof=get('independent-proof.json');ib=get('input-binding.json');member=get('known-root-membership.json');chart=proof['chart']
 need(proof['passed']and all(proof['gates'].values())and set(proof['gates'])=={'uniform_root','known_root','primitive18','independent_frame','positive_rates_angle'},'complete original theorem gates');need(proof['root']['passed']and proof['root']['dimension']==705 and len(proof['root']['coordinate_inclusions'])==705 and all(proof['root']['coordinate_inclusions']),'full705 unique root inclusion');qlo,qhi=bounds(proof['root']['contraction']);need(0<=qlo<=qhi<1,'strict full-J preconditioned contraction');need(chart['shooting_dimension']==705 and chart['parameter_endpoints']==['-49/2048','79/2048']and len(chart['normal'])==len(chart['origin'])==705 and any(F(x)for x in chart['normal']),'same full affine gauge');need([v['divisor']for v in proof['primitive']]==[1,2,3,6,9]and all(v['coordinates']for v in proof['primitive']),'all proper divisors');need(proof['independent_frame_coordinates']and member['passed']and member['coordinates']==705 and member['parameter_inside'],'independent mode plane and prior full-root join')
 source={}
 for n in ['tools/reviews/review_correlated705.py','tools/reviews/review_correlated705_math.py']:
  pin=rt['sources'][n];need(entry(p/'run/source'/n)==pin and hashlib.sha256(subprocess.check_output(['git','show',SPECS['root']['source']+':'+n],cwd=ROOT)).hexdigest()==pin['sha256'],'literal independent source');source[n]=pin
 native=ib['source_native']
 for n in ['src/neural_shrimp/rigorous_cusps.py','src/neural_shrimp/rigorous_folds.py','src/neural_shrimp/rigorous_unit_pair.py','src/neural_shrimp/rigorous_unit_pair_pal.py']:
  need(entry(ROOT/n)==native[n],'same analytic map/terminal/normalization/gauge source');source[n]=native[n]
 o1,o2,o3=[rows[k][2]for k in['first','second','last']];a1,a2,a3=[rows[k][3]('authority-binding.json')for k in['first','second','last']]
 need(all(v['complement_dimension']==11 for v in[o1,o2,o3])and[o1['total_cells'],o2['total_cells'],o3['total_cells']]==[17,4,10],'all31 independently proved cell spectra');need(o1['parameter_endpoints']==['-49/2048','-17/2048']and o2['parameter_endpoints']==['-17/2048','15/2048']and o3['parameter_endpoints']==['15/2048','79/2048'],'exact closed entire-chart union');trace=a1['original_whole'];tp=Path(trace['root']);need(entry(tp/'manifest.json')['sha256']==trace['outer']and entry(tp/'run/manifest.json')['sha256']==trace['worker'],'literal historical polynomial context');tm=read(tp/'run/manifest.json')['files'];need(entry(tp/'run/input-binding.json')==tm['input-binding.json'],'context binding artifact');ti=read(tp/'run/input-binding.json');need({k:ti['index']['independent'][k]for k in('root','source','outer','worker')}==SPECS['root']and ti['prior_card']==ib['card'],'same original correlated root and coefficient authority');need(a2['first_quarter']==a3['first_quarter']==SPECS['first']and a3['second_quarter']==SPECS['second']and a3['prior_outcome']==o2,'same original root genealogy');lo,hi=bounds(member['projection']);need(o2['prior_stable_projection']==o3['prior_stable_projection']==[str(lo),str(hi)]and F(-17,2048)<=lo<=hi<=F(15,2048),'entire prior stable endpoint in stable secondquarter')
 lemma_source='a200f6366f0e4c60ad87af8ac6c6870b98c1972a';tex=subprocess.check_output(['git','show',lemma_source+':paper/gd-route/main.tex'],cwd=ROOT,text=True);start=tex.index(r'\begin{lemma}[Physical stability boundary]');end=tex.index('For our analytic training map',start);lemma=tex[start:end].encode();lemma_binding=dict(source=lemma_source,path='paper/gd-route/main.tex',label='lem:physical-boundary',lemma_and_compact_argument_sha256=hashlib.sha256(lemma).hexdigest())
 hypotheses=[
 dict(name='C3 two-physical-parameter family',authority='Same source-authenticated smooth tanh full-batch gradient map; hidden/output rates are the two actual parameters.',sources=['src/neural_shrimp/rigorous_cusps.py','src/neural_shrimp/rigorous_folds.py']),
 dict(name='Primitive full-state period18 and independent real mode plane',authority='All705 root equations, all five proper-divisor separations, normalized q and q dot r=0, nonzero r coordinate bound.',artifact='root/independent-proof.json'),
 dict(name='Nonreal critical pair',authority='Original actual_geometry enforces both rates,t>0 and t<=1/4 on entire original whole domain. Thus0<sqrt(t)<=1/2<pi; entire terminal series is the lemma return matrix.',sources=['tools/reviews/review_correlated705.py','src/neural_shrimp/rigorous_unit_pair.py']),
 dict(name='Full705 augmented Jacobian nonsingular',authority='Uniform weighted norm of I-C(d)J is below1 for the full state/mode/both-rate/angle Jacobian including both normalizations and affine gauge; square C(d)J invertible implies J invertible.',contraction_upper=str(qhi),sources=['tools/reviews/review_correlated705_math.py','src/neural_shrimp/rigorous_unit_pair_pal.py']),
 dict(name='All11 complementary multipliers strictly inside disk',authority='Three completed independent spectral authorities,31cells, exact whole-chart union; full phase quotient and same-root state/rate/tangent domains checked in each.',artifacts=['first/outcome.json','second/outcome.json','last/outcome.json']),
 dict(name='Connected compact extension of prior30stable charts',authority='Original all705 known-root join plus entire endpoint projection in independently stable secondquarter, exact first/second/third/fourth adjacency; no cross-gauge interpolation.',artifact='root/known-root-membership.json')]
 return dict(passed=True,native_calls=0,new_interval_arithmetic=False,chart=31,parameter_endpoints=chart['parameter_endpoints'],independent_complement_cells=31,hypotheses=hypotheses,authorities=SPECS,artifacts={k:v[4]for k,v in rows.items()},source_equations=source,analytic_lemma=lemma_binding,historical_polynomial_context=dict(authority=trace,input_binding=tm['input-binding.json']),analytic_conclusion='Under the separately reviewed paper lemma Physical stability boundary and its compact connected-family argument, all hypotheses hold on chart31. The attracting side is sigma=log|lambda|<0 on the local physical periodic-orbit branch. This extends the connected attracting-side family from30 to31charts.',limitations='No numerical axis orientation, quantified collar width, first Lyapunov coefficient/nondegenerate NS torus, global embedded boundary, or complete shrimp-window topology is established by this mapping.')
if __name__=='__main__':
 result=check();path=ROOT/'docs/reviews/native705-chart31-boundary-hypotheses.json';path.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n');print(json.dumps({k:result[k]for k in['passed','chart','independent_complement_cells','parameter_endpoints']}))
