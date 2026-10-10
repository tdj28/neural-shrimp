import sys,json,hashlib,subprocess
from pathlib import Path
from fractions import Fraction as F
sys.path[:0]=['tools/reviews','src']
import review_arclength_connection as O
import review_joined_period18_math as M
BASE=Path('/data2/neural-shrimp-data/audits/s2-attraction-remaining/batch004');REUSED=Path('/data2/neural-shrimp-data/audits/s2-attraction-reused/attempt001')
def verify(root):
 O.verify(root,O.entry(root/'manifest.json')['sha256']);assert all(not p.stat().st_mode&0o222 for p in root.rglob('*'))
def worker(root):
 verify(root);verify(root/'run');rt=O.read(root/'run/runtime.json');done=O.read(root/'run/outcome.json');la=O.read(root/'launcher-result.json');assert done['passed']and la['exit_code']==0 and la['owned_live_pids']==[]and la['worker_pid_absent']
 assert la['source_commit']==rt['source_commit']=='d39b017254b5bf8b8cc64b846328a80c3e1d7356'
 for pid in(rt['pid'],la['owned_group_pid']):assert not Path('/proc',str(pid)).exists()
 for n,identity in rt['sources'].items():
  assert O.entry(root/'run/source'/n)==identity
  assert hashlib.sha256(subprocess.check_output(['git','show',rt['source_commit']+':'+n])).hexdigest()==identity['sha256']
 return {'root':str(root),'outer_manifest':O.entry(root/'manifest.json'),'worker_manifest':O.entry(root/'run/manifest.json'),'execution':rt['source_commit'],'outcome':done,'actual_owned_group_peak_kib':la.get('maximum_group_rss_kib',la.get('max_group_rss_kib'))}
verify(BASE);la=O.read(BASE/'launcher-result.json');assert la['passed']and la['failure']is None
rows=[worker(BASE/f'chart{k}')for k in(155,156)];last=worker(REUSED)
intervals=[tuple(map(F,v['exact_endpoints']))for row in rows for v in row['outcome']['slices']]+[tuple(map(F,last['outcome']['coverage']))];intervals.sort();assert len(intervals)==16 and all(intervals[i][1]==intervals[i+1][0]for i in range(15))and intervals[0][0]==F(43,128)and intervals[-1][1]==F(45,128)
prev=O.read(Path('docs/reviews/s2-attraction-remaining-batch003.json'));assert prev['total_connected_independent_slices']==96 and prev['total_connected_exact_interval']==['27/128','43/128']
# Exact endpoint projection from already independently proved native235D roots.
points=[]
for path,selector,k,gauge in [('/data2/neural-shrimp-data/audits/s2-period18-newgauge-chain/unit-002/run/chart-012.json','independent384',0,'27/128'),('/data2/neural-shrimp-data/audits/s2-period18-small-march/attempt-001/run/march-9-384.json',None,1,'45/128')]:
 p=Path(path);v=O.read(p);v=v if selector is None else v[selector];assert v['passed']and v['precision']==384 and v['coordinate_count']==235 and all(v['gates'].values());ball=v['fresh_endpoints'][k][-1];lo,hi=M.interval(ball);points.append({'gauge':gauge,'path':path,'identity':O.entry(p),'selector':selector,'endpoint_index':k,'hidden_rate_enclosure':ball,'exact_rational_endpoints':[str(lo),str(hi)]})
a=F(points[0]['exact_rational_endpoints'][1]);b=F(points[1]['exact_rational_endpoints'][0]);scale=10**6;aa=a*scale;bb=b*scale;lo=F(-(-aa.numerator//aa.denominator),scale);hi=F(bb.numerator//bb.denominator,scale);assert a<lo<hi<b
val={'status':'passed','evidence_level':'computer-assisted proof using shared source-authenticated analytic acquisitions and Arb','tail_batch_root':str(BASE),'tail_batch_outer_manifest':O.entry(BASE/'manifest.json'),'tail_batch_launcher':la,'tail_charts':rows,'reused_piece':last,'new_slices':16,'previous_receipt':'docs/reviews/s2-attraction-remaining-batch003.json','previous_receipt_identity':O.entry(Path('docs/reviews/s2-attraction-remaining-batch003.json')),'previous_connected_independent_slices':96,'total_connected_independent_slices':112,'total_connected_exact_gauge_interval':['27/128','45/128'],'physical_endpoint_proofs':points,'guaranteed_hidden_rate_interval':[str(lo),str(hi)],'fixed_output_rate':'1848862009810323/2251799813685248','physical_interval_derivation':'The independently proved connected continuous native root branch is attracting at every gauge parameter in the closed cover. Its true left hidden rate is below the lower rational bound and true right hidden rate is above the upper rational bound; the intermediate value theorem guarantees an attracting primitive18 orbit for every hidden rate in this inner interval. No monotonicity or absence of other coexisting roots is asserted.','no_acquisition_calls':True,'all_new_manifests_archives_permissions_and_ownership_checked':True,'claim_limit':'One fixed-output-rate connected attraction segment. No whole center-to-fold path, neutral fold attraction, full shrimp geometry or utility claim.'}
Path('docs/reviews/s2-attraction-tail-independent-outcome.json').write_text(json.dumps(val,indent=2,sort_keys=True)+'\n');print('112-piece cover:',float(lo),float(hi),'new worker seconds',[x['outcome']['elapsed_seconds']for x in rows]+[last['outcome']['elapsed_seconds']]);print('batch',val['tail_batch_outer_manifest']);print('reused',last['outer_manifest'])
