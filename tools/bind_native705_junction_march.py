"""Literal completed24-attempt genealogy, no new native evaluations."""
import gc,hashlib,subprocess
from pathlib import Path
import probe_native705_junction_march as W
import bind_native705_resumed as PREV
import launch_native705_connected as L
need,read,entry=W.need,W.read,W.entry
ROOT=W.ROOT;PREVIOUS=W.W.P.BASE/'maintained003';SOURCE='ee4a9c6840a5168f23756dd3684a98380c877d56'
OUTER='71832651ad41b8ba8696c2e0c9fea1bf03a901b66f2916fca76c6e2bccfba3eb'
ENDPOINT='8900e73e16918b16a5f6e391da04f2b59bd51c061230ffb6e20a12da07e9332e'
PARENT=dict(root=str(PREVIOUS),source=SOURCE,outer=OUTER,endpoint=ENDPOINT)
STATE=(-3,0,'same_gauge')

def contract(outcome,receipt):
 need(outcome==receipt['outcome']and outcome['source']==SOURCE and outcome['status']=='attempt_cap_reached'and outcome['passed']and outcome['attempted']==outcome['launched_attempts']==24,'literal complete24-attempt original packet')
 need(outcome['connected_count']==24 and outcome['fallback_count']==0 and not outcome['owned_live_pids']and [r['attempt']for r in outcome['attempts']if not r['joined']]==[],'all accepted and failed history retained')
 last=outcome['attempts'][-1];need(last['joined']and last['transition']==dict(action='advance',halfwidth=-3,strong=0,rotate=False),'actual successful inherited width/gauge policy state')
 return last

def bind(identity,check,save):
 W.inventory(PREVIOUS,OUTER);receipt=read(ROOT/'docs/reviews/native705-maintained003.json');outcome=read(PREVIOUS/'outcome.json');contract(outcome,receipt)
 need(receipt['root']==str(PREVIOUS)and receipt['source']==SOURCE and receipt['outer']==entry(PREVIOUS/'manifest.json')and receipt['connected_endpoint']['sha256']==ENDPOINT,'exact compact original receipt')
 oldpolicy=read(PREVIOUS/'policy.json');need(oldpolicy['initial_parent']==PREV.PARENT and oldpolicy['status']=='measured-reviewed','same prior completed endpoint authority')
 model0,initialchart,initialanchor,current=PREV.bind(identity,check,lambda name,value:save('previous-'+name,value))
 ep=dict(chart={k:initialchart[k]for k in ['shooting_dimension','normal','origin','parameter_center','parameter_endpoints','settings']},endpoint=initialanchor['box'])
 need(read(PREVIOUS/'resumed-parent-binding.json')['parent']==PREV.PARENT,'same original authenticated prior history')
 model=model0;native={n:identity[n]for n in W.W.C.NATIVE_SOURCES};source_hashes={};rows=[];lastchart=None
 for row in outcome['attempts']:
  idx=row['attempt'];folder=PREVIOUS/f'attempt{idx:03d}';run=folder/'run';W.inventory(folder,row['outer_manifest']['sha256']);W.inventory(run,row['worker']['sha256']);rt=read(run/'runtime.json');done=read(run/'outcome.json');card=read(folder/'input-card.json');inp=read(run/'inputs.json');owner=read(folder/'launcher-result.json');owned=read(folder/'owned-process.json')
  need(not owner['owned_live_pids']and owner['stop_reason']is None and not Path('/proc',str(owned['group'])).exists(),'exact original owned worker absent')
  need(rt['commit']==SOURCE and rt['attempt']==done['attempt']==card['attempt']==idx and rt['settings']==card['settings']==inp['settings']and rt['threads']==dict.fromkeys(W.W.THREADS,'1')and rt['nice']>=10,'same original source/card/runtime')
  for n,v in rt['sources'].items():
   if n not in source_hashes:source_hashes[n]=hashlib.sha256(subprocess.check_output(['git','show',SOURCE+':'+n],cwd=ROOT)).hexdigest()
   need(source_hashes[n]==v['sha256']and entry(run/'source'/n)==v,'entire original source archive '+n)
  for n,v in native.items():need(rt['sources'][n]==v,'same resumed native equations '+n)
  if model is None:model=card['map_inputs']
  need(card['source']==SOURCE and card['kernel']=='fresh-cubic705-maintained-v1'and card['map_inputs']==inp['map_inputs']==model and card['parent']==inp['parent']==current and card['policy']==inp['policy']==entry(PREVIOUS/'policy.json')and inp['card']==entry(folder/'input-card.json'),'literal ordered exact source/map/card/parent')
  expected={'box':ep['endpoint']}
  if card['mode']=='same_gauge':
   need(card['normal']==ep['chart']['normal']and card['origin']==ep['chart']['origin'],'literal unchanged inherited gauge');expected.update(normal=ep['chart']['normal'],origin=ep['chart']['origin'],exact_parameter=ep['chart']['parameter_endpoints'][1])
  else:need(card['mode']=='rotated_gauge','literal changed-gauge case')
  need(card['anchor']==inp['anchor']==expected,'full ordered previous705 root box retained')
  need(done['ticks']==331 and done['events']==W.EXPECTED and done['passed']==row['root_passed']and owner['exit_code']==(0 if done['passed']else 2)and not done['connected'],'complete original native331 unjoined worker')
  if row['joined']:
   chart=read(run/'certificate.json.gz');need(W.W.uniform(chart)and done['passed']and done['strict_inclusions']==705 and len(chart['bound']['coordinate_inclusions'])==705,'original complete705 native root and all non-anchor gates')
   join=read(folder/'join.json');need(join['passed']and join['parameter_inside']and len(join['coordinate_inclusions'])==705 and all(join['coordinate_inclusions']),'all literal native root membership coordinates')
   ep=read(folder/'connected-endpoint.json');need(ep['endpoint']==chart['endpoint_root_boxes'][1]and ep['chart']=={k:chart[k]for k in ['shooting_dimension','normal','origin','parameter_center','parameter_endpoints','settings']},'exact accepted original endpoint')
   current=dict(kind='accepted-maintained-endpoint',root=str(folder),source=SOURCE,outer=row['outer_manifest']['sha256'],worker=row['worker']['sha256'],endpoint=entry(folder/'connected-endpoint.json')['sha256']);lastchart={k:chart[k]for k in L.KEEP};del chart
  else:need(not done['passed']and done['status']=='completed_bounds_failed'and row['transition']['action']=='halve','only retained complete bound failures, no skipped failed join')
  rows.append(dict(attempt=idx,joined=row['joined'],outer=row['outer_manifest'],worker=row['worker']));gc.collect();check()
 need(entry(PREVIOUS/'connected-endpoint.json')['sha256']==ENDPOINT and read(PREVIOUS/'connected-endpoint.json')==ep and current['root']==str(PREVIOUS/'attempt024'),'literal completed right endpoint authority')
 save('resumed-parent-binding.json',dict(parent=PARENT,all24=rows,known_root_source='Reviewed complete original native roots and ordered joins; no new independent arithmetic asserted.',native_sources=native,initial_state=list(STATE),physical_endpoint=ep['endpoint'][-3:]))
 return model,lastchart,L.anchor_from(dict(lastchart,endpoint_root_boxes=[None,ep['endpoint']])),current
