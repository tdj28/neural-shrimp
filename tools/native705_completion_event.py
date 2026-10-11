"""Exact full-state completion event for an authenticated local junction."""
from pathlib import Path
import hashlib,subprocess
import probe_native705_connected as W
import probe_native705_junction_lift as K
need,read,entry=W.need,W.read,W.entry
ROOT=W.ROOT

def bind(policy,identity,model,check,save):
 spec=policy['completion_target'];need(spec['kind']=='independently-proved55-native705-lift-v1'and spec['root']==str(K.OUT.parent),'one literal completed junction lift')
 p=Path(spec['root']);W.inventory(p,spec['outer']);W.inventory(p/'run',spec['worker']);rt=read(p/'run/runtime.json');owner=read(p/'launcher-result.json');out=read(p/'run/outcome.json');ib=read(p/'run/input-binding.json');lift=read(p/'run/native705-lift.json');ref=read(p/'run/known-root-refinement.json')
 need(rt['source_commit']==owner['source_commit']==spec['source']and rt['settings']==K.settings()and owner['exit_code']==0 and owner['failure']is None and not owner['owned_live_pids']and owner['worker_pid_absent'],'exact complete owned source/precision lift')
 for k in('launcher_pid','owned_group_pid'):need(not Path('/proc',str(owner[k])).exists(),'actual old lift process absent')
 need(out['passed']and out['status']=='same_independent_junction_lifted_to_native705'and out['target_acquired']and out['native_calls']==108 and out['events']==228,'complete72profile36target lift history')
 need(ib['independent_root']==str(K.PARENT)and ib['source']==K.SOURCE and ib['outer']==K.PINS[0]and ib['worker']==K.PINS[1]and ib['map_inputs']==model,'same independently certified55 root and physical model')
 need(lift['dimension']==13 and lift['period']==18 and lift['shooting_dimension']==705 and len(lift['box'])==705 and lift['native_calls']==36 and lift['closure_contains_zero']and lift['normalization_contains_zero']and ref['passed']and ref['dimension']==55,'entire literal conditional known705 lift')
 need(entry(p/'run/native705-lift.json')==spec['lift'],'exact immutable known-root enclosure')
 need({str(f.relative_to(p/'run/source'))for f in(p/'run/source').rglob('*')if f.is_file()}==set(rt['sources']),'exact entire lift archive membership')
 for n,v in rt['sources'].items():need(entry(p/'run/source'/n)==v and hashlib.sha256(subprocess.check_output(['git','show',spec['source']+':'+n],cwd=ROOT)).hexdigest()==v['sha256'],'entire original junction source '+n);check()
 for n in K.NATIVE:need(identity[n]==rt['sources'][n]==ib['native_source'][n],'same exact continuation and junction native map '+n)
 profile=read(p/'run/profile.json');need(profile['target_arrays_unopened']and profile['forecast']['admitted'],'original measured lift profile retained')
 target=dict(authority=spec,box=lift['box'],native_model=model,scope='Same proved local complex-unit/simpleflip junction in literal705 coordinates; no phase/sign/gauge transformation.')
 save('completion-target-binding.json',dict(authority=spec,independent55=dict(root=str(K.PARENT),source=K.SOURCE,outer=K.PINS[0],worker=K.PINS[1]),native_sources={n:identity[n]for n in K.NATIVE},lift=spec['lift']));return target

def membership(target,chart):
 need(chart['dimension']==13 and chart['period']==18 and chart['shooting_dimension']==705 and W.W.uniform(chart),'completion test only on a passed complete primitive18 native705 tube')
 need(len(chart['bound']['coordinate_inclusions'])==705 and all(chart['bound']['coordinate_inclusions'])and len(target['box'])==705,'all705 uniform root and known coordinates')
 # No old or locally inferred gauge scalar is allowed. Project the entire
 # independently known junction enclosure into THIS exact chart's gauge.
 test=W.join({'box':target['box']},chart)
 passed=bool(test['passed']and test['parameter_inside']and len(test['coordinate_inclusions'])==705 and all(test['coordinate_inclusions']))
 return dict(passed=passed,authority=target['authority'],known_root_join=test,coordinates=705,strict_inclusions=sum(test['coordinate_inclusions']),scope='Strict all705 whole-known-junction membership in this already accepted connected native unit-pair tube. Failed membership is unresolved, never disconnection. Successful identity does not extend complementary spectrum or establish complete window organization.')
