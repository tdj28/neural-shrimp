"""Prospective lower-bootstrap saved-outcome audit: no trajectory or Newton solve."""
import os
THREADS=('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')
for k in THREADS:os.environ[k]='1'
import argparse
from dataclasses import asdict
from datetime import datetime,timezone
import json
from pathlib import Path
import platform
import signal
import subprocess
import sys
import time
import numpy as np
import scipy
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools')]
import audit_s1r_bootstrap_outcome as prior
from audit_s1r_branch_outcome import require,entry,digest,inventory,record,read_json,loss_gradient_hessian
from neural_shrimp.period2_branch_io import Store,canonical,identity
from neural_shrimp.gdmaps import GridMap
from neural_shrimp.trainability import SYSTEMS
from neural_shrimp.period2_branch import Options
BASIS='3346fcaf684fd74b24f58ce4da558358938edcee'
EXECUTION='b5271a4bfeb74c9024eb2c9a40369924c6aa1304'
REVIEW='bb71b9045a6e6850031ebdd6ba0c6770d0f6cb9a'
EXECUTION_ROOT='/home/d7082791602/.codex/worktrees/post-edge-next-steps/neural-shrimp'
MANIFEST='72f88bd7b05570f89a98da90b6af591cac94a4986bb10f89e749bcb03e2c181d'
LAUNCHER_MANIFEST='14c3c2866a1550b46a763c7221fc58ef3803b59353d20a50058672f6112d1d4d'
OLD_SOURCE='586e74927215ff7ab5ebc81fbc29720c9322f291'
OLD_MANIFEST='391092f55c06b7289b852ca58603dd680f7718fa91f747bfc2d749704d5d5974'
TIMING_SOURCE='db10bd1e1b25b0e24b1f0f8f97cb84035742c776'
TIMING_MANIFEST='45c35991b7ec56f7d9888b6dcf28434396ff94f65de7bf90ba9822408c5f3c8b'
SOURCE_FILES=('docs/plans/s1r-lower-association-implementation.md','docs/plans/s1r-next-association-decision.md',
 'src/neural_shrimp/__init__.py','src/neural_shrimp/gdmaps.py','src/neural_shrimp/trainability.py','src/neural_shrimp/models.py',
 'src/neural_shrimp/period2_branch.py','src/neural_shrimp/period2_branch_io.py','src/neural_shrimp/s1r_bootstrap.py',
 'src/neural_shrimp/s1r_lower_association.py','tools/explore_s1r_lower_association.py','tools/launch_s1r_lower_association.py')
AUDIT_FILES=('tools/audit_s1r_lower_outcome.py','tools/audit_s1r_bootstrap_outcome.py','tools/audit_s1r_branch_outcome.py',
 'tests/test_s1r_lower_outcome_audit.py','docs/reviews/s1r-lower-outcome-audit-card.md')
ROLE_SPECS={
 'upper':('00964-node.json','node-0047',1.1,'cb049836108a7a1e1b1f364d3f90dde158b258b6111f281f07578404b1580a3b','b93240d3f0c837c82ea900ae48bb18e2a4e34c84e97ec9d59b2837a6b9d9da1f'),
 'lower':('00107-node.json','node-0001',.95,'443f4616daa6e5951c883f6ec5c3859f06d6b91ae3aadd9d22cee01d0fe14d82','bddae7ec37d4025ebe8cd7f4a449930c754627e3d782046c43e7b58bc38d4702')}
OUTPUT=.6271428571428572
LIMITS=dict(internal_seconds=100,external_seconds=115,kill_grace_seconds=5,nominal_links=75,
 phase_correctors=644,link_attempts=80,store_bytes=255*1024**2,launcher_bytes=1024**2,total_raw_bytes=256*1024**2)
COMPLETE={'completed_common_control_comparison','endpoint_requalification_failed','bridge_gap'}
INCOMPLETE={'resource_stop','failed_or_interrupted'}
STRUCTURAL={'node_pending','node','edge_pending','edge','bridge_anchor','bridge_segment','association'}
ALLOWED=STRUCTURAL|{'bound_inputs','corrector_start','corrector_iteration','corrector_end','line_search','pair_spectrum_pending','terminal'}
exact=prior.exact


def source_inventory():
    result={}
    for name in SOURCE_FILES:
        data=subprocess.check_output(['git','show',BASIS+':'+name],cwd=ROOT)
        require(data==subprocess.check_output(['git','show',EXECUTION+':'+name],cwd=ROOT),'execution differs from reviewed source '+name)
        require(data==(ROOT/name).read_bytes(),'local shared primitive changed '+name)
        result[name]=dict(bytes=len(data),sha256=digest(data))
    return result


def saved_events(folder):
    journal=[json.loads(s) for s in (folder/'journal.jsonl').read_text().splitlines()]
    sequences=[e['sequence'] for e in journal];require(sequences==sorted(set(sequences)),'ordered unique journal')
    out=[];used=set()
    for e in journal:
        name=e['file'];require(Path(name).name==name and name not in used,'journal filename')
        obj=read_json(folder/name);kind,p=record(folder,name)
        require(kind in ALLOWED and obj['sequence']==e['sequence'] and kind==e['kind'],'event schema/sequence')
        require(name==f"{e['sequence']:05d}-{kind}.json",'canonical event filename')
        used.add(name)
        if 'arrays_file' in obj:used.add(obj['arrays_file'])
        out.append((kind,p))
    require(used=={p.name for p in folder.iterdir() if p.name[:5].isdigit()},'all array/event files journaled')
    return out,journal


def immutable(folder):
    require(not folder.is_symlink() and not folder.stat().st_mode&0o222,'immutable real directory')
    require(all(p.is_file() and not p.is_symlink() and not p.stat().st_mode&0o222 for p in folder.iterdir()),'immutable regular files')


def launch_binding(folder,raw,runtime,result,sources):
    m=inventory(folder,LAUNCHER_MANIFEST);immutable(folder)
    require(set(m['files'])=={'launcher-source.py','runtime-start.json','stdout.log','receipt.json'},'launcher inventory')
    require(entry(folder/'launcher-source.py')==sources['tools/launch_s1r_lower_association.py'],'launcher source')
    rt=read_json(folder/'runtime-start.json');r=read_json(folder/'receipt.json')
    require(rt['schema']=='s1r-lower-launcher-v1' and rt['source_commit']==r['source_commit']==EXECUTION and rt['source_review_commit']==r['source_review_commit']==REVIEW,'launcher source/review')
    exact(rt['sources'],sources,'launcher source inventory')
    require(rt['output']==str(raw) and rt['launcher']==str(folder) and rt['cwd']==EXECUTION_ROOT,'launcher paths')
    expected=['timeout','--signal=TERM','--kill-after=5s','115s','nice','-n','10','/data2/neural-shrimp/.venv/bin/python',
      'tools/explore_s1r_lower_association.py','--out',str(raw),'--review-commit',REVIEW]
    exact(rt['command'],expected,'launcher exact command')
    require(rt['nice']>=10 and rt['thread_env']==dict.fromkeys(THREADS,'1'),'launcher CPU limits')
    for key in ('external_seconds','kill_grace_seconds','store_bytes','launcher_bytes','total_raw_bytes'):
        require(rt[key]==LIMITS[key],'launcher fixed limit '+key)
    require(type(rt['launcher_pid']) is int and type(rt['child_pid']) is int and rt['launcher_pid']>0 and rt['child_pid']>0 and rt['child_pid']==r['child_pid'],'launcher owned child identity')
    require(rt['launcher_pid']!=rt['child_pid'] and type(runtime['pid']) is int and runtime['pid']>0 and runtime['pid'] not in (rt['launcher_pid'],rt['child_pid']),'distinct producer process identity')
    require(r['status']==m['status']=='process_finished','completed launcher receipt')
    require(r['failure'] is None and not r['log_truncated'] and r['within_store_cap'],'clean launcher required for audit verdict')
    expected_exit=0 if result['status'] in COMPLETE else 2 if result['status']=='resource_stop' else None
    require(r['process_exit']==expected_exit if expected_exit is not None else r['process_exit']!=0,'exit/status agreement')
    entries={p.name:entry(p) for p in sorted(raw.iterdir())}
    require(r['raw_file_count']==len(entries) and r['raw_bytes']==sum(e['bytes'] for e in entries.values()),'launcher raw inventory size')
    require(r['raw_inventory_sha256']==digest(canonical(entries)),'launcher complete raw inventory digest')
    require(r['raw_manifest']==entry(raw/'manifest.json') and r['raw_result']==entry(raw/'result.json'),'launcher result/manifest binding')
    require(r['stdout']==entry(folder/'stdout.log') and r['stdout']['bytes']<=128*1024,'bounded stdout identity')
    if result['status']!='failed_or_interrupted':exact(read_json(folder/'stdout.log'),result,'stdout result')
    require(0<=result['elapsed_seconds']<=120 and result['elapsed_seconds']<=r['elapsed_seconds'],'recorded child time bound')
    wall=(datetime.fromisoformat(r['finished_utc'])-datetime.fromisoformat(rt['started_utc'])).total_seconds()
    require(np.isfinite(r['elapsed_seconds']) and 0<=r['elapsed_seconds']<=125 and abs(wall-r['elapsed_seconds'])<=5,'launcher clock consistency')
    require(datetime.fromisoformat(rt['started_utc'])<=datetime.fromisoformat(runtime['started_utc'])<=datetime.fromisoformat(r['finished_utc']),'producer start within launcher lifetime')
    require(sum(p.stat().st_size for p in folder.iterdir())<=1024**2,'launcher byte cap')
    return dict(path=str(folder),manifest_sha256=LAUNCHER_MANIFEST,elapsed_seconds=r['elapsed_seconds'],exit_code=r['process_exit'],
      child_pid=r['child_pid'],raw_bytes=r['raw_bytes'],limitation='Owned timeout PID and wait-based termination are source-bound; historical PIDs are not queried for current ownership.')


def bind_inputs(inputs,old,timing):
    m=inventory(old,OLD_MANIFEST);inventory(timing,TIMING_MANIFEST)
    old_rt=read_json(old/'runtime-start.json');time_rt=read_json(timing/'runtime-start.json')
    require(old_rt['source_commit']==OLD_SOURCE and time_rt['source_commit']==TIMING_SOURCE,'prior source identities')
    for rt in (old_rt,time_rt):
        require(rt['nice']>=10 and rt['thread_env']==dict.fromkeys(THREADS,'1'),'timing CPU basis')
        require(rt['numpy']==np.__version__ and rt['scipy']==scipy.__version__,'timing library basis')
    _,original=record(old,'00000-bound_inputs.json');net,x,y,theta=SYSTEMS['S1R'](28002)
    exact(inputs['model'],asdict(net),'model');exact(inputs['identity'],original['identity'],'complete input identity')
    for key,value in dict(x=x,y=y,theta0=theta).items():
        exact(value,original[key],'prior constructor '+key);exact(identity(value),inputs['identity'][key],'identity '+key)
        if key!='theta0':exact(value,inputs[key],'new bound data '+key)
    require(inputs['input_source']==OLD_SOURCE and inputs['input_manifest_sha256']==OLD_MANIFEST,'bound lineage source')
    for role,(filename,node_id,hidden,json_sha,pair_sha) in ROLE_SPECS.items():
        require(entry(old/filename)['sha256']==json_sha,'role record hash')
        kind,node=record(old,filename);require(kind=='node' and node['id']==node_id and node['accepted'],'prior qualified role')
        require(identity(node['pair'])['sha256']==pair_sha,'exact native role pair')
        exact(node['params'],np.array([hidden,OUTPUT]),'role exact controls')
        exact(inputs['endpoints'][role],{k:node[k] for k in ('id','params','rates','pair')},'role source binding')
    t=read_json(timing/'result.json');c=read_json(old/'result.json');size=sum(e['bytes'] for e in m['files'].values())+entry(old/'manifest.json')['bytes']
    require(t['status']=='completed_local_segment' and type(t['phase_corrector_calls']) is int and t['phase_corrector_calls']==42,'timing pilot counts')
    require(c['status']=='completed_common_control_comparison' and c['link_attempts']==12 and size==27218975,'storage projection basis')
    elapsed=t['elapsed_seconds'];require(type(elapsed) in (int,float) and np.isfinite(elapsed) and elapsed>0,'timing duration')
    seconds=2*(elapsed/42)*644+10;raw=size*80/12;ordinary=255*1024**2-(255*1024**2)//4
    require(seconds<=100 and raw<=ordinary,'unchanged cost gate')
    gate=dict(projected_seconds=seconds,projected_raw_bytes=raw,ordinary_byte_allowance=ordinary,timing_elapsed_seconds=elapsed,
      timing_phase_calls=42,collection_bytes=size,collection_link_attempts=12,maximum_phase_calls=644,maximum_link_attempts=80,
      seconds_formula='2 * (timing_elapsed_seconds / 42) * 644 + 10',bytes_formula='27218975 * 80 / 12',passes=True,
      limitation='heuristic bound-input timing/storage projection; new-control conditioning, retries and filesystem costs may differ',
      timing_root=str(timing),timing_manifest_sha256=TIMING_MANIFEST,collection_root=str(old),collection_manifest_sha256=OLD_MANIFEST)
    exact(inputs['cost_gate'],gate,'independent cost gate arithmetic')
    return net,gate


def snap(n):
    if n is None:return None
    return dict(id=n['id'],params=n['params'],rates=n['rates'],pair=n['pair'],carried_phase=int(n.get('carried_phase',0)))


class LowerStructure:
    """Independent lower-unit schedule/durable-state parser; common pair-witness checks reused."""
    node=prior.Structure.node
    def __init__(self,events,result):
        self.queue=[(k,p) for k,p in events if k in STRUCTURAL];self.i=0;self.result=result
        self.nodes={};self.pending={};self.edges=[];self.segments=[];self.last=None;self.accepted=[];self.association=None
    def take(self,kind):
        if self.i==len(self.queue):raise prior.Truncated()
        k,p=self.queue[self.i];require(k==kind,'expected '+kind+' got '+k);self.i+=1;return p
    def edge(self,parent,target,predictor):
        e=prior.Structure.edge(self,parent,target,predictor)
        if e['status']=='accepted':
            native=self.nodes[e['direct']];self.last=snap(native);self.last['carried_phase']=e['parent_match']['phase']
            self.last['pair']=prior.aligned(native['pair'],self.last['carried_phase']);self.accepted.append(e['id'])
        return e
    def segment(self,parent,target,previous,index):
        current=parent;width=abs(target[0]-parent['params'][0]);path=[];halvings=0
        while current['params'][0]!=target[0]:
            remaining=target[0]-current['params'][0]
            proposal=target.copy() if abs(remaining)<=width else np.array([current['params'][0]+np.copysign(width,remaining),target[1]])
            prediction=current['pair']
            if previous is not None:
                denominator=current['params'][0]-previous['params'][0]
                if denominator!=0:prediction=current['pair']+(proposal[0]-current['params'][0])/denominator*(current['pair']-previous['pair'])
            require(len(self.edges)<=80,'global link cap')
            if len(self.edges)==80:raise prior.Truncated()
            e=self.edge(current,proposal,prediction)
            if e['status']!='accepted':
                halvings+=1;width/=2
                if halvings>4 or width<.000125*(1-1e-10):
                    expected=dict(status='unresolved_gap',parent=parent['id'],last_accepted=current['id'],target=target,path=path,
                      failed_edge=e['id'],reason=e['reason'],halvings=halvings);break
            else:
                child=self.nodes[e['direct']].copy();child['carried_phase']=e['parent_match']['phase']
                child['pair']=prior.aligned(child['pair'],child['carried_phase']);previous,current=current,child;path.append(e['id'])
        else:expected=dict(status='connected',parent=parent['id'],target=target,path=path,endpoint=current['id'],halvings=halvings)
        expected['macro_index']=index;exact(self.take('bridge_segment'),expected,'macro path/halving/gap')
        self.segments.append(expected);return current,previous,expected['status']
    def run(self,inputs):
        upper,lower=[inputs['endpoints'][k] for k in ('upper','lower')]
        targets=[np.array([float(upper['params'][0])+(float(lower['params'][0])-float(upper['params'][0]))*i/75,upper['params'][1]]) for i in range(1,75)]+[lower['params'].copy()]
        if 'macro_targets' in self.result:exact(self.result['macro_targets'],[p.tolist() for p in targets],'fixed 75 endpoints')
        else:require(self.result['status'] in INCOMPLETE and not self.queue,'interrupted before schedule only')
        upper_id=lower_id=None;expected_status=None
        try:
            if 'macro_targets' not in self.result:raise prior.Truncated()
            start=self.node(upper['params'],upper['pair'],'saved_upper_requalification');upper_id=start['id']
            comparator=self.node(lower['params'],lower['pair'],'saved_lower_requalification');lower_id=comparator['id']
            if not start['accepted'] or not comparator['accepted']:expected_status='endpoint_requalification_failed'
            else:
                for a,b in ((start,upper),(comparator,lower)):
                    for key in ('params','rates'):exact(a[key],b[key],'requalified exact controls')
                self.last=snap(start);exact(self.take('bridge_anchor'),self.last,'durable anchor')
                current,previous=start,None
                for index,target in enumerate(targets,1):
                    current,previous,status=self.segment(current,target,previous,index)
                    if status!='connected':expected_status='bridge_gap';break
                else:
                    association=prior.common(current,comparator)
                    association['scope']='saved upper-lineage pair versus newly reproduced lower-bootstrap pair at identical controls; finite witnessed numerical path, no historical-state ancestry, interval continuation or uniqueness proof'
                    exact(self.take('association'),association,'common-control comparison');self.association=association
                    expected_status='completed_common_control_comparison'
        except prior.Truncated:require(self.result['status'] in INCOMPLETE,'complete result with unfinished events')
        require(self.i==len(self.queue),'unused structural event or gap jump')
        if expected_status is not None:require(self.result['status']==expected_status,'terminal mathematical status')
        require(self.result.get('upper_requalification')==upper_id and self.result.get('lower_requalification')==lower_id,'endpoint record ownership')
        exact(self.result['bridge'],self.segments,'result macro paths');exact(self.result['association'],self.association,'result association')
        exact(self.result['witnessed_edge_ids'],self.accepted,'accepted durable path')
        require(self.result['last_witnessed_node']==(None if self.last is None else self.last['id']),'last witnessed ID')
        require(self.result['processed_macro_segments']==len(self.segments) and self.result['completed_macro_links']==sum(s['status']=='connected' for s in self.segments),'macro counts')
        return dict(completed_nodes=len(self.nodes),durable_node_starts=len(self.pending),durable_link_starts=len(self.edges),
          accepted_links=len(self.accepted),processed_macro_segments=len(self.segments),association=self.association)


def terminal_audit(events,result,structure):
    terminals=[p for k,p in events if k=='terminal']
    if not terminals:
        require(result['status'] in INCOMPLETE and 'terminal_checkpoint_failure' in result,'missing durable terminal')
        return dict(status='missing_terminal',claim='incomplete only')
    require(len(terminals)==1 and events[-1][0]=='terminal','one final terminal event');t=terminals[0]
    ns={p['id'] for k,p in events if k=='node_pending'};nf={p['id'] for k,p in events if k=='node'}
    es={p['id'] for k,p in events if k=='edge_pending'};ef={p['id'] for k,p in events if k=='edge'}
    require(nf<=ns and ef<=es,'durable completion requires start')
    for key,actual in (('pending_nodes',ns-nf),('durably_finished_nodes',nf),('pending_edges',es-ef),('durably_finished_edges',ef)):
        exact(t[key],sorted(actual),'terminal durable '+key)
    for key,started,prefix in (('unpersisted_node_attempts',ns,'node'),('unpersisted_edge_attempts',es,'edge')):
        unpersisted=t[key];require(unpersisted in ([],[f'{prefix}-{len(started):04d}']),'next unpersisted attempt only')
        if unpersisted:require(result['status'] in INCOMPLETE,'unpersisted completed run')
    require(result['link_attempts']==len(es)+len(t['unpersisted_edge_attempts'])<=80,'total attempted link count')
    require(len(ns)+len(t['unpersisted_node_attempts'])<=322,'maximum node attempts')
    if result['status'] in COMPLETE:require(not (ns-nf or es-ef or t['unpersisted_node_attempts'] or t['unpersisted_edge_attempts']),'complete run with unfinished journal')
    exact(t['last_witnessed'],structure.last,'terminal last durable pair');exact(t['accepted_edges'],structure.accepted,'terminal accepted path')
    corrections=[p for k,p in events if k in ('corrector_start','corrector_iteration','corrector_end')]
    exact(t['latest_corrector'],corrections[-1] if corrections else None,'latest durable proposal')
    return dict(status='consistent',pending_nodes=t['pending_nodes'],pending_edges=t['pending_edges'],
      unpersisted_node_attempts=t['unpersisted_node_attempts'],unpersisted_edge_attempts=t['unpersisted_edge_attempts'],last_witnessed_id=None if structure.last is None else structure.last['id'])


def audit(args,emit):
    require(args.producer_commit==EXECUTION and args.manifest_sha256==MANIFEST and args.launcher_manifest_sha256==LAUNCHER_MANIFEST,'frozen input identities')
    m=inventory(args.collection,MANIFEST);immutable(args.collection)
    runtime=read_json(args.collection/'runtime-start.json');result=read_json(args.collection/'result.json')
    sources=source_inventory();exact(runtime['sources'],sources,'producer full source inventory')
    require(runtime['source_commit']==EXECUTION and runtime['source_review_commit']==REVIEW,'execution/review binding')
    require(result['schema']=='s1r-lower-association-v1' and result['status'] in COMPLETE|INCOMPLETE and m['status']==result['status'],'result schema/status')
    require(result['historical_family_association']=='unresolved','historical scope')
    exact(runtime['limits'],LIMITS,'scientific/resource limits');exact(runtime['scientific_options'],asdict(Options()),'unchanged mathematical gates')
    require(runtime['nice']>=10 and runtime['thread_env']==dict.fromkeys(THREADS,'1') and runtime['device']=='cpu' and not runtime['gpu'] and not runtime['paid_compute'],'CPU runtime')
    require(runtime['input_root']==str(args.original) and runtime['input_manifest_sha256']==OLD_MANIFEST and runtime['timing_root']==str(args.timing) and runtime['timing_manifest_sha256']==TIMING_MANIFEST,'prior input roots')
    raw_bytes=sum(p.stat().st_size for p in args.collection.iterdir());require(raw_bytes<=LIMITS['store_bytes'],'Store cap')
    launcher=launch_binding(args.launcher,args.collection,runtime,result,sources)
    require(raw_bytes+sum(p.stat().st_size for p in args.launcher.iterdir())<=LIMITS['total_raw_bytes'],'combined artifact cap')
    events,journal=saved_events(args.collection)
    exact(read_json(args.collection/'checkpoint.json'),dict(status=result['status'],latest=journal[-1] if journal else None),'final checkpoint')
    emit('bindings',dict(producer_commit=EXECUTION,reviewed_basis=BASIS,source_review=REVIEW,manifest_sha256=MANIFEST,launcher=launcher,
      files=len(m['files'])+1,raw_bytes=raw_bytes))
    bindings=[p for k,p in events if k=='bound_inputs'];require(len(bindings)<=1,'single input record')
    if not bindings:
        require(result['status'] in INCOMPLETE and result['phase_corrector_calls']==0 and result['link_attempts']==0 and result['association'] is None,'interrupted before inputs')
        require(not any(k in STRUCTURAL for k,p in events),'no graph without inputs')
        return dict(status='valid_incomplete_before_inputs',producer_status=result['status'],launcher=launcher)
    inputs=bindings[0];immutable(args.original);immutable(args.timing)
    net,gate=bind_inputs(inputs,args.original,args.timing);exact(result['cost_gate'],gate,'result cost gate')
    emit('bound_input_audit',dict(cost_gate=gate,model=inputs['model'],identity=inputs['identity']))
    history=prior.corrector_history(events,result,max_calls=644,complete_statuses=COMPLETE);emit('corrector_history',history)
    structure=LowerStructure(events,result);graph=structure.run(inputs);terminal=terminal_audit(events,result,structure)
    emit('witness_graph',graph);emit('terminal_consistency',terminal)
    def family(params):return GridMap(net,inputs['x'],inputs['y'],[10.**float(params[0])],[10.**float(params[1])]).cell_map(0)
    qualified=[]
    for key,n in structure.nodes.items():qualified.append(prior.qualification(n,structure.pending[key],family))
    emit('all_native_qualifications',qualified)
    names=[key for key,n in structure.nodes.items() if n['accepted'] and n['purpose'] in ('saved_upper_requalification','saved_lower_requalification')]
    direct=[key for key,n in structure.nodes.items() if n['accepted'] and n['purpose']=='hidden_bridge:direct']
    if direct:names.extend(direct[i] for i in sorted({0,len(direct)//2,len(direct)-1}))
    representatives=[]
    for key in dict.fromkeys(names):
        n=structure.nodes[key];rates=np.r_[np.full(64,n['rates'][0]),np.full(8,n['rates'][1])];cache={}
        def derivatives(t):
            identity=t.tobytes()
            if identity not in cache:cache[identity]=loss_gradient_hessian(t,inputs['x'],inputs['y'],inputs['model'])
            return cache[identity]
        def independent_family(params):
            exact(params,n['params'],'representative controls')
            return lambda t:t-rates*derivatives(t)[1],lambda t:np.eye(72)-rates[:,None]*derivatives(t)[2]
        native=[]
        for phase,t in enumerate(n['pair']):
            loss,gradient,hessian=derivatives(t);native.append(dict(phase=phase,state=t,loss=loss,gradient=gradient,hessian=hessian))
        emit('analytic_native_matrices',dict(node=key,params=n['params'],phases=native))
        representatives.append(prior.qualification(n,structure.pending[key],independent_family,True))
    emit('analytic_representatives',representatives)
    return dict(status='passed_saved_data_consistency' if result['status'] in COMPLETE else 'valid_incomplete_attempt',
      producer_status=result['status'],producer_execution_source=EXECUTION,reviewed_source_basis=BASIS,producer_source_review=REVIEW,
      collection_manifest_sha256=MANIFEST,launcher_manifest_sha256=LAUNCHER_MANIFEST,raw_files=len(m['files'])+1,raw_bytes=raw_bytes,
      launcher=launcher,cost_gate=gate,corrector_history=history,witness_graph=graph,terminal_consistency=terminal,
      qualified_native_pairs=sum(bool(n['accepted']) for n in qualified),independent_representatives=representatives,
      historical_family_association='unresolved',limitations=[
       'All completed native qualifiers use the shared reviewed model Jacobian; at most five role-selected pairs use independent full analytic gradient/Hessian reconstruction.',
       'Saved correction step arithmetic is checked; intermediate Newton residual equations and linear solves are not independently recomputed.',
       'Finite numerical witness paths and endpoint comparisons do not prove interval continuation, uniqueness, historical ancestry, global disconnection or behavioral benefit.',
       'No new bootstrap, profile, free trajectory or Newton correction was run by this audit.'])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('collection','launcher','original','timing','out'):p.add_argument('--'+key,type=Path,required=True)
    for key in ('producer-commit','manifest-sha256','launcher-manifest-sha256'):p.add_argument('--'+key,required=True)
    args=p.parse_args();require(os.getpriority(os.PRIO_PROCESS,0)>=10,'nice10 required')
    source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();bindings={}
    for name in (*SOURCE_FILES,*AUDIT_FILES):
        data=(ROOT/name).read_bytes();require(data==subprocess.check_output(['git','show','HEAD:'+name],cwd=ROOT),'uncommitted audit dependency '+name)
        bindings[name]=entry(ROOT/name)
    store=Store(args.out,31*1024**2);started=time.monotonic()
    runtime=dict(audit_source_commit=source,sources=bindings,started_utc=datetime.now(timezone.utc).isoformat(),pid=os.getpid(),nice=os.getpriority(os.PRIO_PROCESS,0),
      thread_env={k:os.environ[k] for k in THREADS},inputs={k:str(v) for k,v in vars(args).items() if k!='out'},
      limits=dict(internal_seconds=45,external_seconds=60,kill_grace_seconds=5,store_bytes=31*1024**2,launcher_bytes=1024**2,raw_bytes=32*1024**2),numpy=np.__version__,scipy=scipy.__version__,python=platform.python_version(),gpu=False,paid=False)
    store.json('runtime-start.json',runtime)
    def stop(signum,frame):raise TimeoutError('bounded independent audit deadline')
    signal.signal(signal.SIGALRM,stop);signal.signal(signal.SIGTERM,stop);signal.alarm(45)
    result=dict(status='failed',audit_source_commit=source)
    try:result.update(audit(args,store.emit))
    except BaseException as error:result.update(error=type(error).__name__+': '+str(error));raise
    finally:
        signal.alarm(0);result['elapsed_seconds']=time.monotonic()-started;store.finish(result)
        for path in args.out.iterdir():path.chmod(0o444)
        args.out.chmod(0o555)
    print(json.dumps(dict(status=result['status'],elapsed_seconds=result['elapsed_seconds'])))


if __name__=='__main__':main()
