"""Hash and saved-state replay for the bounded S1R pilot; never runs Newton.

Checks the full accepted graph and its parent/half/reverse witnesses. Incomplete
attempts remain incomplete. Successful replay does not prove uniqueness or
associate any historical start family.
"""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name]='1'
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import numpy as np
from scipy.optimize import linear_sum_assignment
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'tools'))
from neural_shrimp.period2_branch import Options,state_checks,phase_match,eigen_residuals,distance
from neural_shrimp.period2_branch_io import file_entry,identity,load_record,canonical
from neural_shrimp.trainability import SYSTEMS
from neural_shrimp.gdmaps import GridMap
import explore_s1r_branch as runner


def require(condition,message):
    if not condition:raise ValueError(message)


def same(actual,saved,message):
    require(canonical(actual)==canonical(saved),message)


def check_node(node,family,o):
    if not node['accepted']:return
    pair=node['pair'];f,jac=family(node['params'])
    require(pair.shape==(2,72),'wrong full-state pair shape')
    require(np.array_equal(node['rates'],np.array([10.**float(v) for v in node['params']])),'binary64 rate mismatch')
    checks=state_checks(f,pair,o);require(checks['passed'],'replayed closure/primitivity failed')
    same(checks,node['checks'],'changed state diagnostics')
    require(len(node['corrections'])==2,'missing independent phase corrections')
    for i,c in enumerate(node['corrections']):
        require(c['converged'] and c['final_relative_correction']<=o.newton_relative_correction,'unaccepted corrector')
        require(c['candidate_distance']<=o.candidate_distance and distance(c['seed'],c['state'])<=o.candidate_distance,'remote root')
        require(np.array_equal(c['state'],pair[i]),'corrector state mismatch')
    independent=phase_match(np.stack([pair[0],f(pair[0])]),np.stack([f(pair[1]),pair[1]]),o,replay=True)
    same(independent,node['independent_phase_match'],'phase refinement mismatch')
    require(independent['passed'] and independent['phase']==0,'independent phase refinement failed')
    for state in pair:
        image=f(state); residual=f(image)-state
        step=np.linalg.solve(jac(image)@jac(state)-np.eye(len(state)),-residual)
        require(float(np.max(np.abs(step)))/max(1.,float(np.max(np.abs(state))))<=o.newton_relative_correction,'replayed undamped correction')
    j0,j1=jac(pair[0]),jac(pair[1])
    spectra=node['spectra'];require(spectra['passed'],'unqualified spectrum')
    require(len(spectra['phases'])==2,'missing phase spectrum')
    for s,matrix in zip(spectra['phases'],(j1@j0,j0@j1)):
        require(np.allclose(matrix,s['matrix'],atol=1e-12,rtol=1e-12),'monodromy changed')
        diagnostics=eigen_residuals(matrix,s['values'],s['left'],s['right'])
        require(max(np.max(diagnostics['right_residual']),np.max(diagnostics['left_residual']))<=o.eigen_residual,'eigenpair residual')
        radius=float(np.max(np.abs(s['values'])))
        stability='attracting' if radius<1-o.stability_margin else 'unstable' if radius>1+o.stability_margin else 'neutral_or_unresolved'
        require(s['stability']==stability and s['radius']==radius,'changed descriptive stability')
    costs=np.abs(spectra['phases'][0]['values'][:,None]-spectra['phases'][1]['values'][None,:])
    rows,cols=linear_sum_assignment(costs)
    require(float(np.max(costs[rows,cols]))<=o.phase_spectrum_agreement,'phase spectrum mismatch')


def match_nodes(nodes,parent,child,o,phase=0,replay=False):
    require(parent in nodes and child in nodes,'missing edge node')
    require(nodes[parent]['accepted'] and nodes[child]['accepted'],'edge uses unqualified node')
    pair=nodes[parent]['pair'][::-1] if phase else nodes[parent]['pair']
    return phase_match(pair,nodes[child]['pair'],o,replay=replay)


def check_edge(edge,nodes,o):
    require(edge['status'] in ('accepted','rejected'),'unfinished edge cannot be final')
    if edge['status']!='accepted':return
    require(edge['parent_phase'] in (0,1),'invalid phase bit')
    names=('direct','half','endpoint','reverse')
    require(edge['nodes']==[edge[k] for k in names],'missing edge witnesses')
    parent,direct,half,end,reverse=[nodes[k] for k in (edge['parent'],*(edge[k] for k in names))]
    require(np.array_equal(direct['params'],edge['target']) and np.array_equal(end['params'],edge['target']),'wrong endpoint parameters')
    require(np.array_equal(half['params'],(parent['params']+edge['target'])/2),'wrong midpoint parameters')
    require(np.array_equal(reverse['params'],parent['params']),'wrong reverse parameters')
    tests=[('parent_match',edge['parent'],edge['direct'],edge['parent_phase'],False),
       ('half_parent_match',edge['parent'],edge['half'],edge['parent_phase'],False),
       ('half_child_match',edge['half'],edge['endpoint'],0,False),
       ('direct_half_match',edge['direct'],edge['endpoint'],0,True),
       ('reverse_match',edge['parent'],edge['reverse'],edge['parent_phase'],True)]
    for key,p,c,phase,replay in tests:
        actual=match_nodes(nodes,p,c,o,phase,replay)
        require(actual['passed'],'edge full-pair gate failed: '+key)
        same(actual,edge[key],'changed edge match: '+key)


def check(output):
    root=Path(output);plan=runner.load_plan();seed=runner.bound_inputs(plan);o=Options()
    manifest=json.loads((root/'manifest.json').read_text())
    require(manifest['schema']=='s1r-branch-manifest-v1','manifest schema')
    require(set(p.name for p in root.iterdir())==set(manifest['files'])|{'manifest.json'},'file inventory')
    for name,entry in manifest['files'].items():
        require(Path(name).name==name and file_entry(root/name)==entry,'artifact hash: '+name)
    runtime=json.loads((root/'runtime-start.json').read_text());result=json.loads((root/'result.json').read_text())
    require(runtime['sources']==runner.sources(),'source inventory/hash mismatch')
    require(runtime['plan_sha256']==file_entry(ROOT/runner.PLAN)['sha256'],'plan digest mismatch')
    require(all(v=='1' for v in runtime['thread_env'].values()),'thread environment')
    same(runtime['inputs'],plan['inputs'],'input bindings')
    same(runtime['limits'],plan['limits'],'budget bindings')
    require(runtime['cpu_threads']==1 and runtime['nice']>=10 and not runtime['gpu'] and not runtime['paid_compute'],'resource scope')
    require(result['phase_corrector_calls']<=160,'phase-corrector budget')
    require(sum(p.stat().st_size for p in root.iterdir())<=plan['limits']['raw_bytes'],'raw byte budget')
    require(result['lineage']==plan['lineage'] and result['historical_family_association']=='unresolved','false lineage association')
    same(result['historical_families'],plan['historical_families'],'historical families changed')
    require(result['status']==manifest['status'],'completion status mismatch')
    same(result['requested_outputs'],plan['output_offsets'],'requested outputs changed')
    events=[json.loads(line) for line in (root/'journal.jsonl').read_text().splitlines()]
    require([e['sequence'] for e in events]==sorted(set(e['sequence'] for e in events)),'journal sequence ordering')
    nodes={};edges={};hidden=[];segments=[];input_record=None;pending_edges=set();pending_nodes=set();calls=[]
    for e in events:
        kind,payload=load_record(root,e['file']);require(kind==e['kind'],'journal kind mismatch')
        if kind=='node_pending':pending_nodes.add(payload['id'])
        if kind=='node':
            require(payload['id'] in pending_nodes and payload['id'] not in nodes,'node state machine')
            pending_nodes.remove(payload['id']);nodes[payload['id']]=payload
        if kind=='edge_pending':pending_edges.add(payload['id'])
        if kind=='edge':
            require(payload['id'] in pending_edges and payload['id'] not in edges,'edge state machine')
            pending_edges.remove(payload['id']);edges[payload['id']]=payload
        if kind=='hidden_link':hidden.append(payload)
        if kind=='segment':segments.append(payload)
        if kind=='inputs':require(input_record is None,'duplicate model inputs');input_record=payload
        if kind=='corrector_start':calls.append(payload['call'])
    require(calls==list(range(1,result['phase_corrector_calls']+1)),'corrector journal count')
    complete=result['status'] in ('completed_local_segment','completed_with_gaps','anchor_failed','hidden_backoff_gap')
    if complete:require(not pending_edges and not pending_nodes,'completed claim with pending work')
    if input_record is None:
        require(not complete and not nodes,'missing model identity');return dict(status='valid_incomplete_attempt',nodes=0)
    net,x,y,theta0=SYSTEMS['S1R'](28002)
    same(input_record['model'],asdict(net),'model identity')
    for key,value in dict(x=x,y=y,theta0=theta0,seed=seed).items():
        same(input_record['identity'][key],identity(value),'input identity '+key)
        require(np.array_equal(value,input_record[key]),'input bytes '+key)
    def family(params):return GridMap(net,x,y,[10.**float(params[0])],[10.**float(params[1])]).cell_map(0)
    for node in nodes.values():check_node(node,family,o)
    rates=np.array([plan['log10_hidden_anchor'],plan['log10_output_anchor']])
    if result['anchor'] is not None:
        require(result['anchor'] in nodes,'missing final anchor')
        anchor=nodes[result['anchor']]
        require(np.array_equal(anchor['params'],rates) and anchor['purpose']=='anchor','wrong anchor')
        require(np.array_equal(anchor['corrections'][0]['seed'],seed),'anchor reseeded')
    current=result['anchor'];ended=False
    require(len(hidden)<=4,'too many hidden links')
    for i,link in enumerate(hidden,1):
        require(not ended and link['parent']==current,'hidden path crosses a gap')
        require(np.array_equal(nodes[link['child']]['params'],np.array([rates[0]-.002*i/4,rates[1]])),'hidden schedule changed')
        ended=link['status']!='accepted';current=link['child']
    if segments:require(len(hidden)==4 and not ended,'output continuation before backoff')
    central=current
    expected_direction=1;expected_offset=.001;current=central
    for segment in segments:
        if segment['direction']==-1 and expected_direction==1:
            expected_direction=-1;expected_offset=.001;current=central
        require(segment['direction']==expected_direction and expected_offset<=.002,'changed segment schedule')
        require(segment['parent']==current,'segment jumps between branches')
        require(np.array_equal(segment['target'],np.array([rates[0]-.002,rates[1]+expected_direction*expected_offset])),'output schedule changed')
        if segment['status']=='connected':
            current=segment['endpoint'];expected_offset+=.001
        else:expected_offset=float('inf')

    for edge in edges.values():check_edge(edge,nodes,o)
    for link in hidden:
        require(link['status'] in ('accepted','unresolved_gap'),'hidden edge status')
        if link['status']=='accepted':
            m=match_nodes(nodes,link['parent'],link['child'],o,link['parent_phase'])
            require(m['passed'],'hidden full-pair match');same(m,link['match'],'hidden match changed')
    same(hidden,result['hidden_links'],'hidden graph differs from result')
    same(segments,result['segments'],'segment graph differs from result')
    for segment in segments:
        current=segment['parent']
        for key in segment['path']:
            require(key in edges and edges[key]['status']=='accepted','path crosses rejected/unresolved edge')
            require(edges[key]['parent']==current,'disconnected path');current=edges[key]['direct']
        if segment['status']=='connected':
            require(current==segment['endpoint'] and np.array_equal(nodes[current]['params'],segment['target']),'false endpoint')
        else:require(segment['status']=='unresolved_gap','unknown segment status')
    if result['status']=='completed_local_segment':
        require(len(segments)==4 and all(s['status']=='connected' for s in segments),'missing five-output coverage')
    return dict(status='passed_saved_state_replay',attempt_status=result['status'],qualified_pairs=sum(n['accepted'] for n in nodes.values()),
        accepted_edges=sum(e['status']=='accepted' for e in edges.values()),pending_nodes=sorted(pending_nodes),pending_edges=sorted(pending_edges),
        historical_family_association='unresolved',evidence='floating-point local consistency; no uniqueness or global ancestry')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('output',type=Path);args=parser.parse_args()
    print(json.dumps(check(args.output),indent=2))


if __name__=='__main__':main()
