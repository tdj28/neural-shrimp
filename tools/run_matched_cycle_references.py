#!/usr/bin/env python3
"""EXP-023: two fixed-parameter bridges and a six-cycle unforced-reference panel.

Historical preparation is metadata only. New neural work is confined to run()
after exact-plan/source/private-push guards; verify() replays saved packets only.
"""
from __future__ import annotations
import argparse
from collections import Counter
import copy
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import resource
import signal
import subprocess
import sys
import time

import run_refined_two_write_memory as common
from run_refined_two_write_memory import work, legacy, codec, runtime, np
import cycle_reference_selection as selection
import matched_cycle_seed_bridge as bridge
import matched_cycle_reference_acquisition as acquisition
import matched_cycle_reference_evaluation as evaluation

ROOT=common.ROOT
EXP='EXP-023'
OLD=ROOT/'evidence/EXP-006/attempt-001'
OLD_SOURCE='ddb568e6cd67ef2cfbdb45e05710cd86025ef548'
OLD_RECEIPT='cdcfc91b9ce8a444a5007e1eda35a1ed34d492810ca1a152598754df27467ac2'
ANCHOR=Path('/data2/neural-shrimp-data/raw/EXP-018/attempt-001')
ANCHOR_SOURCE='7c9b1977f5d12e4354ff41c537778a3a284a4027'
ANCHOR_RECEIPT='b4b56a8fbcc592285cebdbf15c2182a3173c62d3e770ebc1681cd9a142066ce1'
PREPARATION=Path('/data2/neural-shrimp-data/audits/EXP-022/selection-001')
SELECTION_SHA='b8bcf7f11fc82623993e07af54cc3d6a42a338d18561cc148f4316329952416b'
PREPARATION_RECEIPT='56110bbbe632754295490ad24545946ed87577cbc0313c9b306c78dd130416a1'
CLOSURE=Path('/data2/neural-shrimp-data/audits/EXP-022/closure-001')
CLOSURE_SHA='1b4d929b8e720dcc7ef6f964d100f8b17a14bb416e9279af6892c3578f71bab2'
PLAN_REL='experiments/hopfield/EXP-023-plan.json'
REVIEW_REL='docs/EXP-023-local-review.json'
PROTOCOL_REL='docs/EXP-023-protocol.md'
METHODS=('tight-dop853','tight-rk45')
TANGENT_METHODS=('dop853','rk45')
BRIDGE_IDS=('core-rho025','core-rho050')
SOURCE_PATHS=list(dict.fromkeys([
    'tools/run_matched_cycle_references.py','tools/cycle_reference_selection.py',
    'tools/matched_cycle_seed_bridge.py','tools/matched_cycle_reference_acquisition.py',
    'tools/matched_cycle_reference_evaluation.py','tests/test_cycle_reference_selection.py',
    'tests/test_matched_cycle_seed_bridge.py','tests/test_matched_cycle_reference_acquisition.py',
    'tests/test_matched_cycle_reference_evaluation.py','tests/test_matched_cycle_reference_runner.py',
    'tests/test_matched_cycle_reference_integration.py',PROTOCOL_REL,
    'docs/EXP-022-protocol.md','docs/next-cycle-reference-design.md',*common.SOURCE_PATHS]))
CONTRACT=dict(experiment_id=EXP,exploratory=True,bridge_jobs=2,shooting_jobs=12,reference_jobs=6,
    tangent_jobs=96,total_jobs=116,contexts=48,cycle_slots=6,nested_ivp_slots=578,
    bridge_trial_slots=50,shooting_trial_slots=300,cross_forward_slots=24,adjoint_slots=12,tangent_segment_slots=192,
    workers=4,thread_limit=1,producer_seconds=1800,bridge_seconds=180,shooting_seconds=180,
    reference_seconds=180,tangent_seconds=90,sealing_seconds=180,
    launch_ram_bytes=16*common.GIB,reserve_ram_bytes=8*common.GIB,
    owned_rss_bytes=12*common.GIB,raw_budget_bytes=8*common.GIB,
    required_packages=dict(numpy='2.5.3',scipy='1.18.1'),phase_seed=2026091702,phase_count=8,
    tangent_requires_cycle_and_bin_qualification=True,new_finite_writes=False,new_mixture_decoders=False,
    original_studies_requalified=False,paid_compute=False,paid_review=False)
read=common.read
canonical=common.canonical
binding=common.binding


def _exact(a,b,name):
    if canonical(work.baseline_engine.safe(a))!=canonical(work.baseline_engine.safe(b)):
        raise ValueError(name+' differs')


def historical_inputs():
    """Verify inventory memberships before reading historical candidate metadata."""
    result=[]
    for root,pin,source in ((OLD,OLD_RECEIPT,OLD_SOURCE),(ANCHOR,ANCHOR_RECEIPT,ANCHOR_SOURCE)):
        if legacy.sha(root/'receipt.json')!=pin:raise ValueError('Historical receipt pin differs')
        receipt=read(root/'receipt.json')
        if receipt['source_commit']!=source or receipt['status']!='complete':raise ValueError('Historical source/status differs')
        index={x['path']:x for x in receipt['artifacts']}
        if len(index)!=len(receipt['artifacts']):raise ValueError('Historical duplicate inventory')
        result.append(dict(root=str(root),**binding(root/'receipt.json',root)))
        if root==OLD:
            rels=['manifest.json']+[f'continuation/{b}{suffix}' for b in selection.BRANCHES.values() for suffix in ('.json','-nodes.jsonl','-attempts.jsonl')]
        else:rels=[f'references/payloads/{s}--reference-pair/packet.{ext}' for s in selection.SETTINGS for ext in ('json','npz')]
        for rel in rels:
            current=binding(root/rel,root)
            if rel=='manifest.json':
                if current['sha256']!=receipt['manifest_sha256']:raise ValueError('Historical manifest differs')
            elif current!=index.get(rel):raise ValueError('Historical artifact outside receipt: '+rel)
            result.append(dict(root=str(root),**current))
    if legacy.sha(PREPARATION/'receipt.json')!=PREPARATION_RECEIPT or legacy.sha(PREPARATION/'selection.json')!=SELECTION_SHA:
        raise ValueError('Historical selection preparation differs')
    prep=read(PREPARATION/'receipt.json')
    for item in prep['artifacts']:
        if binding(PREPARATION/item['path'],PREPARATION)!=item:raise ValueError('Selection artifact binding differs')
    if legacy.sha(CLOSURE/'report.json')!=CLOSURE_SHA:raise ValueError('Original no-match closure differs')
    closure=read(CLOSURE/'report.json')
    for item in (closure['ledger'],closure['source']):
        path=Path(item['path'])
        if dict(path=str(path),bytes=path.stat().st_size,sha256=legacy.sha(path))!=item:raise ValueError('No-match closure artifact differs')
    for root,names in ((PREPARATION,('receipt.json','selection.json','input-bindings.json','materialize_selection.py')),
                       (CLOSURE,('report.json','not-launched-ledger.json','close_historical_panel.py'))):
        for name in names:result.append(dict(root=str(root),**binding(root/name,root)))
    return result


def proposals():
    """The two new gamma values are immutable metadata choices, not searches."""
    historical_inputs();old=read(PREPARATION/'selection.json');closure=read(CLOSURE/'report.json')
    if closure['selection']['sha256']!=SELECTION_SHA:raise ValueError('Closure/selection association differs')
    slots=copy.deepcopy(old['slots']);specs=[]
    brackets=closure['prospective_gap_brackets']
    if [b['target'] for b in brackets]!=[.25,.5]:raise ValueError('Fixed bridge targets differ')
    for ident,b in zip(BRIDGE_IDS,brackets,strict=True):
        position=next(i for i,s in enumerate(slots) if s['id']==ident);prior=slots[position]
        if prior['status']!='no_historical_match' or prior['available'] is not False:raise ValueError('Original missing slot was changed')
        a,z=b['left'],b['right'];target=prior['target_rho']
        if z['index']!=a['index']+1 or not a['dominant_rho']<target<z['dominant_rho']:raise ValueError('Unique adjacent bracket required')
        fraction=(target-a['dominant_rho'])/(z['dominant_rho']-a['dominant_rho'])
        gamma=a['gamma']+fraction*(z['gamma']-a['gamma']);near=a if fraction<=.5 else z
        if b['proposed_interpolated_gamma']!=gamma or b['proposed_nearest_endpoint_index']!=near['index']:raise ValueError('Fixed interpolation differs')
        nodes=old['branch_audits'][0]['nodes']
        for endpoint in (a,z):
            original=nodes[endpoint['index']]
            if any(original[k]!=v for k,v in endpoint.items()):raise ValueError('Bracket endpoint differs from original selection audit')
        row=dict(copy.deepcopy(prior),available=True,status='bridge_required',seed_kind='fixed_gamma_proposal',
            node_id=near['node_id'],node_index=near['index'],alpha=near['alpha'],gamma=gamma,
            period=near['period'],first_maximum_state=copy.deepcopy(near['first_maximum_state']),dominant_rho=None,reasons=[])
        row['seed_provenance']=dict(original_unavailable_slot=prior,historical_seed_node_id=near['node_id'],
            parameter_changed_from_gamma=near['gamma'],new_gamma_is_not_historical_node=True,
            fixed_bracket=copy.deepcopy(b),original_selection_sha256=SELECTION_SHA,closure_sha256=CLOSURE_SHA,
            source_binding=copy.deepcopy(prior['seed_provenance']['source_binding']))
        slots[position]=row;specs.append(bridge.build_bridge_spec(row,b))
    return dict(slots=slots,bridge_specs=specs,original_selection_sha256=SELECTION_SHA,
                original_missing_slots=list(BRIDGE_IDS),original_experiment_requalified=False)


def stages(phases):
    bridges=[];shoots=[];refs=[];tangents=[]
    for slot in selection.slot_census():
        c=slot['id']
        if c in BRIDGE_IDS:bridges.append(dict(id=c+'--bridge',kind='bridge',case=dict(id=c,setting=c),method=dict(id='tight-dop853')))
        for m in METHODS:shoots.append(dict(id=c+'--'+m,kind='shoot',case=dict(id=c+'--'+m,setting=c),method=dict(id=m)))
        refs.append(dict(id=c+'--reference-pair',kind='reference',case=dict(id=c,setting=c),method=dict(id='paired')))
        for phase in phases:
            cid=c+f'--p{phase["phase_index"]:02d}'
            for m in TANGENT_METHODS:tangents.append(dict(id=cid+'--reference--'+m,kind='tangent',
                case=dict(id=cid,setting=c,reference_id=c+'--reference-pair',**phase),method=dict(id=m)))
    return dict(bridges=bridges,shooting=shoots,references=refs,tangents=tangents)


def nested_ledger(phases):
    tasks=stages(phases);rows=[]
    for stage in ('bridges','shooting'):
        for t in tasks[stage]:
            for i in range(25):rows.append(dict(id=t['id']+f'--trial-{i:02d}',kind='bridge_trial' if stage=='bridges' else 'shooting_trial',stage=stage,worker_id=t['id'],index=i))
    for t in tasks['references']:
        c=t['case']['setting']
        for candidate in METHODS:
            for m in METHODS:rows.append(dict(id=c+'--verify--'+candidate+'--'+m,kind='cross_forward',stage='references',worker_id=t['id'],key=candidate+'--'+m))
        for m in METHODS:rows.append(dict(id=c+'--adjoint--'+m,kind='adjoint',stage='references',worker_id=t['id'],key=m))
    for t in tasks['tangents']:
        for seg in ('pre','post'):rows.append(dict(id=t['id']+'--'+seg,kind='tangent_segment',stage='tangents',worker_id=t['id'],key=seg))
    return rows


def build_plan():
    proposal=proposals();phases=acquisition.phase_rows();ledger=stages(phases);nested=nested_ledger(phases)
    source=[]
    for rel in SOURCE_PATHS:
        if rel==PROTOCOL_REL and not (ROOT/rel).exists():source.append(dict(path=rel,status='pending_protocol'))
        else:source.append(binding(ROOT/rel))
    return dict(**CONTRACT,date='2026-09-21',proposal=proposal,phases=phases,ledger=ledger,nested_ledger=nested,
        selection_contract=selection.contract(),bridge_contract=bridge.contract(),acquisition_contract=acquisition.contract(),
        evaluation_contract=evaluation.contract(),source_files=source,historical_inputs=historical_inputs(),
        census={k:dict(count=len(v),sha256=hashlib.sha256(canonical(v)).hexdigest()) for k,v in ledger.items()},
        nested_census=dict(count=len(nested),sha256=hashlib.sha256(canonical(nested)).hexdigest()))


def validate(plan):
    if any(x.get('status')=='pending_protocol' for x in plan['source_files']):raise ValueError('Protocol must be bound before validation')
    _exact(plan,build_plan(),'Frozen plan reconstruction')
    led=stages(plan['phases']);nested=nested_ledger(plan['phases'])
    if [len(led[k]) for k in led]!=[2,12,6,96] or len(nested)!=578:raise ValueError('Plan census differs')
    if len({t['id'] for v in led.values() for t in v})!=116 or len({r['id'] for r in nested})!=578:raise ValueError('Duplicate planned identity')
    return dict(total_jobs=116,nested_ivp_slots=578,contexts=48,source_files=len(SOURCE_PATHS),historical_inputs=len(plan['historical_inputs']))


def guard(plan,path,head):
    if not re.fullmatch('[0-9a-f]{40}',head):raise ValueError('Exact expected HEAD required')
    if legacy.git('status','--porcelain','--untracked-files=all'):raise ValueError('Clean checkout required')
    branch=legacy.git('branch','--show-current')
    if legacy.git('rev-parse','HEAD')!=head or not branch.startswith('codex/'):raise ValueError('HEAD/branch differs')
    if Path(path).resolve()!=ROOT/PLAN_REL:raise ValueError('Canonical plan required')
    for rel in [*SOURCE_PATHS,PLAN_REL,REVIEW_REL]:
        b=subprocess.check_output(['git','-C',str(ROOT),'show',f'{head}:{rel}'])
        if b!=(ROOT/rel).read_bytes():raise ValueError('Committed source differs')
    review=read(ROOT/REVIEW_REL)
    if review.get('status')!='ready_for_exploratory_execution' or review.get('plan_sha256')!=legacy.sha(path):raise ValueError('Exact-plan review required')
    if legacy.git('remote','get-url','origin')!='git@github-personal:tdj28/neural-shrimp.git':raise ValueError('Expected private origin required')
    if legacy.git('ls-remote','--exit-code','origin',f'refs/heads/{branch}').split()[0]!=head:raise ValueError('Private push not verified')
    return dict(source_commit=head,pushed_branch=branch,remote_verified_at=legacy.now(),review=binding(ROOT/REVIEW_REL))


def external_attempt(data_root,attempt):
    if not isinstance(attempt,str) or re.fullmatch(r'attempt-\d+',attempt) is None:raise ValueError('Named attempt required')
    root=Path(data_root).resolve()
    if root==ROOT or root.is_relative_to(ROOT):raise ValueError('External evidence required')
    for parent in (root,*root.parents):
        if (parent/'receipt.json').exists() or (re.fullmatch(r'attempt-\d+',parent.name) and parent.parent.name.startswith('EXP-')):
            raise ValueError('Never extend historical evidence')
    root.mkdir(parents=True,exist_ok=True);parent=root/EXP;parent.mkdir(exist_ok=True)
    if parent.resolve()!=parent:raise ValueError('Aliased output forbidden')
    dest=parent/attempt;dest.mkdir();return dest


def resource_check(directory,*,launch=False):
    # Frozen EXP019 uses the same explicit RAM/RSS/raw caps, never changed globally.
    for key in ('launch_ram_bytes','reserve_ram_bytes','owned_rss_bytes','raw_budget_bytes'):
        if CONTRACT[key]!=common.CONTRACT[key]:raise ValueError('Resource helper cap mismatch')
    return common.resource_check(directory,launch=launch)


def inputs(plan):
    """Reconstruct selection from originals before any new bridge dispatch."""
    _exact(plan['historical_inputs'],historical_inputs(),'Historical input binding')
    original=read(PREPARATION/'selection.json');branches={};anchors={};bindings={}
    for s,b in selection.BRANCHES.items():
        branch=read(OLD/'continuation'/f'{b}.json')
        if branch['nodes']!=[json.loads(x) for x in (OLD/'continuation'/f'{b}-nodes.jsonl').read_text().splitlines()]:raise ValueError('Historical nodes journal differs')
        if branch['attempts']!=[json.loads(x) for x in (OLD/'continuation'/f'{b}-attempts.jsonl').read_text().splitlines()]:raise ValueError('Historical attempts journal differs')
        branches[b]=branch
        bindings[b]=dict(experiment_id='EXP-006',source_commit=OLD_SOURCE,receipt_sha256=OLD_RECEIPT,
            artifacts=[binding(OLD/'continuation'/f'{b}{suffix}',OLD) for suffix in ('.json','-nodes.jsonl','-attempts.jsonl')])
        p=ANCHOR/f'references/payloads/{s}--reference-pair/packet.json';anchors[s]=codec.load_packet(p)
        bindings['anchor-'+s]=dict(experiment_id='EXP-018',source_commit=ANCHOR_SOURCE,receipt_sha256=ANCHOR_RECEIPT,
            artifacts=[binding(p.with_suffix('.'+ext),ANCHOR) for ext in ('json','npz')])
    reconstructed=selection.select_cycles(branches,anchors,bindings=bindings)
    _exact(original,reconstructed,'Original historical selection')
    _exact(plan['proposal'],proposals(),'Frozen fixed-gamma proposal')
    return dict(proposal=plan['proposal'],original_selection=reconstructed)


def worker(task,context,directory,libraries,parent_pid):
    started=time.monotonic();cpu=time.process_time();directory=Path(directory)
    dest=directory/'payloads'/task['id'];dest.mkdir()
    row=dict(id=task['id'],kind=task['kind'],case_id=task['case']['id'],method=task['method']['id'],
        status='failed',qualified=False,pid=os.getpid(),identity=work.identity(os.getpid()),started_at=legacy.now(),rmse=None)
    def cancelled(signum,frame):
        if os.getppid()!=parent_pid:os._exit(125)
        raise TimeoutError('Owned reference worker interrupted')
    for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,cancelled)
    packet=None
    try:
        if task['kind']=='bridge':packet=bridge.acquire_bridge(context['spec'])
        elif task['kind']=='shoot':packet=acquisition.refine_cycle(context['seed'],task['method']['id'])
        elif task['kind']=='reference':packet=acquisition.crossverify_reference(context['seed'],context['candidates'],context['phases'])
        elif task['kind']=='tangent':packet=acquisition.acquire_tangent(context['context'],task['method']['id'])
        else:raise ValueError('Unknown owned task')
        row.update(status='complete' if packet.get('complete') is True else packet['status'],qualified=packet.get('qualified') is True)
    except BaseException as exc:
        row['error']=repr(exc)
        if hasattr(exc,'payload'):packet=exc.payload
    if packet is not None:
        try:
            seal=codec.save_packet(dest,'packet',packet)
            row.update(packet_path=str(Path(seal['path']).relative_to(directory)),packet_sha256=seal['sha256'],packet_semantic_sha256=seal['semantic_sha256'])
        except BaseException as exc:row.update(status='serialization_failed',error=repr(exc))
    row.update(ended_at=legacy.now(),wall_seconds=time.monotonic()-started,cpu_seconds=time.process_time()-cpu,
        max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    work.write(directory/'records'/f'{task["id"]}.json',row)


def packet_record(directory,task,row):
    """A child packet survives parent termination without promoting its owner."""
    expected=dict(id=task['id'],kind=task['kind'],case_id=task['case']['id'],method=task['method']['id'])
    if any(row.get(k)!=v for k,v in expected.items()):raise ValueError('Terminal identity differs')
    source=row
    if source.get('packet_path') is None:
        p=Path(directory)/'records'/f'{task["id"]}.json'
        if not p.exists():
            if row['status']=='complete':raise ValueError('Complete task lacks packet')
            return None
        source=read(p)
        if any(source.get(k)!=v for k,v in expected.items()) or source.get('pid')!=row.get('pid'):raise ValueError('Partial worker identity differs')
        if source.get('packet_path') is None:return None
    relative=f'payloads/{task["id"]}/packet.json';path=Path(directory)/relative
    if source['packet_path']!=relative or path.resolve()!=path:raise ValueError('Packet path differs')
    if legacy.sha(path)!=source['packet_sha256']:raise ValueError('Packet metadata hash differs')
    packet=codec.load_packet(path)
    if codec.packet_digest(packet)!=source['packet_semantic_sha256']:raise ValueError('Packet semantic hash differs')
    if packet.get('id')!=task['id'] or packet.get('kind')!=task['kind']:raise ValueError('Packet identity differs')
    if row['status']=='complete' and (packet.get('complete') is not True or packet.get('status')!='complete'):raise ValueError('Complete row/packet differs')
    return packet


def all_packets(directory,plan,rows):
    result={}
    for stage,specs in stages(plan['phases']).items():
        index={r['id']:r for r in rows[stage]};result[stage]={}
        for t in specs:
            p=packet_record(Path(directory)/stage,t,index[t['id']])
            if p is not None:result[stage][t['id']]=p
    return result


def packet_index(directory,stage,packets):
    return {ident:dict(metadata=binding(Path(directory)/stage/'payloads'/ident/'packet.json',Path(directory)),
        semantic_sha256=codec.packet_digest(p)) for ident,p in packets.items()}


def save_index(directory,stage,packets,name):
    target=Path(directory)/name;expected=packet_index(directory,stage,packets);work.write(target,expected)
    # Reload arrays at the dependency boundary; a cached JSON hash is insufficient.
    actual={ident:codec.load_packet(Path(directory)/v['metadata']['path']) for ident,v in expected.items()}
    _exact(packet_index(directory,stage,actual),read(target),'Reloaded prerequisite index')
    _exact(read(target),expected,'Written prerequisite index')
    return actual


def nested_records(plan,rows,loaded):
    result=[];outer={stage:{r['id']:r for r in values} for stage,values in rows.items()}
    for spec in nested_ledger(plan['phases']):
        packet=loaded[spec['stage']].get(spec['worker_id']);parent=outer[spec['stage']][spec['worker_id']];saved=None
        if packet is not None:
            if spec['kind'] in ('bridge_trial','shooting_trial'):
                trials=packet.get('trials',[])
                if len(trials)!=25 or len({r['id'] for r in trials})!=25:raise ValueError('Trial census differs')
                saved=trials[spec['index']]
            elif spec['kind']=='cross_forward':saved=packet.get('verifications',{}).get(spec['key'])
            elif spec['kind']=='adjoint':saved=packet.get('adjoints',{}).get(spec['key'])
            else:saved=packet.get('ivp_slots',{}).get(spec['key'])
        if saved is not None and saved.get('id')!=spec['id']:raise ValueError('Nested identity differs')
        row=dict(spec,owner_status=parent['status'],status=saved.get('status','unknown') if saved else 'unavailable',
            packet_available=saved is not None,ivp_started=saved.get('ivp_started') if saved else False,
            ivp_completed=saved.get('ivp_completed') if saved else False)
        if saved is None:row['reason']=parent['status']
        else:
            for key in ('complete','qualified','attempted','accepted','reason','ivp_attempted','ivp_returned','error','failed_stage'):
                if key in saved:row[key]=saved[key]
        result.append(row)
    return result
