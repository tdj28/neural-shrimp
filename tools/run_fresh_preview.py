#!/usr/bin/env python3
"""EXP-013: prospectively frozen fresh-origin preview loss versus charged budget.

prepare/validate are result-free. run alone fits final old-data recipes and
acquires fresh dynamics. All new raw outputs are external and never resumed.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import shutil
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
import fresh_preview_runtime as work
import run_hopfield_refinement as legacy
import numpy as np
import fresh_preview_features as fresh
import fresh_preview_evaluation as evaluation
import preview_final_model as final
import preview_shortlist_features as old_features
import preview_shortlist_model as model
import run_preview_shortlist as old
from hopfield_information_design import build_atlas_geometry
from hopfield_drift_metrics import periodic_interpolate
from hopfield_drift_probe import phase_projected_gram

PLAN=ROOT/'experiments/hopfield/EXP-013-plan.json'
OLD_PLAN=ROOT/'experiments/hopfield/EXP-011-plan.json'
CONTRACT=dict(schema_version='EXP-013-fresh-preview-v1',experiment_id='EXP-013',exploratory=True,
    origins=24,exclusion_origins=36,phases=[675,1350],radii=[.00025,.001,.004,.016],candidate_count=18,
    candidates=3456,shams=48,contexts=192,independent_cases=48,native_traces=10512,independent_traces=96,
    short_traces=10608,baseline_calls=48,sensitivity_calls=48,model_recipes=4,inner_folds=96,
    readouts=6144,bootstrap_intervals=30,bootstrap_resamples=20000,bootstrap_seed=20260915,
    backend='native_cpu',workers=4,thread_limit=1,wall_clock_ceiling_seconds=3600,
    prerequisite_task_seconds=600,preview_task_seconds=120,ram_reserve_bytes=8*1024**3,disk_reserve_bytes=2*1024**3,
    numerical=dict(rmse_difference_atol=1e-4,require_nonincreasing_native_difference=True,
                   activation_rms_atol=1e-3,activation_max_atol=1e-2),
    required_packages={'numpy':'2.5.3','scipy':'1.18.1'},methods=work.METHODS,baseline=work.BASELINE,
    paid_compute=False,pro_consultation='not_run_hard_cap_unavailable')
HISTORICAL_HASHES={
    'evidence/EXP-004/attempt-001/receipt.json':'3424b9eaa388e58cb4d8b91b2a4f0136d701fb6928e02d1a2a30ed2e52051c07',
    'evidence/EXP-004/attempt-001/manifest.json':'a300c6043f1084e12e7c701766d14aa537ffdeb14e22f5d4ef510a0ad4d61205',
    'evidence/EXP-004/attempt-001/selected.json':'0a92e4ce1446cee45038e19037131cfe653b28bce4666afb8dd1dbe51332d86f',
    'evidence/EXP-006/attempt-001/manifest.json':'541b1fdbb0c551e89db0bdc47c3a4ee6a9f23d1857774bced4b3874b7e47cc3e',
    'evidence/EXP-007/attempt-001/manifest.json':'573d817836d6c3271e685a45988be660743c104b47d96a5036414db4e6893ac7',
    'evidence/EXP-008/attempt-001/manifest.json':'fb0e546947e0f36b77ff4978f60769db3541fc64330022bd664da22c20c321cd'}
SOURCE_PATHS=list(dict.fromkeys(['tools/run_fresh_preview.py','tools/fresh_preview_runtime.py',
    'tests/test_fresh_preview_runtime.py','tests/test_fresh_preview.py','tests/test_preview_final_model.py',
    'tests/test_fresh_preview_features.py','tests/test_fresh_preview_evaluation.py',
    'docs/EXP-013-protocol.md','docs/fresh-origin-tradeoff-design.md',
    'docs/consults/2026-09-17-strategy-brief.md','docs/consults/2026-09-17-receipt.json',
    'tools/run_hopfield.py','tools/run_hopfield_drift.py',*fresh.SOURCE_PATHS,*evaluation.SOURCE_PATHS,*old.SOURCE_PATHS]))


def read(path):return json.loads(Path(path).read_text())
def seal(value):value['packet_sha256']=model.digest(value);return value

def input_metadata():
    """Read atlas and origin metadata only; no preview/target metric file decoding."""
    bindings={}
    def load(relative):
        p=ROOT/relative;bindings[relative]=legacy.binding(p)
        if bindings[relative]['sha256']!=HISTORICAL_HASHES[relative]:raise ValueError('Historical origin metadata changed')
        return read(p)
    prior={name:load(f'evidence/{name}/attempt-001/manifest.json') for name in ('EXP-006','EXP-007','EXP-008')}
    for name in ('receipt.json','manifest.json','selected.json'):load(f'evidence/EXP-004/attempt-001/{name}')
    receipt=read(ROOT/'evidence/EXP-004/attempt-001/receipt.json');index={x['path']:x for x in receipt['artifacts']}
    rows=[]
    for row in range(181):
        rel=f'evidence/EXP-004/attempt-001/focus/row-{row:03d}.npz';p=ROOT/rel
        # Original filenames are bound by sorted receipt identity below.
        if not p.exists():break
        bindings[rel]=legacy.binding(p)
        if any(bindings[rel][key]!=index[str(p.relative_to(ROOT/'evidence/EXP-004/attempt-001'))][key] for key in ('sha256','bytes')):
            raise ValueError('Old atlas receipt mismatch')
        with np.load(p,allow_pickle=False) as z:rows.append({k:z[k].copy() for k in ('alpha','gamma','period')})
    if len(rows)!=181:raise ValueError('All181 frozen atlas rows required')
    grid={k:np.asarray([r[k] for r in rows]) for k in ('alpha','gamma','period')}
    return grid,prior,[bindings[k] for k in sorted(bindings)]


def build_plan():
    grid,prior,bindings=input_metadata();selection=fresh.select_origins(grid,prior);census=fresh.build_census(selection['anchors'])
    old_plan=read(OLD_PLAN);old.validate(old_plan,check_bindings=True,reconstruct=False)
    return dict(**CONTRACT,date='2026-09-17',selection=selection,census=census,
        old_plan=legacy.binding(OLD_PLAN),origin_source_bindings=bindings,
        source_files=[legacy.binding(ROOT/p) for p in SOURCE_PATHS],
        fixed_policies=evaluation.policy_catalog(),
        claim_boundary='Exploratory75-unit loss versus charged preview budget in24 atlas-selected fresh origins. '
          'No task-quality margin, guaranteed geometry advantage, measured deployment-speedup,300-unit utility,'
          'full shrimp-boundary topology or trained-network generalization claim.',
        data_root=str(legacy.DEFAULT_DATA_ROOT),review_status='requires_exact_committed_local_review')


def validate(plan, *, bindings=False, reconstruct=False):
    if any(plan.get(k)!=v for k,v in CONTRACT.items()):raise ValueError('Frozen EXP013 contract mismatch')
    census=fresh.build_census(plan['selection']['anchors'])
    if census!=plan['census'] or len(census['cases'])!=3456 or len(census['shams'])!=48 or len(census['sentinel_ids'])!=48:
        raise ValueError('Fresh census mismatch')
    if plan['fixed_policies']!=evaluation.policy_catalog():raise ValueError('Fixed policy mismatch')
    if [x['path'] for x in plan['source_files']]!=SOURCE_PATHS:raise ValueError('Source census mismatch')
    if bindings:
        for item in plan['source_files']+plan['origin_source_bindings']+[plan['old_plan']]:
            if legacy.binding(ROOT/item['path'])!=item:raise ValueError('Source/input binding mismatch: '+item['path'])
        old.validate(read(OLD_PLAN),check_bindings=True,reconstruct=False)
    if reconstruct and plan!=build_plan():raise ValueError('Plan reconstruction mismatch')
    return dict(origins=24,candidates=3456,shams=48,short_traces=10608,readouts=6144)


def source_guard(plan,path):
    if legacy.git('status','--porcelain','--untracked-files=all'):raise ValueError('Clean source checkout required')
    source=legacy.git('rev-parse','HEAD');branch=legacy.git('branch','--show-current')
    for rel in [*SOURCE_PATHS,str(Path(path).resolve().relative_to(ROOT))]:
        data=subprocess.check_output(['git','-C',str(ROOT),'show',f'{source}:{rel}'])
        if hashlib.sha256(data).hexdigest()!=legacy.sha(ROOT/rel):raise ValueError('Uncommitted freeze source')
    if legacy.git('ls-remote','--exit-code','origin',f'refs/heads/{branch}').split()[0]!=source:raise ValueError('Exact private source commit must be pushed')
    return dict(source_commit=source,pushed_branch=branch,remote_head_verified_at=legacy.now())


def task(ident,kind,case,method):return dict(id=ident,kind=kind,case=case,method=method)

def stage_specs(plan):
    anchors=plan['selection']['anchors'];c=plan['census'];sentinels=set(c['sentinel_ids'])
    return dict(models=[task('final-old-models','models',dict(id='old24'),dict(id='four-final-recipes'))],
        baselines=[task(a['id'],'baseline',dict(id=a['id']),dict(id='paired-baseline-certificate')) for a in anchors],
        sensitivities=[task(x['id'],'sensitivity',dict(id=x['id']),dict(id='unperturbed-dop853')) for x in c['sensitivity_contexts']],
        previews=[task(f'{x["id"]}--{m["id"]}','preview',x,m) for x in [*c['cases'],*c['shams']]
                  for m in work.METHODS[:3]+(work.METHODS[3:] if x['id'] in sentinels else [])])


def resources(data_directory=None):
    mem={line.split(':')[0]:int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith(('MemAvailable:','MemTotal:'))}
    return dict(at=legacy.now(),available_ram_bytes=mem['MemAvailable'],total_ram_bytes=mem['MemTotal'],load_average=list(os.getloadavg()),
        logical_cpus=os.cpu_count(),disk_free_bytes=shutil.disk_usage(data_directory or ROOT.parent/'neural-shrimp-data').free)


def require_headroom(data_directory=None):
    value=resources(data_directory)
    if value['available_ram_bytes']<CONTRACT['ram_reserve_bytes'] or value['disk_free_bytes']<CONTRACT['disk_reserve_bytes']:
        raise RuntimeError('Frozen shared resource reserve unavailable')
    return value


def load_baselines(directory,records):
    result={}
    for row in records:
        if row['status']=='complete':
            packet=read(directory/row['result_path']);result[row['case_id']]=packet
    return result


def load_sensitivities(directory,records):
    result={}
    for row in records:
        if row['status']=='complete':
            with np.load(directory/row['raw'],allow_pickle=False) as raw:result[row['case_id']]={k:raw[k].copy() for k in raw.files}
            result[row['case_id']].update(read(directory/row['metadata_path']))
    return result


def preview_contexts(plan,bundle,baselines):
    anchors={a['id']:a for a in plan['selection']['anchors']};rows={r['id']:r for r in bundle['rows']};out={}
    for case in [*plan['census']['cases'],*plan['census']['shams']]:
        a=anchors[case['anchor']];cert=baselines[a['id']]['certificate'];phase=case['phase'];initial=np.asarray(cert['orbit_states'])[phase].copy()
        delta=np.zeros(2) if case['id'] not in rows else case['radius']*np.asarray(rows[case['id']]['direction'])
        alpha,gamma=np.asarray([a['alpha'],a['gamma']])+delta
        out[case['id']]=dict(alpha=float(alpha),gamma=float(gamma),initial=initial,delta=delta,reference=np.asarray(cert['orbit_activations'])[:-1],
                             period=cert['period'],epoch=phase*cert['period']/2700)
    return out


class FreshLabels:
    """Two capabilities: only sealed subsets, then full data after every winner."""
    def __init__(self, directory, records, policy, log_path):
        self.directory=Path(directory);self.records={r['id']:r for r in records};self.policy=policy;self.log=Path(log_path)
        self.rows={r['id']:r for r in policy['rows']};self.winners=None;self.observed={};self.full_resolutions=set()
    def _read(self,ids,resolution):
        values={}
        for ident in ids:
            record=self.records[f'{ident}--{resolution}']
            if record['status']!='complete':raise ValueError('Unavailable preview cannot be revealed')
            saved=read(self.directory/'records'/f'{record["id"]}.json')
            if saved!=record or not isinstance(saved['rmse'],(int,float)) or isinstance(saved['rmse'],bool) or not math.isfinite(saved['rmse']) or not 0<=saved['rmse']<=2:
                raise ValueError('Saved preview identity/score differs')
            values[ident]=saved['rmse']
        return values
    def selected(self,context,resolution,policy_id):
        if policy_id=='full18-b18' or self.winners is not None:raise ValueError('Wrong selected-label stage')
        entry=next(c for c in self.policy['contexts'] if c['context']==context)['policies'][resolution][policy_id]
        ids=entry['selected_ids'];key=(tuple(context),resolution,policy_id)
        if entry['status']!='complete' or key in self.observed:raise ValueError('Unavailable or duplicate selected reveal')
        append_event(self.log,'selected_reveal',context=context,resolution=resolution,policy=policy_id,ids=ids,policy_seal_sha256=self.policy['packet_sha256'])
        rows=[self.rows[i] for i in ids];values=self._read(ids,resolution);self.observed[key]=(rows,values)
        return rows,values
    def attach_winners(self,path):
        value=read(path)
        if value.get('packet_sha256')!=model.digest({k:v for k,v in value.items() if k!='packet_sha256'}) or value['policy_seal_sha256']!=self.policy['packet_sha256'] or len(value['winners'])!=5760:
            raise ValueError('Complete immutable winner seal required')
        expected={(tuple(c['context']),r,d['id']) for c in self.policy['contexts'] for r in final.RESOLUTIONS for d in self.policy['catalog'] if d['budget']!=18}
        seen=set()
        for row in value['winners']:
            key=(tuple(row['context']),row['resolution'],row['policy'])
            if key not in expected or key in seen or key not in self.observed:raise ValueError('Winner census/selected access differs')
            if row['winner']!=evaluation.observe_selected(*self.observed[key]):raise ValueError('Winner differs from selected-only reveal')
            seen.add(key)
        if seen!=expected or set(self.observed)!=expected:raise ValueError('Incomplete selected winner census')
        self.winners=value;append_event(self.log,'winner_seal',sha256=legacy.sha(path),packet_sha256=value['packet_sha256'],count=5760)
    def full(self,resolution):
        if self.winners is None:raise ValueError('Full18 reveal before winner seal')
        if resolution not in final.RESOLUTIONS or resolution in self.full_resolutions:raise ValueError('Unplanned or duplicate full reveal')
        self.full_resolutions.add(resolution)
        ids=[r['id'] for r in self.policy['rows']]
        append_event(self.log,'full18_reveal',resolution=resolution,ids=ids,winner_seal_sha256=self.winners['packet_sha256'])
        return self._read(ids,resolution)


def append_event(path,stage,**fields):work.append(path,dict(stage=stage,at=legacy.now(),monotonic_ns=time.monotonic_ns(),**fields))


def qualify(plan,records,directory,contexts, *, replay):
    expected=stage_specs(plan)['previews'];indexed={r['id']:r for r in records}
    if len(indexed)!=len(records) or set(indexed)!={t['id'] for t in expected}:raise ValueError('Terminal preview census differs')
    for task in expected:
        row=indexed[task['id']];method=task['method']
        if row.get('kind')!='preview' or row.get('case_id')!=task['case']['id'] or row.get('method')!=method['id']:
            raise ValueError('Realized preview task identity differs')
        if row['status']=='complete':
            if row.get('raw')!=f'payloads/{task["id"]}/trace.npz':raise ValueError('Realized preview raw path differs')
            solver=row['solver']
            if any(solver.get(k)!=v for k,v in method.items()):raise ValueError('Realized solver differs')
            if method['kind']=='RK4' and solver.get('steps')!=round(75/method['dt']):raise ValueError('Realized native step count differs')
            if method['kind']=='DOP853' and (solver.get('success') is not True or type(solver.get('nfev')) is not int or solver['nfev']<0):raise ValueError('Independent solver success/work invalid')
    sentinels=set(plan['census']['sentinel_ids']);candidate_ids={x['id'] for x in plan['census']['cases']};details=[];replayed=0
    for case in [*plan['census']['cases'],*plan['census']['shams']]:
        rr={m['id']:indexed[f'{case["id"]}--{m["id"]}'] for m in work.METHODS[:3]+(work.METHODS[3:] if case['id'] in sentinels else [])}
        arrays={};context=contexts.get(case['id'])
        for name,row in rr.items():
            if row['status']!='complete':continue
            p=directory/row['raw']
            if legacy.sha(p)!=row['raw_sha256'] or p.stat().st_size!=row['raw_bytes']:raise ValueError('Raw trace hash mismatch')
            with np.load(p,allow_pickle=False) as z:
                if context is None or not all(np.array_equal(z[k],context[k]) for k in ('initial','alpha','gamma','delta')):raise ValueError('Realized trace intervention mismatch')
                if z['times'].dtype!=np.float64 or z['states'].dtype!=np.float64 or not np.array_equal(z['times'],work.TIMES) or z['states'].shape!=(751,3) or not np.isfinite(z['states']).all():raise ValueError('Trace shape/times/nonfinite mismatch')
                arrays[name]=z['states'].copy()
            if replay:
                result=work.score(context,work.TIMES,arrays[name]);legacy.compare_scoring({k:row[k] for k in result},result);replayed+=1
        all_complete=len(arrays)==len(rr);detail=dict(id=case['id'],sham=case['id'] not in candidate_ids,independent=case['id'] in sentinels,
            statuses={k:r['status'] for k,r in rr.items()},all_required_complete=all_complete,all_native_complete=all(k in arrays for k in ('rk4-0.005','rk4-0.0025','rk4-0.00125')),native_scalar_pass=False,native_waveform_pass=False,independent_pass=case['id'] not in sentinels)
        if all(k in arrays for k in ('rk4-0.005','rk4-0.0025','rk4-0.00125')):
            a,b,c=[rr[k]['rmse'] for k in ('rk4-0.005','rk4-0.0025','rk4-0.00125')];d1=abs(a-b);d2=abs(b-c)
            delta=np.tanh(arrays['rk4-0.0025'])-np.tanh(arrays['rk4-0.00125']);rms=float(np.sqrt(np.mean(delta**2)));maximum=float(np.abs(delta).max())
            detail.update(d1=d1,d2=d2,native_scalar_pass=d2<=1e-4 and d2<=d1,native_waveform_pass=rms<=1e-3 and maximum<=1e-2,
                activation_rms_difference=rms,activation_max_difference=maximum)
        panel=[]
        for a,b in ([('rk4-0.00125','dop853-1e-11'),('rk4-0.00125','dop853-1e-12'),('dop853-1e-11','dop853-1e-12')] if case['id'] in sentinels else []):
            if a not in arrays or b not in arrays:panel.append(dict(pair=[a,b],complete=False,passed=False));continue
            delta=np.tanh(arrays[a])-np.tanh(arrays[b]);rms=float(np.sqrt(np.mean(delta**2)));maximum=float(np.abs(delta).max());difference=abs(rr[a]['rmse']-rr[b]['rmse'])
            phase=None;common=None
            if 'rk4-0.00125' in arrays:
                phase=rr['rk4-0.00125']['metrics']['phase_offset_cycles'];fixed=periodic_interpolate(context['reference'],(work.TIMES+context['epoch'])/context['period']+phase)
                common={name:float(np.sqrt(np.mean((np.tanh(arrays[name])-fixed)**2))) for name in (a,b)}
            panel.append(dict(pair=[a,b],complete=True,passed=difference<=1e-4 and rms<=1e-3 and maximum<=1e-2,rmse_difference=difference,
                activation_rms_difference=rms,activation_max_difference=maximum,common_native_finest_phase=phase,common_phase_rmse=common))
        if panel:detail.update(independent_pass=all(x['passed'] for x in panel),independent_comparisons=panel)
        detail['qualified']=all_complete and detail['native_scalar_pass'] and detail['native_waveform_pass'] and detail['independent_pass'];details.append(detail)
    candidates=[x for x in details if not x['sham']];shams=[x for x in details if x['sham']];panel=[x for x in details if x['independent']]
    return dict(details=details,complete=sum(r['status']=='complete' for r in records),required=10608,score_packets_replayed=replayed,
        native_candidates=all(x['all_native_complete'] and x['native_scalar_pass'] and x['native_waveform_pass'] for x in candidates),
        native_shams=all(x['qualified'] for x in shams),independent_panel=all(x['independent_pass'] for x in panel),
        qualified_candidates=sum(x['qualified'] for x in candidates),qualified_shams=sum(x['qualified'] for x in shams))


def planned_rows(plan):
    return [{k:c[k] for k in ('id','role','anchor','group','phase','radius','candidate')} for c in plan['census']['cases']]


def fill_terminal(specs, existing, reason):
    indexed={r['id']:r for r in existing}
    if len(indexed)!=len(existing) or set(indexed)-{t['id'] for t in specs}:raise ValueError('Invalid terminal task census')
    return [indexed.get(t['id'],dict(id=t['id'],kind=t['kind'],case_id=t['case']['id'],method=t['method']['id'],status=reason,rmse=None)) for t in specs]


def chronology(directory,policy):
    """Audit recorded capability/chronology, not an OS-level unlogged-access claim."""
    events=[json.loads(x) for x in (directory/'events.jsonl').read_text().splitlines()]
    labels=[json.loads(x) for x in (directory/'label-access.jsonl').read_text().splitlines()]
    by_stage={name:[e for e in events if e['stage']==name] for name in ('models_sealed','fresh_dynamics_started','policies_sealed','previews_started')}
    if any(len(v)!=1 for v in by_stage.values()):raise ValueError('Incomplete stage chronology')
    ordered=[by_stage[k][0]['monotonic_ns'] for k in by_stage]
    if ordered!=sorted(ordered) or len(set(ordered))!=4:raise ValueError('Stage seal chronology changed')
    selected=[e for e in labels if e['stage']=='selected_reveal'];winners=[e for e in labels if e['stage']=='winner_seal'];full=[e for e in labels if e['stage']=='full18_reveal']
    if len(selected)!=5760 or len(winners)!=1 or len(full)!=2 or {e['resolution'] for e in full}!=set(final.RESOLUTIONS):
        raise ValueError('Incomplete reveal event census')
    expected={(tuple(c['context']),r,d['id']):c['policies'][r][d['id']]['selected_ids'] for c in policy['contexts'] for r in final.RESOLUTIONS for d in policy['catalog'] if d['budget']!=18}
    seen=set()
    for e in selected:
        key=(tuple(e['context']),e['resolution'],e['policy'])
        if key in seen or e['ids']!=expected[key] or e['policy_seal_sha256']!=policy['packet_sha256']:raise ValueError('Selected access identity differs')
        if not by_stage['policies_sealed'][0]['monotonic_ns']<e['monotonic_ns']<winners[0]['monotonic_ns']:raise ValueError('Selected reveal outside seal order')
        seen.add(key)
    all_ids=[r['id'] for r in policy['rows']]
    if any(e['monotonic_ns']<=winners[0]['monotonic_ns'] or e['ids']!=all_ids for e in full):raise ValueError('Full18 reveal preceded complete winners')
    model_path=directory/by_stage['models_sealed'][0]['path'];policy_path=directory/'policy-seal.json';winner_path=directory/'winner-seal.json'
    if legacy.sha(model_path)!=by_stage['models_sealed'][0]['sha256'] or legacy.sha(policy_path)!=by_stage['policies_sealed'][0]['sha256'] or legacy.sha(winner_path)!=winners[0]['sha256']:
        raise ValueError('Chronology artifact substitution')
    for key,path in [('features_sha256','fresh-features.npz'),('feature_metadata_sha256','fresh-features.json'),('predictions_sha256','predictions.json')]:
        if legacy.sha(directory/path)!=by_stage['policies_sealed'][0][key]:raise ValueError('Feature/prediction chronology substitution')
    if any(e['winner_seal_sha256']!=read(winner_path)['packet_sha256'] for e in full):raise ValueError('Full reveal winner binding differs')
    training=[json.loads(x) for x in (directory/'models/payloads/final-old-models/access.jsonl').read_text().splitlines()]
    if not training or any(e['monotonic_ns']>=by_stage['models_sealed'][0]['monotonic_ns'] for e in training):raise ValueError('Old training access after model seal')
    return dict(passed=True,selected_reveals=5760,full_reveals=2,models_before_fresh_dynamics=True,policies_before_previews=True,
        winners_before_full18=True,scope='Recorded immutable artifacts and reviewed capability paths; not an OS proof of all unlogged file access.')


def numerical_labels(plan,records,qualification):
    indexed={r['id']:r for r in records};flags={x['id']:x for x in qualification['details']}
    return {resolution:{c['id']:dict(status=indexed[f'{c["id"]}--{resolution}']['status'],
        rmse=indexed[f'{c["id"]}--{resolution}'].get('rmse'),qualified=flags[c['id']]['qualified'],
        numerical=flags[c['id']]) for c in plan['census']['cases']} for resolution in final.RESOLUTIONS}


def costs(plan,records,stage_times,policy=None):
    indexed={r['id']:r for r in records};accounting={}
    if policy is not None:
        for resolution in final.RESOLUTIONS:
            accounting[resolution]={}
            for definition in policy['catalog']:
                ids=[i for c in policy['contexts'] for i in (c['policies'][resolution][definition['id']]['selected_ids'] or [])]
                rows=[indexed[f'{i}--{resolution}'] for i in ids]
                complete=len(ids)==192*definition['budget'] and all(r['status']=='complete' for r in rows)
                accounting[resolution][definition['id']]=dict(required_preview_calls=192*definition['budget'],
                    observed_selected_complete_calls=sum(r['status']=='complete' for r in rows),
                    selected_trace_task_seconds=sum(r.get('wall_seconds',0) for r in rows) if complete else None,
                    selected_integration_seconds=sum(r.get('integration_seconds',0) for r in rows) if complete else None,
                    selected_score_seconds=sum(r.get('score_seconds',0) for r in rows) if complete else None,
                    needs_sensitivity=definition['needs_sensitivity'],needs_atlas=definition['needs_atlas'],
                    needs_old_training=definition['kind']=='learned')
    return dict(stage_wall_seconds=stage_times,selected_task_accounting=accounting,amortization_decisions=[1,8,192],
        acquisition=dict(required_short_traces=10608,complete_short_traces=sum(r['status']=='complete' for r in records),
                         planned_baseline_native_calls=48,planned_sensitivity_calls=48,planned_old_model_recipes=4),
        historical_assets=dict(old_training_plan=plan['old_plan'],old_atlas_bound_inputs=len(plan['origin_source_bindings']),
            current_host_replacement_cost_seconds=None,creation_cost_is_zero=False),
        scope='Stage wall times are actual measured pipeline stages. Selected task sums are charged accounting on full acquisition, '
          'not measured controller latency. Worker startup/scheduling is included only in stage wall. Reference and sensitivity '
          'are shared prerequisites; historical atlas and training labels were reused, with replacement cost unknown. No net speedup claim.')


def amortization(plan,stages,policy):
    groups={a['id']:a['group'] for a in plan['selection']['anchors']}
    candidate_ids={r['id'] for r in plan['census']['cases']}
    previews={res:{r['case_id']:r.get('wall_seconds') if r['status']=='complete' else None
        for r in stages['previews'] if r['method']==res and r['case_id'] in candidate_ids} for res in final.RESOLUTIONS}
    references={groups[r['case_id']]:r.get('wall_seconds') if r['status']=='complete' else None for r in stages['baselines']}
    context_ids={c['id']:(c['group'],c['phase']) for c in plan['census']['sensitivity_contexts']}
    sensitivities={context_ids[r['case_id']]:r.get('wall_seconds') if r['status']=='complete' else None for r in stages['sensitivities']}
    return evaluation.amortized_costs(policy,preview_task_seconds=previews,reference_task_seconds=references,
        sensitivity_task_seconds=sensitivities,final_recipe_seconds=None,joint_setup_seconds=None)


def compact_amortization(packet):
    return {**{k:v for k,v in packet.items() if k!='accounting'},'accounting':{
        res:{policy:{h:{k:v for k,v in data.items() if k!='accounting_blocks'} for h,data in horizons.items()}
        for policy,horizons in policies.items()} for res,policies in packet['accounting'].items()}}


def external_attempt(data_root,attempt):
    root=Path(data_root).expanduser().resolve()
    if root.is_relative_to(ROOT) or not attempt or Path(attempt).name!=attempt or attempt in ('.','..'):raise ValueError('Exclusive external data root and simple attempt name required')
    parent=root/'EXP-013';parent.mkdir(parents=True,exist_ok=True)
    if parent.is_symlink() or (parent/attempt).exists() or (parent/attempt).is_symlink():raise ValueError('Attempt exists or symlinked')
    dest=parent/attempt
    if dest.resolve().is_relative_to(ROOT):raise ValueError('Raw path escaped external storage')
    dest.mkdir();return dest


def run(plan_path,data_root,attempt):
    plan=read(plan_path);validate(plan,bindings=True,reconstruct=True);provenance=source_guard(plan,plan_path)
    review_path=ROOT/'docs/EXP-013-local-review.json';review=read(review_path)
    if review.get('status')!='ready_for_exploratory_execution' or review.get('plan_sha256')!=legacy.sha(plan_path):raise ValueError('Exact plan local review required')
    committed=subprocess.check_output(['git','-C',str(ROOT),'show',f'{provenance["source_commit"]}:docs/EXP-013-local-review.json'])
    if hashlib.sha256(committed).hexdigest()!=legacy.sha(review_path):raise ValueError('Uncommitted local review')
    for name,version in CONTRACT['required_packages'].items():
        if importlib.metadata.version(name)!=version:raise ValueError('Unqualified numerical environment')
    directory=external_attempt(data_root,attempt);start=time.monotonic();deadline=start+3600
    manifest=dict(**provenance,plan_sha256=legacy.sha(plan_path),review=legacy.binding(review_path),plan=plan,started_at=legacy.now(),pid=os.getpid(),
        identity=work.identity(os.getpid()),executable=sys.executable,argv=list(sys.argv),working_directory=str(Path.cwd()),python=sys.version,platform=platform.platform(),packages={k:importlib.metadata.version(k) for k in CONTRACT['required_packages']},
        thread_env={k:os.environ.get(k) for k in legacy.THREAD_ENV},resources=resources(directory))
    work.write(directory/'manifest.json',manifest);specs=stage_specs(plan);stages={};stage_times={};error=None;policy=None;bundle=None;contexts={};numerical=None;readout=None;baselines={};sensitivities={};models=None;access=None
    gates={k:False for k in evaluation.QUALIFICATION_KEYS};stage=None
    old_handlers={s:signal.getsignal(s) for s in (signal.SIGTERM,signal.SIGINT,signal.SIGALRM)}
    def stop(signum,frame):raise TimeoutError('Frozen whole producer deadline or external interruption')
    for s in old_handlers:signal.signal(s,stop)
    signal.setitimer(signal.ITIMER_REAL,3600.)
    def execute_stage(name,contexts_for_stage,libraries,per_task):
        nonlocal stage
        stage=name;headroom=require_headroom(directory);append_event(directory/'events.jsonl','stage_started',name=name,resources=headroom)
        t=time.monotonic()
        try:stages[name]=work.execute(specs[name],contexts_for_stage,directory/name,libraries,workers=4,deadline=deadline,per_task=per_task)
        finally:stage_times[name]=time.monotonic()-t
        append_event(directory/'events.jsonl','stage_finished',name=name,status_counts=dict(Counter(r['status'] for r in stages[name])))
        return stages[name]
    try:
        require_headroom(directory);t=time.monotonic();libraries=work.build_libraries(directory/'build',deadline);stage_times['build']=time.monotonic()-t
        old_plan=read(OLD_PLAN);groups=sorted({r['group'] for r in old_plan['rows']})
        model_records=execute_stage('models',{'old24':dict(old_plan=old_plan,groups=groups)},libraries,600)
        gates['models']=all(r['status']=='complete' and r.get('qualified') is True for r in model_records)
        if not gates['models']:raise RuntimeError('Old final model qualification failed; fresh dynamics blocked')
        models_path=directory/'models'/model_records[0]['models_path'];models=read(models_path)
        append_event(directory/'events.jsonl','models_sealed',path=str(models_path.relative_to(directory)),sha256=legacy.sha(models_path),packet_sha256=models['packet_sha256'])
        append_event(directory/'events.jsonl','fresh_dynamics_started')
        baseline_records=execute_stage('baselines',{a['id']:dict(anchor=a) for a in plan['selection']['anchors']},libraries,600)
        baselines=load_baselines(directory/'baselines',baseline_records);gates['references']=all(r['status']=='complete' and r.get('qualified') is True for r in baseline_records)
        if not gates['references']:raise RuntimeError('Full cohort reference qualification failed; dependent stages blocked without replacing origins')
        sens_contexts={}
        for a in plan['selection']['anchors']:
            cert=baselines[a['id']]['certificate']
            for phase in plan['phases']:sens_contexts[f'{a["id"]}-p{phase}']=dict(anchor=a,phase=phase,initial=cert['orbit_states'][phase],epoch=phase*cert['period']/2700)
        sensitivity_records=execute_stage('sensitivities',sens_contexts,libraries,600)
        gates['sensitivities']=all(r['status']=='complete' and r.get('qualified') is True for r in sensitivity_records)
        if not gates['sensitivities']:raise RuntimeError('Full cohort sensitivity acquisition failed; preview stage blocked')
        sensitivities=load_sensitivities(directory/'sensitivities',sensitivity_records);t=time.monotonic()
        grid,_,_=input_metadata();atlas=build_atlas_geometry(grid,[53,60])
        bundle=fresh.build_fresh_features(plan['selection']['anchors'],atlas,{k:v['certificate'] for k,v in baselines.items()},sensitivities)
        work.write(directory/'fresh-features.json',{k:v for k,v in bundle.items() if k not in ('shared','geometry')})
        legacy.atomic_npz(directory/'fresh-features.npz',shared=bundle['shared'],geometry=bundle['geometry'])
        predictions={}
        for recipe,packet in models['models'].items():
            x,names=old.matrix(bundle,packet['feature_schema']);predictions[recipe]=final.predict_prepreview(packet,x,bundle['rows'],names)
        work.write(directory/'predictions.json',predictions)
        policy=evaluation.seal_policies(bundle['rows'],predictions,origin_groups=[a['group'] for a in plan['selection']['anchors']]);work.write(directory/'policy-seal.json',policy)
        stage_times['fresh_features_predictions_sealing']=time.monotonic()-t
        append_event(directory/'events.jsonl','policies_sealed',sha256=legacy.sha(directory/'policy-seal.json'),packet_sha256=policy['packet_sha256'],
            features_sha256=legacy.sha(directory/'fresh-features.npz'),feature_metadata_sha256=legacy.sha(directory/'fresh-features.json'),predictions_sha256=legacy.sha(directory/'predictions.json'))
        gates['models']=gates['models'] and policy['complete'] and policy['numerical_passed']
        if not gates['models']:raise RuntimeError('Fresh prediction solver qualification failed; preview stage blocked')
        contexts=preview_contexts(plan,bundle,baselines);append_event(directory/'events.jsonl','previews_started')
        preview_records=execute_stage('previews',contexts,libraries,120)
        # This label-owning operator retains all terminal evidence. The chooser's
        # interface exposes only the frozen subset until its winners are sealed.
        if all(r['status']=='complete' for r in preview_records):
            t=time.monotonic();store=FreshLabels(directory/'previews',preview_records,policy,directory/'label-access.jsonl');winners=[]
            for context in policy['contexts']:
                for resolution in final.RESOLUTIONS:
                    for definition in policy['catalog']:
                        if definition['budget']==18:continue
                        rows,values=store.selected(context['context'],resolution,definition['id'])
                        winner=evaluation.observe_selected(rows,values)
                        winners.append(dict(context=context['context'],resolution=resolution,policy=definition['id'],winner=winner))
            winner_packet=seal(dict(policy_seal_sha256=policy['packet_sha256'],winners=winners));work.write(directory/'winner-seal.json',winner_packet)
            store.attach_winners(directory/'winner-seal.json')
            for resolution in final.RESOLUTIONS:store.full(resolution)
            access=chronology(directory,policy);gates['access_chronology']=access['passed'];work.write(directory/'access-audit.json',access)
            stage_times['selected_reveal_winner_seal_full_reveal']=time.monotonic()-t
        t=time.monotonic();numerical=qualify(plan,preview_records,directory/'previews',contexts,replay=True);work.write(directory/'numerical-audit.json',numerical)
        gates.update({k:numerical[k] for k in ('native_candidates','native_shams','independent_panel')});stage_times['saved_scoring_and_numerical_qualification']=time.monotonic()-t
        labels=numerical_labels(plan,preview_records,numerical);t=time.monotonic()
        readout=evaluation.evaluate(policy,labels,qualification=gates,winner_seal=read(directory/'winner-seal.json') if (directory/'winner-seal.json').exists() else None)
        work.write(directory/'evaluation.json',readout);stage_times['evaluation_and_bootstrap']=time.monotonic()-t
    except BaseException as exc:
        error=repr(exc);work.write(directory/'producer-error.json',dict(stage=stage,error=error,at=legacy.now(),all_available_raw_retained=True))
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        for s,h in old_handlers.items():signal.signal(s,h)
        for name,spec in specs.items():
            terminal=directory/name/'terminal-records.json'
            if name not in stages and terminal.exists():stages[name]=read(terminal)
            stages[name]=fill_terminal(spec,stages.get(name,[]),'not_started_prerequisite_or_producer_failure')
        if policy is None:
            policy=evaluation.seal_policies(planned_rows(plan),{},origin_groups=[a['group'] for a in plan['selection']['anchors']])
            work.write(directory/'unavailable-policy-ledger.json',policy)
        if readout is None:
            readout=evaluation.evaluate(policy,{},qualification=gates)
            work.write(directory/'evaluation-unavailable.json',readout)
        work.write(directory/'all-terminal-records.json',stages)
        complete=all(r['status']=='complete' for rows in stages.values() for r in rows)
        producer_seconds=time.monotonic()-start
        amortized=amortization(plan,stages,policy);work.write(directory/'amortization.json',amortized)
        cost=costs(plan,stages['previews'],stage_times,policy);cost['amortization']=compact_amortization(amortized);work.write(directory/'cost.json',cost)
        postflight_error=None
        try:validate(plan,bindings=True,reconstruct=False)
        except BaseException as e:postflight_error=repr(e)
        post=resources(directory);work.write(directory/'postflight.json',dict(resources=post,error=postflight_error,source_bindings_unchanged=postflight_error is None))
        summary=dict(experiment_id='EXP-013',complete=complete,qualified=bool(readout['qualified']) and error is None and postflight_error is None,
            error=error,postflight_error=postflight_error,status_counts={name:dict(Counter(r['status'] for r in rr)) for name,rr in stages.items()},
            qualification=gates,native_qualified_candidates=None if numerical is None else numerical['qualified_candidates'],
            native_qualified_shams=None if numerical is None else numerical['qualified_shams'],
            cohort_statistics=readout['cohort_statistics'],producer_and_analysis_seconds=producer_seconds,ended_at=legacy.now(),cost=cost,
            scope=plan['claim_boundary'])
        work.write(directory/'summary.json',summary);seal_start=time.monotonic();artifacts=work.inventory(directory)
        receipt=dict(**provenance,experiment_id='EXP-013',plan_sha256=legacy.sha(plan_path),directory=str(directory),
            status='complete_qualified' if summary['qualified'] else 'complete_unqualified' if complete else 'incomplete',
            artifacts=artifacts,artifact_bytes=sum(x['bytes'] for x in artifacts),inventory_sealed_at=legacy.now(),sealing_seconds=time.monotonic()-seal_start)
        work.write(directory/'receipt.json',receipt)
    print(json.dumps({k:summary[k] for k in ('complete','qualified','error','status_counts','qualification','producer_and_analysis_seconds')},indent=2),flush=True)
    return summary


def verify(directory, *, replay_models=True):
    """Replay saved scoring/features/models/choices; never integrate fresh dynamics."""
    directory=Path(directory).resolve();receipt=read(directory/'receipt.json');manifest=read(directory/'manifest.json');plan=manifest['plan']
    if receipt['plan_sha256']!=manifest['plan_sha256'] or receipt['source_commit']!=manifest['source_commit']:raise ValueError('Manifest/source identity mismatch')
    original=subprocess.check_output(['git','-C',str(ROOT),'show',f'{receipt["source_commit"]}:experiments/hopfield/EXP-013-plan.json'])
    if hashlib.sha256(original).hexdigest()!=receipt['plan_sha256'] or json.loads(original)!=plan:raise ValueError('Frozen committed plan differs')
    validate(plan,bindings=True,reconstruct=False)
    current=work.inventory(directory)
    if current!=receipt['artifacts']:raise ValueError('Complete artifact inventory/hash differs')
    summary=read(directory/'summary.json');terminal=read(directory/'all-terminal-records.json');specs=stage_specs(plan)
    for stage,expected in specs.items():
        rr=terminal[stage]
        if [r['id'] for r in rr]!=[t['id'] for t in expected] or len({r['id'] for r in rr})!=len(expected):raise ValueError('Stage terminal census/order mismatch')
        for task,row in zip(expected,rr,strict=True):
            if any(row.get(k)!=task[k] for k in ('id','kind')) or row['case_id']!=task['case']['id'] or row['method']!=task['method']['id']:raise ValueError('Terminal task identity differs')
            p=directory/stage/'records'/f'{row["id"]}.json'
            if row['status']=='complete' and (not p.exists() or read(p)!=row):raise ValueError('Missing/substituted completed worker record')
        stage_path=directory/stage
        if stage_path.exists():
            cleanup=read(stage_path/'cleanup.json')
            if cleanup['all_original_identities_absent'] is not True or cleanup['all_registered_workers_joined'] is not True:raise ValueError('Recorded owned cleanup incomplete')
            process_path=stage_path/'processes.jsonl'
            processes=[json.loads(x) for x in process_path.read_text().splitlines()] if process_path.exists() else []
            process_index={x['task_id']:x for x in processes}
            if len(process_index)!=len(processes) or set(process_index)-{t['id'] for t in expected}:raise ValueError('Registered process task census differs')
            if len(cleanup['workers'])!=len(processes):raise ValueError('Cleanup process census differs')
            for owned,check in zip(processes,cleanup['workers'],strict=True):
                if any(check[k]!=v for k,v in owned.items()) or check['original_identity_absent'] is not True:raise ValueError('Cleanup worker identity differs')
            saved=read(stage_path/'terminal-records.json')
            if {x['id']:x for x in saved}!={x['id']:x for x in rr}:raise ValueError('Stage terminal record differs')
            for row in rr:
                if row['status']=='complete' and (row['id'] not in process_index or row['pid']!=process_index[row['id']]['pid']):raise ValueError('Completed worker missing owned process')
        elif any(r['status']!='not_started_prerequisite_or_producer_failure' for r in rr):raise ValueError('Executed stage missing cleanup')
    if summary['status_counts']!={stage:dict(Counter(r['status'] for r in rr)) for stage,rr in terminal.items()}:raise ValueError('Summary status census mismatch')
    baselines=load_baselines(directory/'baselines',terminal['baselines']);sensitivities=load_sensitivities(directory/'sensitivities',terminal['sensitivities'])
    anchors={a['id']:a for a in plan['selection']['anchors']};baseline_gates=[]
    for row in terminal['baselines']:
        if row['status']!='complete':baseline_gates.append(False);continue
        if row['result_path']!=f'payloads/{row["id"]}/baselines/{row["id"]}/result.json':raise ValueError('Baseline result identity differs')
        result=baselines[row['id']];anchor=anchors[row['id']]
        path=(directory/'baselines'/row['result_path']).parent
        steps=[]
        for dt in (.01,.005):
            with np.load(path/f'dt-{dt:g}.npz',allow_pickle=False) as z:
                n=int(z['n_events']);q,residual,T=work.baseline_engine.classify(z['events'][:n],z['stats'],work.BASELINE)
                if not np.array_equal(z['initial'],work.BASELINE['initial_state']) or not np.isfinite(z['final']).all() or not 0<=n<=128:raise ValueError('Baseline raw contract differs')
                if q!=int(z['period']) or not np.array_equal(residual,z['residual'],equal_nan=True) or not np.array_equal(T,z['period_time'],equal_nan=True):raise ValueError('Baseline classification replay differs')
                steps.append(work.baseline_engine.safe(dict(dt=dt,period=q,residual=residual,period_time=T,stats=z['stats'])))
        if steps!=result['steps']:raise ValueError('Baseline summary replay differs')
        gate=all(s['period']==9 for s in steps) and work.certificate_ok(result['certificate'],anchor)
        if row['qualified']!=gate:raise ValueError('Reference qualification differs')
        baseline_gates.append(gate)
    model_packet=None;model_replayed=False
    record=terminal['models'][0]
    if record['status']=='complete':
        if record['models_path']!='payloads/final-old-models/models.json':raise ValueError('Final model path differs')
        model_packet=read(directory/'models'/record['models_path'])
        if model_packet['packet_sha256']!=model.digest({k:v for k,v in model_packet.items() if k!='packet_sha256'}):raise ValueError('Model packet seal mismatch')
        old_plan=read(OLD_PLAN);old_bundle=old_features.load_features();groups=sorted({r['group'] for r in old_bundle['rows']})
        old_store=old.LabelStore(old_bundle['rows'],old_plan['label_records'],allowed_training_groups=groups)
        ids=[r['id'] for r in old_bundle['rows']];labels={r:old_store.training(ids,r) for r in final.RESOLUTIONS}
        logged=[json.loads(x) for x in (directory/'models'/'payloads'/'final-old-models'/'access.jsonl').read_text().splitlines()]
        if [{k:v for k,v in e.items() if k!='monotonic_ns'} for e in logged]!=[{k:v for k,v in e.items() if k!='monotonic_ns'} for e in old_store.events]:raise ValueError('Training label access census differs')
        if replay_models:
            replay=final.fit_four_recipes(old_bundle,labels,old_origin_groups=groups);legacy.compare_scoring(model_packet,replay);model_replayed=True
    sensitivity_specs={c['id']:c for c in plan['census']['sensitivity_contexts']}
    for row in terminal['sensitivities']:
        if row['status']!='complete':continue
        ident=row['case_id'];spec=sensitivity_specs[ident];anchor=anchors[spec['anchor']];phase=spec['phase']
        cert=baselines[spec['anchor']]['certificate'];value=sensitivities[ident]
        if row['raw']!=f'payloads/{ident}/sensitivity.npz' or row['metadata_path']!=f'payloads/{ident}/sensitivity.json':raise ValueError('Sensitivity artifact path differs')
        raw=directory/'sensitivities'/row['raw']
        if legacy.sha(raw)!=row['raw_sha256'] or raw.stat().st_size!=row['raw_bytes']:raise ValueError('Sensitivity raw hash differs')
        expected=dict(initial=np.asarray(cert['orbit_states'])[phase],alpha=anchor['alpha'],gamma=anchor['gamma'],phase=phase,epoch=phase*cert['period']/2700)
        if any(not np.array_equal(value[k],v) for k,v in expected.items()):raise ValueError('Sensitivity physical identity differs')
        for key in ('times','states','state_sensitivity','activation_sensitivity','activation_velocity','tail_direction'):
            if value[key].dtype!=np.float64:raise ValueError('Sensitivity precision differs')
        _,orbit_states=fresh._certificate(anchor,cert)
        sensitivity,velocity,direction=fresh._sensitivity(anchor,phase,orbit_states,value)
        if not np.array_equal(phase_projected_gram(sensitivity,velocity)['minimum_eigenvector'],direction):raise ValueError('Saved tail direction differs')
        full_gram=phase_projected_gram(value['activation_sensitivity'],value['activation_velocity'])
        legacy.compare_scoring(work.baseline_engine.safe({k:value[k] for k in full_gram}),work.baseline_engine.safe(full_gram))
        if row['qualified'] is not True:raise ValueError('Completed sensitivity qualification differs')
    fresh_bundle=None;contexts={};policies_path=directory/'policy-seal.json'
    if policies_path.exists():
        grid,_,_=input_metadata();atlas=build_atlas_geometry(grid,[53,60])
        fresh_bundle=fresh.build_fresh_features(plan['selection']['anchors'],atlas,{k:v['certificate'] for k,v in baselines.items()},sensitivities)
        with np.load(directory/'fresh-features.npz',allow_pickle=False) as z:
            if not np.array_equal(z['shared'],fresh_bundle['shared']) or not np.array_equal(z['geometry'],fresh_bundle['geometry']):raise ValueError('Fresh feature replay differs')
        if {k:v for k,v in fresh_bundle.items() if k not in ('shared','geometry')}!=read(directory/'fresh-features.json'):raise ValueError('Fresh feature metadata differs')
        predictions={}
        for recipe,packet in model_packet['models'].items():
            x,names=old.matrix(fresh_bundle,packet['feature_schema']);predictions[recipe]=final.predict_prepreview(packet,x,fresh_bundle['rows'],names)
        legacy.compare_scoring(read(directory/'predictions.json'),predictions)
        policy=evaluation.seal_policies(fresh_bundle['rows'],predictions,origin_groups=[a['group'] for a in plan['selection']['anchors']])
        if policy!=read(policies_path):raise ValueError('Pre-preview policy replay differs')
        contexts=preview_contexts(plan,fresh_bundle,baselines)
    else:
        policy=evaluation.seal_policies(planned_rows(plan),{},origin_groups=[a['group'] for a in plan['selection']['anchors']])
        if policy!=read(directory/'unavailable-policy-ledger.json'):raise ValueError('Blocked-policy ledger differs')
    numerical=qualify(plan,terminal['previews'],directory/'previews',contexts,replay=True)
    if (directory/'numerical-audit.json').exists() and numerical!=read(directory/'numerical-audit.json'):raise ValueError('Numerical audit replay differs')
    qualification=dict(references=all(baseline_gates),sensitivities=all(r['status']=='complete' and r.get('qualified') is True for r in terminal['sensitivities']),
        models=record['status']=='complete' and record.get('qualified') is True and policy['complete'] and policy['numerical_passed'],
        native_candidates=numerical['native_candidates'],native_shams=numerical['native_shams'],independent_panel=numerical['independent_panel'],access_chronology=False)
    winner=None
    if (directory/'winner-seal.json').exists():
        winner=read(directory/'winner-seal.json');access=chronology(directory,policy);qualification['access_chronology']=access['passed']
        if access!=read(directory/'access-audit.json'):raise ValueError('Access audit differs')
    labels=numerical_labels(plan,terminal['previews'],numerical) if (directory/'numerical-audit.json').exists() else {}
    if (directory/'evaluation.json').exists():
        replay=evaluation.evaluate(policy,labels,qualification=qualification,winner_seal=winner);legacy.compare_scoring(read(directory/'evaluation.json'),replay)
    else:
        # Incomplete producer still seals all6144 planned diagnostic rows. Its
        # recorded stage gates show how far it got; numerical false gates are
        # not silently upgraded by post-run analysis.
        replay=evaluation.evaluate(policy,{},qualification=summary['qualification']);legacy.compare_scoring(read(directory/'evaluation-unavailable.json'),replay)
    if replay['cohort_statistics']!=summary['cohort_statistics']:raise ValueError('Summary cohort statistics differ')
    qualified=bool(replay['qualified']) and summary['error'] is None and summary['postflight_error'] is None
    complete=all(r['status']=='complete' for rr in terminal.values() for r in rr)
    if qualified!=summary['qualified'] or complete!=summary['complete']:raise ValueError('Summary terminal conclusion differs')
    stored_cost=read(directory/'cost.json');expected_cost=costs(plan,terminal['previews'],stored_cost['stage_wall_seconds'],policy)
    amortized=amortization(plan,terminal,policy)
    if amortized!=read(directory/'amortization.json'):raise ValueError('Amortization reconstruction differs')
    expected_cost['amortization']=compact_amortization(amortized)
    if stored_cost!=expected_cost or stored_cost!=summary['cost']:raise ValueError('Cost reconstruction differs')
    return dict(status='verified_saved_fresh_preview_evidence',complete=complete,qualified=qualified,source_commit=receipt['source_commit'],
        receipt_sha256=legacy.sha(directory/'receipt.json'),artifacts=len(current),score_packets_replayed=numerical['score_packets_replayed'],
        model_recipes_replayed=4 if model_replayed else 0,cohort_statistics_reproduced=True,
        scope='Same frozen implementation replay; independent audits remain a separate requirement.')


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    for name in ('prepare','validate','run'):
        p=sub.add_parser(name);p.add_argument('--plan',type=Path,default=PLAN)
        if name=='run':p.add_argument('--data-root',type=Path,default=legacy.DEFAULT_DATA_ROOT);p.add_argument('--attempt',default='attempt-001')
    p=sub.add_parser('verify');p.add_argument('directory',type=Path);p.add_argument('--audit-output',type=Path,required=True)
    args=parser.parse_args()
    if args.command=='prepare':
        plan=build_plan();validate(plan,reconstruct=False);work.write(args.plan,plan);print(json.dumps(validate(plan),indent=2))
    elif args.command=='validate':print(json.dumps(validate(read(args.plan),bindings=True,reconstruct=True),indent=2))
    elif args.command=='run':run(args.plan,args.data_root,args.attempt)
    else:
        output=args.audit_output.resolve()
        if output.is_relative_to(ROOT) or output.is_relative_to(args.directory.resolve()):raise ValueError('New external audit directory required')
        output.mkdir(parents=True,exist_ok=False);started=time.monotonic()
        try:result=verify(args.directory);work.write(output/'verification.json',dict(**result,elapsed_seconds=time.monotonic()-started,verified_at=legacy.now()))
        except BaseException as e:
            work.write(output/'failed-verification.json',dict(error=repr(e),elapsed_seconds=time.monotonic()-started));raise
        print(json.dumps(result,indent=2))


if __name__=='__main__':main()
