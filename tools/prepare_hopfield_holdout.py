#!/usr/bin/env python3
"""Compile EXP007 from old atlas/manifest only; no new neural trajectories."""
import json
from pathlib import Path
import run_hopfield as base
from hopfield_holdout_design import load_holdout_packet
from hopfield_drift_design import atlas_margin_direction
from run_hopfield_holdout import PLAN, ARMS, RADII, PHASES, validate
ROOT=Path(__file__).resolve().parents[1]

def compile_plan():
    source=ROOT/'evidence/EXP-004/attempt-001'
    development=ROOT/'evidence/EXP-006/attempt-001/manifest.json'
    packet=load_holdout_packet(source,development)
    grid=base.load_grid(source/'focus');anchors=packet.pop('anchors')
    for anchor in anchors:
        anchor['id']=anchor['anchor']
        anchor['atlas_margin']={str(r):atlas_margin_direction(grid,anchor,r,angle_count=512) for r in RADII}
    # File paths in the holdout packet are recorded separately; bind old inputs here.
    sources=[dict(path=str(path.relative_to(ROOT)),sha256=base.sha(path)) for path in
             [source/'receipt.json',source/'manifest.json',source/'selected.json',development,*sorted((source/'focus').glob('row-*.npz'))]]
    p=dict(experiment_id='EXP-007',date='2026-09-11',status='prospectively frozen private exploratory held-out interventions conditional on a previously observed atlas',
           source_artifacts=sources,atlas_selection=packet,anchors=anchors,arms=ARMS,radii=RADII,phases=PHASES,
           dt=[.01,.005],primary_dt=.005,decision_dt=.005,initial_state=[.1,.1,.1],burn_time=10000,sample_time=2000,
           event_capacity=128,max_peak_period=32,minimum_repeats=4,recurrence_atol=2e-6,recurrence_rtol=2e-5,
           preview_duration=75,probe_duration=300,sample_dt=.1,phase_grid=128,workers=4,wall_clock_ceiling_seconds=1800,
           bootstrap_samples=10000,bootstrap_seed=20260912,hybrid_local_through_radius=.004,
           primary_endpoint='Activation RMSE over samples75<t<=300 of all3neurons relative to original certified waveform; one phase offset fitted only to this scored interval. Fine dt=.005. No frequency fitting or time warp.',
           secondary_endpoints=['Separate full0..300waveform RMSE','Coarse dt=.01 using same fine-preview policy decisions','Zero-dose floor','All candidate preview scores, chosen actions and costs'],
           baseline_rule='Independent common initial x(.1,.1,.1), both steps nine peaks, attracting independent shooting and2701orbit samples. Failed anchors retained without replacement.',
           phase_rule='Exact orbit indices337,1012,1687,2362 of2700endpoint-excluded samples; epochs index*T/2700. Approximately odd eighths, not rounded exact eighths.',
           decision_rule='21candidate entries:16uniformangles with exactaxes,local+/-,tangent+/-,atlas. No deduplication. Each75unit preview starts from same orbit state atdt.005. Minimum preview global-phase RMSE, exact tie lowestindex. All candidate failures retained; any failure invalidates requiring subset.',
           local_rule='Fixed-initial-state sensitivity integrated0..300, phase-projected Gram computed only75<t<=300 in native weight units; two signs selected using75unit previews.',
           policies={'atlas':'old512angle pixel-margin rule; no preview required',
             'local':'preview best ofindices16,17','tangent':'preview best of18,19',
             'axes':'preview best of0,4,8,12','uniform':'preview best ofindices0..15; no atlas orlocal derivatives','search':'preview best ofall21entries; augmented search usesatlas andlocal',
             'hybrid':'local for radii<=.004;atlas for.016; development-informed rule frozen before newtargets'},
           primary_comparisons=['local minus hybrid error','tangent minus hybrid error','axes minus hybrid error','uniform minus hybrid error','search minus hybrid error'],
           aggregation='Equal4phases,4doses andeligibleanchors.10000pairedanchorbootstrappercentile95intervals descriptiveonly. Any requiredmissingcondition invalidates affectedmeans/contrasts; no success-only deletion.',
           cost_rule='Actualpreview21entriescollectedoncepercondition; counterfactualpolicybudgets0atlas,2local,2tangent,4axes,16uniform,21search;hybrid2small0large. Local/search/hybrid require baseline sensitivity. Atlas/tangent/augmentedsearch require oldmap. Report cost separately; no equal-total-budget or net-speedup claim.',
           failure_rules=['Exclusive attempt, no resume/retry/replacements','Complete clean source pushed beforetargets','All failed previews/targets and partial nativepayloads retained','1800secondhardcap includesverification','Sensitivityfailure stopswholeattempt, preservespayload'],
           resource_basis='EXP006 same CPU/sensitivity/native metrics423.87s for4320short+longtrials. EXP007 removeslongtargetassays;4032previews75units+2784targets300units+48sensitivities+24baselinecells. Streamonecondition (<25MBrawarrayworkingmemory);4workers;1800shardceiling.',
           claim_boundary='New anchor waveform intervention test within seen published3neuronatlas; not newnetwork, unseenmap, repeatedcontroller, independenttemporalrandomsample, learnedtask, storagecapacity or paid-resource benefit. Simulatorpreview requires model and cloned initialstate; endpoint excludes preview samples but future remains dynamicallydependent.',
           paid_compute=False,paid_review=False)
    validate(p);return p

if __name__=='__main__':
    p=compile_plan();base.write(PLAN,p);print(json.dumps(dict(plan=str(PLAN),targets=validate(p),previews=4032)))
