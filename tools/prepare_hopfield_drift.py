#!/usr/bin/env python3
"""Compile EXP006 from already retained EXP004/005 evidence; no new integration."""
from dataclasses import asdict
import json
from pathlib import Path
import run_hopfield as base
from hopfield_drift_design import load_anchor_packet, atlas_margin_direction
from hopfield_continuation import ContinuationConfig
from run_hopfield_drift import PLAN, ARMS, RADII, validate

ROOT=Path(__file__).resolve().parents[1]

def compile_plan():
    source=ROOT/'evidence/EXP-004/attempt-001'
    packet=load_anchor_packet(source,count_per_stratum=4,seed=20260911)
    grid=base.load_grid(source/'focus')
    anchors=packet.pop('anchors')
    for a in anchors:
        a['id']=a['anchor']
        a['atlas_margin']={str(r):atlas_margin_direction(grid,a,r,angle_count=512) for r in RADII}
    sources=[{'path':str((source/item['path']).relative_to(ROOT)),'sha256':item['sha256']} for item in packet['source_files']]
    control_summary='evidence/EXP-005/attempt-001/summary.json'
    sources.append(dict(path=control_summary,sha256=base.sha(ROOT/control_summary)))
    config=asdict(ContinuationConfig());config.pop('parameter_minimum');config.pop('parameter_maximum')
    p=dict(experiment_id='EXP-006',date='2026-09-11',status='prospective exploratory utility pilot selected after EXP004/005',
           source_artifacts=sources,atlas_selection=packet,anchors=anchors,arms=ARMS,radii=RADII,phases=[0,1,2,3],
           dt=[.01,.005],primary_dt=.005,initial_state=[.1,.1,.1],burn_time=10000,sample_time=2000,
           event_capacity=128,max_peak_period=32,minimum_repeats=4,recurrence_atol=2e-6,recurrence_rtol=2e-5,
           probe_duration=300,sample_dt=.1,phase_grid=128,workers=4,wall_clock_ceiling_seconds=1800,
           bootstrap_samples=10000,bootstrap_seed=20260911,
           baseline_rule='Both independent common-initialization steps must retain nine peaks and finest independent shooting must verify attraction. Failed anchors retained without replacement; all their planned trial records labeled baseline_not_evaluable.',
           intervention='Raw Frobenius weight norm equal to declared radius; weight changes held throughout each independent short or long trajectory. Eleven arms share anchor, exact initial orbit phase, radius and numerical step. One zero-dose sham per phase and step.',
           geometry_policy='atlas_margin: maximum old-atlas observed pixel margin among 512 fixed unit directions; smallest-angle tie break, explicit inward fallback when all scores zero. Crop-censored scores and every angle score retained.',
           local_policy='Mean of two signs of minimum eigenvector of baseline activation parameter-sensitivity Gram after projecting off one global time-shift vector. Native alpha/gamma units, no atlas. No sign chosen from target outcomes.',
           primary_endpoint='Activation RMSE over all three neurons and 300 physical time units at dt=.005 relative to original certified waveform, allowing one global phase offset only. No frequency fitting, time warping or per-cycle realignment.',
           primary_comparisons=['local_symmetric minus atlas_margin error','random_symmetric minus atlas_margin error','shuffled_inward minus atlas_margin error'],
           aggregation='Equal phases within anchor and equal four radii; eligible anchors equally weighted. 10000 anchor bootstrap draws and percentile95 intervals, descriptive only. Any incomplete required eligible trial makes affected contrasts undefined. No success-only deletion.',
           secondary_endpoints=['all arms/doses including inward normal diagnostic','same-time waveform RMSE and constant phase offset','zero-dose interpolation/integration floor','cadence spectrum diagnostic retained without exclusion','independent dt=.01 sensitivity','long-horizon nine-peak retention, physical period and FTLE'],
           continuation=dict(source_summary=control_summary,seeds=['core','failed-near-edge'],directions=[-1,1],config=config,
             parameterization='fixed alpha; gamma=seed_gamma+0.41*s; gamma bounds[9.9,10.7]; scales[1,1,20], logT and s scale1',
             separation='Continuation never selects drift anchors/actions/outcomes. Keep unstable nodes and every failure; +1/-1 brackets are numerical candidates, not exact boundary or basin-loss proofs.'),
           failure_rules=['Exclusive attempt directory and no rerun/resume','Clean committed pushed complete source before targets','Retain failed baselines, all attempted cells, native partial payloads, continuation correctors and timeout','Hard1800second timer includes all stages; no adaptation to outcomes'],
           resource_basis='Local non-target 300-time native trace .0121s; sensitivity .770s; waveform metric .013s. EXP00439,523cells140.54s; long-run workload roughly eight billion native steps, conservative whole-attempt30minutehardcap.',
           cost_scope='Report actual wall, local-sensitivity cost and old atlas140.54seconds separately. No zero-cost atlas, net speedup, or paid-resource claim.',
           claim_boundary='Does this frozen atlas policy reduce error for an explicitly defined small recurrent-network waveform task beyond local parameter sensitivity? No generic geometry benefit, exact full shrimp boundary topology, modern training/reasoning benefit or population inference from12anchorattempts.',
           paid_compute=False,paid_review=False)
    assert validate(p)==4320
    return p

if __name__=='__main__':
    p=compile_plan();base.write(PLAN,p)
    print(json.dumps(dict(plan=str(PLAN),anchor_attempts=len(p['anchors']),trial_step_calls=validate(p))))
