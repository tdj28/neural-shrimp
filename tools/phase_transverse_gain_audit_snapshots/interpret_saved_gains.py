#!/usr/bin/env python3
"""Saved-only EXP015 arithmetic/interpretation; no gain/ODE/fit recomputation."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
from collections import Counter
import hashlib,json,datetime,sys,math,itertools
import numpy as np
ROOT=Path('/data2/neural-shrimp');RAW=Path('/data2/neural-shrimp-data/raw/EXP-015/attempt-001');OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'tools'))
import run_phase_memory_recovery as codec

def read(p):return json.loads(Path(p).read_text())
def bind(p):
 p=Path(p);return dict(path=str(p),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
checks=[]
def check(name,value):
 checks.append(dict(name=name,passed=bool(value)))
 if not value:raise AssertionError(name)
receipt=read(RAW/'receipt.json');summary=read(RAW/'summary.json');manifest=read(RAW/'manifest.json');rows=read(RAW/'terminal-records.json');index={r['path']:r for r in receipt['artifacts']}
check('receipt_pinned',bind(RAW/'receipt.json')['sha256']=='502a2613b7bac2704a7f93030d0450d7ae7a501c8c9dd8c8a47114dfac30c986')
check('source_and_plan',receipt['source_commit']==manifest['source_commit']=='e62b1dac4291c8aca288b723c70aea85259800d1' and receipt['plan_sha256']==manifest['plan_sha256']=='6bc005d773af7aa60d3f8c2e7b85a97db6ddb70909815bbd787ea64fa54716e1')
for name in ('summary.json','manifest.json','terminal-records.json','postflight.json'):
 b=bind(RAW/name);check('bound:'+name,index[name]==dict(path=name,bytes=b['bytes'],sha256=b['sha256']))
check('complete_sourcevalid',summary['complete'] is True and summary['source_valid'] is True and summary['error'] is None and summary['postflight_error'] is None)
check('full192_census',len(rows)==192 and [r['cell'] for r in rows]==manifest['plan']['cells_census'])
packets=[];failedtests=[];cellcompact=[]
for row in rows:
 ident=row['cell']['id'];p=RAW/row['packet_path'];check('packet:'+ident,bind(p)['sha256']==row['packet_sha256'])
 r=codec.load_packet(p);packets.append(r);check('cell/status:'+ident,r['cell']==row['cell'] and r['complete'] is True and r['status']=='complete' and len(r['compatibility'])==6)
 check('flags:'+ident,all(r[k]==row[k] for k in ('qualified','qualified_algebraic','qualified_ideal_neutral')))
 for key,value in r['unit']['algebraic_checks'].items():
  if not value['passed']:
   failure=dict(cell_id=ident,kind='unit_algebra',test=key,values=value)
   if key=='state_partition':
    phase=r['unit']['state']['phase_algebraic'];trans=r['unit']['state']['transverse'];total=r['unit']['state']['total'];res=phase+trans-total;tol=1e-14+1e-12*np.abs(total);inds=np.argwhere(np.abs(res)>tol)
    failure['failed_entries']=[dict(index=i.tolist(),phase=float(phase[tuple(i)]),transverse=float(trans[tuple(i)]),total=float(total[tuple(i)]),residual=float(res[tuple(i)]),tolerance=float(tol[tuple(i)]),ratio=float(abs(res[tuple(i)])/tol[tuple(i)])) for i in inds]
   failedtests.append(failure)
 for c in r['compatibility']:
  check('compat_complete:'+ident+':'+c['id'],c['status']=='complete' and c['complete'] is True)
  for key,v in c['checks'].items():check('old_compatibility:'+ident+':'+c['id']+':'+key,v['passed'] is True)
  for key,v in c['mathematics']['algebraic_checks'].items():
   if not v['passed']:failedtests.append(dict(cell_id=ident,pulse_id=c['id'],kind='pulse_algebra',test=key,values=v))
  for key,v in c['approximation'].items():
   expected=v['state_max']<=1e-6 and v['activation_max']<=1e-7 and v['primary_standardized_readout']<=.01
   check('approximation_decision:'+ident+':'+c['id']+':'+key,v['passed']==expected)
   if not v['passed']:failedtests.append(dict(cell_id=ident,pulse_id=c['id'],kind='ideal_approximation',test=key,values=v,state_failed=v['state_max']>1e-6,activation_failed=v['activation_max']>1e-7,readout_failed=v['primary_standardized_readout']>.01))
 expected_alg=r['gates']['input_identity'] and r['gates']['algebraic'] and r['gates']['old_linear_compatibility'];expected_ideal=expected_alg and r['gates']['normalization'] and r['gates']['ideal_approximation']
 check('qualification_logic:'+ident,r['qualified_algebraic']==expected_alg and r['qualified_ideal_neutral']==expected_ideal)
 cellcompact.append(dict(cell=r['cell'],qualified_algebraic=r['qualified_algebraic'],qualified_ideal_neutral=r['qualified_ideal_neutral'],gates=r['gates'],observable=r['unit']['observable'],reference_defects=r['unit']['reference_defects'],state_gain=r['unit']['state_gain'],activation_gain=r['unit']['activation_gain'],unit_noise_rows=r['unit']['noise_rows']))
check('all_summary_counts',sum(r['qualified_algebraic'] for r in packets)==summary['algebraically_qualified_cells']==191 and sum(r['qualified_ideal_neutral'] for r in packets)==summary['ideal_neutral_qualified_cells']==179 and summary['realized_pulse_rows']==dict(required=1152,complete=1152,failed=0,unavailable=0))
strata=[];horizons=[]
for setting,direction in itertools.product(('core','failed-near-edge'),('phase_write','phase_null','neuron_one')):
 selected=[r for r in packets if r['cell']['setting']==setting and r['cell']['direction']==direction]
 for delay in (1,2,4,8):
  group=[r for r in selected if r['cell']['delay_cycles']==delay]
  ranges={name:[min(r['unit']['observable'][name] for r in group),max(r['unit']['observable'][name] for r in group)] for name in ('total','phase_algebraic','transverse','initial_time_shift')}
  extra={}
  if direction=='phase_write':
   ratios=[abs(r['unit']['observable']['transverse']/r['unit']['observable']['phase_algebraic']) for r in group]
   extra=dict(transverse_to_algebraic_phase_magnitude_range=[min(ratios),max(ratios)],negative_total_count=sum(r['unit']['observable']['total']<0 for r in group),negative_transverse_count=sum(r['unit']['observable']['transverse']<0 for r in group))
  strata.append(dict(setting=setting,direction=direction,delay_cycles=delay,required_cells=8,qualified_algebraic=sum(r['qualified_algebraic'] for r in group),qualified_ideal_neutral=sum(r['qualified_ideal_neutral'] for r in group),signed_gain_ranges=ranges,**extra))
 for phase in range(8):
  group=sorted((r for r in selected if r['cell']['phase_index']==phase),key=lambda r:r['cell']['delay_cycles'])
  vals=[r['unit']['observable']['transverse'] for r in group]
  horizons.append(dict(setting=setting,phase_index=phase,direction=direction,delays=[1,2,4,8],signed_transverse_gains=vals,algebraic_flags=[r['qualified_algebraic'] for r in group],ideal_flags=[r['qualified_ideal_neutral'] for r in group],absolute_nondecreasing_steps=[(1,2,4,8)[i:i+2] for i in range(3) if abs(vals[i+1])>=abs(vals[i])]))
selected=next(r for r in packets if r['cell']['id']=='failed-near-edge--p1--phase_null--k4')
selected_pulses=[dict(id=r['id'],observable=r['mathematics']['observable'],primary_noise=r['noise_rows'][1],original_finite_phase_only_q=r['original_finite_phase_only_q'],compatibility_checks=r['checks']) for r in selected['compatibility'][:2]]
# The probability below is solely a check against the already reported original
# linear prediction, not a new BA outcome/cell or a new analysis qualification.
zplus=selected_pulses[1]['primary_noise']['signed_standardized']['total'];zminus=selected_pulses[0]['primary_noise']['signed_standardized']['total'];old_linear_probability=.25*(2+math.erf(zplus/math.sqrt(2))+math.erf(-zminus/math.sqrt(2)))
failed_approx=[r for r in failedtests if r['kind']=='ideal_approximation'];badcells=[r for r in cellcompact if not r['qualified_ideal_neutral']]
report=dict(schema_version='EXP015-saved-scientific-interpretation-v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scope='Saved reference-gain outcomes and stored arithmetic diagnostics only; no new ODE, fit, reference-gain solve or finite-pulse outcome analysis. Full independent raw arithmetic audit is separate.',script=bind(__file__),source_commit=receipt['source_commit'],plan_sha256=receipt['plan_sha256'],receipt=bind(RAW/'receipt.json'),raw_summary=bind(RAW/'summary.json'),artifact_count=len(receipt['artifacts']),artifact_bytes=receipt['artifact_bytes'],checks=len(checks),failures=0,
 counts=dict(complete_cells=192,complete_realized_compatibility_rows=1152,algebraically_qualified_cells=191,ideal_neutral_qualified_cells=179,failed_overall_cells=13,algebra_failed_cells=1,ideal_approximation_failed_cells=12),failed_cells=badcells,exact_failed_tests=failedtests,
 failure_interpretation=dict(all_old_linear_compatibility_checks_pass=True,unit_state_partition_failed_entries=sum(len(r.get('failed_entries',[])) for r in failedtests),ideal_failure_atomic_tests=len(failed_approx),ideal_failures_only_largest_dose=all('--d2--' in r['pulse_id'] for r in failed_approx),ideal_failures_only_transport_or_combined=all(r['test'] in ('transport_defect','combined_ideal_clock_defect') for r in failed_approx),standardized_readout_approximation_failures=sum(r['readout_failed'] for r in failed_approx),clock_flow_failures=sum(r['test']=='clock_flow_defect' for r in failed_approx),maximum_approximation_standardized_q=max(v['primary_standardized_readout'] for r in packets for c in r['compatibility'] for v in c['approximation'].values()),maximum_normalization_residual=max(r['unit']['reference_defects']['normalization'] for r in packets)),
 stratum_ranges_all_cells=strata,horizon_sequences_all48=horizons,
 phase_null=dict(required=64,algebra_qualified=sum(r['qualified_algebraic'] for r in packets if r['cell']['direction']=='phase_null'),ideal_qualified=sum(r['qualified_ideal_neutral'] for r in packets if r['cell']['direction']=='phase_null'),maximum_absolute_unit_phase_gain=max(abs(r['unit']['observable']['phase_algebraic']) for r in packets if r['cell']['direction']=='phase_null'),scope='Nonzero transverse readout with nearzero timing projection; null direction was constructed from the adjoint, so nearzero phase is by design.'),
 selected_previously_reported_example=dict(cell=selected['cell'],gates=selected['gates'],unit_observable=selected['unit']['observable'],smallest_realized_signed_pulses=selected_pulses,compatibility_with_original_full_linear_probability=old_linear_probability,scope='Same previously reported example, selected before this analysis; not the best new cell. Algebraic phase is zero for the unit direction and only rounding-scale for realized pulses. Agreement explains the sealed linear prediction by ordinary transverse response, conditional on inherited propagators; it does not requalify the original setting or prove new nonlinear predictive utility.'),
 conclusion='Conditional linear mechanism is explicit: adjoint-phase-null inputs can drive the frozen clock readout through transverse dynamics, and phase-writing inputs also contain substantial oblique transverse components. Phase-writing transverse contribution opposes the phase part at every phase/window in both selected settings. Core transverse readout decays rapidly across the four windows; near-edge transverse response persists, with transient growth at some early windows. These are contrasts of two selected systems, not a geometry effect.',
 recommendation='The next most decisive bounded step is a separately frozen comparison of these sealed linear/neutral/transverse predictions against the already acquired numerically consistent smaller-dose pulse census: all2settings×8phases×3directions×2smaller doses×2signs=192 physical pulses plus16shams, with all3original methods retained (624 trajectories), four fixed clock windows and the same three noise scales. No new ODE or fitted decoder is needed. Preserve every case, the one algebraic attribution failure, requested/realized rounding, and all old large-dose failures; show signed/odd/even residuals and error-versus-dose without pooled setting means, posthoc decoder changes or utility acceptance threshold. This tests whether the generic linear mechanism explains actual dynamics beyond algebraic self-consistency. A later independent reference/variational refinement is required before claiming globally accurate transverse operators.',
 caution=['All ranges include the full stated eight-phase strata and retain their failed qualification counts; they are not qualified means.','The exact algebraic split is valid as an identity; one finite-precision strict arithmetic check fails and remains failed. No tolerance adjustment or normalized projector repair.','Twelve ideal-neutral approximation failures reflect inherited physical-flow transport defects at the largest declared displacement, not twelve newly observed pulse failures.','Noise-standardized q approximation passes do not waive failed state/activation approximation gates.','Unit phase-writing direction Z/||Z|| is not generally tangent f0/||f0||; transverse response is not unexpected contamination or novel topology mechanism.','No eigenvalue-modulus-only causal explanation, trained-network claim, memory-capacity theorem, original setting average or shrimp utility claim is established.'])
(OUT/'all192-cell-readouts.json').write_text(json.dumps(cellcompact,indent=2)+'\n');report['all_cells']=bind(OUT/'all192-cell-readouts.json')
(OUT/'saved-checks.json').write_text(json.dumps(checks,indent=2)+'\n');report['check_details']=bind(OUT/'saved-checks.json')
f=OUT/'EXP-015-scientific-interpretation.json';f.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:report[k] for k in ('counts','failure_interpretation','phase_null')},indent=2));print(json.dumps(bind(f),indent=2))
