"""Prepare held Task37 JSONs from reviewed local bytes; never launches a job."""
from pathlib import Path
import hashlib
import json

HERE=Path(__file__).resolve().parent
REPO=Path('/home/d7082791602/.codex/worktrees/llm-hostile-fold-crosscheck/neural-shrimp')
TARGET=REPO/'codex/presubmit-task37'
RAW=Path('/data2/neural-shrimp-data/scratch/audit-codex-1019')
SNAP=Path('/data2/neural-shrimp-data/models/hf/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775')
def pin(path,alias=None,known_hash=None):
    path=Path(path)
    return {'path':str(alias or path),'bytes':path.stat().st_size,
            'sha256':known_hash or hashlib.sha256(path.read_bytes()).hexdigest()}
def save(name,data): (HERE/name).write_text(json.dumps(data,indent=2)+'\n')
sources=[pin(HERE/n,TARGET/n) for n in ('p6_kite.py','periodic_audit.py')]+[
 pin(REPO/'codex/crosscheck/qwen_map.py'),pin(REPO/'codex/certify-chaos/supervise.py')]
inventory=json.loads((HERE/'p6-seed-inventory.json').read_text())
prior='/data2/neural-shrimp-data/codex/certificate-ladder/phase-cascade-p6-001-supervision/receipt.json'
inputs=[pin(HERE/'p6-seed-inventory.json',TARGET/'p6-seed-inventory.json')]
for row in inventory['seeds']:inputs.extend([row['states'],row['curve']])
inputs.extend([
 pin(SNAP/'model.safetensors',known_hash='fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe'),
 pin(SNAP/'config.json',known_hash='18e18afcaccafade98daf13a54092927904649e1dd4eba8299ab717d5d94ff45'),
 pin(SNAP/'tokenizer.json',known_hash='c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539'),
 pin('/data2/neural-shrimp-data/explore/llm-phase-map/vectors-l12/vectors.npz',known_hash='b630bb7bef06bde87f8e6a7ca5e8672f83d24bc0585bf19048e9e494a4357f41'),pin(prior)])
limits={'as_bytes':18*1024**3,'cpu_seconds':15000,'wall_seconds':15300,'output_bytes':512*1024**2}
card={
 'schema':'codex-task37-p6-r-pair-card-v1','status':'held_for_private_checkpoint_and_coordinated_capacity',
 'target':{'label':'R','alpha':.072912,'log10_temperature':-.154104,'period':6},
 'seed_inventory':str(TARGET/'p6-seed-inventory.json'),'tokenizer':str(SNAP/'tokenizer.json'),
 'source_files':sources,'input_files':inputs,'prior_closed_receipt':prior,'capacity_window':None,
 'operator_limits':{'composed_calls':160,'jvp_columns':1200,'wall_seconds':7200.,'newton_updates':8,
  'newton_jvp_columns':192,'gmres_restart':24,'gmres_cycles':2,'line_trials':8,
  'residual_tolerance':1e-11,'max_seed_displacement':.25},
 'external_limits':limits,'threads':1,'dense_batch_columns':8,
 'maximum_extra_forward_calls':{'prefix':1,'phase_logits_per_seed':6},
 'purpose':'Independent exploratory finite numerical check of two saved P6 seeds at the requested rounded R; no interval certificate or normal-form coefficient.',
 'gates':{'all_six_phase_returns_relative':1e-11,'proper_divisor_relative_separation':1e-7,
  'identity_relative_max_phase_distance':1e-10,'distinct_relative_max_phase_distance':1e-7,
  'dense_scalar_jvp_relative_error':1e-10,'all_dense_eigen_residuals_normalized':1e-10,
  'full_eigenvalues':896,'reported_largest':8,'fd_final_relative_error':1e-5,
  'random_fd_h_squared_interval_required':True,'stability_numerical_margin':1e-7},
 'conditional_followup':'Only if both R orbits qualify and are distinct: prepare separately pinned allocations for the two requested body cells using the actual closed R states; no automatic launch or extension.',
 'cost_basis':{'reference':'codex/certificate-ladder-task34/dense-cascade-p6-001-completion.json',
  'reference_source':'19b63a2002b72ff5bcdaea441a141b58ed05e272','reference_elapsed_seconds_rounded_up':2472.136234,
  'transfer':'Same independent 24-layer/896-dimensional/full-vocabulary float64 CPU F^6 and eight-column batching; new orbit and Newton workload. Timing is a workload reference, not a measured new-family or scalar-JVP forecast.',
  'two_dense_reference_seconds':4944.272468,
  'allocation_rationale':'One serial CPU, at most two 896-column final spectra plus bounded Newton/FD/phase work; 15000CPU/15300wall provides a finite stop with all partial outputs retained. No rate or six-thread speedup promise.'},
 'preparation_checks':'Source/interface review, synthetic numerical controls and focused wrapper controls are recorded in execution-preparation-review.md; no target model execution.',
 'admission_remaining':['save exact reviewed files in owned worktree',
  'ordinary private-main checkpoint and verified fetch equality','coordinated stable memory window covering hard wall',
  'fresh shared capacity and actual owned process census','explicit dated card/plan status amendment'],
 'retry_policy':'No automatic retry, extension, seed replacement, floor weakening, or body-point launch.'}
save('r-pair-001-card.json',card)
plan={'status':card['status'],'repository':str(REPO),'raw_root':str(RAW),
 'python':'/data2/neural-shrimp/.venv/bin/python','cpu_threads':1,'cpu_ids':[0],'startup_ram_gib':32,
 'source_files':sources+[pin(HERE/'r-pair-001-card.json',TARGET/'r-pair-001-card.json')],
 'lanes':{'r_pair':{'entry':str(TARGET/'p6_kite.py'),
  'arguments':['--card',str(TARGET/'r-pair-001-card.json'),'--out',str(RAW/'r-pair-001'),
               '--admission',str(RAW/'r-pair-001-supervision/admission.json')],
  'output':str(RAW/'r-pair-001'),'supervision':str(RAW/'r-pair-001-supervision'),
  'preflight_pins':inputs,'prior_receipts':[prior],'limits':limits}},
 'execution_environment':'HF_HUB_OFFLINE=1 and all numerical threads1; existing external supervisor imposes nice10, CPU affinity and resource/available-memory/disk caps.',
 'capacity_status':'HELD after prior local RAM-floor stops; no admission from a momentary passing RAM reading.',
 'no_target_run':True}
save('r-pair-001-plan.json',plan)
print(json.dumps({'card':pin(HERE/'r-pair-001-card.json'),'plan':pin(HERE/'r-pair-001-plan.json'),
                  'source_pins':len(sources),'input_pins':len(inputs),'target_calls':0},indent=2))
