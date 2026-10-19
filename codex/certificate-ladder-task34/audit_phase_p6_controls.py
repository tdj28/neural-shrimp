"""New P6 delta/interface controls; manufactured evidence only, no target reads.

Source/card/plan reads are permitted. All original raw, Git, and host process
interfaces are mocked. This is neither an actual saved-evidence audit nor a
numerical/model qualification, and it repeats no prior control suite.
"""
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

SOURCE = Path('/tmp/audit_phase_p6.py')
EXPECTED = '61c8a466235eedbfcc9454d5965e8e999a46a88a79552c3ab7e799f7f2aaa92e'
assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == EXPECTED
spec = importlib.util.spec_from_file_location('phase_p6_subject', SOURCE)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
import numpy as np

def forbid_target_access(event, args):
    if event == 'open' and isinstance(args[0], (str, bytes, os.PathLike)):
        path = os.fsdecode(args[0])
        assert not path.startswith('/data2/neural-shrimp-data/'), ('forbidden actual raw/model read', path)
        assert not path.startswith('/proc/'), ('forbidden actual process read', path)
    assert event not in {'subprocess.Popen', 'os.system', 'os.posix_spawn'}, ('forbidden subprocess', event)
sys.addaudithook(forbid_target_access)

card = json.loads(m.CARD.read_text())
plan = json.loads(m.PLAN.read_text())
assert hashlib.sha256(m.CARD.read_bytes()).hexdigest() == m.CARD_SHA
assert hashlib.sha256(m.PLAN.read_bytes()).hexdigest() == m.PLAN_SHA
plan_pin = dict(path=str(m.PLAN), bytes=len(m.PLAN.read_bytes()), sha256=m.PLAN_SHA)
lane = plan['lanes']['phase']
argv = [plan['python'], '-I', '-B', lane['entry'], *lane['arguments']]
receipt = dict(argv=argv, lane='phase', source_commit=m.SOURCE, returncode=0, failure=None,
               worker_reaped=True, worker_identity_absent=True, worker_group_session_absent=True,
               worker_pid=10001, external_elapsed_seconds=12.0,
               output_files=[dict(path=p, bytes=1, sha256='0'*64) for p in sorted(m.expected_outputs())])
admission = dict(argv=argv, lane='phase', source_commit=m.SOURCE, supervisor=dict(pid=10002),
                 plan=plan_pin, limits=lane['limits'], threads={n:'1' for n in m.THREADS},
                 source_blobs=[dict(relative_path=str(Path(e['path']).relative_to(m.R)), bytes=e['bytes'],
                                    sha256=e['sha256']) for e in plan['source_files']])
cell = m.expected_cell(card)
row = dict(cell=cell, period=6, certificate_passed=False, product_order=m.PRODUCT_ORDER,
           phases=[dict(phase=i, seconds=0.0, relative_l2_closure=0.0) for i in range(6)])
history = np.ones((513, 2), dtype=np.float64)
history[-6:] = np.array([[i+1., i+2.] for i in range(6)])
matrices = np.array([[[.25, (i+1)/64], [0., .5]] for i in range(6)])
product = np.eye(2)
for a in matrices: product = a @ product
inverse = np.eye(2)  # Supplied proposal; no new inverse solve.
denominator = np.eye(2)-product
left = np.eye(2)-inverse@denominator
right = np.eye(2)-denominator@inverse
arrays = dict(A=matrices, B=inverse, centers=history[-6:].copy(),
              parameters=np.array([cell['alpha'], cell['log10T']]), product=product,
              left_inverse_defect=left, right_inverse_defect=right)
phase_files = [(matrices[i].copy(), dict(center=history[-6+i].copy(),
               image=history[-6+(i+1)%6].copy(), residual=np.zeros(2))) for i in range(6)]
row['diagnostics'] = dict(left_inverse_defect_inf=float(np.linalg.norm(left, ord=np.inf)),
    right_inverse_defect_inf=float(np.linalg.norm(right, ord=np.inf)),
    product_norm_inf=float(np.linalg.norm(product, ord=np.inf)),
    inverse_norm_inf=float(np.linalg.norm(inverse, ord=np.inf)),
    condition_2_estimate=float(np.linalg.cond(denominator)))
results=[]

def check(name, thunk, refusal=None):
    try:
        value=thunk()
    except Exception as error:
        assert refusal is not None, (name, type(error).__name__, str(error))
        assert refusal in str(error), (name, refusal, type(error).__name__, str(error))
        results.append(dict(name=name, outcome='expected_refusal', error_type=type(error).__name__,
                            message=str(error)))
    else:
        assert refusal is None, (name, 'accepted forbidden fixture')
        results.append(dict(name=name, outcome='pass'))
        return value

def authority_change(change):
    values=copy.deepcopy([receipt, admission, plan, card, plan_pin])
    change(*values)
    return m.allocation(*values)

def row_change(change):
    value=copy.deepcopy(row); change(value)
    return m.row_authority(dict(rows=[value]), cell)

def array_change(change):
    values=copy.deepcopy([row, history, arrays, phase_files]); change(*values)
    return m.check_saved_arrays(*values, dimension=2)

check('original_shared_card_named_p6_and_exact_14_files', lambda:m.allocation(receipt,admission,plan,card,plan_pin))
check('reject_p3_source', lambda:authority_change(lambda r,a,p,c,q:r.update(source_commit='32c7e4126cd88c6e0c2030dcf1d287f356ee2db8')), 'wrong original P6 execution source')
check('reject_other_period6_command', lambda:authority_change(lambda r,a,p,c,q:r['argv'].__setitem__(7,'band-p6')), 'wrong exact original phase command')
check('reject_duplicate_named_p6_cell', lambda:authority_change(lambda r,a,p,c,q:c['cells'].append(copy.deepcopy(cell))), 'duplicate named')
check('reject_period3_cell', lambda:authority_change(lambda r,a,p,c,q:next(x for x in c['cells'] if x['name']==m.CELL).update(period=3)), 'cell/period')
check('reject_other_period6_history', lambda:authority_change(lambda r,a,p,c,q:next(x for x in c['cells'] if x['name']==m.CELL).update(states=str(m.B/'settle-001/band-p6-states.npy'))), 'history/seed')
check('reject_wrong_exact_cell_ratio', lambda:authority_change(lambda r,a,p,c,q:next(x for x in c['cells'] if x['name']==m.CELL).update(alpha_ratio=[1,16])), 'binary64 cell ratios')
check('reject_missing_phase5_matrix', lambda:authority_change(lambda r,a,p,c,q:r.update(output_files=[e for e in r['output_files'] if not e['path'].endswith('phase-5-matrix.npy')])), 'six-phase original output inventory')
check('reject_missing_phase5_residual', lambda:authority_change(lambda r,a,p,c,q:r.update(output_files=[e for e in r['output_files'] if not e['path'].endswith('phase-5-residual.npz')])), 'six-phase original output inventory')
check('reject_extra_phase6_file', lambda:authority_change(lambda r,a,p,c,q:r['output_files'].append(dict(path=str(m.RAW/m.CELL/'phase-6-matrix.npy'),bytes=1,sha256='0'*64))), 'six-phase original output inventory')
check('reject_duplicate_output', lambda:authority_change(lambda r,a,p,c,q:r['output_files'].append(r['output_files'][0])), 'duplicate original file paths')
check('reject_source_inventory_omission', lambda:authority_change(lambda r,a,p,c,q:a['source_blobs'].pop()), 'complete source inventory')
def remove_required_source(r,a,p,c,q):
    name='codex/certificate-ladder-task34/phase_preconditioner.py'
    p['source_files']=[e for e in p['source_files'] if e['path']!=str(m.R/name)]
    a['source_blobs']=[e for e in a['source_blobs'] if e['relative_path']!=name]
check('reject_consistently_omitted_required_source', lambda:authority_change(remove_required_source), 'scientific source/card')
check('reject_preflight_original_history_omission', lambda:authority_change(lambda r,a,p,c,q:p['lanes']['phase'].update(preflight_pins=[e for e in p['lanes']['phase']['preflight_pins'] if e['path']!=cell['states']])), 'inputs missing')
check('reject_original_closure_flag_false', lambda:authority_change(lambda r,a,p,c,q:r.update(worker_identity_absent=False)), 'closure flags')
check('reject_original_worker_pid_missing', lambda:authority_change(lambda r,a,p,c,q:r.update(worker_pid=None)), 'process identifiers')
check('reject_original_nonzero_returncode', lambda:authority_change(lambda r,a,p,c,q:r.update(returncode=1)), 'did not succeed')
check('reject_original_plan_pin_change', lambda:authority_change(lambda r,a,p,c,q:a['plan'].update(sha256='1'*64)), 'plan pin differs')
check('six_phase_report_order',lambda:m.row_authority(dict(rows=[row]),cell))
check('reject_missing_phase5_row',lambda:row_change(lambda r:r['phases'].pop()),'all six')
check('reject_reordered_phase_rows',lambda:row_change(lambda r:r['phases'].reverse()),'all six')
check('reject_duplicate_phase5_row',lambda:row_change(lambda r:r['phases'].__setitem__(5,copy.deepcopy(r['phases'][4]))),'all six')
check('reject_row_foreign_cell',lambda:row_change(lambda r:r['cell'].update(name='band-p6')),'named cascade-P6 cell')
check('reject_row_wrong_product_order',lambda:row_change(lambda r:r.update(product_order='A[0] @ ... @ A[5]')),'scope/order')
check('reject_row_certificate_claim',lambda:row_change(lambda r:r.update(certificate_passed=True)),'scope/order')
checked=check('tiny_six_phase_seven_array_adapter_with_phase5_wrap',lambda:m.check_saved_arrays(row,history,arrays,phase_files,dimension=2))
assert checked['phases_checked']==6 and checked['dimension']==2 and checked['certificate_passed'] is False
check('reject_last_matrix_pair_disagrees',lambda:array_change(lambda r,h,a,p:p[5][0].__setitem__((0,0),.75)),'stacked A')
check('reject_five_matrix_residual_pairs',lambda:array_change(lambda r,h,a,p:p.pop()),'all six')
check('reject_last_residual_field_omitted',lambda:array_change(lambda r,h,a,p:p[5][1].pop('residual')),'residual fields')
check('reject_last_residual_wraps_to_wrong_center',lambda:array_change(lambda r,h,a,p:p[5][1].update(image=a['centers'][5].copy())),'centre/image/residual')
check('reject_three_phase_stacked_A',lambda:array_change(lambda r,h,a,p:a.update(A=a['A'][:3])),'binary64 A')
check('reject_wrong_original_last_six_centers',lambda:array_change(lambda r,h,a,p:a['centers'].__setitem__((5,0),20.)),'last six')
check('reject_nonbinary64_proposal',lambda:array_change(lambda r,h,a,p:a.update(B=a['B'].astype(np.float32))),'binary64 B')
check('reject_missing_seventh_array',lambda:array_change(lambda r,h,a,p:a.pop('right_inverse_defect')),'seven original')
check('reject_eighth_array',lambda:array_change(lambda r,h,a,p:a.update(unexpected=np.ones(2))),'seven original')
check('reject_history_not_513_rows',lambda:m.check_saved_arrays(row,history[:512],arrays,phase_files,dimension=2),'binary64 original history')
check('reject_other_period6_parameters',lambda:array_change(lambda r,h,a,p:a['parameters'].__setitem__(0,.07)),'exact named cell')
check('mocked_original_host_closed',lambda:m.original_host_closed(receipt,admission,lambda p:None,lambda p:[]))
check('reject_mocked_original_supervisor_alive',lambda:m.original_host_closed(receipt,admission,lambda p:dict(pid=p) if p==10002 else None,lambda p:[]),'closure pending')
check('reject_mocked_original_worker_group_alive',lambda:m.original_host_closed(receipt,admission,lambda p:None,lambda p:[p] if p==10001 else []),'closure pending')

args_list=['--attempt',m.ATTEMPT,'--source',m.SOURCE,'--cell',m.CELL,
           '--receipt-sha256','2'*64,'--admission-sha256','3'*64,'--output',str(m.RESULT)]
args=check('explicit_future_original_receipt_pins',lambda:m.arguments(args_list))
bad_args=args_list.copy();bad_args[5]='band-p6'
check('reject_other_named_period6_argument',lambda:m.arguments(bad_args),'producer allocation')
bad_args=args_list.copy();bad_args[-1]=str(m.RAW/'result.json')
check('reject_output_inside_original_raw',lambda:m.arguments(bad_args),'separate named P6')

# Entire future loader/control flow is exercised with in-memory raw arrays,
# synthetic receipts, mocked file/Git/host interfaces, and source-only blobs.
# No actual raw path is opened, statted, resolved, listed, or loaded.
source_data={e['relative_path']:(m.R/e['relative_path']).read_bytes() for e in admission['source_blobs']}
assert all(hashlib.sha256(source_data[e['relative_path']]).hexdigest()==e['sha256'] for e in admission['source_blobs'])
pins={e['path']:e for e in [*receipt['output_files'],*card['inputs'],*plan['source_files'],plan_pin]}
pins.update({str(m.CARD):dict(path=str(m.CARD),bytes=len(m.CARD.read_bytes()),sha256=m.CARD_SHA),
 str(m.SUP/'receipt.json'):dict(path=str(m.SUP/'receipt.json'),bytes=1,sha256='2'*64),
 str(m.SUP/'admission.json'):dict(path=str(m.SUP/'admission.json'),bytes=1,sha256='3'*64)})
reads={str(m.CARD):card,str(m.PLAN):plan,str(m.SUP/'receipt.json'):receipt,
       str(m.SUP/'admission.json'):admission,str(m.RAW/'report.json'):dict(rows=[row])}
loads={cell['states']:history,str(m.RAW/m.CELL/'preconditioner.npz'):arrays}
for i,(matrix,values) in enumerate(phase_files):
    loads[str(m.RAW/m.CELL/f'phase-{i}-matrix.npy')]=matrix
    loads[str(m.RAW/m.CELL/f'phase-{i}-residual.npz')]=values
class Archive:
    def __init__(self,values):self.values=values;self.files=list(values)
    def __getitem__(self,key):return self.values[key]
    def __enter__(self):return self
    def __exit__(self,*args):return False
def load_mock(path,allow_pickle):
    assert allow_pickle is False
    v=loads[str(path)]
    return Archive(v) if isinstance(v,dict) else v
listed=[]
for value in receipt['output_files']:
    p=MagicMock();p.__str__.return_value=value['path'];p.is_file.return_value=True;listed.append(p)
def check_adapter(r,h,a,p):
    assert r is row and h is history and len(p)==6 and set(a)==m.ARRAYS
    assert all(p[i][0] is phase_files[i][0] for i in range(6))
    return dict(mocked_array_checker=True,dimension=896,phases_checked=6)
def full_mocked_adapter():
    with patch.object(m,'pin',lambda p:pins[str(p)]),patch.object(m,'verify',lambda p:p), \
         patch.object(m,'pinned_source',lambda p,d:pins[str(p)]),patch.object(m,'read',lambda p:reads[str(p)]), \
         patch.object(m,'git_blob',lambda p:source_data[str(Path(p).relative_to(m.R))]), \
         patch.object(type(m.RAW),'rglob',lambda p,s:listed if p==m.RAW else (_ for _ in ()).throw(AssertionError('foreign listing'))), \
         patch.object(np,'load',load_mock),patch.object(m,'check_saved_arrays',check_adapter), \
         patch.dict(sys.modules,supervise=SimpleNamespace(__file__=str(m.R/'codex/certify-chaos/supervise.py'),ident=lambda p:None,members=lambda p:[])):
        result=m.run(args)
    assert result['period']==6 and result['outputs_checked']==14 and result['certificate_passed'] is False
    assert result['checked']['dimension']==896 and result['host_closure_rechecked'] is True
    return result
mocked_result=check('complete_loader_adapter_mocked_files_git_host_and_numeric_interface',full_mocked_adapter)

scratch=Path(tempfile.mkdtemp(prefix='phase-p6-audit-controls-'))
sentinel=scratch/'synthetic-original-evidence.txt';sentinel.write_text('unchanged manufactured original\n')
sentinel_sha=hashlib.sha256(sentinel.read_bytes()).hexdigest()
out=scratch/'audit-001/result.json'
def retained_failure():
    with patch.object(m,'RESULT',out),patch.object(m,'arguments',lambda:args), \
         patch.object(m,'run',side_effect=ValueError('manufactured admission refusal')):
        try:m.main()
        except ValueError as error:assert str(error)=='manufactured admission refusal'
        else:raise AssertionError('main swallowed manufactured refusal')
    value=json.loads((out.parent/'failure.json').read_text())
    assert not out.exists() and value['status']=='audit_refused' and value['certificate_passed'] is False
    assert value['requested_receipt_sha256']=='2'*64 and value['requested_admission_sha256']=='3'*64
    assert value['period']==6 and value['cell']==m.CELL and value['source_commit']==m.SOURCE
    assert hashlib.sha256(sentinel.read_bytes()).hexdigest()==sentinel_sha
check('fresh_failure_retains_source_receipt_namedcell_false_claims_and_original_sentinel',retained_failure)

print(json.dumps(dict(schema='codex-task34-phase-p6-audit-manufactured-delta-controls-v1',
    subject=dict(path=str(SOURCE),sha256=EXPECTED,bytes=SOURCE.stat().st_size),
    control_source=dict(path=__file__,sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),
    source_card=dict(path=str(m.CARD),sha256=m.CARD_SHA),source_plan=plan_pin,
    passed=len(results),failed=0,controls=results,scratch=str(scratch),
    scope='New manufactured P6 authority/interface/failure controls only. Tiny arrays dimension2; full loader numerical interface mocked. No original raw/model reads, target evaluation, Git operation, or host/process query. No prior control suite repeated.',
    actual_saved_phase_audit_performed=False,certificate_passed=False),indent=2,allow_nan=False))
