"""Manufactured decoded-record controls only. No target reads or producer imports."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
from fractions import Fraction as F

SOURCE = Path('/tmp/tube_saved_scalar.py')
spec = importlib.util.spec_from_file_location('independent_scalar', SOURCE)
a = importlib.util.module_from_spec(spec)
spec.loader.exec_module(a)
RESULT = Path('/tmp/tube-saved-scalar-controls.json')
FLAGS = dict(a.FLAGS)


def b(mid=0, rad=0):
    def pair(q):
        q = F(q)
        assert q.denominator & (q.denominator-1) == 0
        return [q.numerator, -(q.denominator.bit_length()-1)]
    return {'mid': pair(mid), 'rad': pair(rad)}


def p(path, value):
    data = json.dumps(value, sort_keys=True, separators=(',', ':')).encode()
    return {'path': path, 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}


def fixture(zero=False):
    c0 = [b() for _ in range(a.DIM)]
    c1 = [b(F(i,1024)) for i in range(a.DIM)]
    delta = F(0) if zero else F(1,1024)
    radius = 2*delta if delta else F(1,2**2048)
    initial = {'step': 0, 'centre': c0, 'trajectory_enclosure': False, 'certificate_passed': False}
    step = {'step': 1, 'input_centre': c0, 'next_centre': c1,
            'full_image': [b(F(i,1024),delta) for i in range(a.DIM)],
            'defects': [b(0,delta) for _ in range(a.DIM)],
            'defect_inf_upper': b(delta), 'trajectory_enclosure': False, 'certificate_passed': False}
    initialpin=p('/manufactured/thin/centre-000.json', initial)
    steppin=p('/manufactured/thin/step-001.json', step)
    embedding = {'tensor':'model.embed_tokens.weight','stored_dtype':'BF16','shape':[151936,a.DIM],
                 'row':198,'row_bytes':2*a.DIM,
                 'exact_rational_values_sha256':a.canonical([a.qr(F(0)) for _ in c0])}
    centre_hash=a.canonical(c1)
    row={'file':steppin,'step':1,'component_count':a.DIM,
         'checks':{k:True for k in a.STEP_CHECKS},
         'computed_defect_endpoint_max':a.qr(delta),'saved_defect_inf_upper_exact':a.qr(delta),
         'next_centre_canonical_sha256':centre_hash}
    report={'schema':'codex-task33-thin-pilot-saved-audit-v1','status':'complete_success',
            'trajectory_enclosure':False,'certificate_passed':False,'reachability_proved':False,
            'original_partial_evidence_preserved':True,'steps_checked':20,
            'steps':[row]+[{'synthetic_other_step':i} for i in range(2,21)],
            'initial_state':{'file':initialpin,'component_count':a.DIM,'exact_checkpoint_row_match':True,
                             'embedding':embedding},
            'entry_candidate':{'file':steppin,'step':1,'delta0':a.qr(delta),'e1_upper':a.qr(delta),
                               'next_centre_canonical_sha256':centre_hash}}
    reportpin=p('/manufactured/audit002/result.json',report)
    upstream={'initial_file':initialpin,'step1_file':steppin,'audit_result':reportpin}
    selected={'schema':'codex-task33-first-tube-selection-v1',**upstream,'embedding':embedding,
              'centre':c1,'centre_canonical_sha256':centre_hash,'delta0':a.qr(delta),'radius':a.qr(radius),
              'radius_rule':'2*delta0 if positive; otherwise exact 2^-2048',
              'incoming_error_le_radius':True,**FLAGS}
    tube={'centres':copy.deepcopy(c1),'nominal_radius':a.qr(radius),
          'live_primal':[b(F(i,1024),radius) for i in range(a.DIM)],
          'live_directions':[[b(0,1)] for _ in range(a.DIM)],
          'dimension':a.DIM,'direction_count':1,'precision':2048,**FLAGS}
    derivative=[[b(F(i%7-3,8),F(i%5+1,4))] for i in range(a.DIM)]
    derivative[-1]=[b(F(-1,8),F(5,2))]
    rows=[max(abs(q) for q in a.ball(x[0])[2:]) for x in derivative]
    operator={'schema':'conditional-whole-tube-infinity-operator-v1',**copy.deepcopy(tube),'threads':1,
              'full_output':[b(F(i,1024),F(i+1,4096)) for i in range(a.DIM)],
              'signed_derivative':derivative,'row_abs_endpoint_upper':[a.qr(q) for q in rows],
              'L':a.qr(max(rows))}
    binding={'source_commit':'a'*40,'synthetic_metadata_only':True,**FLAGS}
    records={'admission.json':copy.deepcopy(binding),'selection.json':selected,
             'prefix-binding.json':{'synthetic_metadata_only':True,**FLAGS},
             'tube-inputs.json':{**copy.deepcopy(tube),'threads':1},
             'actual-receiver.json':copy.deepcopy(tube),'tube-operator.json':operator,
             'coverage.json':{'started_layers':list(range(24)),'ended_layers':list(range(24)),
                  'head_passes':[[k,i,min(i+1024,151936)] for k in ('head_logits','head_derivatives') for i in range(0,151936,1024)],
                  'actual_receiver_equals_original':True,'all_direction_cube_retained':True,**FLAGS}}
    inventory={name:p(a.RAW+'/'+name,value) for name,value in records.items()}
    records['status.json']={'schema':'codex-task33-one-tube-status-v1','status':'operator_measurement_completed',
             'full_domain_gates_succeeded':True,'target_calls':1,'binding':copy.deepcopy(binding),
             'selection':inventory['selection.json'],'artifacts':list(inventory.values()),'L':operator['L'],**FLAGS}
    inventory['status.json']=p(a.RAW+'/status.json',records['status.json'])
    receipt={'returncode':0,'failure':None,'worker_reaped':True,'worker_identity_absent':True,
             'worker_group_session_absent':True,'output_files':list(inventory.values())}
    return [records,inventory,receipt,binding,initial,step,report,upstream]


checks=[]

def run(name, mutate=None, expected=None, zero=False):
    args=fixture(zero)
    if mutate:mutate(args)
    try:
        result=a.check_bundle(*args)
    except ValueError as e:
        if expected is None or expected not in str(e):raise
        checks.append({'name':name,'outcome':'refused_as_expected','reason':str(e)})
    else:
        assert expected is None, name+' unexpectedly accepted'
        assert result['operator_bound_admitted'] is False
        assert result['original_file_authentication_performed'] is False
        assert result['original_source_authentication_performed'] is False
        assert result['L_candidate']==a.qr(F(21,8))
        assert result['components_checked']==896 and result['head_chunks_checked']==[149,149]
        checks.append({'name':name,'outcome':'scalar_consistency_only','result':result})

try:
    run('all896_nonuniform_signed_rows_and_first_step_join')
    run('exact_zero_defect_uses_declared_positive_fallback',zero=True)
    run('understated_last_saved_derivative_row',lambda x:x[0]['tube-operator.json']['row_abs_endpoint_upper'].__setitem__(-1,a.qr(F(2))), 'every original signed row')
    run('understated_saved_max_L',lambda x:x[0]['tube-operator.json'].__setitem__('L',a.qr(F(2))), 'exact maximum L')
    run('last_full_primal_output_nonfinite_encoding',lambda x:x[0]['tube-operator.json']['full_output'][-1].__setitem__('rad',['inf',0]),'finite native integer')
    run('actual_receiver_native_radius_differs_but_still_contains_tube',lambda x:x[0]['actual-receiver.json']['live_primal'][-1].__setitem__('rad',[1,-8]),'all original native input')
    run('last_nominal_primal_coordinate_too_narrow',lambda x:x[0]['tube-inputs.json']['live_primal'][-1].__setitem__('rad',[1,-10]),'nominal closed cube')
    run('serialized_last_direction_coordinate_too_narrow',lambda x:x[0]['tube-inputs.json']['live_directions'][-1][0].__setitem__('rad',[1,-1]),'independent unit direction')
    run('selected_radius_differs_from_exact_first_defect',lambda x:x[0]['selection.json'].__setitem__('radius',a.qr(F(1,1024))),'exact selected centre/radius')
    run('last_local_defect_omits_saved_image_endpoint',lambda x:x[5]['defects'].__setitem__(-1,b()),'full signed defect')
    run('first_saved_bound_understates_signed_defects',lambda x:x[5].__setitem__('defect_inf_upper',b(F(1,2048))),'saved bound covers')
    run('audit_entry_incoming_scalar_differs',lambda x:x[6]['entry_candidate'].__setitem__('e1_upper',a.qr(F(0))),'audited exact first-step')
    run('audit_first_step_original_pin_differs',lambda x:x[6]['steps'][0].__setitem__('file',p('/manufactured/wrong.json',{})),'audited first-step pin')
    run('initial_checkpoint_row_value_hash_differs',lambda x:x[6]['initial_state']['embedding'].__setitem__('exact_rational_values_sha256','0'*64),'audited exact embedding')
    run('partial_original_thin_report_is_not_current_success',lambda x:x[6].__setitem__('status','closed_partial_failure'),'complete successful original thin audit')
    run('missing_last_derivative_head_chunk',lambda x:x[0]['coverage.json']['head_passes'].pop(),'both complete ordered 149')
    run('wrong_last_head_endpoint',lambda x:x[0]['coverage.json']['head_passes'][-1].__setitem__(2,151935),'both complete ordered 149')
    run('wrong_decoder_end_order',lambda x:x[0]['coverage.json']['ended_layers'].__setitem__(-1,22),'complete ordered 24')
    run('boolean_saved_layer_index',lambda x:x[0]['coverage.json']['started_layers'].__setitem__(1,True),'complete ordered 24')
    run('operator_file_without_final_status',lambda x:x[0].pop('status.json'),'exact eight successful')
    run('successful_internal_status_but_outer_resource_failure',lambda x:x[2].update(returncode=-15,failure='available_ram_floor'),'successful complete original supervision')
    run('outer_zero_but_failure_not_null',lambda x:x[2].__setitem__('failure','retained refusal'),'successful complete original supervision')
    run('outer_reap_without_group_closure',lambda x:x[2].__setitem__('worker_group_session_absent',False),'successful complete original supervision')
    run('status_original_admission_join_differs',lambda x:x[0]['status.json']['binding'].__setitem__('source_commit','b'*40),'status original admission')
    run('status_artifact_pin_substitution',lambda x:x[0]['status.json']['artifacts'].__setitem__(-1,p(a.RAW+'/coverage.json',{})),'status full preceding')
    run('nonconditional_certificate_flag',lambda x:x[0]['tube-operator.json'].__setitem__('certificate_passed',True),'conditional-only saved scope')
    result={'schema':'manufactured-tube-scalar-controls-v1','source':str(SOURCE),
            'source_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
            'fixture_scope':'Synthetic 896-component decoded JSON. Pins and receipt fields are manufactured, never authenticated originals.',
            'status':'passed','controls':checks,'count':len(checks),'target_evidence_reads':0,'model_calls':0,
            'old_controls_repeated':False,'runtime_admission':False}
    with RESULT.open('x')as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({'status':'passed','count':len(checks),'source_sha256':result['source_sha256'],'result_sha256':hashlib.sha256(RESULT.read_bytes()).hexdigest()}))
except BaseException as e:
    with Path('/tmp/tube-saved-scalar-controls-failure.json').open('x')as f:
        json.dump({'error_type':type(e).__name__,'error':str(e),'completed':checks,'source_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest()},f,indent=2)
    raise
