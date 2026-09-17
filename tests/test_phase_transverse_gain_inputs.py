"""Synthetic metadata/arrays only; never evaluates reference gains or target ODE."""
import copy
import json
from pathlib import Path
import sys
import numpy as np
import pytest
root=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(Path(__file__).resolve().parent),str(root/'tools')]
import phase_transverse_gain_inputs as a


def fabricated_inputs():
    refs={};props={};contexts={};cases={};period=8.
    for setting in a.SETTINGS:
        phases=[]
        for phase in range(8):
            initial=np.array([float(phase),0.,0.]);flow=np.array([0.,1.,0.]);z=np.array([0.,1.,0.])
            directions={d:np.array([1.,0.,0.]) for d in a.DIRECTIONS};orientations={d:1 for d in a.DIRECTIONS}
            phases.append(dict(phase_index=phase,phase_epoch=float(phase),initial=initial,flow=flow,phase_gradient=z,directions=directions,classifier_orientations=orientations))
        refs[setting]=dict(setting=setting,status='complete',qualified=True,gates={'synthetic':True},period=period,phases=phases)
        for phase in range(8):
            cid=f'{setting}--p{phase}';fields=phases[phase]
            props[cid]=dict(setting=setting,phase_index=phase,phase_epoch=float(phase),period=period,initial=fields['initial'],status='complete',qualified=True,gates={'synthetic':True},times=np.linspace(0.,8.,5),phi=np.repeat(np.eye(3)[None],5,axis=0))
            readouts={str(d):dict(times=np.linspace(float(d-1)*8,float(d)*8,32),weights=np.zeros((32,3)),reference_activations=np.zeros((32,3))) for d in a.DELAYS}
            contexts[cid]=dict(id=cid,setting=setting,status='complete',period=period,**fields,readouts=readouts,
                diagnostic_times=np.linspace(0.,320.,17),reference_states=np.full((17,3),123.),reference_velocity=np.full((17,3),456.),reference_adjoint=np.full((17,3),789.))
            for direction in a.DIRECTIONS:
                for dose in range(3):
                    for sign in (-1,1):
                        ident=f'{cid}--d{dose}--{direction}--s{sign:+d}'
                        meta=dict(setting=setting,phase_index=phase,kind='pulse',dose_index=dose,epsilon=(1e-4,1e-3,1e-2)[dose],direction=direction,sign=sign)
                        pred={str(d):dict(phase_only_states=np.zeros((32,3)),phase_only_activations=np.zeros((32,3)),full_linear_state_delta=np.full((32,3),float(sign*(dose+1))),full_linear_activation_delta=np.full((32,3),sign*(dose+1)/2),noise_gain=.1,phase_only_q=0.,full_linear_q=0.,zero_response_q=0.) for d in a.DELAYS}
                        cases[ident]=dict(id=ident,context_id=cid,status='complete',case=meta,pulse={'realized_delta':np.array([sign*meta['epsilon'],0.,0.]),'requested_delta':np.array([sign*meta['epsilon'],0.,0.])},orientation=1,pulse_used_for_phase_only='requested_delta',pulse_used_for_full_linear='realized_delta',readouts=pred,
                            full_linear_state_delta=np.full((17,3),444.),full_linear_activation_delta=np.full((17,3),555.),phase_only_activations=np.full((17,3),666.))
            cases[cid+'--sham']=dict(id=cid+'--sham',dense=np.full((17,3),888.))
    prepared=dict(schema_version='synthetic-original',contexts=contexts,cases=cases,tasks=a.original.build_tasks(),complete=True,qualified=True,failures=[])
    return refs,props,prepared


def test_exact192_cells_six_pulses_and_phi_order_mapping():
    cells=a.cell_census();assert len(cells)==192 and len({c['id'] for c in cells})==192
    for c in cells:
        mapping=a.cell_input_map(c);ids=a.pulse_ids(c['setting'],c['phase_index'],c['direction'])
        assert len(ids)==6 and mapping['sealed_case_ids']==ids and mapping['sealed_prediction_key']==str(c['delay_cycles'])
        assert 'Phi_phase(s) @ matrix_power(M_phase,k) @ realized_delta' in mapping['propagator_convention']
    changed=dict(cells[1],delay_cycles=True)
    with pytest.raises(ValueError):a.cell_input_map(changed)


def test_compact_projection_preserves_values_and_shares_arrays():
    refs,props,prepared=fabricated_inputs();bundle=a.compact_inputs(refs,props,prepared)
    assert a.validate_compact(bundle)==dict(references=2,propagators=16,contexts=16,pulse_cases=288,cells=192,pulse_compatibility_records=1152)
    assert len(bundle['cases'])==288 and all(not (set(c)&a.CASE_DROP) for c in bundle['cases'].values())
    for cell in a.cell_census():
        kwargs=a.cell_inputs(bundle,cell);ident=f'{cell["setting"]}--p{cell["phase_index"]}'
        assert kwargs['reference'] is refs[cell['setting']] and kwargs['propagator'] is props[ident]
        assert len(kwargs['sealed_cases'])==6 and kwargs['sealed_context']['readouts'] is prepared['contexts'][ident]['readouts']
        for key,value in kwargs['sealed_cases'].items():
            assert value['pulse'] is prepared['cases'][key]['pulse']
            for delay in a.DELAYS:
                assert value['readouts'][str(delay)]['full_linear_state_delta'] is prepared['cases'][key]['readouts'][str(delay)]['full_linear_state_delta']
    assert not any('sham' in key for key in bundle['cases'])


@pytest.mark.parametrize('mutation',['missing_ref','missing_prop','wrong_prop','period','bool_phase','missing_case','case_sign','bool_sign','direction','orientation','convention','float32','shape','nan','readout','unqualified'])
def test_bad_input_identities_and_arrays_fail_closed(mutation):
    refs,props,prepared=fabricated_inputs();case=prepared['cases']['core--p0--d0--phase_write--s-1'];context=prepared['contexts']['core--p0']
    if mutation=='missing_ref':refs.pop('core')
    elif mutation=='missing_prop':props.pop('core--p0')
    elif mutation=='wrong_prop':props['core--p0']['phase_index']=1
    elif mutation=='period':context['period']=9.
    elif mutation=='bool_phase':context['phase_index']=False
    elif mutation=='missing_case':prepared['cases'].pop(case['id'])
    elif mutation=='case_sign':case['case']['sign']=1
    elif mutation=='bool_sign':prepared['cases']['core--p0--d0--phase_write--s+1']['case']['sign']=True
    elif mutation=='direction':context['directions']=dict(context['directions'],phase_write=np.array([0.,0.,1.]))
    elif mutation=='orientation':case['orientation']=-1
    elif mutation=='convention':case['pulse_used_for_full_linear']='requested_delta'
    elif mutation=='float32':case['readouts']['1']['full_linear_state_delta']=np.zeros((32,3),dtype=np.float32)
    elif mutation=='shape':case['readouts']['1']['full_linear_state_delta']=np.zeros((31,3))
    elif mutation=='nan':case['readouts']['1']['full_linear_state_delta'][0,0]=np.nan
    elif mutation=='readout':case['readouts'].pop('8')
    else:prepared['qualified']=False
    with pytest.raises((ValueError,KeyError)):a.compact_inputs(refs,props,prepared)


def test_projected_numeric_codec_skips_dense_outputs_and_replays_selected_arrays(tmp_path):
    refs,props,prepared=fabricated_inputs();sealed=a.codec.save_packet(tmp_path,'reference-seal',prepared)
    projected,receipt=a.load_projected_packet(sealed['path'],project_seal=True)
    assert receipt['retained_array_count']<receipt['original_array_count']
    assert receipt['original_semantic_sha256']==sealed['semantic_sha256']
    bundle=a.compact_inputs(refs,props,projected);assert len(bundle['cases'])==288
    source=prepared['cases']['core--p0--d0--phase_write--s-1']['readouts']['4']['full_linear_state_delta']
    target=bundle['cases']['core--p0--d0--phase_write--s-1']['readouts']['4']['full_linear_state_delta']
    assert np.array_equal(source,target) and source.dtype==target.dtype
    metadata=json.loads((tmp_path/'reference-seal.json').read_text());metadata['arrays']['sha256']='0'*64
    (tmp_path/'reference-seal.json').write_text(json.dumps(metadata))
    with pytest.raises(ValueError,match='hash'):a.load_projected_packet(tmp_path/'reference-seal.json',project_seal=True)


def make_source(tmp_path,monkeypatch):
    refs,props,prepared=fabricated_inputs();root=tmp_path/'old';root.mkdir();ledger={'references':[],'propagators':[]}
    a.codec.save_packet(root,'reference-seal',prepared)
    for stage,packets in [('references',refs),('propagators',props)]:
        for ident,packet in packets.items():
            dest=root/stage/'payloads'/ident;sealed=a.codec.save_packet(dest,'packet',packet)
            row=dict(id=ident,status='complete',qualified=True,packet_path=f'payloads/{ident}/packet.json',packet_sha256=sealed['sha256'],packet_semantic_sha256=sealed['semantic_sha256'])
            (root/stage/'records').mkdir(exist_ok=True);(root/stage/'records'/f'{ident}.json').write_text(json.dumps(row));ledger[stage].append(row)
    values={'manifest.json':dict(source_commit=a.OLD_SOURCE,plan_sha256='b'*64),'summary.json':{'intentionally':'metadata-bound only'},'all-terminal-records.json':ledger,'postflight.json':dict(error=None,resources={}),'reference-seal-replay.json':{'complete':True}}
    for name,value in values.items():(root/name).write_text(json.dumps(value))
    (root/'events.jsonl').write_text('{}\n')
    artifacts=[a.binding(p,root) for p in sorted(root.rglob('*')) if p.is_file()]
    (root/'receipt.json').write_text(json.dumps(dict(source_commit=a.OLD_SOURCE,plan_sha256='b'*64,artifacts=artifacts)))
    monkeypatch.setattr(a,'OLD',root);monkeypatch.setattr(a,'OLD_RECEIPT_SHA',a.sha(root/'receipt.json'))
    return root


def test_exact63_file_inventory_and_full_load_without_numerical_evaluation(tmp_path,monkeypatch):
    root=make_source(tmp_path,monkeypatch);inventory=a.input_inventory(root)
    assert len(inventory['files'])==63 and len(inventory['packets'])==18
    assert a.metadata_input_paths(root)==[r['path'] for r in inventory['files']]
    assert a.input_file_bindings(root)==inventory['files']
    bundle=a.load_bound_inputs(root);assert len(bundle['cases'])==288 and len(bundle['projection_receipts'])==19
    assert all(len(a.cell_inputs(bundle,c)['sealed_cases'])==6 for c in a.cell_census())
    assert not any('/traces/' in x['path'] or x['path'].startswith('traces/') for x in inventory['files'])
    (root/'summary.json').write_text('{}')
    with pytest.raises(ValueError,match='differs'):a.input_inventory(root)


def test_rejects_packet_alias_and_original_receipt_drift(tmp_path,monkeypatch):
    root=make_source(tmp_path,monkeypatch)
    (root/'receipt.json').write_text('{}')
    with pytest.raises(ValueError,match='receipt'):a.input_inventory(root)
