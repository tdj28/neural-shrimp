"""Read-only identity/projection adapter for EXP015 saved reference cells.

No ODE, interpolation, SVD, gain, score or directional-response computation.
Dense original case diagnostic arrays are excluded by explicit tree projection;
all retained scalar/array values are unmodified saved inputs.
"""
from __future__ import annotations
import gc
import hashlib
import itertools
import json
from pathlib import Path
import numpy as np
import run_phase_memory_recovery as codec
import phase_memory_acquisition as original

OLD=Path('/data2/neural-shrimp-data/raw/EXP-014/attempt-002')
OLD_SOURCE='406b89553b956bac1b97ef05413c799e383a9103'
OLD_RECEIPT_SHA='2494966914cb68d8c5286a5dca6710212ade171ead42c9cfaed82cebd44c12a8'
SETTINGS=('core','failed-near-edge');DIRECTIONS=('phase_write','phase_null','neuron_one');DELAYS=(1,2,4,8)
CONTEXT_DROP={'diagnostic_times','reference_states','reference_velocity','reference_adjoint'}
CASE_DROP={'full_linear_state_delta','full_linear_activation_delta','phase_only_activations'}
READOUT_FIELDS={'phase_only_states','phase_only_activations','full_linear_state_delta','full_linear_activation_delta','noise_gain','phase_only_q','full_linear_q','zero_response_q'}


def read(path):return json.loads(Path(path).read_text())
def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()
def binding(path,root):return dict(path=str(Path(path).relative_to(root)),bytes=Path(path).stat().st_size,sha256=sha(path))

def cell_census():
    return [dict(id=f'{s}--p{p}--{direction}--k{d}',setting=s,phase_index=p,direction=direction,delay_cycles=d)
        for s,p,direction,d in itertools.product(SETTINGS,range(8),DIRECTIONS,DELAYS)]

def pulse_ids(setting,phase_index,direction):
    return [f'{setting}--p{phase_index}--d{dose}--{direction}--s{sign:+d}' for dose in range(3) for sign in (-1,1)]

def _case_ids(*,shams):
    result=[]
    for s,p in itertools.product(SETTINGS,range(8)):
        result.extend(ident for direction in DIRECTIONS for ident in pulse_ids(s,p,direction))
        if shams:result.append(f'{s}--p{p}--sham')
    return set(result)

def _project_seal(tree):
    contexts=tree['contexts'];cases=tree['cases'];expected={f'{s}--p{p}' for s,p in itertools.product(SETTINGS,range(8))}
    if set(contexts)!=expected or set(cases)!=_case_ids(shams=True):raise ValueError('Original16context/304case census differs')
    if tree['tasks']!=original.build_tasks() or tree['complete'] is not True or tree['qualified'] is not True or tree['failures']!=[]:raise ValueError('Original reference seal is incomplete or task census differs')
    return dict(schema_version=tree['schema_version'],complete=True,qualified=True,
        contexts={k:{name:v for name,v in value.items() if name not in CONTEXT_DROP} for k,value in contexts.items()},
        cases={k:{name:v for name,v in value.items() if name not in CASE_DROP} for k,value in cases.items() if k in _case_ids(shams=False)})


def _array_markers(tree):
    result=[]
    def visit(value):
        if isinstance(value,dict):
            if '__array__' in value:
                if set(value)!={'__array__','dtype','shape'}:raise ValueError('Malformed array marker')
                result.append(value['__array__'])
            else:
                for item in value.values():visit(item)
        elif isinstance(value,list):
            for item in value:visit(item)
    visit(tree)
    if len(result)!=len(set(result)):raise ValueError('Duplicate original array marker')
    return result


def load_projected_packet(path,*,project_seal=False):
    """Hash full NPZ, decode only selected arrays; no numerical transformations."""
    path=Path(path);metadata=read(path)
    if path.is_symlink() or set(metadata)!={'schema','tree','arrays','semantic_sha256'} or metadata['schema']!='numeric-packet-v1':raise ValueError('Original numeric packet schema/alias differs')
    entry=metadata['arrays'];raw=path.parent/entry['path']
    if Path(entry['path']).name!=entry['path'] or raw.is_symlink() or not raw.resolve().is_relative_to(path.parent.resolve()):raise ValueError('Original raw path escapes packet')
    if raw.stat().st_size!=entry['bytes'] or sha(raw)!=entry['sha256']:raise ValueError('Original full-array hash differs')
    all_keys=_array_markers(metadata['tree']);tree=_project_seal(metadata['tree']) if project_seal else metadata['tree'];keys=_array_markers(tree)
    with np.load(raw,allow_pickle=False) as data:
        if len(data.files)!=len(set(data.files)) or set(data.files)!=set(all_keys):raise ValueError('Original exact NPZ array census differs')
        arrays={k:data[k].copy() for k in keys}
    used=set();value=codec._decode(tree,arrays,used)
    if used!=set(keys):raise ValueError('Projected array census differs')
    if not project_seal and codec.packet_digest(value)!=metadata['semantic_sha256']:raise ValueError('Original semantic packet hash differs')
    return value,dict(path=str(path),original_semantic_sha256=metadata['semantic_sha256'],original_array_count=len(all_keys),retained_array_count=len(keys),retained_array_bytes=sum(v.nbytes for v in arrays.values()),projection='omit dense context/case diagnostic arrays and16shams' if project_seal else 'full original packet')


def input_inventory(directory=OLD):
    """Return the exact63 receipt-bound input files using metadata and hashing."""
    directory=Path(directory).resolve()
    if directory!=OLD.resolve():raise ValueError('Only frozen EXP014 recovery attempt002 is an allowed source')
    if sha(directory/'receipt.json')!=OLD_RECEIPT_SHA:raise ValueError('Original receipt SHA differs')
    receipt=read(directory/'receipt.json');items=receipt['artifacts'];index={r['path']:r for r in items}
    if len(index)!=len(items) or receipt['source_commit']!=OLD_SOURCE:raise ValueError('Original receipt source/census differs')
    def checked(name):
        path=directory/name
        if path.is_symlink() or not path.resolve().is_relative_to(directory) or index.get(name)!=binding(path,directory):raise ValueError('Original bound file differs: '+name)
        return read(path)
    manifest=checked('manifest.json');ledger=checked('all-terminal-records.json');post=checked('postflight.json')
    if manifest['source_commit']!=OLD_SOURCE or manifest['plan_sha256']!=receipt['plan_sha256'] or post['error'] is not None:raise ValueError('Original source/postflight identity differs')
    names=['receipt.json','manifest.json','summary.json','all-terminal-records.json','postflight.json','reference-seal.json','reference-seal.npz','reference-seal-replay.json','events.jsonl'];packets={}
    for stage,expected in [('references',set(SETTINGS)),('propagators',{f'{s}--p{p}' for s,p in itertools.product(SETTINGS,range(8))})]:
        rows=ledger[stage]
        if len(rows)!=len(expected) or {r['id'] for r in rows}!=expected:raise ValueError('Original prerequisite terminal census differs')
        for row in rows:
            ident=row['id'];meta=f'{stage}/payloads/{ident}/packet.json'
            if row['status']!='complete' or row.get('qualified') is not True or row['packet_path']!=f'payloads/{ident}/packet.json':raise ValueError('Original prerequisite record incomplete/misbound')
            descriptor=checked(meta)
            if sha(directory/meta)!=row['packet_sha256'] or descriptor['semantic_sha256']!=row['packet_semantic_sha256']:raise ValueError('Original record-to-packet binding differs')
            array_name=descriptor['arrays']['path']
            if Path(array_name).name!=array_name:raise ValueError('Original array path escape')
            record=f'{stage}/records/{ident}.json'
            if checked(record)!=row:raise ValueError('Original terminal/record identity differs')
            names.extend([record,meta,str(Path(meta).parent/array_name)]);packets[ident]=meta
    if len(names)!=63 or len(set(names))!=63:raise AssertionError('Exact63 source artifact contract differs')
    bindings=[]
    for name in names:
        path=directory/name;item=binding(path,directory)
        if path.is_symlink() or not path.resolve().is_relative_to(directory) or name!='receipt.json' and index.get(name)!=item:raise ValueError('Required immutable input differs: '+name)
        bindings.append(item)
    return dict(directory=str(directory),source_commit=OLD_SOURCE,receipt_sha256=OLD_RECEIPT_SHA,plan_sha256=manifest['plan_sha256'],files=bindings,packets=packets,
        census=dict(references=2,propagators=16,sealed_contexts=16,sealed_pulse_cases=288,cells=192,realized_pulse_records=1152),scope='Reference-derived inputs only; no target trace packets or labels read.')


def _qualified(packet,label):
    if packet.get('status')!='complete' or packet.get('qualified') is not True or not packet.get('gates') or any(v is not True for v in packet['gates'].values()):raise ValueError('Saved prerequisite not qualified: '+label)


def validate_compact(bundle):
    expected_contexts={f'{s}--p{p}' for s,p in itertools.product(SETTINGS,range(8))}
    if set(bundle['references'])!=set(SETTINGS) or set(bundle['propagators'])!=expected_contexts or set(bundle['contexts'])!=expected_contexts or set(bundle['cases'])!=_case_ids(shams=False):raise ValueError('Compact full input census differs')
    for setting in SETTINGS:
        ref=bundle['references'][setting];_qualified(ref,setting)
        if ref['setting']!=setting or len(ref['phases'])!=8 or [x['phase_index'] for x in ref['phases']]!=list(range(8)):raise ValueError('Reference eight-phase identity differs')
    for setting,phase in itertools.product(SETTINGS,range(8)):
        ident=f'{setting}--p{phase}';ref=bundle['references'][setting];prop=bundle['propagators'][ident];context=bundle['contexts'][ident];_qualified(prop,ident)
        expected=dict(id=ident,setting=setting,phase_index=phase,status='complete',period=ref['period'],phase_epoch=phase*ref['period']/8)
        if any(type(context.get(k)) is not type(v) or context.get(k)!=v for k,v in expected.items()):raise ValueError('Sealed context identity differs')
        if prop['setting']!=setting or type(prop['phase_index']) is not int or prop['phase_index']!=phase or prop['period']!=ref['period'] or prop['phase_epoch']!=context['phase_epoch']:raise ValueError('Phase propagator identity differs')
        saved_phase=ref['phases'][phase]
        for key in ('initial','flow','phase_gradient'):
            if not np.array_equal(context[key],saved_phase[key]):raise ValueError('Reference/seal phase binding differs: '+key)
        if not np.array_equal(prop['initial'],context['initial']):raise ValueError('Phase propagator initial differs')
        if set(context['directions'])!=set(DIRECTIONS) or context['classifier_orientations']!=saved_phase['classifier_orientations']:raise ValueError('Direction/orientation census differs')
        for name in DIRECTIONS:
            if not np.array_equal(context['directions'][name],saved_phase['directions'][name]):raise ValueError('Sealed direction differs')
        if set(context['readouts'])!={str(d) for d in DELAYS} or set(context)&CONTEXT_DROP:raise ValueError('Compact clock field census differs')
        for direction in DIRECTIONS:
            for dose,sign in itertools.product(range(3),(-1,1)):
                cid=f'{ident}--d{dose}--{direction}--s{sign:+d}';packet=bundle['cases'][cid]
                expected_case=dict(setting=setting,phase_index=phase,kind='pulse',dose_index=dose,epsilon=(1e-4,1e-3,1e-2)[dose],direction=direction,sign=sign)
                if packet['id']!=cid or packet['context_id']!=ident or packet['status']!='complete' or packet['case']!=expected_case or any(type(packet['case'].get(k)) is not type(v) for k,v in expected_case.items()) or set(packet)&CASE_DROP:raise ValueError('Projected pulse case identity differs')
                if packet['orientation']!=context['classifier_orientations'][direction] or set(packet['readouts'])!={str(d) for d in DELAYS}:raise ValueError('Sealed pulse readout identity differs')
                if packet['pulse_used_for_phase_only']!='requested_delta' or packet['pulse_used_for_full_linear']!='realized_delta':raise ValueError('Original prediction pulse convention differs')
                for delay in DELAYS:
                    row=packet['readouts'][str(delay)]
                    if set(row)!=READOUT_FIELDS:raise ValueError('Six-pulse sealed prediction fields differ')
                    for key in ('phase_only_states','phase_only_activations','full_linear_state_delta','full_linear_activation_delta'):
                        a=row[key]
                        if not isinstance(a,np.ndarray) or a.dtype!=np.float64 or a.shape!=(32,3) or not np.isfinite(a).all():raise ValueError('Sealed prediction array identity differs')
    return dict(references=2,propagators=16,contexts=16,pulse_cases=288,cells=192,pulse_compatibility_records=1152)


def compact_inputs(references,propagators,prepared):
    """Pure field projection; no derived numerical values or dense case copies."""
    if prepared.get('complete') is not True or prepared.get('qualified') is not True:raise ValueError('Original prepared seal not qualified')
    projected=_project_seal(prepared) if 'tasks' in prepared else prepared
    props={f'{k[0]}--p{k[1]}' if isinstance(k,tuple) else k:v for k,v in propagators.items()}
    bundle=dict(schema_version='EXP015-reference-inputs-v1',references=references,propagators=props,contexts=projected['contexts'],cases=projected['cases'])
    validate_compact(bundle);return bundle


def load_bound_inputs(directory=OLD):
    inventory=input_inventory(directory);directory=Path(directory);references={};props={};loads={}
    prepared,loads['reference-seal']=load_projected_packet(directory/'reference-seal.json',project_seal=True)
    for ident,path in inventory['packets'].items():
        packet,loads[ident]=load_projected_packet(directory/path)
        if ident in SETTINGS:references[ident]=packet
        else:props[ident]=packet
    bundle=compact_inputs(references,props,prepared);bundle['provenance']=inventory;bundle['projection_receipts']=loads
    gc.collect();return bundle


def cell_inputs(bundle,cell):
    """Return shared original packets as evaluate_cell keyword arguments."""
    if cell not in cell_census() or any(type(cell[k]) is not int for k in ('phase_index','delay_cycles')):raise ValueError('Exact planned cell identity required')
    setting,phase,direction=cell['setting'],cell['phase_index'],cell['direction'];context=f'{setting}--p{phase}'
    return dict(reference=bundle['references'][setting],propagator=bundle['propagators'][context],sealed_context=bundle['contexts'][context],
        sealed_cases={ident:bundle['cases'][ident] for ident in pulse_ids(setting,phase,direction)})


def input_file_bindings(directory=OLD):
    return input_inventory(directory)['files']


def metadata_input_paths(directory=OLD):
    return [row['path'] for row in input_file_bindings(directory)]


def cell_input_map(cell):
    """Exact source artifact/key mapping, independent of numerical cell outputs."""
    if cell not in cell_census() or any(type(cell[k]) is not int for k in ('phase_index','delay_cycles')):raise ValueError('Unplanned cell')
    setting,phase,direction=cell['setting'],cell['phase_index'],cell['direction'];context=f'{setting}--p{phase}'
    return dict(id=cell['id'],reference_packet=f'references/payloads/{setting}/packet.json',
        propagator_packet=f'propagators/payloads/{context}/packet.json',reference_seal='reference-seal.json',
        sealed_context_key=context,sealed_clock_key=str(cell['delay_cycles']),
        sealed_case_ids=pulse_ids(setting,phase,direction),sealed_prediction_key=str(cell['delay_cycles']),
        propagator_convention='For elapsed t=kT+s, Phi_phase(s) @ matrix_power(M_phase,k) @ realized_delta; M acts first.',
        activation_convention='(1-tanh(reference_state)^2)*full_linear_state_delta at original32clock samples',
        output_cell_id=cell['id'])
