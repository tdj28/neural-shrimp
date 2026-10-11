"""Fabricated output packets only; no target gains or integrations."""
import copy
import json
from pathlib import Path
import sys
import numpy as np
import pytest
root=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(Path(__file__).resolve().parent),str(root/'tools')]
import plot_phase_transverse_gain as p
import run_phase_memory_recovery as codec


def maths(cell,scale=1.):
    phase=(-1 if cell['phase_index']%2 else 1)*(cell['phase_index']+1)*scale
    transverse=phase*.4/cell['delay_cycles']
    obs=dict(total=phase+transverse,phase_algebraic=phase,transverse=transverse,phase_physical=phase,phase_clock=phase,transport_defect=0.,clock_flow_defect=0.,initial_time_shift=phase,clock_normalization_defect=0.,ideal_accounting_residual=0.)
    return dict(observable=obs,noise_gain=.2,orientation=1,noise_rows=[dict(sigma=s,noise_sd=s*.2,signed_standardized={k:v/(s*.2) for k,v in obs.items()}) for s in p.NOISES])


def fabricated():
    cells=p.cell_census();rows=[];packets={}
    for cell in cells:
        ids=p.pulse_ids(cell);unit=maths(cell);compat=[]
        for i,ident in enumerate(ids):
            m=maths(cell,(1e-4,1e-3,1e-2)[i//2]*(-1 if i%2==0 else 1))
            compat.append(dict(id=ident,status='complete',complete=True,mathematics=m,noise_rows=m['noise_rows'],original_finite_phase_only_q=m['observable']['phase_algebraic'],approximation={'synthetic':True}))
        flags=dict(qualified=True,qualified_algebraic=True,qualified_ideal_neutral=True)
        packets[cell['id']]=dict(cell=cell,status='complete',complete=True,**flags,unit=unit,compatibility=compat,clock_sha256='c'*64,error=None,example_array=np.eye(3))
        rows.append(dict(cell=cell,status='complete',**flags,error=None,compatibility_ids=ids,compatibility_counts=dict(required=6,complete=6,failed=0,unavailable=0)))
    summary=dict(experiment_id='EXP-015',required_cells=192,status_counts={'complete':192},calculated_cells=192,all_gates_pass_cells=192,algebraically_qualified_cells=192,ideal_neutral_qualified_cells=192,realized_pulse_rows=dict(required=1152,complete=1152,failed=0,unavailable=0),setting_means=None,original_study_requalified=False,complete=True,source_valid=True,error=None,postflight_error=None)
    manifest=dict(plan=dict(cells_census=cells),source_commit='a'*40,plan_sha256='b'*64)
    return summary,manifest,rows,packets


@pytest.fixture
def data():return fabricated()


def test_exact_census_signed_fields_and_all_noise_rows(data):
    report=p.collect(*data)
    assert tuple(len(report[k]) for k in ('cells','unit','unit_noise','realized','realized_noise'))==(192,192,576,1152,3456)
    assert report['display']['setting_means'] is None
    values=[r['phase_algebraic'] for r in report['unit']]
    assert min(values)<0<max(values)


@pytest.mark.parametrize('bad',['missing_cell','duplicate_cell','bool_phase','missing_compat','counts','qualification','noise','sigma','nonfinite','means','extra_packet'])
def test_visible_saved_output_corruption_rejected(data,bad):
    summary,manifest,rows,packets=data;cell=rows[0]['cell'];packet=packets[cell['id']]
    if bad=='missing_cell':rows.pop()
    elif bad=='duplicate_cell':rows[-1]=rows[0]
    elif bad=='bool_phase':rows[0]['cell']=dict(rows[0]['cell'],phase_index=False)
    elif bad=='missing_compat':packet['compatibility'].pop()
    elif bad=='counts':summary['realized_pulse_rows']['complete']-=1
    elif bad=='qualification':packet['qualified_algebraic']=False
    elif bad=='noise':packet['unit']['noise_rows'][0]['signed_standardized']['total']+=1.
    elif bad=='sigma':packet['unit']['noise_rows'][0]['sigma']=.1
    elif bad=='nonfinite':packet['unit']['observable']['transverse']=float('nan')
    elif bad=='means':summary['setting_means']={'core':1.}
    else:packets['unplanned']={}
    with pytest.raises((ValueError,KeyError)):p.collect(*data)


def test_failed_qualification_remains_plotted_without_erasing_values(data):
    summary,manifest,rows,packets=data
    for r in (rows[0],packets[rows[0]['cell']['id']]):r.update(qualified=False,qualified_algebraic=False,qualified_ideal_neutral=False)
    for k in ('all_gates_pass_cells','algebraically_qualified_cells','ideal_neutral_qualified_cells'):summary[k]-=1
    report=p.collect(*data);assert report['unit'][0]['status']=='complete' and report['unit'][0]['total'] is not None
    assert report['cells'][0]['qualified_algebraic'] is False


def test_missing_cell_retains_all_compatibility_and_noise_placeholders(data):
    summary,manifest,rows,packets=data;packets.pop(rows[0]['cell']['id']);rows[0].update(status='not_started',qualified=False,qualified_algebraic=False,qualified_ideal_neutral=False,compatibility_counts=dict(required=6,complete=0,failed=0,unavailable=6))
    summary.update(status_counts={'complete':191,'not_started':1},calculated_cells=191,all_gates_pass_cells=191,algebraically_qualified_cells=191,ideal_neutral_qualified_cells=191,realized_pulse_rows=dict(required=1152,complete=1146,failed=0,unavailable=6),complete=False)
    report=p.collect(*data);assert len(report['realized'])==1152
    assert sum(r['status']=='unavailable' for r in report['unit_noise'])==3
    assert sum(r['status']=='unavailable' for r in report['realized_noise'])==18


def seal(tmp_path,data):
    summary,manifest,rows,packets=data;source=tmp_path/'source';source.mkdir();(source/'cells').mkdir()
    for row in rows:
        cell=row['cell'];packet=codec.save_packet(source/'cells',cell['id'],packets[cell['id']]);row.update(packet_path='cells/'+cell['id']+'.json',packet_sha256=packet['sha256'],packet_semantic_sha256=packet['semantic_sha256'])
    initial=[dict(cell=c,status='not_started',qualified=False,qualified_algebraic=False,qualified_ideal_neutral=False,compatibility_ids=p.pulse_ids(c),compatibility_counts=dict(required=6,complete=0,failed=0,unavailable=6)) for c in p.cell_census()]
    for name,value in [('manifest.json',manifest),('summary.json',summary),('terminal-records.json',rows),('planned-cells.json',p.cell_census()),('initial-terminal-records.json',initial),('postflight.json',dict(error=None,source_valid=True))]:(source/name).write_text(json.dumps(value))
    artifacts=[p.binding(path,source) for path in sorted(source.rglob('*')) if path.is_file()]
    receipt=dict(experiment_id='EXP-015',source_commit=manifest['source_commit'],plan_sha256=manifest['plan_sha256'],status='complete',artifacts=artifacts,artifact_bytes=sum(a['bytes'] for a in artifacts))
    (source/'receipt.json').write_text(json.dumps(receipt));return source


def test_independent_codec_receipt_and_actual_figure_pipeline(data,tmp_path):
    source=seal(tmp_path,data);report,provenance=p.load_saved(source)
    assert provenance['artifacts_verified']==390 and report['counts']['calculated_cells']==192
    output=tmp_path/'report';result=p.write_report(report,provenance,output)
    assert (output/'phase-transverse-gains.png').stat().st_size>10000 and (output/'phase-transverse-gains.pdf').exists()
    assert len((output/'realized_noise.csv').read_text().splitlines())==3457
    receipt=p.read(output/'receipt.json');assert receipt['artifacts']==[p.binding(path,output) for path in sorted(output.iterdir()) if path.name!='receipt.json']
    with pytest.raises(FileExistsError):p.write_report(report,provenance,output)
    with pytest.raises(ValueError):p.write_report(report,provenance,source/'report')
    (source/'summary.json').write_text('{}')
    with pytest.raises(ValueError,match='hash'):p.load_saved(source)


def test_array_codec_strict_shape_and_semantic_hash(tmp_path):
    packet=codec.save_packet(tmp_path,'packet',{'x':np.eye(3),'failure':float('nan')});value,digest=p.load_packet(packet['path'])
    assert np.array_equal(value['x'],np.eye(3)) and np.isnan(value['failure']) and digest==packet['semantic_sha256']
    meta=p.read(packet['path']);meta['tree']['x']['shape']=[9];Path(packet['path']).write_text(json.dumps(meta))
    with pytest.raises(ValueError,match='shape'):p.load_packet(packet['path'])
