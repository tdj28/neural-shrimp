"""Synthetic archive/receipt controls; no producer data or code execution."""
import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tarfile

import pytest

PATH = Path(__file__).resolve().parents[1] / 'codex/huginn-independent-replay/admission.py'
spec = importlib.util.spec_from_file_location('task8_admission_test', PATH)
a = importlib.util.module_from_spec(spec); spec.loader.exec_module(a)


def pin(path):
    return {'path': str(path), 'bytes': path.stat().st_size, 'sha256': a.digest(path)}


def fixture(tmp_path, monkeypatch, part='B', partition_rows=None):
    source = tmp_path/'analysis.py'; source.write_text('# synthetic source; never executed\n')
    monkeypatch.setattr(a, 'ANALYSIS', a.digest(source))
    receipt = tmp_path/'receipt.json'
    receipt.write_text(json.dumps({'completed': True, 'hash_verified': True, 'terminated': True,
                                   'source': a.FROZEN, 'assets': 'synthetic', 'order': 'before'}))
    producer = tmp_path/'producer.json'; producer.write_text('{"primary_success": false}')
    rows = []
    if partition_rows is None:
        partition_rows = [[{'index': i}] for i in range(1 if part == 'A' else 2)]
    for i, records in enumerate(partition_rows):
        p = tmp_path/f'rows-{i}.jsonl'
        p.write_text(''.join(json.dumps(row) + '\n' for row in records)); rows.append(p)
    files = {'source': pin(source), 'receipt': pin(receipt), 'producer': pin(producer),
             **{f'rows{i}': pin(p) for i, p in enumerate(rows)}}
    members = [{'member': f'part{part}/rows-{i}.jsonl', 'file': f'rows{i}'} for i in range(len(rows))]
    roles = {'rows': [f'rows{i}' for i in range(len(rows))], 'producer_analysis': 'producer', 'analysis_source': 'source'}
    if part == 'A':
        p = tmp_path/'fixed-spectrum.json'; p.write_text('{"fixed_point_residual_F": 0.01}')
        files['spectrum0'] = pin(p)
        members.append({'member': 'partA/spectra/q0000/fixed-spectrum.json', 'file': 'spectrum0'})
        roles.update(spectra={str(q): 'spectrum0' if q == 0 else None for q in range(64)},
            absent_spectra={str(q): {'path': str(tmp_path/f'q{q:04d}'/'fixed-spectrum.json'),
                 'archive': 'archive', 'member': f'partA/spectra/q{q:04d}/fixed-spectrum.json'} for q in range(1,64)})
    archive = tmp_path/'results.tgz'
    with tarfile.open(archive, 'w:gz') as tar:
        for row in members:tar.add(files[row['file']]['path'], arcname=row['member'])
    files['archive'] = pin(archive)
    assertions = [{'file': 'receipt', 'format': 'json', 'pointer': '/completed', 'equals': True},
                  {'file': 'receipt', 'format': 'json', 'pointer': '/hash_verified', 'equals': True},
                  {'file': 'receipt', 'format': 'json', 'pointer': '/terminated', 'equals': True},
                  {'file': 'receipt', 'format': 'json', 'pointer': '/source', 'equals': a.FROZEN},
                  {'file': 'receipt', 'format': 'json', 'pointer': '/assets', 'equals': 'synthetic'}]
    obligations = {'complete_producer_scope': [0], 'retrieval': [1], 'verified_termination': [2],
                   'source_and_invocation': [3], 'resolved_assets': [4], 'analysis_completed': [0]}
    if part == 'A':
        assertions.append({'file': 'receipt', 'format': 'json', 'pointer': '/order', 'equals': 'before'})
        obligations['predictor_before_readout'] = [5]
    m = {'schema': 'codex-huginn-replay-inputs-v1', 'part': part, 'frozen_commit': a.FROZEN,
         'frozen_analysis_sha256': a.ANALYSIS, 'files': files, 'roles': roles,
         'receipt_assertions': assertions, 'obligations': obligations,
         'archives': [{'file': 'archive', 'members': members}]}
    return m


def run(tmp_path, manifest):
    path = tmp_path/'inputs.json'; path.write_text(json.dumps(manifest))
    return a.authenticate(path, a.digest(path), manifest['part'], root=tmp_path, max_input_bytes=10**7)


@pytest.mark.parametrize('part', ['A','B'])
def test_complete_local_receipts_archive_and_all_pins(tmp_path, monkeypatch, part):
    m=fixture(tmp_path, monkeypatch, part); got=run(tmp_path,m)
    assert got['report']['model_or_network_called'] is False
    assert got['report']['certificate_passed'] is False
    assert len(got['rows']) == (1 if part=='A' else 2)
    if part=='A':assert len(got['spectra'])==64 and got['spectra'][1] is None


@pytest.mark.parametrize('mutation', ['manifest','source','rows','producer','archive','receipt'])
def test_tamper_refuses_before_analysis(tmp_path,monkeypatch,mutation):
    m=fixture(tmp_path,monkeypatch); p=tmp_path/'inputs.json';p.write_text(json.dumps(m));expected=a.digest(p)
    if mutation=='manifest':p.write_text('{}')
    else:Path(m['files'][{'rows':'rows0'}.get(mutation,mutation)]['path']).write_bytes(b'changed')
    with pytest.raises(ValueError):a.authenticate(p,expected,'B',root=tmp_path,max_input_bytes=10**7)


@pytest.mark.parametrize('mutation',['missing_obligation','empty_obligation','bool_index','unused_assertion','wrong_source','alias_role','duplicate_partition','absent_file','budget'])
def test_incomplete_or_false_authority_refused(tmp_path,monkeypatch,mutation):
    m=fixture(tmp_path,monkeypatch)
    if mutation=='missing_obligation':del m['obligations']['verified_termination']
    elif mutation=='empty_obligation':m['obligations']['retrieval']=[]
    elif mutation=='bool_index':m['obligations']['retrieval']=[True]
    elif mutation=='unused_assertion':m['receipt_assertions'].append(copy.deepcopy(m['receipt_assertions'][0]))
    elif mutation=='wrong_source':m['frozen_commit']='0'*40
    elif mutation=='alias_role':m['roles']['producer_analysis']='rows0'
    elif mutation=='duplicate_partition':m['roles']['rows']=['rows0','rows0']
    elif mutation=='absent_file':Path(m['files']['rows0']['path']).unlink()
    elif mutation=='budget':m['files']['archive']['bytes']=10**8
    with pytest.raises((ValueError,KeyError)):run(tmp_path,m)


def test_receipt_boolean_is_not_integer(tmp_path,monkeypatch):
    m=fixture(tmp_path,monkeypatch);m['receipt_assertions'][0]['equals']=1
    with pytest.raises(ValueError,match='assertion failed'):run(tmp_path,m)


def test_jsonl_termination_and_escaped_pointer(tmp_path,monkeypatch):
    m=fixture(tmp_path,monkeypatch);p=tmp_path/'termination.jsonl';p.write_text('{"x/y":{"~z":true}}\n')
    m['files']['termination']=pin(p);m['receipt_assertions'][2]={'file':'termination','format':'jsonl','pointer':'/0/x~1y/~0z','equals':True}
    assert run(tmp_path,m)['report']['receipt_assertions'][2]['format']=='jsonl'


def test_extracted_bytes_disagree_with_original_archive(tmp_path,monkeypatch):
    m=fixture(tmp_path,monkeypatch);p=Path(m['files']['rows0']['path']);p.write_text('{"index": 7}\n');m['files']['rows0']=pin(p)
    with pytest.raises(ValueError,match='differs from original archive'):run(tmp_path,m)


@pytest.mark.parametrize('case',['missing','duplicate','symlink','traversal'])
def test_archive_members_refuse_missing_duplicate_links_and_paths(tmp_path,monkeypatch,case):
    m=fixture(tmp_path,monkeypatch);archive=Path(m['files']['archive']['path'])
    if case=='traversal':m['archives'][0]['members'][0]['member']='../outside'
    else:
        with tarfile.open(archive,'w:gz') as tar:
            for i,row in enumerate(m['archives'][0]['members']):
                if case=='missing' and i==0:continue
                if case=='symlink' and i==0:
                    item=tarfile.TarInfo(row['member']);item.type=tarfile.SYMTYPE;item.linkname='/tmp/outside';tar.addfile(item)
                else:tar.add(m['files'][row['file']]['path'],arcname=row['member'])
                if case=='duplicate' and i==0:tar.add(m['files'][row['file']]['path'],arcname=row['member'])
        m['files']['archive']=pin(archive)
    with pytest.raises(ValueError):run(tmp_path,m)


def test_missing_spectrum_cannot_hide_present_archive_member(tmp_path,monkeypatch):
    m=fixture(tmp_path,monkeypatch,'A');archive=Path(m['files']['archive']['path'])
    with tarfile.open(archive,'w:gz') as tar:
        for row in m['archives'][0]['members']:tar.add(m['files'][row['file']]['path'],arcname=row['member'])
        item=tarfile.TarInfo('partA/spectra/q0001/fixed-spectrum.json');item.size=2;tar.addfile(item,io.BytesIO(b'{}'))
    m['files']['archive']=pin(archive)
    with pytest.raises(ValueError,match='exists in original archive'):run(tmp_path,m)


def test_changed_inputs_and_new_missing_file_after_admission_refused(tmp_path,monkeypatch):
    m=fixture(tmp_path,monkeypatch,'A');got=run(tmp_path,m);p=tmp_path/'inputs.json';expected=a.digest(p)
    absent=Path(m['roles']['absent_spectra']['1']['path']);absent.parent.mkdir();absent.write_text('{}')
    with pytest.raises(ValueError,match='appeared'):a.recheck(p,expected,got['manifest'],root=tmp_path)


def test_duplicate_json_key_refused(tmp_path,monkeypatch):
    m=fixture(tmp_path,monkeypatch);p=Path(m['files']['producer']['path']);p.write_text('{"n":1,"n":2}');m['files']['producer']=pin(p)
    with pytest.raises(ValueError,match='duplicate JSON'):run(tmp_path,m)


def test_symlink_input_refused(tmp_path,monkeypatch):
    m=fixture(tmp_path,monkeypatch);original=Path(m['files']['producer']['path']);alias=tmp_path/'alias.json';alias.symlink_to(original);m['files']['producer']['path']=str(alias)
    with pytest.raises(ValueError,match='canonical'):run(tmp_path,m)


def test_canonical_spectrum_question_cannot_be_reassigned(tmp_path,monkeypatch):
    m=fixture(tmp_path,monkeypatch,'A');m['archives'][0]['members'][-1]['member']='partA/spectra/q0001/fixed-spectrum.json'
    with pytest.raises(ValueError,match='canonical archive path'):run(tmp_path,m)


def test_extra_spectrum_question_in_original_archive_refused(tmp_path,monkeypatch):
    m=fixture(tmp_path,monkeypatch,'A');archive=Path(m['files']['archive']['path'])
    with tarfile.open(archive,'w:gz') as tar:
        for row in m['archives'][0]['members']:tar.add(m['files'][row['file']]['path'],arcname=row['member'])
        item=tarfile.TarInfo('partA/spectra/q0064/fixed-spectrum.json');item.size=2;tar.addfile(item,io.BytesIO(b'{}'))
    m['files']['archive']=pin(archive)
    with pytest.raises(ValueError,match='extra spectrum'):run(tmp_path,m)


# Amendment 001: invented three-part 200x4 panel, not original Huginn rows.
def three_b_partitions():
    return [[{'index': q, 'arm': arm, 'lam': float(arm), 'split': 'test',
              'steps': 32, 'prompt_style': 'cot8', 'seed': q, 'correct': q % 2 == 0,
              'correct_strict': q % 3 == 0, 'new_tokens': 17, 'strict_pred': None}
             for q in range(start, stop) for arm in ('1.0', '1.05', '1.15', '1.20')]
            for start, stop in ((0, 96), (96, 100), (100, 200))]


def independent_b(rows):
    source = PATH.parent / 'statistics.py'
    definition = importlib.util.spec_from_file_location('partition_test_independent_statistics', source)
    module = importlib.util.module_from_spec(definition); definition.loader.exec_module(module)
    # Small synthetic bootstrap only; actual replay's unchanged default is10000.
    return module.analyze_b(rows, count=200, boot=8)


def test_three_part_archive_admission_and_complete_200_by_four_reducer(tmp_path, monkeypatch):
    partitions = three_b_partitions()
    m = fixture(tmp_path, monkeypatch, partition_rows=partitions)
    got = run(tmp_path, m)
    assert got['rows'] == [row for part in partitions for row in part]
    assert len(got['rows']) == 800
    assert len(got['report']['archive_members']) == 3
    result = independent_b(got['rows'])
    assert result['n'] == 200 and len(result['all_rows_retained']) == 800
    assert {row['index'] for row in result['all_rows_retained']} == set(range(200))
    assert got['report']['certificate_passed'] is False


@pytest.mark.parametrize('case', ['missing_partition', 'missing_cell', 'duplicate_cell', 'extra_cell'])
def test_partition_count_never_replaces_exact_scientific_coverage(tmp_path, monkeypatch, case):
    partitions = three_b_partitions()
    if case == 'missing_partition':
        partitions.pop(1)
    elif case == 'missing_cell':
        partitions[1].pop()
    elif case == 'duplicate_cell':
        partitions[1].append(copy.deepcopy(partitions[0][-1]))
    else:
        row = copy.deepcopy(partitions[2][-1]); row.update(index=200, seed=200)
        partitions[2].append(row)
    m = fixture(tmp_path, monkeypatch, partition_rows=partitions)
    # Archives can be byte-authentic yet scientifically incomplete or aliased.
    got = run(tmp_path, m)
    with pytest.raises(ValueError, match='Cartesian|extra or duplicate'):
        independent_b(got['rows'])


@pytest.mark.parametrize('part,row_ids', [('B', []), ('B', ['rows0', 'rows0']),
    ('B', ['rows0', '']), ('B', ['rows0', 1]), ('B', ['rows0', ['rows1']]),
    ('A', []), ('A', ['rows0', 'rows1'])])
def test_partition_role_list_is_nonempty_unique_typed_and_A_exactly_one(tmp_path, monkeypatch, part, row_ids):
    m = fixture(tmp_path, monkeypatch, part)
    m['roles']['rows'] = row_ids
    with pytest.raises(ValueError, match='partition list'):
        a.validate_manifest(m, part)


def test_three_part_duplicate_physical_files_are_not_unique_partitions(tmp_path, monkeypatch):
    m = fixture(tmp_path, monkeypatch, partition_rows=three_b_partitions())
    m['files']['rows1'] = copy.deepcopy(m['files']['rows0'])
    with pytest.raises(ValueError, match='duplicate physical input path'):
        run(tmp_path, m)
