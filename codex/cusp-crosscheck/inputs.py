"""Literal Task14 candidate metadata. This module never loads state arrays."""
import hashlib
import json
import math
from pathlib import Path
import stat

AUTHORITY=Path('/data2/neural-shrimp-data/codex/llm-cusp-crosscheck/input-authority-001')
AUTHORITY_SHA='1bb981cc55cbf617069ab09c1dbc23a1d9b79e39394c4e6e5b2863d3ec5866e4'
PINS={'states.npy':'35bdd25867f9cfba51cc10722f4d56be18f234c81cfd472e6595a8829b728bd0',
      'curve.json':'0bf6f562b6343e21e7fe35a55c109407d92e0a1cc02983607f8ea28ce408e321'}
CANDIDATES={13:dict(log10_temperature=.1893809804983369,alpha=.06661104169386665),
            14:dict(log10_temperature=.1893794755174379,alpha=.06660969640358283)}
ANCESTRY=Path('/data2/neural-shrimp-data/codex/llm-cusp-crosscheck/map-ancestry-001')
ANCESTRY_SHA='89b9a8ac8254235b12f2fb6b8b8960a23d724b9be48ac43f4b05529544b1fac0'
RECIPE=dict(model='qwen0.5b',layer=12,prompt='open',opener='Hello!',window=1,
            vectors='llm-phase-map/vectors-l12-p1b',concept='hostility-vs-kindness')
PERIOD=7
DIMENSION=896


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024**2),b''):digest.update(chunk)
    return digest.hexdigest()


def metadata(curve):
    if curve.get('period')!=PERIOD or curve.get('kind')!='fold' or curve['args']['alpha_scale']!=10. or len(curve['points'])!=40:
        raise ValueError('literal period7/40-row fold metadata required')
    selected={}
    for index,wanted in CANDIDATES.items():
        row=curve['points'][index]
        if row['index']!=index or any(row[k]!=v for k,v in wanted.items()):
            raise ValueError('exact row-to-parameter association differs')
        if row['theta'][0]!=wanted['log10_temperature'] or not math.isclose(row['theta'][1]/10.,wanted['alpha'],rel_tol=0,abs_tol=2e-17):
            raise ValueError('scaled theta and physical alpha disagree')
        selected[str(index)]=dict(wanted,theta=row['theta'])
    return selected


def ancestry_metadata(branch,report):
    if branch.get('period')!=PERIOD or any(branch['args'].get(k)!=v for k,v in RECIPE.items()):
        raise ValueError('producer branch map recipe differs')
    if report['declared_map']!=RECIPE or report['period']!=PERIOD or report['alpha_scale']!=10. or report['prefix_decodes_to_expected_prompt'] is not True:
        raise ValueError('reviewed map recipe differs')
    return dict(declared_map=RECIPE.copy(),prefix_ids=report['prefix_ids'],
                actual_checkpoint_sha256=report['independent_checkpoint_sha256'],
                vector_sha256=report['current_vector_sha256'],limitations=report['limits'])


def authenticate():
    if sha(AUTHORITY/'manifest.json')!=AUTHORITY_SHA:
        raise ValueError('immutable candidate authority differs')
    manifest=json.loads((AUTHORITY/'manifest.json').read_text())
    records={row['name']:row for row in manifest['files']}
    for name,pin in PINS.items():
        path=AUTHORITY/name
        if path.stat().st_mode & (stat.S_IWUSR|stat.S_IWGRP|stat.S_IWOTH):
            raise ValueError('candidate copy is writable')
        if records[name]['sha256']!=pin or path.stat().st_size!=records[name]['bytes'] or sha(path)!=pin or manifest['same_bytes_in_returned_archive'][name]['sha256']!=pin:
            raise ValueError('candidate source/archive bytes differ')
    if sha(ANCESTRY/'manifest.json')!=ANCESTRY_SHA:raise ValueError('map ancestry authority differs')
    ancestry=json.loads((ANCESTRY/'manifest.json').read_text())
    if ancestry['prior_candidate_manifest_sha256']!=AUTHORITY_SHA or ancestry['archive_sha256']!=manifest['archive_sha256']:
        raise ValueError('ancestry/candidate archive differs')
    for row in ancestry['files']:
        path=ANCESTRY/row['name']
        if path.stat().st_mode & 0o222 or path.stat().st_size!=row['bytes'] or sha(path)!=row['sha256']:
            raise ValueError('immutable ancestry file differs')
    branch=json.loads((ANCESTRY/'branch.json').read_text())
    report=json.loads((ANCESTRY/'task14-map-ancestry-check.json').read_text())
    recipe=ancestry_metadata(branch,report)
    if json.loads((AUTHORITY/'curve.json').read_text())['args']['start']!=branch['args']['out']:
        raise ValueError('curve does not inherit the authenticated branch')
    return dict(authority_sha256=AUTHORITY_SHA,ancestry_manifest_sha256=ANCESTRY_SHA,ancestry=recipe,inputs=PINS.copy(),
                selected=metadata(json.loads((AUTHORITY/'curve.json').read_text())),
                producer_source=manifest['source_commit'],array_values_decoded=False)
