"""Original command selects the captured plan; no model or target evidence."""
from pathlib import Path

import pytest

from test_codex_llm_derivative_evidence import full, collect, read, seal, write
import derivative_evidence as ev


def duplicate_plan(full):
    attempt=Path(full['attempts'][0]['attempt'])
    original=attempt/'source/codex/certify-llm/test-plan.json'
    selected=original.with_name('test-plan-recovery.json')
    selected.write_bytes(original.read_bytes())
    manifest=read(attempt/'manifest.json')
    manifest['command'][8]=str(selected)
    write(attempt/'manifest.json',manifest)
    full['attempts'][0]=seal(attempt)
    return attempt,original,selected


def test_identical_retained_plans_admit_only_original_command_selected_file(full,monkeypatch):
    attempt,original,selected=duplicate_plan(full)
    actual=ev.producer.read_json;plans_read=[]
    def tracking(path):
        if Path(path) in (original,selected):plans_read.append(Path(path))
        return actual(path)
    monkeypatch.setattr(ev.producer,'read_json',tracking)
    matrices,report=collect(full)
    assert report['coverage']['coverage_complete'] and len(matrices)==2
    assert plans_read==[selected]
    assert original.read_bytes()==selected.read_bytes()


@pytest.mark.parametrize('fault',[
    'missing_selected','unlisted_selected','missing_command','short_command',
    'nonstring_argument','wrong_flag','outside','wrong_source_directory','relative',
    'parent_traversal','non_json','wrong_hash','semantic_mismatch','wrong_driver',
])
def test_duplicate_hash_never_substitutes_for_bad_command_plan(full,fault):
    attempt,original,selected=duplicate_plan(full)
    manifest=read(attempt/'manifest.json')
    if fault=='missing_selected':selected.unlink()
    elif fault=='unlisted_selected':
        # Remove the selected file from both inventories; the unused identical
        # plan remains authenticated, but is not a replacement for this path.
        manifest['source']=[r for r in manifest['source'] if r['path']!=str(selected.relative_to(attempt/'source'))]
        write(attempt/'manifest.json',manifest)
        receipt=read(attempt/'receipt.json')
        receipt['artifacts']=[r for r in receipt['artifacts'] if r['path']!=str(selected.relative_to(attempt))]
        for row in receipt['artifacts']:
            if row['path']=='manifest.json':row.update(bytes=(attempt/'manifest.json').stat().st_size,sha256=ev.digest(attempt/'manifest.json'))
        write(attempt/'receipt.json',receipt)
        full['attempts'][0]['receipt_sha256']=ev.digest(attempt/'receipt.json')
        with pytest.raises(ValueError):collect(full)
        return
    elif fault=='missing_command':manifest.pop('command')
    elif fault=='short_command':manifest['command'].pop()
    elif fault=='nonstring_argument':manifest['command'][8]=None
    elif fault=='wrong_flag':manifest['command'][7]='--other-plan'
    elif fault=='outside':
        outside=attempt.parent/'outside.json';outside.write_bytes(original.read_bytes());manifest['command'][8]=str(outside)
    elif fault=='wrong_source_directory':
        other=attempt/'source/elsewhere.json';other.write_bytes(original.read_bytes());manifest['command'][8]=str(other)
    elif fault=='relative':manifest['command'][8]=str(selected.relative_to(attempt))
    elif fault=='parent_traversal':manifest['command'][8]=str(selected.parent)+'/../certify-llm/'+selected.name
    elif fault=='non_json':
        other=selected.with_suffix('.txt');other.write_bytes(selected.read_bytes());manifest['command'][8]=str(other)
    elif fault=='wrong_hash':selected.write_bytes(selected.read_bytes()+b'\n')
    elif fault=='semantic_mismatch':
        changed=read(selected);changed['start_column']=2;write(selected,changed)
        inputs=read(attempt/'run/inputs.json');inputs['plan_sha256']=ev.digest(selected);write(attempt/'run/inputs.json',inputs)
    elif fault=='wrong_driver':manifest['command'][4]=str(attempt/'source/codex/certify-llm/not-the-driver.py')
    else:raise AssertionError(fault)
    write(attempt/'manifest.json',manifest)
    full['attempts'][0]=seal(attempt)
    with pytest.raises(ValueError):collect(full)
