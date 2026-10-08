"""Summarize immutable S1R pilot records; no model/map/Jacobian evaluation."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from neural_shrimp.period2_branch_io import canonical,file_entry,load_record


def summarize(raw,output):
    manifest=json.loads((raw/'manifest.json').read_text())
    assert set(p.name for p in raw.iterdir())==set(manifest['files'])|{'manifest.json'}
    assert all(file_entry(raw/name)==entry for name,entry in manifest['files'].items())
    runtime=json.loads((raw/'runtime-start.json').read_text());result=json.loads((raw/'result.json').read_text())
    assert runtime['source_commit']=='db10bd1e1b25b0e24b1f0f8f97cb84035742c776'
    assert all(file_entry(ROOT/name)==entry for name,entry in runtime['sources'].items())
    log=raw.parent/(raw.name+'.checker.log');checker=json.loads(log.read_text())
    assert checker['status']=='passed_saved_state_replay'
    nodes={};edges=[]
    for line in (raw/'journal.jsonl').read_text().splitlines():
        event=json.loads(line);kind,payload=load_record(raw,event['file'])
        if kind=='node':nodes[payload['id']]=payload
        if kind=='edge':edges.append(payload)
    qualified=[n for n in nodes.values() if n['accepted']]
    assert len(qualified)==checker['qualified_pairs']
    def matched_cost(m):return m['costs'][m['phase']]
    all_spectra=[s for n in qualified for s in n['spectra']['phases']]
    main_ids=[result['hidden_links'][-1]['child']]+[s['endpoint'] for s in result['segments'] if s['status']=='connected']
    receipt=dict(schema='s1r-local-branch-receipt-v1',evidence_rung=1,status=result['status'],
        execution_source_commit=runtime['source_commit'],source_review_commit='cdf2ad51034db515fab99846fbd88e61a4528a46',
        summary_source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        summary_source_sha256=file_entry(Path(__file__))['sha256'],external_root=str(raw),
        raw_files={name:file_entry(raw/name) for name in ('manifest.json','runtime-start.json','result.json','journal.jsonl','checkpoint.json')},
        raw_file_count=len(manifest['files'])+1,raw_total_bytes=sum(p.stat().st_size for p in raw.iterdir()),
        launcher=dict(path=str(raw.parent/(raw.name+'.launcher.log')),**file_entry(raw.parent/(raw.name+'.launcher.log'))),
        checker=dict(path=str(log),**file_entry(log),result=checker),
        preparation_failure=dict(path=str(raw.parent/'launch-preparation-failure-001.json'),**file_entry(raw.parent/'launch-preparation-failure-001.json')),
        source_inventory=runtime['sources'],input_inventory=runtime['inputs'],lineage=result['lineage'],
        historical_family_association=result['historical_family_association'],historical_families=result['historical_families'],
        runtime={k:result[k] for k in ('elapsed_seconds','max_rss_kib','phase_corrector_calls')},
        pid=runtime['pid'],nice=runtime['nice'],threads=runtime['cpu_threads'],limits=runtime['limits'],
        qualified_pairs=len(qualified),accepted_output_links=sum(e['status']=='accepted' for e in edges),
        rejected_edges=sum(e['status']=='rejected' for e in edges),
        maxima=dict(normalized_closure=max(max(n['checks']['closure']) for n in qualified),
          normalized_transition=max(max(n['checks']['transitions']) for n in qualified),
          eigen_residual=max(max(np.max(s['left_residual']),np.max(s['right_residual'])) for s in all_spectra),
          phase_spectrum_multiset_error=max(n['spectra']['multiset_error'] for n in qualified),
          independent_phase_agreement=max(matched_cost(n['independent_phase_match']) for n in qualified),
          direct_half_agreement=max(matched_cost(e['direct_half_match']) for e in edges if e['status']=='accepted'),
          reverse_agreement=max(matched_cost(e['reverse_match']) for e in edges if e['status']=='accepted'),
          output_parent_displacement=max(matched_cost(e['parent_match']) for e in edges if e['status']=='accepted')),
        minimum_phase_separation=min(n['checks']['separation'] for n in qualified),
        main_points=[dict(node_id=key,log10_rates=nodes[key]['params'].tolist(),
               spectral_radii=[s['radius'] for s in nodes[key]['spectra']['phases']],
               stability=[s['stability'] for s in nodes[key]['spectra']['phases']]) for key in main_ids],
        scope='local numerical branch consistency at tested step sizes; no uniqueness, historical-family association, distant ancestry, invariant circle or utility')
    output.mkdir(parents=True,exist_ok=False)
    with (output/'receipt.json').open('xb') as h:h.write(canonical(receipt))
    with (output/'manifest.json').open('xb') as h:h.write(canonical({'receipt.json':file_entry(output/'receipt.json')}))
    for p in output.iterdir():p.chmod(0o444)
    output.chmod(0o555)
    print(canonical({k:receipt[k] for k in ('status','runtime','raw_total_bytes','maxima','minimum_phase_separation','main_points')}).decode())


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('raw',type=Path);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();summarize(args.raw,args.out)
