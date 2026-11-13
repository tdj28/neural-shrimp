"""Bind the completed FGR2 studies and their independently audited conclusions."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
from fold_global_return import sha,write_json,backup_file
from fold_return_curve import render as render_curve
from audit_fold_return_cycle_map import render as render_cycle


def finalize(args):
    line,curve,cycle,out=map(Path,(args.line,args.curve,args.cycle,args.out))
    out.mkdir(parents=True,exist_ok=False)
    audits=[json.loads((p/'audit.json').read_text()) for p in (line,curve,cycle)]
    a,c,d=audits
    assert a['hashes_verified_on_separate_disks'] and c['hash_verified_on_separate_disks'] and d['all_manifest_hashes_verified_on_separate_disks']
    assert sum(a['counts'].values())==64 and a['batch_iterations']==4096
    assert c['test_pairs']==31 and d['heldout_returning_pairs']==c['returned_test_pairs']
    assert d['excluded_P3_pairs']==c['original_P3_test_pairs']
    bindings={}
    for label,path,audit in zip(('line','scalar_excursion','complete_cycle'),(line,curve,cycle),audits):
        run,backup=Path(audit['raw_path']),Path(audit['backup_path'])
        assert sha(run/'manifest.json')==sha(backup/'manifest.json')==audit['manifest_sha256']
        assert sha(path/'audit.json')==sha(backup/'audit/audit.json')
        bindings[label]={'audit_path':str(path),'audit_sha256':sha(path/'audit.json'),
                         'raw_path':str(run),'manifest_sha256':audit['manifest_sha256'],
                         'plan_sha256':audit['plan_sha256'],'backup_path':str(backup),
                         'execution_source':audit['execution_source']}
    rows=json.loads((line/'rows.json').read_text());captured=[r for r in rows if r['kind']=='reference_p3']
    unresolved=[r for r in rows if r['kind']=='unresolved_at_horizon']
    line_summary={k:a[k] for k in ('counts','returned_at_least_once','returned_then_P3','total_channel_visits','natural_exits','wall_seconds','retained_valid_updates','initial_line_prediction')}
    line_summary.update(capture_detection_updates_range=[min(r['valid_updates'] for r in captured),max(r['valid_updates'] for r in captured)] if captured else None,
                        unresolved_visit_range=[min(r['visit_count'] for r in unresolved),max(r['visit_count'] for r in unresolved)] if unresolved else None,
                        captured_visit_range=[min(r['visit_count'] for r in captured),max(r['visit_count'] for r in captured)] if captured else None)
    prediction={k:v for k,v in a['prediction'].items() if k not in ('observations','predictions','test_event_ids')}
    events=json.loads((line/'events.json').read_text())
    replay=[]
    for row in json.loads((cycle/'rows.json').read_text()):
        if row['id']%2:continue
        expected_update=row['parent_update']+row['departure_to_departure_updates']
        matches=[e for e in events if e['row']==row['parent_row'] and e['exit']==expected_update]
        assert len(matches)==1,('full-cycle replay timing',row['parent_row'],expected_update)
        error=abs(matches[0]['exit_v_over_U']-row['next_departure_v_over_U'])
        assert error<1e-6,('full-cycle saved-state coordinate replay',row['parent_row'],error)
        replay.append(error)
    run=Path(c['raw_path']);plan=json.loads((run/'plan.json').read_text())
    with np.load(run/'inputs.npz') as z:inputs={k:z[k] for k in z.files}
    render_curve(out,inputs,json.loads((curve/'rows.json').read_text()),json.loads((curve/'pairs.json').read_text()),c,plan['width'])
    # Preserve scientific figures rather than regenerate or modify the audited originals.
    import shutil
    for source in (line/'repeated-returns.png',line/'natural-return-prediction.png'):
        shutil.copyfile(source,out/source.name)
    render_cycle(out,json.loads((cycle/'pairs.json').read_text()),d,
                 background=events)
    receipt={'status':'completed exploratory local follow-up; bounded numerical conclusions',
             'source':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
             'model':'Qwen/Qwen2.5-0.5B-Instruct','source_parameters':plan['source'],
             'epsilon':plan['epsilon'],'U':plan['width'],'bindings':bindings,
             'line_scan':line_summary,'simple_prediction_comparisons':prediction,
             'scalar_reconstruction':{k:v for k,v in c.items() if k not in ('parent_replay_files','raw_path','backup_path','replay_gate')},
             'complete_return_cycle':{k:v for k,v in d.items() if k not in ('raw_path','backup_path','replay_gate')},
             'full_cycle_replay_against_original_line':{'pairs':len(replay),'all_departure_times_match':True,'max_departure_coordinate_difference_U':max(replay)},
             'interpretation':[
                 'Arbitrary transverse changes at fixed slow coordinate can change the next destination (FGR1).',
                 'After a sufficiently long slow passage, sampled natural exit states admit an accurate one-coordinate reconstruction.',
                 'The reconstructed full state predicts the tested next excursion and complete return cycle when evolved through the actual network.',
                 'The global return function remains nonlinear and is evaluated by the network, not derived solely from local fold coefficients.',
                 'A return is compatible with later approach to P3; the finite-horizon scan does not establish permanent recurrence or chaos.'],
             'limits':['One fixed model, prompt, steering direction and parameter setting.',
                       'Exploratory adaptive analyses with complete initial-state trajectories held out; not a confirmatory population study.',
                       'The line sample is not a basin-volume or escape-probability estimate.',
                       'No rigorous error enclosure, global basin-boundary proof, or guaranteed indefinite scalar iteration.'],
             'storage':{'primary_physical_disk':'nvme0n1','independent_backup_physical_disk':'sda','raw_and_figures_outside_git':True},
             'figures':{p.name:{'path':str(p),'sha256':sha(p),'bytes':p.stat().st_size} for p in out.glob('*.png')}}
    write_json(out/'receipt.json',receipt)
    backup=Path(a['backup_path']).parent/'closeout'
    for path in out.iterdir():backup_file(path,out,backup)
    write_json(ROOT/'docs/fold-return-mechanism.json',receipt)
    print(json.dumps({'line':line_summary,'excursion_max_error_U':c['max_return_v_over_U_error'],
                      'cycle_max_error_U':d['max_next_departure_v_over_U_error'],'out':str(out)},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('line','curve','cycle','out'):p.add_argument('--'+k,required=True)
    finalize(p.parse_args())
