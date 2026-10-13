"""Small post-replay comparison. Every predeclared row and discrepancy is saved."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import compare

HERE=Path(__file__).resolve().parent
RAW=Path('/data2/neural-shrimp-data/codex/llm-misalignment-audit')
PUBLISHED=Path('/data2/neural-shrimp-data/explore/llm-msm')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['descriptive','l7','l8'],required=True)
    ap.add_argument('--attempt',choices=['001','002'],required=True)
    ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    directory=RAW/(a.mode+'-'+a.attempt)
    result=json.loads((directory/'result.json').read_text())
    if not result['completed']:raise ValueError('incomplete replay cannot be summarized as complete')
    registry=json.loads((HERE/'claims.json').read_text())
    reference_pins=json.loads((HERE/'comparison-inputs.json').read_text())
    if a.mode=='descriptive':
        own=json.loads((directory/'descriptive.json').read_text())
        pubpath=PUBLISHED/'p1-controller/controller.json'
        if hashlib.sha256(pubpath.read_bytes()).hexdigest()!=reference_pins[str(pubpath)]:raise ValueError('published reference changed')
        claims=compare.descriptive_claims(own,registry)
        machine=compare.controller_reference(own,json.loads(pubpath.read_text()))
        extra={'seed_share_differences':{p:own['capture'][p]['share']-own['capture'][p+'s2']['share']
                for p in ['p1-h050','p1-h060']},
               'release_seed_and_pooled_rows':'descriptive.json retains every per-turn row and seed result',
               'controller_anti_hostility_efficiencies':{p:own['controller'][p]['efficiency'] for p in
                   ['p1-ctl-trig-cancel','p1-ctl-trig-over','p1-ctl-trig-long']}}
    else:
        experiment=a.mode.upper();own=compare.detector_rows(directory)
        expected=30 if a.mode=='l7' else 18
        if len(own)!=expected:raise ValueError('incomplete fixed detection row set')
        pubpath=PUBLISHED/('p1-detection-corrected/detection.json' if a.mode=='l7' else
                           'p1-detection-confirmation-amended2/confirmation.json')
        if hashlib.sha256(pubpath.read_bytes()).hexdigest()!=reference_pins[str(pubpath)]:raise ValueError('published reference changed')
        claims=compare.detection_claims(own,registry,experiment)
        machine=compare.detection_reference(own,json.loads(pubpath.read_text()),experiment)
        extra={}
        if a.mode=='l7':
            extra['operating_point']=compare.l7_oof_operating_point(directory/'event1-h3-scores.npz')
            for key,expected in [('baseline',.23),('D3_state_probe',.27)]:
                claims.append(compare.check('L7.c1.h3.OOFflag.'+key,
                    extra['operating_point']['models'][key]['flagged_positive'],expected,.005,
                    'paper/llm/main.tex:865-866','descriptive OOF negative quantile; exact historical threshold provenance not located'))
    out={'mode':a.mode,'replay_result_sha256':hashlib.sha256((directory/'result.json').read_bytes()).hexdigest(),
         'published_path':str(pubpath),'published_sha256':hashlib.sha256(pubpath.read_bytes()).hexdigest(),
         'claims_registry_sha256':hashlib.sha256((HERE/'claims.json').read_bytes()).hexdigest(),
         'manuscript_comparisons':claims,'machine_readable_comparisons':machine,'extra':extra,
         'manuscript_discrepancies':[c for c in claims if not c['within_rounding']],
         'machine_discrepancies':[c for c in machine if not c['within_rounding']]}
    with a.output.open('x') as f:json.dump(out,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({'mode':a.mode,'claims':len(claims),'manuscript_discrepancies':len(out['manuscript_discrepancies']),
                      'machine_discrepancies':len(out['machine_discrepancies'])}))


if __name__=='__main__':main()
