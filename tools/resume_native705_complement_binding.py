"""Exact immutable pending-cell admission; no mathematical acquisition."""
import hashlib,json
from pathlib import Path
from fractions import Fraction as F
from neural_shrimp.complement_partition import coverage
ROOT=Path(__file__).resolve().parents[1]
OLD=Path('/data2/neural-shrimp-data/explore/eigenpair-locus-prototype/generalized-chain/unit-pair/pal/complement/march002')
PIN='5e91a422b88e3ec61b15caf9f743f31332e1975beda76d65c814b373294ef9c3'
def need(ok,message):
    if not ok:raise ValueError(message)
def entry(path):
    h=hashlib.sha256()
    with path.open('rb')as f:
        while a:=f.read(1024**2):h.update(a)
    return dict(bytes=path.stat().st_size,sha256=h.hexdigest())
def read(p):return json.loads(p.read_text())
def accepted(previous):
    out={i:[]for i in range(7,31)}
    for row in previous['attempts']:
        if row['action']=='accept':out[row['assignment']['ordinal']].append(row['assignment']['cell'])
    return out

def prior(card,*,full):
    r=card['resume'];need(r['root']==str(OLD)and r['outer']['sha256']==PIN and entry(OLD/'manifest.json')==r['outer'],'literal immutable prior48 manifest')
    m=read(OLD/'manifest.json');need(m['files']['launcher-result.json']==r['outcome']and entry(OLD/'launcher-result.json')==r['outcome'],'exact original complete pending decisions')
    need(not OLD.stat().st_mode&0o222,'sealed prior root')
    if full:
        actual={str(p.relative_to(OLD))for p in OLD.rglob('*')if p.is_file()and p!=OLD/'manifest.json'};need(actual==set(m['files']),'complete prior inventory')
        for name,row in m['files'].items():need(entry(OLD/name)==row and not(OLD/name).stat().st_mode&0o222,'exact immutable prior file '+name)
    oldcard=read(OLD/'fixed-index.json');need(card['charts']==oldcard['charts']and card['retained_schema_failure']==oldcard['retained_schema_failure'],'same original24 roots/source and retained first failure')
    need(entry(ROOT/'docs/plans/native705-complement-march-inputs.json')==r['source_index']and oldcard==read(ROOT/'docs/plans/native705-complement-march-inputs.json'),'literal historical index')
    done=read(OLD/'launcher-result.json');need(done['source_commit']==r['source']and done['launched']==48 and len(done['attempts'])==48 and done['failure']is None and not done['passed']and done['complete_charts']==16 and not done['owned_live_pids'],'complete original capped result')
    got=accepted(done)
    for row in done['attempts']:
        a=row['assignment'];owner=row['owner'];out=row['outcome'];need(a['source_commit']==r['source']and a['input_index']==r['source_index']and row['action']in['accept','split']and out['native_calls']==360 and out['target_native_calls']==out['profile_native_calls']==180 and not owner['owned_live_pids']and owner['worker_pid_absent'],'complete same-source original action')
    for row in card['charts']:
        ordinal=row['ordinal'];need(coverage(row['settings']['parameter'],'1/128',got[ordinal])==done['coverage'][str(ordinal)],'recomputed exact accepted coverage')
    expected=[(26,1,1),(25,1,1),(24,1,1),(23,1,1),(27,0,0),(28,0,0),(29,0,0),(30,0,0)]
    need([(o,c['depth'],c['index'])for o,c in done['pending']]==expected,'literal eight pending cells only')
    return done

def admissible(card,job):
    # Worker authenticates compact predecessor decisions; supervisor additionally
    # hashes the entire sealed9.18GB inventory before allocating any worker.
    previous=prior(card,full=False);c=job['cell'];lo,hi=map(F,c['endpoints'])
    need(any(o==job['ordinal']and F(p['endpoints'][0])<=lo<hi<=F(p['endpoints'][1])for o,p in previous['pending']),'only a pending cell or strict descendant; no accepted-cell reacquisition')
