"""Own fixed RM1 saved-data adapter. No producer/model/PCA execution."""
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import resource
import time
import traceback
import zipfile

HERE=Path(__file__).resolve().parent
SPEC=importlib.util.spec_from_file_location('own_rm_core',HERE/'core.py')
C=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(C)
np=C.np
INPUT_SHA='6145d0edf686f3465f9aef396642c4007ea7a79290da22d646d8f8ddbd33962a'
CORE_SHA='01e4ba536a2aecfeebbf1398a0c8f16d4259b26bf6c7eb27bdd05ef11c1a98a0'
ORIGINAL='0a15d1a66a4e9d34ff43d1d56e8cb42c5126fc9e'
OUTPUT='/data2/neural-shrimp-data/codex/night-results-review-task21/task21g/run-001'
GROUPS={'body_edge':[0,1,2],'hostile_line':[3,4,5,6],'original_figure_rows':[0,1,2,5]}
LIMITS={'cpu_threads':1,'nice':10,'as_bytes':1073741824,'cpu_seconds':55,'wall_seconds':60,
        'max_output_bytes':134217728,'max_input_bytes':8388608,'startup_available_gib':20,'runtime_available_gib':16}


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def authenticate(pin):
    p=Path(pin['path'])
    if p.stat().st_size!=pin['bytes'] or sha(p)!=pin['sha256']:raise ValueError('input/source hash mismatch: '+str(p))


def pins_in(v):
    if isinstance(v,dict):
        if {'path','sha256','bytes'}<=v.keys():yield {k:v[k] for k in ('path','sha256','bytes')}
        else:
            for x in v.values():yield from pins_in(x)
    elif isinstance(v,list):
        for x in v:yield from pins_in(x)


class Sink:
    def __init__(self,out,limit):self.out=Path(out);self.limit=limit;self.bytes=0;self.entries={}
    def write(self,name,value,append=False):
        data=(json.dumps(C.plain(value),allow_nan=False,separators=(',',':'))+'\n').encode()
        if self.bytes+len(data)>self.limit:raise RuntimeError('declared output ceiling')
        if '/' in name or name.startswith('.'):raise ValueError('output basename')
        mode='ab' if append and name in self.entries else 'xb'
        with open(self.out/name,mode) as f:f.write(data)
        self.bytes+=len(data);self.entries[name]=self.entries.get(name,0)+1
    def manifest(self):
        return [{'path':k,'bytes':(self.out/k).stat().st_size,'sha256':sha(self.out/k),'records':n} for k,n in sorted(self.entries.items())]


def compare(own,saved):
    if own is None:return {'status':'undefined_own','own':None,'saved':saved,'signed_difference':None,'exact':None}
    finite=math.isfinite(float(own)) and math.isfinite(float(saved))
    return {'status':'finite' if finite else 'nonfinite','own':own,'saved':saved,
            'signed_difference':float(own)-float(saved) if finite else None,
            'exact':bool(own==saved),'both_nan':math.isnan(float(own)) and math.isnan(float(saved))}


def load_inputs(packet):
    if packet['attempt']!='RM1' or packet['source_commit']!=ORIGINAL:raise ValueError('original attempt/source')
    if (packet['requested_cell_count'],packet['phase_count'],packet['expected_sequence_count'])!=(7,7,49):raise ValueError('fixed cell/phase counts')
    if packet['group_memberships']!=GROUPS:raise ValueError('frozen claim groups')
    meta=packet['metadata'];spec={x['member'][:-4]:x for x in meta['npz_members']}
    keys={'cells','period'}|{f'c{c}_p{p}' for c in range(7) for p in range(7)}
    if len(spec)!=51 or len(meta['npz_members'])!=51 or set(spec)!=keys:raise ValueError('exact member inventory')
    path=packet['files']['return-map.npz']['path']
    with zipfile.ZipFile(path) as z:
        names=z.namelist()
        if len(names)!=len(set(names)) or set(names)!={k+'.npy' for k in keys}:raise ValueError('NPZ member inventory')
    with np.load(path,allow_pickle=False) as z:
        arrays={}
        for k,s in spec.items():
            a=z[k]
            if a.dtype.hasobject or a.dtype.str!=s['dtype'] or list(a.shape)!=s['shape'] or a.nbytes!=s['array_nbytes']:raise ValueError('array schema '+k)
            arrays[k]=a.copy()
    coords=np.asarray(meta['coordinate_and_period_metadata']['cells'],dtype=np.float64)
    if coords.shape!=(7,2) or not np.isfinite(coords).all() or not np.array_equal(arrays['cells'],coords):raise ValueError('cell coordinates/order')
    if arrays['period'].shape!=() or arrays['period'].dtype.kind not in 'iu' or int(arrays['period'])!=7:raise ValueError('nominal period')
    for k in keys-{'cells','period'}:
        if arrays[k].shape!=(1000,) or arrays[k].dtype!=np.dtype('float64'):raise ValueError('fixed scalar sequence')
    saved=json.loads(Path(packet['files']['return-map.json']['path']).read_text())
    if set(saved)!={'args','cells','seconds'} or saved['args']!=meta['args'] or len(saved['cells'])!=7:raise ValueError('summary schema/args/count')
    for c,rec in enumerate(saved['cells']):
        if set(rec)!={'alpha','log10_temperature','phases'} or [rec['alpha'],rec['log10_temperature']]!=coords[c].tolist() or len(rec['phases'])!=7:raise ValueError('summary cell association')
        for p,row in enumerate(rec['phases']):
            if set(row)!={'phase','turning_points','explained_1','explained_2','spread'} or type(row['phase'])!=int or row['phase']!=p or type(row['turning_points'])!=int:raise ValueError('summary phase schema')
            if any(type(row[k]) not in (float,int) for k in ('explained_1','explained_2','spread')):raise ValueError('summary numeric schema')
    return arrays,saved


def compact_partition(part):
    out={k:v for k,v in part.items() if k not in ('rows','runs')}
    if part['status']!='complete':return out
    out.update(occupied_bins=sum(x['count']>0 for x in part['rows']),singleton_bins=sum(x['count']==1 for x in part['rows']),
               total_pairs=sum(x['count'] for x in part['rows']),
               parity_counts_by_bin=[{'bin':x['bin'],'count':x['count'],
                                     'parity_counts':[p['y']['count'] for p in x.get('parity',[])],
                                     'odd_minus_even_median_y':x.get('odd_minus_even_median_y')} for x in part['rows']],
               runs=part['runs'])
    return out


def calculate(packet,sink):
    arrays,saved=load_inputs(packet);summaries=[]
    for c in range(7):
        for p in range(7):
            seq=arrays[f'c{c}_p{p}'];own=C.analyze_sequence(seq);producer=saved['cells'][c]['phases'][p]
            with np.errstate(all='ignore'):spread=float(np.ptp(seq))
            literal=own['literal_producer_count'];turn=literal['count'] if literal['status']=='complete' else None
            identity={'cell':c,'phase':p,'coordinate':arrays['cells'][c],
                      'groups':[name for name,ids in GROUPS.items() if c in ids]}
            comparisons={'literal_turning_count':compare(turn,producer['turning_points']),
                         'sequence_spread':compare(spread,producer['spread'])}
            inherited={'explained_1':producer['explained_1'],'explained_2':producer['explained_2'],
                       'both_finite':all(math.isfinite(producer[k]) for k in ('explained_1','explained_2')),
                       'scope':'producer PCA shares inherited; native states/directions/means/singular values unavailable, no PCA recalculation'}
            sink.write('phases.jsonl',{**identity,'sequence':seq,'producer_summary':producer,'comparisons':comparisons,
                                       'inherited_pca':inherited,'own':own},True)
            summaries.append({**identity,'producer_summary':producer,'comparisons':comparisons,'inherited_pca':inherited,
                              'scalar_scope':{k:v for k,v in own.items() if k not in ('partitions','literal_producer_count')},
                              'literal_counter':{k:v for k,v in literal.items() if k!='rows'},
                              'partitions':[compact_partition(q) for q in own['partitions']]})
    return {'schema':'codex-task21g-saved-return-maps-v1','computational_completed':True,
            'scientific_claims_accepted':False,'original_attempt':'RM1','original_source_commit':ORIGINAL,
            'cells':7,'phases_per_cell':7,'sequences':49,'states_per_sequence':1000,'pairs_per_sequence':999,
            'partition_count_per_sequence':10,'group_memberships':GROUPS,'original_args':saved['args'],
            'producer_elapsed_seconds_inherited':saved['seconds'],'all_phase_summaries':summaries,
            'comparison_counts':{k:{'total':49,'finite':sum(r['comparisons'][k]['status']=='finite' for r in summaries),
                                    'exact':sum(r['comparisons'][k]['exact'] is True for r in summaries),
                                    'nonexact_finite':sum(r['comparisons'][k]['status']=='finite' and r['comparisons'][k]['exact'] is False for r in summaries),
                                    'undefined_or_nonfinite':sum(r['comparisons'][k]['status']!='finite' for r in summaries)} for k in ('literal_turning_count','sequence_spread')},
            'no_selected_prominence_or_unimodality_gate':True,
            'scope':'Finite scalar projection geometry only; nominal phase clock, PCA and native dimension/conjugacy are not reconstructed. Producer result exposure disclosed.'}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--card',required=True);ap.add_argument('--out',required=True);args=ap.parse_args()
    card=json.loads(Path(args.card).read_text());out=Path(args.out)
    if card.get('schema')!='codex-task21g-card-v1' or card.get('admission')!='reviewed' or card.get('limits')!=LIMITS or str(out)!=OUTPUT or card.get('output')!=OUTPUT:raise ValueError('prepared/unrecognized card/output refuses execution')
    if card['inputs']['sha256']!=INPUT_SHA or sha(HERE/'core.py')!=CORE_SHA:raise ValueError('fixed metadata/core association')
    resource.setrlimit(resource.RLIMIT_AS,(LIMITS['as_bytes'],LIMITS['as_bytes']));resource.setrlimit(resource.RLIMIT_CPU,(LIMITS['cpu_seconds'],LIMITS['cpu_seconds']))
    nice=os.getpriority(os.PRIO_PROCESS,0)
    if nice>10:raise ValueError('nice exceeds10')
    if nice<10:os.nice(10-nice)
    out.mkdir(exist_ok=False);sink=Sink(out,LIMITS['max_output_bytes']);start=time.perf_counter()
    try:
        authenticate(card['inputs'])
        if len(card['sources'])!=2 or {p['name'] for p in card['sources']}!={'core.py','run.py'}:raise ValueError('own source inventory')
        for p in card['sources']:
            if sha(HERE/p['name'])!=p['sha256']:raise ValueError('own source changed')
        packet=json.loads(Path(card['inputs']['path']).read_text())
        for k in ('authority','source_map'):authenticate(packet[k])
        source_map=json.loads(Path(packet['source_map']['path']).read_text())
        pins=list(pins_in(packet))+list(pins_in(source_map))+[card['inputs']];unique={}
        for p in pins:
            if p['path'] in unique and unique[p['path']]!=p:raise ValueError('conflicting input pin')
            unique[p['path']]=p
        if sum(x['bytes'] for x in unique.values())>LIMITS['max_input_bytes']:raise ValueError('input byte cap')
        for p in unique.values():authenticate(p)
        report=calculate(packet,sink)
        for p in unique.values():authenticate(p)
        report.update(input_authentication=list(unique.values()),actual_source_pins=card['sources'],card_sha256=sha(args.card),
                      operational_scope='original archive/receipt/recorded termination inherited from accepted coordinator authority; own selected byte hashes before/after',
                      resource_observation={'seconds_through_posthash_before_final_report':time.perf_counter()-start,
                                            'peak_rss_kib_before_final_report_serialization':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                                            'scope':'includes input IO, calculations and streamed phase writes; excludes final report/manifest serialization/write; external supervisor observes full worker wall, no separate full-peak claim'})
        sink.write('report.json',report);sink.write('manifest.json',{'schema':'codex-task21g-output-v1','computational_completed':True,'files':sink.manifest()})
    except BaseException:
        with open(out/'failure.json','x') as f:json.dump({'computational_completed':False,'error':traceback.format_exc(),'source':sha(__file__)},f,indent=2)
        raise


if __name__=='__main__':main()
