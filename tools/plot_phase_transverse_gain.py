#!/usr/bin/env python3
"""Limited independent saved-output census and figures; no gain recomputation."""
from __future__ import annotations
import argparse
from collections import Counter
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1] if Path(__file__).resolve().parent.name=='tools' else Path('/data2/neural-shrimp')
SETTINGS=('core','failed-near-edge');DIRECTIONS=('phase_write','phase_null','neuron_one');DELAYS=(1,2,4,8);NOISES=(1e-5,1e-4,1e-3)
COMPONENTS=('total','phase_algebraic','transverse','phase_physical','phase_clock','transport_defect','clock_flow_defect','initial_time_shift','clock_normalization_defect','ideal_accounting_residual')
FLAGS=('qualified','qualified_algebraic','qualified_ideal_neutral')
SCOPE=('Limited saved-output census, packet/receipt integrity and visible scalar-normalization checks; '
       'no independent propagator solve or gain replay. Conditional linear reference predictions, '
       'not observed task accuracy or finite-pulse qualification. No setting means, direction selection or original-study requalification.')


def read(path):return json.loads(Path(path).read_text())
def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()
def binding(path,root):return dict(path=str(Path(path).relative_to(root)),bytes=Path(path).stat().st_size,sha256=sha(path))
def finite(v):
    if type(v) not in (int,float) or not math.isfinite(v):raise ValueError('Finite nonboolean scalar required')
    return float(v)
def typed_bool(v):
    if type(v) is not bool:raise ValueError('Strict Boolean required')
    return v

def cell_census():
    return [dict(id=f'{s}--p{p}--{d}--k{k}',setting=s,phase_index=p,direction=d,delay_cycles=k) for s,p,d,k in itertools.product(SETTINGS,range(8),DIRECTIONS,DELAYS)]
def pulse_ids(cell):return [f'{cell["setting"]}--p{cell["phase_index"]}--d{d}--{cell["direction"]}--s{sign:+d}' for d in range(3) for sign in (-1,1)]


def load_packet(path):
    """Independent numeric-packet-v1 decoder with strict array/hash census."""
    path=Path(path);meta=read(path)
    if set(meta)!={'schema','tree','arrays','semantic_sha256'} or meta['schema']!='numeric-packet-v1':raise ValueError('Unknown numeric packet schema')
    entry=meta['arrays'];raw=path.parent/entry['path']
    if raw.is_symlink() or Path(entry['path']).name!=entry['path'] or not raw.resolve().is_relative_to(path.parent.resolve()):raise ValueError('Packet array alias/escape')
    if raw.stat().st_size!=entry['bytes'] or sha(raw)!=entry['sha256']:raise ValueError('Packet array bytes/hash differ')
    with np.load(raw,allow_pickle=False) as data:
        if len(data.files)!=len(set(data.files)):raise ValueError('Duplicate NPZ member')
        arrays={k:data[k].copy() for k in data.files}
    used=set()
    def decode(value):
        if isinstance(value,list):return [decode(v) for v in value]
        if not isinstance(value,dict):return value
        if '__array__' in value:
            if set(value)!={'__array__','dtype','shape'}:raise ValueError('Malformed array marker')
            key=value['__array__']
            if key in used or key not in arrays:raise ValueError('Duplicate/missing array marker')
            used.add(key);a=arrays[key]
            if a.dtype.hasobject or a.dtype.str!=value['dtype'] or list(a.shape)!=value['shape']:raise ValueError('Array dtype/shape differs')
            return a
        if '__nonfinite__' in value:
            if set(value)!={'__nonfinite__'} or value['__nonfinite__'] not in ('nan','inf','-inf'):raise ValueError('Nonfinite tag differs')
            return float(value['__nonfinite__'])
        return {k:decode(v) for k,v in value.items()}
    packet=decode(meta['tree'])
    if used!=set(arrays):raise ValueError('Unbound numerical array')
    digest=hashlib.sha256(canonical(dict(tree=meta['tree'],arrays={k:hashlib.sha256(v.tobytes(order='C')).hexdigest() for k,v in arrays.items()}))).hexdigest()
    if digest!=meta['semantic_sha256']:raise ValueError('Packet semantic hash differs')
    return packet,digest


def observable_rows(mathematics,noise_rows,orientation,base):
    """Read saved q and check its displayed normalization; never derive a gain."""
    if mathematics is None:
        return dict(base,status='unavailable',orientation=None,noise_gain=None,**{k:None for k in COMPONENTS}),[dict(base,status='unavailable',sigma=s,noise_sd=None,**{k:None for k in COMPONENTS}) for s in NOISES]
    if type(orientation) is not int or orientation not in (-1,1):raise ValueError('Frozen orientation differs')
    values=mathematics['observable']
    if set(values)!=set(COMPONENTS):raise ValueError('Observable component census differs')
    gain=finite(mathematics['noise_gain'])
    if gain<=0:raise ValueError('Positive noise gain required')
    raw={k:finite(v) for k,v in values.items()}
    if len(noise_rows)!=3 or [r['sigma'] for r in noise_rows]!=list(NOISES) or any(type(r['sigma']) is not float for r in noise_rows):raise ValueError('Three fixed noise rows required')
    standard=[]
    for row in noise_rows:
        sd=finite(row['noise_sd']);expected_sd=row['sigma']*gain
        if sd!=expected_sd or set(row['signed_standardized'])!=set(COMPONENTS):raise ValueError('Saved noise scale/component census differs')
        for key in COMPONENTS:
            value=finite(row['signed_standardized'][key]);expected=orientation*raw[key]/sd
            if not math.isclose(value,expected,rel_tol=1e-12,abs_tol=1e-14):raise ValueError('Saved scalar noise normalization differs')
        standard.append(dict(base,status='complete',sigma=row['sigma'],noise_sd=sd,**row['signed_standardized']))
    return dict(base,status='complete',orientation=orientation,noise_gain=gain,**{k:orientation*v for k,v in raw.items()}),standard


def collect(summary,manifest,rows,packets):
    cells=cell_census()
    if manifest['plan']['cells_census']!=cells or [row['cell'] for row in rows]!=cells or len(rows)!=192:raise ValueError('Exact ordered192-cell census required')
    if summary['experiment_id']!='EXP-015' or summary['setting_means'] is not None or summary['original_study_requalified'] is not False:raise ValueError('Summary scope differs')
    if set(packets)-{c['id'] for c in cells}:raise ValueError('Unplanned output packet')
    unit_rows=[];unit_noise=[];realized=[];realized_noise=[];cell_rows=[]
    for row,cell in zip(rows,cells,strict=True):
        if any(type(row['cell'][k]) is not type(v) for k,v in cell.items()):raise ValueError('Cell identity type differs')
        ids=pulse_ids(cell);counts=row['compatibility_counts']
        if row['compatibility_ids']!=ids or set(counts)!={'required','complete','failed','unavailable'} or any(type(v) is not int or v<0 for v in counts.values()) or counts['required']!=6 or sum(counts[k] for k in ('complete','failed','unavailable'))!=6:raise ValueError('Exact six-pulse denominator differs')
        flags={k:typed_bool(row[k]) for k in FLAGS}
        if flags['qualified']!=flags['qualified_ideal_neutral'] or flags['qualified_ideal_neutral'] and not flags['qualified_algebraic']:raise ValueError('Qualification implication differs')
        result=packets.get(cell['id']);unit=None;compatibility={};orientation=None
        if result is not None:
            if result['cell']!=cell or result['complete'] is not (result['status']=='complete') or [r['id'] for r in result['compatibility']]!=ids:raise ValueError('Output packet identity/completeness differs')
            if any(type(result[k]) is not bool for k in FLAGS) or result['qualified']!=result['qualified_ideal_neutral'] or result['qualified_ideal_neutral'] and not result['qualified_algebraic']:raise ValueError('Packet qualification differs')
            operational=row['status'] in ('failed','interrupted') and row.get('error') is not None
            if operational:
                if any(flags.values()):raise ValueError('Operational failure cannot qualify')
            elif row['status']!=result['status'] or any(row[k]!=result[k] for k in FLAGS):raise ValueError('Terminal/packet status differs')
            complete=sum(r['status']=='complete' for r in result['compatibility'])
            if counts!=dict(required=6,complete=complete,failed=6-complete,unavailable=0):raise ValueError('Saved compatibility counts differ')
            if result['status']=='complete' and complete!=6:raise ValueError('Completed cell requires six complete pulse rows')
            unit=result['unit'];orientation=unit.get('orientation') if unit else None;compatibility={r['id']:r for r in result['compatibility']}
        elif row['status']=='complete' or any(flags.values()) or counts!=dict(required=6,complete=0,failed=0,unavailable=6):raise ValueError('Unavailable cell claims successful outputs')
        base=dict(cell,cell_status=row['status'],**flags)
        u,un=observable_rows(unit,unit.get('noise_rows') if unit else None,orientation,base);unit_rows.append(u);unit_noise.extend(un)
        cell_rows.append(dict(base,clock_sha256=result.get('clock_sha256') if result else None,error=row.get('error') or (result.get('error') if result else None),compatibility_counts=counts))
        for ident in ids:
            packet=compatibility.get(ident);maths=packet.get('mathematics') if packet else None
            if packet and packet['complete'] is not (packet['status']=='complete'):raise ValueError('Pulse status/completeness differs')
            rbase=dict(base,pulse_id=ident,pulse_status=packet['status'] if packet else 'unavailable')
            r,rn=observable_rows(maths,packet.get('noise_rows') if packet else None,orientation,rbase)
            r.update(original_finite_phase_only_q=packet.get('original_finite_phase_only_q') if packet else None,approximation=packet.get('approximation') if packet else None)
            realized.append(r);realized_noise.extend(rn)
    expected=dict(required_cells=192,status_counts=dict(Counter(r['status'] for r in rows)),calculated_cells=sum(r['status']=='complete' for r in rows),all_gates_pass_cells=sum(r['qualified'] is True for r in rows),algebraically_qualified_cells=sum(r['qualified_algebraic'] is True for r in rows),ideal_neutral_qualified_cells=sum(r['qualified_ideal_neutral'] is True for r in rows),realized_pulse_rows={k:sum(r['compatibility_counts'][k] for r in rows) for k in ('required','complete','failed','unavailable')})
    if any(summary.get(k)!=v for k,v in expected.items()):raise ValueError('Saved summary census differs')
    if typed_bool(summary['complete'])!=(summary.get('error') is None and all(r['status']=='complete' for r in rows)):raise ValueError('Producer completion differs')
    typed_bool(summary['source_valid'])
    if (len(unit_rows),len(unit_noise),len(realized),len(realized_noise))!=(192,576,1152,3456):raise AssertionError('Report denominator changed')
    return dict(schema_version='EXP015-reference-report-v1',scope=SCOPE,counts=expected,producer_complete=summary['complete'],source_valid=summary['source_valid'],cells=cell_rows,unit=unit_rows,unit_noise=unit_noise,realized=realized,realized_noise=realized_noise,display=dict(quantity='Frozen-orientation signed linear q per unit input direction',linthresh=.001,setting_means=None,all_delays=list(DELAYS),all_phases=list(range(8))))


def load_saved(directory):
    directory=Path(directory).resolve();receipt=read(directory/'receipt.json');manifest=read(directory/'manifest.json');summary=read(directory/'summary.json');post=read(directory/'postflight.json');rows=read(directory/'terminal-records.json')
    actual=[]
    for p in sorted(directory.rglob('*')):
        if p.is_symlink():raise ValueError('Immutable artifact alias forbidden')
        if p.is_file() and p!=directory/'receipt.json':actual.append(binding(p,directory))
    if receipt['artifacts']!=actual or receipt['artifact_bytes']!=sum(x['bytes'] for x in actual):raise ValueError('Exact artifact inventory bytes/hash differ')
    if receipt['experiment_id']!='EXP-015' or receipt['source_commit']!=manifest['source_commit'] or receipt['plan_sha256']!=manifest['plan_sha256']:raise ValueError('Source/plan identity differs')
    if read(directory/'planned-cells.json')!=cell_census():raise ValueError('Initial planned cell census differs')
    initial=read(directory/'initial-terminal-records.json')
    if len(initial)!=192 or [r['cell'] for r in initial]!=cell_census() or any(r['status']!='not_started' or r['compatibility_ids']!=pulse_ids(r['cell']) or r['compatibility_counts']!=dict(required=6,complete=0,failed=0,unavailable=6) for r in initial):raise ValueError('Initial full denominator differs')
    if summary['postflight_error']!=post['error'] or summary['source_valid'] is not (post['error'] is None) or post['source_valid'] is not (post['error'] is None):raise ValueError('Postflight source flags differ')
    expected_status='complete' if summary['complete'] and summary['source_valid'] else 'incomplete_or_invalid'
    if receipt['status']!=expected_status:raise ValueError('Receipt completion differs')
    packets={}
    for row in rows:
        if 'packet_path' not in row:continue
        expected='cells/'+row['cell']['id']+'.json'
        if row['packet_path']!=expected or sha(directory/expected)!=row['packet_sha256']:raise ValueError('Cell packet file binding differs')
        packet,digest=load_packet(directory/expected)
        if digest!=row['packet_semantic_sha256']:raise ValueError('Terminal semantic binding differs')
        packets[row['cell']['id']]=packet
    provenance=dict(source_directory=str(directory),source_commit=manifest['source_commit'],plan_sha256=manifest['plan_sha256'],receipt=binding(directory/'receipt.json',directory),artifacts_verified=len(actual),artifact_bytes_verified=receipt['artifact_bytes'],scope=SCOPE)
    return collect(summary,manifest,rows,packets),provenance


def plot_report(report,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    figure,axes=plt.subplots(3,2,figsize=(18,12),sharex=True,sharey=True)
    index={r['id']:r for r in report['unit']};palette=plt.get_cmap('tab10');maximum=max((abs(r[k]) for r in report['unit'] if r['status']=='complete' for k in ('phase_algebraic','transverse')),default=.001)
    for di,direction in enumerate(DIRECTIONS):
        for si,setting in enumerate(SETTINGS):
            ax=axes[di,si];ax.axhline(0,color='#666666',lw=.6)
            for phase in range(8):
                rows=[index[f'{setting}--p{phase}--{direction}--k{d}'] for d in DELAYS]
                for name,style,marker in [('phase_algebraic','-','o'),('transverse','--','s')]:
                    values=[r[name] if r['status']=='complete' else np.nan for r in rows]
                    ax.plot(range(4),values,color=palette(phase),ls=style,marker=marker,markersize=3,lw=1.,alpha=.85)
                for x,r in enumerate(rows):
                    if r['status']!='complete':ax.text(x,.03+.022*phase,'×',color='#b42020',ha='center',transform=ax.get_xaxis_transform(),fontsize=10)
                    elif not r['qualified_algebraic']:ax.scatter(x,r['phase_algebraic'],marker='x',color='#b42020',s=45,zorder=7)
                    elif not r['qualified_ideal_neutral']:ax.scatter(x,r['phase_algebraic'],marker='v',facecolors='none',edgecolors='#333333',s=42,zorder=6)
            ax.set_title(f'{setting} · {direction.replace("_"," ")}',fontsize=11);ax.set_yscale('symlog',linthresh=.001);ax.set_ylim(-max(.003,maximum*1.4),max(.003,maximum*1.4));ax.grid(alpha=.16)
            ax.set_xticks(range(4),[str(d) for d in DELAYS]);ax.tick_params(labelsize=8)
            if di==2:ax.set_xlabel('Declared readout delay (periods)')
            if si==0:ax.set_ylabel('Signed unit-input readout q\n(shared symlog; linear ±.001)')
    counts=report['counts'];figure.suptitle('EXP-015: signed phase and transverse linear readout contributions\nAll 16 setting/phase contexts × 3 directions × 4 clocks; conditional on saved references',fontsize=15,y=.985)
    phase_handles=[Line2D([0],[0],color=palette(p),label=f'Phase {p}') for p in range(8)]
    style_handles=[Line2D([0],[0],color='#333333',ls='-',marker='o',label='Algebraic phase'),Line2D([0],[0],color='#333333',ls='--',marker='s',label='Transverse'),Line2D([0],[0],color='#333333',marker='v',markerfacecolor='none',ls='',label='Ideal-neutral gate fails'),Line2D([0],[0],color='#b42020',marker='x',ls='',label='Algebraic failure / unavailable')]
    figure.legend(handles=phase_handles,loc='lower center',bbox_to_anchor=(.5,.074),ncol=8,fontsize=9);figure.legend(handles=style_handles,loc='lower center',bbox_to_anchor=(.5,.044),ncol=4,fontsize=9)
    figure.text(.06,.015,f'Complete cells {counts["calculated_cells"]}/192; algebraically qualified {counts["algebraically_qualified_cells"]}/192; ideal-neutral qualified {counts["ideal_neutral_qualified_cells"]}/192. No means, selected direction or finite-pulse performance claim.\nFrozen orientation retained. Contributions are oblique linear components, not orthogonal energy fractions. All six realized pulses and three fixed noise scales remain in CSV/JSON.',fontsize=9)
    figure.subplots_adjust(left=.075,right=.985,top=.91,bottom=.15,hspace=.26,wspace=.12)
    for ext in ('png','pdf'):figure.savefig(Path(out)/f'phase-transverse-gains.{ext}',dpi=170,facecolor='white')
    plt.close(figure)


def write_csv(path,rows):
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader();writer.writerows({k:json.dumps(v,sort_keys=True) if isinstance(v,(dict,list)) else v for k,v in r.items()} for r in rows)

def write_report(report,provenance,out):
    out=Path(out).resolve();source=Path(provenance['source_directory']).resolve() if provenance.get('source_directory') else None
    if out==ROOT or ROOT in out.parents or source is not None and (out==source or source in out.parents):raise ValueError('Output must be outside repository and immutable raw source')
    out.parent.mkdir(parents=True,exist_ok=True);out.mkdir()
    (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    for name in ('cells','unit','unit_noise','realized','realized_noise'):write_csv(out/(name+'.csv'),report[name])
    plot_report(report,out)
    (out/'README.md').write_text('# EXP-015 saved-reference gain report\n\n'+SCOPE+'\n\nThe six panels retain every phase, direction and delay. Solid circles are algebraic-phase readouts; dashed squares are complementary transverse readouts. Raw q is multiplied only by the original frozen orientation. The shared symmetric-log display has a fixed linear region ±.001; it is not a scientific eligibility threshold. No target traces are loaded. All declared noise scales and realized-pulse compatibility rows are exported, with unavailable rows retained.\n\n'+json.dumps(provenance,indent=2)+'\n')
    artifacts=[binding(p,out) for p in sorted(out.iterdir()) if p.is_file()]
    receipt=dict(source=provenance,script=dict(path=str(Path(__file__).resolve()),bytes=Path(__file__).stat().st_size,sha256=sha(__file__)),artifacts=artifacts,artifact_bytes=sum(x['bytes'] for x in artifacts),scope=SCOPE)
    (out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    return dict(directory=str(out),receipt_sha256=sha(out/'receipt.json'),counts=report['counts'])

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--directory',required=True);parser.add_argument('--out',required=True);args=parser.parse_args();report,source=load_saved(args.directory)
    print(json.dumps(write_report(report,source,args.out),indent=2),flush=True)

if __name__=='__main__':main()
