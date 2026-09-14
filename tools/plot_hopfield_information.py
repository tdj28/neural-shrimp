#!/usr/bin/env python3
"""EXP008 frozen figures from completed saved evidence; no ODE integration."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import tempfile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Rectangle, Patch
from matplotlib.font_manager import FontProperties
from matplotlib.transforms import Bbox
import numpy as np
from hopfield_drift_metrics import periodic_interpolate
from plot_hopfield import exact_npz, sha, json_write
from plot_hopfield_holdout import close_identity, replay_array, number
from run_hopfield_information import summarize, validate, specs

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = Path(__file__).resolve()
PLAN = ROOT/'experiments/hopfield/EXP-008-plan.json'
POLICIES = ['baseline', 'geometry', 'shuffled_geometry', 'min_preview']
LEARNERS = POLICIES[:3]
LABELS = ['Shared baseline', '+ Real geometry', '+ Mismatched geometry', 'Minimum preview']
COLORS = ['#D55E00', '#0072B2', '#CC79A7', '#009E73']
STYLES = ['-', '--', '-.', ':']
CONTRASTS = ['baseline_minus_geometry', 'shuffled_minus_geometry', 'min_preview_minus_geometry']
CONTRAST_LABELS = ['Baseline - geometry', 'Mismatch - geometry', 'Preview - geometry']
COVERAGE_LABELS = ['Core, uncensored', 'Other, uncensored', 'Inside, censored', 'Outside crop: unknown']
COVERAGE_COLORS = ['#56B4E9', '#E69F00', '#999999', '#CC79A7']
FAMILIES = ['neuralshrimp-information-map', 'neuralshrimp-information-utility',
            'neuralshrimp-information-waveforms']
INDEX = 'hopfield-information-figure-provenance.json'
plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':10, 'axes.labelsize':10,
    'axes.titlesize':11, 'legend.fontsize':9.5, 'xtick.labelsize':10, 'ytick.labelsize':10,
    'axes.spines.top':False, 'axes.spines.right':False, 'pdf.fonttype':42,
    'svg.fonttype':'none', 'svg.hashsalt':'neural-shrimp-exp008-v1'})


def load(runroot):
    # This first read is deliberately the completion gate. No partial outcome
    # or model file is read when the target collection has not completed.
    receipt = json.loads((runroot/'receipt.json').read_text())
    if receipt.get('status') != 'complete':
        raise ValueError('A completed EXP008 receipt is required')
    if (runroot/'failure.json').exists() or (runroot/'timeout.json').exists():
        raise ValueError('Attempt failure is present; no completed-result rendering')
    p = json.loads((runroot/'manifest.json').read_text()); validate(p)
    if p != json.loads(PLAN.read_text()) or sha(runroot/'manifest.json') != receipt['manifest_sha256']:
        raise ValueError('Frozen manifest drift')
    paths = [item['path'] for item in receipt['artifacts']]
    if len(paths) != len(set(paths)):
        raise ValueError('Duplicate receipt artifact')
    for item in receipt['artifacts']:
        path = runroot/item['path']
        if sha(path) != item['sha256'] or path.stat().st_size != item['bytes']:
            raise ValueError('Saved evidence drift: '+item['path'])
    for item in p['source_artifacts']+p['source_files']:
        if sha(ROOT/item['path']) != item['sha256']:
            raise ValueError('Frozen input/source drift: '+item['path'])
    features, baselines, records = {}, {}, []
    for role in ['development', 'test']:
        features[role] = json.loads((runroot/f'{role}-features.json').read_text())
        baselines.update(json.loads((runroot/f'{role}-baselines.json').read_text()))
        rr = [json.loads(line) for line in (runroot/f'{role}-targets.jsonl').read_text().splitlines()]
        expected = {(s['anchor'],s['phase'],s['radius'],s['candidate'],s['dt']) for s in specs(p,role)}
        actual = {(r['anchor'],r['phase'],r['radius'],r['candidate'],r['dt']) for r in rr}
        if len(rr) != len(expected) or actual != expected:
            raise ValueError('Complete attempted target census required')
        records.extend(rr)
    if set(baselines) != {a['id'] for a in p['anchors']}:
        raise ValueError('Baseline census drift')
    decisions = json.loads((runroot/'test-decisions.json').read_text())
    summary = json.loads((runroot/'summary.json').read_text())
    close_identity(summary, summarize(records, decisions, p), 'summary')
    model = json.loads((runroot/'model.json').read_text())
    model_seal = json.loads((runroot/'model-seal.json').read_text())
    old, prior = [], []
    for item in p['source_artifacts']:
        path = ROOT/item['path']
        if '/focus/row-' in item['path']:
            with np.load(path, allow_pickle=False) as raw:
                old.append((float(raw['gamma']),raw['alpha'].copy(),raw['period'].copy()))
        elif any(eid in item['path'] for eid in ['EXP-006/','EXP-007/']) and path.name=='manifest.json':
            prior.extend(json.loads(path.read_text())['anchors'])
    old.sort(key=lambda row:row[0])
    if not old or any(not np.array_equal(row[1],old[0][1]) for row in old):
        raise ValueError('Old map coordinates differ')
    sensitivity = [json.loads((runroot/rel).read_text()) for rel in paths
                   if rel.startswith('sensitivity/') and rel.endswith('.json') and not rel.endswith('-failure.json')]
    costs = dict(collection_wall_seconds=receipt['wall_seconds'], fit_seconds=model_seal['fit_seconds'],
        sensitivity_sum_call_seconds=sum(s['wall_seconds'] for s in sensitivity),
        sensitivity_calls=len(sensitivity), old_map_seconds=140.54,
        preview_sum_overlapping_call_seconds=sum(r['wall_seconds'] for rr in features.values() for r in rr),
        target_sum_overlapping_call_seconds=sum(r['wall_seconds'] for r in records),
        planned_previews=p['planned_previews'], received_previews=sum(map(len,features.values())),
        complete_previews=sum(r['status']=='complete' for rr in features.values() for r in rr),
        planned_targets=p['planned_targets'], received_targets=len(records),
        complete_targets=sum(r['status']=='complete' for r in records))
    return dict(plan=p,receipt=receipt,features=features,baselines=baselines,records=records,
        decisions=decisions,summary=summary,model=model,costs=costs,
        alpha=old[0][1],gamma=np.array([r[0] for r in old]),periods=np.stack([r[2] for r in old]),
        prior=prior,runroot=runroot,run_sources=sorted(set(paths)|{'receipt.json'}))


def title(ax,letter,text):
    ax.set_title(f'{letter}  {text}',loc='left',pad=9)


def place_labels(ax, points, labels, colors=None):
    """Deterministic display-space label placement; plotted data never move."""
    ax.figure.canvas.draw()
    renderer=ax.figure.canvas.get_renderer(); frame=ax.get_window_extent()
    pixels=ax.transData.transform(np.asarray(points)); boxes=[]
    font=FontProperties(family='DejaVu Sans',size=9.5)
    offsets=[(distance*np.cos(angle),distance*np.sin(angle))
             for distance in [14,23,33,46,62,82] for angle in np.arange(16)*np.pi/8]
    for i,(point,label) in enumerate(zip(pixels,labels)):
        width,height,descent=renderer.get_text_width_height_descent(label,font,ismath=False)
        width+=5;height+=7
        def cost(offset):
            center=point+offset; box=Bbox.from_bounds(center[0]-width/2,center[1]-height/2,width,height)
            outside=max(0,frame.x0-box.x0)+max(0,box.x1-frame.x1)+max(0,frame.y0-box.y0)+max(0,box.y1-frame.y1)
            overlap=sum(max(0,min(box.x1,b.x1)-max(box.x0,b.x0))*max(0,min(box.y1,b.y1)-max(box.y0,b.y0)) for b in boxes)
            point_overlap=sum(box.x0-3<x<box.x1+3 and box.y0-3<y<box.y1+3 for x,y in pixels)
            return 1e6*outside+1e4*overlap+1e3*point_overlap+np.dot(offset,offset),box,center
        _,box,center=min((cost(np.asarray(o)) for o in offsets),key=lambda v:v[0]);boxes.append(box)
        xytext=ax.transData.inverted().transform(center)
        ax.annotate(label,points[i],xytext=xytext,textcoords='data',ha='center',va='center',
            fontsize=9.5,color='black' if colors is None else colors[i],
            bbox=dict(fc='white',ec='none',alpha=.82,pad=.2),
            arrowprops=dict(arrowstyle='-',color='#666666',lw=.5,shrinkA=1,shrinkB=3))


def coverage_class(row):
    if row.get('geometry') is None:
        return -1
    g = row['geometry']
    return 3 if not g[4] else 2 if g[5] else 0 if g[2] else 1


def map_figure(data):
    p=data['plan']; aa=data['alpha']; gg=data['gamma']; origin=np.array(p['selection']['normalization_origin'])
    widths=np.array(p['selection']['normalization_widths']); halo=p['selection']['exclusion_radius']
    coords=np.array([[a['alpha'],a['gamma']] for a in p['anchors']])
    roles=np.array([a['role'] for a in p['anchors']]); eligible=np.array([data['baselines'][a['id']]['eligible'] for a in p['anchors']],bool)
    prior=np.array([[a['alpha'],a['gamma']] for a in data['prior']])
    blocks=np.full(data['periods'].shape,-1,int)
    for i,b in enumerate(p['selection']['blocks']):
        m=np.array(b['members'],int); blocks[m[:,0],m[:,1]]=i
    feature_rows=[r for role in ['development','test'] for r in data['features'][role]]
    destinations=np.array([[number(r.get('alpha')),number(r.get('gamma'))] for r in feature_rows])
    coverage=np.array([coverage_class(r) for r in feature_rows],int)
    counts=np.zeros((2,4,5),int)
    for ri,role in enumerate(['development','test']):
        for j,radius in enumerate(p['radii']):
            for row in data['features'][role]:
                if row['radius']==radius:
                    c=coverage_class(row); counts[ri,j,c if c>=0 else 4]+=1
    arrays=dict(alpha=aa,gamma=gg,old_peak_period=data['periods'],test_block_index=blocks,
        origin=origin,widths=widths,anchor_coordinates=coords,anchor_roles=roles,
        anchor_ids=np.array([a['id'] for a in p['anchors']]),eligible=eligible,prior_coordinates=prior,
        halo=np.array(halo),destination_coordinates=destinations,coverage_class=coverage,
        coverage_counts=counts,radii=np.array(p['radii']))
    fig,axs=plt.subplots(2,2,figsize=(12.6,8.8)); ax=axs[0,0]
    mask=np.where(data['periods']==9,1,np.where(data['periods']==18,2,0))
    ax.pcolormesh(aa,gg,mask,shading='nearest',rasterized=True,
                 cmap=ListedColormap(['#f1f1f1','#c6dbea','#f1d4bb']),vmin=0,vmax=2)
    for i,point in enumerate(coords):
        test=roles[i]=='test'; color=COLORS[1] if test else COLORS[0]; marker='o' if test else '^'
        ax.scatter(*point,marker=marker,s=40,edgecolors=color,facecolors=color if eligible[i] else 'white',zorder=3)
        if not eligible[i]:ax.scatter(*point,marker='x',s=22,color='black',zorder=4)
    ax.set(xlabel=r'Weight $\alpha$',ylabel=r'Weight $\gamma$',xlim=(aa[0],aa[-1]),ylim=(gg[0],gg[-1]))
    title(ax,'A','Every attempted origin on the old atlas')
    ax.legend(handles=[Line2D([],[],marker='^',color=COLORS[0],ls='',label='D: development'),
        Line2D([],[],marker='o',color=COLORS[1],ls='',label='T: test')],loc='upper right',fontsize=9.5)
    ax=axs[0,1]; normal=(coords-origin)/widths
    ax.pcolormesh((aa-origin[0])/widths[0],(gg-origin[1])/widths[1],np.ma.masked_less(blocks,0),
        shading='nearest',cmap=ListedColormap(plt.get_cmap('tab20').colors[:12]),vmin=-.5,vmax=11.5,rasterized=True)
    for point in (prior-origin)/widths:
        ax.add_patch(Circle(point,halo,color='#555555',alpha=.13,lw=.6))
    for i,point in enumerate(normal[roles=='test']):
        ax.scatter(*point,s=32,facecolors='white',edgecolors='black',zorder=3)
    ax.set(xlim=(0,1),ylim=(0,1),xlabel='Normalized alpha',ylabel='Normalized gamma')
    ax.set_aspect('equal',adjustable='box')
    title(ax,'B','Test pixel groups and prior-origin halos')
    ax=axs[1,0]
    for c,(label,color) in enumerate(zip(COVERAGE_LABELS,COVERAGE_COLORS)):
        xy=destinations[coverage==c]
        ax.scatter(xy[:,0],xy[:,1],s=7,c=color,alpha=.35,rasterized=True,label=label)
    ax.add_patch(Rectangle((aa[0],gg[0]),np.ptp(aa),np.ptp(gg),fc='none',ec='black',lw=1.2))
    ax.set(xlabel=r'Destination $\alpha$',ylabel=r'Destination $\gamma$')
    ax.margins(.035);title(ax,'C','All candidate destinations: map coverage')
    ax=axs[1,1]; x=np.arange(4); width=.34
    for ri,role in enumerate(['development','test']):
        bottom=np.zeros(4)
        denominator=counts[ri].sum(axis=1)
        for c,color in enumerate(COVERAGE_COLORS+['white']):
            values=np.divide(counts[ri,:,c],denominator,out=np.zeros(4),where=denominator>0)
            ax.bar(x+(ri-.5)*width,values,bottom=bottom,width=width,color=color,edgecolor='#444444',
                   linewidth=.3,hatch='//' if c==4 else None)
            bottom+=values
    ax.set(xticks=x,xticklabels=[f'{r:g}' for r in p['radii']],xlabel='Native weight-change norm',
           ylabel='Fraction of declared candidates',ylim=(0,1.02))
    title(ax,'D','Coverage by dose; development / test pairs')
    handles=[Patch(fc=c,label=l) for c,l in zip(COVERAGE_COLORS,COVERAGE_LABELS)]
    if counts[:,:,4].sum():handles.append(Patch(fc='white',hatch='//',label='Unavailable feature'))
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.052),ncol=4,fontsize=9.5)
    fig.suptitle('EXP-008: fixed samples and availability of the old-map features',y=.995)
    fig.text(.5,.013,'A: blue = nine peaks; tan = eighteen; gray = other labels. Open/crossed origins failed qualification.\n'
        'B: gray halos exclude normalized distance ≤0.04 from 24 prior origins; blocks contain eligible pixels, not unseen rectangles.\n'
        'C-D: all 3,456 candidate entries; outside-crop membership is unknown. Paired bars: development left, test right.',
        ha='center',fontsize=9.5)
    fig.subplots_adjust(left=.075,right=.98,top=.93,bottom=.19,hspace=.48,wspace=.25)
    place_labels(axs[0,0],coords,[('T' if r=='test' else 'D')+f'{i%12:02d}' for i,r in enumerate(roles)],
                 [COLORS[1] if r=='test' else COLORS[0] for r in roles])
    place_labels(axs[0,1],normal[roles=='test'],[f'T{i:02d}' for i in range(12)])
    return fig,arrays,dict(panels=4,anchor_ids=arrays['anchor_ids'].tolist(),eligible=eligible.tolist(),
        coverage_order=COVERAGE_LABELS+['Unavailable feature'],coverage_counts=counts.tolist(),
        selection=p['selection']['scope'],missing_features=int((coverage<0).sum()))


def utility_figure(data):
    p,s=data['plan'],data['summary']; radii=np.array(p['radii']); steps=np.array(p['dt'])
    means=np.full((2,4,4),np.nan); regrets=means.copy(); oracle=np.full((2,4),np.nan)
    contrasts=np.full((2,3,3),np.nan); perblock=np.full((12,5),np.nan)
    for di,dt in enumerate(steps):
        panels=s['steps'][str(dt)]['panels']
        for j,r in enumerate(radii):
            for k,policy in enumerate(POLICIES):
                means[di,k,j]=number(panels[str(r)]['means'][policy]['selected_rmse'])
                regrets[di,k,j]=number(panels[str(r)]['means'][policy]['regret'])
            values=[c['oracle_rmse'] for c in s['steps'][str(dt)]['conditions'] if c['radius']==r]
            if len(values)==24 and all(v is not None for v in values):oracle[di,j]=np.mean(values)
        for k,name in enumerate(CONTRASTS):
            row=panels['pooled']['contrasts'][name]
            if row['status']=='complete':contrasts[di,k]=[row['mean_benefit'],*row['ci95']]
    fine=list(steps).index(p['primary_dt'])
    panels=s['steps'][str(p['primary_dt'])]['panels']
    for j,dose in enumerate([*map(str,radii),'pooled']):
        row=panels[dose]['contrasts']['baseline_minus_geometry']
        if row['status']=='complete':perblock[:,j]=row['per_block']
    lambdas=np.array(p['lambda_grid']);cv=np.full((3,5),np.nan);chosen=np.full(3,-1,int)
    for i,policy in enumerate(LEARNERS):
        item=data['model'].get('models',{}).get(policy)
        if item:
            cv[i]=item['cv']['equal_group_mse'];chosen[i]=item['chosen_lambda_index']
    arrays=dict(radii=radii,steps=steps,policy_names=np.array(POLICIES),selected_rmse=means,
        regret=regrets,oracle_rmse=oracle,pooled_contrast=contrasts,contrast_names=np.array(CONTRASTS),
        block_benefit=perblock,block_ids=np.array(s['blocks']),lambdas=lambdas,cv_mse=cv,
        chosen_lambda_index=chosen,cost_names=np.array(list(data['costs'])),cost_values=np.array(list(data['costs'].values()),float))
    fig,axes=plt.subplots(2,3,figsize=(12.6,7.8));flat=axes.ravel()
    for pos,di in enumerate([fine,1-fine]):
        ax=flat[pos]
        for k,label in enumerate(LABELS):
            ax.plot(radii,means[di,k],STYLES[k],c=COLORS[k],marker=['o','s','^','D'][k],ms=4,label=label,lw=1.5)
        ax.plot(radii,oracle[di],c='#444444',ls=':',label='Finite-18 oracle',lw=1.3)
        ax.set(xscale='log',xlabel='Native weight norm',ylabel='Selected tail RMSE',xticks=radii,
               xticklabels=[f'{r:g}' for r in radii]);ax.grid(alpha=.15)
        title(ax,'AB'[pos],f"All policies: dt={steps[di]:g}"+(' (primary)' if di==fine else ''))
    flat[0].legend(loc='upper left',fontsize=9.5)
    ax=flat[2]
    for di,dt in enumerate(steps):
        y=np.arange(3)+(di-.5)*.18
        for k in range(3):
            value,low,high=contrasts[di,k]
            if np.isfinite(value):ax.errorbar(value,y[k],xerr=[[value-low],[high-value]],
                fmt='o' if di==fine else 's',mfc='#222222' if di==fine else 'white',mec='#222222',
                color='#222222',capsize=3,ms=5,label=f'dt={dt:g}' if k==0 else None)
    ax.axvline(0,c='#777777',ls=':',lw=1);ax.set(yticks=np.arange(3),yticklabels=CONTRAST_LABELS,
        xlabel='Pooled RMSE benefit from geometry');ax.invert_yaxis();ax.grid(axis='x',alpha=.15)
    title(ax,'C','Pooled benefit; 95% intervals')
    ax=flat[3]; finite_values=np.abs(perblock[np.isfinite(perblock)])
    extent=float(finite_values.max()) if len(finite_values) else 1.
    if extent==0:extent=1e-12
    im=ax.imshow(np.ma.masked_invalid(perblock),cmap='RdBu',vmin=-extent,vmax=extent,aspect='auto')
    for (i,j),value in np.ndenumerate(perblock):
        ax.text(j,i,'NA' if not np.isfinite(value) else f'{value:+.3f}',ha='center',va='center',fontsize=9.5,
                color='white' if np.isfinite(value) and abs(value)>.60*extent else 'black')
    ax.set(xticks=range(5),xticklabels=[f'{r:g}' for r in radii]+['Pooled'],
           yticks=range(12),yticklabels=[f'T{i:02d}' for i in range(12)],xlabel='Native norm / pooled')
    title(ax,'D','All blocks: baseline - geometry')
    ax=flat[4]
    for k,label in enumerate(LABELS):
        ax.plot(radii,regrets[fine,k],STYLES[k],c=COLORS[k],marker=['o','s','^','D'][k],ms=4,label=label,lw=1.5)
    ax.set(xscale='log',xticks=radii,xticklabels=[f'{r:g}' for r in radii],xlabel='Native weight norm',
           ylabel='Excess RMSE over finite oracle');ax.grid(alpha=.15)
    title(ax,'E','Fine-step finite-action regret')
    ax=flat[5]
    for i,label in enumerate(LABELS[:3]):
        ax.plot(lambdas,cv[i],STYLES[i],color=COLORS[i],marker='o',ms=4,label=label)
        if chosen[i]>=0:
            ax.scatter(lambdas[chosen[i]],cv[i,chosen[i]],s=95,facecolor='none',edgecolor=COLORS[i],lw=1.5)
    ax.set(xscale='log',xlabel='Ridge penalty',ylabel='Validation prediction MSE');ax.grid(alpha=.15)
    title(ax,'F','Development CV; penalty circled')
    c=data['costs']; gate='defined' if s['primary']['status']=='complete' else 'UNDEFINED: required data/choice failure'
    fig.suptitle('EXP-008: fixed predictor ablation, task utility and finite-action regret',y=.995)
    fig.text(.5,.027,f"Primary {gate}. Positive favors geometry; descriptive paired-block intervals. C: filled dt=.005, open dt=.01. T00-T11: all blocks.\n"
        f"Targets {c['complete_targets']}/{c['planned_targets']}; previews {c['complete_previews']}/{c['planned_previews']}. "
        f"Collection {c['collection_wall_seconds']:.2f} s (includes fit {c['fit_seconds']:.2f} s); prior map {c['old_map_seconds']:.2f} s separately.\n"
        'Oracle outcomes are evaluation cost. The same 18 previews feed each policy; no equal-total-cost or net-speedup claim.',
        ha='center',fontsize=9.5)
    fig.subplots_adjust(left=.065,right=.99,top=.92,bottom=.19,hspace=.50,wspace=.52)
    return fig,arrays,dict(panels=6,primary=s['primary'],costs=c,bootstrap=s['bootstrap'],
        oracle='Minimum of exactly18 candidate outcomes; identical oracle cancels in primary contrast',
        missing_entries_remain_nan=True,policies=POLICIES,steps=p['dt'])


def waveform_figure(data):
    p=data['plan'];e=p['display_contract']['example']; aid=e['anchor'];phase=e['phase'];radius=e['radius'];dt=e['dt']
    baseline=data['baselines'][aid]
    lookup={(r['anchor'],r['phase'],r['radius'],r['candidate'],r['dt']):r for r in data['records']}
    decision=next(r for r in data['decisions'] if (r['anchor'],r['phase'],r['radius'])==(aid,phase,radius))
    features={r['candidate']:r for r in data['features']['test'] if (r['anchor'],r['phase'],r['radius'])==(aid,phase,radius)}
    actual=np.array([number(lookup[(aid,phase,radius,c,dt)].get('rmse')) for c in range(18)])
    preview=np.array([number(features[c].get('preview_score')) for c in range(18)])
    choices=np.array([decision['policies'][policy].get('candidate') if decision['policies'][policy].get('candidate') is not None else -1 for policy in POLICIES],int)
    times=np.arange(3001)*.1;times[-1]=300.
    values=np.full((4,3001,3),np.nan); aligned=values.copy();sync=np.full((3001,3),np.nan)
    phases=np.full(4,np.nan);errors=values.copy();eligible=bool(baseline['eligible'])
    if eligible:
        cert=baseline['certificate']; reference=np.array(cert['orbit_activations'][:-1]);T=cert['period'];epoch=phase*T/2700
        sync=periodic_interpolate(reference,(times+epoch)/T)
        for k,c in enumerate(choices):
            if c<0:continue
            row=lookup[(aid,phase,radius,int(c),dt)]
            if row['status']!='complete':continue
            with np.load(data['runroot']/row['raw'],allow_pickle=False) as raw:
                np.testing.assert_array_equal(times,raw['times']); values[k]=np.tanh(raw['states'])
            phases[k]=row['tail_metrics']['phase_offset_cycles']
            aligned[k]=periodic_interpolate(reference,(times+epoch)/T+phases[k]);errors[k]=values[k]-aligned[k]
            close_identity(float(np.sqrt(np.mean(errors[k,times>75]**2))),row['rmse'],f'example.{POLICIES[k]}.rmse')
    arrays=dict(times=times,candidate_activations=values,original_epoch_reference=sync,
        saved_phase_aligned_reference=aligned,saved_phase_aligned_errors=errors,
        saved_tail_phase_cycles=phases,selected_candidate_ids=choices,all_candidate_ids=np.arange(18),
        all_preview_rmse=preview,all_tail_rmse=actual,eligible=np.array(eligible),
        anchor_id=np.array(aid),phase_index=np.array(phase),radius=np.array(radius),dt=np.array(dt))
    fig,axs=plt.subplots(2,2,figsize=(12.6,8.5))
    for neuron,ax in enumerate(axs.ravel()[:3]):
        ax.plot(times,sync[:,neuron],c='#555555',ls='--',lw=1.,alpha=.7,label='Reference: original epoch')
        for k,label in enumerate(LABELS):
            ax.plot(times,values[k,:,neuron],STYLES[k],color=COLORS[k],lw=1.05,label=label,alpha=.85)
        ax.axvspan(0,75,color='#777777',alpha=.12);ax.axvline(75,c='#777777',ls=':',lw=.8)
        ax.set(xlim=(0,300),ylim=(-1.07,1.07),xlabel='Time after weight change',ylabel=fr'Activation $u_{neuron+1}$')
        title(ax,'ABC'[neuron],f'Neuron {neuron+1}: original time origin');ax.grid(alpha=.12)
        if not eligible:ax.text(.5,.5,'Frozen example ineligible; no replacement',transform=ax.transAxes,ha='center')
    ax=axs[1,1];good=np.isfinite(preview)&np.isfinite(actual)
    ax.scatter(preview[good],actual[good],s=28,c='#555555',zorder=3)
    for k,c in enumerate(choices):
        if c>=0 and good[c]:ax.scatter(preview[c],actual[c],s=130+k*45,facecolors='none',edgecolors=COLORS[k],
                                       marker=['o','s','^','D'][k],lw=1.3,zorder=4)
    maximum=max(np.nanmax(preview) if np.isfinite(preview).any() else 1.,np.nanmax(actual) if np.isfinite(actual).any() else 1.)
    ax.plot([0,maximum],[0,maximum],c='#aaaaaa',ls=':',lw=1.)
    ax.set(xlabel='Full 0-75 preview RMSE',ylabel='Later 75<t≤300 RMSE');ax.margins(.10);ax.grid(alpha=.12)
    title(ax,'D','All 18 candidates; outlined policy choices')
    handles=[Line2D([],[],color='#555555',ls='--',label='Reference: original epoch')]
    handles += [Line2D([],[],color=COLORS[k],ls=STYLES[k],marker=['o','s','^','D'][k],
                     label=f'{LABELS[k]} (c{choices[k] if choices[k]>=0 else "NA"})') for k in range(4)]
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.080),ncol=3,fontsize=9.5)
    fig.suptitle(f'Frozen EXP-008 example: {aid}, phase {phase}, norm {radius:g}, dt={dt:g}',y=.995)
    if len(set(choices))==1 and choices[0]>=0:
        fig.text(.5,.957,f'All four policies select candidate {choices[0]}; candidate waveforms coincide.',ha='center',fontsize=10)
    fig.text(.5,.018,'The first declared test origin is retained regardless of qualification or which policy wins. Gray interval: preview 0-75.\n'
        'Waveforms use the original epoch. Reported RMSE fits one constant phase on its scored interval, separately for each policy.\n'
        'No frequency fit or time warp. Candidate labels are 0-17; overlapping candidate entries remain separate charged evaluations.',
        ha='center',fontsize=9.5)
    fig.subplots_adjust(left=.075,right=.975,top=.90,bottom=.23,hspace=.40,wspace=.25)
    place_labels(axs[1,1],np.column_stack((preview[good],actual[good])),[str(c) for c in np.flatnonzero(good)])
    return fig,arrays,dict(panels=4,selection=e,eligible=eligible,selected_candidate_ids=choices.tolist(),
        tail_phase_cycles=[None if not np.isfinite(v) else float(v) for v in phases],
        example_not_replaced=True,scored_interval='75<t<=300',waveform_display='Original epoch; each metric uses its own saved global phase')


PLOTS=[map_figure,utility_figure,waveform_figure]


def source_entries(data):
    result=[dict(namespace='run',path=rel,sha256=sha(data['runroot']/rel)) for rel in data['run_sources']]
    result += [dict(namespace='repo',**item) for item in data['plan']['source_artifacts']+data['plan']['source_files']]
    return result


def build(runroot,destination):
    data=load(runroot);destination.mkdir(parents=True,exist_ok=True);sources=source_entries(data);index=[]
    generators=[SCRIPT,ROOT/'tools/plot_hopfield.py',ROOT/'tools/plot_hopfield_holdout.py',ROOT/'tools/hopfield_drift_metrics.py']
    for stem,plot in zip(FAMILIES,PLOTS):
        fig,arrays,conditions=plot(data);exact_npz(destination/f'{stem}.plot-data.npz',arrays)
        for ext in ['pdf','svg','png']:
            metadata=({'CreationDate':None,'ModDate':None,'Creator':'neural-shrimp saved-data renderer'} if ext=='pdf'
                      else {'Date':None} if ext=='svg' else {'Software':'neural-shrimp saved-data renderer'})
            fig.savefig(destination/f'{stem}.{ext}',dpi=300,metadata=metadata)
        plt.close(fig)
        receipt=dict(schema_version=1,experiment='EXP-008',stem=stem,generators=[dict(path=str(g.relative_to(ROOT)),sha256=sha(g)) for g in generators],
            plan=dict(path=str(PLAN.relative_to(ROOT)),sha256=sha(PLAN)),sources=sources,conditions=conditions,
            plotted_arrays=dict(path=f'{stem}.plot-data.npz',sha256=sha(destination/f'{stem}.plot-data.npz')),
            outputs=[dict(path=f'{stem}.{ext}',sha256=sha(destination/f'{stem}.{ext}')) for ext in ['pdf','svg','png']],
            scope='Frozen4+6+4 panel contract; complete saved evidence only; no target integrations; no outcome-selected examples')
        path=destination/f'{stem}.receipt.json';json_write(path,receipt);index.append(dict(path=path.name,sha256=sha(path)))
    json_write(destination/INDEX,dict(schema_version=1,experiment='EXP-008',figures=index))


def check(runroot,destination):
    data=load(runroot);index=json.loads((destination/INDEX).read_text())
    if [r['path'] for r in index['figures']] != [f'{stem}.receipt.json' for stem in FAMILIES]:
        raise ValueError('Figure family census drift')
    for entry,plot in zip(index['figures'],PLOTS):
        path=destination/entry['path']
        if sha(path)!=entry['sha256']:raise ValueError('Figure receipt drift')
        r=json.loads(path.read_text())
        for item in r['generators']+[r['plan']]:
            if sha(ROOT/item['path'])!=item['sha256']:raise ValueError('Generator or plan drift')
        for item in r['sources']:
            parent=runroot if item['namespace']=='run' else ROOT
            if sha(parent/item['path'])!=item['sha256']:raise ValueError('Input hash drift: '+item['path'])
        for item in r['outputs']+[r['plotted_arrays']]:
            if sha(destination/item['path'])!=item['sha256']:raise ValueError('Output hash drift: '+item['path'])
        fig,arrays,conditions=plot(data);plt.close(fig)
        close_identity(r['conditions'],conditions,'plot conditions')
        with np.load(destination/r['plotted_arrays']['path'],allow_pickle=False) as saved:
            if set(saved.files)!=set(arrays):raise ValueError('Plotted array census drift')
            for name,value in arrays.items():replay_array(saved[name],np.asarray(value),name)
    print('PASS: EXP008 input/output hashes, full census, summary and plotted arrays; renderer bytes are not regenerated for comparison.')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('runroot',type=Path)
    parser.add_argument('--destination',required=True,type=Path);parser.add_argument('--check',action='store_true')
    args=parser.parse_args()
    if args.check:check(args.runroot.resolve(),args.destination.resolve())
    else:build(args.runroot.resolve(),args.destination.resolve());print('Built EXP008 three families /14 empirical panels.')

if __name__=='__main__':main()
