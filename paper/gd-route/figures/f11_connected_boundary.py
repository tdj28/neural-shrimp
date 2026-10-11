"""Draw the independently proved native boundary and same-root multiplier bounds.

Saved data only. Endpoint midpoint polylines visualize the projection, not a
linear enclosure of the entire curve. Regauged portions remain separate lines.
"""
import json, math
from fractions import Fraction
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import common as c


def load(p):return json.loads(Path(p).read_text())
def ball(v):
    m,e=v['mid'];r,f=v['rad'];assert all(type(x)is int for x in(m,e,r,f)) and r>=0
    return Fraction(m)*Fraction(2)**e,Fraction(r)*Fraction(2)**f

def entry(row,inputs):
    p=Path(row['path']);assert p.stat().st_size==row['bytes'] and c.sha256(p)==row['sha256'];inputs.append(p);return load(p)


def data():
    ixp=c.ROOT/'docs/reviews/eigenpair-connected107-proof-index.json';ix=load(ixp);inputs=[ixp]
    assert ix['total_connected_charts']==len(ix['charts'])==107 and ix['period']==18 and ix['same_native_component']
    mem=ix['membership'];assert mem['outcome']['passed'] and mem['outcome']['native_coordinate_inclusions']==470
    for key in ['native_domain_check','membership','input_bindings']:entry(mem[key],inputs)
    groups={};rows=[]
    for i,row in enumerate(ix['charts'],1):
        assert row['global_chart']==i
        proof=entry(row['independent384'],inputs);assert proof['passed']
        ends=row['independently_recomputed_native_rate_endpoints'];assert ends['source']==row['independent384'] and ends['native_coordinate_indices']==[468,469]
        actual=proof['recomputed_endpoints'];assert len(actual)==2 and all(len(v)==470 for v in actual)
        assert ends['left']==actual[0][-2:] and ends['right']==actual[1][-2:]
        pts=[[float(ball(v)[0])for v in e[-2:]]for e in actual]
        groups.setdefault(row['gauge_family'],[]).extend(pts);rows.append({'chart':i,'rates':pts})
        for p in row['join'].get('localization_proofs',[]):assert entry(p,inputs)['passed']
    jpath=c.ROOT/'docs/reviews/generalized-chain-independent-outcome.json';j=load(jpath);inputs.append(jpath)
    assert j['outcome']['passed'] and j['outcome']['exact_algebraic_multiplicity_two'] and j['quotient']['all_discs_strictly_inside']
    organizer=[float(ball(v)[0])for v in j['outcome']['actual_rate_enclosures']]
    discs=[]
    for d in j['quotient']['discs']:
        assert d['strict_inside'] and sum(ball(d['recomputed_modulus_upper']))<1
        discs.append([float(ball(d['center_real'])[0]),float(ball(d['center_imag'])[0]),float(sum(ball(d['radius'])))])
    assert len(discs)==11
    bound=max(float(sum(ball(d['recomputed_modulus_upper'])))for d in j['quotient']['discs'])
    ap=c.ROOT/'docs/reviews/s2-attraction-tail-independent-outcome.json';a=load(ap);inputs.append(ap);assert a['status']=='passed' and a['total_connected_independent_slices']==112
    stable=[float(Fraction(v))for v in a['guaranteed_hidden_rate_interval']];output=float(Fraction(a['fixed_output_rate']))
    cp=c.ROOT/'certificates/s2-period18-receipt.json';cr=load(cp);inp=Path(cr['external_root'])/'inputs.json';inputs.extend([cp,inp]);assert c.sha256(inp)==cr['files']['inputs.json']['sha256'];orig=load(inp);center=float(Fraction(orig['lr_hidden']));assert Fraction(orig['lr_output'])==Fraction(a['fixed_output_rate'])
    # Canonical fold comes from the exact original augmented certificate.
    fp=Path(ix['charts'][0]['join']['parent_certificate']);fc=load(fp);inputs.append(fp)
    frp=c.ROOT/'certificates/s2-period18-fold-receipt.json';fr=load(frp);inputs.append(frp)
    assert fp==Path(fr['external_root'])/'run/certificate.json'
    assert c.sha256(fp)==fr['files']['run/certificate.json']['sha256']
    assert fc['result']['passed'] and all(fc['result']['gates'].values())
    assert fc['result']['box'][13]==fr['bounds']['hidden_rate']['ball']
    fold=float(ball(fr['bounds']['hidden_rate']['ball'])[0])
    # Connected unit-pair extension: use independently recomputed full-state endpoints.
    unit_groups={};unit_rows=[]
    def bound_proof(root,relative):
        manifest=root/'run/manifest.json';inv=load(manifest)['files'];inputs.append(manifest)
        item=root/'run'/relative;v=inv[relative]
        assert item.stat().st_size==v['bytes']and c.sha256(item)==v['sha256'];inputs.append(item)
        proof=load(item);assert proof['inclusions']==705
        # Uniform proof omits anchor identification; the separate join is checked below.
        assert all(proof['gates'][k]for k in ['uniform_root','frame_independent','positive_angle','primitive_period'])
        assert [v['divisor']for v in proof['primitive']]==[1,2,3,6,9]and all(v['separating_coordinates']for v in proof['primitive'])
        ends=proof['recomputed_endpoints'];assert len(ends)==2 and all(len(v)==705 for v in ends)
        return [[float(ball(v)[0])for v in z[702:704]]for z in ends]
    names=['native705-independent-outcome','native705-batch-independent-outcome','native705-forward-independent-outcome']+[f'native705-rotated-independent-{i:03d}-outcome'for i in range(1,4)]
    jp=c.ROOT/'docs/reviews/native705-endpoint-independent-outcome.json';jr=load(jp);inputs.append(jp)
    jroot=Path(jr['root']);jm=jroot/'manifest.json';assert c.sha256(jm)==jr['outer_manifest_sha256'];inputs.append(jm)
    jf=jroot/'run/outcome.json';assert c.sha256(jf)==load(jm)['files']['run/outcome.json']['sha256'];inputs.append(jf)
    jo=load(jf);assert jo==jr['outcome']and jo['passed']and jo['connected']and jo['anchor_coordinates']==705
    previous=None
    for packet,name in enumerate(names):
        rp=c.ROOT/'docs/reviews'/(name+'.json');rec=load(rp);inputs.append(rp);root=Path(rec['root'])
        outer=root/'manifest.json';assert c.sha256(outer)==rec['outer_manifest_sha256'];inputs.append(outer)
        inv=load(outer)['files'];wm=root/'run/manifest.json';assert c.sha256(wm)==inv['run/manifest.json']['sha256']
        op=root/'run/outcome.json';assert c.sha256(op)==inv['run/outcome.json']['sha256'];inputs.append(op)
        o=load(op);assert o==rec['outcome']and o['passed']
        count=1 if packet==0 else o['connected_count']
        if packet:
            assert o['connected']and all(z['passed']and z['coordinates']==705 for z in o['joins'])
            if packet>=2:
                ip=root/'run/input-binding.json';assert c.sha256(ip)==inv['run/input-binding.json']['sha256'];inputs.append(ip)
                identity=load(ip)['card']['prior'];assert identity['root']==str(previous)and identity['outer']==c.sha256(previous/'manifest.json')
        group=0 if packet<3 else packet-2
        for i in range(1,count+1):
            relative='native705-384.json'if packet==0 else f'chart-{i}/uniform.json'
            pts=bound_proof(root,relative);unit_groups.setdefault(group,[]).extend(pts);unit_rows.append(dict(packet=packet,chart=i,rates=pts))
        previous=root
    assert len(unit_rows)==30
    up=c.ROOT/'docs/reviews/native-unit-pair-independent-outcome.json';uc=load(up);inputs.append(up)
    assert uc['outcome']['passed'];finite_unit=[[float(ball(v)[0])for v in z]for z in uc['endpoint_actual_rate_enclosures']]
    return locals()


def main():
    d=data();c.style();fig,(ax,bx)=plt.subplots(1,2,figsize=(6.8,3.3),gridspec_kw={'width_ratios':[1.8,1]});fig.subplots_adjust(left=.09,right=.98,bottom=.19,top=.88,wspace=.36)
    for pts in d['groups'].values():
        p=np.asarray(pts);ax.plot(p[:,0],p[:,1],color=c.FOLD,lw=1.7)
    for pts in [d['finite_unit']]+list(d['unit_groups'].values()):
        p=np.asarray(pts);ax.plot(p[:,0],p[:,1],color='#b36b19',lw=2)
    ax.plot([],[],color='#b36b19',lw=2,label='Connected complex-unit pair')
    fold=d['fold']
    ax.plot([d['center'],fold],[d['output']]*2,color=c.MUTED,lw=1.5,label='Connected period-18 branch')
    ax.plot(d['stable'],[d['output']]*2,color=c.NS,lw=4,solid_capstyle='butt',label='Certified attracting interval')
    ax.scatter([d['center']],[d['output']],color=c.INK,s=20,zorder=4)
    ax.scatter([fold],[d['output']],color=c.FLIP,marker='s',s=25,zorder=5)
    ox,oy=d['organizer'];ax.scatter([ox],[oy],color=c.FLIP,marker='D',s=28,zorder=6)
    ax.annotate('1:1 resonance',xy=(ox,oy),xytext=(71.25,.82865),fontsize=8,arrowprops={'arrowstyle':'-','lw':.6,'color':c.MUTED})
    ax.annotate('Interior orbit',xy=(d['center'],d['output']),xytext=(71.3,.8221),fontsize=7,arrowprops={'arrowstyle':'-','lw':.6,'color':c.MUTED})
    ax.annotate('Saddle-node',xy=(fold,d['output']),xytext=(74.6,.8221),fontsize=7,arrowprops={'arrowstyle':'-','lw':.6,'color':c.MUTED})
    ax.plot([],[],color=c.FOLD,lw=1.7,label=r'Connected $+1$ locus')
    ax.set(xlim=(71,76.1),ylim=(.8207,.8295),xlabel=r'Hidden-layer learning rate $\eta_h$',ylabel=r'Output-layer learning rate $\eta_o$');ax.set_title('a  Certified connections in the learning-rate plane');ax.legend(loc='upper left',bbox_to_anchor=(0,.69),frameon=False,fontsize=6.4)
    bx.add_patch(Circle((0,0),1,fill=False,lw=.9,color=c.MUTED));bx.add_patch(Circle((0,0),d['bound'],fill=False,lw=.8,ls=':',color=c.NS))
    for real,imag,radius in d['discs']:
        bx.add_patch(Circle((real,imag),radius,facecolor=c.FOLD,edgecolor=c.FOLD,lw=.5));bx.plot(real,imag,'.',ms=4,color=c.FOLD)
    bx.plot(1,0,'D',ms=5,color=c.FLIP);bx.annotate('Double +1',xy=(1,0),xytext=(.1,.72),fontsize=8,arrowprops={'arrowstyle':'-','lw':.6,'color':c.MUTED})
    bx.text(0,-.69,f"{len(d['discs'])} complementary modes\ninside radius {math.ceil(d['bound']*1000)/1000:.3f}",ha='center',fontsize=7,color=c.SECONDARY)
    bx.axhline(0,color=c.NEUTRAL,lw=.6,zorder=0);bx.axvline(0,color=c.NEUTRAL,lw=.6,zorder=0);bx.set(xlim=(-1.12,1.12),ylim=(-1.12,1.12),xlabel='Real multiplier',ylabel='Imaginary multiplier',xticks=[-1,0,1],yticks=[-1,0,1]);bx.set_aspect('equal');bx.set_title('b  Spectrum at the organizer')
    c.save(fig,'f11_connected_boundary',sorted(set(d['inputs'])),__file__,'3: computer-assisted proof; displayed polylines are endpoint midpoint guides','The original period18 orbit is joined to a saddle-node at fixed output rate. A107-chart native plus-one locus joins that fold to a generic1:1 resonance; a finite704-dimensional chart and30 further705-dimensional charts continue a complex-unit-pair branch. The resonance has exactly two unit multipliers and eleven strictly attracting complementary modes.',derived={'unit_pair_charts':d['unit_rows'],'finite_unit':d['finite_unit'],'charts':d['rows'],'organizer':d['organizer'],'complement_discs':d['discs'],'complement_bound':d['bound'],'attracting_hidden_interval':d['stable'],'fixed_output':d['output'],'interior_hidden':d['center'],'fold_hidden':fold},notes='Endpoint boxes and proper full-state joins come from independent saved384-bit arithmetic. No line is interpolated across changes of gauge. Orange is connected unit-pair geometry; all eleven transverse modes are independently stable on the initial finite arc and first five705 charts. Lines inside each group guide the eye; they are not polygonal interval enclosures. Spectrum dots locate Gershgorin centers; true enclosure radii are drawn but mostly below image resolution. Attraction is proved on the green cross-section, not claimed along the whole grey line or blue locus.')
if __name__=='__main__':main()
