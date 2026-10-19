"""Saved final refinement points for Task34 cascade-p3; no model execution or proof release.

External plan pins the already-reviewed original supervisor, source and final
iteration. Retain original point radii and prove exact endpoint containment
when decoding at assembly precision. Root/primitive/attraction remain separate.
"""
import hashlib,json,subprocess
from pathlib import Path
from fractions import Fraction

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def dy(v):
    if not isinstance(v,list) or len(v)!=2 or any(type(x)is not int for x in v):raise ValueError('dyadic pair')
    return Fraction(v[0])*Fraction(2)**v[1]
def endpoints(v):
    if not isinstance(v,dict) or set(v)!={'mid','rad'}:raise ValueError('ball schema')
    m,r=dy(v['mid']),dy(v['rad'])
    if r<0:raise ValueError('negative radius')
    return m-r,m+r

def decode_original(v,unpack,pack):
    lo,hi=endpoints(v);x=unpack(v)
    if not x.is_finite():raise ValueError('nonfinite point')
    low,high=endpoints(pack(x))
    if low>lo or high<hi:raise ValueError('decoded point lost original endpoints')
    return x

def authenticate(plan,*,repository,unpack,pack,precision):
    from flint import ctx
    if precision not in (1024,2048) or ctx.prec!=precision or ctx.threads!=1:raise ValueError('assembly precision')
    p=plan['period'];n=896
    if type(p)is not int or p!=3:raise ValueError('period')
    if plan.get('cell')!='cascade-p3':raise ValueError('named point cell')
    def pinned(key):
        e=plan[key];path=Path(e['path'])
        if path.is_symlink() or not path.is_file() or sha(path)!=e['sha256']:raise ValueError('external pin '+key)
        return path,read(path)
    rp,receipt=pinned('receipt');ap,admission=pinned('admission');dp,definition=pinned('definition');cp,card=pinned('card')
    source=plan['source_commit']
    if receipt['source_commit']!=source or admission['source_commit']!=source:raise ValueError('original source')
    if receipt['returncode']!=0 or receipt['failure'] is not None or not receipt['worker_reaped'] or not receipt['worker_group_session_absent']:raise ValueError('original unsuccessful closure')
    rows=receipt['output_files'];inventory={e['path']:e for e in rows}
    if len(inventory)!=len(rows):raise ValueError('duplicate inventory')
    for name,e in inventory.items():
        q=Path(name)
        if q.is_symlink() or not q.is_file() or q.stat().st_size!=e['bytes'] or sha(q)!=e['sha256']:raise ValueError('original artifact '+name)
    for e in admission['source_blobs']:
        if e['sha256']!=plan['source_sha256'].get(e['relative_path']):raise ValueError('unreviewed source blob')
        b=subprocess.check_output(['git','show',source+':'+e['relative_path']],cwd=repository)
        if hashlib.sha256(b).hexdigest()!=e['sha256']:raise ValueError('source bytes')
    if {e['relative_path'] for e in admission['source_blobs']}!=set(plan['source_sha256']):raise ValueError('source coverage')
    if sha(cp)!=plan['source_sha256'].get(plan['card_relative_path']):raise ValueError('original card source binding')
    if card.get('cell')!='cascade-p3':raise ValueError('original point cell')
    if (card['period'],card['precision'],card['alpha'],card['log10T'])!=(p,2048,.0650,-.14798):raise ValueError('original point map')
    if definition['period']!=p or definition['dimension']!=n or definition['map']!=plan['map']:raise ValueError('definition map')
    if list(float(card['alpha']).as_integer_ratio())!=plan['map']['alpha_ratio'] or list(float(card['log10T']).as_integer_ratio())!=plan['map']['log10_temperature_ratio']:raise ValueError('exact parameter bridge')
    # Full source/card authority is external and pinned, not inferred from output.
    inputs={e['path']:e for e in card['inputs']}
    if len(inputs)!=len(card['inputs']):raise ValueError('duplicate card inputs')
    for name,e in inputs.items():
        q=Path(name)
        if q.is_symlink() or not q.is_file() or q.stat().st_size!=e['bytes'] or sha(q)!=e['sha256']:raise ValueError('original card input '+name)
    prefix_path=Path(card['prefix_output'])/'inputs.json'
    if str(prefix_path) not in inputs:raise ValueError('unbound prefix metadata')
    meta=read(prefix_path)
    if meta.get('cell')!='cascade-p3':raise ValueError('prefix point cell')
    for key in ('pinned_assets','prefix_ids','steering_layer','alpha_ratio','log10_temperature_ratio','temperature','rotary'):
        if meta[key]!=plan['map'][key]:raise ValueError('prefix/map bridge '+key)
    if meta['precision']!=2048 or meta['versions']!=plan['point_versions']:raise ValueError('point backend authority')
    if meta['source_commit']!=card['prefix_source_commit']:raise ValueError('prefix original source')
    raw=Path(plan['output'])
    if set(str(q) for q in raw.rglob('*') if q.is_file())!={name for name in inventory if Path(name).is_relative_to(raw)}:raise ValueError('whole output inventory')
    argv=receipt['argv']
    if argv!=admission['argv'] or argv[argv.index('--card')+1]!=str(cp) or argv[argv.index('--out')+1]!=str(raw):raise ValueError('original command binding')
    def original(name):
        path=raw/name
        if str(path) not in inventory:raise ValueError('missing original point artifact')
        return read(path)
    report=original('report.json');it=plan['iteration']
    if type(it)is not int or it!=16 or report['corrections_applied']!=it or report['period']!=p or report['precision']!=2048:raise ValueError('final iteration')
    label=f'iteration-{it:02d}';centers=original(label+'-centers.json')
    if centers!=definition['centers'] or len(centers)!=p or any(len(row)!=n for row in centers):raise ValueError('exact centre binding')
    if any(endpoints(v)[0]!=endpoints(v)[1] for row in centers for v in row):raise ValueError('nonpoint centre')
    images=[];original_images=[]
    for phase in range(p):
        obj=original(f'{label}-phase-{phase}.json')
        if len(obj['value'])!=n or len(obj['tangent'])!=n:raise ValueError('full state')
        if any(endpoints(v)!=(0,0) for v in obj['tangent']):raise ValueError('nonzero tangent')
        images.append([decode_original(v,unpack,pack) for v in obj['value']]);original_images.append(obj['value'])
    return images,dict(period=p,dimension=n,source_commit=source,point_precision=2048,assembly_precision=precision,definition_sha256=sha(dp),receipt_sha256=sha(rp),original_images=original_images,original_centers=centers,certificate_passed=False,note='Original local receipt/source authority only; reviewed evaluator is an external trust prerequisite; original card inputs and prefix/assets/map bridge checked. No fresh model execution.')
