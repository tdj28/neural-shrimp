"""Task21b exact-inventory saved-spectrum calculation. No producer/model execution."""
import argparse, hashlib, json, time, sys
from decimal import Decimal
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import adapter as A


def require(ok,message):
    if not ok: raise ValueError(message)


def decode(data):
    return json.loads(data,parse_float=Decimal,parse_constant=Decimal)


def encode(value):
    def decimal_only(x):
        if isinstance(x,Decimal):return str(x)
        raise TypeError('unsupported output object: '+type(x).__name__)
    return json.dumps(value,indent=2,default=decimal_only,allow_nan=False)+'\n'


def read(pin,seen):
    p=Path(pin['path']);data=p.read_bytes()
    require(len(data)==pin['bytes'] and hashlib.sha256(data).hexdigest()==pin['sha256'],'input pin mismatch: '+str(p))
    seen[str(p)]=pin
    return data


def equal(a,b):
    if isinstance(a,list) or isinstance(b,list):
        return isinstance(a,list) and isinstance(b,list) and len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    if isinstance(a,(int,float,Decimal)) and not isinstance(a,bool) and isinstance(b,(int,float,Decimal,str)):
        try:return float(a)==float(b)
        except ValueError:return False
    return a==b


def calculate(card):
    tic=time.monotonic();seen={};manifest=decode(read(card['input_manifest'],seen))
    authority=decode(read(manifest['authority'],seen));read(manifest['source_metadata_manifest'],seen)
    require(manifest['requested_cells']==38 and manifest['completed_cells']==35 and manifest['selected_file_bytes']==26017,'fixed complete input scope')
    require([r['id'] for r in manifest['runs']]==['KY1','KY2','TC'],'exact original run order')
    for p in card.get('additional_metadata_pins',[]):read(p,seen)
    rows={};runs=[]
    for spec in manifest['runs']:
        read(spec['source'],seen)
        selection=decode(read(spec['cells_file'],seen))['cells']
        payload=decode(read(spec['file'],seen));args=payload['args'];saved=payload['cells']
        requested=spec['requested_indices'];completed=spec['completed_indices'];missing=spec['missing_indices']
        require(len(set(requested))==len(requested) and completed==requested[:len(completed)],'frozen requested/completed prefix')
        require(missing==requested[len(completed):],'explicit missing suffix')
        require(args['cells_index']==requested and [r['index'] for r in saved]==completed,'actual requested/completed identity')
        require(payload['complete'] is spec['expected_complete'] and payload['complete']==(len(completed)==len(requested)),'complete flag')
        require(args['k']==spec['k'] and args['transient']==spec['transient'] and args['record']==spec['record'],'original estimator sizes')
        require(args['checkpoints']==spec['checkpoint_steps'],'original requested checkpoints')
        require(set(args)==set(spec['original_args_metadata']) and all(equal(args.get(k),v) for k,v in spec['original_args_metadata'].items()),'original source/card argument association')
        require(spec['recorded_termination_verified'] is True,'original local termination authority')
        require(len(requested)==spec['requested_count'] and len(completed)==spec['completed_count'],'actual metadata counts')
        require([r['index'] for r in spec['selected_coordinates']]==requested and all(equal(selection[r['index']],r['coordinates']) for r in spec['selected_coordinates']),'original selected coordinate association')
        require(all(type(i)is int and 0<=i<len(selection) for i in requested),'requested indices in original cell file')
        keys=[]
        for r in saved:
            key=spec['id']+':'+str(r['index']);require(key not in rows,'duplicate original run/index')
            result=A.cell(r,expected_index=r['index'],expected_coordinates=selection[r['index']],k=spec['k'],
                expected_checkpoint_steps=spec['checkpoint_steps'],record_steps=spec['record'])
            result.update(key=key,run=spec['id']);rows[key]=result;keys.append(key)
        runs.append(dict(id=spec['id'],requested_count=len(requested),completed_count=len(completed),missing_indices=missing,
            complete=payload['complete'],requested_indices=requested,completed_indices=completed,source=spec['source'],
            original_receipt_disposition=spec['original_disposition'],original_args=args,row_keys=keys,
            selection_policy=card['selection_policy'][spec['id']],source_commit=spec['source_commit'],leading_spectrum_authority=spec['leading_spectrum_authority'],ambient_dimension=spec['ambient_dimension'],ambient_dimension_scope=spec['ambient_dimension_scope'], original_spectrum_file=spec['file']))
    groups={}
    memberships={name:[spec['id']+':'+str(i) for i in indices] for spec in manifest['runs'] for name,indices in spec['group_memberships'].items()}
    memberships['headline_union']=memberships['head']+memberships['prior_headline']
    memberships['all_requested']=[spec['id']+':'+str(i) for spec in manifest['runs'] for i in spec['requested_indices']]
    require(len(rows)==35 and set(memberships['all_requested'])==set(rows)|{'KY1:18','KY1:19','TC:406'},'all completed and missing cells retained')
    for name,keys in memberships.items():
        require(len(set(keys))==len(keys),'duplicate group key')
        available=[rows[k] for k in keys if k in rows];missing=[k for k in keys if k not in rows]
        groups[name]=dict(A.group(available),requested_row_keys=keys,missing_requested_row_keys=missing,
            fixed_membership_basis=card['group_membership_basis'][name])
    for pin in list(seen.values()):read(pin,{})
    return dict(schema='task21b-independent-saved-ky-v1',computational_status='complete',
        rows=list(rows.values()),completed_cell_count=len(rows),runs=runs,groups=groups,
        input_manifest=card['input_manifest'],original_authority=manifest['authority'],owner_provenance=authority,read_pins=list(seen.values()),
        source_order_is_retained=True,classification_policy='Positive leading finite estimate means strictly greater than zero; the original selection labels/thresholds remain metadata.',
        range_policy='Actual exact extrema of supported point estimates; outward2/3decimal bounds. Tail lower bounds and nonpositive/missing/invalid rows remain separately visible.',
        limitations=['This is arithmetic on original finite QR estimates, not a model-generation or tangent replay.',
            'Four retained estimates are not the full ambient spectrum; a nonnegative final retained sum gives only a conditional lower bound.',
            'Stable sorting preserves original QR columns and permutation; source-order substitution is an explicitly labelled diagnostic.',
            'A finite positive estimate and D_KY do not establish asymptotic chaos, exact attractor dimension, smooth scalar conjugacy or a causal temperature effect.',
            'Original requested but unsaved cells are missing, not negative, imputed or silently excluded.'],
        wall_seconds=time.monotonic()-tic)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--card',required=True);ap.add_argument('--out',required=True);args=ap.parse_args()
    card=decode(Path(args.card).read_bytes());require(card['status']=='reviewed','card not reviewed')
    for name,sha in card['source_sha256'].items():
        require(Path(name).name==name,'adjacent source basename')
        require(hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()==sha,'executed source hash')
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    report=calculate(card);data=encode(report).encode();require(len(data)<=16*1024*1024,'output cap')
    (out/'report.json').write_bytes(data)
    (out/'manifest.json').write_text(encode({'files':[{'path':'report.json','bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}]}))

if __name__=='__main__':main()
