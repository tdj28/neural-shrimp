"""Small saved-spectrum report functions; no input files are read on import."""
from fractions import Fraction
from pathlib import Path
import math, sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import kaplan_yorke as K


def fraction(rec):
    return Fraction(int(rec['numerator']), int(rec['denominator']))


def dimension(result):
    return None if result['dimension'] is None else fraction(result['dimension'])


def spectrum(values):
    """Preserve native QR order; compare saved-decimal, represented and literal arithmetic."""
    decimal = K.estimate(values, leading_spectrum=True)
    out = dict(saved_decimal=decimal, original_qr_order=[str(x) for x in values])
    if decimal['disposition'] == 'invalid saved spectrum':
        return dict(out, represented_binary64=None, literal_binary64=None, finite=False)
    native = list(map(float, values))
    if not all(math.isfinite(x) for x in native):
        return dict(out, represented_binary64=None, literal_binary64=None, finite=False,
                    limitation='Saved decimal value is outside finite binary64 range.')
    binary = K.estimate(native, leading_spectrum=True)
    order = sorted(range(len(native)), key=lambda i: native[i], reverse=True)
    lam = [native[i] for i in order]; sums = []; total=0.
    for x in lam:
        total += x; sums.append(total)
    j=max([0]+[i+1 for i,s in enumerate(sums) if s>=0])
    point=None if j==len(lam) else float(j)+(sums[j-1] if j else 0.)/abs(lam[j])
    de,bi=dimension(decimal),dimension(binary)
    native_exact=list(map(K.exact,values)); native_sums=[]; native_total=Fraction(0)
    for v in native_exact:
        native_total+=v; native_sums.append(native_total)
    native_j=max([0]+[i+1 for i,v in enumerate(native_sums) if v>=0])
    native_point=None if native_j==len(values) or native_exact[native_j]==0 else Fraction(native_j)+(native_sums[native_j-1] if native_j else 0)/abs(native_exact[native_j])
    out.update(finite=True, represented_binary64=binary,
        native_qr_order_ky_diagnostic=dict(cumulative_sums=[K.number(v) for v in native_sums],j=native_j,
            candidate_dimension=None if native_point is None else K.number(native_point),
            agrees_with_required_descending_order=decimal['input_already_nonincreasing'],
            candidate_minus_sorted_dimension=None if native_point is None or de is None else K.number(native_point-de),
            limitation='Unsorted native-order substitution is a historical comparison only; the Kaplan–Yorke formula requires descending exponents.'),
        binary64_native_values=native, binary64_hex=[x.hex() for x in native],
        positive_count_native=sum(x>0 for x in native),zero_count_native=sum(x==0 for x in native),
        native_first_positive=native[0]>0,sorted_leading_positive=lam[0]>0,
        native_first_two_nonnegative=None if len(native)<2 else K.exact(values[0])+K.exact(values[1])>=0,
        native_first_two_sum=None if len(native)<2 else K.number(K.exact(values[0])+K.exact(values[1])),
        sorted_first_two_sum=None if len(values)<2 else K.number(sum(sorted(map(K.exact,values),reverse=True)[:2])),
        sorted_first_two_nonnegative=None if len(values)<2 else sum(sorted(map(K.exact,values),reverse=True)[:2])>=0,
        literal_binary64=dict(sorted_to_original_index=order, sorted_values=lam,cumulative_sums=sums,j=j,dimension=point),
        exact_decimal_minus_exact_binary_dimension=None if de is None or bi is None else K.number(de-bi),
        literal_minus_exact_binary_dimension=None if point is None or bi is None else K.number(K.exact(point)-bi))
    return out


def cell(row, *, expected_index, expected_coordinates, k, expected_checkpoint_steps, record_steps):
    if row['index'] != expected_index or row['k'] != k or len(row['exponents']) != k:
        raise ValueError('cell identity or retained exponent count mismatch')
    if any(float(row[key])!=float(val) for key,val in zip(('log10_temperature','alpha'),expected_coordinates)):
        raise ValueError('cell coordinates disagree with original selected cells')
    cps=row['checkpoints']
    if [c['steps'] for c in cps] != expected_checkpoint_steps or any(len(c['exponents'])!=k for c in cps):
        raise ValueError('checkpoint order/count differs from original source/card')
    if len(set(expected_checkpoint_steps))!=len(expected_checkpoint_steps) or any(s<=0 or s>record_steps for s in expected_checkpoint_steps):
        raise ValueError('invalid frozen checkpoint selection')
    final=spectrum(row['exponents']); checkpoints=[]
    for cp in cps:
        checked=spectrum(cp['exponents']); cf=dimension(checked['saved_decimal']); ff=dimension(final['saved_decimal'])
        raw=[K.number(K.exact(a)-K.exact(b)) for a,b in zip(cp['exponents'],row['exponents'])] if checked['finite'] and final['finite'] else None
        sortdiff=None
        if checked['finite'] and final['finite']:
            sortdiff=[K.number(a-b) for a,b in zip(sorted(map(K.exact,cp['exponents']),reverse=True), sorted(map(K.exact,row['exponents']),reverse=True))]
        checkpoints.append(dict(steps=cp['steps'],spectrum=checked,native_column_minus_final=raw,sorted_rank_minus_final=sortdiff,
            dimension_minus_final=None if cf is None or ff is None else K.number(cf-ff),
            final_step_checkpoint_exactly_matches=cp['exponents']==row['exponents'] if cp['steps']==record_steps else None))
    return dict(index=expected_index,log10_temperature=str(row['log10_temperature']),alpha=str(row['alpha']),
        k=k,record_steps=record_steps,entropy_mean_original=str(row['entropy_mean']),original_record=row,
        final=final,checkpoints=checkpoints)


def group(rows):
    """Every passed row remains in both the group ledger and its denominator."""
    supported=[r['final']['saved_decimal'] for r in rows]
    positive=[i for i,r in enumerate(rows) if r['final'].get('sorted_leading_positive') is True]
    positive_results=[supported[i] for i in positive]
    out=dict(total_completed_rows=len(rows),row_keys=[r['key'] for r in rows],
        positive_leading_row_keys=[rows[i]['key'] for i in positive],positive_leading_count=len(positive),
        nonpositive_or_invalid_row_keys=[r['key'] for i,r in enumerate(rows) if i not in positive],
        native_first_positive_count=sum(r['final'].get('native_first_positive') is True for r in rows),
        positive_exponent_counts=[dict(key=r['key'],count=r['final'].get('positive_count_native')) for r in rows],
        sorted_first_two_nonnegative_row_keys=[r['key'] for r in rows if r['final'].get('sorted_first_two_nonnegative') is True],
        all_supported_dimension_range=K.dimension_range(supported,3),
        positive_leading_dimension_range=K.dimension_range(positive_results,3),
        positive_leading_dimension_range_2dp=K.dimension_range(positive_results,2),
        positive_range_local_to_group_row_indices=positive)
    deltas=[]
    for r in rows:
        for cp in r['checkpoints']:
            if cp['dimension_minus_final'] is not None:
                deltas.append((r['key'],cp['steps'],abs(fraction(cp['dimension_minus_final']))))
    maximum=max((v for _,_,v in deltas),default=None)
    out['checkpoint_dimension_differences_by_steps']={}
    for step in sorted({step for _,step,_ in deltas}):
        subset=[(key,v) for key,s,v in deltas if s==step]; high=max(v for _,v in subset)
        out['checkpoint_dimension_differences_by_steps'][str(step)]=dict(supported_count=len(subset),
            maximum_absolute_difference=K.number(high),row_keys_at_maximum=[key for key,v in subset if v==high])
    out['maximum_absolute_checkpoint_dimension_difference']=None if maximum is None else K.number(maximum)
    out['maximum_checkpoint_difference_locations']=[dict(key=key,steps=step) for key,step,v in deltas if v==maximum]
    return out
