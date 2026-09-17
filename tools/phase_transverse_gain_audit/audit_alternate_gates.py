#!/usr/bin/env python3
"""Bounded saved-only companion to immutable audit_gain/report; no changed gates.

Separate the units of strict arithmetic discrepancies and determine whether
the independently reconstructed physical/activation/noise approximation gates
have the same classifications. This supplements rather than replaces the
first audit and its retained failures. No producer imports, ODE or fitting.
"""
from collections import Counter
import json
from pathlib import Path
import time
import numpy as np
import audit_gain as a


def main():
    begin=time.monotonic();base=Path(__file__).parent;original=a.read(base/'report.json')
    if a.sha(base/'report.json')!='0196cdd659f50bf1fbd34a9a42aaf4461e94653ceaed46f242001d0d870b6eaf':raise ValueError('Primary audit changed')
    if a.sha(a.RAW/'receipt.json')!=a.RECEIPT:raise ValueError('Raw receipt changed')
    evidence=a.Audit();bundle=a.packet(a.RAW/'inputs-seal.json',evidence)
    rows=a.read(a.RAW/'terminal-records.json');results=[];maxima={};changes=[];strict_units=Counter();strict_category=Counter()
    for failure in original['alternate_arithmetic']['failures']:
        path=failure['check'];error=failure['details']['maximum_absolute_difference']
        group='diagnostic_tolerance_ratio' if path.endswith('maximum_tolerance_ratio') else 'relative_flow_ratio' if path.endswith('maximum_relative_physical_flow_transport') else 'alternate_direct_residual' if '/alternate_direct' in path else 'state_array' if '/state/' in path else 'activation_array' if '/activation/' in path else 'observable_scalar' if '/observable/' in path else 'reference_defect_diagnostic' if '/reference_defects/' in path else 'other'
        strict_units[group]+=1
        entry=maxima.setdefault('strict_failed_'+group,dict(maximum_absolute_difference=0.,location=None))
        if error is not None and error>entry['maximum_absolute_difference']:entry.update(maximum_absolute_difference=error,location=path)
    for failure in original['alternate_arithmetic_categories']['failures']:
        path=failure['check'];strict_category['gate_boolean' if path.endswith('/passed') else 'argmax_location' if '/maximum_location/' in path else 'other']+=1
    for cell,row in zip(a.census(),rows,strict=True):
        saved=a.packet(a.RAW/row['packet_path'],evidence);ctx=bundle['contexts'][f'{cell["setting"]}--p{cell["phase_index"]}']
        common=a.context_arrays(bundle['references'][cell['setting']],bundle['propagators'][ctx['id']],ctx,ctx['readouts'][str(cell['delay_cycles'])])
        direction=ctx['directions'][cell['direction']];unit=a.derive(common,direction)
        for item,d,original_math in [(None,direction,saved['unit'])]+[(pulse,pulse['realized_delta'],pulse['mathematics']) for pulse in saved['compatibility']]:
            derived=unit if item is None else a.derive(common,d)
            prefix=cell['id']+'/'+('unit' if item is None else item['id'])
            for quantity in ('state','activation'):
                for key,value in derived[quantity].items():
                    delta=float(np.max(np.abs(value-original_math[quantity][key])))
                    entry=maxima.setdefault(('unit' if item is None else 'realized')+'_'+quantity,dict(maximum_absolute_difference=0.,location=None))
                    if delta>entry['maximum_absolute_difference']:entry.update(maximum_absolute_difference=delta,location=prefix+'/'+key)
            for key,value in derived['observable'].items():
                delta=abs(value-original_math['observable'][key]);normdelta=delta/(1e-4*derived['noise_gain'])
                for name,size in [('readout_q',delta),('primary_standardized_readout',normdelta)]:
                    entry=maxima.setdefault(('unit' if item is None else 'realized')+'_'+name,dict(maximum_absolute_difference=0.,location=None))
                    if size>entry['maximum_absolute_difference']:entry.update(maximum_absolute_difference=size,location=prefix+'/'+key)
            if item is None:continue
            alternative=a.approximation(derived);record=dict(cell=cell['id'],pulse=item['id'],original={key:v['passed'] for key,v in item['approximation'].items()},independent={key:v['passed'] for key,v in alternative.items()})
            results.append(record)
            if record['original']!=record['independent']:changes.append(dict(**record,independent_values=alternative,original_values=item['approximation']))
    report=dict(scope=__doc__,primary_audit=a.file_binding(base/'report.json'),source=a.file_binding(Path(__file__)),arithmetic_source=a.file_binding(base/'audit_gain.py'),raw_receipt_sha256=a.RECEIPT,
        independent_approximation_rows=len(results),independent_approximation_gate_tests=3*len(results),changed_approximation_classifications=changes,
        original_failed_approximation_rows=sum(not all(row['original'].values()) for row in results),independent_failed_approximation_rows=sum(not all(row['independent'].values()) for row in results),
        maxima=maxima,strict_numeric_failure_units=dict(strict_units),strict_category_failure_types=dict(strict_category),
        alternate_direct_boolean_differences=[r for r in original['alternate_arithmetic_categories']['failures'] if r['check'].endswith('/passed')],
        codec_evidence_checks=evidence.checks,codec_evidence_failures=evidence.failures,elapsed_seconds=time.monotonic()-begin,new_odes=0,new_fits=0)
    output=base/'alternate-gates-and-error-units.json'
    with output.open('x') as f:json.dump(report,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(dict(output=str(output),sha256=a.sha(output),bytes=output.stat().st_size,gate_rows=len(results),changed=len(changes),codec_failures=len(evidence.failures),elapsed_seconds=report['elapsed_seconds'])))


if __name__=='__main__':main()
