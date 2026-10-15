"""Single finite Task16 saved-array replay; external lifecycle belongs to Inference."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time

HERE = Path(__file__).resolve().parent
DATA = Path('/data2/neural-shrimp-data/explore/llm-soft-loop/pod64')
EXPECTED = {
    'q05-alpha0': (512, 1, 64), 'llama8-alpha0': (512, 1, 64),
    'q7-alpha0': (512, 1, 64), 'olmo7-alpha0': (512, 1, 64),
    **{f'q05-{x}': (128, 128, 48) for x in ('random1', 'random2', 'random3', 'philosophy', 'gratitude')},
    **{f'llama8-layer16-{x}': (64, 64, 32) for x in ('concept', 'random1', 'random2')},
    **{f'llama8-layer19-{x}': (64, 64, 32) for x in ('concept', 'saeH', 'saeR', 'gauss')},
}
CAPS = {'cpu_threads': 1, 'nice': 10, 'external_seconds': 120, 'hard_as_bytes': 4 * 1024**3,
        'output_bytes': 128 * 1024**2, 'startup_memavailable_gib': 20, 'runtime_memavailable_gib': 16}


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024**2), b''):
            h.update(block)
    return h.hexdigest()


def need(ok, message):
    if not ok:
        raise ValueError(message)


def pinned(row):
    p = Path(row['path'])
    need(p.is_file() and p.stat().st_size == row['bytes'] and sha(p) == row['sha256'], 'authority hash/size: ' + str(p))
    return json.loads(p.read_text())


def specifications(meta, authority, commands, refs):
    """Adapt immutable metadata only; no scientific array values are used here."""
    need(meta['datasets'] == 16 and len(meta['segments']) == 19, 'fixed inventory count')
    need(authority['header_inventory']['sha256'] == refs['inventory']['sha256'] == commands['header_inventory']['sha256'], 'header association')
    need(commands['prospective_root_review']['sha256'] == refs['join_policy']['sha256'], 'join policy association')
    need(authority['random_partition_authority']['sha256'] == refs['commands']['sha256'], 'command authority association')
    ids = [s['segment_id'] for s in meta['segments']]
    need(ids == [f'segment-{i:02d}' for i in range(19)], 'exact ordered segment IDs')
    need(set(s['dataset_id'] for s in meta['segments']) == set(EXPECTED), 'exact dataset IDs')
    result = []
    for name, (nt, na, maximum) in EXPECTED.items():
        source = [s for s in meta['segments'] if s['dataset_id'] == name]
        random = name.startswith('q05-random')
        need(len(source) == (2 if random else 1), 'dataset segment count')
        segments = []
        for k, s in enumerate(source):
            matches = [(r, b) for r in authority['records'] for b in r['selected_member_bindings'] if b['segment_id'] == s['segment_id']]
            need(len(matches) == 1, 'unique original segment authority')
            run, member = matches[0]
            need(run['recorded_termination_verified'] is True and run['hash_verified'] is True, 'original recorded closure')
            need(member['matches_header_inventory'] is True and member['current_path'] == s['path'] and
                 member['sha256'] == s['sha256'] and member['bytes'] == s['bytes'], 'original member binding')
            partial = random and k == 0
            shape = [128 if partial else (32 if random else nt), na]
            need(s['axes']['log10_temperature']['shape'] == [shape[0]] and s['axes']['alpha']['shape'] == [na], 'frozen axis shape')
            need(s['filename'] == ('scan-partial.npz' if partial else 'scan.npz'), 'fixed schema filename')
            need(s['members']['lyapunov']['shape'] == ([(shape[0] * na)] if partial else shape) and
                 s['members']['period']['shape'] == s['members']['lyapunov']['shape'], 'header science shape')
            need(s['members']['lyapunov']['dtype'] == '<f8' and s['members']['period']['dtype'] == '<i8', 'recorded native dtypes')
            if s['scan_args'] is not None:
                need(s['scan_args']['max_period'] == maximum and s['scan_args']['resolution'] == shape, 'recorded scan limits')
            count = shape[0] * na
            selected = [0, 96 * na] if partial else [0, count]
            segments.append({'path': str(Path(s['path']).relative_to(DATA)), 'sha256': s['sha256'], 'bytes': s['bytes'],
                'shape': shape, 'layout': 'partial' if partial else 'full', 'expected_done': s['done'] if partial else count,
                'row_offset': 96 if random and k == 1 else 0, 'selected_local_flat_range': selected,
                'temperature_hex': s['axes']['log10_temperature']['float_hex'], 'alpha_hex': s['axes']['alpha']['float_hex'],
                'original_attempt_outcome': {'runner_returncode': run['runner_returncode'], 'status_lines': run['status_lines']},
                'outcome_authority': {'run_id': run['run_id'], 'receipt': run['original_receipt'], 'archive': run['archive'],
                    'member': member, 'source_commit': run['commit'], 'termination_recorded': True},
                'scan_args': s['scan_args'], 'original_jobset_args': run['upload']['jobset_args']})
        result.append({'id': name, 'shape': [nt, na], 'max_period': maximum, 'axis_segment': 0, 'segments': segments,
            'ordinal_amendment': refs['join_policy']['sha256'] if random else None,
            'metadata': {'original_args_by_segment': [s['scan_args'] for s in segments],
                'original_commands_by_segment': [s['original_jobset_args'] for s in segments],
                'scope': 'Frozen named directions; gratitude is not asserted norm-matched to philosophy.'}})
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', required=True); parser.add_argument('--output', required=True)
    args = parser.parse_args()
    plan = json.loads(Path(args.plan).read_text())
    need(plan['schema'] == 'task16-saved-scans-v1' and plan['status'] == 'reviewed', 'prepared/unknown plan is unlaunchable')
    need(plan['resources'] == CAPS and str(Path(args.output)) == plan['output'], 'fixed output/resource contract')
    need(isinstance(plan['source_commit'], str) and len(plan['source_commit']) == 40 and all(c in '0123456789abcdef' for c in plan['source_commit']), 'exact source commit required')
    need(set(plan['sources']) == {'run.py', 'tabulate.py'}, 'source closure')
    for name, expected in plan['sources'].items():
        need(sha(HERE / name) == expected, 'source hash: ' + name)
    priority = os.getpriority(os.PRIO_PROCESS, 0)
    need(priority <= 10, 'priority stronger than exact nice10 envelope')
    if priority < 10:
        os.nice(10 - priority)
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    need(hard == resource.RLIM_INFINITY or hard >= CAPS['hard_as_bytes'], 'inherited hard AS too small')
    resource.setrlimit(resource.RLIMIT_AS, (CAPS['hard_as_bytes'], CAPS['hard_as_bytes']))
    for key in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[key] = '1'
    sys.path.insert(0, str(HERE))
    import tabulate as T
    out = Path(args.output); out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic(); total_bytes = 0
    def write(name, value, lines=False):
        nonlocal total_bytes
        with open(out / name, 'x') as f:
            for row in value if lines else [value]:
                text = json.dumps(row, allow_nan=False, separators=(',', ':')) + '\n'
                total_bytes += len(text.encode())
                need(total_bytes <= CAPS['output_bytes'] - 65536, 'output bound including closing reserve')
                f.write(text)
    status = {'computational_completed': False, 'all_datasets_admitted': False,
              'source_commit': plan['source_commit'], 'plan_sha256': sha(args.plan), 'sources': plan['sources']}
    try:
        refs = plan['authorities']
        for ref in refs.values():
            need(Path(ref['path']).stat().st_size == ref['bytes'] and sha(ref['path']) == ref['sha256'], 'pre authority hash/size')
        meta, authority, commands = [pinned(refs[k]) for k in ('inventory', 'runs', 'commands')]
        norms = pinned(refs['vector_metadata'])
        policy = refs['join_policy']; need(sha(policy['path']) == policy['sha256'] and Path(policy['path']).stat().st_size == policy['bytes'], 'join note pin')
        specs = specifications(meta, authority, commands, refs)
        write('input-admission.json', {'authorities': refs, 'specifications': specs,
              'limits': authority['trust_limits'], 'vector_metadata': norms, 'original_git_archive_checks': 'inherited from accepted Inference metadata authority; not rerun'})
        reports, rows_by_id = [], {}
        for spec in specs:
            report, rows, sources = T.tabulate_dataset(DATA, spec)
            reports.append(report); rows_by_id[spec['id']] = rows
            write(spec['id'] + '-cells.jsonl', rows, True)
            write(spec['id'] + '-sources.jsonl', sources, True)
            write(spec['id'] + '-summary.json', report)
            need(time.monotonic() - started < CAPS['external_seconds'], 'worker elapsed bound')
        groups = [{'id': prefix, 'datasets': [s['id'] for s in specs if s['id'].startswith(prefix) and 'alpha0' not in s['id']],
                   'metadata_basis': 'actual-coordinate intersection; named directions, no population or equal-norm inference'}
                  for prefix in ('q05-', 'llama8-layer16-', 'llama8-layer19-')]
        write('report.json', {'schema': 'task16-saved-report-v1', 'prior_producer_summary_exposure': True,
              'scope': 'independent unblinded arithmetic reproduction, not fresh confirmation',
              'datasets': reports, 'exact_common_valid_groups': T.matched_groups(rows_by_id, groups),
              'vector_metadata_authority': refs['vector_metadata'],
              'expected_selected_cells': 112640, 'actual_cell_records': sum(len(x) for x in rows_by_id.values())})
        for ref in refs.values():
            need(sha(ref['path']) == ref['sha256'], 'post authority hash')
        for name, expected in plan['sources'].items():
            need(sha(HERE / name) == expected, 'post source hash')
        for spec in specs:
            for seg in spec['segments']:
                T.authenticate(DATA, seg)
        status.update(computational_completed=True, all_datasets_admitted=all(r['admitted'] for r in reports))
    except Exception as e:
        status['error'] = type(e).__name__ + ': ' + str(e)
    status.update(elapsed_seconds=time.monotonic() - started, peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  numpy_version=T.np.__version__, python_version=sys.version, output_bytes_before_status=total_bytes)
    if status['elapsed_seconds'] >= CAPS['external_seconds']:
        status['computational_completed'] = False
        status['resource_failure'] = 'elapsed_seconds'
    (out / 'status.json').write_text(json.dumps(status, allow_nan=False, indent=2) + '\n')
    return 0 if status['computational_completed'] and status['all_datasets_admitted'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
