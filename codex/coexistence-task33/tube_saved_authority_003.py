"""One saved-only Task33 audit; no producer, model, backend or cache decoding.

Composes the four frozen helpers. Historical arithmetic validity is not inferred
from hashes: even complete consistency returns only L_candidate and false
scientific admission flags. Before/after checks are not an atomic snapshot.
"""
import argparse
import ast
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import resource
import stat
import subprocess
import traceback
import types

ROOT = Path('/home/d7082791602/.codex/worktrees/llm-hostile-fold-crosscheck/neural-shrimp')
D = ROOT / 'codex/coexistence-task33'
B = Path('/data2/neural-shrimp-data/codex/coexistence')
TARGET = B / 'tube-target-001'
OUT = B / 'tube-target-001-audit-003'
CARD = D / 'tube-target-001-audit-003-card.json'
PLAN = D / 'tube-target-001-audit-003-plan.json'
SELF = D / 'tube_saved_authority_003.py'
SUP = Path(str(OUT) + '-supervision')
PYTHON = '/data2/neural-shrimp/.venv/bin/python'
ORIGINAL_SOURCE = 'ed679786a0a231bd9c8d05c9e1cc0d83ca17c537'
HELPERS = {
    'tube_saved_files.py': '8feace44ca4fe1f8226ae18e344670eab6f8dca5b52e39fb25b4860030ec5561',
    'tube_saved_execution.py': '1435a1b13c4a36095c2f28a369e4c7c717e186103ccbd6a60e48485bf9cc95a4',
    'tube_saved_assets.py': '1068ab9db8e8c2deed22aa198b5bfce9ab3e58ed48037a591c14bf7564b93169',
    'tube_saved_scalar.py': '90b0518f4bc976f84c6d56adf45c90726f617fa55e316ab3b7be7e50d9dcc096',
}
EXPECTED_PIN = dict(path=str(D / 'tube-target-001-audit-001-expected.json'), bytes=6598,
                    sha256='6d0870c84d558e915a2247337b4fb0f4c44e1810a98b7b06afdeec569de79ee2')
CLOSURE_PIN = dict(path=str(D / 'tube-target-001-original-closure.json'), bytes=15911,
                   sha256='53ece975b074890ca042c9ea99e1c52e92d19c4f9255e290e5d442ba4d0db753')
CACHE_PIN = dict(path=str(B / 'prefix-001/prefix-cache.json.gz'), bytes=68579136,
                 sha256='52ead55348240b014ab3d0d41a925a58dfc477ceb8fdd2d3801e4ff7c5996634')
PREFIX_INPUT_PIN = dict(path=str(B / 'prefix-001/inputs.json'), bytes=1117,
                        sha256='726c125e1f20d85ab9396b70bc5aad6dab8ad6589c24d2a71e05e976796cf67d')
REPORT_PIN = dict(path=str(B / 'thin-pilot-001-audit-002/result.json'), bytes=51399,
                  sha256='e17d77123a04b01d325fcb83cf03e5997e843cc3c33577c398ece4c401866375')
# One historical producer input only; this is not a fallback for changed files.
THIN_PRODUCER_PIN = dict(path=str(ROOT / 'tools/explore_llm_soft_loop.py'), bytes=122549,
                        sha256='28418408507409f52df9217c71255131a1de8cfe7d059a35761971738883e120')
THIN_CLASS_SHA256 = 'c590a4c0bb2ee3742333d66cfa204add121b63bcb43d6446f96e8cd047f3e50b'
THIN_AUTHORITY = dict(
    source_commit='e8db350006098683ab170645bf7ee7c317ca3b56',
    card=dict(path=str(D / 'thin-pilot-001-card.json'), bytes=4859,
              sha256='33922427cb39bb72662122ae06c461acdbfc9314344e4e9f8a468d2b47164213'),
    plan=dict(path=str(D / 'thin-pilot-001-plan.json'), bytes=9344,
              sha256='bf07856a9637d1f174bcd4dfe65f66d2055091d59ac8758f6001ca0c3ef4ac90'),
    receipt=dict(path=str(B / 'thin-pilot-001-supervision/receipt.json'), bytes=9588,
                 sha256='8cc26e4a6eba2227bf6712588c7ff4fce381be855cb8b6b04b1fd6fbe13f4711'),
    admission=dict(path=str(B / 'thin-pilot-001-supervision/admission.json'), bytes=6296,
                   sha256='18ef3c15b21502ae758b06c37c2e93038a4953ae53f546a0bacd2ff8c27c0c17'),
    lane='thin', output=str(B / 'thin-pilot-001'),
    argv=[PYTHON, '-I', '-B', str(D / 'thin_pilot.py'), '--card',
          str(D / 'thin-pilot-001-card.json'), '--out', str(B / 'thin-pilot-001')])
LIMITS = dict(as_bytes=4 * 2**30, cpu_seconds=300, wall_seconds=360, output_bytes=32 * 2**20)
FLAGS = dict(scientific_verdict_admitted=False, operator_bound_admitted=False,
             conditional_operator_bound_admitted=False, trajectory_enclosure=False,
             reachability_proved=False, certificate_passed=False)
PREFIX_IDS = [151644,8948,198,2610,525,458,15235,7404,448,2441,15235,13,1446,614,4583,11290,13,31733,1910,311,
              22729,8820,498,1366,13,151645,198,151644,872,198,9707,0,151645,198,151644,77091,198]


def need(ok, message):
    if not ok:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def load_helpers():
    """Execute only literal hash-checked helper bytes, never producer modules."""
    modules = []
    for name, expected in HELPERS.items():
        path = D / name; flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_DIRECTORY
        directory = os.open('/', flags)
        try:
            for part in path.parent.parts[1:]:
                child = os.open(part, flags, dir_fd=directory); os.close(directory); directory = child
            fd = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory)
            try:
                info = os.fstat(fd)
                need(stat.S_ISREG(info.st_mode) and info.st_nlink == 1 and info.st_size <= 65536, 'bounded regular helper required')
                payload = b''
                while len(payload) <= 65536:
                    block = os.read(fd, min(8192, 65537 - len(payload)))
                    if not block: break
                    payload += block
            finally: os.close(fd)
        finally: os.close(directory)
        need(sha(payload) == expected, 'frozen helper bytes differ')
        module = types.ModuleType(name.removesuffix('.py')); module.__file__ = str(path)
        exec(compile(payload, str(path), 'exec'), module.__dict__)
        modules.append(module)
    return tuple(modules)  # files, execution, assets, scalar


class ClosedOriginalFailure(ValueError):
    def __init__(self, evidence):
        super().__init__('authenticated original is failed/incomplete; no scalar helper called')
        self.evidence = evidence


class Evidence:
    """Bounded original bytes, strict same-byte decoding and complete flat sets."""
    def __init__(self, files, execution, assets):
        self.f, self.e, self.a = files, execution, assets
        self.pins, self.payloads, self.inventories, self.git_checks = {}, {}, [], []
        self.thin_producer_bytes = None
        self.historical_input_provenance = []
        self.cache_observation = None
        self.assets_result = None

    def thin_producer(self, authority, pin):
        """Read one original Git source; never label it a current-file check."""
        self.f.validate_pin(pin)
        need(authority == THIN_AUTHORITY and pin == THIN_PRODUCER_PIN,
             'exact original thin producer authority and pin required')
        need(pin['path'] not in self.pins and pin['path'] not in self.payloads,
             'historical producer cannot share the current-file cache')
        if self.thin_producer_bytes is None:
            relative = Path(pin['path']).relative_to(ROOT).as_posix()
            payload = self.sources(authority['source_commit'], [pin])[relative]
            need(type(payload) is bytes and len(payload) == pin['bytes'] and sha(payload) == pin['sha256'],
                 'historical thin producer bytes differ')
            text = payload.decode('utf-8')
            nodes = [node for node in ast.parse(text).body if isinstance(node, ast.ClassDef) and node.name == 'SoftLoop']
            need(len(nodes) == 1 and sha(ast.get_source_segment(text, nodes[0]).encode()) == THIN_CLASS_SHA256,
                 'historical thin producer class differs')
            self.thin_producer_bytes = payload
            self.historical_input_provenance.append(dict(
                source_commit=authority['source_commit'], source_pin=pin.copy(),
                original_card=authority['card'].copy(), original_plan=authority['plan'].copy(),
                class_source_segment_sha256=THIN_CLASS_SHA256, current_file_authenticated=False,
                scope='Exact original Git bytes only; current producer file is neither read nor substituted.'))
        return self.thin_producer_bytes

    def assets(self):
        if self.assets_result is None: self.assets_result = self.a.check_target_assets()
        return self.assets_result

    def verify_pins(self, rows):
        """Authenticate every declared input; only literal cache/assets stream."""
        definitions = self.a._target_definitions()
        asset_pins = {value['asset']['path']:value['asset'] for value in definitions.values()}
        for pin in self.e.pins(rows).values():
            if pin['path'] == CACHE_PIN['path']:
                need(pin == CACHE_PIN, 'declared original cache pin differs')
                if self.cache_observation is None: self.stream_cache()
            elif pin['path'] in asset_pins:
                need(pin == asset_pins[pin['path']], 'declared original asset pin differs'); self.assets()
            else: self.read(pin)

    def read(self, pin, decode=False):
        self.f.validate_pin(pin)
        name = pin['path']
        need(name not in self.pins or self.pins[name] == pin, 'conflicting pin for one original path')
        if name not in self.payloads:
            need(sum(p['bytes'] for p in self.pins.values()) + pin['bytes'] <= 256 * 2**20, 'saved-byte budget exceeded')
            self.payloads[name] = self.f.read_pinned_bytes(pin); self.pins[name] = pin.copy()
        payload = self.payloads[name]
        return self.f.strict_json_bytes(payload) if decode else payload

    def current(self, path):
        """Capture own runtime metadata; decode only the subsequently pinned bytes."""
        path = self.f._path(str(path))
        with self.f._directory(path.parent) as parent:
            info = os.stat(path.name, dir_fd=parent, follow_symlinks=False); self.f._regular(info)
            need(info.st_size <= self.f.MAX_FILE_BYTES, 'bounded runtime record required')
            fd = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent)
            try:
                need(self.f._version(os.fstat(fd)) == self.f._version(info), 'runtime capture identity differs')
                payload = self.f._read_fd(fd, info.st_size)
                need(self.f._version(os.fstat(fd)) == self.f._version(info), 'runtime capture changed')
            finally: os.close(fd)
        pin = dict(path=str(path), bytes=len(payload), sha256=sha(payload))
        return pin, self.read(pin, True)

    def blob(self, source, pin):
        self.e.digest(source, 40); self.f.validate_pin(pin)
        relative = self.f._path(pin['path']).relative_to(ROOT).as_posix()
        spec = source + ':' + relative
        size = subprocess.check_output(['git', 'cat-file', '-s', spec], cwd=ROOT, timeout=15).strip()
        need(size.isdigit() and int(size) == pin['bytes'], 'original Git blob size differs')
        payload = subprocess.check_output(['git', 'cat-file', 'blob', spec], cwd=ROOT, timeout=15)
        need(len(payload) == pin['bytes'] and sha(payload) == pin['sha256'], 'original Git bytes differ')
        return payload

    def sources(self, source, pins):
        rows = self.f._pins(pins)
        blobs = {str(Path(name).relative_to(ROOT)): self.blob(source, pin) for name, pin in rows.items()}
        self.f.check_source_blobs(str(ROOT), list(rows.values()), blobs)
        self.git_checks.append(dict(source_commit=source, files=list(rows.values())))
        return blobs

    def historical_source(self, source, filename, digest):
        self.e.digest(source,40); self.e.digest(digest); relative = self.f._path(filename).relative_to(ROOT).as_posix()
        size = subprocess.check_output(['git','cat-file','-s',source+':'+relative],cwd=ROOT,timeout=15).strip()
        need(size.isdigit() and int(size) <= self.f.MAX_FILE_BYTES, 'bounded original historical source required')
        pin = dict(path=filename,bytes=int(size),sha256=digest)
        return self.sources(source,[pin])[relative]

    def snapshot(self, root, rows):
        root = self.f._path(str(root)); bypath = self.e.pins(rows)
        need(all(Path(path).parent == root for path in bypath), 'direct flat original inventory required')
        with self.f._directory(root) as fd:
            before = self.f._version(os.fstat(fd)); names = []
            with os.scandir(fd) as entries:
                for entry in entries:
                    need(len(names) < len(rows), 'extra original output entry'); names.append(entry.name)
            need({str(root / name) for name in names} == set(bypath), 'complete original directory differs')
            versions, identities = {}, set()
            for name in names:
                info = os.stat(name, dir_fd=fd, follow_symlinks=False); self.f._regular(info)
                need((info.st_dev, info.st_ino) not in identities, 'original output inode alias')
                identities.add((info.st_dev, info.st_ino)); versions[name] = self.f._version(info)
            need(self.f._version(os.fstat(fd)) == before, 'original directory changed during snapshot')
        return before, versions

    def inventory(self, root, rows):
        before = self.snapshot(root, rows)
        for pin in rows:
            if pin['path'] == CACHE_PIN['path']:
                need(pin == CACHE_PIN, 'original cache pin differs')
                if self.cache_observation is None: self.stream_cache()
            else: self.read(pin)
        need(self.snapshot(root, rows) == before, 'original inventory changed during authentication')
        self.inventories.append((str(root), rows, before))
        return self.e.pins(rows)

    def stream_cache(self):
        """Only the fixed ordinary prefix cache: no symlink or decompression."""
        pin = CACHE_PIN; path = Path(pin['path'])
        with self.a._parent(path) as (parent, directories):
            before = os.stat(path.name, dir_fd=parent, follow_symlinks=False); self.f._regular(before)
            need(before.st_size == pin['bytes'], 'original prefix cache size differs')
            fd = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent)
            try:
                need(self.f._version(os.fstat(fd)) == self.f._version(before), 'cache open identity differs')
                digest = self.a._hash_fd(fd, pin['bytes'])
                need(self.f._version(os.fstat(fd)) == self.f._version(before) and
                     self.f._version(os.stat(path.name, dir_fd=parent, follow_symlinks=False)) == self.f._version(before), 'cache changed during streaming')
            finally: os.close(fd)
        with self.a._parent(path) as (parent, after_dirs):
            need(after_dirs == directories and self.f._version(os.stat(path.name, dir_fd=parent, follow_symlinks=False)) == self.f._version(before), 'cache canonical path changed')
        need(digest == pin['sha256'], 'full original prefix cache hash differs')
        observation = dict(pin=pin.copy(), directories=directories, version=list(self.f._version(before)))
        need(self.cache_observation is None or self.cache_observation == observation, 'prefix cache changed across audit')
        self.cache_observation = observation
        return observation

    def recheck(self):
        for pin in self.pins.values(): self.f.read_pinned_bytes(pin)
        if self.thin_producer_bytes is not None:
            need(self.blob(THIN_AUTHORITY['source_commit'], THIN_PRODUCER_PIN) == self.thin_producer_bytes,
                 'historical thin producer changed across audit')
        for root, rows, before in self.inventories:
            need(self.snapshot(root, rows) == before, 'whole original inventory changed before report')
        if self.cache_observation is not None: self.stream_cache()


def successful(receipt):
    need(type(receipt['returncode']) is int and receipt['returncode'] == 0 and receipt['failure'] is None and
         all(receipt.get(key) is True for key in ('worker_reaped', 'worker_identity_absent', 'worker_group_session_absent')),
         'successful complete original receipt required')


def source_join(io, source, plan, admission, card_pin):
    planned = io.e.pins(plan['source_files'])
    recorded = []
    for row in admission['source_blobs']:
        need(set(row) == {'relative_path', 'bytes', 'sha256'}, 'source blob schema differs')
        need(type(row['relative_path']) is str and str(Path(row['relative_path'])) == row['relative_path'] and
             '..' not in Path(row['relative_path']).parts, 'canonical source-relative blob path required')
        path = str(ROOT / row['relative_path']); io.e.path(path)
        need(Path(path).is_relative_to(ROOT) and not Path(row['relative_path']).is_absolute(), 'source blob escaped repository')
        recorded.append(dict(path=path, bytes=row['bytes'], sha256=row['sha256']))
    need(io.e.pins(recorded) == planned and planned.get(card_pin['path']) == card_pin, 'complete original source/card join differs')
    io.sources(source, list(planned.values()))
    return planned


def upstream_preflight(io, authority, rows):
    """Only the exact thin producer pin uses historical Git; all others stay live."""
    pins = io.e.pins(rows)
    historical_path = THIN_PRODUCER_PIN['path']
    if authority == THIN_AUTHORITY or historical_path in pins:
        need(authority == THIN_AUTHORITY and pins.get(historical_path) == THIN_PRODUCER_PIN,
             'exact thin historical preflight context and pin required')
        for pin in pins.values():
            if pin['path'] == historical_path: io.thin_producer(authority, pin)
            else: io.verify_pins([pin])
    else:
        io.verify_pins(list(pins.values()))


def original_execution(io, authority):
    need(type(authority) is dict and set(authority) == {'source_commit','card','plan','receipt','admission','lane','output','argv'}, 'complete upstream execution authority required')
    source = io.e.digest(authority['source_commit'], 40)
    card, plan, receipt, admission = [io.read(authority[k], True) for k in ('card','plan','receipt','admission')]
    io.sources(source, [authority['card'], authority['plan']])
    successful(receipt)
    lane = plan['lanes'][authority['lane']]
    argv = [plan['python'], '-I', '-B', lane['entry'], *lane['arguments']]
    need(plan['status'] == 'reviewed_for_execution' and plan['repository'] == str(ROOT) and plan['raw_root'] == str(B)
         and plan['python'] == PYTHON and type(plan['cpu_threads']) is int and plan['cpu_threads'] == 1, 'upstream reviewed allocation differs')
    need(argv == authority['argv'] == receipt['argv'] == admission['argv'] and
         authority['lane'] == receipt['lane'] == admission['lane'] and source == receipt['source_commit'] == admission['source_commit'], 'upstream source/lane/argv differs')
    need(lane['output'] == authority['output'] and Path(authority['output']).parent == B and
         authority['receipt']['path'] == lane['supervision'] + '/receipt.json' and
         authority['admission']['path'] == lane['supervision'] + '/admission.json', 'upstream allocation differs')
    ap = dict(admission['plan'])
    if not Path(ap['path']).is_absolute(): ap['path'] = str(ROOT / ap['path'])
    need(ap == authority['plan'] and io.e.limits(admission['limits']) == io.e.limits(lane['limits']) and
         set(admission['threads']) == set(io.e.THREADS) and all(v == '1' for v in admission['threads'].values()), 'upstream plan/resource binding differs')
    planned = source_join(io, source, plan, admission, authority['card'])
    preflight = io.e.pins(lane['preflight_pins'])
    for field in ('inputs','input_files'):
        if field in card:
            need(all(preflight.get(path) == pin for path,pin in io.e.pins(card[field]).items()), 'upstream declared inputs absent from preflight')
    upstream_preflight(io, authority, list(preflight.values()))
    files = io.inventory(authority['output'], receipt['output_files'])
    need(type(receipt['actual_output_bytes']) is int and receipt['actual_output_bytes'] == sum(p['bytes'] for p in files.values()), 'upstream output total differs')
    return dict(card=card, plan=plan, receipt=receipt, admission=admission, files=files, sources=planned)


def _required_sources(upstream, expected):
    for relative, digest in expected.items():
        need(upstream['sources'].get(str(ROOT / relative), {}).get('sha256') == digest, 'reviewed upstream source differs')


def join_upstream(io, card, scalar):
    """Authenticate successful audit002 first, then its exact successful thin run."""
    audit, thin = card['audit_execution'], card['thin_execution']
    need(audit['source_commit'] == '70b043192b775f2cc67860c4709bfc867de096eb' and
         audit['card']['path'] == str(D / 'thin-pilot-001-audit-002-card.json') and audit['card']['sha256'] == '538e6ea648fa1e115dc35eceb4392d0798da9bc19f4c77c91098ac227fed8dd7' and
         audit['plan']['path'] == str(D / 'thin-pilot-001-audit-002-plan.json') and audit['plan']['sha256'] == 'd719c9366fc84380fa4983fd9c579a425ed8faeb99c33aad29c6373c3f0292a6' and
         audit['output'] == str(B / 'thin-pilot-001-audit-002') and card['audit_result'] == REPORT_PIN, 'exact original audit002 authority required')
    au = original_execution(io, audit)
    auditor = 'b2a84b5022f71cadfce91e6fc21fa4ad9953bb0a1a0b0c1d62d24d43f0c7e4d5'
    _required_sources(au, {'codex/coexistence-task33/audit_thin_pilot_002.py': auditor})
    need(au['card']['schema'] == 'codex-task33-thin-audit-execution-card-v1' and au['card']['auditor_sha256'] == auditor and
         au['card']['original_execution'] == thin and au['card']['result_path'] == REPORT_PIN['path'] and
         set(au['files']) == {REPORT_PIN['path']} and au['files'][REPORT_PIN['path']] == REPORT_PIN, 'audit report not in exact original successful inventory')
    need(thin['source_commit'] == 'e8db350006098683ab170645bf7ee7c317ca3b56' and thin['lane'] == 'thin' and thin['output'] == str(B / 'thin-pilot-001') and
         thin['card']['path'] == str(D / 'thin-pilot-001-card.json') and thin['card']['sha256'] == '33922427cb39bb72662122ae06c461acdbfc9314344e4e9f8a468d2b47164213' and
         thin['plan']['path'] == str(D / 'thin-pilot-001-plan.json'), 'exact original thin authority required')
    need(thin['argv'] == [PYTHON,'-I','-B',str(D / 'thin_pilot.py'),'--card',thin['card']['path'],'--out',thin['output']], 'thin exact command differs')
    need(audit['argv'] == [PYTHON,'-I','-B',str(D / 'audit_thin_pilot_002.py'),'--source',thin['source_commit'],
         '--card',thin['card']['path'],'--card-sha256',thin['card']['sha256'],'--plan',thin['plan']['path'],'--plan-sha256',thin['plan']['sha256'],
         '--receipt-sha256',thin['receipt']['sha256'],'--admission-sha256',thin['admission']['sha256'],'--output',REPORT_PIN['path']], 'audit002 exact command differs')
    tu = original_execution(io, thin); tc = tu['card']
    _required_sources(tu, {'codex/coexistence-task33/thin_pilot.py':'a12fd306fb49392c0d4922b6c4d7aabec1dd56e0b0b835373ecf74287ff05cbc',
        'codex/coexistence-task33/thin_pseudo_orbit.py':'1e9974f2bd5cddfc0ddfd6b1038c7d046df7f8abdb6f061b67fb764a6189369e',
        'codex/crosscheck/qwen_map.py':io.e.SOURCE_HASHES['codex/crosscheck/qwen_map.py']})
    need(all(tc.get(k) == v for k,v in io.e.PREFIX.items()) and tc['alpha'] == .06468 and tc['log10T'] == -.148, 'thin exact cell/prefix differs')
    for key, value in (('precision',2048),('steps',20),('initial_token_id',198)): scalar.integer(tc[key],value,'thin '+key)
    names = {'admission.json','centre-000.json','report.json', *(f'step-{i:03d}.json' for i in range(1,21))}
    need(set(tu['files']) == {thin['output']+'/'+name for name in names}, 'successful complete20-step thin inventory required')
    report = io.read(REPORT_PIN, True); scalar.thin_flags(report)
    need(report['schema'] == 'codex-task33-thin-pilot-saved-audit-v1' and report['status'] == 'complete_success' and
         report['worker_failure'] is None and report['worker_completion_present'] is True and report['reachability_proved'] is False and
         report['original_partial_evidence_preserved'] is True, 'successful audit002 outcome required')
    scalar.integer(report['steps_checked'],20,'all audit rows'); need(len(report['steps']) == 20, 'all20 saved audit rows required')
    original = report['original_execution']
    expected_original = {k:thin[k] for k in ('source_commit','receipt','admission','card','plan')}
    expected_original.update(raw_root=thin['output'], output_files=tu['receipt']['output_files'], source_blobs=tu['admission']['source_blobs'],
                             returncode=0, failure=None, host_original_groups_absent=True, elapsed_seconds=tu['receipt']['external_elapsed_seconds'])
    need(original == expected_original, 'complete audit-to-thin original execution join differs')
    for index, row in enumerate(report['steps'], 1):
        scalar.integer(row['step'],index,'audit row index'); scalar.integer(row['component_count'],896,'audit full dimension')
        need(row['file'] == tu['files'][thin['output']+f'/step-{index:03d}.json'] and set(row['checks']) == scalar.STEP_CHECKS and
             all(flag is True for flag in row['checks'].values()), 'every inherited audit row must be gated')
    first = tu['files'][thin['output']+'/centre-000.json']; step = tu['files'][thin['output']+'/step-001.json']
    initial, point = io.read(first,True), io.read(step,True)
    authority = tc['producer_initial_state_authority']
    need(report['initial_state']['source_authority'] == authority and authority['source']['path'] == str(ROOT/'tools/explore_llm_soft_loop.py') and
         authority['source']['sha256'] == '28418408507409f52df9217c71255131a1de8cfe7d059a35761971738883e120' and
         authority['method'] == 'SoftLoop.prepare: embedding of last prefix token, K=1' and
         authority['scope'] == 'exact stored checkpoint constants; not bitwise identity with floating execution' and
         authority['original_source_commit'] == '019168d9c28b92ae80f7f2766592dccbfae69e2d' and
         authority['original_source_sha256'] == 'ead3b44a82c556c1a376f3e58c389cd3fb2f7329786ef0a712a3d92e7a920843' and
         authority['fixed_start_class_sha256'] == 'c590a4c0bb2ee3742333d66cfa204add121b63bcb43d6446f96e8cd047f3e50b', 'original SoftLoop authority differs')
    need(io.e.pins(tc['inputs']).get(authority['source']['path']) == authority['source'], 'fixed-start source missing from thin inputs')
    producer = io.thin_producer(thin, authority['source'])
    need(io.blob(thin['source_commit'],authority['source']) == producer, 'thin-pinned producer source differs from original Git')
    historical = io.historical_source(authority['original_source_commit'],authority['source']['path'],authority['original_source_sha256'])
    for payload in (producer,historical):
        text = payload.decode('utf-8'); nodes = [node for node in ast.parse(text).body if isinstance(node,ast.ClassDef) and node.name == 'SoftLoop']
        need(len(nodes) == 1 and sha(ast.get_source_segment(text,nodes[0]).encode()) == authority['fixed_start_class_sha256'], 'original fixed-start class witness differs')
    internal = io.read(tu['files'][thin['output']+'/admission.json'],True); scalar.thin_flags(internal)
    need(internal['card_path'] == thin['card']['path'] and internal['card_sha256'] == thin['card']['sha256'] and
         internal['driver_sha256'] == tu['sources'][str(D/'thin_pilot.py')]['sha256'] and internal['loop_sha256'] == tu['sources'][str(D/'thin_pseudo_orbit.py')]['sha256'] and
         internal['producer_initial_state_authority'] == authority and internal['prefix_authority'] == io.e.PREFIX and
         internal['alpha_ratio'] == list((.06468).as_integer_ratio()) and internal['log10_temperature_ratio'] == list((-.148).as_integer_ratio()) and
         internal['initial_error_exact_zero'] is True, 'thin internal source/initial-state binding differs')
    for key,value in (('dimension',896),('layers',24),('vocabulary',151936),('precision',2048),('initial_token_id',198)):
        scalar.integer(internal[key],value,'thin internal '+key)
    # Reuse the authenticated audit002 BF16 row conversion; never decode it again.
    embedding = report['initial_state']['embedding']; scalar.integer(embedding['row_byte_offset'],embedding['row_byte_offset'],'row byte offset')
    need(embedding['row_byte_offset'] >= 8 and embedding['row_byte_offset'] + 1792 <= embedding['asset']['bytes'], 'audited row offset outside checkpoint')
    io.e.digest(embedding['row_raw_sha256'])
    completion = io.read(tu['files'][thin['output']+'/report.json'],True)
    need(completion['status'] == 'thin_point_defects_completed' and completion['steps'] == 20 and completion['precision'] == 2048 and
         completion['local_image_and_defect_radii_preserved'] is True and completion['midpoint_selection_is_explicit'] is True and
         all(completion[k] is False for k in ('trajectory_error_propagated','whole_tube_derivatives_evaluated','reachability_test_performed')), 'original thin completion differs')
    return dict(audit=au, thin=tu, report=report, initial=initial, step=point,
                upstream_pins=dict(initial_file=first,step1_file=step,audit_result=REPORT_PIN), embedding=embedding)


def join_prefix(io, card, records, assets):
    authority = card['prefix_execution']; prefix = io.e.PREFIX
    need(authority['source_commit'] == prefix['prefix_source_commit'] and authority['lane'] == 'prefix' and authority['output'] == prefix['prefix_output'] and
         authority['receipt']['path'] == prefix['prefix_receipt'] and authority['receipt']['sha256'] == prefix['prefix_receipt_sha256'] and
         authority['admission']['path'] == prefix['prefix_admission_path'] and authority['admission']['sha256'] == prefix['prefix_admission_sha256'] and
         authority['card']['path'] == str(D/'prefix-card.json') and authority['card']['sha256'] == '6c2ad90ef0332733f8f249833ec9d2adcd0799b1ec6c4dc415202df0b78e851c' and
         authority['plan']['path'] == str(D/'prefix-plan.json') and authority['argv'] == [PYTHON,'-I','-B',str(D/'prefix_setup.py'),'--card',str(D/'prefix-card.json'),'--out',prefix['prefix_output']], 'exact original prefix execution required')
    upstream = original_execution(io, authority)
    _required_sources(upstream, {'codex/coexistence-task33/prefix_setup.py':'72a95c042b4d5e229264b07886733b23c385f37b0dd95009dfa067935877fbc3',
                                 'codex/certify-llm/ball_qwen.py':prefix['prefix_map_source_sha256']})
    need(upstream['card']['alpha'] == .06468 and upstream['card']['log10T'] == -.148 and
         type(upstream['card']['precision']) is int and upstream['card']['precision'] == 2048, 'prefix cell/precision differs')
    need(set(upstream['files']) == {str(B/'prefix-001'/name) for name in ('inputs.json','layers.jsonl','prefix-cache.json.gz','report.json')} and
         upstream['files'].get(CACHE_PIN['path']) == CACHE_PIN and upstream['files'].get(PREFIX_INPUT_PIN['path']) == PREFIX_INPUT_PIN, 'exact original prefix files required')
    meta = io.read(PREFIX_INPUT_PIN,True); report = io.read(upstream['files'][str(B/'prefix-001/report.json')],True)
    need(report['completed'] is True and type(report['precision']) is int and report['precision'] == 2048 and report['layers'] == list(range(24)), 'original full prefix incomplete')
    value = records['prefix-binding.json']; map_value = card['map']; backend = card['backend']
    expected_map = dict(schema='codex-qwen-exact-soft-map-v1', dimension=896, vocabulary=151936, layers=24,
        pinned_assets={name:record['asset']['sha256'] for name,record in assets['assets'].items()}, prefix_ids=PREFIX_IDS, steering_layer=12,
        alpha_ratio=list((.06468).as_integer_ratio()), log10_temperature_ratio=list((-.148).as_integer_ratio()),
        temperature='exp(log(10)*exact_dyadic_log10T)', rotary='real base^(-2j/head_dim), real sin/cos',
        state='one appended full-dimensional soft embedding after the fixed prefix',
        readout='full-vocabulary expected embedding with the checkpoint embedding matrix',
        checkpoint_constants='stored binary constants interpreted as exact dyadics')
    need(map_value == expected_map and value['map'] == map_value and value['map_sha256'] == card['map_sha256'] == io.e.canonical_hash(map_value), 'exact target real map differs')
    need(value['backend'] == backend and value['backend_sha256'] == card['backend_sha256'] == io.e.canonical_hash(backend) and
         type(backend['precision']) is int and backend['precision'] == 2048 and type(backend['threads']) is int and backend['threads'] == 1, 'original arithmetic backend metadata differs')
    need(meta['source_commit'] == authority['source_commit'] and meta['pinned_assets'] == expected_map['pinned_assets'] and
         meta['prefix_ids'] == PREFIX_IDS and meta['precision'] == 2048 and meta['steering_layer'] == 12 and
         all(meta[k] == expected_map[k] for k in ('alpha_ratio','log10_temperature_ratio','temperature','rotary')) and
         meta['versions'] == {'python_flint':backend['python_flint'],'flint':backend['flint']}, 'original prefix metadata/precision/backend differs')
    expected_fields = dict(prefix=prefix, prefix_files={'inputs.json':PREFIX_INPUT_PIN['sha256'],'prefix-cache.json.gz':CACHE_PIN['sha256']},
        original_prefix_execution=authority, checkpoint=assets['assets']['model.safetensors'], config=assets['assets']['config.json'],
        stored_precision=2048, receiver_precision=2048, original_radii_retained=True, map=map_value, map_sha256=card['map_sha256'],
        backend=backend, backend_sha256=card['backend_sha256'], regularity_contract=io.e.REGULARITY_CONTRACT,
        regularity_review=card['regularity_review'], **io.e.SAVED_FLAGS)
    need(value == expected_fields and type(value['stored_precision']) is int and type(value['receiver_precision']) is int and value['original_radii_retained'] is True,
         'complete original prefix/retained-radius binding differs')
    inputs = io.e.pins(card['input_files'])
    for pin in [CACHE_PIN,PREFIX_INPUT_PIN,authority['receipt'],authority['admission'], *[r['asset'] for r in assets['assets'].values()]]:
        need(inputs.get(pin['path']) == pin, 'prefix/asset missing from exact target input pins')
    return dict(execution=authority, cache=io.cache_observation, metadata_pin=PREFIX_INPUT_PIN,
                radii_retention_basis='Exact original cache bytes, producer/importer source and authenticated original metadata; no cache decoding or radius reset.')


def host_join(closure, host, expected, receipt):
    need(closure['source_commit'] == expected['source_commit'] and closure['receipt'] == expected['receipt'] and
         closure['admission'] == expected['admission'] and closure['worker_startup'] == expected['worker_startup'] and
         closure['original_card'] == expected['card'] and closure['original_plan'] == expected['plan'] and
         closure['output_inventory'] == receipt['output_files'] and closure['output_bytes'] == receipt['actual_output_bytes'] and
         type(closure['session_reaped_exit_code']) is int and closure['session_reaped_exit_code'] == receipt['returncode'], 'Root original operational closure join differs')
    need(CLOSURE_PIN['path'] + ' SHA256 ' + CLOSURE_PIN['sha256'] in host['observation_reference'], 'normalized host record lacks pinned Root observation reference')
    actual = closure['actual_host_closure']
    need(datetime.fromisoformat(actual['checked_at']) == datetime.fromisoformat(host['observed_at_utc'].replace('Z','+00:00')) and
         actual['matching_groups_or_sessions'] == [], 'Root contemporaneous actual-host closure differs')
    ids = {row['pid']:row for row in actual['identities']}
    need(len(ids) == len(actual['identities']) == 2, 'two original host identities required')
    for key in ('worker_identity','supervisor_identity'):
        ident = host[key]; row = ids.get(ident['pid'],{})
        need(row.get('expected_starttime') == ident['starttime'] and row.get('pid_absent') is True and row.get('original_absent') is True and
             ident['pgid'] in actual['group_ids_checked'] and ident['sid'] in actual['group_ids_checked'], 'Root host identity/group observation differs')


def integrate(io, modules, audit_card):
    files, execution, assets_helper, scalar = modules
    expected = io.read(audit_card['original_expected'],True); closure = io.read(audit_card['original_closure'],True)
    need(expected['source_commit'] == ORIGINAL_SOURCE, 'exact original target source required')
    card, plan, receipt, admission, startup, host = [io.read(expected[k],True) for k in ('card','plan','receipt','admission','worker_startup','host_closure')]
    io.sources(ORIGINAL_SOURCE, [expected['card'],expected['plan']]); source_join(io, ORIGINAL_SOURCE, plan, admission, expected['card'])
    need(receipt['source_commit'] == admission['source_commit'] == ORIGINAL_SOURCE and receipt['lane'] == admission['lane'] == 'tube' and
         receipt['argv'] == admission['argv'] == [PYTHON,'-I','-B',str(D/'tube_target_driver_002.py'),'--card',expected['card']['path'],'--card-sha256',expected['card']['sha256'],'--out',str(TARGET)] and
         type(receipt['returncode']) is int and (receipt['failure'] is None or type(receipt['failure']) is str) and
         all(receipt.get(k) is True for k in ('worker_reaped','worker_identity_absent','worker_group_session_absent')), 'original target source/command/closure metadata differs')
    execution._host_closure(host,expected,execution.identity(startup['identity']),execution.identity(admission['supervisor']))
    host_join(closure,host,expected,receipt)
    if receipt['returncode'] != 0 or receipt['failure'] is not None:
        inventory = io.inventory(TARGET, receipt['output_files'])
        raise ClosedOriginalFailure(dict(receipt=expected['receipt'],returncode=receipt['returncode'],failure=receipt['failure'],output_files=list(inventory.values())))
    bundle = files.read_flat_output_inventory(str(TARGET), receipt['output_files'])
    io.inventory(TARGET, receipt['output_files']); records = bundle['records']
    joined = execution.check_target_execution(expected,card,plan,receipt,admission,startup,records['admission.json'],records['status.json'],host)
    preflight = execution.pins(plan['lanes']['tube']['preflight_pins'])
    need(execution.pins(closure['preflight_pins_verified']) == preflight and closure['source_blobs_verified'] == admission['source_blobs'], 'Root complete source/preflight inventories differ')
    io.verify_pins(list(preflight.values()))
    upstream = join_upstream(io,card,scalar)
    assets = io.assets()
    embedding = upstream['embedding']; need(embedding['asset'] == assets['assets']['model.safetensors']['asset'] and
        embedding['snapshot_authority'] == assets['assets']['model.safetensors']['snapshot_authority'], 'audited BF16 row uses different checkpoint/aliases')
    prefix = join_prefix(io,card,records,assets)
    # All inherited authority rows are joined before any new scalar evaluation.
    result = scalar.check_bundle(records,bundle['pins'],receipt,joined['reconstructed_binding'],
                                 upstream['initial'],upstream['step'],upstream['report'],upstream['upstream_pins'])
    io.recheck(); need(assets_helper.check_target_assets() == assets, 'complete current assets/aliases changed across audit')
    need(files.read_flat_output_inventory(str(TARGET),receipt['output_files'])['pins'] == bundle['pins'], 'final original output set differs')
    return dict(schema='codex-task33-integrated-saved-audit-v1', status='saved_consistency_completed_pending_external_release',
        original_expected=expected, original_closure=audit_card['original_closure'], target_execution=joined,
        upstream_audit=card['audit_execution'], upstream_thin=card['thin_execution'], prefix=prefix, assets=assets,
        authenticated_file_pins=list(io.pins.values()), original_git_checks=io.git_checks,
        historical_input_provenance=io.historical_input_provenance,
        scalar_check={k:v for k,v in result.items() if k != 'scope'}, L_candidate=result['L_candidate'], **FLAGS,
        scope='Saved-byte/source/record consistency and exact first-step/operator/coverage arithmetic only. No native arithmetic trace attestation, model replay, basin result, trajectory or scientific admission. Root must separately authenticate this audit execution/closure and release any conditional bound.')


def audit_admission(io, card_pin, card):
    need(card['schema'] == 'codex-task33-tube-saved-audit-card-v1' and card['status'] == 'reviewed_for_saved_audit' and
         card['repository'] == str(ROOT) and card['output'] == str(OUT) and card['original_expected'] == EXPECTED_PIN and
         card['original_closure'] == CLOSURE_PIN and card['helper_sha256'] == HELPERS and
         card['supervision_plan'] == str(PLAN) and card['supervision_admission'] == str(SUP/'admission.json'), 'reviewed exact saved-audit card required')
    need(all(card.get(k) is False for k in FLAGS) and io.e.limits(card['limits']) == LIMITS and card['cpu_ids'] == [0] and
         type(card['cpu_ids'][0]) is int, 'saved-audit scope/resource card differs')
    admission_pin, admission = io.current(SUP/'admission.json'); plan_pin = dict(admission['plan'])
    if not Path(plan_pin['path']).is_absolute(): plan_pin['path'] = str(ROOT/plan_pin['path'])
    need(plan_pin['path'] == str(PLAN), 'exact runtime audit plan required'); plan = io.read(plan_pin,True)
    source = io.e.digest(admission['source_commit'],40); lane = plan['lanes']['audit']
    argv = [PYTHON,'-I','-B',str(SELF),'--card',str(CARD),'--card-sha256',card_pin['sha256'],'--out',str(OUT)]
    need(plan['status'] == 'reviewed_for_execution' and plan['repository'] == str(ROOT) and plan['raw_root'] == str(B) and
         plan['python'] == PYTHON and type(plan['cpu_threads']) is int and plan['cpu_threads'] == 1 and io.e.cpus(plan['cpu_ids']) == [0] and
         lane['entry'] == str(SELF) and [PYTHON,'-I','-B',lane['entry'],*lane['arguments']] == argv == admission['argv'] and
         lane['output'] == str(OUT) and lane['supervision'] == str(SUP) and admission['lane'] == 'audit' and
         io.e.limits(lane['limits']) == io.e.limits(admission['limits']) == LIMITS, 'exact current saved-audit allocation differs')
    need(all(os.environ.get(k) == admission['threads'].get(k) == '1' for k in io.e.THREADS) and os.environ.get('HF_HUB_OFFLINE') == '1', 'offline single-thread runtime required')
    need(type(plan['startup_ram_gib']) is int and plan['startup_ram_gib'] == 24 and
         type(admission['capacity']['available_ram_bytes']) is int and admission['capacity']['available_ram_bytes'] >= 24*2**30 and
         type(admission['capacity']['disk_free_bytes']) is int and admission['capacity']['disk_free_bytes'] >= 16*2**30, 'recorded saved-audit startup capacity below floors')
    need(os.getpriority(os.PRIO_PROCESS,0) == 10 and sorted(os.sched_getaffinity(0)) == [0] and
         os.getresuid() == (1000,1000,1000), 'actual nice10 CPU0 UID1000 required')
    for key, kind in (('as_bytes',resource.RLIMIT_AS),('cpu_seconds',resource.RLIMIT_CPU),('output_bytes',resource.RLIMIT_FSIZE)):
        need(resource.getrlimit(kind) == (LIMITS[key],)*2, 'actual bounded audit rlimit differs')
    planned = source_join(io,source,plan,admission,card_pin)
    need(planned.get(str(SELF),{}).get('sha256') == card['auditor_sha256'], 'auditor missing from original runtime source inventory')
    need(planned.get(str(ROOT/'codex/certify-chaos/supervise.py'),{}).get('sha256') == io.e.SOURCE_HASHES['codex/certify-chaos/supervise.py'], 'reviewed supervisor/runtime16GiB-floor source missing')
    for name,digest in HELPERS.items(): need(planned.get(str(D/name),{}).get('sha256') == digest, 'frozen helper absent from audit source inventory')
    for pin in (EXPECTED_PIN,CLOSURE_PIN,io.read(EXPECTED_PIN,True)['host_closure']):
        need(planned.get(pin['path']) == pin, 'reviewed original authority record absent from audit source inventory')
    for pin in planned.values(): io.read(pin)
    io.sources(source,[plan_pin]); preflight = io.e.pins(lane['preflight_pins'])
    for pin in (EXPECTED_PIN,CLOSURE_PIN): need(preflight.get(pin['path']) == pin, 'audit authority missing from preflight')
    io.verify_pins(list(preflight.values()))
    return dict(source_commit=source,card=card_pin,plan=plan_pin,admission=admission_pin,source_files=list(planned.values()))


def pure_arguments(args):
    need(args.card == str(CARD) and args.out == str(OUT) and type(args.card_sha256) is str and
         re.fullmatch('[0-9a-f]{64}',args.card_sha256), 'exact fresh audit001 arguments required')


def _allocate():
    flags = os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW; parent = os.open('/',flags)
    try:
        for part in OUT.parent.parts[1:]:
            child = os.open(part,flags,dir_fd=parent); os.close(parent); parent = child
        os.mkdir(OUT.name,mode=0o700,dir_fd=parent)
        return os.open(OUT.name,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=parent)
    finally: os.close(parent)


def _save(directory,name,value):
    need(name in ('result.json','failure.json'), 'fixed audit output basename required')
    _check_output_identity(directory)
    payload = (json.dumps(value,indent=2,allow_nan=False)+'\n').encode()
    need(len(payload) <= LIMITS['output_bytes'], 'audit report exceeds output cap')
    fd = os.open(name,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600,dir_fd=directory)
    with os.fdopen(fd,'wb') as stream: stream.write(payload); stream.flush(); os.fsync(stream.fileno())
    os.fsync(directory)
    _check_output_identity(directory)


def _check_output_identity(directory):
    flags = os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW; parent = os.open('/',flags)
    try:
        for part in OUT.parent.parts[1:]:
            child = os.open(part,flags,dir_fd=parent); os.close(parent); parent = child
        current = os.stat(OUT.name,dir_fd=parent,follow_symlinks=False); opened = os.fstat(directory)
        need(stat.S_ISDIR(current.st_mode) and (current.st_dev,current.st_ino) == (opened.st_dev,opened.st_ino), 'exclusive audit output path moved or replaced')
    finally: os.close(parent)


def execute(args):
    pure_arguments(args); directory = _allocate(); io = None
    try:
        modules = load_helpers(); io = Evidence(*modules[:3])
        card_pin, card = io.current(CARD)
        need(card_pin['sha256'] == args.card_sha256, 'external audit card hash differs')
        admission = audit_admission(io,card_pin,card)
        result = integrate(io,modules,card); result['audit_execution_binding'] = admission
        _save(directory,'result.json',result)
        return 0
    except BaseException as error:
        failure = dict(schema='codex-task33-integrated-saved-audit-failure-v1',
            status='closed_original_failure_no_L' if isinstance(error,ClosedOriginalFailure) else 'saved_audit_refused',
            error_type=type(error).__name__, message=str(error), traceback=traceback.format_exc(),
            authenticated_file_pins=[] if io is None else list(io.pins.values()),
            historical_input_provenance=[] if io is None else io.historical_input_provenance, **FLAGS)
        if isinstance(error,ClosedOriginalFailure): failure['original_failure'] = error.evidence
        _save(directory,'failure.json',failure)
        return 1
    finally: os.close(directory)


def main():
    parser = argparse.ArgumentParser()
    for name in ('card','card-sha256','out'): parser.add_argument('--'+name,required=True)
    raise SystemExit(execute(parser.parse_args()))


if __name__ == '__main__': main()
