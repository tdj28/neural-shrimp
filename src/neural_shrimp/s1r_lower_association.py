"""One fixed numerical lower-bootstrap comparison; no trajectories or search.

The continuation and scientific gates are imported unchanged. Native node
records remain native; only copies carry cyclic phase choices between links.
"""
from copy import deepcopy
import numpy as np
from .period2_branch import Options
from .period2_branch_io import identity
from .s1r_bootstrap import compare_common_controls

NOMINAL_LINKS = 75
MAX_LINKS = 80
MAX_PHASES = 644
INTERNAL_SECONDS = 100
TOTAL_BYTES = 256 * 1024**2
LAUNCHER_BYTES = 1024**2
STORE_BYTES = TOTAL_BYTES - LAUNCHER_BYTES
OUTPUT_LOG = 0.6271428571428572
INPUT_SOURCE = '586e74927215ff7ab5ebc81fbc29720c9322f291'
INPUT_MANIFEST = '391092f55c06b7289b852ca58603dd680f7718fa91f747bfc2d749704d5d5974'
TIMING_SOURCE = 'db10bd1e1b25b0e24b1f0f8f97cb84035742c776'
TIMING_MANIFEST = '45c35991b7ec56f7d9888b6dcf28434396ff94f65de7bf90ba9822408c5f3c8b'
ENDPOINTS = {
    'upper': dict(record='00964-node.json', node='node-0047', hidden=1.10,
        json_sha256='cb049836108a7a1e1b1f364d3f90dde158b258b6111f281f07578404b1580a3b',
        npz_sha256='1994ad4eb2b10853ad2447c16b60b1e8e1cf80f2a030e8ae706f20d5dc14b8ab',
        pair='b93240d3f0c837c82ea900ae48bb18e2a4e34c84e97ec9d59b2837a6b9d9da1f',
        params='01dc43202cbb4dfc536aea06c644e8e2b22f056f6ff77bf8e789b5cd553d1a3d',
        rates='a9f8e9b8a9efdf126de96197580647cd950bec2dea1e5615a5d6baa66f72ad21'),
    'lower': dict(record='00107-node.json', node='node-0001', hidden=.95,
        json_sha256='443f4616daa6e5951c883f6ec5c3859f06d6b91ae3aadd9d22cee01d0fe14d82',
        npz_sha256='299be7b1d96138d583a1e4d9e1b4af47d064b4f488876a9a51d04b71cdc6fcfa',
        pair='bddae7ec37d4025ebe8cd7f4a449930c754627e3d782046c43e7b58bc38d4702',
        params='81d210b5d2df5a7c543735730869dcdc093754ccbee4d374f596e8e779e8b040',
        rates='c4914601242a56a22657f720e3330531a936f6ffdd4bc47654ccb1d7e429a2e8')}


def require(value, message):
    if not value:
        raise ValueError(message)


def validate_endpoint(node, role):
    spec = ENDPOINTS[role]
    require(node['id'] == spec['node'] and node['accepted'] is True, 'qualified fixed endpoint')
    for name, shape in (('pair', (2, 72)), ('params', (2,)), ('rates', (2,))):
        a = node[name]
        require(isinstance(a, np.ndarray) and a.dtype == np.float64 and a.shape == shape
                and a.flags.c_contiguous and np.isfinite(a).all(), 'finite native array ' + name)
        require(identity(a)['sha256'] == spec[name], 'fixed endpoint identity ' + name)
    expected = np.array([spec['hidden'], OUTPUT_LOG], dtype=np.float64)
    require(node['params'].tobytes() == expected.tobytes(), 'fixed endpoint control bytes')
    rates = np.array([10.**float(v) for v in node['params']], dtype=np.float64)
    require(node['rates'].tobytes() == rates.tobytes(), 'historical rate arithmetic')


def measured_cost_gate(timing, collection, collection_bytes):
    """Reduce bound historical receipts only; this is not a new model profile."""
    require(timing['status'] == 'completed_local_segment', 'completed timing pilot')
    require(type(timing['phase_corrector_calls']) is int and timing['phase_corrector_calls'] == 42,
            'timing count must be 42')
    elapsed = timing['elapsed_seconds']
    require(type(elapsed) in (int, float) and np.isfinite(elapsed) and elapsed > 0, 'finite positive timing')
    require(collection['status'] == 'completed_common_control_comparison'
            and type(collection['link_attempts']) is int and collection['link_attempts'] == 12,
            'completed 12-link collection')
    require(type(collection_bytes) is int and collection_bytes == 27218975, 'fixed collection raw size')
    seconds = 2 * (elapsed / 42) * MAX_PHASES + 10
    raw = collection_bytes * MAX_LINKS / 12
    ordinary = STORE_BYTES - STORE_BYTES // 4
    require(np.isfinite(seconds) and seconds <= INTERNAL_SECONDS, 'existing timing requires disclosed redesign')
    require(raw <= ordinary, 'existing storage requires disclosed redesign')
    return dict(projected_seconds=seconds, projected_raw_bytes=raw, ordinary_byte_allowance=ordinary,
        timing_elapsed_seconds=elapsed, timing_phase_calls=42, collection_bytes=collection_bytes,
        collection_link_attempts=12, maximum_phase_calls=MAX_PHASES, maximum_link_attempts=MAX_LINKS,
        seconds_formula='2 * (timing_elapsed_seconds / 42) * 644 + 10',
        bytes_formula='27218975 * 80 / 12', passes=True,
        limitation='heuristic bound-input timing/storage projection; new-control conditioning, retries and filesystem costs may differ')


def macro_targets(upper, lower):
    for p in (upper, lower):
        require(isinstance(p, np.ndarray) and p.dtype == np.float64 and p.shape == (2,)
                and np.isfinite(p).all(), 'finite binary64 controls')
    require(upper[1:].tobytes() == lower[1:].tobytes() and upper[0] > lower[0], 'fixed output and backward hidden direction')
    h_start, h_end = float(upper[0]), float(lower[0])
    return [np.array([h_start + (h_end - h_start) * i / NOMINAL_LINKS, upper[1]])
            for i in range(1, NOMINAL_LINKS)] + [lower.copy()]


def snapshot(node):
    if node is None:
        return None
    return dict(id=node['id'], params=node['params'].copy(), rates=node['rates'].copy(),
                pair=node['pair'].copy(), carried_phase=int(node.get('carried_phase', 0)))


class Progress:
    """Keep the last DURABLY emitted complete link and latest corrector record.

    A qualified direct node is not a witnessed bridge endpoint until all of
    its link's checks have passed and the accepted edge has been persisted.
    """
    def __init__(self, store):
        self.store = store
        self.engine = None
        self.last_accepted = None
        self.latest_corrector = None
        self.accepted_edges = []
        self.started_edges = set()
        self.finished_edges = set()

    def emit(self, kind, payload):
        terminal = kind == 'corrector_end' and payload.get('reason') == 'interrupted'
        self.store.emit(kind, payload, final=terminal)
        if kind in ('corrector_start', 'corrector_iteration', 'corrector_end'):
            self.latest_corrector = deepcopy(payload)
        if kind == 'edge_pending':
            self.started_edges.add(payload['id'])
        if kind == 'edge':
            self.finished_edges.add(payload['id'])
        if kind == 'edge' and payload['status'] == 'accepted':
            native = self.engine.nodes[payload['direct']]
            carried = snapshot(native)
            carried['carried_phase'] = int(payload['parent_match']['phase'])
            if carried['carried_phase']:
                carried['pair'] = carried['pair'][::-1].copy()
            self.last_accepted = carried
            self.accepted_edges.append(payload['id'])

    def terminal(self):
        engine = self.engine
        return dict(last_witnessed=snapshot(self.last_accepted), accepted_edges=self.accepted_edges.copy(),
            latest_corrector=deepcopy(self.latest_corrector),
            pending_nodes=[] if engine is None else [n['id'] for n in engine.nodes.values() if n['reason'] == 'pending'],
            pending_edges=sorted(self.started_edges - self.finished_edges),
            unpersisted_edge_attempts=[] if engine is None else [e['id'] for e in engine.edges if e['id'] not in self.started_edges],
            durably_finished_edges=sorted(self.finished_edges),
            scope='latest corrector is a proposal; only last_witnessed belongs to the completed link path')


def compare_along_bridge(engine, upper, lower, result, progress):
    """Exactly one direction, 75 macros; no association after a gap/stop."""
    targets = macro_targets(upper['params'], lower['params'])
    result['macro_targets'] = [p.tolist() for p in targets]
    start = engine.node(upper['params'].copy(), upper['pair'].copy(), 'saved_upper_requalification')
    result['upper_requalification'] = start['id']
    comparator = engine.node(lower['params'].copy(), lower['pair'].copy(), 'saved_lower_requalification')
    result['lower_requalification'] = comparator['id']
    if not start['accepted'] or not comparator['accepted']:
        result['status'] = 'endpoint_requalification_failed'
        return
    for new, original in ((start, upper), (comparator, lower)):
        for key in ('params', 'rates'):
            require(new[key].tobytes() == original[key].tobytes(), 'requalification exact ' + key)
    progress.last_accepted = snapshot(start)
    engine.emit('bridge_anchor', progress.last_accepted)
    parent, previous = start, None
    for i, target in enumerate(targets, 1):
        engine.budget.check()
        segment, parent, previous = engine.hidden_segment(parent, target, previous=previous,
                                                          min_step=.000125, max_halvings=4)
        segment['macro_index'] = i
        engine.emit('bridge_segment', segment)
        result['bridge'].append(segment)
        if segment['status'] != 'connected':
            result['status'] = 'bridge_gap'
            return
    engine.budget.check()
    association = compare_common_controls(parent, comparator, engine.o)
    association['scope'] = ('saved upper-lineage pair versus newly reproduced lower-bootstrap pair at identical controls; '
                            'finite witnessed numerical path, no historical-state ancestry, interval continuation or uniqueness proof')
    engine.emit('association', association)
    result['association'] = association
    result['status'] = 'completed_common_control_comparison'
