"""New synthetic retention-interface controls; not executed during preparation.

Only the new helper is loaded when this fixture is explicitly run. No Torch,
NumPy, producer tools, model, original evidence, filesystem output allocation,
Git or process-control interface is used. These fixtures cannot qualify seed
closure, primitive period, backend precision or execution authority.
"""
import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace
import unittest


SOURCE = Path(__file__).with_name('seed_phase_capture.py')
spec = importlib.util.spec_from_file_location('seed_phase_capture_subject', SOURCE)
subject = importlib.util.module_from_spec(spec)
spec.loader.exec_module(subject)


class FakeFinite:
    def __init__(self, finite):
        self.finite = finite

    def all(self):
        return self

    def item(self):
        return self.finite


class FakeTensor:
    """Tiny stdlib implementation of only the helper's declared duck interface."""
    def __init__(self, values=None, *, shape=(1, 1, 896), dtype='torch.float64',
                 device='cpu', norm_override=None):
        self.values = list(values if values is not None else [1.0] * 896)
        self.shape = shape
        self.dtype = dtype
        self.device = SimpleNamespace(type=device)
        self.norm_override = norm_override
        self.detach_calls = 0
        self.clone_calls = 0

    def isfinite(self):
        return FakeFinite(all(math.isfinite(value) for value in self.values))

    def norm(self):
        if self.norm_override is not None:
            return self.norm_override
        return math.hypot(*self.values)

    def __sub__(self, other):
        return FakeTensor([left - right for left, right in zip(self.values, other.values)])

    def detach(self):
        self.detach_calls += 1
        return self

    def clone(self):
        self.clone_calls += 1
        return FakeTensor(self.values, shape=self.shape, dtype=self.dtype,
                          device=self.device.type, norm_override=self.norm_override)


class FakeCell:
    def __init__(self, transform=None, events=None):
        self.transform = transform or (lambda value, step: FakeTensor([x + 1.0 for x in value.values]))
        self.events = events if events is not None else []
        self.inputs = []
        self.outputs = []
        self.calls = 0

    def f(self, value):
        self.calls += 1
        self.events.append(('map', self.calls))
        self.inputs.append(value)
        result = self.transform(value, self.calls)
        self.outputs.append(result)
        return result


def legacy_return_distances(cell, X, max_period):
    """Original six-line formula over synthetic values only; no producer import."""
    hist = [X]
    for _ in range(max_period):
        hist.append(cell.f(hist[-1]))
    return [float((hist[q] - X).norm() / X.norm()) for q in range(1, max_period + 1)]


class CaptureControls(unittest.TestCase):
    def test_01_exact_five_calls_and_immediate_callback_order(self):
        events, snapshots = [], []
        X = FakeTensor()
        cell = FakeCell(events=events)

        def callback(index, value):
            events.append(('callback', index))
            snapshots.append(value)

        distances = subject.capture_seed_phases(cell, X, callback)
        expected = [('callback', 0)]
        for step in range(1, 6):
            expected.extend([('map', step), ('callback', step)])
        self.assertEqual(events, expected)
        self.assertEqual(cell.calls, 5)
        self.assertEqual(len(distances), 5)
        self.assertEqual(len(snapshots), 6)
        self.assertEqual([item.values[0] for item in snapshots], [1., 2., 3., 4., 5., 6.])
        self.assertTrue(all(item.shape == (1, 1, 896) for item in snapshots))
        self.assertIs(cell.inputs[0], X)
        for step in range(1, 5):
            self.assertIs(cell.inputs[step], cell.outputs[step - 1])
        for state in [X, *cell.outputs]:
            self.assertEqual((state.detach_calls, state.clone_calls), (1, 1))

    def test_02_exact_legacy_distances_on_nonuniform_synthetic_states(self):
        values = [0.125 + (index % 7) / 8.0 for index in range(896)]
        transform = lambda value, step: FakeTensor([x * .5 + step / 8.0 for x in value.values])
        captured_cell, legacy_cell = FakeCell(transform), FakeCell(transform)
        actual = subject.capture_seed_phases(captured_cell, FakeTensor(values), lambda *_: None)
        expected = legacy_return_distances(legacy_cell, FakeTensor(values), 5)
        self.assertEqual(actual, expected)
        self.assertEqual((captured_cell.calls, legacy_cell.calls), (5, 5))

    def test_03_callback_mutation_cannot_change_recurrence_or_distances(self):
        X, cell = FakeTensor(), FakeCell()

        def mutate(index, snapshot):
            snapshot.values[:] = [-1000.0 - index] * 896

        actual = subject.capture_seed_phases(cell, X, mutate)
        expected = legacy_return_distances(FakeCell(), FakeTensor(), 5)
        self.assertEqual(actual, expected)
        self.assertEqual(X.values, [1.0] * 896)
        self.assertEqual([state.values[0] for state in cell.outputs], [2., 3., 4., 5., 6.])

    def test_04_snapshots_are_independent_of_each_other_and_live_states(self):
        saved = []
        X, cell = FakeTensor(), FakeCell()
        subject.capture_seed_phases(cell, X, lambda index, state: saved.append(state))
        live = [X, *cell.outputs]
        self.assertEqual(len({id(item.values) for item in [*saved, *live]}), 12)
        live[1].values[0] = 999.0
        saved[0].values[0] = -999.0
        self.assertEqual(saved[1].values[0], 2.0)
        self.assertEqual(saved[2].values[0], 3.0)
        self.assertEqual(X.values[0], 1.0)

    def test_05_initial_callback_failure_makes_no_map_call(self):
        cell, seen = FakeCell(), []
        failure = RuntimeError('synthetic initial callback failure')

        def callback(index, snapshot):
            seen.append(index)
            raise failure

        with self.assertRaises(RuntimeError) as caught:
            subject.capture_seed_phases(cell, FakeTensor(), callback)
        self.assertIs(caught.exception, failure)
        self.assertEqual((cell.calls, seen), (0, [0]))

    def test_06_later_callback_failure_stops_and_preserves_earlier_evidence(self):
        cell, seen = FakeCell(), []
        failure = RuntimeError('synthetic callback failure at X2')

        def callback(index, snapshot):
            seen.append((index, snapshot.values[0]))
            if index == 2:
                raise failure

        with self.assertRaises(RuntimeError) as caught:
            subject.capture_seed_phases(cell, FakeTensor(), callback)
        self.assertIs(caught.exception, failure)
        self.assertEqual(cell.calls, 2)
        self.assertEqual(seen, [(0, 1.), (1, 2.), (2, 3.)])

    def test_07_invalid_initial_values_or_norm_never_reach_callback_or_map(self):
        invalid = [FakeTensor([0.0] * 896), FakeTensor([math.nan] + [1.] * 895),
                   FakeTensor([math.inf] + [1.] * 895), FakeTensor(norm_override=math.inf)]
        for X in invalid:
            with self.subTest(values=X.values[:1], norm=X.norm_override):
                cell, seen = FakeCell(), []
                with self.assertRaises(ValueError):
                    subject.capture_seed_phases(cell, X, lambda index, value: seen.append(index))
                self.assertEqual((cell.calls, seen), (0, []))

    def test_08_invalid_initial_declared_metadata_never_reaches_map(self):
        invalid = [FakeTensor(shape=(896,)), FakeTensor(shape=(1, 2, 448)),
                   FakeTensor(shape=(True, 1, 896)), FakeTensor(dtype='torch.float32'),
                   FakeTensor(device='cuda')]
        for X in invalid:
            with self.subTest(shape=X.shape, dtype=X.dtype, device=X.device.type):
                cell, seen = FakeCell(), []
                with self.assertRaises(ValueError):
                    subject.capture_seed_phases(cell, X, lambda index, value: seen.append(index))
                self.assertEqual((cell.calls, seen), (0, []))

    def test_09_late_invalid_state_preserves_only_valid_partial_callbacks(self):
        for bad in [FakeTensor([math.nan] + [1.] * 895), FakeTensor(shape=(1, 896)),
                    FakeTensor(dtype='torch.float32'), FakeTensor(device='cuda')]:
            with self.subTest(shape=bad.shape, dtype=bad.dtype, device=bad.device.type):
                seen = []
                cell = FakeCell(lambda value, step: bad if step == 3 else
                                FakeTensor([x + 1 for x in value.values]))
                with self.assertRaises(ValueError):
                    subject.capture_seed_phases(cell, FakeTensor(),
                                                lambda index, value: seen.append((index, value.values[0])))
                self.assertEqual(cell.calls, 3)
                self.assertEqual(seen, [(0, 1.), (1, 2.), (2, 3.)])

    def test_10_invalid_period_or_callback_is_refused_before_map(self):
        for period in (True, 5.0, 0, 4, 6):
            with self.subTest(period=period):
                cell = FakeCell()
                with self.assertRaises(ValueError):
                    subject.capture_seed_phases(cell, FakeTensor(), lambda *_: None, period=period)
                self.assertEqual(cell.calls, 0)
        cell = FakeCell()
        with self.assertRaises(ValueError):
            subject.capture_seed_phases(cell, FakeTensor(), None)
        self.assertEqual(cell.calls, 0)
        with self.assertRaises(ValueError):
            subject.capture_seed_phases(SimpleNamespace(f=None), FakeTensor(), lambda *_: None)

    def test_11_map_exception_propagates_without_later_calls(self):
        seen = []
        failure = RuntimeError('synthetic map failure at X3')

        def transform(value, step):
            if step == 3:
                raise failure
            return FakeTensor([x + 1 for x in value.values])

        cell = FakeCell(transform)
        with self.assertRaises(RuntimeError) as caught:
            subject.capture_seed_phases(cell, FakeTensor(), lambda index, value: seen.append(index))
        self.assertIs(caught.exception, failure)
        self.assertEqual((cell.calls, seen), (3, [0, 1, 2]))

    def test_12_nonfinite_distance_refuses_after_preserving_all_finite_images(self):
        cell = FakeCell(lambda value, step: FakeTensor([1e200] * 896))
        seen = []
        with self.assertRaisesRegex(ValueError, 'finite seed return distances'):
            subject.capture_seed_phases(cell, FakeTensor([1e-200] * 896),
                                        lambda index, value: seen.append(index))
        self.assertEqual((cell.calls, seen), (5, [0, 1, 2, 3, 4, 5]))


if __name__ == '__main__':
    unittest.main(verbosity=2)
