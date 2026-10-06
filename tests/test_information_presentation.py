"""Post-run scientific display checks using controlled saved data only."""
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import plot_hopfield_information as presentation


def test_each_neuron_panel_has_only_its_corresponding_reference(tmp_path):
    # Distinct constant columns expose cross-neuron leakage without an ODE,
    # phase optimizer, regression fit, or dependence on collected outcomes.
    times = np.arange(3001) * .1
    times[-1] = 300.
    reference = np.array([.1, -.25, .4])
    expected = np.tile(reference, (len(times), 1))
    anchor, phase, radius, dt = 'synthetic', 675, .016, .005
    records, features = [], []
    for candidate in range(18):
        states = np.arctanh(expected + candidate * .002)
        raw = f'candidate-{candidate}.npz'
        if candidate < 4:
            np.savez(tmp_path / raw, times=times, states=states)
        records.append(dict(anchor=anchor, phase=phase, radius=radius, dt=dt,
            candidate=candidate, status='complete', raw=raw,
            rmse=float(np.sqrt(np.mean((np.tanh(states)[times > 75] - reference) ** 2))),
            tail_metrics=dict(phase_offset_cycles=0.)))
        features.append(dict(anchor=anchor, phase=phase, radius=radius,
                             candidate=candidate, preview_score=.001 * candidate))
    data = dict(plan=dict(display_contract=dict(example=dict(anchor=anchor,
        phase=phase, radius=radius, dt=dt))), runroot=tmp_path,
        baselines={anchor:dict(eligible=True, certificate=dict(period=10.,
            orbit_activations=np.tile(reference, (9, 1)).tolist()))},
        records=records, features={'test':features}, decisions=[dict(anchor=anchor,
            phase=phase, radius=radius, policies={policy:dict(candidate=index)
                for index, policy in enumerate(presentation.POLICIES)})])

    figure, arrays, _ = presentation.waveform_figure(data)
    try:
        np.testing.assert_array_equal(arrays['original_epoch_reference'], expected)
        for neuron, axis in enumerate(figure.axes[:3]):
            lines = [line for line in axis.lines
                     if line.get_label() == 'Reference: original epoch']
            assert len(lines) == 1, 'One reference neuron must appear per neuron panel'
            np.testing.assert_array_equal(lines[0].get_xdata(), times)
            np.testing.assert_array_equal(lines[0].get_ydata(), expected[:, neuron])
    finally:
        presentation.plt.close(figure)
