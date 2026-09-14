"""Synthetic EXP008 learning/analysis checks: no neural targets or integrators."""
import copy
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import hopfield_information_model as m


def development():
    rng = np.random.default_rng(314)
    groups = [f"dev-{a:02d}" for a in range(12) for _ in range(8)]
    ids = [f"dev-{a:02d}-row-{c}" for a in range(12) for c in range(8)]
    base = np.column_stack((np.tile(np.linspace(-1, 1, 8), 12), np.repeat(np.arange(12), 8),
                            np.full(96, 3.)))
    geometry = rng.uniform(-1, 1, size=(96, 2))
    shuffled = np.roll(geometry.reshape(12, 8, 2), -5, axis=0).reshape(96, 2)
    y = .7 + .15 * geometry[:, 0] - .1 * geometry[:, 1] + .03 * base[:, 0]
    return dict(base_features=base, geometry_features=geometry, target_rmse=y,
                group_ids=groups, shuffled_geometry_features=shuffled,
                base_feature_names=["action", "coordinate", "constant"],
                geometry_feature_names=["margin", "boundary_alignment"],
                row_ids=ids, expected_row_ids=ids.copy(),
                expected_group_ids=[f"dev-{a:02d}" for a in range(12)],
                shuffle_metadata={"method": "cyclic donor shift", "shift": 5})


def test_development_fit_recovers_synthetic_information_and_serializes():
    data = development()
    packet = m.fit_information_models(**data)
    assert packet["status"] == "complete"
    assert packet["packet_sha256"] == m.packet_digest(packet)
    assert json.loads(json.dumps(packet, allow_nan=False)) == packet
    assert packet["development_rows"] == 96
    assert set(packet["models"]) == set(m.LEARNERS)
    geom = packet["models"]["geometry"]
    for policy in ("baseline", "shuffled_geometry"):
        assert min(geom["cv"]["equal_group_mse"]) < min(packet["models"][policy]["cv"]["equal_group_mse"]) / 100
    for model in packet["models"].values():
        assert model["cv"]["lambda_grid"] == list(m.LAMBDA_GRID)
        assert model["chosen_lambda_index"] == int(np.argmin(model["cv"]["equal_group_mse"]))
        assert model["cost"] == {"cv_folds": 12, "ridge_solves": 65}


def test_every_validation_scaler_excludes_held_out_group_and_prediction_is_fixed():
    data = development()
    data["base_features"][-8:, 1] = 1e6
    packet = m.fit_information_models(**data)
    assert packet["status"] == "complete"
    fold = next(f for f in packet["models"]["baseline"]["cv"]["folds"] if f["held_out_group"] == "dev-11")
    np.testing.assert_allclose(fold["feature_mean"], np.mean(data["base_features"][:-8], axis=0), rtol=1e-14, atol=1e-14)
    assert "dev-11" not in fold["train_groups"]
    assert fold["train_rows"] == 88 and fold["validation_rows"] == 8
    original = copy.deepcopy(packet)
    new_base = np.full((18, 3), 1e5)
    result = m.predict_information_models(packet, new_base, np.zeros((18, 2)),
                                          shuffled_geometry_features=np.zeros((18, 2)))
    assert result["status"] == "complete"
    assert packet == original  # No test-driven centering, scaling or update.
    new_base[:] = 0
    assert result["scores"]["baseline"] != [packet["models"]["baseline"]["intercept"]] * 18


def test_ridge_matches_balanced_analytic_solution_and_does_not_clip_predictions():
    groups = [f"a{i}" for i in range(12) for _ in range(2)]
    x = np.tile([-1., 1.], 12)
    data = dict(base_features=x[:, None], geometry_features=np.ones((24, 1)),
                target_rmse=.3 + .2*x, group_ids=groups,
                shuffled_geometry_features=np.ones((24, 1)), base_feature_names=["x"],
                geometry_feature_names=["constant"], row_ids=[f"row{i}" for i in range(24)],
                expected_row_ids=[f"row{i}" for i in range(24)], expected_group_ids=sorted(set(groups)))
    packet = m.fit_information_models(**data)
    model = packet["models"]["baseline"]
    assert model["intercept"] == pytest.approx(.3, abs=2e-15)
    assert model["coefficients"] == pytest.approx([.2/(1+model["chosen_lambda"])], abs=2e-15)
    assert model["chosen_lambda"] == m.LAMBDA_GRID[0]
    result = m.predict_information_models(packet, [[-100.], [100.]], [[1.], [1.]],
                                          shuffled_geometry_features=[[1.], [1.]])
    assert result["status"] == "complete"
    assert result["scores"]["baseline"][0] < 0 and result["scores"]["baseline"][1] > 2


def test_anchor_weights_do_not_change_when_one_groups_rows_are_duplicated():
    data = development()
    original = m.fit_information_models(**data)
    for key in ("base_features", "geometry_features", "shuffled_geometry_features", "target_rmse"):
        data[key] = np.concatenate((data[key], data[key][:8]), axis=0)
    data["group_ids"] += data["group_ids"][:8]
    data["row_ids"] += [f"duplicate-row-{i}" for i in range(8)]
    data["expected_row_ids"] = data["row_ids"].copy()
    duplicated = m.fit_information_models(**data)
    assert duplicated["status"] == "complete"
    for policy in m.LEARNERS:
        for key in ("feature_mean", "feature_scale", "coefficients", "intercept"):
            np.testing.assert_allclose(original["models"][policy][key], duplicated["models"][policy][key], rtol=1e-9, atol=1e-12)
        np.testing.assert_allclose(original["models"][policy]["cv"]["equal_group_mse"],
                                   duplicated["models"][policy]["cv"]["equal_group_mse"], rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize("bad", [None, np.nan, np.inf, -.01, 2.01, True])
def test_invalid_development_label_invalidates_whole_packet(bad):
    data = development()
    data["target_rmse"] = data["target_rmse"].tolist()
    data["target_rmse"][0] = bad
    packet = m.fit_information_models(**data)
    assert packet["status"] == "training_failed" and packet["models"] == {}
    result = m.predict_information_models(packet, [[0, 0, 0]], [[0, 0]], shuffled_geometry_features=[[0, 0]])
    assert result["status"] == "prediction_failed"


def test_missing_development_row_and_nonfinite_feature_are_never_filtered():
    data = development()
    for key in ("base_features", "geometry_features", "shuffled_geometry_features", "target_rmse", "group_ids", "row_ids"):
        data[key] = data[key][1:]
    packet = m.fit_information_models(**data)
    assert packet["status"] == "training_failed" and "census" in packet["error"]
    data = development()
    data["shuffled_geometry_features"][0, 0] = np.nan
    assert m.fit_information_models(**data)["status"] == "training_failed"
    data = development()
    data["expected_group_ids"] = data["expected_group_ids"][:-1]
    assert m.fit_information_models(**data)["status"] == "training_failed"


def test_missing_census_and_duplicate_names_fail_before_training():
    data = development()
    data.pop("expected_row_ids")
    assert m.fit_information_models(**data)["status"] == "training_failed"
    data = development()
    data["geometry_feature_names"][0] = "action"
    assert m.fit_information_models(**data)["status"] == "training_failed"


def test_corrupted_packet_and_wrong_feature_order_cannot_predict():
    packet = m.fit_information_models(**development())
    altered = copy.deepcopy(packet)
    altered["models"]["baseline"]["coefficients"][0] += .1
    assert m.predict_information_models(altered, [[0, 0, 0]], [[0, 0]],
        shuffled_geometry_features=[[0, 0]])["status"] == "prediction_failed"
    assert m.predict_information_models(packet, [[0, 0, 0]], [[0, 0]],
        shuffled_geometry_features=[[0, 0]], base_feature_names=["coordinate", "action", "constant"])["status"] == "prediction_failed"


def test_solver_failure_retains_invalid_packet_without_lambda_fallback(monkeypatch):
    calls = []
    def fail(*args):
        calls.append(1)
        raise np.linalg.LinAlgError("injected ridge failure")
    monkeypatch.setattr(m.np.linalg, "solve", fail)
    packet = m.fit_information_models(**development())
    assert packet["status"] == "training_failed"
    assert "injected ridge failure" in packet["error"]
    assert len(calls) == 1  # No easier penalty or surviving-row replacement.
    assert packet["development_rows"] == 96 and "development_data_sha256" in packet


def test_exact_ties_lowest_id_near_ties_not_rounded_and_failure_not_replaced():
    predictions = dict(status="complete", scores={p: [0.1]*18 for p in m.LEARNERS})
    decision = m.select_candidate_actions(predictions, [.1]*18, candidate_ids=list(reversed(m.CANDIDATES)))
    assert all(row["candidate"] == 0 for row in decision["policies"].values())
    predictions["scores"]["geometry"][17] -= 1e-15
    decision = m.select_candidate_actions(predictions, [.1]*18)
    assert decision["policies"]["geometry"]["candidate"] == 17
    assert len(decision["predictions"]["geometry"]) == len(decision["preview_scores"]) == 18
    failed = dict(status="prediction_failed", scores={p: None for p in m.LEARNERS})
    decision = m.select_candidate_actions(failed, [.1]*18)
    assert all(decision["policies"][p]["status"] == "selector_failed" for p in m.LEARNERS)
    assert decision["policies"]["min_preview"]["status"] == "complete"
    with pytest.raises(ValueError):
        m.select_candidate_actions(predictions, [.1]*18, candidate_ids=list(range(17)))


@pytest.mark.parametrize("bad", [None, np.nan, np.inf, -.01, 2.01, True])
def test_failed_preview_never_shrinks_minimum_search(bad):
    previews = [.1]*18
    previews[5] = bad
    predictions = dict(status="complete", scores={p: [0.1]*18 for p in m.LEARNERS})
    result = m.select_candidate_actions(predictions, previews)
    assert result["policies"]["min_preview"]["status"] == "selector_failed"
    assert result["policies"]["min_preview"]["candidate"] is None


def outcome_fixture():
    blocks = [f"test-{i:02d}" for i in range(12)]
    records, decisions = [], []
    for b, block in enumerate(blocks):
        for pi, phase in enumerate(m.PHASES):
            for ri, radius in enumerate(m.RADII):
                choices = dict(baseline=7, geometry=3, shuffled_geometry=10, min_preview=12)
                decisions.append(dict(block=block, phase=phase, radius=radius,
                    policies={p: dict(status="complete", candidate=c) for p, c in choices.items()}))
                for dt in m.DTS:
                    for candidate in m.CANDIDATES:
                        records.append(dict(block=block, phase=phase, radius=radius, dt=dt,
                            candidate=candidate, status="complete",
                            rmse=b/128 + ri/512 + pi/1024 + candidate/64))
    return blocks, records, decisions


def test_exhaustive_census_equal_weights_and_exact_oracle_cancellation():
    blocks, records, decisions = outcome_fixture()
    result = m.summarize_information_outcomes(records, decisions, block_ids=blocks)
    assert result["planned_candidates"] == result["received_candidates"] == 3456
    assert result["planned_decisions"] == result["received_decisions"] == 96
    assert result["primary"] == dict(status="complete", mean_benefit=1/16,
        ci95=[1/16, 1/16], positive_blocks=12, per_block=[1/16]*12)
    for dt in m.DTS:
        pooled = result["steps"][str(dt)]["panels"]["pooled"]
        assert pooled["means"]["baseline"]["regret"] == 7/64
        assert pooled["means"]["geometry"]["regret"] == 3/64
        assert pooled["means"]["baseline"]["regret"] - pooled["means"]["geometry"]["regret"] == 1/16
        assert len(result["steps"][str(dt)]["conditions"]) == 96
    # A common per-condition error shift changes the measured oracle but not
    # either regret or its paired reduction; this is not an independent endpoint.
    shifted = [{**row, "rmse": row["rmse"] + .5} for row in records]
    shifted_result = m.summarize_information_outcomes(shifted, decisions, block_ids=blocks)
    assert shifted_result["primary"] == result["primary"]


def test_missing_unselected_candidate_invalidates_primary_despite_oracle_cancellation():
    blocks, records, decisions = outcome_fixture()
    missing = next(row for row in records if row["dt"] == .005 and row["candidate"] == 17)
    records.remove(missing)
    result = m.summarize_information_outcomes(records, decisions, block_ids=blocks)
    assert result["candidate_status_counts"]["missing"] == 1
    assert result["primary"]["status"] == "undefined_incomplete"
    assert result["steps"]["0.005"]["complete_candidate_conditions"] == 95
    # No complete-case subset interval is substituted. The numerical check
    # retains its own complete declared census.
    assert result["steps"]["0.01"]["panels"]["pooled"]["contrasts"]["baseline_minus_geometry"]["status"] == "complete"


def test_coarse_only_failure_invalidates_primary_but_retains_fine_diagnostics():
    blocks, records, decisions = outcome_fixture()
    missing = next(row for row in records if row["dt"] == .01 and row["candidate"] == 17)
    records.remove(missing)
    result = m.summarize_information_outcomes(records, decisions, block_ids=blocks)
    assert result["primary"]["status"] == "undefined_incomplete"
    assert not result["primary_required_candidate_census_complete"]
    assert result["steps"]["0.005"]["panels"]["pooled"]["contrasts"]["baseline_minus_geometry"]["status"] == "complete"
    assert result["steps"]["0.005"]["complete_candidate_conditions"] == 96


def test_variable_block_effects_use_paired_blocks_and_equal_phase_dose_weights():
    blocks, records, decisions = outcome_fixture()
    for row in records:
        b = blocks.index(row["block"])
        r = m.RADII.index(row["radius"])
        ph = m.PHASES.index(row["phase"])
        row["rmse"] = .1 + row["candidate"]*(b+1)*(r+1)*(ph+1)/1000
    result = m.summarize_information_outcomes(records, decisions, block_ids=blocks)
    # (7-3) * mean(1,2,3,4) * mean(1,2) / 1000 per block index+1.
    expected = np.arange(1, 13) * .015
    draws = np.random.default_rng(20260913).integers(12, size=(10000, 12))
    ci = np.percentile(expected[draws].mean(axis=1), [2.5, 97.5])
    np.testing.assert_allclose(result["primary"]["per_block"], expected, rtol=1e-12, atol=1e-14)
    assert result["primary"]["mean_benefit"] == pytest.approx(expected.mean(), abs=1e-14)
    np.testing.assert_allclose(result["primary"]["ci95"], ci, rtol=1e-12, atol=1e-14)


def test_failed_outcome_or_decision_is_retained_and_malformed_census_rejected():
    blocks, records, decisions = outcome_fixture()
    bad = next(row for row in records if row["dt"] == .005)
    bad.update(status="failed", rmse=None)
    result = m.summarize_information_outcomes(records, decisions, block_ids=blocks)
    assert result["candidate_status_counts"]["failed"] == 1
    assert result["primary"]["status"] == "undefined_incomplete"
    blocks, records, decisions = outcome_fixture()
    decisions[0]["policies"]["geometry"] = dict(status="selector_failed", candidate=None)
    assert m.summarize_information_outcomes(records, decisions, block_ids=blocks)["primary"]["status"] == "undefined_incomplete"
    with pytest.raises(ValueError, match="duplicate"):
        m.summarize_information_outcomes(records+[records[0]], decisions, block_ids=blocks)
    with pytest.raises(ValueError, match="duplicate"):
        m.summarize_information_outcomes(records, decisions+[decisions[0]], block_ids=blocks)
    with pytest.raises(ValueError, match="frozen"):
        m.summarize_information_outcomes(records, decisions, block_ids=blocks, phases=(675, 2025))


def add_prediction_fixture(blocks, records, decisions):
    lookup = {(r['block'], r['phase'], r['radius'], r['candidate']): r['rmse']
              for r in records if r['dt'] == .005}
    for decision in decisions:
        b = blocks.index(decision['block']) + 1
        ph = m.PHASES.index(decision['phase']) + 1
        r = m.RADII.index(decision['radius']) + 1
        actual = np.array([lookup[(decision['block'], decision['phase'], decision['radius'], c)] for c in m.CANDIDATES])
        decision['candidate_ids'] = list(m.CANDIDATES)
        decision['predictions'] = dict(baseline=(actual+b*ph*r*np.arange(1, 19)/8192).tolist(),
                                      geometry=(actual-.5).tolist(), shuffled_geometry=[3.]*18)


def test_all_candidate_prediction_mse_uses_each_weighting_level_and_no_clipping():
    blocks, records, decisions = outcome_fixture()
    original = m.summarize_information_outcomes(records, decisions, block_ids=blocks)['primary']
    add_prediction_fixture(blocks, records, decisions)
    result = m.summarize_information_outcomes(records, decisions, block_ids=blocks)
    assert result['primary'] == original
    expected = (np.mean(np.arange(1, 13)**2) * np.mean(np.arange(1, 3)**2)
                * np.mean(np.arange(1, 5)**2) * np.mean(np.arange(1, 19)**2) / 8192**2)
    for dt in m.DTS:
        panels = result['steps'][str(dt)]['panels']
        assert panels['pooled']['prediction']['baseline']['mse'] == pytest.approx(expected, abs=1e-16)
        for ri, radius in enumerate(m.RADII, 1):
            expected_dose = expected*ri**2/np.mean(np.arange(1, 5)**2)
            assert panels[str(radius)]['prediction']['baseline']['mse'] == pytest.approx(expected_dose, abs=1e-16)
        geom = panels['pooled']['prediction']['geometry']
        assert geom['status'] == geom['range_status'] == 'complete'
        assert geom['mse'] == .25 and geom['maximum'] < 0
        assert geom['below_zero_fraction'] == geom['out_of_range_fraction'] == 1
        assert geom['above_two_fraction'] == 0
        shuffled = panels['pooled']['prediction']['shuffled_geometry']
        assert shuffled['minimum'] == shuffled['maximum'] == 3
        assert shuffled['above_two_fraction'] == shuffled['out_of_range_fraction'] == 1
        assert len(panels['pooled']['prediction_per_block']['baseline']['mse']) == 12


def test_prediction_mse_reuses_fine_forecast_at_coarse_step_and_explicit_candidate_order():
    blocks, records, decisions = outcome_fixture()
    add_prediction_fixture(blocks, records, decisions)
    for row in records:
        if row['dt'] == .01:
            row['rmse'] += .125
    expected = m.summarize_information_outcomes(records, decisions, block_ids=blocks)
    assert expected['steps']['0.005']['panels']['pooled']['prediction']['geometry']['mse'] == .25
    assert expected['steps']['0.01']['panels']['pooled']['prediction']['geometry']['mse'] == .625**2
    for decision in decisions:
        decision['candidate_ids'].reverse()
        for policy in m.LEARNERS:
            decision['predictions'][policy].reverse()
    reordered = m.summarize_information_outcomes(records, decisions, block_ids=blocks)
    assert reordered == expected


def test_prediction_range_counts_strict_out_of_bounds_without_clipping():
    blocks, records, decisions = outcome_fixture()
    add_prediction_fixture(blocks, records, decisions)
    for decision in decisions:
        decision['predictions']['baseline'] = [0., 2., -.125, 2.125] + [1.]*14
    result = m.summarize_information_outcomes(records, decisions, block_ids=blocks)
    diagnostic = result['steps']['0.005']['panels']['pooled']['prediction']['baseline']
    assert diagnostic['minimum'] == -.125 and diagnostic['maximum'] == 2.125
    assert diagnostic['below_zero_fraction'] == pytest.approx(1/18)
    assert diagnostic['above_two_fraction'] == pytest.approx(1/18)
    assert diagnostic['out_of_range_fraction'] == pytest.approx(2/18)


def test_missing_prediction_or_outcome_invalidates_secondary_without_survivor_mean():
    blocks, records, decisions = outcome_fixture()
    add_prediction_fixture(blocks, records, decisions)
    decisions[0]['predictions']['geometry'][17] = None
    result = m.summarize_information_outcomes(records, decisions, block_ids=blocks)
    for dt in m.DTS:
        prediction = result['steps'][str(dt)]['panels']['pooled']['prediction']
        assert prediction['geometry']['status'] == 'undefined_incomplete'
        assert prediction['geometry']['mse'] is None and prediction['geometry']['minimum'] is None
        assert prediction['baseline']['status'] == 'complete'
    # Existing action-primary semantics are not replaced by this secondary.
    assert result['primary']['status'] == 'complete'
    blocks, records, decisions = outcome_fixture()
    add_prediction_fixture(blocks, records, decisions)
    next(r for r in records if r['dt'] == .01 and r['candidate'] == 17).update(status='failed', rmse=None)
    result = m.summarize_information_outcomes(records, decisions, block_ids=blocks)
    for policy in m.LEARNERS:
        coarse = result['steps']['0.01']['panels']['pooled']['prediction'][policy]
        assert coarse['mse'] is None and coarse['status'] == 'undefined_incomplete'
        assert coarse['range_status'] == 'complete'  # Forecast values still exist.
        assert result['steps']['0.005']['panels']['pooled']['prediction'][policy]['status'] == 'complete'


def test_overflowing_prediction_mse_is_undefined_and_range_remains_visible():
    blocks, records, decisions = outcome_fixture()
    add_prediction_fixture(blocks, records, decisions)
    decisions[0]['predictions']['baseline'] = [1e308]*18
    result = m.summarize_information_outcomes(records, decisions, block_ids=blocks)
    first = result['steps']['0.005']['conditions'][0]['prediction']['baseline']
    assert first['status'] == 'undefined_nonfinite_metric' and first['mse'] is None
    pooled = result['steps']['0.005']['panels']['pooled']['prediction']['baseline']
    assert pooled['mse'] is None and pooled['maximum'] == 1e308
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize('bad', [None, 1., np.array(1.), [None]*18, [np.nan]*18])
def test_absent_or_malformed_prediction_array_yields_undefined_secondary(bad):
    blocks, records, decisions = outcome_fixture()
    add_prediction_fixture(blocks, records, decisions)
    decisions[0]['predictions']['geometry'] = bad
    result = m.summarize_information_outcomes(records, decisions, block_ids=blocks)
    assert result['steps']['0.005']['panels']['pooled']['prediction']['geometry']['mse'] is None
