"""Independent analytic cross-checks and fail-closed controls for the C1 pilot."""
from dataclasses import asdict
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from flint import arb, ctx

from neural_shrimp.gdmaps import TwoLayerNet
from neural_shrimp.rigorous_orbits import BallTanhMap, canonical_bytes, check_orbit, dyadic


def fixture_inputs(net=None, period=1):
    net = net or TwoLayerNet(d=1, width=1, bias=False, ridge=0.5)
    network = asdict(net)
    for k in ("in_scale", "out_scale", "loss_scale", "ridge"):
        network[k] = dyadic(network[k])
    return {"schema": "neural-shrimp-c1-input-v1", "network": network,
            "x": [["0"] * net.d], "y": ["0"], "lr_hidden": "1/2", "lr_output": "1/2",
            "period": period, "approximation": ["0"] * net.n_parameters}


def test_map_and_jacobian_agree_with_independent_frozen_implementation():
    rng = np.random.default_rng(127)
    for bias in (True, False):
        net = TwoLayerNet(d=2, width=3, bias=bias, ridge=0.03, in_scale=0.7, out_scale=0.4, loss_scale=0.8)
        state = rng.normal(size=net.n_parameters)
        x, y = rng.normal(size=(5, 2)), rng.normal(size=5)
        inp = fixture_inputs(net)
        inp.update(x=[[dyadic(v) for v in row] for row in x], y=[dyadic(v) for v in y])
        with ctx.workprec(192):
            actual, jac = BallTanhMap(inp).step([arb(float(v)) for v in state])
        expected = state - 0.5 * net.gradient(state[None, :], x, y)[0]
        expected_jac = np.eye(net.n_parameters) - 0.5 * net.hessian(state, x, y)
        np.testing.assert_allclose([float(v.mid()) for v in actual], expected, rtol=2e-13, atol=2e-13)
        np.testing.assert_allclose([[float(jac[i,j].mid()) for j in range(net.n_parameters)]
                                   for i in range(net.n_parameters)], expected_jac, rtol=2e-13, atol=2e-13)


def test_interval_encloses_high_precision_interior_values_and_jacobians():
    inp = fixture_inputs(TwoLayerNet(d=1, width=1, bias=True, ridge=0.02))
    inp.update(x=[["1/2"], ["-3/4"]], y=["1/4", "-1/2"])
    with ctx.workprec(192):
        model = BallTanhMap(inp)
        box = [arb(i / 8, "1/1024") for i in range(model.n)]
        values, jac = model.step(box)
    with ctx.workprec(384):
        model = BallTanhMap(inp)
        for sign in (-1, 1):
            sample = [arb(i / 8) + sign * arb("1/2048") for i in range(model.n)]
            v, j = model.step(sample)
            assert all(values[i].contains(v[i]) for i in range(model.n))
            assert all(jac[i,k].contains(j[i,k]) for i in range(model.n) for k in range(model.n))


def test_exact_linear_positive_and_false_period_controls():
    # With x=0, no bias, and y=0, the network loss is ridge alone:
    # F(theta)=3 theta/4 exactly. Zero is a unique attracting fixed point.
    good = check_orbit(fixture_inputs(), newton_steps=1, max_power=8)
    assert good["passed"] and good["evidence_rung"] == 3
    assert good["attraction_witness_power"] == 1
    repeated = check_orbit(fixture_inputs(period=2), newton_steps=1, max_power=8)
    assert not repeated["passed"]
    assert repeated["exact_statement"]["existence_and_local_uniqueness"]
    assert not repeated["exact_statement"]["primitive_period_excluded_divisors"]


def test_poor_center_and_unstable_fixed_point_fail_closed():
    inp = fixture_inputs()
    inp["approximation"] = ["1", "1"]
    assert not check_orbit(inp, newton_steps=0, max_power=8)["passed"]
    inp = fixture_inputs()
    inp["lr_hidden"] = inp["lr_output"] = "5"
    result = check_orbit(inp, newton_steps=1, max_power=8)
    assert result["exact_statement"]["existence_and_local_uniqueness"]
    assert not result["exact_statement"]["attraction"]
    assert not result["passed"]


def test_invalid_input_rejected():
    inp = fixture_inputs()
    inp["x"] = [[float("nan")]]
    with pytest.raises(ValueError):
        check_orbit(inp)
    inp = fixture_inputs()
    inp["period"] = True
    with pytest.raises(ValueError):
        check_orbit(inp)


def load_tool(name):
    path = Path(__file__).resolve().parents[1] / "tools" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_offline_verifier_recomputes_and_rejects_corrupted_input(tmp_path):
    runner, verifier = load_tool("certify_c1"), load_tool("verify_certificates")
    raw = tmp_path / "attempt"
    assert runner.run(fixture_inputs(), raw, {"precision": 192, "radius_exponent": -50,
                                            "newton_steps": 1, "max_power": 8}) == 0
    receipt = raw / "receipt.json"
    assert verifier.verify(receipt)["passed"]
    inputs = json.loads((raw / "inputs.json").read_bytes())
    inputs["lr_hidden"] = "17"
    (raw / "inputs.json").write_bytes(canonical_bytes(inputs))
    with pytest.raises(ValueError, match="hash/size mismatch"):
        verifier.verify(receipt)


def test_verifier_rejects_incomplete_source_inventory(tmp_path):
    runner, verifier = load_tool("certify_c1"), load_tool("verify_certificates")
    raw = tmp_path / "attempt"
    runner.run(fixture_inputs(), raw, {"precision": 192, "radius_exponent": -50,
                                     "newton_steps": 1, "max_power": 8})
    receipt_path = raw / "receipt.json"
    receipt = json.loads(receipt_path.read_bytes())
    receipt["source_sha256"] = {}
    receipt_path.write_bytes(canonical_bytes(receipt))
    with pytest.raises(ValueError, match="incomplete checker"):
        verifier.verify(receipt_path)


def test_verifier_rejects_false_summary_number(tmp_path):
    runner, verifier = load_tool("certify_c1"), load_tool("verify_certificates")
    raw = tmp_path / "attempt"
    runner.run(fixture_inputs(), raw, {"precision": 192, "radius_exponent": -50,
                                     "newton_steps": 1, "max_power": 8})
    receipt_path = raw / "receipt.json"
    receipt = json.loads(receipt_path.read_bytes())
    receipt["attraction_witness_power"] = 99
    receipt_path.write_bytes(canonical_bytes(receipt))
    with pytest.raises(ValueError, match="receipt summary"):
        verifier.verify(receipt_path)
