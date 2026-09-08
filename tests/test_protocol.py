import copy
import json
from pathlib import Path
import pytest
from neural_shrimp.cli import validate_manifest, trial_plan, verify, save_json, sha256

MANIFESTS = Path(__file__).resolve().parents[1] / "experiments" / "manifests"


def test_independent_counts_and_control_matching():
    expected = {"EXP-001": 65, "EXP-002": 162, "EXP-003": 96}
    for path in MANIFESTS.glob("*.json"):
        config = json.loads(path.read_text())
        assert validate_manifest(config) == expected[config["experiment_id"]]
        rows = trial_plan(config)
        assert len({r["trial_id"] for r in rows}) == len(rows)
        assert rows == trial_plan(copy.deepcopy(config))
        if config["kind"] == "utility":
            for seed in config["seeds"]:
                for h, o in config["learning_rate_pairs"]:
                    group = [r for r in rows if (r["seed"],r["lr_hidden"],r["lr_output"]) == (seed,h,o)]
                    assert len(group) == 4
                    assert {r["policy"] for r in group} == {"fixed", "scheduled_decay", "stagnation_decay", "recurrence_decay"}


def test_pilot_and_utility_seeds_do_not_overlap():
    configs = {c["experiment_id"]: c for p in MANIFESTS.glob("*.json") if (c := json.loads(p.read_text()))}
    assert set(configs["EXP-002"]["seeds"]).isdisjoint(configs["EXP-003"]["seeds"])


def test_missing_control_and_inadequate_horizon_rejected():
    config = json.loads((MANIFESTS / "EXP-003-utility-pilot.json").read_text())
    config["policies"].remove("scheduled_decay")
    with pytest.raises(ValueError): validate_manifest(config)
    config = json.loads((MANIFESTS / "EXP-002-training-pilot.json").read_text())
    config["tail"] = 3
    with pytest.raises(ValueError): validate_manifest(config)
