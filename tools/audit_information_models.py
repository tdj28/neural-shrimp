#!/usr/bin/env python3
"""Independent EXP008 saved-model audit; imports no experiment implementation.

Only a complete, failure-free receipt enables outcome reads. Regression is
reconstructed by weighted SVD rather than the experiment's normal-equation
solves. No ODE integration or target modification is performed.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time

import numpy as np


SOURCE = "3c5604497ec57b19d62fbb595da7b190102f4b33"
MANIFEST = "fb0e546947e0f36b77ff4978f60769db3541fc64330022bd664da22c20c321cd"
LAMBDAS = np.array([1e-6, 1e-4, 1e-2, 1., 100.])
LEARNERS = ("baseline", "geometry", "shuffled_geometry")
POLICIES = (*LEARNERS, "min_preview")
RADII = (.00025, .001, .004, .016)
PHASES = (675, 1350)
DTS = (.01, .005)


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def close(actual, expected, model=False):
    np.testing.assert_allclose(actual, expected, rtol=1e-7 if model else 1e-12,
                               atol=1e-9 if model else 1e-14)


def plain(value):
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return plain(value.tolist())
    if isinstance(value, np.generic):
        return plain(value.item())
    return value


def standardize(x, y, groups):
    """Average explicit within-group moments, avoiding the original reduction."""
    unique = sorted(set(groups))
    group = np.asarray(groups)
    mean = np.mean([np.mean(x[group == g], axis=0) for g in unique], axis=0)
    variance = np.mean([np.mean((x[group == g]-mean)**2, axis=0) for g in unique], axis=0)
    scale = np.sqrt(variance)
    scale[(np.ptp(x, axis=0) == 0) | (scale == 0)] = 1.
    intercept = float(np.mean([np.mean(y[group == g]) for g in unique]))
    weights = np.array([1/(len(unique)*np.count_nonzero(group == g)) for g in groups])
    z = (x-mean)/scale
    return mean, scale, intercept, weights, z


def svd_fit_grid(x, y, groups):
    mean, scale, intercept, weights, z = standardize(x, y, groups)
    design = np.sqrt(weights)[:, None]*z
    u, singular, vh = np.linalg.svd(design, full_matrices=False)
    projection = u.T @ (np.sqrt(weights)*(y-intercept))
    coefs = np.array([vh.T @ (singular/(singular**2+lam)*projection) for lam in LAMBDAS])
    return dict(mean=mean, scale=scale, intercept=intercept, weights=weights,
                standardized=z, singular=singular, coefficients=coefs)


def audit_model(x, y, groups, model):
    group = np.asarray(groups)
    folds = []
    for held_out in sorted(set(groups)):
        train = group != held_out
        independent = svd_fit_grid(x[train], y[train], group[train].tolist())
        saved = next(f for f in model["cv"]["folds"] if f["held_out_group"] == held_out)
        assert saved["train_groups"] == sorted(set(group[train]))
        assert saved["train_rows"] == train.sum() and saved["validation_rows"] == (~train).sum()
        close(saved["feature_mean"], independent["mean"])
        close(saved["feature_scale"], independent["scale"])
        validation = ((x[~train]-independent["mean"])/independent["scale"])
        predictions = independent["intercept"] + independent["coefficients"] @ validation.T
        losses = np.mean((predictions-y[~train])**2, axis=1)
        close(saved["mse_by_lambda"], losses, model=True)
        singular = independent["singular"]
        smallest = float(singular[-1]**2) if len(singular) == x.shape[1] else 0.
        folds.append(dict(held_out=held_out, independent_mse=losses,
                          max_saved_loss_difference=float(np.max(np.abs(losses-saved["mse_by_lambda"]))),
                          largest_grid_normal_condition_number=(float(singular[0]**2)+LAMBDAS[0])/(smallest+LAMBDAS[0])))
    cv = np.mean([f["independent_mse"] for f in folds], axis=0)
    close(model["cv"]["equal_group_mse"], cv, model=True)
    close(model["cv"]["lambda_grid"], LAMBDAS)
    best = int(np.argmin(cv))
    assert best == model["chosen_lambda_index"]
    assert model["chosen_lambda"] == LAMBDAS[best]
    final = svd_fit_grid(x, y, groups)
    close(model["feature_mean"], final["mean"])
    close(model["feature_scale"], final["scale"])
    close(model["intercept"], final["intercept"])
    beta = final["coefficients"][best]
    saved_beta = np.asarray(model["coefficients"])
    coefficient_close = np.isclose(saved_beta, beta, rtol=1e-7, atol=1e-9)
    z = (x-np.asarray(model["feature_mean"]))/np.asarray(model["feature_scale"])
    w = final["weights"]
    h = z.T @ (w[:, None]*z) + LAMBDAS[best]*np.eye(x.shape[1])
    rhs = z.T @ (w*(y-model["intercept"]))
    residual = np.linalg.norm(h @ saved_beta-rhs)
    backward = residual/(np.linalg.norm(h, 2)*np.linalg.norm(saved_beta)+np.linalg.norm(rhs))
    sigma = final["singular"]
    eigen_min = float(sigma[-1]**2) if len(sigma) == x.shape[1] else 0.
    conditioned = (float(sigma[0]**2)+LAMBDAS[best])/(eigen_min+LAMBDAS[best])
    fitted = model["intercept"] + z @ saved_beta
    close(model["training_equal_group_mse"], np.sum(w*(fitted-y)**2))
    svd_fitted = final["intercept"] + final["standardized"] @ beta
    top = np.argsort(np.abs(saved_beta))[-8:][::-1]
    cv_sorted = sorted(cv)
    return dict(chosen_lambda=float(LAMBDAS[best]), cv_equal_group_mse=cv,
        cv_best_to_runnerup_gap=float(cv_sorted[1]-cv_sorted[0]),
        independent_folds=folds, feature_count=x.shape[1],
        numerical_design_rank=int(np.linalg.matrix_rank(final["standardized"])),
        constant_columns=[model["feature_names"][i] for i in np.flatnonzero(np.ptp(x, axis=0) == 0)],
        regularized_normal_condition_number=conditioned, relative_backward_residual=float(backward),
        max_coefficient_svd_difference=float(np.max(np.abs(saved_beta-beta))),
        coefficient_comparison_passed=bool(coefficient_close.all()),
        coefficient_mismatch_count=int(np.count_nonzero(~coefficient_close)),
        coefficient_mismatches=[dict(feature=model["feature_names"][i], saved=float(saved_beta[i]),
                                     independent_svd=float(beta[i]), difference=float(saved_beta[i]-beta[i]))
                                for i in np.flatnonzero(~coefficient_close)],
        max_training_prediction_svd_difference=float(np.max(np.abs(fitted-svd_fitted))),
        largest_standardized_coefficients=[dict(feature=model["feature_names"][i], value=float(saved_beta[i])) for i in top],
        independent_final=dict(mean=final["mean"], scale=final["scale"], intercept=final["intercept"], coefficient=beta))


def audit(attempt):
    start = time.perf_counter()
    receipt_path = attempt/"receipt.json"
    if not receipt_path.exists():
        raise RuntimeError("Receipt is not complete; no outcome may be read")
    receipt = read(receipt_path)
    if receipt.get("status") != "complete" or (attempt/"failure.json").exists() or (attempt/"timeout.json").exists():
        raise RuntimeError("A complete failure-free attempt is required")
    assert receipt["source_commit"] == SOURCE and receipt["manifest_sha256"] == MANIFEST
    assert sha(attempt/"manifest.json") == MANIFEST
    p = read(attempt/"manifest.json")
    assert p["phases"] == list(PHASES) and p["radii"] == list(RADII) and p["dt"] == list(DTS)
    paths = [r["path"] for r in receipt["artifacts"]]
    assert len(paths) == len(set(paths))
    for artifact in receipt["artifacts"]:
        f = attempt/artifact["path"]
        assert sha(f) == artifact["sha256"] and f.stat().st_size == artifact["bytes"], f
    packet = read(attempt/"model.json")
    content = {k:v for k,v in packet.items() if k != "packet_sha256"}
    assert packet["packet_sha256"] == hashlib.sha256(json.dumps(content, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    seal = read(attempt/"model-seal.json")
    dseal = read(attempt/"test-decision-seal.json")
    begin = read(attempt/"test-target-started.json")
    for key, file in (("model_sha256", "model.json"), ("development_targets_sha256", "development-targets.jsonl"),
                      ("development_features_sha256", "development-features.json")):
        assert seal[key] == sha(attempt/file)
    assert dseal["model_seal_sha256"] == begin["model_seal_sha256"] == sha(attempt/"model-seal.json")
    assert dseal["decisions_sha256"] == sha(attempt/"test-decisions.json")
    assert dseal["features_sha256"] == sha(attempt/"test-features.json")
    assert begin["decision_seal_sha256"] == sha(attempt/"test-decision-seal.json")
    assert receipt["started_at"] <= seal["sealed_at"] <= dseal["sealed_at"] <= begin["started_at"] <= receipt["ended_at"]
    records = {role: rows(attempt/f"{role}-targets.jsonl") for role in ("development", "test")}
    features = {role: read(attempt/f"{role}-features.json") for role in records}
    anchors = {role: [a for a in p["anchors"] if a["role"] == role] for role in records}
    for role in records:
        assert len(anchors[role]) == 12 and len(records[role]) == 3504 and len(features[role]) == 1728
        assert all(r["role"] == role for r in records[role]+features[role])
        expected = [f'{a["id"]}-p{ph}-r{r:g}-c{c:02d}' for a in anchors[role] for ph in PHASES for r in RADII for c in range(18)]
        assert [r["id"] for r in features[role]] == expected
        expected_targets = {f'{fid}-dt{dt:g}' for fid in expected for dt in DTS}
        expected_targets |= {f'{a["id"]}-p{ph}-r0-zero-dt{dt:g}' for a in anchors[role] for ph in PHASES for dt in DTS}
        assert {r["id"] for r in records[role]} == expected_targets
    assert not ({r["anchor"] for r in features["development"]} & {r["anchor"] for r in features["test"]})
    assert all(r["test_decision_seal_sha256"] == begin["decision_seal_sha256"] for r in records["test"])
    report = dict(source_commit=SOURCE, manifest_sha256=MANIFEST, receipt_sha256=sha(receipt_path),
        verified_artifacts=len(paths), model_status=packet["status"],
        target_status={role:dict(Counter(r["status"] for r in rr)) for role,rr in records.items()},
        feature_status={role:dict(Counter(r["status"] for r in rr)) for role,rr in features.items()},
        chronology=dict(model_sealed=seal["sealed_at"], decisions_sealed=dseal["sealed_at"], test_targets_started=begin["started_at"]),
        model_fit_seconds=seal["fit_seconds"], collection_seconds=receipt["wall_seconds"])
    if packet["status"] != "complete":
        assert read(attempt/"summary.json")["primary"]["status"] != "complete"
        report["disposition"] = "Learner failed; no complete learner result is audited"
        return report
    assert all(r["status"] == "complete" for r in records["development"]+features["development"])
    matrices = {}
    for role in records:
        rr = features[role]
        by_context = {(r["anchor"], r["phase"], r["radius"], r["candidate"]):r for r in rr}
        donor = {a["id"]:anchors[role][(i+5)%12]["id"] for i,a in enumerate(anchors[role])}
        base = np.asarray([r["base"] for r in rr], float)
        geometry = np.asarray([r["geometry"] for r in rr], float)
        shuffled = np.asarray([by_context[(donor[r["anchor"]], r["phase"], r["radius"], r["candidate"])]["geometry"] for r in rr], float)
        assert base.shape == (1728,51) and geometry.shape == shuffled.shape == (1728,6)
        assert np.isfinite(base).all() and np.isfinite(geometry).all() and np.isfinite(shuffled).all()
        matrices[role] = dict(baseline=base, geometry=np.column_stack((base, geometry)),
                              shuffled_geometry=np.column_stack((base, shuffled)), raw_geometry=geometry, raw_shuffled=shuffled)
    assert packet["base_feature_names"] == p["base_feature_names"]
    assert packet["geometry_feature_names"] == p["geometry_feature_names"]
    groups = [r["block"] for r in features["development"]]
    dev_ids = [r["id"] for r in features["development"]]
    assert packet["development_groups"] == sorted(set(groups)) and packet["development_row_ids"] == dev_ids
    dev_records = {r["id"]:r for r in records["development"]}
    y = np.asarray([dev_records[f'{fid}-dt0.005']["rmse"] for fid in dev_ids])
    assert y.shape == (1728,) and np.isfinite(y).all() and np.all((y>=0)&(y<=2))
    input_hash = hashlib.sha256()
    for value in (matrices["development"]["baseline"], matrices["development"]["raw_geometry"], matrices["development"]["raw_shuffled"], y):
        input_hash.update(str(value.shape).encode())
        input_hash.update(np.ascontiguousarray(value, dtype="<f8").tobytes())
    input_hash.update(json.dumps([dev_ids,groups], separators=(",", ":")).encode())
    assert input_hash.hexdigest() == packet["development_data_sha256"]
    model_reports = {policy:audit_model(matrices["development"][policy], y, groups, packet["models"][policy]) for policy in LEARNERS}
    decisions = read(attempt/"test-decisions.json")
    assert len(decisions) == 96
    predictions = {}
    svd_predictions = {}
    for policy in LEARNERS:
        model = packet["models"][policy]
        independent = model_reports[policy]["independent_final"]
        x = matrices["test"][policy]
        predictions[policy] = model["intercept"] + ((x-model["feature_mean"])/model["feature_scale"]) @ np.asarray(model["coefficients"])
        svd_predictions[policy] = independent["intercept"] + ((x-independent["mean"])/independent["scale"]) @ independent["coefficient"]
    counts = {policy:Counter() for policy in POLICIES}
    svd_choices_changed = {policy:[] for policy in LEARNERS}
    margins = {policy:[] for policy in LEARNERS}
    decision_index = {}
    for i, decision in enumerate(decisions):
        rr = features["test"][18*i:18*(i+1)]
        context = tuple(rr[0][k] for k in ("block", "phase", "radius"))
        assert tuple(decision[k] for k in ("block", "phase", "radius")) == context
        assert decision["candidate_ids"] == list(range(18))
        assert decision["model_packet_sha256"] == packet["packet_sha256"]
        decision_index[context] = decision
        for policy in POLICIES:
            scores = [r["preview_score"] for r in rr] if policy == "min_preview" else decision["predictions"][policy]
            assert len(scores) == 18 and np.isfinite(scores).all()
            if policy != "min_preview":
                close(scores, predictions[policy][18*i:18*(i+1)])
                alt = int(np.argmin(svd_predictions[policy][18*i:18*(i+1)]))
                ordered = np.sort(scores)
                change = float(np.max(np.abs(np.asarray(scores)-svd_predictions[policy][18*i:18*(i+1)])))
                margins[policy].append(dict(context=context, best_to_second_gap=float(ordered[1]-ordered[0]),
                                            largest_prediction_change=change))
                if alt != int(np.argmin(scores)):
                    svd_choices_changed[policy].append(dict(context=context, saved=int(np.argmin(scores)), svd=alt))
            else:
                close(scores, decision["preview_scores"])
            chosen = min(range(18), key=lambda c:(scores[c],c))
            assert decision["policies"][policy]["status"] == "complete" and decision["policies"][policy]["candidate"] == chosen
            counts[policy][chosen] += 1
    for policy in LEARNERS:
        values = predictions[policy]
        model_reports[policy].update(test_prediction_minimum=float(values.min()), test_prediction_maximum=float(values.max()),
            test_prediction_below_zero_fraction=float(np.mean(values<0)), test_prediction_above_two_fraction=float(np.mean(values>2)),
            test_prediction_below_zero_count=int(np.count_nonzero(values<0)), test_prediction_above_two_count=int(np.count_nonzero(values>2)),
            max_test_prediction_svd_difference=float(np.max(np.abs(values-svd_predictions[policy]))),
            svd_changed_choices=svd_choices_changed[policy], selected_candidate_counts=dict(counts[policy]))
        gap = np.array([v["best_to_second_gap"] for v in margins[policy]])
        uncertain = [v for v in margins[policy] if v["best_to_second_gap"] <= 2*v["largest_prediction_change"]]
        model_reports[policy]["action_margin_diagnostics"] = dict(minimum=float(gap.min()), median=float(np.median(gap)),
            p05=float(np.percentile(gap,5)), maximum=float(gap.max()), exact_ties=int(np.count_nonzero(gap==0)),
            margins_not_exceeding_twice_prediction_difference=uncertain,
            smallest_context=margins[policy][int(np.argmin(gap))])
        model_reports[policy].pop("independent_final")
    report["models"] = model_reports
    report["minimum_preview_selected_candidate_counts"] = dict(counts["min_preview"])
    test_by_key = {(r["block"],r["phase"],r["radius"],r["candidate"],r["dt"]):r for r in records["test"]}
    test_blocks = [a["block_id"] for a in anchors["test"]]
    summary = read(attempt/"summary.json")
    draws = np.random.default_rng(20260913).integers(12,size=(10000,12))
    result_steps = {}
    for dt in DTS:
        per_dose = {}
        for radius in RADII:
            errors = {policy:[] for policy in POLICIES}
            regret = {policy:[] for policy in POLICIES}
            prediction_mse = {policy:[] for policy in LEARNERS}
            oracle_values = []
            for block in test_blocks:
                phase_errors = {policy:[] for policy in POLICIES}
                phase_regret = {policy:[] for policy in POLICIES}
                phase_mse = {policy:[] for policy in LEARNERS}
                oracles = []
                for phase in PHASES:
                    rr = [test_by_key[(block,phase,radius,c,dt)] for c in range(18)]
                    if not all(r["status"] == "complete" for r in rr):
                        raise RuntimeError("Incomplete test condition: requires explicit failed-result audit")
                    actual = np.array([r["rmse"] for r in rr]); oracle=float(actual.min());oracles.append(oracle)
                    decision=decision_index[(block,phase,radius)]
                    for policy in POLICIES:
                        chosen=decision["policies"][policy]["candidate"]
                        phase_errors[policy].append(float(actual[chosen]));phase_regret[policy].append(float(actual[chosen]-oracle))
                        if policy in LEARNERS:
                            phase_mse[policy].append(float(np.mean((np.asarray(decision["predictions"][policy])-actual)**2)))
                oracle_values.append(float(np.mean(oracles)))
                for policy in POLICIES:
                    errors[policy].append(float(np.mean(phase_errors[policy])))
                    regret[policy].append(float(np.mean(phase_regret[policy])))
                    if policy in LEARNERS:prediction_mse[policy].append(float(np.mean(phase_mse[policy])))
            per_dose[str(radius)] = dict(error=errors, regret=regret, prediction_mse=prediction_mse, oracle=oracle_values)
        pooled = {metric: {policy:np.mean([per_dose[str(r)][metric][policy] for r in RADII], axis=0)
                   for policy in (LEARNERS if metric == "prediction_mse" else POLICIES)}
                  for metric in ("error","regret","prediction_mse")}
        pooled["oracle"] = np.mean([per_dose[str(r)]["oracle"] for r in RADII],axis=0)
        per_dose["pooled"] = pooled
        result_steps[str(dt)] = {}
        for dose, values in per_dose.items():
            panel=summary["steps"][str(dt)]["panels"][dose]
            for policy in POLICIES:
                close(panel["per_block"][policy]["selected_rmse"],values["error"][policy])
                close(panel["per_block"][policy]["regret"],values["regret"][policy])
                if policy in LEARNERS:close(panel["prediction_per_block"][policy]["mse"],values["prediction_mse"][policy])
            comparisons = {}
            for name,left,right in (("baseline_minus_geometry","baseline","geometry"),
                                     ("baseline_minus_shuffled","baseline","shuffled_geometry"),
                                     ("shuffled_minus_geometry","shuffled_geometry","geometry"),
                                     ("min_preview_minus_geometry","min_preview","geometry")):
                difference = np.asarray(values["error"][left])-np.asarray(values["error"][right])
                ci=np.percentile(difference[draws].mean(axis=1),[2.5,97.5])
                stored=panel["contrasts"][name]
                close(stored["mean_benefit"],difference.mean());close(stored["ci95"],ci)
                comparisons[name]=dict(mean=float(difference.mean()),ci95=ci,positive_blocks=int((difference>0).sum()),per_block=difference)
            result_steps[str(dt)][dose]=dict(mean_error={k:float(np.mean(v)) for k,v in values["error"].items()},
                mean_regret={k:float(np.mean(v)) for k,v in values["regret"].items()},
                mean_prediction_mse={k:float(np.mean(v)) for k,v in values["prediction_mse"].items()},
                mean_oracle=float(np.mean(values["oracle"])),contrasts=comparisons)
    report["outcomes"] = result_steps
    primary=result_steps["0.005"]["pooled"]["contrasts"]["baseline_minus_geometry"]
    close(summary["primary"]["mean_benefit"],primary["mean"])
    report["primary"] = primary
    paired = {}
    for role, rr in records.items():
        by_key={(r["anchor"],r["phase"],r["radius"],r["candidate"],r["dt"]):r for r in rr}
        delta=[abs(r["rmse"]-by_key[(r["anchor"],r["phase"],r["radius"],r["candidate"],.01)]["rmse"])
               for r in rr if r["dt"] == .005 and r["status"] == "complete"]
        paired[role]=dict(pairs=len(delta),mean=float(np.mean(delta)),median=float(np.median(delta)),
                          p95=float(np.percentile(delta,95)),maximum=float(max(delta)))
    report["coarse_fine_absolute_rmse_differences"] = paired
    zero = {}
    for role, rr in records.items():
        for dt in DTS:
            values = [r["rmse"] for r in rr if r["candidate"] == -1 and r["dt"] == dt]
            assert len(values) == 24
            zero[f"{role}_{dt}"] = dict(n=24, mean=float(np.mean(values)), maximum=float(max(values)))
    report["zero_shams"] = zero
    selected_step_differences = {}
    for policy in POLICIES:
        delta = [abs(test_by_key[(*context,d["policies"][policy]["candidate"],.005)]["rmse"]
                     -test_by_key[(*context,d["policies"][policy]["candidate"],.01)]["rmse"])
                 for context,d in decision_index.items()]
        selected_step_differences[policy] = dict(n=len(delta), mean=float(np.mean(delta)), maximum=float(max(delta)))
    report["selected_policy_coarse_fine_differences"] = selected_step_differences
    report["policy_action_identity_counts"] = {f'{a}__{b}':sum(d["policies"][a]["candidate"]==d["policies"][b]["candidate"] for d in decisions)
        for i,a in enumerate(POLICIES) for b in POLICIES[i+1:]}
    report["disposition"] = ("CV, sealed prediction/selection and block-outcome checks passed; "
        "independent coefficient comparisons retained individually without changing their tolerances")
    report["audit_wall_seconds"] = time.perf_counter()-start
    return plain(report)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("attempt",type=Path)
    parser.add_argument("--out",type=Path)
    args=parser.parse_args()
    attempt=args.attempt.resolve()
    if args.out is not None and args.out.resolve().is_relative_to(attempt):
        parser.error("Audit outputs must not modify the immutable attempt")
    result=audit(attempt)
    rendered=json.dumps(result,indent=2,allow_nan=False)+"\n"
    if args.out is not None:args.out.write_text(rendered)
    else:print(rendered)


if __name__ == "__main__":main()
