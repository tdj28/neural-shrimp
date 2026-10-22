"""BSC6 preparation: fresh local admission of a conditional mixed-junction target.

Continue the near-crease intersection tau=0, f_plus=0 at three declared betas;
measure f_minus there. Every correction needs new smooth-field stencils. Trace
value projections are explicit conditional operations, never measured zeros.
This pilot supplies no pseudo-arclength continuation or connectivity claim.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[key] = "8"
import numpy as np
from baby_shrimp_memory_connection_probe import (
    ROOT, DATA, SOURCES as PRIOR_SOURCES, require, digest, save, checked, replay_seed)
from contraction_valley_geometry import quadratic_jet, cusp_branch_limit_jets, hessian_valley_feature

SOURCES = list(dict.fromkeys(PRIOR_SOURCES + [
    "tools/baby_shrimp_mixed_junction_probe.py", "tools/contraction_valley_geometry.py",
    "tools/baby_shrimp_contraction_minima_finish.py"]))
SIDES = ("positive_tau_side", "negative_tau_side")
SCOPE = ("Exploratory fresh-field admission of a conditional feature intersection. "
         "Empirical error balls and local quadratic transfer are not verified neural error bounds. "
         "No exact trace zero, intrinsic ridge connectivity, or superstability certificate.")


def read(path):
    return json.loads(Path(path).read_text(),
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


def finite(value, shape=None):
    try:
        a = np.asarray(value, float)
        return bool(np.isfinite(a).all() and (shape is None or a.shape == shape))
    except (TypeError, ValueError):
        return False


def validate_plan(plan):
    c = plan["mixed_junction"]
    require(plan["id"] == "BSC6-mixed-junction-probe" and plan["period"] == 18
            and type(plan["maximum_evaluations"]) is int and 1 <= plan["maximum_evaluations"] <= 120,
            "wrong pilot identity, period or point budget")
    require(type(c["allow_scientifically_partial_bsc5"]) is bool, "explicit predecessor policy required")
    target = c["target"]
    require(finite(target["beta"]) and abs(target["beta"]) <= plan["maximum_abs_beta"]
            and finite(target["coordinate"], (2,)) and finite(target["moving_center_beta"], (2,)), "invalid declared target")
    for side in SIDES:
        ref = target["references"][side]
        require(finite(ref, (2,)) and np.linalg.norm(ref) > 0, "invalid transported reference")
    offsets = c["beta_offsets"]
    require(len(offsets) == 3 and finite(offsets) and offsets[0] == 0 and offsets[1] < 0 < offsets[2]
            and offsets[1] == -offsets[2] and offsets[2] <= 1e-8, "declare central and symmetric nearby betas")
    require(c["stencil_steps"] == [.0003, .0001] and c["primary_step"] == .0003 and c["maximum_corrections"] == 2,
            "two declared matched scales and at most two corrections required")
    positive = ("maximum_step", "maximum_total_displacement", "coordinate_tolerance", "trace_tolerance",
        "f_plus_tolerance", "f_minus_tolerance", "maximum_projection_displacement",
        "maximum_predictor_projection_displacement", "minimum_trace_gradient", "jacobian_step",
        "maximum_jacobian_condition", "minimum_normalized_jacobian_singular_value", "minimum_reference_overlap", "noise_multiplier", "signal_multiple",
        "maximum_feature_uncertainty", "maximum_f_minus_total_uncertainty", "maximum_intersection_location_uncertainty", "maximum_jacobian_perturbation",
        "orbit_alignment_tolerance")
    require(all(finite(c[k]) and c[k] > 0 for k in positive), "all admission scales must be positive finite values")
    require(c["maximum_step"] <= .0005 and c["maximum_total_displacement"] <= .0005
            and c["coordinate_tolerance"] <= 2e-6 and c["trace_tolerance"] <= 1e-12 and c["f_plus_tolerance"] <= 1e-8
            and c["maximum_projection_displacement"] <= c["coordinate_tolerance"]
            and c["maximum_predictor_projection_displacement"] <= c["maximum_total_displacement"]
            and c["orbit_alignment_tolerance"] <= 1e-4 and c["minimum_reference_overlap"] < 1
            and c["maximum_jacobian_perturbation"] < 1 and .2 <= c["minimum_normalized_jacobian_singular_value"] < 1
            and c["noise_multiplier"] >= 1 and c["signal_multiple"] >= 10
            and c["maximum_f_minus_total_uncertainty"] <= 5e-8 and c["maximum_intersection_location_uncertainty"] <= 5e-6,
            "pilot trust, precision or signal gate loosened")
    for field in ("tau", "delta"):
        require(finite(c["field_noise_floor"][field]) and c["field_noise_floor"][field] > 0, "positive empirical noise floor required")
        require(all(finite(c["jet_difference_limits"][field][k]) and c["jet_difference_limits"][field][k] > 0
                    for k in ("value", "gradient", "hessian")), "explicit paired-jet discrepancy limits required")
    return c


def field_jet(values, step):
    a = np.asarray(values, float)
    require(a.shape == (3, 3) and finite(a) and finite(step) and step > 0, "finite nine-point field stencil required")
    cross = (a[2, 2]-a[0, 2]-a[2, 0]+a[0, 0])/(4*step*step)
    return {"value": float(a[1, 1]),
        "gradient": [float((a[1, 2]-a[1, 0])/(2*step)), float((a[2, 1]-a[0, 1])/(2*step))],
        "hessian": [[float((a[1, 2]-2*a[1, 1]+a[1, 0])/step**2), float(cross)],
                    [float(cross), float((a[2, 1]-2*a[1, 1]+a[0, 1])/step**2)]]}


def conditional_geometry(fields, references, cfg, *, source="fresh measured field jet"):
    """Keep raw tau; explicitly change only its value for conditional limits."""
    tau, delta = fields["tau"], fields["delta"]
    gt = np.asarray(tau["gradient"], float); norm = float(np.linalg.norm(gt))
    require(norm > cfg["minimum_trace_gradient"], "regular trace gradient unresolved")
    displacement = -tau["value"]*gt/norm**2
    require(np.linalg.norm(displacement) <= cfg["maximum_predictor_projection_displacement"],
            "conditional trace projection leaves declared trust region")
    projected = {**tau, "value": 0.}
    limits = cusp_branch_limit_jets(projected, delta)
    branches = {}
    for side in SIDES:
        feature = hessian_valley_feature(limits[side], reference=references[side])
        require(feature["defined"] and feature["reference_overlap"] >= cfg["minimum_reference_overlap"],
                "one-sided feature or transported normal unresolved: "+side)
        branches[side] = {"jet": limits[side], "valley": feature}
    return {"input_scope": source, "input_tau": tau["value"], "input_delta": delta["value"],
        "projection": {"kind": "conditional trace-value projection at the same coordinate", "input_tau": tau["value"],
            "enforced_tau": 0., "absolute_trace_change": abs(tau["value"]),
            "predicted_normal_displacement": displacement.tolist(), "predicted_displacement_norm": float(np.linalg.norm(displacement)),
            "spatial_displacement_applied": False, "scope": "The recorded displacement is a local prediction, not a new measured point."},
        "cusp": limits["cusp"], "branches": branches, "scope": SCOPE}


def local_equations(fields, displacement, references, cfg):
    translated = {name: quadratic_jet(fields[name], displacement) for name in ("tau", "delta")}
    geometry = conditional_geometry(translated, references, cfg, source="translated quadratic field predictor")
    return np.array([translated["tau"]["value"], geometry["branches"][SIDES[0]]["valley"]["feature"],
                     geometry["branches"][SIDES[1]]["valley"]["feature"]])


def newton_correction(fields, references, cfg):
    """Numerical Jacobian of the pure field model, never a fitted cusp Hessian."""
    origin = local_equations(fields, np.zeros(2), references, cfg)
    h = cfg["jacobian_step"]
    def derivatives(step):
        return np.column_stack([(local_equations(fields, step*e, references, cfg)
                                 -local_equations(fields, -step*e, references, cfg))/(2*step) for e in np.eye(2)])
    derivative = derivatives(h)
    half_derivative = derivatives(h/2)
    J = derivative[:2]
    scales = np.array([cfg["trace_tolerance"], cfg["f_plus_tolerance"]])
    scaled = J/scales[:, None]
    condition = float(np.linalg.cond(scaled))
    row_norms = np.linalg.norm(J, axis=1)
    require(np.all(row_norms > 0), "zero intersection Jacobian row")
    normalized_singular_values = np.linalg.svd(J/row_norms[:, None], compute_uv=False)
    require(finite(derivative) and finite(condition) and condition <= cfg["maximum_jacobian_condition"],
            "intersection corrector Jacobian unresolved")
    require(normalized_singular_values[-1] >= cfg["minimum_normalized_jacobian_singular_value"],
            "normalized intersection constraints unresolved")
    correction = np.linalg.solve(scaled, -origin[:2]/scales)
    numerical_perturbation = float(np.linalg.norm(np.linalg.inv(scaled), 2)
        *np.linalg.norm((half_derivative[:2]-J)/scales[:, None], 2))
    require(numerical_perturbation <= cfg["maximum_jacobian_perturbation"], "pure-model Jacobian half-step agreement unresolved")
    require(finite(correction), "nonfinite intersection correction")
    return {"equations": origin[:2].tolist(), "f_minus": float(origin[2]), "jacobian_step": h,
        "jacobian": J.tolist(), "f_minus_gradient": derivative[2].tolist(), "equation_scales": scales.tolist(),
        "normalized_jacobian_singular_values": normalized_singular_values.tolist(),
        "jacobian_check_step": h/2, "jacobian_half_step": half_derivative[:2].tolist(),
        "f_minus_gradient_half_step": half_derivative[2].tolist(), "half_step_jacobian_perturbation": numerical_perturbation,
        "scaled_condition": condition, "correction": correction.tolist(), "correction_norm": float(np.linalg.norm(correction)),
        "scope": "Local quadratic-field Newton predictor. A correction needs fresh field measurements."}


def intersection_transfer(primary_corrector, check_corrector, equation_uncertainty, cfg):
    """Conditional inverse perturbation with its possible directional rotation."""
    scales = np.asarray(primary_corrector["equation_scales"])
    J = np.asarray(primary_corrector["jacobian"])/scales[:, None]
    dJ = (np.asarray(check_corrector["jacobian"])-primary_corrector["jacobian"])/scales[:, None]
    inv = np.linalg.inv(J)
    primary_half = (np.asarray(primary_corrector["jacobian_half_step"])-primary_corrector["jacobian"])/scales[:, None]
    check_half = (np.asarray(check_corrector["jacobian_half_step"])-check_corrector["jacobian"])/scales[:, None]
    jacobian_errors = {"matched_fields": float(np.linalg.norm(dJ, 2)),
                      "primary_half_step": float(np.linalg.norm(primary_half, 2)),
                      "check_half_step": float(np.linalg.norm(check_half, 2))}
    perturbation = float(np.linalg.norm(inv, 2)*sum(jacobian_errors.values()))
    require(perturbation <= cfg["maximum_jacobian_perturbation"], "paired intersection Jacobians unresolved")
    equation_error = (np.asarray(equation_uncertainty)+abs(np.asarray(primary_corrector["equations"])))/scales
    nominal_components = abs(inv) @ equation_error
    nominal_norm = float(np.linalg.norm(nominal_components))
    location_error = nominal_norm/(1-perturbation)
    gradient = np.asarray(primary_corrector["f_minus_gradient"])
    gradient_difference = float(np.linalg.norm(gradient-check_corrector["f_minus_gradient"])
        +np.linalg.norm(gradient-primary_corrector["f_minus_gradient_half_step"])
        +np.linalg.norm(np.asarray(check_corrector["f_minus_gradient"])-check_corrector["f_minus_gradient_half_step"]))
    sensitivities = gradient @ inv
    nominal_transfer = float(abs(sensitivities) @ equation_error)
    rotation_transfer = float(np.linalg.norm(gradient)*perturbation*location_error)
    gradient_transfer = gradient_difference*location_error
    return {"jacobian_perturbation": perturbation, "scaled_jacobian_discrepancy_norms": jacobian_errors,
        "nominal_location_box_components": nominal_components.tolist(),
        "nominal_location_norm": nominal_norm, "location_error_norm": location_error,
        "scaled_equation_errors": equation_error.tolist(), "f_minus_scaled_equation_sensitivities": sensitivities.tolist(),
        "f_minus_gradient_difference": gradient_difference, "f_minus_nominal_transfer": nominal_transfer,
        "inverse_rotation_transfer": rotation_transfer, "f_minus_gradient_transfer": gradient_transfer,
        "f_minus_location_transfer": nominal_transfer+rotation_transfer+gradient_transfer,
        "scope": "Inflated location norm only; nominal component box is not a perturbed componentwise bound. Conditional local linear-model transfer."}


def empirical_uncertainty(primary, check, primary_geometry, check_geometry, primary_corrector, check_corrector,
                          prior_variations, cyclic_variations, cfg):
    """Algebraic envelopes conditional on empirical field-error balls.

    Noise spans and paired stencils are not certified error bounds. The
    algebraic propagation is explicit so the experimental assumption remains
    separate from the full-map measurements and the local-model transfer.
    """
    h = primary["step"]; fields = primary["fields"]
    require(h == cfg["primary_step"] and check["step"] == cfg["stencil_steps"][1], "frozen primary/check scale differs")
    errors, differences = {}, {}
    for name in ("tau", "delta"):
        a, b = check["fields"][name], fields[name]
        diff = {"value": abs(a["value"]-b["value"]),
                "gradient": float(np.linalg.norm(np.asarray(a["gradient"])-b["gradient"])),
                "hessian": float(np.linalg.norm(np.asarray(a["hessian"])-b["hessian"], 2))}
        epsilon = (cfg["noise_multiplier"]*max(prior_variations[name], cyclic_variations[name], cfg["field_noise_floor"][name])
                   +diff["value"])
        differences[name] = diff
        errors[name] = {"value": epsilon, "gradient": diff["gradient"]+np.sqrt(2)*epsilon/h,
                        "hessian": diff["hessian"]+5*epsilon/h**2}
    tau, delta = fields["tau"], fields["delta"]
    gt, gd = np.asarray(tau["gradient"]), np.asarray(delta["gradient"])
    a, b = float(np.linalg.norm(gt)), float(np.linalg.norm(gd))
    Et, Ed = errors["tau"], errors["delta"]
    require(delta["value"]+Ed["value"] < 0 and a > Et["gradient"], "empirical balls lose opposite signs or regular crease")
    r, rmin = np.sqrt(-delta["value"]), np.sqrt(-delta["value"]-Ed["value"])
    I1, I3 = 1/rmin-1/r, 1/rmin**3-1/r**3
    EG = Ed["gradient"]/(2*rmin)+b*I1/2+Et["gradient"]/2
    EH = ((2*a*Et["gradient"]+Et["gradient"]**2)/(4*rmin)+a*a*I1/4
          +Ed["hessian"]/(2*rmin)+np.linalg.norm(delta["hessian"], 2)*I1/2
          +(2*b*Ed["gradient"]+Ed["gradient"]**2)/(4*rmin**3)+b*b*I3/4+Et["hessian"]/2)
    En = min(2., 2*Et["gradient"]/(a-Et["gradient"]))
    B = float(np.asarray(primary_geometry["cusp"]["normal"]) @ gd)
    EB = Ed["gradient"]+b*En
    slope_error = (EB/rmin+abs(B)*I1+Et["gradient"])/2
    projection_radius = (abs(tau["value"])+Et["value"])/(a-Et["gradient"])
    require(projection_radius <= cfg["maximum_projection_displacement"], "conditional projection uncertainty unresolved")
    direction = gt/a
    projected = [local_equations(fields, sign*projection_radius*direction,
                 {s: primary_geometry["branches"][s]["valley"]["normal"] for s in SIDES}, cfg) for sign in (-1, 1)]
    branches = {}
    for j, side in enumerate(SIDES, 1):
        primary_branch = primary_geometry["branches"][side]
        valley = primary_branch["valley"]; gap = valley["eigenvalue_gap"]
        require(EH < gap/2, "empirical Hessian ball loses a simple valley normal")
        Enormal = np.sqrt(2)*EH/(gap-EH)
        paired = abs(valley["feature"]-check_geometry["branches"][side]["valley"]["feature"])
        projection = max(abs(p[j]-valley["feature"]) for p in projected)
        Ef = EG+np.linalg.norm(primary_branch["jet"]["gradient"])*Enormal+paired+projection
        branches[side] = {"gradient_error": float(EG), "hessian_error": float(EH),
            "normal_error": float(Enormal), "paired_feature_difference": paired, "projection_sensitivity": projection,
            "feature_error": float(Ef), "positive_curvature_lower": float(valley["largest_eigenvalue"]-EH),
            "gap_lower": float(gap-2*EH), "transport_overlap_lower": float(valley["reference_overlap"]-Enormal)}
    transfer = intersection_transfer(primary_corrector, check_corrector,
                                     [Et["value"], branches[SIDES[0]]["feature_error"]], cfg)
    transfer["f_minus_total_error"] = branches[SIDES[1]]["feature_error"]+transfer["f_minus_location_transfer"]
    return {"primary_step": h, "check_step": check["step"], "field_error_balls": errors, "paired_jet_differences": differences,
        "branch_envelopes": branches, "cusp": {"normal_error": En, "trace_gradient_lower": a-Et["gradient"], "slope_error": float(slope_error),
            "left_upper": float(primary_geometry["cusp"]["left_slope"]+slope_error),
            "right_lower": float(primary_geometry["cusp"]["right_slope"]-slope_error)},
        "projection_radius": float(projection_radius),
        "normalized_jacobian_minimum_singular_values": [primary_corrector["normalized_jacobian_singular_values"][-1],
                                                        check_corrector["normalized_jacobian_singular_values"][-1]],
        "intersection": transfer,
        "assumptions": "Algebraic propagation over empirical field-error balls; local quadratic Jacobian and projection transfer. Not validated neural error bounds."}


def admission(geometry, corrector, uncertainty, cyclic_point, cfg):
    branches = uncertainty["branch_envelopes"]
    checks = {"actual_cyclic_trace": max(abs(p["tau"]) for p in cyclic_point["phases"]) <= cfg["trace_tolerance"],
        "actual_primary_trace": abs(geometry["input_tau"]) <= cfg["trace_tolerance"],
        "f_plus_residual": abs(corrector["equations"][1]) <= cfg["f_plus_tolerance"],
        "coordinate_correction": corrector["correction_norm"] <= cfg["coordinate_tolerance"],
        "projection_displacement": geometry["projection"]["predicted_displacement_norm"] <= cfg["maximum_projection_displacement"],
        "paired_field_jets": all(d <= cfg["jet_difference_limits"][name][key]
                                 for name, ds in uncertainty["paired_jet_differences"].items() for key, d in ds.items()),
        "regular_crease_with_uncertainty": uncertainty["cusp"]["trace_gradient_lower"] >= cfg["minimum_trace_gradient"],
        "both_normalized_jacobians": min(uncertainty["normalized_jacobian_minimum_singular_values"]) >= cfg["minimum_normalized_jacobian_singular_value"],
        "strict_cusp_slopes": uncertainty["cusp"]["left_upper"] < 0 < uncertainty["cusp"]["right_lower"],
        "positive_simple_transported_normals": all(b["positive_curvature_lower"] > 0 and b["gap_lower"] > 0
                                                   and b["transport_overlap_lower"] >= cfg["minimum_reference_overlap"] for b in branches.values()),
        "feature_precision": all(b["feature_error"] <= cfg["maximum_feature_uncertainty"] for b in branches.values()),
        "intersection_precision": uncertainty["intersection"]["location_error_norm"] <= cfg["maximum_intersection_location_uncertainty"],
        "f_minus_total_precision": uncertainty["intersection"]["f_minus_total_error"] <= cfg["maximum_f_minus_total_uncertainty"]}
    error = uncertainty["intersection"]["f_minus_total_error"]
    value = corrector["f_minus"]
    status = ("resolved_positive" if value > cfg["signal_multiple"]*error else
              "resolved_negative" if value < -cfg["signal_multiple"]*error else
              "residual_qualified_near_zero" if abs(value) <= cfg["f_minus_tolerance"] and error <= cfg["maximum_f_minus_total_uncertainty"] else
              "unresolved_sign")
    passed = all(checks.values())
    return {"passed": passed, "checks": {k: bool(v) for k, v in checks.items()},
        "f_minus": {"value": value, "empirical_total_error": error, "status": status if passed else "not_admitted",
                    "ungated_sign_diagnostic": status,
                    "scope": "Conditional local feature value at a numerically corrected intersection; not an exact zero."}, "scope": SCOPE}


def measured_jet(evaluate, center, beta, step):
    center = np.asarray(center, float)
    points = [[evaluate(center+step*np.array([dx, dy]), beta) for dx in (-1, 0, 1)] for dy in (-1, 0, 1)]
    return {"center": center.tolist(), "beta": beta, "step": step,
        "point_indices": [[p["index"] for p in row] for row in points],
        "fields": {name: field_jet([[p[name] for p in row] for row in points], step) for name in ("tau", "delta")}}


def case_predictor(target, offset, central=None):
    """Outward beta predictors retain the freshly admitted central correction."""
    if offset == 0:
        base = np.asarray(target["coordinate"])
        parent = {"kind": "frozen conditional target", "beta": target["beta"], "coordinate": base.tolist(), "point_index": None}
    else:
        require(central is not None and central.get("complete") is True and central.get("status") == "admitted"
                and central["beta_offset"] == 0, "outward predictor requires admitted central intersection")
        base = np.asarray(central["final_center"])
        parent = {"kind": "fresh admitted central intersection", "beta": central["beta"],
                  "coordinate": base.tolist(), "point_index": central["final_point_index"]}
    parent.update(beta_displacement=offset, moving_center_beta=target["moving_center_beta"],
                  scope="Historical moving-center secant is a predictor only; fresh two-equation corrections follow.")
    return base+offset*np.asarray(target["moving_center_beta"]), parent


def probe_case(evaluate, predicted_center, beta, beta_offset, references, prior_variations, cfg, checkpoint, *, predictor_parent=None):
    """Fresh, bounded 2D corrections, followed by matched-jet admission."""
    origin = np.asarray(predicted_center, float); center = origin.copy()
    references = {k: np.asarray(v).tolist() for k, v in references.items()}
    case = {"beta": beta, "beta_offset": beta_offset, "predicted_center": origin.tolist(),
            "predictor_parent": predictor_parent, "initial_references": references,
            "iterations": [], "status": "running", "complete": False}
    checkpoint(case)
    for count in range(cfg["maximum_corrections"]+1):
        row = {"center": center.tolist(), "beta": beta, "corrections_already_applied": count}
        case["iterations"].append(row); checkpoint(case)
        coarse = row["coarse_jet"] = measured_jet(evaluate, center, beta, cfg["stencil_steps"][0]); checkpoint(case)
        geom = row["coarse_geometry"] = conditional_geometry(coarse["fields"], references, cfg)
        correction = row["coarse_correction"] = newton_correction(coarse["fields"], references, cfg); checkpoint(case)
        basic = (abs(geom["input_tau"]) <= cfg["trace_tolerance"]
                 and abs(correction["equations"][1]) <= cfg["f_plus_tolerance"]
                 and correction["correction_norm"] <= cfg["coordinate_tolerance"])
        if basic:
            fine = row["fine_jet"] = measured_jet(evaluate, center, beta, cfg["stencil_steps"][1]); checkpoint(case)
            row["fine_geometry"] = conditional_geometry(fine["fields"], references, cfg)
            row["fine_correction"] = newton_correction(fine["fields"], references, cfg)
            cyclic = evaluate(center, beta, cyclic=True)
            row["cyclic_point_index"] = cyclic["index"]
            row["cyclic_variations"] = {name: float(np.ptp([p[name] for p in cyclic["phases"]]
                                                          +[coarse["fields"][name]["value"]])) for name in ("tau", "delta")}
            checkpoint(case)
            row["uncertainty"] = empirical_uncertainty(coarse, fine, geom, row["fine_geometry"], correction,
                row["fine_correction"], prior_variations, row["cyclic_variations"], cfg)
            row["admission"] = admission(geom, correction, row["uncertainty"], cyclic, cfg)
            checkpoint(case)
            if row["admission"]["passed"]:
                case.update(status="admitted", complete=True, final_center=center.tolist(),
                    final_point_index=cyclic["index"], f_minus=row["admission"]["f_minus"],
                    final_references={side: geom["branches"][side]["valley"]["normal"] for side in SIDES})
                checkpoint(case); return case
            # A failed precision gate cannot be cured by merely moving center.
            unresolved = [k for k, v in row["admission"]["checks"].items() if not v
                          and k not in ("actual_cyclic_trace", "actual_primary_trace", "f_plus_residual", "coordinate_correction", "projection_displacement")]
            if unresolved:
                case.update(status="unresolved precision", failed_checks=unresolved); checkpoint(case)
                raise ValueError("target admission precision unresolved: "+", ".join(unresolved))
            # The primary scale is frozen; the fine jet never replaces it.
        if count == cfg["maximum_corrections"]:
            case["status"] = "correction budget exhausted"; checkpoint(case)
            raise ValueError("no admitted target after bounded fresh corrections")
        step = np.asarray(correction["correction"]); next_center = center+step
        require(np.linalg.norm(step) <= cfg["maximum_step"]
                and np.linalg.norm(next_center-origin) <= cfg["maximum_total_displacement"], "target corrector left trust region")
        row["step_applied"] = {"old_center": center.tolist(), "new_center": next_center.tolist(), "displacement": step.tolist(),
                               "norm": float(np.linalg.norm(step)), "cumulative_displacement": float(np.linalg.norm(next_center-origin))}
        references = {side: geom["branches"][side]["valley"]["normal"] for side in SIDES}
        center = next_center; checkpoint(case)
    raise RuntimeError("unreachable correction fallthrough")


def bind_prior(plan):
    """Re-run the exact completed BSC5 auditor on its immutable raw snapshot."""
    from baby_shrimp_memory_orbit_alignment import METADATA, git_bytes
    from baby_shrimp_contraction_minima_finish import run_audit, SOURCES as AUDIT_SOURCES
    cfg = plan["mixed_junction"]; bound = plan["bsc5_audit"]
    path = (DATA/bound["path"]).resolve()
    require(path.is_relative_to(DATA.resolve()) and path.is_file() and path.stat().st_size == bound["bytes"]
            and digest(path) == bound["sha256"], "BSC5 receipt binding differs")
    receipt = read(path)
    require(receipt["id"] == "BSC5-contraction-minima-offline-audit" and receipt.get("audit_finished") is True
            and type(receipt["complete"]) is bool and receipt["center_qualified"] is True, "independently audited BSC5 center required")
    require(receipt["complete"] or cfg["allow_scientifically_partial_bsc5"], "scientifically partial predecessor not allowed by plan")
    require(all(plan[k] == receipt["scientific_metadata"][k] for k in METADATA if k != "id"), "map or coordinate metadata changed")
    require(set(AUDIT_SOURCES) <= set(receipt["analysis_source_sha256"]), "BSC5 audit source manifest incomplete")
    for source, sha in receipt["analysis_source_sha256"].items():
        require(hashlib.sha256(git_bytes(receipt["analysis_commit"], source)).hexdigest() == sha
                and digest(ROOT/source) == sha, "BSC5 replay requires the exact bound auditor source: "+source)
    controller = receipt["lifecycle"].get("controller")
    args = {}
    if controller:
        require(controller["worker_reaped"] is True, "prior worker not reaped")
        args = {"controller_manifest": DATA/controller["manifest"]["path"], "manifest_sha": controller["manifest"]["sha256"],
                "controller_status": DATA/controller["status_artifact"]["path"], "status_sha": controller["status_artifact"]["sha256"]}
    else:
        require(receipt["lifecycle"]["python_finalization"] is True, "prior terminal lifecycle absent")
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1, user_api="blas"):
        replayed = run_audit(Path(receipt["raw_directory"]), allow_partial=cfg["allow_scientifically_partial_bsc5"], **args)
    ignored = {"generated_utc", "analysis_commit", "analysis_source_sha256", "threads"}
    require({k: v for k, v in receipt.items() if k not in ignored} == {k: v for k, v in replayed.items() if k not in ignored},
            "BSC5 independent replay differs from bound receipt")
    require(receipt["baseline"].get("passed") is True and receipt["seed_replay"].get("passed") is True
            and receipt["expanded_spectrum"]["complete"] is True and len(receipt["local_jets"]) >= 2
            and receipt["expanded_spectrum"]["point_index"] == receipt["center_point_index"], "BSC5 supporting setup incomplete")
    if not receipt["complete"]:
        review = cfg["partial_bsc5_review"]
        require(review["receipt_sha256"] == bound["sha256"]
                and review["retained_unresolved_stages"] == receipt["untested_or_incomplete_stages"]
                and review["section_statuses"] == [{"arm_id": s["arm_id"], "longitudinal": s["longitudinal"],
                    "status": s["search"]["status"], "complete": s["search"]["complete"]} for s in receipt["sections"]],
                "partial BSC5 outcomes require an exact disclosed review in the frozen plan")
    run = Path(receipt["raw_directory"])
    directory = run/f"point-{receipt['center_point_index']:04d}"
    point = read(directory/"point.json")
    require(point["qualified"] is True and [p["phase"] for p in point["phases"]] == [0, 6, 12], "BSC5 seed lacks cyclic point")
    orbit = np.load(directory/"points.npy", allow_pickle=False)
    bases = {}
    for phase in (0, 6, 12):
        with np.load(directory/f"phase-{phase}.npz", allow_pickle=False) as arrays:
            bases[phase] = arrays["q"].copy()
    return {"receipt": receipt, "seed": {"point": point, "directory": directory, "orbit": orbit, "bases": bases},
        "provenance": {"audit": bound, "supporting_audit_blas_threads": 1, "scientific_complete": receipt["complete"],
            "retained_unresolved_stages": receipt["untested_or_incomplete_stages"],
            "section_statuses": [{"arm_id": s["arm_id"], "longitudinal": s["longitudinal"], "status": s["search"]["status"],
                                  "complete": s["search"]["complete"]} for s in receipt["sections"]],
            "expanded_spectrum": receipt["expanded_spectrum"],
            "scope": "Bound BSC5 supporting spectra, not fresh expanded spectra at a BSC6 target. All supporting raw gates independently replayed."}}


class AdmittedEvaluator:
    """Audit every raw point and its full-state phase alignment before use."""
    def __init__(self, evaluator, reference_point, plan, records, checkpoint):
        self.ev, self.plan, self.records, self.checkpoint = evaluator, plan, records, checkpoint
        self.reference_point = reference_point
        self.reference = np.load(evaluator.out/f"point-{reference_point['index']:04d}"/"points.npy", allow_pickle=False)
        self.audited = set()

    def __call__(self, center, beta, cyclic=False):
        p = checked(self.ev, center, beta, cyclic=cyclic)
        if p["index"] in self.audited:
            return p
        from baby_shrimp_memory_finish import audit_point
        row = {"point_index": p["index"], "status": "started", "reference_point_index": self.reference_point["index"]}
        self.records.append(row); self.checkpoint()
        try:
            _, summary = audit_point(self.ev.out/f"point-{p['index']:04d}", self.plan)
            orbit = np.load(self.ev.out/f"point-{p['index']:04d}"/"points.npy", allow_pickle=False)
            distances = [float(np.max(np.linalg.norm(np.roll(orbit, shift, axis=0)-self.reference, axis=1)
                                     /np.linalg.norm(self.reference, axis=1))) for shift in range(18)]
            row.update(audit=summary, all_alignment_maximum_relative_changes=distances, best_right_roll=int(np.argmin(distances)))
            row["passed"] = bool(summary["closure_recomputed_from_saved_endpoint"] and row["best_right_roll"] == 0
                and distances[0] < self.plan["mixed_junction"]["orbit_alignment_tolerance"])
            row["status"] = "passed" if row["passed"] else "failed orbit alignment"
            self.checkpoint()
            require(row["passed"], "full-state orbit alignment or saved closure gate failed")
        except BaseException as error:
            row.update(status="failed", error_type=type(error).__name__, error=str(error)); self.checkpoint()
            raise
        self.audited.add(p["index"])
        return p


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--seconds", type=int, default=5400)
    parser.add_argument("--source-commit")
    args = parser.parse_args()
    require(90 < args.seconds <= 5400, "bounded local runtime must be within 5400 seconds")
    started = time.monotonic()
    plan_path = ROOT/args.plan; plan = read(plan_path); cfg = validate_plan(plan)
    commit = args.source_commit or subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    require(len(commit) == 40 and all(c in "0123456789abcdef" for c in commit), "exact execution commit required")
    hashes = {source: digest(ROOT/source) for source in SOURCES}
    for source, sha in {**hashes, args.plan: digest(plan_path)}.items():
        require(hashlib.sha256(subprocess.check_output(["git", "show", commit+":"+source], cwd=ROOT)).hexdigest() == sha,
                "execution source and plan must be committed: "+source)
    out = args.out.resolve()
    require(out.is_relative_to((DATA/"explore").resolve()) and not out.exists(), "exclusive external output required")
    out.mkdir(parents=True)
    def emit(stage, **kw):
        row = {"time": datetime.now(timezone.utc).isoformat(), "stage": stage, **kw}
        with (out/"progress.jsonl").open("a") as stream:
            stream.write(json.dumps(row, allow_nan=False)+"\n")
        print(json.dumps(row, allow_nan=False), flush=True)
    protocol = {"commit": commit, "source_sha256": hashes, "plan": plan, "plan_sha256": digest(plan_path),
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "started_utc": datetime.now(timezone.utc).isoformat()}
    save(out/"protocol.json", protocol)
    result = {"complete": False, "execution_finished": False, "baseline": None, "seed_replay": None,
              "prior_support": None, "point_admissions": [], "cases": [], "scope": SCOPE}
    ev = None
    def checkpoint():
        save(out/"probe-progress.json", result)
    try:
        bound = bind_prior(plan)
        protocol["prior_bsc5"] = bound["provenance"]; save(out/"protocol.json", protocol)
        result["prior_support"] = bound["provenance"]; checkpoint()
        from baby_shrimp_saddle_memory import Evaluator
        from baby_shrimp_memory_resume import seed_evaluator
        for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            os.environ[key] = "8"
        evaluation_args = argparse.Namespace(**vars(args))
        evaluation_args.seconds = int(args.seconds-(time.monotonic()-started)-90)
        require(evaluation_args.seconds > 0, "preflight consumed model budget")
        ev = Evaluator(evaluation_args, plan, out, emit)
        result["baseline"] = ev.baseline(); checkpoint()
        seed = bound["seed"]
        seed_evaluator(ev, {"warm": [(np.asarray(seed["point"]["coordinate"]), seed["point"]["beta"], seed["orbit"][0], seed["bases"])]})
        result["seed_replay"] = replay_seed(ev, seed, plan); checkpoint()
        fresh_directory = out/f"point-{result['seed_replay']['fresh_point_index']:04d}"
        fresh = read(fresh_directory/"point.json")
        fresh_orbit = np.load(fresh_directory/"points.npy", allow_pickle=False)
        fresh_bases = {}
        for phase in (0, 6, 12):
            with np.load(fresh_directory/f"phase-{phase}.npz", allow_pickle=False) as arrays:
                fresh_bases[phase] = arrays["q"].copy()
        seed_evaluator(ev, {"warm": [(np.asarray(fresh["coordinate"]), fresh["beta"], fresh_orbit[0], fresh_bases)]})
        evaluate = AdmittedEvaluator(ev, fresh, plan, result["point_admissions"], checkpoint)
        target = cfg["target"]
        central_references = target["references"]
        for offset in cfg["beta_offsets"]:
            beta = float(target["beta"]+offset)
            predicted, parent = case_predictor(target, offset, result["cases"][0] if offset else None)
            def case_checkpoint(case):
                if not result["cases"] or result["cases"][-1]["beta_offset"] != offset:
                    result["cases"].append(case)
                else:
                    result["cases"][-1] = case
                checkpoint()
            case = probe_case(evaluate, predicted, beta, offset, central_references,
                              bound["receipt"]["expanded_spectrum"]["variations"], cfg, case_checkpoint, predictor_parent=parent)
            if offset == 0:
                central_references = case["final_references"]
            emit("target_admitted", beta=beta, point_index=case["final_point_index"], f_minus=case["f_minus"])
        lower, upper = result["cases"][1:]
        labels = {lower["f_minus"]["status"], upper["f_minus"]["status"]}
        result.update(complete=True, execution_finished=True, evaluations=ev.attempts,
            resolved_f_minus_sign_bracket=labels == {"resolved_positive", "resolved_negative"},
            completed_utc=datetime.now(timezone.utc).isoformat())
        save(out/"result.json", result)
    except BaseException as error:
        if result["cases"] and result["cases"][-1]["status"] == "running":
            result["cases"][-1].update(status="unresolved", error_type=type(error).__name__, error=str(error))
        save(out/"failure.json", {"type": type(error).__name__, "error": str(error),
             "evaluations": None if ev is None else ev.attempts, "time": datetime.now(timezone.utc).isoformat()})
        raise
    finally:
        checkpoint()
        save(out/"manifest.json", {"complete": result["complete"], "files": [
            {"path": str(p.relative_to(out)), "bytes": p.stat().st_size, "sha256": digest(p)}
            for p in sorted(out.rglob("*")) if p.is_file() and p.name != "manifest.json"]})


if __name__ == "__main__":
    main()
