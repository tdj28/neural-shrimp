"""Independent saved-data replay of the BSC6 mixed-junction pilot.

No model calls. Conditional trace-zero extension jets are recomputed directly;
they are not a common Hessian of the nonsmooth radius or an exact-zero claim.
"""
import os
for name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[name] = "1"
import numpy as np
import baby_shrimp_contraction_minima_finish as previous

ROOT, DATA = previous.ROOT, previous.DATA
require, close, read, digest, artifact = previous.require, previous.close, previous.read, previous.digest, previous.artifact
point_audit = previous.point_audit


def finite_jet(jet, name):
    require(isinstance(jet, dict) and {"value", "gradient", "hessian"} <= set(jet), "missing "+name+" jet")
    value = jet["value"]
    require(not isinstance(value, (bool, np.bool_)) and np.isscalar(value) and np.isrealobj(value)
            and np.isfinite(value), "invalid "+name+" value")
    gradient, hessian = np.asarray(jet["gradient"]), np.asarray(jet["hessian"])
    require(gradient.shape == (2,) and hessian.shape == (2, 2)
            and gradient.dtype.kind in "iuf" and hessian.dtype.kind in "iuf"
            and np.isfinite(gradient).all() and np.isfinite(hessian).all()
            and np.array_equal(hessian, hessian.T), "invalid "+name+" derivatives")
    return float(value), gradient.astype(float), hessian.astype(float)


def independent_limit_jets(tau, delta):
    """Recompute two regular-cusp extension jets with actual residual retained.

    Setting the *field value* tau to zero here is a conditional algebraic
    projection at unchanged coordinates. The linear normal displacement is
    merely a diagnostic; it is not applied and supplies no certified bound.
    """
    t, gt, ht = finite_jet(tau, "tau")
    d, gd, hd = finite_jet(delta, "delta")
    norm = float(np.linalg.norm(gt))
    require(d < 0 and norm > 0 and np.isfinite(norm), "regular negative-determinant conditional crease required")
    r = float(np.sqrt(-d))
    n = gt/norm
    with np.errstate(over="raise", divide="raise", invalid="raise"):
        displacement = -t*gt/(norm*norm)
        common = np.outer(gt, gt)/(4*r)-hd/(2*r)-np.outer(gd, gd)/(4*r**3)
        branches = {}
        for sign, key in ((1, "positive_tau_side"), (-1, "negative_tau_side")):
            gradient = -gd/(2*r)+sign*gt/2
            hessian = common+sign*ht/2
            require(np.isfinite(gradient).all() and np.isfinite(hessian).all(), "nonfinite conditional branch derivatives")
            branches[key] = {"value": r, "gradient": gradient.tolist(), "hessian": hessian.tolist(), "branch_sign": sign}
        smooth_slope = -float(n@gd)/(2*r)
        left, right = smooth_slope-norm/2, smooth_slope+norm/2
    require(np.isfinite(displacement).all() and np.isfinite([left, right]).all(), "nonfinite conditional displacement/slopes")
    return {"actual_tau": t, "conditional_tau": 0., "normal_displacement_estimate": displacement.tolist(),
            "normal_displacement_norm": float(np.linalg.norm(displacement)), "tau_gradient_norm": norm,
            "trace_normal": n.tolist(), "left_slope": left, "right_slope": right, **branches,
            "scope": "Conditional field-value projection at unchanged coordinates; two distinct extension jets, no common cusp Hessian or certified displacement bound."}


def independent_feature(jet, reference):
    """Recompute transported Hessian-valley feature without worker helpers."""
    value, gradient, H = finite_jet(jet, "radius extension")
    reference = np.asarray(reference, float)
    require(value > 0 and reference.shape == (2,) and np.isfinite(reference).all()
            and np.linalg.norm(reference) > 0, "invalid radius or transported reference")
    reference = reference/np.linalg.norm(reference)
    values, vectors = np.linalg.eigh(H)
    gap = float(values[1]-values[0])
    require(np.isfinite(values).all() and np.isfinite(gap), "nonfinite extension eigenvalues")
    out = {"defined": False, "feature": None, "normal": None, "eigenvalues": values.tolist(),
           "largest_eigenvalue": float(values[1]), "eigenvalue_gap": gap, "reference_overlap": None}
    if gap == 0:
        return {**out, "status": "degenerate_largest_eigenvalue"}
    if values[1] <= 0:
        return {**out, "status": "nonpositive_largest_eigenvalue"}
    normal = vectors[:, 1]
    overlap = float(normal@reference)
    out["reference_overlap"] = abs(overlap)
    if overlap == 0:
        return {**out, "status": "orthogonal_reference"}
    if overlap < 0:
        normal = -normal
    feature = float(normal@gradient)
    require(np.isfinite(feature), "nonfinite transported feature")
    return {**out, "defined": True, "feature": feature, "normal": normal.tolist(), "status": "defined"}


def recompute_stencil(jet, raw):
    """Replay smooth tau/Delta stencils without imposing a saddle condition."""
    center, beta, step = np.asarray(jet["center"], float), jet["beta"], jet["step"]
    indices = np.asarray(jet["point_indices"])
    require(center.shape == (2,) and np.isfinite(center).all() and np.isfinite(beta)
            and np.isfinite(step) and step > 0 and indices.shape == (3, 3)
            and np.issubdtype(indices.dtype, np.integer), "invalid matched stencil coordinates/indices")
    fields = {}
    for field in ("tau", "delta"):
        values = np.empty((3, 3))
        for iy, dy in enumerate((-1, 0, 1)):
            for ix, dx in enumerate((-1, 0, 1)):
                p = previous.checked_point(raw, indices[iy, ix], center+step*np.array([dx, dy]), beta)
                values[iy, ix] = p[field]
        f, h = values, step
        gradient = np.array([(f[1, 2]-f[1, 0])/(2*h), (f[2, 1]-f[0, 1])/(2*h)])
        mixed = (f[2, 2]-f[0, 2]-f[2, 0]+f[0, 0])/(4*h*h)
        H = np.array([[(f[1, 2]-2*f[1, 1]+f[1, 0])/h**2, mixed],
                      [mixed, (f[2, 1]-2*f[1, 1]+f[0, 1])/h**2]])
        actual = {"value": float(f[1, 1]), "gradient": gradient.tolist(), "hessian": H.tolist()}
        for key, value in actual.items():
            close(jet["fields"][field][key], value, "stencil differs: "+field+"/"+key,
                  rtol=1e-10, atol=1e-18 if key == "value" else 1e-13)
        fields[field] = actual
    return {"center": center.tolist(), "beta": beta, "step": step, "point_indices": indices.tolist(), "fields": fields}


def independent_jet_error_bounds(tau, delta, errors, references):
    """Propagate declared empirical jet-error balls through the cusp algebra.

    These are conditional algebraic bounds. This function does not establish
    that the caller's finite-difference discrepancies bound the true errors.
    """
    _, gt, ht = finite_jet(tau, "tau")
    d, gd, hd = finite_jet(delta, "delta")
    keys = ("delta_value", "tau_gradient", "delta_gradient", "tau_hessian", "delta_hessian")
    require(all(k in errors and np.isfinite(errors[k]) and errors[k] >= 0 for k in keys), "invalid declared jet errors")
    ed, etg, edg, eth, edh = (float(errors[k]) for k in keys)
    require(-d > ed, "determinant error ball reaches zero")
    r, lower_r = float(np.sqrt(-d)), float(np.sqrt(-d-ed))
    a, b, hd_norm = float(np.linalg.norm(gt)), float(np.linalg.norm(gd)), float(np.linalg.norm(hd, 2))
    i1, i3 = 1/lower_r-1/r, 1/lower_r**3-1/r**3
    eg = edg/(2*lower_r)+b*i1/2+etg/2
    eh = ((2*a*etg+etg*etg)/(4*lower_r)+a*a*i1/4+edh/(2*lower_r)+hd_norm*i1/2
          +(2*b*edg+edg*edg)/(4*lower_r**3)+b*b*i3/4+eth/2)
    require(np.isfinite([eg, eh, i1, i3]).all(), "nonfinite conditional derivative error bounds")
    limits = independent_limit_jets(tau, delta)
    branches = {}
    for key in ("positive_tau_side", "negative_tau_side"):
        feature = independent_feature(limits[key], references[key])
        gap = feature["eigenvalue_gap"]
        orientation_error = float(np.sqrt(2)*eh/(gap-eh)) if feature["defined"] and eh < gap/2 else None
        gradient_norm = float(np.linalg.norm(limits[key]["gradient"]))
        branches[key] = {"positive_curvature_lower": feature["largest_eigenvalue"]-eh,
            "gap_lower": gap-2*eh, "eigenvector_error": orientation_error,
            "feature_error": None if orientation_error is None else eg+gradient_norm*orientation_error,
            "transport_sign_resolved": bool(orientation_error is not None and feature["reference_overlap"] > orientation_error)}
    slope_error, normal_error = None, None
    if a > etg:
        normal_error = min(2., 2*etg/(a-etg))
        n = gt/a; B = float(n@gd)
        eb = edg+b*normal_error
        slope_error = (eb/lower_r+abs(B)*i1+etg)/2
    return {"radius_lower": lower_r, "gradient_error": eg, "hessian_error": eh,
            "branches": branches, "trace_normal_error": normal_error, "cusp_slope_error": slope_error,
            "scope": "Algebraic propagation conditional on declared empirical jet-error balls; not a validated error bound."}


SIDES = ("positive_tau_side", "negative_tau_side")


def compare_numeric_tree(saved, expected, name):
    """Compare independently reconstructed scientific fields; ignore prose."""
    if isinstance(expected, dict):
        require(isinstance(saved, dict) and set(expected) <= set(saved), "missing replay field: "+name)
        for key, value in expected.items():
            compare_numeric_tree(saved[key], value, name+"/"+key)
    elif isinstance(expected, (list,tuple)):
        require(isinstance(saved,(list,tuple)) and len(saved)==len(expected),'replay sequence differs: '+name)
        for index,(actual,wanted) in enumerate(zip(saved,expected)):
            compare_numeric_tree(actual,wanted,name+'/'+str(index))
    elif expected is None or isinstance(expected, (bool, str)):
        require(type(saved) is type(expected) and saved == expected, "replay label differs: "+name)
    else:
        # Differencing independently evaluated smooth features amplifies their
        # last-bit algebraic rounding; the scalar gradient discrepancy can
        # therefore differ around 1e-13 while its transferred error agrees.
        difference_names=('/f_minus_gradient_difference','/half_step_jacobian_perturbation')
        atol=1e-12 if name.endswith(difference_names) else 1e-17
        close(saved, expected, "replay value differs: "+name, rtol=1e-8, atol=atol)


def geometry_from_fields(fields, references, cfg):
    limits = independent_limit_jets(fields["tau"], fields["delta"])
    require(limits["tau_gradient_norm"] > cfg["minimum_trace_gradient"]
            and limits["normal_displacement_norm"] <= cfg["maximum_predictor_projection_displacement"],
            "regular conditional projection leaves declared predictor gates")
    branches = {}
    for side in SIDES:
        feature = independent_feature(limits[side], references[side])
        require(feature["defined"] and feature["reference_overlap"] >= cfg["minimum_reference_overlap"],
                "transported conditional feature is unresolved")
        branches[side] = {"jet": limits[side], "valley": feature}
    left, right = limits["left_slope"], limits["right_slope"]
    strict = left < 0 < right
    delta_gradient = np.asarray(fields["delta"]["gradient"])
    return {"input_tau": limits["actual_tau"], "input_delta": fields["delta"]["value"],
        "projection": {"kind": "conditional trace-value projection at the same coordinate", "input_tau": limits["actual_tau"],
            "enforced_tau": 0., "absolute_trace_change": abs(limits["actual_tau"]),
            "predicted_normal_displacement": limits["normal_displacement_estimate"],
            "predicted_displacement_norm": limits["normal_displacement_norm"], "spatial_displacement_applied": False},
        "cusp": {"value": limits[SIDES[0]]["value"], "normal": limits["trace_normal"],
            "tau_gradient_norm": limits["tau_gradient_norm"], "normal_delta_gradient": float(np.asarray(limits["trace_normal"])@delta_gradient),
            "left_slope": left, "right_slope": right, "slope_margin": min(-left,right),
            "strict_first_order_minimum": strict, "status": "strict_first_order_minimum" if strict else
                "not_a_normal_minimum" if left > 0 or right < 0 else "first_order_degenerate"}, "branches": branches}


def translated_equations(fields, q, references, cfg):
    q = np.asarray(q, float)
    shifted = {}
    for name in ("tau", "delta"):
        value, g, H = finite_jet(fields[name], name)
        shifted[name] = {"value": float(value+g@q+.5*q@H@q), "gradient": (g+H@q).tolist(), "hessian": H.tolist()}
    geometry = geometry_from_fields(shifted, references, cfg)
    return np.array([shifted["tau"]["value"]]+[geometry["branches"][s]["valley"]["feature"] for s in SIDES])


def correction_from_fields(fields, references, cfg):
    origin = translated_equations(fields, np.zeros(2), references, cfg)
    h = cfg["jacobian_step"]
    derivative = np.column_stack([(translated_equations(fields,h*e,references,cfg)
                                   -translated_equations(fields,-h*e,references,cfg))/(2*h) for e in np.eye(2)])
    half_derivative = np.column_stack([(translated_equations(fields,(h/2)*e,references,cfg)
                                       -translated_equations(fields,-(h/2)*e,references,cfg))/h for e in np.eye(2)])
    J, scales = derivative[:2], np.array([cfg["trace_tolerance"], cfg["f_plus_tolerance"]])
    row_norms = np.linalg.norm(J,axis=1)
    require(min(row_norms)>0,"zero intersection constraint gradient")
    normalized_singular_values=np.linalg.svd(J/row_norms[:,None],compute_uv=False)
    scaled = J/scales[:,None]
    condition = float(np.linalg.cond(scaled))
    require(np.isfinite(derivative).all() and np.isfinite(condition) and condition <= cfg["maximum_jacobian_condition"],
            "conditional intersection Jacobian fails declared gate")
    require(normalized_singular_values[-1]>=cfg["minimum_normalized_jacobian_singular_value"],
            "normalized intersection constraints unresolved")
    correction = np.linalg.solve(scaled,-origin[:2]/scales)
    numerical_perturbation=float(np.linalg.norm(np.linalg.inv(scaled),2)*np.linalg.norm((half_derivative[:2]-J)/scales[:,None],2))
    require(numerical_perturbation<=cfg['maximum_jacobian_perturbation'],'pure-model Jacobian half-step agreement unresolved')
    require(np.isfinite(correction).all(), "nonfinite intersection correction")
    return {"equations": origin[:2].tolist(), "f_minus": float(origin[2]), "jacobian_step": h,
            "jacobian": J.tolist(), "f_minus_gradient": derivative[2].tolist(), "equation_scales": scales.tolist(),
            "normalized_jacobian_singular_values":normalized_singular_values.tolist(),
            "jacobian_check_step":h/2,"jacobian_half_step":half_derivative[:2].tolist(),
            "f_minus_gradient_half_step":half_derivative[2].tolist(),"half_step_jacobian_perturbation":numerical_perturbation,
            "scaled_condition": condition, "correction": correction.tolist(), "correction_norm": float(np.linalg.norm(correction))}


def uncertainty_from_jets(primary, check, primary_geometry, check_geometry, corrector, check_corrector,
                          prior_variations, cyclic_variations, cfg):
    h = primary["step"]
    require(h == cfg["primary_step"] and check["step"] == cfg["stencil_steps"][1], "matched uncertainty scales differ")
    errors, differences = {}, {}
    for name in ("tau", "delta"):
        a,b = check["fields"][name], primary["fields"][name]
        ds = {"value": abs(a["value"]-b["value"]),
              "gradient": float(np.linalg.norm(np.asarray(a["gradient"])-b["gradient"])),
              "hessian": float(np.linalg.norm(np.asarray(a["hessian"])-b["hessian"],2))}
        noise = cfg["noise_multiplier"]*max(prior_variations[name],cyclic_variations[name],cfg["field_noise_floor"][name])+ds['value']
        differences[name]=ds
        errors[name]={"value":noise,"gradient":ds["gradient"]+np.sqrt(2)*noise/h,"hessian":ds["hessian"]+5*noise/h**2}
    Et,Ed=errors["tau"],errors["delta"]
    refs={side:primary_geometry["branches"][side]["valley"]["normal"] for side in SIDES}
    bounds=independent_jet_error_bounds(primary["fields"]["tau"],primary["fields"]["delta"],
        {"delta_value":Ed["value"],"tau_gradient":Et["gradient"],"delta_gradient":Ed["gradient"],
         "tau_hessian":Et["hessian"],"delta_hessian":Ed["hessian"]},refs)
    gt=np.asarray(primary["fields"]["tau"]["gradient"]);norm=float(np.linalg.norm(gt))
    require(norm>Et["gradient"],"empirical trace-gradient ball reaches singular crease")
    radius=(abs(primary["fields"]["tau"]["value"])+Et["value"])/(norm-Et["gradient"])
    require(radius<=cfg["maximum_projection_displacement"],"conditional projection uncertainty exceeds gate")
    projected=[translated_equations(primary["fields"],sign*radius*gt/norm,refs,cfg) for sign in (-1,1)]
    branches={}
    for j,side in enumerate(SIDES,1):
        valley=primary_geometry["branches"][side]["valley"]
        bound=bounds["branches"][side]
        require(bound["eigenvector_error"] is not None,"empirical Hessian ball loses simple normal")
        paired=abs(valley["feature"]-check_geometry["branches"][side]["valley"]["feature"])
        projection=max(abs(v[j]-valley["feature"]) for v in projected)
        branches[side]={"gradient_error":bounds["gradient_error"],"hessian_error":bounds["hessian_error"],
            "normal_error":bound["eigenvector_error"],"paired_feature_difference":paired,"projection_sensitivity":projection,
            "feature_error":bound["feature_error"]+paired+projection,
            "positive_curvature_lower":valley["largest_eigenvalue"]-bounds["hessian_error"],
            "gap_lower":valley["eigenvalue_gap"]-2*bounds["hessian_error"],
            "transport_overlap_lower":valley["reference_overlap"]-bound["eigenvector_error"]}
    scales=np.asarray(corrector["equation_scales"])
    J=np.asarray(corrector["jacobian"])/scales[:,None]
    dJ=(np.asarray(check_corrector["jacobian"])-corrector["jacobian"])/scales[:,None]
    equation_errors=(np.array([Et["value"],branches[SIDES[0]]["feature_error"]])+abs(np.asarray(corrector["equations"])))/scales
    gradient=np.asarray(corrector["f_minus_gradient"])
    primary_half=(np.asarray(corrector['jacobian_half_step'])-corrector['jacobian'])/scales[:,None]
    check_half=(np.asarray(check_corrector['jacobian_half_step'])-check_corrector['jacobian'])/scales[:,None]
    discrepancies={'matched_fields':float(np.linalg.norm(dJ,2)),'primary_half_step':float(np.linalg.norm(primary_half,2)),
                   'check_half_step':float(np.linalg.norm(check_half,2))}
    gradient_difference=float(np.linalg.norm(gradient-check_corrector['f_minus_gradient'])
        +np.linalg.norm(gradient-corrector['f_minus_gradient_half_step'])
        +np.linalg.norm(np.asarray(check_corrector['f_minus_gradient'])-check_corrector['f_minus_gradient_half_step']))
    transferred=independent_location_transfer(J,dJ,equation_errors,gradient,gradient_difference,
                                               jacobian_error_norm=sum(discrepancies.values()))
    require(transferred["jacobian_perturbation"]<=cfg["maximum_jacobian_perturbation"]<1,
            "paired Jacobian perturbation exceeds gate")
    return {"primary_step":h,"check_step":check["step"],"field_error_balls":errors,"paired_jet_differences":differences,
        "branch_envelopes":branches,"projection_radius":radius,
        "cusp":{"normal_error":bounds["trace_normal_error"],"trace_gradient_lower":norm-Et['gradient'],
                "slope_error":bounds["cusp_slope_error"],
                "left_upper":primary_geometry["cusp"]["left_slope"]+bounds["cusp_slope_error"],
                "right_lower":primary_geometry["cusp"]["right_slope"]-bounds["cusp_slope_error"]},
        "normalized_jacobian_minimum_singular_values":[corrector['normalized_jacobian_singular_values'][-1],check_corrector['normalized_jacobian_singular_values'][-1]],
        "intersection":{"jacobian_perturbation":transferred['jacobian_perturbation'],
            "scaled_jacobian_discrepancy_norms":discrepancies,
            "nominal_location_box_components":transferred['nominal_location_component_box'],
            "nominal_location_norm":transferred['nominal_location_radius'],
            "location_error_norm":transferred['location_error_norm'],
            "f_minus_scaled_equation_sensitivities":transferred['f_minus_scaled_equation_sensitivities'],
            "f_minus_nominal_transfer":transferred['f_minus_nominal_location_transfer'],
            "inverse_rotation_transfer":transferred['f_minus_inverse_perturbation_transfer'],
            "f_minus_gradient_transfer":transferred['f_minus_gradient_change_transfer'],
            "f_minus_location_transfer":transferred['f_minus_location_transfer'],
            "scaled_equation_errors":equation_errors.tolist(),
            "f_minus_gradient_difference":gradient_difference,
            "f_minus_total_error":branches[SIDES[1]]["feature_error"]+transferred["f_minus_location_transfer"]}}


def independent_location_transfer(jacobian, jacobian_difference, equation_errors, gradient, gradient_error, *, jacobian_error_norm=None):
    """Conditional inverse-perturbation radius and scalar transfer envelope.

    A perturbation can rotate the inverse into a direction annihilated by the
    nominal scalar row. Thus only the nominal transfer may use that row;
    its perturbation remainder needs the full gradient norm.
    """
    J,dJ,e,g=(np.asarray(x,float) for x in (jacobian,jacobian_difference,equation_errors,gradient))
    require(J.shape==dJ.shape==(2,2) and e.shape==g.shape==(2,) and all(np.isfinite(x).all() for x in (J,dJ,e,g))
            and min(e)>=0 and np.isfinite(gradient_error) and gradient_error>=0,"invalid location-transfer inputs")
    inv=np.linalg.inv(J)
    error_norm=float(np.linalg.norm(dJ,2)) if jacobian_error_norm is None else jacobian_error_norm
    require(np.isfinite(error_norm) and error_norm>=np.linalg.norm(dJ,2),'invalid combined Jacobian discrepancy bound')
    k=float(np.linalg.norm(inv,2)*error_norm)
    require(k<1,"Jacobian perturbation does not admit inverse bound")
    components=abs(inv)@e
    nominal_radius=float(np.linalg.norm(components))
    remainder=k*nominal_radius/(1-k)
    location=nominal_radius/(1-k)
    sensitivities=g@inv
    nominal_transfer=float(abs(sensitivities)@e)
    rotation_transfer=float(np.linalg.norm(g)*remainder)
    gradient_transfer=float(gradient_error*location)
    return {"jacobian_perturbation":k,"nominal_location_component_box":components.tolist(),
            "nominal_location_radius":nominal_radius,"inverse_perturbation_radius":remainder,
            "location_error_norm":location,"f_minus_scaled_equation_sensitivities":sensitivities.tolist(),
            "f_minus_nominal_location_transfer":nominal_transfer,"f_minus_inverse_perturbation_transfer":rotation_transfer,
            "f_minus_gradient_change_transfer":gradient_transfer,
            "f_minus_location_transfer":nominal_transfer+rotation_transfer+gradient_transfer}


def admission_from_values(geometry, corrector, uncertainty, cyclic, cfg):
    require([p['phase'] for p in cyclic['phases']]==[0,6,12],"target admission lacks cyclic spectrum")
    branches=uncertainty['branch_envelopes']
    checks={
        'actual_cyclic_trace':max(abs(p['tau']) for p in cyclic['phases'])<=cfg['trace_tolerance'],
        'actual_primary_trace':abs(geometry['input_tau'])<=cfg['trace_tolerance'],
        'f_plus_residual':abs(corrector['equations'][1])<=cfg['f_plus_tolerance'],
        'coordinate_correction':corrector['correction_norm']<=cfg['coordinate_tolerance'],
        'projection_displacement':geometry['projection']['predicted_displacement_norm']<=cfg['maximum_projection_displacement'],
        'paired_field_jets':all(d<=cfg['jet_difference_limits'][name][key] for name,ds in uncertainty['paired_jet_differences'].items() for key,d in ds.items()),
        'regular_crease_with_uncertainty':uncertainty['cusp']['trace_gradient_lower']>=cfg['minimum_trace_gradient'],
        'both_normalized_jacobians':min(uncertainty['normalized_jacobian_minimum_singular_values'])>=cfg['minimum_normalized_jacobian_singular_value'],
        'strict_cusp_slopes':uncertainty['cusp']['left_upper']<0<uncertainty['cusp']['right_lower'],
        'positive_simple_transported_normals':all(b['positive_curvature_lower']>0 and b['gap_lower']>0
            and b['transport_overlap_lower']>=cfg['minimum_reference_overlap'] for b in branches.values()),
        'feature_precision':all(b['feature_error']<=cfg['maximum_feature_uncertainty'] for b in branches.values()),
        'intersection_precision':uncertainty['intersection']['location_error_norm']<=cfg['maximum_intersection_location_uncertainty'],
        'f_minus_total_precision':uncertainty['intersection']['f_minus_total_error']<=cfg['maximum_f_minus_total_uncertainty']}
    value,error=corrector['f_minus'],uncertainty['intersection']['f_minus_total_error']
    status=('resolved_positive' if value>cfg['signal_multiple']*error else 'resolved_negative' if value < -cfg['signal_multiple']*error
            else 'residual_qualified_near_zero' if abs(value)<=cfg['f_minus_tolerance'] and error<=cfg['maximum_f_minus_total_uncertainty']
            else 'unresolved_sign')
    passed=all(checks.values())
    return {'passed':passed,'checks':{k:bool(v) for k,v in checks.items()},
            'f_minus':{'value':value,'empirical_total_error':error,'status':status if passed else 'not_admitted',
                       'ungated_sign_diagnostic':status}}


def audit_cases(result, raw, admitted_indices, prior_variations, cfg):
    target=cfg['target']; cases=result.get('cases',[])
    require(len(cases)<=len(cfg['beta_offsets']),"too many target cases")
    references=target['references']; outputs=[]
    def check_point(index,center,beta,cyclic=False):
        require(index in admitted_indices,"target used a point without independent state admission")
        p=previous.checked_point(raw,index,center,beta)
        require(not cyclic or [v['phase'] for v in p['phases']]==[0,6,12],"target lacks fresh cyclic point")
        return p
    for case_index,(case,offset) in enumerate(zip(cases,cfg['beta_offsets'])):
        beta=float(target['beta']+offset)
        if offset==0:
            base=np.asarray(target['coordinate'])
            parent={'kind':'frozen conditional target','beta':target['beta'],'coordinate':base.tolist(),'point_index':None}
        else:
            require(outputs and outputs[0]['complete'],'outward predictor lacks admitted central case')
            base=np.asarray(outputs[0]['final_center'])
            parent={'kind':'fresh admitted central intersection','beta':outputs[0]['beta'],'coordinate':base.tolist(),
                    'point_index':outputs[0]['final_point_index']}
        parent.update(beta_displacement=offset,moving_center_beta=target['moving_center_beta'])
        origin=base+offset*np.asarray(target['moving_center_beta'])
        require(case['beta_offset']==offset and case['beta']==beta and type(case['complete']) is bool,
                "case beta order or status differs")
        close(case['predicted_center'],origin,"case predictor coordinate differs",rtol=0,atol=1e-13)
        compare_numeric_tree(case['initial_references'],references,'case initial references')
        compare_numeric_tree(case['predictor_parent'],parent,'case predictor parent')
        center=origin.copy(); iteration_references=references
        iterations=case['iterations']; checked=[]
        require(len(iterations)<=cfg['maximum_corrections']+1,"case correction budget exceeded")
        for count,row in enumerate(iterations):
            require(row['beta']==beta and row['corrections_already_applied']==count,"iteration beta/count differs")
            close(row['center'],center,"iteration coordinate differs",rtol=0,atol=1e-13)
            replay={'center':center.tolist(),'beta':beta,'corrections_already_applied':count}
            for prefix,step in (('coarse',cfg['stencil_steps'][0]),('fine',cfg['stencil_steps'][1])):
                require(not any(prefix+suffix in row for suffix in ('_geometry','_correction')) or prefix+'_jet' in row,
                        'derived target fields lack their measured stencil')
                if prefix+'_jet' not in row:continue
                jet=row[prefix+'_jet']
                require(jet['center']==row['center'] and jet['beta']==beta and jet['step']==step,"target matched stencil placement differs")
                for index in np.asarray(jet['point_indices']).reshape(-1):
                    require(int(index) in admitted_indices,"stencil uses point without admitted alignment")
                replay[prefix+'_jet']=recompute_stencil(jet,raw)
                if prefix+'_geometry' in row:
                    geometry=geometry_from_fields(replay[prefix+'_jet']['fields'],iteration_references,cfg)
                    compare_numeric_tree(row[prefix+'_geometry'],geometry,prefix+' geometry')
                    replay[prefix+'_geometry']=geometry
                if prefix+'_correction' in row:
                    require(prefix+'_geometry' in replay,"corrector lacks geometry")
                    correction=correction_from_fields(replay[prefix+'_jet']['fields'],iteration_references,cfg)
                    compare_numeric_tree(row[prefix+'_correction'],correction,prefix+' correction')
                    replay[prefix+'_correction']=correction
            if 'fine_jet' in replay:
                require('coarse_correction' in replay and abs(replay['coarse_geometry']['input_tau'])<=cfg['trace_tolerance']
                        and abs(replay['coarse_correction']['equations'][1])<=cfg['f_plus_tolerance']
                        and replay['coarse_correction']['correction_norm']<=cfg['coordinate_tolerance'],
                        "fine check began without primary candidate gates")
            if 'cyclic_point_index' in row:
                require('fine_correction' in replay,"cyclic target lacks matched corrector")
                cyclic=check_point(row['cyclic_point_index'],center,beta,cyclic=True)
                variations={k:float(np.ptp([p[k] for p in cyclic['phases']]+[replay['coarse_jet']['fields'][k]['value']])) for k in ('tau','delta')}
                require(row['cyclic_variations']==variations,"target cyclic variation differs")
                replay.update(cyclic_point_index=cyclic['index'],cyclic_variations=variations)
            if 'uncertainty' in row:
                require('cyclic_point_index' in replay,"uncertainty lacks cyclic data")
                # Both correctors were independently reconstructed above. Use
                # their validated saved floating-point matrices for subsequent
                # cancellation diagnostics, so last-bit differences between
                # algebraically equivalent eigenvector formulas do not become
                # invented changes in the measured half-step discrepancies.
                uncertainty=uncertainty_from_jets(replay['coarse_jet'],replay['fine_jet'],replay['coarse_geometry'],replay['fine_geometry'],
                    row['coarse_correction'],row['fine_correction'],prior_variations,replay['cyclic_variations'],cfg)
                compare_numeric_tree(row['uncertainty'],uncertainty,'target uncertainty')
                replay['uncertainty']=uncertainty
            if 'admission' in row:
                require('uncertainty' in replay,"admission lacks uncertainty")
                admission=admission_from_values(replay['coarse_geometry'],replay['coarse_correction'],replay['uncertainty'],cyclic,cfg)
                compare_numeric_tree(row['admission'],admission,'target admission')
                replay['admission']=admission
            if 'step_applied' in row:
                require('coarse_correction' in replay and count<cfg['maximum_corrections']
                        and not replay.get('admission',{}).get('passed'),"correction applied without predictor or after admission")
                if 'admission' in replay:
                    correctable={'actual_cyclic_trace','actual_primary_trace','f_plus_residual',
                                 'coordinate_correction','projection_displacement'}
                    require(not any(not passed and gate not in correctable
                                    for gate,passed in replay['admission']['checks'].items()),
                            'fresh correction continued after a failed precision gate')
                step=np.asarray(replay['coarse_correction']['correction']);next_center=center+step
                norm,total=float(np.linalg.norm(step)),float(np.linalg.norm(next_center-origin))
                require(norm<=cfg['maximum_step'] and total<=cfg['maximum_total_displacement'],"correction exceeds declared trust region")
                applied={'old_center':center.tolist(),'new_center':next_center.tolist(),'displacement':step.tolist(),
                         'norm':norm,'cumulative_displacement':total}
                compare_numeric_tree(row['step_applied'],applied,'applied correction')
                replay['step_applied']=applied
                center=next_center
                iteration_references={side:replay['coarse_geometry']['branches'][side]['valley']['normal'] for side in SIDES}
            elif count<len(iterations)-1:
                require(False,"next fresh iteration lacks applied correction")
            checked.append(replay)
        if case['complete']:
            require(case['status']=='admitted' and checked and checked[-1].get('admission',{}).get('passed') is True
                    and 'step_applied' not in checked[-1],"completed case lacks passed terminal admission")
            last=checked[-1]
            require(case['final_point_index']==last['cyclic_point_index'],"case endpoint differs")
            close(case['final_center'],last['center'],'case final coordinate differs',rtol=0,atol=1e-13)
            compare_numeric_tree(case['f_minus'],last['admission']['f_minus'],'case f_minus')
            final_references={side:last['coarse_geometry']['branches'][side]['valley']['normal'] for side in SIDES}
            compare_numeric_tree(case['final_references'],final_references,'case final references')
            if offset==0:references=final_references
        else:
            require(case_index==len(cases)-1 and case['status'] in ('running','unresolved','unresolved precision','correction budget exhausted'),
                    "unfinished case has later target or unknown outcome")
            if case['status']=='unresolved':require('error_type' in case and 'error' in case,'unresolved case lacks failure evidence')
            if case['status']=='unresolved precision':
                require(checked and 'admission' in checked[-1],"precision failure lacks measured admission")
                failed=[k for k,v in checked[-1]['admission']['checks'].items() if not v and k not in
                        ('actual_cyclic_trace','actual_primary_trace','f_plus_residual','coordinate_correction','projection_displacement')]
                require(failed and case['failed_checks']==failed,"unresolved precision labels differ")
            if case['status']=='correction budget exhausted':
                require(len(checked)==cfg['maximum_corrections']+1 and not checked[-1].get('admission',{}).get('passed'),
                        "correction-budget outcome lacks exhausted unadmitted target")
        outputs.append({**case,'iterations':checked})
    if result.get('complete'):
        require(len(outputs)==len(cfg['beta_offsets']) and all(c['complete'] for c in outputs),"complete pilot lacks every admitted target")
        signs={outputs[1]['f_minus']['status'],outputs[2]['f_minus']['status']}
        require(result['resolved_f_minus_sign_bracket'] is (signs=={'resolved_positive','resolved_negative'}),"f_minus sign-bracket label differs")
    return outputs


EXECUTION_SOURCES = {
    'tools/baby_shrimp_mixed_junction_probe.py','tools/contraction_valley_geometry.py','tools/baby_shrimp_contraction_minima_finish.py',
    'tools/baby_shrimp_memory_connection_probe.py','tools/baby_shrimp_saddle_memory.py','tools/baby_shrimp_memory.py',
    'tools/baby_shrimp_crossing.py','tools/baby_shrimp_memory_orbit_alignment.py','tools/baby_shrimp_memory_finish.py',
    'tools/baby_shrimp_memory_resume.py','tools/baby_shrimp_orbits.py','tools/baby_shrimp_memory_determinant_gradient.py',
    'tools/baby_shrimp_geometry.py','tools/doubly_superstable.py','tools/explore_llm_soft_loop.py',
    'tools/explore_llm_attractors.py','tools/explore_llm_phase_map.py','tools/explore_latent_loop.py','tools/llm_feigenbaum.py'}
SOURCES=['tools/baby_shrimp_mixed_junction_probe_finish.py','tools/baby_shrimp_contraction_minima_finish.py',
         'tools/baby_shrimp_memory_finish.py','tools/baby_shrimp_memory_orbit_alignment.py']


def bind_bsc5(plan):
    import hashlib
    import re
    from pathlib import Path
    bound=plan['bsc5_audit'];cfg=plan['mixed_junction'];path=(DATA/bound['path']).resolve()
    require(path.is_relative_to(DATA.resolve()) and path.stat().st_size==bound['bytes'] and digest(path)==bound['sha256'],
            'prior BSC5 receipt binding differs')
    receipt=read(path)
    require(receipt['id']=='BSC5-contraction-minima-offline-audit' and receipt.get('audit_finished') is True
            and type(receipt['complete']) is bool and receipt['center_qualified'] is True,'qualified audited BSC5 center required')
    require(receipt['complete'] or cfg['allow_scientifically_partial_bsc5'],'partial predecessor not authorized by frozen plan')
    require(isinstance(receipt.get('analysis_commit'),str) and re.fullmatch(r'[0-9a-f]{40}',receipt['analysis_commit']),
            'exact prior analysis commit required')
    require(all(plan[k]==receipt['scientific_metadata'][k] for k in previous.binding.METADATA if k!='id'),
            'BSC6 changed bound scientific map or coordinates')
    require(set(previous.SOURCES)<=set(receipt['analysis_source_sha256']),'prior analysis source set incomplete')
    for source,sha in receipt['analysis_source_sha256'].items():
        require(hashlib.sha256(point_audit.git_bytes(receipt['analysis_commit'],source)).hexdigest()==sha
                and digest(ROOT/source)==sha,'exact prior auditor source unavailable: '+source)
    controller=receipt['lifecycle']['controller'];kwargs={}
    if controller:
        require(controller['worker_reaped'] is True,'prior worker is not reaped')
        kwargs={'controller_manifest':DATA/controller['manifest']['path'],'manifest_sha':controller['manifest']['sha256'],
                'controller_status':DATA/controller['status_artifact']['path'],'status_sha':controller['status_artifact']['sha256']}
    else:require(receipt['lifecycle']['python_finalization'] is True,'prior terminal lifecycle absent')
    replay=previous.run_audit(Path(receipt['raw_directory']),allow_partial=cfg['allow_scientifically_partial_bsc5'],**kwargs)
    ignore={'generated_utc','analysis_commit','analysis_source_sha256','threads'}
    require({k:v for k,v in receipt.items() if k not in ignore}=={k:v for k,v in replay.items() if k not in ignore},
            'full independent BSC5 replay differs from prior receipt')
    require(receipt['baseline'].get('passed') is True and receipt['seed_replay'].get('passed') is True
            and receipt['expanded_spectrum']['complete'] is True and len(receipt['local_jets'])>=2
            and receipt['expanded_spectrum']['point_index']==receipt['center_point_index'],'prior supporting setup incomplete')
    statuses=[{'arm_id':s['arm_id'],'longitudinal':s['longitudinal'],'status':s['search']['status'],
               'complete':s['search']['complete']} for s in receipt['sections']]
    if not receipt['complete']:
        review=cfg['partial_bsc5_review']
        require(review['receipt_sha256']==bound['sha256'] and review['retained_unresolved_stages']==receipt['untested_or_incomplete_stages']
                and review['section_statuses']==statuses,'partial predecessor outcomes lack exact frozen disclosure')
    run=Path(receipt['raw_directory']);index=receipt['center_point_index'];point=read(run/f'point-{index:04d}'/'point.json')
    require(point['qualified'] is True and [r['phase'] for r in point['phases']]==[0,6,12],'prior center lacks cyclic state')
    provenance={'audit':bound,'scientific_complete':receipt['complete'],'retained_unresolved_stages':receipt['untested_or_incomplete_stages'],
                'section_statuses':statuses,'expanded_spectrum':receipt['expanded_spectrum'],
                'supporting_audit_blas_threads':1}
    return {'receipt':receipt,'run':run,'seed':{'center':point['coordinate'],'parameter':point['beta'],'cyclic_point_index':index},
            'provenance':provenance}


def audit_point_admissions(run,records,raw,summaries,reference,cfg):
    by_index={row['index']:row for row in summaries}
    reference_orbit=np.load(run/f"point-{reference['index']:04d}"/'points.npy',allow_pickle=False)
    require([r['phase'] for r in reference['phases']]==[0,6,12],'state-alignment reference lacks cyclic seed')
    checked=[];seen=set();admitted=set()
    for row in records:
        index=row['point_index']
        require(type(index) is int and index not in seen and row['reference_point_index']==reference['index']
                and row['status'] in ('started','passed','failed','failed orbit alignment'),'invalid state-admission record identity/status')
        seen.add(index)
        require(index in raw,'state admission lacks qualified raw point')
        expected={'point_index':index,'reference_point_index':reference['index'],'status':row['status']}
        if 'audit' in row:
            require(row['audit']==by_index[index],'embedded raw-point audit differs')
            orbit=np.load(run/f'point-{index:04d}'/'points.npy',allow_pickle=False)
            distances=[float(np.max(np.linalg.norm(np.roll(orbit,shift,axis=0)-reference_orbit,axis=1)/np.linalg.norm(reference_orbit,axis=1))) for shift in range(18)]
            best=int(np.argmin(distances))
            close(row['all_alignment_maximum_relative_changes'],distances,'full-state alignment distances differ',rtol=1e-10,atol=1e-15)
            require(row['best_right_roll']==best,'state-alignment minimizing phase differs')
            passed=bool(by_index[index]['closure_recomputed_from_saved_endpoint'] and best==0 and distances[0]<cfg['orbit_alignment_tolerance'])
            require(row['passed'] is passed,'state-alignment gate differs')
            expected.update(audit=by_index[index],all_alignment_maximum_relative_changes=distances,best_right_roll=best,passed=passed)
            if row['status']=='passed':
                require(passed and 'error' not in row,'passed state admission contains failure')
                admitted.add(index)
        else:require(row['status'] not in ('passed','failed orbit alignment'),'resolved state admission lacks saved audit')
        if row['status']=='failed orbit alignment':
            require(not expected.get('passed'),'alignment-failed status contains a passing comparison')
        if row['status']=='failed':
            require('error_type' in row and 'error' in row,'failed admission lacks error evidence')
            expected.update(error_type=row['error_type'],error=row['error'])
        checked.append(expected)
    return checked,admitted


def terminal_evidence(run,protocol,controller_manifest=None,manifest_sha=None,controller_status=None,status_sha=None):
    from pathlib import Path
    supplied=(controller_manifest,manifest_sha,controller_status,status_sha)
    require(not any(supplied) or all(supplied),'controller status/manifest need explicit paired hashes')
    controller=None
    result=read(run/'result.json') if (run/'result.json').exists() else None
    if all(supplied):
        paths=[Path(p).resolve() for p in (controller_manifest,controller_status)]
        require(all(p.is_file() and p.is_relative_to(DATA.resolve()) for p in paths)
                and digest(paths[0])==manifest_sha and digest(paths[1])==status_sha,'controller packet location/hash differs')
        status=read(paths[1]);finished=bool(result and result.get('execution_finished') is True)
        scientific=bool(result and result.get('complete') is True)
        require(status['worker_reaped'] is True and status['source_commit']==protocol['commit']
                and Path(status['output']).resolve()==run and type(status['exit_code']) is int
                and type(status['timed_out']) is bool and type(status['complete']) is bool
                and status['complete']==(status['exit_code']==0 and finished)
                and type(status.get('scientific_complete')) is bool and status['scientific_complete']==scientific
                and not(status['complete'] and status['timed_out']) and 'ended_utc' in status,
                'controller lacks consistent reaped terminal execution')
        require(status['hard_seconds']==protocol['args']['seconds']<=5400 and 0<=status['termination_grace_seconds']<=10,
                'controller lifetime differs')
        command=status['command']
        for flag,value in (('--source-commit',protocol['commit']),('--out',str(run)),('--plan',protocol['args']['plan']),
                           ('--device',protocol['args']['device']),('--seconds',str(protocol['args']['seconds']))):
            require(command.count(flag)==1 and command.index(flag)+1<len(command) and command[command.index(flag)+1]==value,
                    'controller command differs: '+flag)
        require('tools/baby_shrimp_mixed_junction_probe.py' in command,'controller ran another worker')
        previous.verify_snapshot(run,read(paths[0]))
        controller={'manifest':artifact(paths[0]),'status_artifact':artifact(paths[1]),
                    **{k:status[k] for k in ('worker_pid','controller_pid','worker_reaped','exit_code','timed_out','elapsed_seconds','ended_utc')},
                    'execution_complete':status['complete'],'scientific_complete':status['scientific_complete']}
    path=run/'manifest.json';manifest=read(path) if path.exists() else None
    if manifest is not None:
        require(type(manifest.get('complete')) is bool,'Python manifest completion missing')
        previous.verify_snapshot(run,manifest['files'],omit_python_manifest=True)
        require(result is not None or (run/'failure.json').exists(),'Python finalization lacks outcome')
        require(manifest['complete']==(result or {'complete':False})['complete'],'Python manifest/result completion differs')
        require(not manifest['complete'] or not (run/'failure.json').exists(),'complete Python result contains failure')
    else:require(controller is not None,'missing Python finalization needs a hash-bound terminal snapshot')
    return {'python_finalization':manifest is not None,'python_manifest':artifact(path) if manifest else None,
            'python_complete':bool(manifest and manifest['complete']),'controller':controller}


def run_audit(run,*,allow_partial=False,controller_manifest=None,manifest_sha=None,controller_status=None,status_sha=None):
    from pathlib import Path
    from datetime import datetime,timezone
    run=Path(run).resolve()
    require(run.is_relative_to((DATA/'explore').resolve()),'BSC6 raw attempt must remain outside Git')
    protocol=read(run/'protocol.json')
    anchor=previous.bind_execution(protocol,expected_id='BSC6-mixed-junction-probe',required_sources=EXECUTION_SOURCES)
    plan,cfg=protocol['plan'],protocol['plan']['mixed_junction']
    require(type(plan['maximum_evaluations']) is int and 1<=plan['maximum_evaluations']<=120
            and 90<protocol['args']['seconds']<=5400 and protocol['args']['device'] in ('cpu','cuda')
            and Path(protocol['args']['out']).resolve()==run and protocol['args'].get('source_commit') in (None,protocol['commit']),
            'BSC6 resource or execution arguments differ')
    require(cfg['beta_offsets']==[0.,-3e-9,3e-9] and cfg['stencil_steps']==[.0003,.0001] and cfg['primary_step']==.0003
            and cfg['maximum_corrections']==2 and cfg['trace_tolerance']<=1e-12 and cfg['f_plus_tolerance']<=1e-8
            and cfg['coordinate_tolerance']<=2e-6 and cfg['maximum_intersection_location_uncertainty']<=5e-6
            and cfg['maximum_f_minus_total_uncertainty']<=5e-8 and cfg['signal_multiple']>=10
            and cfg['minimum_normalized_jacobian_singular_value']>=.2,'BSC6 declared design or admission cap differs')
    terminal_args=(controller_manifest,manifest_sha,controller_status,status_sha)
    lifecycle=terminal_evidence(run,protocol,*terminal_args)
    path=run/'result.json' if (run/'result.json').exists() else run/'probe-progress.json'
    result=read(path) if path.exists() else {'complete':False,'execution_finished':False}
    require(type(result.get('complete')) is bool and type(result.get('execution_finished')) is bool,'result completion flags missing')
    points=previous.audit_saved_points(run,plan);raw=points.pop('raw')
    require(points['runtime'] is not None or not result['complete'],'complete run lacks precision/runtime metadata')
    baseline=previous.audit_baseline(run,raw,anchor) if (run/'baseline.json').exists() else {'status':'untested'}
    if result.get('baseline') is not None:require(result['baseline']==read(run/'baseline.json'),'embedded baseline differs')
    prior=bind_bsc5(plan)
    if 'prior_bsc5' in protocol:compare_numeric_tree(protocol['prior_bsc5'],prior['provenance'],'protocol prior provenance')
    if result.get('prior_support') is not None:compare_numeric_tree(result['prior_support'],prior['provenance'],'result prior provenance')
    replay=previous.audit_seed(run,result.get('seed_replay'),raw,prior,prior['seed'])
    require(replay is None or baseline.get('passed') is True and result.get('prior_support') is not None
            and 'prior_bsc5' in protocol,'fresh seed lacks baseline/prior admission')
    admissions=[];admitted=set()
    if result.get('point_admissions'):
        require(replay is not None,'state admissions lack fresh seed')
        admissions,admitted=audit_point_admissions(run,result['point_admissions'],raw,points['qualified_points'],
                                                   raw[replay['fresh_point_index']],cfg)
    require(not result.get('cases') or replay is not None,'target cases lack baseline and seed admission')
    cases=audit_cases(result,raw,admitted,prior['receipt']['expanded_spectrum']['variations'],cfg)
    failure=read(run/'failure.json') if (run/'failure.json').exists() else None
    scientific=result['complete']
    if result['execution_finished'] or scientific:
        require(result['execution_finished'] and scientific and len(cases)==3 and all(c['complete'] for c in cases)
                and result['evaluations']==points['attempted_points'] and failure is None
                and not points['unqualified_or_incomplete_points'] and all(r['status']=='passed' for r in admissions),
                'complete BSC6 execution contains missing or failed scientific evidence')
    complete=bool(scientific and lifecycle['python_finalization'] and lifecycle['python_complete'])
    if lifecycle['controller']:
        complete=complete and lifecycle['controller']['execution_complete'] and not lifecycle['controller']['timed_out']
    require(complete or allow_partial,'incomplete BSC6 target probe requires explicit --allow-partial')
    missing=[]
    for name,done in (('dense baseline',baseline.get('passed') is True),('fresh BSC5 center replay',replay is not None),
                      ('central intersection',len(cases)>0 and cases[0]['complete']),
                      ('lower-beta intersection',len(cases)>1 and cases[1]['complete']),
                      ('upper-beta intersection',len(cases)>2 and cases[2]['complete'])):
        if not done:missing.append(name)
    require(terminal_evidence(run,protocol,*terminal_args)==lifecycle,'terminal raw evidence changed during audit')
    return {'id':'BSC6-mixed-junction-offline-audit','generated_utc':datetime.now(timezone.utc).isoformat(),
        'audit_finished':True,'complete':complete,'execution_finished':result['execution_finished'],
        'status':'all saved-data gates passed' if complete else 'partial or unresolved attempt; available saved measurements audited',
        'execution_source_commit':protocol['commit'],'execution_source_sha256':protocol['source_sha256'],
        'plan_path':protocol['args']['plan'],'plan_sha256':protocol['plan_sha256'],
        'scientific_metadata':{k:plan[k] for k in previous.binding.METADATA},'raw_directory':str(run),
        'lifecycle':lifecycle,**points,'baseline':baseline,'bsc5_audit':plan['bsc5_audit'],'prior_support':prior['provenance'],
        'seed_replay':replay,'point_admissions':admissions,'cases':cases,
        'resolved_f_minus_sign_bracket':result.get('resolved_f_minus_sign_bracket') if scientific else None,
        'untested_or_incomplete_stages':missing,'terminal_failure':failure,
        'manifest':[artifact(p,run) for p in sorted(run.rglob('*')) if p.is_file()],
        'limits':['Saved-array replay, not a fresh model run, full-spectrum proof or interval certificate.',
            'Expanded 12-column support comes from the audited BSC5 center; BSC6 target spectra are fresh six-column cyclic measurements.',
            'Conditional tau-value projections keep coordinates unchanged. Actual measured tau and predicted displacement remain explicit; neither is an exact trace-zero certificate.',
            'The two one-sided extension jets are separate. There is no common radius Hessian across the cusp.',
            'Jet-error balls, half-step Jacobian discrepancies and first-order location transfer are empirical model assumptions, not verified neural error bounds.',
            'Admitted tau/f_plus intersections and observed f_minus signs do not certify a common zero, intrinsic ridge connectivity or exact superstability.',
            'Failed precision/admission gates are retained; absent later targets are untested, not evidence against a possible junction.']}


def main():
    import argparse,hashlib,json,subprocess
    from pathlib import Path
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--allow-partial',action='store_true')
    parser.add_argument('--controller-manifest',type=Path);parser.add_argument('--controller-manifest-sha256')
    parser.add_argument('--controller-status',type=Path);parser.add_argument('--controller-status-sha256')
    args=parser.parse_args();run,out=args.run.resolve(),args.out.resolve()
    require(out.is_relative_to(DATA.resolve()) and not out.exists() and not out.is_relative_to(run)
            and not run.is_relative_to(out),'fresh external audit directory disjoint from raw required')
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    hashes={source:digest(ROOT/source) for source in SOURCES}
    require(all(hashlib.sha256(point_audit.git_bytes(commit,path)).hexdigest()==sha for path,sha in hashes.items()),
            'commit auditor sources before execution')
    out.mkdir(parents=True)
    try:
        receipt=run_audit(run,allow_partial=args.allow_partial,controller_manifest=args.controller_manifest,
            manifest_sha=args.controller_manifest_sha256,controller_status=args.controller_status,status_sha=args.controller_status_sha256)
        receipt.update(analysis_commit=commit,analysis_source_sha256=hashes,threads=1)
        path=out/'receipt.json';path.write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n')
        print(json.dumps({'out':str(path),'sha256':digest(path),'complete':receipt['complete'],'qualified_points':len(receipt['qualified_points'])}))
    except BaseException as error:
        (out/'failed.json').write_text(json.dumps({'type':type(error).__name__,'error':str(error)},indent=2)+'\n')
        raise


if __name__=='__main__':main()
