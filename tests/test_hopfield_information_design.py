"""Synthetic EXP008 design checks. No neural trajectories or new outcomes."""
import copy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import hopfield_drift_design as old
import hopfield_holdout_design as holdout
import hopfield_information_design as design


def atlas():
    alpha = np.linspace(-.05, .05, 41)
    gamma = np.linspace(-.5, .5, 31)
    period = np.zeros((31, 41), dtype=int)
    period[4:27, 8:30] = 9
    period[1:3, 1:4] = 9
    grid = {"alpha": np.tile(alpha, (31, 1)), "gamma": gamma, "period": period,
            "states": np.zeros((31, 41, 3))}
    exp006 = old.select_anchors(grid, (15, 20))["anchors"]
    exp007 = holdout.select_holdout_anchors(grid, (15, 20), exp006)["anchors"]
    return grid, exp006, exp007


def square():
    q = np.zeros((11, 11), dtype=int)
    q[2:9, 2:9] = 9
    grid = {"alpha": np.arange(11.), "gamma": np.arange(11.), "period": q}
    anchor = {"row": 5, "column": 5, "alpha": 5., "gamma": 5.}
    return grid, anchor


def test_blocks_partition_every_eligible_pixel_and_represent_their_centroids():
    grid, e6, e7 = atlas()
    p = design.select_information_anchors(grid, (15, 20), e6+e7)
    assert p == design.select_information_anchors(grid, (15, 20), e6+e7)
    assert len(p["anchors"]) == len(p["blocks"]) == 12
    assert len(p["partition_splits"]) == 11
    assert len({a["id"] for a in p["anchors"]}) == 12
    assert p["phase_indices"] == [675, 1350] and p["reference_samples"] == 2700
    context = design.build_atlas_geometry(grid, (15, 20))
    widths = np.asarray(p["normalization_widths"])
    origin = np.asarray(p["normalization_origin"])
    prior = np.array([[a["alpha"], a["gamma"]] for a in e6+e7])
    expected = set()
    for row, column in np.argwhere(context.component):
        point = np.array([context.alpha[column], context.gamma[row]])
        edge = min(point[0]-context.alpha[0], context.alpha[-1]-point[0],
                   point[1]-context.gamma[0], context.gamma[-1]-point[1])
        if context.signed_margin[row, column] < edge and np.linalg.norm((prior-point)/widths,axis=1).min() > .04:
            expected.add((int(row), int(column)))
    all_members = []
    for block, anchor in zip(p["blocks"], p["anchors"]):
        members = [tuple(m) for m in block["members"]]
        assert block["member_count"] == len(members) > 0
        all_members.extend(members)
        pts = np.array([[context.alpha[c], context.gamma[r]] for r,c in members])
        normalized = (pts-origin)/widths
        np.testing.assert_allclose(block["normalized_centroid"],normalized.mean(axis=0),rtol=0,atol=1e-15)
        scores = np.sum((normalized-normalized.mean(axis=0))**2,axis=1)
        selected = min(range(len(members)),key=lambda i:(scores[i],*members[i]))
        assert members[selected] == (anchor["row"],anchor["column"])
        assert anchor["block_id"] == block["id"] and anchor["id"] == block["anchor_id"]
        point = np.array([anchor["alpha"],anchor["gamma"]])
        assert np.linalg.norm((prior-point)/widths,axis=1).min() > .04
        assert anchor["raw_margin"] < anchor["crop_edge_lower_bound"]
        # The split path is a complete inspectable membership rule.
        for split in p["partition_splits"]:
            if block["split_path"].startswith(split["path"]) and len(block["split_path"]) > len(split["path"]):
                axis = 0 if split["axis"] == "normalized_alpha" else 1
                side = block["split_path"][len(split["path"])]
                assert np.all(normalized[:,axis] <= split["threshold"]) if side == "L" else np.all(normalized[:,axis] > split["threshold"])
    assert len(all_members) == len(set(all_members)) == len(expected)
    assert set(all_members) == expected


def test_selection_and_features_ignore_outcomes_and_leave_old_atlas_unchanged():
    grid, e6, e7 = atlas()
    before = {k:v.copy() for k,v in grid.items()}
    prior = e6+e7
    result = design.select_information_anchors(grid, (15, 20), prior)
    contaminated = copy.deepcopy(prior)
    for i,a in enumerate(contaminated):
        a.update(eligible=False, tail_metrics={"rms":i*1000}, target_rank=-i)
    assert result == design.select_information_anchors(grid, (15, 20), contaminated)
    context = design.build_atlas_geometry(grid,(15,20))
    anchor = result["anchors"][0]
    expected = design.geometry_features(context,anchor,[1.,0.],.001)
    assert expected == design.geometry_features(context,{**anchor,"target_error":-999},[1.,0.],.001)
    for k in grid:np.testing.assert_array_equal(grid[k],before[k])


def test_closed_halo_exclusion_and_no_replacements():
    grid, _ = square()
    prior=[{"alpha":5.,"gamma":5.}]
    packet=design.select_information_anchors(grid,(5,5),prior,block_count=4,exclusion_radius=.2)
    forbidden={(5,3),(5,7),(3,5),(7,5),(5,5)}
    assert not forbidden.intersection(tuple(m) for b in packet["blocks"] for m in b["members"])
    with pytest.raises(ValueError,match="no fallback"):
        design.select_information_anchors(grid,(5,5),prior,exclusion_radius=10.)
    with pytest.raises(ValueError,match="no fallback"):
        design.select_information_anchors(grid,(5,5),prior,block_count=1000)
    with pytest.raises(ValueError,match="unique"):
        design.select_information_anchors(grid,(5,5),prior*2)
    with pytest.raises(ValueError,match="Finite prior"):
        design.select_information_anchors(grid,(5,5),[])


def test_raw_signed_pixel_margin_component_membership_and_deterministic_normal():
    grid,a=square();context=design.build_atlas_geometry(grid,(5,5))
    # Nearest opposite-class pixels tie; row/column order chooses (1,5).
    near=design.geometry_features(context,a,[0.,-1.],3.)
    assert near['feature_names']==list(design.GEOMETRY_FEATURE_NAMES)
    assert near['values']==[4.,1.,1.,1.,1.,0.]
    assert near['diagnostics']['anchor_nearest_non_core_pixel']==[1,5]
    outside=design.geometry_features(context,a,[0.,-1.],4.)
    assert outside['values']==[4.,-1.,0.,1.,1.,1.]
    inward=design.geometry_features(context,a,[0.,1.],3.)
    assert inward['values'][3]==-1.
    # Normalization is absent from the physical margin: alpha spacing doubled
    # makes the nearest raw boundary along gamma, not a dimensionless norm.
    grid['alpha']*=2;a['alpha']*=2
    scaled=design.build_atlas_geometry(grid,(5,5))
    assert design.geometry_features(scaled,a,[0.,-1.],3.)['values']==near['values']


def test_lower_pixel_ties_and_outside_crop_are_explicitly_missing_not_known_negative():
    grid,a=square();context=design.build_atlas_geometry(grid,(5,5))
    tie=design.geometry_features(context,a,[1.,0.],3.5)
    assert tie['diagnostics']['nearest_pixel_column']==8
    assert tie['values'][1:3]==[1.,1.]
    missing=design.geometry_features(context,a,[1.,0.],5.1)
    assert missing['values'][1:3]==[0.,0.]
    assert missing['values'][4:]==[0.,1.]
    assert missing['diagnostics']['destination_signed_pixel_margin'] is None
    assert missing['diagnostics']['destination_in_core_component'] is None
    assert missing['diagnostics']['destination_crop_edge_lower_bound'] < 0


def test_disconnected_nine_peak_pixels_are_not_the_core_component():
    grid,a=square();grid['period'][0,5]=9
    context=design.build_atlas_geometry(grid,(5,5))
    f=design.geometry_features(context,a,[0.,-1.],5.)
    assert grid['period'][0,5]==9 and not context.component[0,5]
    assert f['values'][2]==0 and f['values'][1]<0


def test_invalid_geometry_inputs_fail_and_cached_arrays_are_read_only():
    grid,a=square();context=design.build_atlas_geometry(grid,(5,5))
    for direction in ([2.,0.],[float('nan'),0.],[1.,0.,0.]):
        with pytest.raises(ValueError,match="native-unit"):
            design.geometry_features(context,a,direction,1.)
    for radius in (0.,-1.,float('nan'),True):
        with pytest.raises(ValueError,match="positive|Positive"):
            design.geometry_features(context,a,[1.,0.],radius)
    with pytest.raises(ValueError,match="coordinates"):
        design.geometry_features(context,{**a,'alpha':5.1},[1.,0.],1.)
    with pytest.raises(ValueError):context.period[5,5]=0
    with pytest.raises(ValueError,match="nine-peak"):
        design.build_atlas_geometry(grid,(True,5))


def test_packet_hashes_old_inputs_but_exposes_only_origin_data(tmp_path):
    grid,e6,e7=atlas();directory=tmp_path/'old';(directory/'focus').mkdir(parents=True)
    files=[]
    for i in range(len(grid['gamma'])):
        p=directory/'focus'/f'row-{i:03d}.npz'
        np.savez(p,alpha=grid['alpha'][i],gamma=grid['gamma'][i],period=grid['period'][i],states=grid['states'][i]);files.append(p)
    (directory/'manifest.json').write_text(json.dumps({'experiment_id':'EXP-004','grids':[
        {'name':'focus','alpha':[-.05,.05,41],'gamma':[-.5,.5,31]}]}))
    (directory/'selected.json').write_text(json.dumps({'core':{'row':15,'column':20}}))
    files += [directory/'manifest.json',directory/'selected.json']
    (directory/'receipt.json').write_text(json.dumps({'status':'complete','source_commit':'a'*40,
        'artifacts':[{'path':str(p.relative_to(directory)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),
                      'bytes':p.stat().st_size} for p in files]}))
    m6=tmp_path/'six.json';m7=tmp_path/'seven.json'
    m6.write_text(json.dumps({'experiment_id':'EXP-006','anchors':e6}))
    m7.write_text(json.dumps({'experiment_id':'EXP-007','anchors':[{**a,'forbidden_outcome':999} for a in e7]}))
    packet=design.load_information_packet(directory,m6,m7)
    assert len(packet['development_anchors'])==len(packet['anchors'])==12
    assert not {a['id'] for a in packet['development_anchors']} & {a['id'] for a in packet['anchors']}
    assert all('forbidden_outcome' not in a for a in packet['development_anchors'])
    assert len(packet['source_files'])==len(files)+1
    assert packet['prior_manifests'][1]['sha256']==hashlib.sha256(m7.read_bytes()).hexdigest()
    assert packet['source_freeze_commit']=='a'*40
    files[0].write_bytes(b'altered')
    with pytest.raises(ValueError,match='receipt mismatch'):
        design.load_information_packet(directory,m6,m7)
