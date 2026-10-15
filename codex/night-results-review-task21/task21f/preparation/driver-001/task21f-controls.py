"""Manufactured-only focused controls and finite serialization measurements."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import time
import traceback


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--core",required=True)
    ap.add_argument("--out",required=True)
    args=ap.parse_args()
    resource.setrlimit(resource.RLIMIT_AS,(1024**3,1024**3))
    resource.setrlimit(resource.RLIMIT_CPU,(25,25))
    nice=os.getpriority(os.PRIO_PROCESS,0)
    if nice > 10:
        raise RuntimeError("cannot meet exact nice10")
    if nice < 10:
        os.nice(10-nice)
    out=Path(args.out); out.mkdir(exist_ok=False)
    begin=time.perf_counter()
    spec=importlib.util.spec_from_file_location("own_task21f",args.core)
    c=importlib.util.module_from_spec(spec); spec.loader.exec_module(c)
    np=c.np
    rows=[]

    def check(name,fn):
        t=time.perf_counter()
        try:
            fn(); rows.append({"name":name,"passed":True,"seconds":time.perf_counter()-t})
        except BaseException:
            rows.append({"name":name,"passed":False,"traceback":traceback.format_exc(),"seconds":time.perf_counter()-t})

    def require(v):
        if not v:
            raise AssertionError("condition failed")

    def refusal(fn):
        try:
            fn()
        except (ValueError,KeyError):
            return
        raise AssertionError("expected source-schema refusal")

    def grid(n=2,m=2,ys=None):
        shape=(n,m); phase=7
        ct=np.ones(shape+(phase,)); ct[...,0]=-.5; ct[...,1]=2
        g={k:np.ones(shape) for k in c.SCALARS}
        g.update(alpha=np.linspace(-.1,.1,m),log10_temperature=np.linspace(.1,.2,n) if ys is None else np.asarray(ys,dtype=float),
                 mu=-np.ones(shape),mu_imag=np.zeros(shape),next_modulus=np.full(shape,.001),
                 newton=np.full(shape,1e-14),product=-np.ones(shape),factors=ct.copy(),
                 ok=np.ones(shape,dtype=bool),clean=np.ones(shape,dtype=bool),same_tokens=np.ones(shape,dtype=bool),
                 factors_transported=ct.copy(),stretchings=np.abs(ct),transport_cosine=np.ones(shape))
        return g

    def identity():
        result=c.cell_rows(grid(),7)
        require(len(result)==4)
        for row in result:
            require(row["transport_product_vs_mu_real"]["signed"] == 0)
            require((row["phase_absolute_factor_vs_stretching"]["signed"] == 0).all())
            require(row["absolute_product_vs_stretching_product"]["signed"] == 0)
        g=grid(); g["factors_transported"][0,0,0]=.5
        r=c.cell_rows(g,7)[0]
        require(r["transport_product_vs_mu_real"]["signed"] == 2)
        require((r["phase_absolute_factor_vs_stretching"]["signed"] == 0).all())
    check("saved scalar products, phase absolute identities and sign discrepancy",identity)

    def separate_masks():
        g=grid(); g["mu_imag"][0,0]=1e-9; g["mu_imag"][0,1]=np.nan; g["transport_cosine"][1,0]=.3
        m=c.masks(g)
        require(m["seed_real_threshold_finite"][0,0] and not m["plot_real_threshold_finite"][0,0])
        require(m["plot_zoom_not_complex"][0,1] and not m["finite_spectral"][0,1])
        require(not m["transport_seed_real_carry_above_point3"][1,0])
        require(not m["exact_real_finite"][0,0])
    check("literal NaN zoom and distinct source reality/carry thresholds",separate_masks)

    def schema():
        g=grid(); del g["stretchings"]; refusal(lambda:c.validate_grid(g,7))
        g=grid(); g["alpha"][1]=g["alpha"][0]; refusal(lambda:c.validate_grid(g,7))
        g=grid(); g["ok"]=np.ones((2,2)); refusal(lambda:c.validate_grid(g,7))
        g=grid(); g["factors"]=np.ones((2,2,6)); refusal(lambda:c.validate_grid(g,7))
    check("duplicate axis, dtype, partial bundle and phase count refusals",schema)

    def simple_contour():
        r=c.contours([[-1,1],[-1,1]],np.ones((2,2),bool),[0.,1.],[0.,1.])
        require(len(r["segments"])==1)
        require(np.array_equal(r["segments"][0]["points"],[[.5,0],[.5,1]]))
        r=c.contours([[1,-1],[-1,1]],np.ones((2,2),bool),[0.,1.],[0.,1.])
        require(r["literal_checkerboards"]==1 and r["quads"][0]["cut_pairs"]==[(0,3),(1,2)])
        require(np.array_equal(r["segments"][0]["points"],[[.5,0],[0,.5]]))
    check("ordered edges and source mean-zero four-cut tie",simple_contour)

    def zeros_nonfinite():
        m=np.ones((2,2),bool)
        r=c.contours(np.zeros((2,2)),m,[0.,1.],[0.,1.])
        require(len(r["segments"])==2 and len(r["quads"][0]["zero_corner_indices"])==4)
        r=c.contours([[0,1],[1,1]],m,[0.,1.],[0.,1.])
        require(len(r["segments"])==0 and len(r["quads"][0]["cuts"])==1)
        r=c.contours([[0,np.nan],[1,1]],m,[0.,1.],[0.,1.])
        require(not r["quads"][0]["finite"] and len(r["quads"][0]["cuts"])==1)
        require(not np.isfinite(r["quads"][0]["cuts"][0]["point"]).all())
        r=c.contours([[1,-1],[-1,1]],[[1,1],[0,1]],[0.,1.],[0.,1.])
        require(not r["quads"][0]["eligible"] and not r["segments"])
    check("zero/NaN/missing-corner cases preserved",zeros_nonfinite)

    def underflow():
        r=c.contours([[1e-200,-1e-200],[-1e-200,1e-200]],np.ones((2,2),bool),[0.,1.],[0.,1.])
        require(r["literal_checkerboards"]==1 and not r["segments"])
        require(r["quads"][0]["opposite_sign_underflow_edges"]==[0,1,2,3])
    check("literal product underflow differs from sign-based contour",underflow)

    def segment(points):
        return {"points":points}

    def crossings():
        a=[segment([[0,0],[1,0]])]
        b=[segment([[1,0],[1,1]]),segment([[.25,0],[.75,0]]),segment([[.5,0],[.5,0]]),
           segment([[0,0],[1,1e-31]]),segment([[2,0],[2,1]])]
        r=c.intersections(a,b)
        require(len(r["intersections"])==1 and r["intersections"][0]["t"]==1)
        require(r["parallel_cutoff_pairs"]==3 and len(r["collinear_overlaps"])==2)
        require(len(r["near_parallel_candidate_pairs"])==1 and r["bbox_disjoint_finite_pairs"]==1)
        r=c.intersections([segment([[.5,0],[.5,0]])],[segment([[0,0],[1,0]])])
        require(len(r["collinear_overlaps"])==1 and len(r["intersections"])==0)
    check("inclusive endpoints, near-parallel cutoff and separate collinear overlaps",crossings)

    def overlaps():
        a,b=grid(),grid(); b["factors_transported"][...,0]*=-1; b["factors_transported"][...,1]*=-1
        rows=c.exact_overlaps([("a",a),("b",b)],["mu","factors_transported"])
        require(len(rows)==4 and rows[0]["fields"]["mu"]["signed"]==0)
        require((rows[0]["fields"]["factors_transported"]["signed"] != 0).any())
        b["alpha"][0]=np.nextafter(b["alpha"][0],np.inf)
        require(len(c.exact_overlaps([("a",a),("b",b)],["mu"]))==2)
    check("exact-coordinate-only overlaps preserve same-product phase gauge differences",overlaps)

    def joins():
        a,b,d=grid(2,2,[0,1]),grid(2,2,[1,2]),grid(2,2,[2,3])
        b["mu"][0,0]=42; d["log10_temperature"][0]=np.nextafter(2.,np.inf)
        r=c.plot_zoom([("a",a),("b",b),("d",d)])
        require(r["literal_zoom_cells"]==8 and len(r["row_ledger"])==4)
        require(r["joins"][0]["dropped_row_fields"]["mu"]["signed"][0]==43)
        require(r["joins"][1]["dropped_temperature_vs_previous_last"]["signed"] != 0)
        require(r["literal_zoom_min_modulus"]==1)
    check("literal drop-first-row selection keeps discarded differences",joins)

    def progress():
        g=grid(); g["mu"][1,1]=np.nan
        p=[{"seed":[0,0]}, {"i":0,"j":0,"converged":False},
           {"i":0,"j":0,"converged":True,"mu":-2.,"split":1.,"alpha":-.1,"log10_temperature":.1},
           {"i":99,"j":0,"converged":False}]
        r=c.progress_coverage(g,p)
        require(len(r["noncell_records"])==1 and len(r["duplicate_progress_indices"])==1)
        require(r["invalid_index_sequences"]==[3] and r["all_progress_records"][1]["saved_field_comparisons"]["mu"]["signed"]==-1)
        require(r["no_snapshot_reconstruction"])
    check("BFS failures, duplicates, invalid indices and snapshot differences retained",progress)

    def finite_stats():
        r=c.stats([np.nan,np.inf,-np.inf,0,2,2,-1])
        require(r["finite"]==4 and r["minimum"]==-1 and r["maximum"]==2)
        require(len(r["maximum_indices"])==2)
        encoded=json.dumps(c.plain({"r":r,"nan":float("nan")}),allow_nan=False)
        require('"nonfinite": "nan"' in encoded)
    check("true finite extrema/duplicate witnesses and explicit nonfinite JSON",finite_stats)

    def factor_pairs():
        g=grid(); g["factors_transported"][...,0]=[[-1,1],[-1,1]]
        g["factors_transported"][...,1]=[[-1,-1],[1,1]]
        r=c.factor_geometry(g,"factors_transported",g["ok"])
        require(len(r["phases"])==7 and len(r["phase_pairs"])==21)
        require(len(r["phase_pairs"][0]["crossings"]["intersections"])==1)
        require(not c.factor_geometry(g,"missing",g["ok"])["available"])
    check("all seven phase fields and 21 distinct pairs; absent methods explicit",factor_pairs)

    profiles=[]
    try:
        for name,n,m,oscillatory in [("manufactured_smooth_35_by_33",35,33,False),
                                     ("manufactured_checkerboard_12_by_12",12,12,True)]:
            t=time.perf_counter(); g=grid(n,m); x,y=np.meshgrid(g["alpha"],g["log10_temperature"])
            for k in range(7):
                g["factors_transported"][...,k]=((-1.)**(np.indices((n,m)).sum(axis=0)+k)
                                                   if oscillatory else x+(k+1)*.07*(y-.15))
            g["stretchings"]=np.abs(g["factors_transported"])
            result={"cells":c.cell_rows(g,7),"geometry":c.factor_geometry(g,"factors_transported",g["ok"])}
            text=json.dumps(c.plain(result),allow_nan=False,separators=(",",":"))
            profiles.append({"name":name,"rows":n,"columns":m,"phases":7,
                             "serialized_bytes":len(text.encode()),"seconds_through_serialization":time.perf_counter()-t,
                             "peak_rss_kib_observed":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                             "segment_pairs":sum(p["crossings"]["pairs"] for p in result["geometry"]["phase_pairs"]),
                             "intersections_retained":sum(len(p["crossings"]["intersections"]) for p in result["geometry"]["phase_pairs"]),
                             "collinear_overlaps_retained":sum(len(p["crossings"]["collinear_overlaps"]) for p in result["geometry"]["phase_pairs"])})
            del result,text,g
        passed=all(r["passed"] for r in rows)
        report={"manufactured_only":True,"target_data_read":False,"core_sha256":hashlib.sha256(Path(args.core).read_bytes()).hexdigest(),
                "controls":rows,"all_passed":passed,"profiles":profiles,"seconds_before_report_write":time.perf_counter()-begin,
                "cpu_as_nice_limits":{"cpu_seconds":25,"as_bytes":1024**3,"nice":os.getpriority(os.PRIO_PROCESS,0)},
                "scope":"One core-control run; no historical source import, target/model/AD/eigen work or actual-data admission."}
        (out/"report.json").write_text(json.dumps(c.plain(report),indent=2,allow_nan=False)+"\n")
        print(json.dumps({"all_passed":passed,"controls":len(rows),"profiles":profiles}))
        return 0 if passed else 1
    except BaseException:
        (out/"failure.json").write_text(json.dumps(c.plain({"controls":rows,"profiles":profiles,"traceback":traceback.format_exc()}),indent=2,allow_nan=False)+"\n")
        raise


if __name__=="__main__":
    raise SystemExit(main())
