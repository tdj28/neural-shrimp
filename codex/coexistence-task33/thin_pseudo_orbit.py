"""Thin-centre proposal loop with retained rigorous defects, not a trajectory proof."""

def pack(value):
    if not value.is_finite():
        raise ValueError("finite enclosure required")
    return {"mid":list(map(int,value.mid().man_exp())),
            "rad":list(map(int,value.rad().man_exp()))}

def generate(initial, forward, steps, save):
    if type(steps) is not int or not 1 <= steps <= 20:
        raise ValueError("bounded 1..20 steps required")
    centre=list(initial)
    if not centre or any(not v.is_finite() or not v.is_exact() for v in centre):
        raise ValueError("exact finite initial centre required")
    dimension=len(centre)
    save("centre-000.json", {"step":0,"centre":[pack(v) for v in centre]})
    for step in range(1,steps+1):
        values,tangent=forward(centre)
        if len(values)!=dimension or len(tangent)!=dimension:
            raise ValueError("complete image and tangent required")
        if any(not v.is_finite() for v in values):
            raise ValueError("nonfinite image")
        if any(not v.is_exact() or v!=0 for v in tangent):
            raise ValueError("unexpected tangent")
        following=[v.mid() for v in values]
        defects=[v-c for v,c in zip(values,following)]
        save(f"step-{step:03d}.json",dict(step=step,
             input_centre=[pack(v) for v in centre],
             full_image=[pack(v) for v in values],
             next_centre=[pack(v) for v in following],
             defects=[pack(v) for v in defects],
             defect_inf_upper=pack(max(v.abs_upper() for v in defects)),
             trajectory_enclosure=False))
        centre=following
    return centre
