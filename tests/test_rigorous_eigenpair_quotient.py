import pytest
from flint import arb,arb_mat,ctx
from neural_shrimp.rigorous_eigenpair_quotient import coordinate_frame,quotient_blocks,qualify


def test_nonnormal_native_quotient_and_cyclic_order():
    with ctx.workprec(128):
        v0=arb_mat([[1,2],[0,1]]);v1=arb_mat([[1,0],[2,1]])
        t0=arb_mat([[1,3],[0,arb('1/2')]]);t1=arb_mat([[1,-2],[0,arb('1/3')]])
        js=[v1*t0*v0.inv(),v0*t1*v1.inv()];qs=[[v0[i,0]for i in range(2)],[v1[i,0]for i in range(2)]]
        bs,meta=quotient_blocks(js,qs)
        assert (bs[1]*bs[0])[0,0].contains(arb('1/6'))
        assert qualify(js,qs,precision=128)['passed']
        assert meta[1]['pivot']==1


def test_neutral_or_unstable_complement_not_accepted():
    for other in ['1','-1','2']:
        with ctx.workprec(128):
            try:r=qualify([arb_mat([[1,0],[0,arb(other)]])],[[arb(1),arb(0)]],precision=128)
            except (ValueError,ZeroDivisionError):continue
            assert not r['passed']


def test_contracting_negative_mode_allowed():
    with ctx.workprec(128):assert qualify([arb_mat([[1,3],[0,arb('-1/2')]])],[[arb(1),arb(0)]],precision=128)['passed']


def test_interval_mode_and_rescaling():
    with ctx.workprec(128):
        q=[arb(2,arb('1e-12')),arb(1,arb('1e-12'))]
        v,vi,m=coordinate_frame(q)
        assert m['pivot']==0
        assert (vi*v)[0,0].contains(1)
        assert (vi*v)[1,0].contains(0)


def test_noninvariant_complement_does_not_require_zero_top_right():
    with ctx.workprec(128):
        bs,_=quotient_blocks([arb_mat([[1,99],[0,arb('1/2')]])],[[arb(1),arb(0)]])
        assert bs[0][0,0]==arb('1/2')


def test_first_column_mismatch_rejected():
    with pytest.raises(ValueError,match='compatibility'):quotient_blocks([arb_mat([[2,0],[0,arb('1/2')]])],[[arb(1),arb(0)]])


def test_zero_pivot_and_nonfinite_rejected():
    for q in [[arb(0),arb(0)],[arb(0,1),arb(0,1)],[arb('nan'),arb(1)]]:
        with pytest.raises(ValueError):coordinate_frame(q)


def test_exact_tie_first_pivot():
    assert coordinate_frame([arb(-1),arb(1)])[2]['pivot']==0


def test_return_is_preserved_before_observer_interruption():
    rows=[]
    def observe(k,v):
        rows.append(k)
        if k=='transformed_return':raise TimeoutError('retained')
    with pytest.raises(TimeoutError):quotient_blocks([arb_mat([[1,0],[0,arb('1/2')]])],[[arb(1),arb(0)]],observe)
    assert rows[-1]=='transformed_return'


def test_three_dimensional_quotient_keeps_matrix_order():
    with ctx.workprec(128):
        q=[arb(1),arb(0),arb(0)]
        a=arb_mat([[1,2,3],[0,arb('1/2'),arb('1/4')],[0,0,arb('1/3')]])
        b=arb_mat([[1,-3,1],[0,arb('1/4'),0],[0,arb('1/5'),arb('1/3')]])
        blocks,_=quotient_blocks([a,b],[q,q])
        product=blocks[1]*blocks[0];expected=b*a
        for i in range(2):
            for j in range(2):assert product[i,j].overlaps(expected[i+1,j+1])
        assert not product[0,1].overlaps((blocks[0]*blocks[1])[0,1])


def test_saved_native_layout_and_bad_period():
    import importlib.util
    from pathlib import Path
    from neural_shrimp.rigorous_orbits import ball_json
    path=Path(__file__).resolve().parents[1]/'tools/qualify_eigenpair_quotient.py'
    spec=importlib.util.spec_from_file_location('quotient_driver',path);d=importlib.util.module_from_spec(spec);spec.loader.exec_module(d)
    z=ball_json(arb(0));rows=[[z for _ in range(470)]for _ in range(470)];box=[z]*470
    for p in range(18):
        rows[26*p][26*p]=ball_json(arb(p+1));box[26*p+13]=ball_json(arb(p+2))
    chart={'period':18,'dimension':13,'sigma':1,'shooting_dimension':470,'passed':True,'gates':{'root':True},'whole_box':box,'whole_jacobian':rows}
    js,qs=d.extract_native(chart)
    assert all(js[p][0,0]==p+1 and qs[p][0]==p+2 for p in range(18))
    chart['sigma']=-1
    with pytest.raises(ValueError):d.extract_native(chart)
