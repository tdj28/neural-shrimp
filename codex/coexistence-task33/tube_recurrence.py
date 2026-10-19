"""Exact scalar tube arithmetic; caller must separately authenticate whole-tube bounds.

This module does not establish any derivative bound or certify target evidence.
A row radius is the domain on which its Lipschitz upper bound was proved.
"""
from fractions import Fraction

def nonnegative(value):
    if isinstance(value, bool) or not isinstance(value, (int, Fraction)):
        raise ValueError("exact integer/Fraction required")
    value = Fraction(value)
    if value < 0:
        raise ValueError("nonnegative bound required")
    return value

def propagate(initial_error, rows):
    """e(next) <= defect + L*e; stop before using a bound outside its domain.

For a block row, intermediate[j] = (a,b,r) asserts a separately proved
bound ||F^j(x)-c_j|| <= a+b*e0 and its required radius r. All paths must
fit before the block L may be used. No periodic monodromy is inferred.
"""
    error = nonnegative(initial_error)
    history = []
    for index, row in enumerate(rows):
        steps = row["steps"]
        if type(steps) is not int or steps < 1:
            raise ValueError("positive exact step count required")
        if len(row["intermediate"]) != steps-1:
            raise ValueError("every block intermediate path required")
        radius = nonnegative(row["input_radius"])
        if error > radius:
            return dict(passed=False, reason="input tube exceeded", index=index, error=error, history=history)
        paths = []
        for triple in row["intermediate"]:
            if len(triple) != 3:
                raise ValueError("intermediate affine bound and radius required")
            a,b,r = map(nonnegative,triple)
            upper = a+b*error
            if upper > r:
                return dict(passed=False, reason="intermediate tube exceeded", index=index, error=error, history=history)
            paths.append(upper)
        defect = nonnegative(row["defect"])
        lipschitz = nonnegative(row["lipschitz"])
        following = defect+lipschitz*error
        history.append(dict(input_error=error, output_error=following, intermediate=paths, steps=steps))
        error = following
    return dict(passed=True, error=error, history=history, target_certificate=False)
