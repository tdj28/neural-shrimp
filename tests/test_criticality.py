import numpy as np
import pytest

from neural_shrimp.criticality import finite_difference_forms, ns_coefficient


def to_c(v):
    return v[..., 0] + 1j * v[..., 1]


def to_r(z):
    return np.stack([z.real, z.imag], axis=-1)


class PlanarMap:
    """z -> m z + g20/2 z^2 + g11 z zbar + g02/2 zbar^2 + g21/2 z^2 zbar in real coordinates."""

    def __init__(self, m, g20, g11, g02, g21):
        self.m, self.g20, self.g11, self.g02, self.g21 = m, g20, g11, g02, g21

    def __call__(self, x):
        z = to_c(x)
        zb = np.conj(z)
        return to_r(self.m * z + self.g20 / 2 * z * z + self.g11 * z * zb + self.g02 / 2 * zb * zb
                    + self.g21 / 2 * z * z * zb)

    def jvp(self, X, V):
        z, v = to_c(X), to_c(V)
        zb, vb = np.conj(z), np.conj(v)
        return to_r(self.m * v + self.g20 * z * v + self.g11 * (z * vb + zb * v) + self.g02 * zb * vb
                    + self.g21 / 2 * (2 * z * zb * v + z * z * vb))

    def A(self):
        c, s = self.m.real, self.m.imag
        return np.array([[c, -s], [s, c]])

    def B(self, u, v):
        a, b = to_c(u), to_c(v)
        return to_r(self.g20 * a * b + self.g11 * (a * np.conj(b) + np.conj(a) * b) + self.g02 * np.conj(a * b))

    def C(self, u, v, w):
        a, b, c = to_c(u), to_c(v), to_c(w)
        return to_r(self.g21 * (a * b * np.conj(c) + a * c * np.conj(b) + b * c * np.conj(a)))


def planar_d(m, g20, g11, g02, g21):
    """Kuznetsov's explicit first Lyapunov coefficient for planar maps in complex form."""
    return ((np.conj(m) * g21).real / 2
            - ((1 - 2 * m) * np.conj(m) ** 2 / (2 * (1 - m)) * g20 * g11).real
            - abs(g11) ** 2 / 2 - abs(g02) ** 2 / 4)


@pytest.mark.parametrize("seed", [11, 12, 13, 14])
def test_matches_explicit_planar_formula(seed):
    rng = np.random.default_rng(seed)
    theta = rng.uniform(0.05, 0.2) * 2 * np.pi
    m = np.exp(1j * theta)
    g = rng.normal(size=4) + 1j * rng.normal(size=4)
    f = PlanarMap(m, *g)
    exact = ns_coefficient(f.A(), f.B, f.C)
    B, C = finite_difference_forms(f.jvp, np.zeros(2))
    numeric = ns_coefficient(f.A(), B, C)
    # q is normalized to |q| = 1, i.e. w = z / sqrt(2): the cubic coefficient doubles.
    assert exact["d"] == pytest.approx(2 * planar_d(m, *g), rel=1e-10, abs=1e-12)
    assert numeric["d"] == pytest.approx(exact["d"], rel=1e-7, abs=1e-9)
    assert exact["theta_over_2pi"] == pytest.approx(theta / (2 * np.pi))


@pytest.mark.parametrize("sign", [-1.0, 1.0])
def test_sign_predicts_soft_or_hard_loss_of_stability(sign):
    theta = 0.13 * 2 * np.pi
    m0 = np.exp(1j * theta)
    g21 = 2 * sign * 0.5 * m0  # pure resonant cubic term: d = Re(conj(m) g21) / 2 * 2 = sign * 0.5 * 2
    d = ns_coefficient(PlanarMap(m0, 0, 0, 0, g21).A(), PlanarMap(m0, 0, 0, 0, g21).B,
                       PlanarMap(m0, 0, 0, 0, g21).C)["d"]
    assert np.sign(d) == sign
    eps = 1e-3
    f = PlanarMap((1 + eps) * m0, 0, 0, 0, g21)
    x = np.array([1e-3, 0.0])
    for _ in range(200000):
        x = f(x)
        if not np.all(np.isfinite(x)) or np.hypot(*x) > 1.0:
            break
    radius = np.hypot(*x)
    if sign < 0:
        # attracting circle of radius sqrt(eps / |Re(conj(m) g21 / 2)|) in z coordinates
        assert radius == pytest.approx(np.sqrt(eps / 0.5), rel=2e-2)
    else:
        assert not np.isfinite(radius) or radius > 0.5
