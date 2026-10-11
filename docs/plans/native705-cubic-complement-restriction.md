# Cubic restriction for later boundary complement bounds

After complete producer complementary coverage of originalcharts7–30, extend
the same source-reviewed correlation method to later independently proved
cubic705 charts. Initial source-only helper recenters p=c+sδ+aδ²/2+bδ³ exactly,
retains original full705 correction radii, and checks |shift|+h<=H. It returns
full predictor and predictor-derivative domains, plus separate residualTaylor
bounds R=c0+c1δ+c2δ²+c3δ³+c4(I)δ⁴ and
R′=c1+2c2δ+3c3δ²+4c4(I)δ³. This uses Taylor's theorem for R′ independently;
it does not differentiate an interval remainder. Same fullnative root provenance,
all705 coordinates/rates/angle, actual original path acquisition domains and
fresh local J/C are caller obligations. No old-J reuse or existence claim.

Five controls exercise positive/negative/centered translation, retained root
correction, quintic/quartic analytic residuals and invaliddomains. Initial
synthetic version used Python Arb exponentiation on a sign-changing interval,
which returns nan in this runtime; controls caught it before any target. All
integer powers now use repeated multiplication, including negative shifts.
No target wrapper, proof acquisition or new complementary result exists yet.
