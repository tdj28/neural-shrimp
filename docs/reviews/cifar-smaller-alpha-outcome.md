# CIFAR smaller-alpha result

The retained inexact Newton direction does improve orbit closure when damped sufficiently. All eight fixed trials from1/32 through1/4096 pass the unchanged Armijo test;1/16 fails. The largest passing step1/32 reduces squared residual merit by2.89%, but relative closure remains4.90077e-6 against the1e-8gate. We still have no accepted CIFAR cycle or stability result.

The one reviewed unit took1.467s internally/3.144s including the launcher, with exactly20direct gradients and no HVP/JVP/GMRES. All nine trials start from the original state and reuse the same400-space direction. Full worker/launcher inventories are immutable and hash-verified; owned processes exited. The companion binds source, review, input and all outcomes. Independent outcome review is pending.

This supports a damped inexact-Newton follow-up, not the claim that increased Krylov space alone solves the orbit. The small accepted fraction shows substantial nonlinear curvature along this direction. The input-span diagnostic already rejected complementary coordinates as the main residual source; reducing dimension is not by itself a demonstrated convergence fix.
