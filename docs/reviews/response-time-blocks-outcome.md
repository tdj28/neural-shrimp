# Temporal blocks improve response accuracy; fixed efficiency screen fails

The one-period block conditioner improves the two saved e0 readouts markedly. For p10, PCG converges in236iterations with true normal relative residual6.94e-11 and readout error4.59e-9; matched unpreconditioned CG reaches300iterations with error.05675. For p18, PCG reaches300iterations with residual9.01e-9 and readout error1.82e-7, versus baseline.07949. The p18info300 is retained: requested1e-10 solver tolerance was not met, so the predeclared endpoint fails even though its looser actual residual/readout screens pass.

Neither slot reaches the required half-callback screen: p10uses76000phaseactions versus96480baseline; p18uses173664forboth. Three small Cholesky factors and8024/10200triangular solves add cost; walltimes.638/1.094s versus.591/.794s. The dense small blocks are a feasibility oracle, not evidence of a large-model HVP speedup or training utility. One coordinate readout is not a trace/spectral count, and finite-window bias persists.

The unit passed its synthetic resource gate and completed4.612s at62396KiB. All3400rawfiles and owned process exits were verified; the separately written post-outcome saved-arithmetic check recomputes K/KT, true residual/readout, full callback retention and factor residuals (max2.23e-16), without a new factorization or solver run. It is producer-agent checking; independent personnel review remains pending. Companion JSON binds exact source, input, raw and checker identities.

The disclosed next method comparison fixes two-period and four-period blocks on the same pair, preserves all methods/results and counts both operator work and dense factor cost. No selected-winner or scalable utility claim is justified by an adapted small-model improvement.
