# One retained direction under the residual-merit strategy

The1600MINRES result remains a failed inexact-Newton attempt:forcing0.65985
is greater than the fixed0.5 gate. Its saved direction nevertheless descends
the exact native residual merit and predicts56.46% merit reduction; norm8.80%
of state scale. Rather than rerunningKrylov, test this single retained direction
under the already used actual residual-merit strategy, explicitly retaining
the failedNewtonlabel. This is a disclosed method amendment, not a retroactive
pass or relaxed orbitcriterion.

Repeat2native map calls and2HVP to reproduce actualaction bit-for-bit. Require
predicted descent and directionnorm<=0.1state. At most13fixedbacktracking
trials use the original actual merit/Armijo/rho/separation gates. No direction
search, data/holdout/spectrum or orbit acceptance. Cap2HVP28grad,90sworker,
110+5external117supervisor,2GiBgroup255+1MiBraw,1CPUthreadnice10.
Measured previousnative maximumcost×2+20 mustfit90s. Preserve everytrial.
