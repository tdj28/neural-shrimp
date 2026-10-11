# Bounded stationary seed continuation preparation

2026-10-11 UTC. Root's saved-merit001 numerical step decreases gradient merit
at alpha1/8 with true forcing.765; the independent saved audit is pending.
The original fixed.5 failure remains unchanged. The new preparatory helper
uses at most four fresh256-action MINRES directions at the actual current
state, each with a fresh true action, forcing<=.9 and negative measured slope,
then13dyadic actual gradient-merit Armijo trials. No old-state action is reused.
The original midpoint5% trust ball persists across the already accepted step
and all future updates; it is never reset around the latest endpoint.

This helper derives transparently from cifar_stationary.stationary with only
the disclosed forcing and explicit-original-origin changes. Stationarity1e-10
is unchanged; a positive-mode stage, if later included, remains conditional on
stationarity. No eta change, eigenpair, cycle or rigorous stationarity claim.

Before any actual run: independent saved-merit audit must pass; literal exact
accepted state/gradient and original origin, data/model/source/runtime must be
bound; fresh same-state symmetry/FD and measured total-budget gates; all native
attempted inputs/returns retained. A separate reviewed wrapper/card will fix
source, complete1063HVP/59gradient ceiling (including qualification and possible
33mode actions),400s/2GiB RSS/3GiB raw, source-bound actual profile and ownership.
This preparation does not authorize target acquisition or adaptive iteration
expansion. Controls cover a genuine .75-forcing descent, global trust retention,
and stopping before trial when forcing exceeds.9.
