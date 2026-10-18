# Phase preconditioner admission

Root reviewed full phase_preconditioner.py and reused generic center_seed.propose_product_and_inverse. Exact five saved phases, own float64 full896one-step Jacobians, relativeL2closure<1e-10 each phase, no failed-phase omission. Matrix product order A[p-1]...A[0], B=solve(I-product,I), both inverse defects and condition are saved; no claim they are rigorous enclosures. Reuse existing accepted generic inverse proposal controls and composed-map controls; new tiny order/defect checks recorded below. No old period8 caller is executed.

Admit one phase-proposal-001 CPU1+supervisor,12GiBAS,3800CPU/3900wall,128MiB output. May overlap dense-001 at4slots total,24GiB combinedAS plus16GiBheadroom;40GiB available startup gate. Original dense source/card/map remains untouched. Fresh exact source/input/private checkpoint and capacity required. No retry/extension. Every phase saved before inverse solve; failures preserved.
