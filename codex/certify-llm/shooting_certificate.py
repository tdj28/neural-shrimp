"""Join conditional shooting, primitive-period and dynamical-attraction gates.

This module consumes externally authenticated mathematical enclosures. It does
not read receipts, run a transformer or establish the evaluator contract. It
never emits certificate_passed=True, even when every conditional gate succeeds.
All inputs and the arithmetic context must remain unchanged during a call.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

from flint import ctx
import proof_gates as proof
import shooting_gates as shooting


def check_shooting_certificate(centers, radii, images, jacobians, derivatives,
        inverse_cycle, *, contract, bound_method='factored', powers=(1, 2),
        stability_weights=None):
    """Check one exact ordered phase tuple under its external enclosure contract.

    The primitive test separates the root's phase boxes; those boxes are not
    asserted to enclose F^d(X_0). Multiplying J_(p-1)...J_0 encloses the derivative
    of F^p at that root, since its corresponding phases belong to the boxes.
    A power norm below one establishes local attraction, not whole-box trapping.
    Each gate runs and is retained even when another gate fails.
    """
    precision, threads = ctx.prec, ctx.threads
    root = shooting.shooting_root_gate(centers, radii, images, jacobians,
        derivatives, inverse_cycle, contract=contract, bound_method=bound_method)
    primitive = shooting.primitive_phase_gate(centers, radii)
    if primitive['domain_sha256'] != root['domain_sha256']:
        raise ValueError('primitive and shooting domains differ')
    p, n = root['period'], root['dimension']
    monodromy = proof._eye(n)
    multiplication = []
    for phase, jacobian in enumerate(jacobians):
        # Root admission has checked every input and its bound hash. Preserve
        # signed interval multiplication and its actual order in the witness.
        monodromy = jacobian * monodromy
        proof._matrix(monodromy, f'partial monodromy through phase {phase}', n=n)
        multiplication.append(dict(phase=phase,
            jacobian_sha256=shooting._matrix_sha256(jacobian),
            partial_product_sha256=shooting._matrix_sha256(monodromy)))
    if len(multiplication) != p:
        raise ValueError('all ordered phase Jacobians are required')
    attraction = proof.attraction_gate(monodromy, powers=powers,
                                       weights=stability_weights)
    if (ctx.prec, ctx.threads) != (precision, threads):
        raise ValueError('arithmetic context changed during conditional gates')
    flags = dict(root_existence_and_local_uniqueness=root['conditional_gates_passed'],
        primitive_period=primitive['conditional_primitive_period'],
        local_attraction=attraction['passed'])
    return dict(schema='codex-conditional-shooting-certificate-v1',
        certificate_passed=False, external_authentication_performed=False,
        enclosure_validity_checked_by_this_module=False,
        conditional_gates_passed=all(flags.values()), conditional_gates=flags,
        period=p, dimension=n, precision=precision, threads=threads,
        contract=root['contract'], domain_sha256=root['domain_sha256'],
        enclosures_sha256=root['enclosures_sha256'],
        proposals_sha256=root['proposals_sha256'],
        source_sha256={Path(module.__file__).name:
            hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
            for module in (proof, shooting)},
        wrapper_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        root=root, primitive=primitive,
        monodromy=dict(order='J_(p-1) ... J_1 J_0',
            phase_product_witnesses=multiplication,
            matrix_sha256=shooting._matrix_sha256(monodromy),
            scope='Derivative of F^p at any shooting root in these same phase boxes; no full-box forward invariance is asserted.'),
        attraction=attraction,
        statement='Conditional on independently established whole-box derivative and exact-centre image enclosures for the bound map: all three gates imply an existing locally unique primitive period-p orbit that is locally attracting. Numerical proposals, metadata and this conditional calculation alone do not establish that contract. A failed gate is an inconclusive sufficient-condition test, not evidence that no orbit exists.')
