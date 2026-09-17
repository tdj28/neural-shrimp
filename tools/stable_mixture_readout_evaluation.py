"""EXP021 explicit numerical injection into the unchanged EXP020 evaluator.

Statistical definitions, fixed census, controls, aggregation and failure rules
are inherited. No imported module is patched and no target data is loaded.
"""
import mixture_readout_evaluation as frozen
import gaussian_mixture_readout_stable as stable

cell_census = frozen.cell_census
calculation_census = frozen.calculation_census
observation_census = frozen.observation_census
methods_for_cell = frozen.methods_for_cell
assemble_cell = frozen.assemble_cell
summarize = frozen.summarize


def contract():
    result = frozen.contract()
    result['numerical_successor'] = 'EXP-021 high-precision nominal point evaluation'
    result['statistical_design_changed'] = False
    result['frozen_evaluator_reused'] = True
    return result


def seal_cell(cell, old_context_seal, alternate_tangent, *, define_fn=None):
    return frozen.seal_cell(cell, old_context_seal, alternate_tangent,
        define_fn=stable.define_classifier if define_fn is None else define_fn)


def evaluate_cell(cell, sealed, finite_scores, *, probability_fn=None,
                  linear_probability_fn=None, progress_fn=None):
    return frozen.evaluate_cell(cell, sealed, finite_scores,
        probability_fn=stable.probability if probability_fn is None else probability_fn,
        linear_probability_fn=stable.linear_probability if linear_probability_fn is None else linear_probability_fn,
        progress_fn=progress_fn)


def evaluate_calculation(cell, sealed, finite_scores, slot_spec, *,
                         probability_fn=None, linear_probability_fn=None):
    return frozen.evaluate_calculation(cell, sealed, finite_scores, slot_spec,
        probability_fn=stable.probability if probability_fn is None else probability_fn,
        linear_probability_fn=stable.linear_probability if linear_probability_fn is None else linear_probability_fn)
