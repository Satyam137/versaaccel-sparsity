"""
layer_cost.py -- the single canonical per-layer cost function.

Both Idea 1 (dp_scheduler) and Idea 2 (integrated_scheduler) previously carried
their own near-duplicate copies of this logic, and they had drifted apart:
Idea 2 silently multiplied D-M-R cost by 1.12 and DM-R by 1.06 ("Benes
overhead", unmeasured) while Idea 1 did not. The same workload therefore
scored differently depending on which script you ran. See FIX_PLAN.md M3.

This module is now the only implementation. Any cost adjustment belongs here,
applies uniformly, and must be justified in `ADJUSTMENTS` below.

Operand convention (FIX_PLAN.md B5)
-----------------------------------
A neural-network layer computes  activations(M x K) @ weights(K x P).

VersaAccel's sparse dataflows compress the FIRST operand (matrix A) and stream
it row-wise (Gustavson). The operand worth compressing is the statically-known
pruned weight matrix, so we map:

    sparsity_A  <- weight_sparsity      (static, known at deploy time)
    sparsity_B  <- activation_sparsity  (dynamic, predicted at runtime)

The previous code did `sparsity_A = max(activation_sp, weight_sp)` and never
set `sparsity_B` at all. That had two distinct defects:
  1. weight sparsity was attributed to the activation matrix, and
  2. activation sparsity was *discarded entirely* whenever weights were
     sparser -- which, on the pruned traces, is 70% of rows.

Operator selection follows from which operands are actually sparse.
"""

from .cost_model import CostModel

CONFIG_NAMES = ['DMR', 'D-MR', 'DM-R', 'D-M-R']

# Threshold above which an operand is worth treating as sparse. Below this the
# index/metadata overhead of a compressed format outweighs the skipped MACs.
SPARSE_THRESHOLD = 0.3

# Per-config cost multipliers. Empty by design: every previous entry here was an
# unmeasured constant that biased the comparison. Populate only from synthesis
# or RTL measurement, and say so in the paper.
ADJUSTMENTS = {}


def select_operator(weight_sparsity, activation_sparsity):
    """Pick the VersaAccel operator implied by the two operand sparsities."""
    a_sparse = weight_sparsity > SPARSE_THRESHOLD
    b_sparse = activation_sparsity > SPARSE_THRESHOLD
    if a_sparse and b_sparse:
        return 'SpMSpM'
    if a_sparse:
        return 'SpMM'
    return 'MM'


def get_layer_costs(cost_model, layer_info, activation_sparsity,
                    config_names=CONFIG_NAMES):
    """
    Execution cost for one layer under each config.

    Args:
        cost_model: a CostModel instance
        layer_info: mapping with M, K, P, weight_sparsity
        activation_sparsity: input-activation sparsity for this layer
                             (actual when measuring, predicted when scheduling)

    Returns:
        dict {config_name: total_cost_cycles}
    """
    M = int(layer_info['M'])
    K = int(layer_info['K'])
    P = int(layer_info['P'])
    weight_sp = float(layer_info['weight_sparsity'])
    act_sp = float(activation_sparsity)

    op = select_operator(weight_sp, act_sp)
    # Measured structure where the trace has it (all real sets). It is the
    # measured value even when activation_sparsity is a prediction: section
    # 5.3b shows f never changes a configuration choice (0 of 125,000 points),
    # so this cannot leak a decision to a predictor.
    skip = layer_info.get('skippable_frac') if hasattr(layer_info, 'get') else None

    costs = {}
    for cfg in config_names:
        result = cost_model.cost_for_config(
            cfg, op, M, K, P,
            sparsity_A=weight_sp,   # compressed operand: the pruned weights
            sparsity_B=act_sp,      # streamed operand: the activations
            skippable=skip,         # "rtl" model: measured k-slice structure
        )
        costs[cfg] = result['total_cost'] * ADJUSTMENTS.get(cfg, 1.0)
    return costs


# ---------------------------------------------------------------------------
# Decision relevance
# ---------------------------------------------------------------------------
# A config-match rate computed over all layers is dominated by layers whose
# optimal config does not depend on activation sparsity at all. On those the
# predictor is irrelevant by construction: any prediction, however wrong,
# selects the same config, so they inflate the metric toward 100%.
#
# At 8x8 this was 100% of layers, which is why a constant predictor scored a
# perfect match there (FIX_PLAN.md F1). Restricting the metric to the
# decision-relevant subset is what makes it discriminate.

RELEVANCE_PROBE = (0.0, 0.3, 0.6, 0.99)


def is_decision_relevant(cost_model, M, K, P, weight_sparsity,
                         probe=RELEVANCE_PROBE):
    """True if sweeping activation sparsity changes this layer's optimal config."""
    chosen = {
        cost_model.evaluate(
            select_operator(weight_sparsity, sp), int(M), int(K), int(P),
            sparsity_A=weight_sparsity, sparsity_B=sp,
        )["best_config"]
        for sp in probe
    }
    return len(chosen) > 1
