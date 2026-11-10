"""
cost_model.py — VersaAccel cost model (Paper Section IV, Eq. 1).

Given an operator S, system bandwidth BW, and matrix properties,
selects the optimal configuration C* that minimizes total execution cost:

    C* = argmin  [ Costcomp(S, C, BW) + Costmem(S, C, BW) ]
         C ∈ Call

Where:
    Costcomp = compute cycles (from config_modes)
    Costmem  = Data(S, C) / BW  (memory transfer time)
"""

import math
import os

from .config_modes import ALL_CONFIGS
from .memory_model import MemoryModel

# Which compute term the model charges (Rev 19).
#
#   "mac"  ADOPTED (B). The published model, with every configuration's compute
#          held to the MAC lower bound of the work it claims to do: dense MM
#          M*K*P MACs; a sparse configuration on SpMM / SpMSpM only the
#          nonzero work, M*K*P*(1-sA) (*(1-sB) for SpMSpM); over Nr*Nc PEs.
#          The published D-MR MM formula ceil(P/Nc)*(Nr+M+Nc-1) does not depend
#          on K and D-M-R's divides by Nc twice, so on real layer shapes both
#          charged 3-40x below what Nr*Nc PEs can do; their Table II check is
#          at 2x2, where the gap is invisible (section 3.0).
#   "rtl"  D (a finding, not the paper's model). The two RTL-measured
#          corrections together, with the same bound:
#          B11  DMR's dense MM is charged the RTL's tile cost,
#               ceil(M/Nr)*ceil(P/Nc)*(K+Nr+Nc-1) + Nc  (section 7.12, cycle-exact),
#               instead of the ideal-utilisation bound ceil(MKP/(Nr*Nc)).
#          B6   the sparse configurations are charged their own DENSE compute,
#               credited only what a shared-operand array can skip:
#               comp * (1 - f*K/(K+Nr+Nc)), f = skippable k-slice fraction
#               (sections 2.6, 5.3b, 7.7). Memory keeps the nnz term --
#               compressed operands still move less data. It costs the sparse
#               configurations as if run on THIS RTL's shared-operand array.
#   "nnz"  the model as published (VersaAccel, Table II): compute in proportion to
#          the nonzero count. base_simulator/validate.py, section 7.12 and
#          section 5.3b use it explicitly, since comparing against it is their point.
#
# f is measured per layer and input where the trace carries it (skippable_frac);
# otherwise it is estimated as if the zeros were i.i.d. -- an all-zero k-slice of
# the activations across Nr rows or of the weights across Nc columns:
#   f = 1 - (1 - a^Nr) * (1 - w^Nc).
COST_MODEL = os.environ.get("VERSAACCEL_COST_MODEL", "mac")
assert COST_MODEL in ("mac", "rtl", "nnz"), COST_MODEL
_DENSE_OPS = ("MM", "SpMM", "SpMSpM")


class CostModel:
    """
    Implements VersaAccel's cost model for configuration selection.

    Usage:
        model = CostModel(Nr=8, Nc=8)
        result = model.evaluate('SpMM', M=64, K=64, P=64,
                                sparsity_A=0.7, sparsity_B=0.0)
        print(result)  # {'best_config': 'DM-R', 'cycles': 234, ...}
    """

    def __init__(self, Nr=8, Nc=8, memory=None, model=None):
        self.Nr = Nr
        self.Nc = Nc
        self.memory = memory or MemoryModel()
        self.model = model or COST_MODEL
        assert self.model in ("mac", "rtl", "nnz"), self.model

    def iid_skippable(self, sparsity_A, sparsity_B):
        """Skippable k-slice fraction if zeros were i.i.d. (A = weights, B = activations)."""
        w = min(max(sparsity_A, 0.0), 1.0)
        a = min(max(sparsity_B, 0.0), 1.0)
        return 1.0 - (1.0 - a ** self.Nr) * (1.0 - w ** self.Nc)

    def _compute_nnz(self, total_elements, sparsity):
        """Convert sparsity fraction to number of nonzeros."""
        density = 1.0 - sparsity
        return max(1, int(total_elements * density))

    def cost_for_config(self, config_name, op, M, K, P,
                        sparsity_A=0.0, sparsity_B=0.0, bw_override=None, skippable=None):
        """
        Compute total cost (Costcomp + Costmem) for a specific configuration.

        Args:
            config_name: One of 'DMR', 'D-MR', 'DM-R', 'D-M-R'
            op: 'MV', 'MM', 'SpMV', 'SpMM', 'SpMSpM'
            M, K, P: Matrix dimensions (A is M×K, B is K×P)
            sparsity_A: Fraction of zeros in A (0.0 = dense, 0.9 = 90% sparse)
            sparsity_B: Fraction of zeros in B
            bw_override: Override bandwidth in elements/cycle (None = use system BW)
            skippable: measured skippable k-slice fraction f ("rtl" model only;
                       None = i.i.d. estimate from the two sparsities)

        Returns:
            dict with costcomp, costmem, total_cost, config
        """
        config = ALL_CONFIGS[config_name]
        bw = bw_override if bw_override is not None else self.memory.bw_elements_per_cycle

        # Convert sparsity to NNZ counts
        nnz_A = self._compute_nnz(M * K, sparsity_A)
        nnz_B = self._compute_nnz(K * P, sparsity_B)

        # Compute cost
        if hasattr(config, 'compute_cycles'):
            method = config.compute_cycles
            # Some configs accept bw parameter
            import inspect
            sig = inspect.signature(method)
            if 'bw' in sig.parameters:
                costcomp = method(op, M, K, P, nnz_A, nnz_B, self.Nr, self.Nc, bw=bw)
            else:
                costcomp = method(op, M, K, P, nnz_A, nnz_B, self.Nr, self.Nc)

        if self.model == "rtl" and op in _DENSE_OPS:
            # B11 / B6 (see COST_MODEL): dense compute, with the RTL's fill for
            # DMR and only the realisable skip credited to the sparse configs.
            if config_name == "DMR":
                costcomp = (math.ceil(M / self.Nr) * math.ceil(P / self.Nc)
                            * (K + self.Nr + self.Nc - 1) + self.Nc)
            else:
                dense = max(self.model_dense_compute(config, M, K, P, bw),
                            math.ceil(M * K * P / (self.Nr * self.Nc)))   # MAC bound
                f = self.iid_skippable(sparsity_A, sparsity_B) if skippable is None \
                    or skippable != skippable else min(max(float(skippable), 0.0), 1.0)
                costcomp = dense * (1.0 - f * K / (K + self.Nr + self.Nc))

        if self.model == "mac" and op in _DENSE_OPS:
            macs = M * K * P
            if config_name != "DMR" and op != "MM":
                macs *= (1.0 - sparsity_A) * ((1.0 - sparsity_B) if op == "SpMSpM" else 1.0)
            costcomp = max(costcomp, math.ceil(macs / (self.Nr * self.Nc)))

        # Memory cost
        data_elements = config.data_accessed(op, M, K, P, nnz_A, nnz_B)
        costmem = self.memory.memory_access_cycles(data_elements)

        total_cost = costcomp + costmem

        return {
            'config': config_name,
            'costcomp': costcomp,
            'costmem': costmem,
            'total_cost': total_cost,
            'data_elements': data_elements,
        }

    def model_dense_compute(self, config, M, K, P, bw):
        """A configuration's own compute for the dense MM (nnz = all elements)."""
        import inspect
        method = config.compute_cycles
        if 'bw' in inspect.signature(method).parameters:
            return method('MM', M, K, P, M * K, K * P, self.Nr, self.Nc, bw=bw)
        return method('MM', M, K, P, M * K, K * P, self.Nr, self.Nc)

    def evaluate(self, op, M, K, P=1, sparsity_A=0.0, sparsity_B=0.0, skippable=None):
        """
        Find optimal configuration C* = argmin Cost(S, C, BW).

        Returns:
            dict with best_config, cycles, and per-config breakdown
        """
        # For MV ops, P = 1
        if op in ('MV', 'SpMV'):
            P = 1

        results = {}
        for config_name in ALL_CONFIGS:
            results[config_name] = self.cost_for_config(
                config_name, op, M, K, P, sparsity_A, sparsity_B, skippable=skippable
            )

        # Find minimum total cost
        best = min(results.values(), key=lambda x: x['total_cost'])

        return {
            'best_config': best['config'],
            'total_cost': best['total_cost'],
            'costcomp': best['costcomp'],
            'costmem': best['costmem'],
            'all_configs': results,
        }

    def evaluate_all(self, op, M, K, P=1, sparsity_A=0.0, sparsity_B=0.0):
        """Return costs for ALL configurations (for comparison/analysis)."""
        result = self.evaluate(op, M, K, P, sparsity_A, sparsity_B)
        return result['all_configs']
