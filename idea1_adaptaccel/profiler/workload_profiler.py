"""
workload_profiler.py — Step 1.1: Profile PyTorch models to extract per-layer sparsity traces.

For each layer in a model, captures:
  - layer_name, layer_type (Linear, Conv2d, etc.)
  - op_type mapped to VersaAccel operator (MM, MV, SpMM, etc.)
  - dimensions M, K, P
  - input_sparsity (fraction of zeros in input activation)
  - output_sparsity (fraction of zeros in output activation)
  - weight_sparsity (fraction of zeros in weight tensor)
  - a_kslice_zero / w_kslice_zero / skippable_frac: WHERE the zeros are, at the
    tile granularity the RTL actually skips on (section 7.7). Density says how
    many; these say whether the array can do anything about them.

Runs inference on N_SAMPLES inputs and records a trace per input.
Saves results to CSV files in profiling_data/ directory.

Usage:
    cd project/idea1_adaptaccel
    python -m profiler.workload_profiler
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
import sys
from collections import OrderedDict

# ============================================================================
# Sparsity measurement
# ============================================================================

def measure_sparsity(tensor):
    """Fraction of zeros in a tensor (0.0 = fully dense, 1.0 = all zeros)."""
    if tensor is None or tensor.numel() == 0:
        return 0.0
    return float((tensor == 0).sum().item() / tensor.numel())


# ----------------------------------------------------------------------------
# Structural sparsity (B5a)
#
# Scalar density says how MANY zeros there are. It says nothing about WHERE,
# and section 7.7 measured that a shared-operand systolic array can only skip a
# zero when an ENTIRE k-slice of a tile is zero -- 50% unstructured zeros save
# 0%, the same density structured along k saves 12.8-28.2%.
#
# These functions capture the quantity that IS decision-relevant, at the same
# tile granularity the RTL skips on.
# ----------------------------------------------------------------------------

TILE_NR = 16   # must match NR in idea2_veriflex/rtl/common/versaaccel_pkg.sv
TILE_NC = 16   # must match NC


def _operand_a(module, x):
    """Activation operand as a dense [M, K] matrix, im2col'd for Conv2d.

    This is the matrix the array actually streams, not the NCHW tensor -- a
    k-slice is a column of THIS, so the metric has to be computed after im2col
    or it measures nothing the hardware sees.
    """
    if isinstance(module, nn.Conv2d):
        # F.unfold IS im2col: [N, Cin*kh*kw, L] with L the output positions.
        cols = torch.nn.functional.unfold(
            x, module.kernel_size, dilation=module.dilation,
            padding=module.padding, stride=module.stride)
        return cols.permute(0, 2, 1).reshape(-1, cols.shape[1])   # [N*L, K]
    return x.reshape(-1, x.shape[-1])


def _operand_b(module):
    """Weight operand as a dense [K, P] matrix.

    A grouped convolution (B9: MobileNetV2's depthwise layers, groups = Cin) is
    run on a GEMM array as ONE GEMM against im2col over all Cin*kh*kw inputs
    with a block-diagonal weight matrix -- output channels of group g read only
    group g's inputs. Its zeros are therefore structural and lie along whole
    k-slices of each column tile, which is what the array can skip. groups = 1
    (every ResNet-50 and BERT layer) is unchanged.
    """
    w = module.weight.detach()
    if isinstance(module, nn.Conv2d):
        if module.groups > 1:
            g = module.groups
            cout, cin_g = w.shape[0], w.shape[1]
            kk = w.shape[2] * w.shape[3]
            co_g = cout // g
            B = w.new_zeros(cin_g * g * kk, cout)
            for i in range(g):
                B[i * cin_g * kk:(i + 1) * cin_g * kk, i * co_g:(i + 1) * co_g] = \
                    w[i * co_g:(i + 1) * co_g].reshape(co_g, -1).t()
            return B
        return w.reshape(w.shape[0], -1).t()      # [Cout, K] -> [K, P]
    return w.t()                                   # Linear [P, K] -> [K, P]


def kslice_structure(module, x):
    """Zero-structure of a layer at the granularity the RTL skips on.

    Returns (a_kslice_zero, w_kslice_zero, skippable_frac):

      a_kslice_zero   fraction of (row-tile, k) pairs of A that are ALL zero
      w_kslice_zero   fraction of (k, col-tile) pairs of B that are ALL zero
      skippable_frac  fraction of (row-tile, k, col-tile) steps the array can
                      skip -- i.e. 1 - mean(a_nz AND b_nz), which is exactly
                      the `mask` computed in versaaccel_top.sv

    Partial tiles are zero-padded, matching the hardware: a padded row really
    is zero and really does make its slice skippable.
    """
    try:
        with torch.no_grad():
            A = _operand_a(module, x.detach()).float()
            B = _operand_b(module).float()
            M, K = A.shape
            if K == 0 or M == 0 or B.shape[0] != K:
                return 0.0, 0.0, 0.0
            P = B.shape[1]

            n_ti = (M + TILE_NR - 1) // TILE_NR
            n_tj = (P + TILE_NC - 1) // TILE_NC
            if n_ti * TILE_NR > M:
                A = torch.nn.functional.pad(A, (0, 0, 0, n_ti * TILE_NR - M))
            if n_tj * TILE_NC > P:
                B = torch.nn.functional.pad(B, (0, n_tj * TILE_NC - P))

            a_nz = (A.reshape(n_ti, TILE_NR, K) != 0).any(dim=1)   # [n_ti, K]
            b_nz = (B.reshape(K, n_tj, TILE_NC) != 0).any(dim=2)   # [K, n_tj]

            a_kslice_zero = 1.0 - float(a_nz.float().mean())
            w_kslice_zero = 1.0 - float(b_nz.float().mean())

            # mean over (ti, k, tj) of (a_nz & b_nz), without forming the
            # n_ti x K x n_tj product: for each k the count is |a|_k * |b|_k.
            ca = a_nz.sum(dim=0).double()
            cb = b_nz.sum(dim=1).double()
            work = float((ca * cb).sum()) / float(n_ti * K * n_tj)
            return a_kslice_zero, w_kslice_zero, 1.0 - work
    except Exception:
        # A structural metric must never take down a profiling run.
        return float('nan'), float('nan'), float('nan')


def get_op_type(module, input_shape, output_shape):
    """Map a PyTorch module to a VersaAccel operator type."""
    if isinstance(module, nn.Linear):
        # Linear: output = input @ weight.T → MM or MV
        batch_size = input_shape[0] if len(input_shape) > 1 else 1
        if batch_size == 1 and len(input_shape) <= 2:
            return 'MV'
        return 'MM'
    elif isinstance(module, nn.Conv2d):
        # Conv2d via im2col → MM
        return 'MM'
    else:
        return 'OTHER'


def get_dimensions(module, input_shape, output_shape):
    """Extract M, K, P dimensions from a layer for VersaAccel."""
    if isinstance(module, nn.Linear):
        # Linear(in_features, out_features): weight is [out, in]
        # MM: A(batch × in) × B(in × out) → C(batch × out)
        K = module.in_features
        P = module.out_features
        M = 1
        for d in input_shape[:-1]:  # all dims except last (features)
            M *= d
        return M, K, P

    elif isinstance(module, nn.Conv2d):
        # Conv2d via im2col: A(N*Hout*Wout × Cin*kH*kW) × B(Cin*kH*kW × Cout)
        Cin = module.in_channels
        Cout = module.out_channels
        kH, kW = module.kernel_size
        K = Cin * kH * kW
        P = Cout
        # M = batch * output_spatial
        batch = input_shape[0]
        H_out = output_shape[2] if len(output_shape) > 2 else 1
        W_out = output_shape[3] if len(output_shape) > 3 else 1
        M = batch * H_out * W_out
        return M, K, P

    return 1, 1, 1


# ============================================================================
# Profiler class
# ============================================================================

class LayerProfiler:
    """
    Hooks into a PyTorch model and records per-layer statistics during inference.
    """

    def __init__(self, model, model_name="model"):
        self.model = model
        self.model_name = model_name
        self.records = []
        self.hooks = []
        self._input_idx = 0

    def _make_hook(self, name, module):
        """Create a forward hook for a specific layer."""
        def hook_fn(mod, inp, out):
            # Get input tensor
            x = inp[0] if isinstance(inp, tuple) else inp
            y = out if isinstance(out, torch.Tensor) else (out[0] if isinstance(out, tuple) else None)

            if x is None or not isinstance(x, torch.Tensor):
                return
            if y is None or not isinstance(y, torch.Tensor):
                return

            input_shape = tuple(x.shape)
            output_shape = tuple(y.shape)

            op_type = get_op_type(mod, input_shape, output_shape)
            if op_type == 'OTHER':
                return  # Skip non-compute layers (ReLU, BatchNorm, etc.)

            M, K, P = get_dimensions(mod, input_shape, output_shape)
            a_ks, w_ks, skip_f = kslice_structure(mod, x)

            record = {
                'input_idx': self._input_idx,
                'layer_name': name,
                'layer_type': mod.__class__.__name__,
                'op_type': op_type,
                'M': M, 'K': K, 'P': P,
                'input_sparsity': measure_sparsity(x.detach()),
                'output_sparsity': measure_sparsity(y.detach()),
                # grouped conv: zeros of the block-diagonal GEMM operand (B9)
                'weight_sparsity': (measure_sparsity(_operand_b(mod)) if getattr(mod, 'groups', 1) > 1
                                    else measure_sparsity(mod.weight.detach()) if hasattr(mod, 'weight') and mod.weight is not None else 0.0),
                'a_kslice_zero': a_ks,
                'w_kslice_zero': w_ks,
                'skippable_frac': skip_f,
                'input_shape': str(input_shape),
                'output_shape': str(output_shape),
            }
            self.records.append(record)

        return hook_fn

    def register_hooks(self):
        """Register forward hooks on all Linear and Conv2d layers."""
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                h = module.register_forward_hook(self._make_hook(name, module))
                self.hooks.append(h)
        print(f"  Registered hooks on {len(self.hooks)} layers")

    def remove_hooks(self):
        """Remove all hooks."""
        for h in self.hooks:
            h.remove()
        self.hooks.clear()

    def profile(self, dataloader, n_samples=100):
        """
        Run inference on n_samples inputs and record per-layer traces.

        Args:
            dataloader: PyTorch DataLoader yielding (input, target) batches
            n_samples: Number of inputs to profile
        """
        self.model.eval()
        self.records = []
        self.register_hooks()

        count = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(dataloader):
                if inputs.device != next(self.model.parameters()).device:
                    inputs = inputs.to(next(self.model.parameters()).device)

                # Process each sample individually for per-input traces
                for i in range(inputs.size(0)):
                    if count >= n_samples:
                        break
                    self._input_idx = count
                    single_input = inputs[i:i+1]
                    self.model(single_input)
                    count += 1

                    if count % 50 == 0:
                        print(f"  Profiled {count}/{n_samples} inputs...")

                if count >= n_samples:
                    break

        self.remove_hooks()
        print(f"  Done. Collected {len(self.records)} records from {count} inputs.")
        return pd.DataFrame(self.records)

    def save_traces(self, output_dir, provenance=None):
        """Save profiling records to CSV.

        `provenance` records WHERE the numbers came from -- pretrained weights or
        random init, real inputs or noise. Activation sparsity measured on a
        randomly-initialised network fed Gaussian noise says nothing about the
        real network, and a trace file that does not say which it is will
        eventually be mistaken for the other. See FIX_PLAN.md M7.
        """
        os.makedirs(output_dir, exist_ok=True)
        df = pd.DataFrame(self.records)
        if provenance:
            for k, v in provenance.items():
                df[k] = v
        path = os.path.join(output_dir, f"{self.model_name}_traces.csv")
        df.to_csv(path, index=False)
        print(f"  Saved to {path} ({len(df)} rows)")
        if provenance:
            print(f"  Provenance: {provenance}")
        return path
