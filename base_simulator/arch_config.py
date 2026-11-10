"""
arch_config.py -- the accelerator's operating point, in one place.

Why this module exists
----------------------
The array size was previously written as a literal `8` at eleven call sites
(plus `64` as a magic PE count in the prefetch model and NR/NC in the RTL
package). That made the operating point impossible to change coherently, and it
hid a result: FIX_PLAN.md F1 / E3 showed that activation sparsity NEVER changes
the optimal configuration at 4x4 or 8x8, but does so for 12.5-41.7% of
layer/sparsity combinations at 16x16 and above.

Since runtime sparsity prediction is the premise of this project, the project is
scoped to arrays where that premise holds. 16x16 is the smallest such size.

    4x4, 8x8    0.0% sparsity-sensitive  -> prediction cannot help;
                                            config selection is compile-time
    16x16       12.5-20.8%               -> prediction becomes load-bearing
    32x32       33.3-41.7%               -> all four configs win somewhere

Override for sweeps or comparison against the original 8x8 results:

    VERSAACCEL_ARRAY=8 python idea1_adaptaccel/scheduler/dp_scheduler.py
"""

import os

DEFAULT_ARRAY_DIM = 16

ARRAY_ROWS = int(os.environ.get("VERSAACCEL_ARRAY", DEFAULT_ARRAY_DIM))
ARRAY_COLS = ARRAY_ROWS
NUM_PES = ARRAY_ROWS * ARRAY_COLS

# Smallest array at which activation sparsity changes the optimal config on any
# tested layer shape (experiments/array_size_sweep.py). Below this the predictor
# is provably unable to improve config selection.
PREDICTION_USEFUL_MIN_DIM = 16


def describe():
    """One-line banner so every script states its operating point."""
    note = "" if ARRAY_ROWS >= PREDICTION_USEFUL_MIN_DIM else \
        f"  [WARNING: below {PREDICTION_USEFUL_MIN_DIM}x{PREDICTION_USEFUL_MIN_DIM}, " \
        f"activation sparsity cannot change the optimal config]"
    return f"Array: {ARRAY_ROWS}x{ARRAY_COLS} ({NUM_PES} PEs){note}"
