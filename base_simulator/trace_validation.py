"""
trace_validation.py -- integrity checks for profiling traces.

Sparsity is a fraction of zeros in [0, 1]. Generators that add Gaussian noise
have historically emitted negative values (see FIX_PLAN.md B1); this module
makes such data a hard error rather than something that silently propagates
into the cost model.
"""

SPARSITY_COLUMNS = (
    "input_sparsity",
    "output_sparsity",
    "weight_sparsity",
    "attention_sparsity",
    "nm_compatibility",
)


def validate_traces(df, source="<dataframe>", raise_on_error=True):
    """Check sparsity columns lie in [0, 1]. Returns a list of problem strings."""
    problems = []
    for col in SPARSITY_COLUMNS:
        if col not in df.columns:
            continue
        bad = df[(df[col] < 0.0) | (df[col] > 1.0)]
        if len(bad):
            problems.append(
                f"{source}: {len(bad)} rows have {col} outside [0,1] "
                f"(min={df[col].min():.4f}, max={df[col].max():.4f})"
            )
    if problems and raise_on_error:
        raise ValueError("Invalid trace data:\n  " + "\n  ".join(problems))
    return problems


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------
# Sparsity numbers are only as meaningful as the network and inputs that
# produced them. A trace from a randomly-initialised model fed Gaussian noise
# is a measurement of ReLU, not of ResNet-50 (FIX_PLAN.md M7), and once it is a
# CSV on disk nothing distinguishes it from a real one. The profilers now stamp
# `data_source` and `weights` columns; this reports what a trace set actually
# contains so no downstream script can silently treat noise as real.

SYNTHETIC_SOURCES = ("gaussian-noise", "random-tokens")
SYNTHETIC_WEIGHTS = ("random-init",)


def describe_provenance(df, source="<dataframe>"):
    """Return (is_real, message) for a trace set."""
    if "data_source" not in df.columns and "weights" not in df.columns:
        return False, (
            f"{source}: UNSTAMPED -- predates provenance tracking. Assume it was "
            f"generated from a randomly-initialised model on synthetic inputs "
            f"(the pre-M7 default) unless you know otherwise. Regenerate with "
            f"profiler/profile_resnet.py or profile_bert.py."
        )
    src = str(df["data_source"].iloc[0]) if "data_source" in df.columns else "?"
    wts = str(df["weights"].iloc[0]) if "weights" in df.columns else "?"
    synthetic = src in SYNTHETIC_SOURCES or wts in SYNTHETIC_WEIGHTS
    tag = "SYNTHETIC" if synthetic else "real"
    return (not synthetic), f"{source}: {tag} (inputs={src}, weights={wts})"


def warn_if_synthetic(df, source="<dataframe>"):
    """Print a provenance line; return True if the traces are real."""
    is_real, msg = describe_provenance(df, source)
    print(("  " if is_real else "  !! ") + msg)
    return is_real
