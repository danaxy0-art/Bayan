"""Lab 6 starter: bootstrap confidence intervals."""

import numpy as np


def bootstrap_ci(values, *, n_boot=2000, seed=42, alpha=0.05):
    """Compute a point estimate (mean) and a bootstrap confidence
    interval for a list of per-example scores (e.g. 0/1 correctness,
    or continuous metrics).

    Returns (point_estimate, lo, hi) where [lo, hi] is the
    (1 - alpha) bootstrap percentile confidence interval.
    """

    values = np.asarray(values, dtype=float)
    n = len(values)

    point = float(values.mean())

    rng = np.random.default_rng(seed)

    boot_means = np.empty(n_boot)
    for i in range(n_boot):
        sample_idx = rng.integers(0, n, size=n)
        boot_means[i] = values[sample_idx].mean()

    lower_pct = (alpha / 2) * 100
    upper_pct = (1 - alpha / 2) * 100

    lo = float(np.percentile(boot_means, lower_pct))
    hi = float(np.percentile(boot_means, upper_pct))

    return point, lo, hi


def paired_bootstrap_diff(a, b, *, n_boot=2000, seed=42, alpha=0.05):
    """Compute a paired bootstrap confidence interval for the mean
    difference (a - b), resampling matched pairs together so the
    pairing between a[i] and b[i] is preserved in every resample.

    Returns (delta, lo, hi).

    Raises ValueError if a and b have different lengths.
    """

    if len(a) != len(b):
        raise ValueError(
            f"a and b must have the same length, got {len(a)} and {len(b)}"
        )

    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n = len(a)

    diffs = a - b
    delta = float(diffs.mean())

    rng = np.random.default_rng(seed)

    boot_deltas = np.empty(n_boot)
    for i in range(n_boot):
        # Resample PAIRED indices, so a[idx] and b[idx] always stay
        # matched together, preserving the pairing structure.
        sample_idx = rng.integers(0, n, size=n)
        boot_deltas[i] = diffs[sample_idx].mean()

    lower_pct = (alpha / 2) * 100
    upper_pct = (1 - alpha / 2) * 100

    lo = float(np.percentile(boot_deltas, lower_pct))
    hi = float(np.percentile(boot_deltas, upper_pct))

    return delta, lo, hi