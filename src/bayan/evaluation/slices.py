"""Lab 6 starter: sliced evaluation report."""

import pandas as pd

from bayan.evaluation.bootstrap import bootstrap_ci


MIN_SLICE_SIZE = 30


def _length_bucket(text: str) -> str:
    """Bucket a text by character length into short/medium/long."""

    length = len(str(text))

    if length < 40:
        return "short (<40 chars)"
    elif length < 100:
        return "medium (40-100 chars)"
    else:
        return "long (>100 chars)"


def sliced_report(
    df: pd.DataFrame,
    correct_col: str = "correct",
    lang_col: str = "lang",
    dialect_col: str = "dialect_region",
    class_col: str = "topic",
    text_col: str = "text",
    min_slice_size: int = MIN_SLICE_SIZE,
):
    """Compute a per-slice accuracy report with bootstrap confidence
    intervals, sliced by language, dialect, class (topic), and text
    length bucket.

    df must contain a boolean/0-1 `correct_col` indicating whether the
    model's prediction was correct for that row, plus the slicing
    columns.

    Returns a list of dicts, one per slice, each with:
      slice_type, slice_value, n, accuracy, ci_lo, ci_hi, flagged
    (flagged=True when n < min_slice_size, meaning the CI is unreliable).
    """

    if correct_col not in df.columns:
        raise ValueError(f"df is missing required column: {correct_col}")

    rows = []

    slice_definitions = [
        ("language", lang_col),
        ("dialect", dialect_col),
        ("class", class_col),
    ]

    for slice_type, col in slice_definitions:
        if col not in df.columns:
            continue

        for value, group in df.groupby(col, dropna=True):
            n = len(group)
            if n == 0:
                continue

            point, lo, hi = bootstrap_ci(group[correct_col].tolist())

            rows.append({
                "slice_type": slice_type,
                "slice_value": str(value),
                "n": n,
                "accuracy": point,
                "ci_lo": lo,
                "ci_hi": hi,
                "flagged": n < min_slice_size,
            })

    # Length-bucket slice, derived from text_col
    if text_col in df.columns:
        length_buckets = df[text_col].map(_length_bucket)

        for value, group in df.groupby(length_buckets, dropna=True):
            n = len(group)
            if n == 0:
                continue

            point, lo, hi = bootstrap_ci(group[correct_col].tolist())

            rows.append({
                "slice_type": "length",
                "slice_value": str(value),
                "n": n,
                "accuracy": point,
                "ci_lo": lo,
                "ci_hi": hi,
                "flagged": n < min_slice_size,
            })

    return rows


def format_slices_table(rows) -> str:
    """Render sliced_report() output as a Markdown table."""

    lines = [
        "| Slice type | Slice value | n | Accuracy | 95% CI | Flag |",
        "|---|---|---:|---:|---|---|",
    ]

    for r in rows:
        flag = "⚠️ small n" if r["flagged"] else ""
        lines.append(
            f"| {r['slice_type']} | {r['slice_value']} | {r['n']} | "
            f"{r['accuracy']:.4f} | "
            f"[{r['ci_lo']:.4f}, {r['ci_hi']:.4f}] | {flag} |"
        )

    return "\n".join(lines)