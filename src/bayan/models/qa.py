"""Lab 3B QA post-processing contracts."""

import numpy as np


def best_span(
    start_logits,
    end_logits,
    offsets,
    *,
    null_score,
    null_threshold,
    max_answer_len=30,
    top_k=20,
):
    """Select the best answer span from start/end logits.

    Returns a dict with key "answer":
        - a (start_char, end_char) tuple if a valid, confident span exists
        - None if no valid span exists, or if the model is more confident
          there's no answer (null_score) than in the best span found
    """

    starts = np.argsort(start_logits)[-top_k:]
    ends = np.argsort(end_logits)[-top_k:]

    best_score = float("-inf")
    best_start = None
    best_end = None

    for s in starts:
        for e in ends:

            # Skip positions with no character offset (special tokens,
            # question tokens, etc. — they can't map back to the context)
            if offsets[s] is None or offsets[e] is None:
                continue

            # End can't come before start
            if e < s:
                continue

            # Reject spans longer than the allowed answer length
            if e - s + 1 > max_answer_len:
                continue

            score = float(start_logits[s] + end_logits[e])

            if score > best_score:
                best_score = score
                best_start = s
                best_end = e

    # No valid span was found at all
    if best_start is None:
        return {
            "answer": None,
            "reason": "no_valid_span",
        }

    # The model is more confident there's no answer than in this span
    if null_score - best_score > null_threshold:
        return {
            "answer": None,
            "reason": "no_answer_in_context",
            "margin": float(null_score - best_score),
        }

    start_char = offsets[best_start][0]
    end_char = offsets[best_end][1]

    return {
        "answer": (start_char, end_char),
        "start": start_char,
        "end": end_char,
        "score": best_score,
    }