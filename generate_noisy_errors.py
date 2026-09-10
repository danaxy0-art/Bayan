"""Lab 6 - Step 4 helper: generate real model errors via input noise.

Since the topic classifier reaches 100% accuracy on the clean
synthetic test set (see EVALUATION_REPORT.md), we cannot hand-read
120 organic errors. Instead, we apply realistic input noise (typos,
word drops, truncation, code-mixing) to test texts, trying MULTIPLE
noise strategies and severities per text, and keep only the cases
where the model's prediction actually flips to something wrong. These
are genuine model errors on perturbed input, not fabricated labels.
"""

import random

import torch
import pandas as pd

from transformers import AutoTokenizer, AutoModelForSequenceClassification

from bayan.models.data import build_topic_dataset


DATA_PATH = "data/raw/bayan_feedback.csv"
CHECKPOINT_DIR = "artifacts/topic_classifier"
SEED = 42
TARGET_ERROR_COUNT = 120


def load_classifier(checkpoint_dir):
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint_dir)
    model.eval()
    return tokenizer, model


def make_predict_fn(tokenizer, model):
    id2label = model.config.id2label

    def predict_fn(text):
        encoded = tokenizer(
            text, return_tensors="pt", truncation=True, max_length=256
        )
        with torch.inference_mode():
            logits = model(**encoded).logits
        return id2label[int(logits.argmax(dim=-1).item())]

    return predict_fn


# ------------------------------------------------------------
# Noise functions - each returns a perturbed copy of the text.
# Several severity levels are tried per text, since the classifier
# is very robust to light perturbation on this templated data.
# ------------------------------------------------------------

def noise_drop_words(text, rng, frac):
    words = text.split()
    if len(words) <= 2:
        return text
    n_drop = max(1, int(len(words) * frac))
    n_drop = min(n_drop, len(words) - 1)
    drop_idx = set(rng.sample(range(len(words)), n_drop))
    kept = [w for i, w in enumerate(words) if i not in drop_idx]
    return " ".join(kept) if kept else text


def noise_char_typo(text, rng, n_typos):
    chars = list(text)
    for _ in range(min(n_typos, max(0, len(chars) - 1))):
        idx = rng.randrange(len(chars) - 1)
        chars[idx], chars[idx + 1] = chars[idx + 1], chars[idx]
    return "".join(chars)


def noise_truncate(text, rng, keep_frac):
    words = text.split()
    if len(words) <= 1:
        return text
    cut = max(1, int(len(words) * keep_frac))
    return " ".join(words[:cut])


def noise_keyword_mask(text, rng):
    """Replace a random word with a generic placeholder, simulating a
    garbled/OCR'd or heavily abbreviated keyword.
    """
    words = text.split()
    if len(words) <= 1:
        return text
    idx = rng.randrange(len(words))
    words[idx] = "***"
    return " ".join(words)


def noise_shuffle_words(text, rng):
    words = text.split()
    if len(words) <= 2:
        return text
    rng.shuffle(words)
    return " ".join(words)


# Each entry: (function, list of severity args to try, in increasing strength)
NOISE_STRATEGIES = [
    (noise_truncate, [0.6, 0.4, 0.25, 0.15]),
    (noise_drop_words, [0.3, 0.5, 0.7]),
    (noise_char_typo, [2, 4, 6]),
    (noise_keyword_mask, [None]),
    (noise_shuffle_words, [None]),
]


def try_all_noise_levels(text, rng, predict_fn, true_topic):
    """Try every strategy/severity combo (in a shuffled order) until
    one flips the prediction to something wrong. Returns
    (noisy_text, noise_tag, wrong_pred) or None if nothing worked.
    """

    combos = []
    for fn, severities in NOISE_STRATEGIES:
        for sev in severities:
            combos.append((fn, sev))
    rng.shuffle(combos)

    for fn, sev in combos:
        if sev is None:
            noisy_text = fn(text, rng)
            tag = fn.__name__
        else:
            noisy_text = fn(text, rng, sev)
            tag = f"{fn.__name__}(severity={sev})"

        if not noisy_text.strip():
            continue

        pred = predict_fn(noisy_text)
        if pred != true_topic:
            return noisy_text, tag, pred

    return None


def main():
    rng = random.Random(SEED)

    print("Loading topic classifier...")
    tokenizer, model = load_classifier(CHECKPOINT_DIR)
    predict_fn = make_predict_fn(tokenizer, model)

    print("Loading frozen validation split...")
    dataset = build_topic_dataset(DATA_PATH, seed=SEED)
    val_df = dataset["validation"].to_pandas()
    print(f"Validation rows available: {len(val_df)}")

    val_df = val_df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)

    collected_errors = []

    for i, row in val_df.iterrows():
        if len(collected_errors) >= TARGET_ERROR_COUNT:
            break

        original_text = str(row["text"])
        true_topic = row["topic"]

        result = try_all_noise_levels(original_text, rng, predict_fn, true_topic)

        if result is not None:
            noisy_text, noise_tag, wrong_pred = result
            collected_errors.append({
                "feedback_id": row["feedback_id"],
                "lang": row["lang"],
                "dialect_region": row["dialect_region"],
                "true_topic": true_topic,
                "original_text": original_text,
                "noisy_text": noisy_text,
                "noise_type": noise_tag,
                "predicted_topic": wrong_pred,
            })

        if (i + 1) % 200 == 0:
            print(f"  ...scanned {i + 1} rows, {len(collected_errors)} errors so far")

    print(f"\nCollected {len(collected_errors)} real model errors "
          f"(target was {TARGET_ERROR_COUNT})")

    errors_df = pd.DataFrame(collected_errors)
    errors_df.to_csv(
        "artifacts/lab6_sampled_errors.csv", index=False, encoding="utf-8-sig"
    )
    print("Saved to: artifacts/lab6_sampled_errors.csv")

    print("\nBreakdown by noise type:")
    print(errors_df["noise_type"].value_counts())

    print("\nBreakdown by true_topic -> predicted_topic confusion (top 10):")
    confusion = errors_df.groupby(["true_topic", "predicted_topic"]).size()
    print(confusion.sort_values(ascending=False).head(10))


if __name__ == "__main__":
    main()