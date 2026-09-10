"""Lab 6 - Step 5 helper: generate the NER model card."""

import json

import numpy as np
import torch
from jinja2 import Template

from seqeval.metrics import f1_score as seqeval_f1
from transformers import AutoTokenizer, AutoModelForTokenClassification

from bayan.evaluation.bootstrap import bootstrap_ci
from bayan.models.ner import align_labels


CONLL_PATH = "data/models/bayan_ner.conll"
CHECKPOINT_DIR = "artifacts/ner"
SEED = 42


def read_conll(path):
    sentences = []
    tokens, tags = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.strip() == "":
                if tokens:
                    sentences.append((tokens, tags))
                    tokens, tags = [], []
                continue
            parts = line.split("\t")
            if len(parts) != 2:
                parts = line.split()
                if len(parts) != 2:
                    continue
            word, tag = parts
            tokens.append(word)
            tags.append(tag)
    if tokens:
        sentences.append((tokens, tags))
    return sentences


def main():
    print("Loading NER model...")
    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT_DIR)
    model = AutoModelForTokenClassification.from_pretrained(CHECKPOINT_DIR)
    model.eval()

    id2label = model.config.id2label

    print("Loading and re-splitting CoNLL data (same seed as training)...")
    sentences = read_conll(CONLL_PATH)

    rng = np.random.default_rng(SEED)
    indices = np.arange(len(sentences))
    rng.shuffle(indices)

    n = len(indices)
    n_train = int(n * 0.8)
    n_val = int(n * 0.1)
    test_idx = indices[n_train + n_val:]

    test_sentences = [sentences[i] for i in test_idx]
    print(f"Test sentences: {len(test_sentences)}")

    # ------------------------------------------------------------
    # Run predictions and compute per-sentence correctness (whole
    # sentence's entity tags all correct == 1, else 0) for bootstrap
    # ------------------------------------------------------------

    per_sentence_correct = []
    all_true, all_pred = [], []

    for tokens, true_tags in test_sentences:
        encoded = tokenizer(
            tokens,
            truncation=True,
            is_split_into_words=True,
            return_tensors="pt",
        )
        word_ids = encoded.word_ids(batch_index=0)

        with torch.inference_mode():
            logits = model(**encoded).logits[0]
        pred_ids = logits.argmax(dim=-1).tolist()

        # Take only first-subword predictions, matching each original word
        pred_tags = []
        prev_word_id = None
        for wid, pid in zip(word_ids, pred_ids):
            if wid is None:
                continue
            if wid != prev_word_id:
                pred_tags.append(id2label[pid])
            prev_word_id = wid

        pred_tags = pred_tags[:len(true_tags)]
        while len(pred_tags) < len(true_tags):
            pred_tags.append("O")

        all_true.append(true_tags)
        all_pred.append(pred_tags)

        sentence_correct = int(pred_tags == true_tags)
        per_sentence_correct.append(sentence_correct)

    entity_f1 = seqeval_f1(all_true, all_pred)
    point, lo, hi = bootstrap_ci(per_sentence_correct)

    print(f"\nEntity-level F1: {entity_f1:.4f}")
    print(f"Sentence-exact-match accuracy: {point:.4f} 95% CI [{lo:.4f}, {hi:.4f}]")

    metrics_table = (
        "| Metric | Value |\n"
        "|---|---:|\n"
        f"| Entity-level F1 (seqeval) | {entity_f1:.4f} |\n"
        f"| Sentence exact-match accuracy | {point:.4f} [{lo:.4f}, {hi:.4f}] |\n"
    )

    # ------------------------------------------------------------
    # Render model card
    # ------------------------------------------------------------

    with open("templates/model_card.md.j2", encoding="utf-8") as f:
        template = Template(f.read())

    rendered = template.render(
        model_name="Bayan NER Model",
        intended_use=(
            "Extracts SERVICE, LOCATION, DATE, and REFERENCE entities "
            "from bilingual (Arabic/English) citizen feedback text."
        ),
        checkpoint=CHECKPOINT_DIR,
        preproc_version="bayan_ar_v1",
        data_version="bayan_ner.conll (synthetic)",
        metrics_table=metrics_table,
        slices_table="Not computed for this model (see EVALUATION_REPORT.md for topic classifier slices).",
        behavioural_table="Not run for this model.",
    )

    with open("model_card_ner.md", "w", encoding="utf-8") as f:
        f.write(rendered)

    print("\nModel card written to: model_card_ner.md")


if __name__ == "__main__":
    main()
