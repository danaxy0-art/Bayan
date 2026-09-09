"""Lab 3B starter: fine-tune token classification with correct alignment."""

import argparse
from pathlib import Path

import numpy as np
import torch

from seqeval.metrics import classification_report, f1_score

from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
    set_seed,
)

from datasets import Dataset, DatasetDict

from bayan.models.ner import align_labels


CHECKPOINT = "CAMeL-Lab/bert-base-arabic-camelbert-mix"

CONLL_PATH = "data/models/bayan_ner.conll"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default="artifacts/ner",
        help="Where to save the trained NER artefact (local path or mounted Drive path).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    return parser.parse_args()


def read_conll(path):
    """Parse a CoNLL-format file into a list of (tokens, tags) sentences."""

    sentences = []
    tokens = []
    tags = []

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")

            if line.strip() == "":
                if tokens:
                    sentences.append((tokens, tags))
                    tokens = []
                    tags = []
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


def build_label_list(sentences):
    """Collect the sorted, unique set of BIO tags seen in the data."""

    label_set = set()
    for _, tags in sentences:
        label_set.update(tags)

    labels = sorted(label_set - {"O"})
    return ["O"] + labels


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)

    print("CUDA available:", torch.cuda.is_available())

    sentences = read_conll(CONLL_PATH)
    print(f"Total sentences: {len(sentences)}")

    label_list = build_label_list(sentences)
    label2id = {label: i for i, label in enumerate(label_list)}
    id2label = {i: label for i, label in enumerate(label_list)}

    print("Labels:", label_list)

    rng = np.random.default_rng(args.seed)
    indices = np.arange(len(sentences))
    rng.shuffle(indices)

    n = len(indices)
    n_train = int(n * 0.8)
    n_val = int(n * 0.1)

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    def make_split(idx_list):
        return {
            "tokens": [sentences[i][0] for i in idx_list],
            "ner_tags": [
                [label2id[tag] for tag in sentences[i][1]]
                for i in idx_list
            ],
        }

    dataset = DatasetDict({
        "train": Dataset.from_dict(make_split(train_idx)),
        "validation": Dataset.from_dict(make_split(val_idx)),
        "test": Dataset.from_dict(make_split(test_idx)),
    })

    print(dataset)

    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)

    def tokenize_and_align(batch):
        tokenized = tokenizer(
            batch["tokens"],
            truncation=True,
            is_split_into_words=True,
        )

        all_labels = []
        for i, labels in enumerate(batch["ner_tags"]):
            word_ids = tokenized.word_ids(batch_index=i)
            aligned = align_labels(word_ids, labels)
            all_labels.append(aligned)

        tokenized["labels"] = all_labels
        return tokenized

    tokenized_dataset = dataset.map(
        tokenize_and_align,
        batched=True,
    )

    model = AutoModelForTokenClassification.from_pretrained(
        CHECKPOINT,
        num_labels=len(label_list),
        id2label=id2label,
        label2id=label2id,
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)

        true_predictions = []
        true_labels = []

        for pred_row, label_row in zip(predictions, labels):
            pred_seq = []
            label_seq = []
            for p, l in zip(pred_row, label_row):
                if l == -100:
                    continue
                pred_seq.append(id2label[p])
                label_seq.append(id2label[l])
            true_predictions.append(pred_seq)
            true_labels.append(label_seq)

        entity_f1 = f1_score(true_labels, true_predictions)

        return {
            "entity_f1": entity_f1,
        }

    training_args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),

        learning_rate=2e-5,

        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,

        num_train_epochs=4,

        weight_decay=0.01,

        fp16=torch.cuda.is_available(),

        eval_strategy="epoch",
        save_strategy="epoch",

        load_best_model_at_end=True,
        metric_for_best_model="entity_f1",
        greater_is_better=True,

        save_total_limit=2,
        save_safetensors=False,

        logging_steps=50,

        seed=args.seed,
        report_to="none",
    )

    data_collator = DataCollatorForTokenClassification(
        tokenizer=tokenizer
    )

    trainer = Trainer(
        model=model,
        args=training_args,

        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],

        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    test_results = trainer.evaluate(
        tokenized_dataset["test"],
        metric_key_prefix="test",
    )

    print("\nFINAL TEST RESULTS")
    print(test_results)

    predictions, labels, _ = trainer.predict(tokenized_dataset["test"])
    predictions = np.argmax(predictions, axis=-1)

    true_predictions = []
    true_labels = []
    for pred_row, label_row in zip(predictions, labels):
        pred_seq = []
        label_seq = []
        for p, l in zip(pred_row, label_row):
            if l == -100:
                continue
            pred_seq.append(id2label[p])
            label_seq.append(id2label[l])
        true_predictions.append(pred_seq)
        true_labels.append(label_seq)

    print("\nDetailed entity-level report:")
    print(classification_report(true_labels, true_predictions))

    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    print(f"\nSaved artefact to: {output_dir}")


if __name__ == "__main__":
    main()