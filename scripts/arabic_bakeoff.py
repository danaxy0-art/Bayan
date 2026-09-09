"""Lab 4 starter: compare Arabic-centric checkpoints by all/Gulf/MSA slices."""

import numpy as np
import torch

from sklearn.metrics import f1_score

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)

from bayan.models.data import TOPICS, build_topic_dataset


SEED = 42

DATA_PATH = "data/raw/bayan_feedback.csv"

# Candidate Arabic-centric checkpoints to compare.
# MARBERT included as an optional third candidate.
CHECKPOINTS = {
    "CAMeLBERT-mix": "CAMeL-Lab/bert-base-arabic-camelbert-mix",
    "CAMeLBERT-DA": "CAMeL-Lab/bert-base-arabic-camelbert-da",
    "MARBERT": "UBC-NLP/MARBERT",
}


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    macro_f1 = f1_score(labels, predictions, average="macro")
    return {"macro_f1": macro_f1}


def slice_macro_f1(trainer, tokenized_test, dialect_regions, target_region):
    """Compute macro-F1 on the subset of the test set matching a dialect region.

    target_region: one of "Gulf", "MSA", or None (meaning: no filter, all rows).
    """

    if target_region is None:
        subset = tokenized_test
    else:
        mask = [
            region == target_region
            for region in dialect_regions
        ]
        indices = [i for i, keep in enumerate(mask) if keep]
        subset = tokenized_test.select(indices)

    if len(subset) == 0:
        return None, 0

    predictions = trainer.predict(subset)
    logits = predictions.predictions
    labels = predictions.label_ids
    preds = np.argmax(logits, axis=-1)

    macro_f1 = f1_score(labels, preds, average="macro")
    return macro_f1, len(subset)


def run_bake_off_for_checkpoint(name, checkpoint, dataset):
    print(f"\n{'=' * 60}")
    print(f"Bake-off: {name} ({checkpoint})")
    print("=" * 60)

    set_seed(SEED)

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=256,
        )

    tokenized_dataset = dataset.map(
        tokenize,
        batched=True,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint,
        num_labels=len(TOPICS),
        id2label={i: topic for i, topic in enumerate(TOPICS)},
        label2id={topic: i for i, topic in enumerate(TOPICS)},
    )

    training_args = TrainingArguments(
        output_dir=f"artifacts/arabic_bakeoff/{name}",

        learning_rate=2e-5,

        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,

        # Reduced epochs: this is a comparative bake-off, not a final
        # production model, so we favour a faster head-to-head comparison.
        num_train_epochs=2,

        weight_decay=0.01,

        fp16=torch.cuda.is_available(),

        eval_strategy="epoch",
        save_strategy="no",

        logging_steps=50,

        seed=SEED,
        report_to="none",
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    # ------------------------------------------------------------
    # Evaluate on the frozen test set, sliced by dialect region
    # ------------------------------------------------------------

    test_dialect_regions = list(dataset["test"]["dialect_region"])

    results = {}

    for slice_name in ["all", "Gulf", "MSA"]:
        region_filter = None if slice_name == "all" else slice_name
        f1, n = slice_macro_f1(
            trainer,
            tokenized_dataset["test"],
            test_dialect_regions,
            region_filter,
        )
        results[slice_name] = f1
        print(f"  {slice_name:6s} slice (n={n:4d}): macro-F1 = {f1}")

    return results


def main():
    dataset = build_topic_dataset(DATA_PATH, seed=SEED)

    all_results = {}

    for name, checkpoint in CHECKPOINTS.items():
        try:
            results = run_bake_off_for_checkpoint(name, checkpoint, dataset)
            all_results[name] = results
        except Exception as exc:
            print(f"\nSkipping {name} ({checkpoint}): {exc}")
            all_results[name] = {"all": None, "Gulf": None, "MSA": None}

    # ------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------

    print("\n" + "=" * 60)
    print("SUMMARY: Arabic model bake-off (macro-F1 by slice)")
    print("=" * 60)
    print(f"{'Model':15s} {'All':>10s} {'Gulf':>10s} {'MSA':>10s}")

    for name, results in all_results.items():
        row = f"{name:15s}"
        for slice_name in ["all", "Gulf", "MSA"]:
            val = results.get(slice_name)
            row += f" {val:10.4f}" if val is not None else f" {'N/A':>10s}"
        print(row)

    # ------------------------------------------------------------
    # Append to BENCHMARKS.md
    # ------------------------------------------------------------

    with open("BENCHMARKS.md", "a", encoding="utf-8") as f:
        f.write("\n## Lab 4 - Arabic Model Bake-off\n\n")
        f.write("| Model | All macro-F1 | Gulf macro-F1 | MSA macro-F1 |\n")
        f.write("|---|---:|---:|---:|\n")
        for name, results in all_results.items():
            def fmt(v):
                return f"{v:.4f}" if v is not None else "N/A"
            f.write(
                f"| {name} | {fmt(results.get('all'))} | "
                f"{fmt(results.get('Gulf'))} | {fmt(results.get('MSA'))} |\n"
            )


if __name__ == "__main__":
    main()