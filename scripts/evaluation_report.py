"""Lab 6 starter: generate EVALUATION_REPORT.md + model-card evidence."""

import numpy as np
import torch
import pandas as pd
from jinja2 import Template

from transformers import AutoTokenizer, AutoModelForSequenceClassification

from bayan.evaluation.bootstrap import bootstrap_ci
from bayan.evaluation.slices import sliced_report, format_slices_table
from bayan.evaluation.behavioural import run_behavioural_suite
from bayan.models.data import TOPICS, build_topic_dataset


DATA_PATH = "data/raw/bayan_feedback.csv"
CHECKPOINT_DIR = "artifacts/topic_classifier"
SEED = 42


def load_classifier(checkpoint_dir):
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint_dir)
    model.eval()
    return tokenizer, model


def make_predict_fn(tokenizer, model):
    """Wrap the model into a simple text -> topic_label callable, for
    use by both the sliced report and the behavioural test suite.
    """

    id2label = model.config.id2label

    def predict_fn(text):
        encoded = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=256,
        )
        with torch.inference_mode():
            logits = model(**encoded).logits
        pred_id = int(logits.argmax(dim=-1).item())
        return id2label[pred_id]

    return predict_fn


def run_predictions_on_test(predict_fn, test_df):
    """Run predict_fn over every row of the frozen test split, adding
    a `correct` (0/1) column comparing prediction to the true topic.
    """

    preds = [predict_fn(text) for text in test_df["text"]]

    df = test_df.copy()
    df["prediction"] = preds
    df["correct"] = (df["prediction"] == df["topic"]).astype(int)

    return df


def render_model_card(
    model_name,
    intended_use,
    checkpoint,
    preproc_version,
    data_version,
    metrics_table,
    slices_table,
    behavioural_table,
    output_path,
):
    with open("templates/model_card.md.j2", encoding="utf-8") as f:
        template = Template(f.read())

    rendered = template.render(
        model_name=model_name,
        intended_use=intended_use,
        checkpoint=checkpoint,
        preproc_version=preproc_version,
        data_version=data_version,
        metrics_table=metrics_table,
        slices_table=slices_table,
        behavioural_table=behavioural_table,
    )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(rendered)

    print(f"Model card written to: {output_path}")


def format_behavioural_table(behavioural_results):
    lines = [
        "| Suite | Pass rate |",
        "|---|---:|",
    ]
    for suite_name, suite in behavioural_results.items():
        rate = suite["pass_rate"]
        rate_str = f"{rate:.2%}" if rate is not None else "N/A"
        lines.append(f"| {suite_name} | {rate_str} |")
    return "\n".join(lines)


def main():
    print("Loading topic classifier...")
    tokenizer, model = load_classifier(CHECKPOINT_DIR)
    predict_fn = make_predict_fn(tokenizer, model)

    print("Loading frozen test split...")
    dataset = build_topic_dataset(DATA_PATH, seed=SEED)
    test_df = dataset["test"].to_pandas()

    print(f"Running predictions on {len(test_df)} test rows...")
    results_df = run_predictions_on_test(predict_fn, test_df)

    # ------------------------------------------------------------
    # Overall bootstrap CI on accuracy
    # ------------------------------------------------------------

    point, lo, hi = bootstrap_ci(results_df["correct"].tolist())
    print(f"\nOverall accuracy: {point:.4f}  95% CI [{lo:.4f}, {hi:.4f}]")

    metrics_table = (
        "| Metric | Value | 95% CI |\n"
        "|---|---:|---|\n"
        f"| Accuracy | {point:.4f} | [{lo:.4f}, {hi:.4f}] |\n"
    )

    # ------------------------------------------------------------
    # Sliced report
    # ------------------------------------------------------------

    print("\nComputing sliced report...")
    slice_rows = sliced_report(results_df)
    slices_table = format_slices_table(slice_rows)
    n_slices = len(slice_rows)
    print(f"Computed {n_slices} slices")

    # ------------------------------------------------------------
    # Behavioural test suite
    # ------------------------------------------------------------

    print("\nRunning behavioural test suite...")
    behavioural_results = run_behavioural_suite(predict_fn)
    behavioural_table = format_behavioural_table(behavioural_results)

    for suite_name, suite in behavioural_results.items():
        rate = suite["pass_rate"]
        print(f"  {suite_name}: {rate:.2%}" if rate is not None else f"  {suite_name}: N/A")

    # ------------------------------------------------------------
    # Write EVALUATION_REPORT.md
    # ------------------------------------------------------------

    with open("EVALUATION_REPORT.md", "a", encoding="utf-8") as f:
        f.write("\n## Topic Classifier Evaluation\n\n")
        f.write("### Overall metrics\n\n")
        f.write(metrics_table)
        f.write(f"\n### Sliced report ({n_slices} slices)\n\n")
        f.write(slices_table)
        f.write("\n\n### Behavioural tests\n\n")
        f.write(behavioural_table)
        f.write("\n")

    print("\nAppended results to EVALUATION_REPORT.md")

    # ------------------------------------------------------------
    # Render model card
    # ------------------------------------------------------------

    render_model_card(
        model_name="Bayan Topic Classifier",
        intended_use=(
            "Classifies bilingual (Arabic/English) citizen feedback "
            "text into one of 8 municipal service topics."
        ),
        checkpoint=CHECKPOINT_DIR,
        preproc_version="bayan_ar_v1",
        data_version="bayan_feedback.csv (synthetic)",
        metrics_table=metrics_table,
        slices_table=slices_table,
        behavioural_table=behavioural_table,
        output_path="model_card_topic_classifier.md",
    )


if __name__ == "__main__":
    main()