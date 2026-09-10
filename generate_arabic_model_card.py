"""Lab 6 - Step 5 helper: generate the Arabic dialect-aware model card,
using the measured results already recorded from the Lab 4 bake-off
(see BENCHMARKS.md), rather than re-training the model from scratch.
"""

from jinja2 import Template


def main():
    metrics_table = (
        "| Metric | All | Gulf | MSA |\n"
        "|---|---:|---:|---:|\n"
        "| macro-F1 | 0.9992 | 1.0000 | 1.0000 |\n"
    )

    slices_table = (
        "| Slice type | Slice value | macro-F1 |\n"
        "|---|---|---:|\n"
        "| dialect | all | 0.9992 |\n"
        "| dialect | Gulf | 1.0000 |\n"
        "| dialect | MSA | 1.0000 |\n"
        "\n"
        "(See BENCHMARKS.md, Lab 4 Arabic Model Bake-off, for the full "
        "comparison against CAMeLBERT-mix and MARBERT.)"
    )

    behavioural_table = "Not run for this model (behavioural suite was run on the topic classifier only; see EVALUATION_REPORT.md)."

    with open("templates/model_card.md.j2", encoding="utf-8") as f:
        template = Template(f.read())

    rendered = template.render(
        model_name="Bayan Arabic Dialect-Aware Classifier (CAMeLBERT-DA)",
        intended_use=(
            "Topic classification for Gulf-dialect and MSA Arabic "
            "citizen feedback, evaluated as a dialect-aware alternative "
            "to the general-purpose CAMeLBERT-mix checkpoint used in "
            "production (see DECISIONS.md#arabic-model)."
        ),
        checkpoint="CAMeL-Lab/bert-base-arabic-camelbert-da (fine-tuned, 2 epochs, Lab 4 bake-off)",
        preproc_version="bayan_ar_v1",
        data_version="bayan_feedback.csv (synthetic), frozen test split",
        metrics_table=metrics_table,
        slices_table=slices_table,
        behavioural_table=behavioural_table,
    )

    with open("model_card_arabic_da.md", "w", encoding="utf-8") as f:
        f.write(rendered)

    print("Model card written to: model_card_arabic_da.md")


if __name__ == "__main__":
    main()
