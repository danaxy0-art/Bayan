"""Lab 3B starter: run the 12-question QA smoke set."""

import json

import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

from bayan.models.qa import best_span


# ⚠️ لم يُذكر checkpoint محدد بـ docs/LABS.md لهذه المهمة.
# بما إنه بيانات qa_smoke_set.json بالإنجليزية بالكامل (نمط SQuAD)،
# استخدمنا موديل QA إنجليزي عام مدرب مسبقاً على SQuAD.
CHECKPOINT = "distilbert-base-cased-distilled-squad"


def main():
    with open("data/eval/qa_smoke_set.json", encoding="utf-8") as f:
        smoke_data = json.load(f)

    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)
    model = AutoModelForQuestionAnswering.from_pretrained(CHECKPOINT).eval()

    correct_answerable = 0
    total_answerable = 0
    correct_null = 0
    total_null = 0

    for entry in smoke_data["data"]:
        for paragraph in entry["paragraphs"]:
            context = paragraph["context"]

            for qa in paragraph["qas"]:
                question = qa["question"]
                is_impossible = qa["is_impossible"]

                encoded = tokenizer(
                    question,
                    context,
                    return_tensors="pt",
                    return_offsets_mapping=True,
                    truncation=True,
                )

                offset_mapping = encoded.pop("offset_mapping")[0].tolist()
                offsets = [
                    tuple(o) if o != [0, 0] else None
                    for o in offset_mapping
                ]

                with torch.inference_mode():
                    outputs = model(**encoded)

                start_logits = outputs.start_logits[0].numpy()
                end_logits = outputs.end_logits[0].numpy()

                # null_score: confidence that token 0 ([CLS]) is the answer
                null_score = float(start_logits[0] + end_logits[0])

                result = best_span(
                    start_logits,
                    end_logits,
                    offsets,
                    null_score=null_score,
                    null_threshold=1.0,
                )

                predicted_answer = result.get("answer")

                if is_impossible:
                    total_null += 1
                    if predicted_answer is None:
                        correct_null += 1
                else:
                    total_answerable += 1
                    if predicted_answer is not None:
                        start_char, end_char = predicted_answer
                        predicted_text = context[start_char:end_char]
                        expected_text = qa["answers"][0]["text"]
                        if predicted_text.strip() == expected_text.strip():
                            correct_answerable += 1

    print(
        f"Answerable: {correct_answerable}/{total_answerable} correct spans"
    )
    print(f"Unanswerable: {correct_null}/{total_null} correctly null")


if __name__ == "__main__":
    main()