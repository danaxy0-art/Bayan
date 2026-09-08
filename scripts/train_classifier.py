"""Lab 3A starter: fine-tune the Bayan topic classifier."""

import argparse
from pathlib import Path

import numpy as np
import torch

from sklearn.metrics import f1_score

# كل الأدوات اللي محتاجاها من مكتبة transformers لتدريب موديل BERT
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    set_seed,
)

# نفس دالة بناء الداتاست وقائمة المواضيع اللي جهزتها بملف data.py
from bayan.models.data import TOPICS, build_topic_dataset


# الموديل الأساسي (checkpoint) اللي رح أبني عليه — نفس اللي اخترته بـ Lab 1
CHECKPOINT = "CAMeL-Lab/bert-base-arabic-camelbert-mix"


def parse_args():
    # بستخدم argparse عشان أقدر أمرر إعدادات من سطر الأوامر
    # بدل ما أعدل بالكود كل مرة (زي مكان الحفظ أو الـ seed)
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output-dir",
        default="artifacts/topic_classifier",
        help="Where to save the trained classifier artefact.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    return parser.parse_args()


def compute_metrics(eval_pred):
    # هاي الدالة بيستدعيها Trainer تلقائياً بعد كل epoch
    # عشان يحسب أداء الموديل على بيانات التقييم
    logits, labels = eval_pred

    # نأخذ الفئة (topic) اللي عندها أعلى احتمال من مخرجات الموديل
    predictions = np.argmax(logits, axis=-1)

    # نحسب Macro-F1 (المطلوبة بالتقييم)
    macro_f1 = f1_score(
        labels,
        predictions,
        average="macro",
    )

    # نحسب الدقة العادية (accuracy) كمان كمقياس إضافي
    accuracy = (predictions == labels).mean()

    return {
        "macro_f1": macro_f1,
        "accuracy": accuracy,
    }


def main():
    # نقرأ الإعدادات (arguments) اللي أدخلناها من سطر الأوامر
    args = parse_args()

    # نجهز مجلد الحفظ، ونسويه لو مش موجود أصلاً
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # نثبت الـ seed عشان النتائج تكون قابلة للتكرار كل مرة أشغل الكود
    set_seed(args.seed)

    # نتأكد إذا عندي GPU متاح أو رح يشتغل على CPU بس
    print("CUDA available:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
    else:
        print("Training on CPU")

    # 1. نحمّل الـ tokenizer الخاص بنفس الموديل اللي اخترته بـ Lab 1
    tokenizer = AutoTokenizer.from_pretrained(
        CHECKPOINT
    )

    # 2. نبني الداتاست المقسّم (train/validation/test) اللي جهزته بـ Step 2
    dataset = build_topic_dataset(
        "data/raw/bayan_feedback.csv",
        seed=args.seed,
    )

    print(dataset)

    # 3. نحول كل نص لتوكنز يفهمها الموديل
    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=256,
        )

    # نطبق دالة التحويل هاي على كل صفوف الداتاست دفعة وحدة (batched)
    tokenized_dataset = dataset.map(
        tokenize,
        batched=True,
    )

    # 4. نحمّل الموديل المدرب مسبقاً ونضيفله رأس تصنيف (classification head)
    # مناسب لعدد المواضيع (topics) اللي عندي
    model = AutoModelForSequenceClassification.from_pretrained(
        CHECKPOINT,
        num_labels=len(TOPICS),
        id2label={
            i: topic
            for i, topic in enumerate(TOPICS)
        },
        label2id={
            topic: i
            for i, topic in enumerate(TOPICS)
        },
    )

    # 5. إعدادات التدريب
    training_args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),

        # معدل التعلم (learning rate) اللي رح يستخدمه الموديل
        learning_rate=2e-5,

        # حجم الدفعة (batch) بالتدريب والتقييم
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,

        # عدد مرات مرور الموديل على كامل بيانات التدريب
        num_train_epochs=4,

        warmup_ratio=0.1,
        weight_decay=0.01,

        # نفعّل fp16 بس إذا كان عندي GPU (يسرّع التدريب)
        fp16=torch.cuda.is_available(),

        # نقيّم ونحفظ نسخة من الموديل بعد كل epoch
        eval_strategy="epoch",
        save_strategy="epoch",

        # بالنهاية نرجع لأفضل نسخة حسب Macro-F1، مو آخر نسخة بالضرورة
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,

        # نحتفظ بآخر نسختين بس عشان ما ناخد مساحة كبيرة عالقرص
        save_total_limit=2,

        # عطلت safetensors عشان تجنب خطأ tensor غير متجاور صار يطلعلي
        save_safetensors=False,

        # كل 50 خطوة يطبع تقرير عن التقدم
        logging_steps=50,

        seed=args.seed,
        report_to="none",
    )

    # نجهز الـ data collator اللي بيتكفل بعمل padding تلقائي لكل batch
    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer
    )

    # 6. ننشئ كائن Trainer ونجمعله كل شي محتاجه: الموديل، الإعدادات، البيانات
    trainer = Trainer(
        model=model,
        args=training_args,

        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],

        data_collator=data_collator,

        compute_metrics=compute_metrics,

        # لو الأداء ما تحسن لمدة epoch-ين متتاليين، نوقف التدريب بدري
        # (early stopping) عشان ما نضيع وقت زيادة
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=2
            )
        ],
    )

    # 7. نبدأ التدريب الفعلي (fine-tuning)
    trainer.train()

    # 8. بعد ما يخلص التدريب، نقيّم الموديل على بيانات الـ test المجمّدة
    # (اللي ما شافها الموديل أبداً طول فترة التدريب)
    test_results = trainer.evaluate(
        tokenized_dataset["test"],
        metric_key_prefix="test",
    )

    print("\nFINAL TEST RESULTS")
    print(test_results)

    # 9. نحفظ النسخة النهائية من الموديل والـ tokenizer عشان أقدر أستخدمهم لاحقاً
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    print(f"\nSaved artefact to: {output_dir}")


if __name__ == "__main__":
    main()