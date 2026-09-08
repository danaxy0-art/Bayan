"""Lab 3 starter: dataset construction and split integrity."""


import pandas as pd

from datasets import Dataset, DatasetDict
from sklearn.model_selection import GroupShuffleSplit

from bayan.preprocessing.core import preprocess

# قائمة المواضيع (topics) الممكنة اللي بيتصنف عليها كل تعليق
TOPICS = [
    "roads",
    "lighting",
    "waste",
    "water",
    "billing",
    "digital_services",
    "licensing",
    "parks",
]


def build_topic_dataset(
    csv_path: str = "data/raw/bayan_feedback.csv",
    seed: int = 42,
) -> DatasetDict:
    # رقم ثابت للعشوائية (seed) عشان النتائج تكون قابلة للتكرار كل مرة

    # ------------------------------------------------------------
    # 1) تحميل البيانات
    # ------------------------------------------------------------

    df = pd.read_csv(csv_path)

    # ------------------------------------------------------------
    # 2) نفس التنظيف اللي سويته بـ Lab 1
    # ------------------------------------------------------------

    df["text"] = df["text"].astype(str).map(preprocess)

    # ------------------------------------------------------------
    # 3) تحويل اسم الموضوع (topic) لرقم (label) يفهمه الموديل
    # ------------------------------------------------------------

    # مثال:
    # roads -> 0
    # lighting -> 1
    # ...
    df["label"] = df["topic"].map(
        {topic: i for i, topic in enumerate(TOPICS)}
    )

    # ------------------------------------------------------------
    # 4) أول تقسيم (grouped split)
    # ------------------------------------------------------------

    # 80% -> بيانات تدريب مؤقتة (temporary train)
    # 20% -> بيانات اختبار مجمّدة (frozen test)
    #
    # التقسيم بيصير حسب citizen_group_id
    # (يعني كل مجموعة مواطنين تروح كاملة لطرف واحد بس،
    # مو مبعثرة بين train و test، عشان نتجنب data leakage)

    # GroupShuffleSplit هي أداة من scikit-learn تقسم البيانات
    # بس بشكل يحافظ على تجميع كل مجموعة (group) مع بعضها

    # n_splits يعني بس نبي تقسيمة وحدة بين train و test
    first_split = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=seed,
    )

    train_idx, test_idx = next(
        first_split.split(
            df,
            groups=df["citizen_group_id"],
        )
    )

    train_df = df.iloc[train_idx].copy()
    test_df = df.iloc[test_idx].copy()

    # ------------------------------------------------------------
    # 5) التقسيم الثاني (grouped split)
    # ------------------------------------------------------------

    # هلق بدي أفصل جزء validation من الـ 80% (train) اللي طلعت فوق
    #
    # test_size=0.125 من الـ 80% يعني تقريباً 10% من إجمالي البيانات الأصلية
    #
    # النتيجة النهائية:
    # Train      ≈ 70%
    # Validation ≈ 10%
    # Test       ≈ 20%

    second_split = GroupShuffleSplit(
        n_splits=1,
        test_size=0.125,
        random_state=seed,
    )

    final_train_idx, validation_idx = next(
        second_split.split(
            train_df,
            groups=train_df["citizen_group_id"],
        )
    )

    final_train_df = train_df.iloc[final_train_idx].copy()
    validation_df = train_df.iloc[validation_idx].copy()

    # ------------------------------------------------------------
    # 6) التأكد إنه ما في أي تداخل بين المجموعات
    # ------------------------------------------------------------

    # بتحقق إنه كل مجموعة مواطنين (citizen_group_id) موجودة بطرف واحد بس
    # (يعني ما تكرر نفس المجموعة بين train وvalidation وtest بنفس الوقت)

    train_groups = set(final_train_df["citizen_group_id"])
    validation_groups = set(validation_df["citizen_group_id"])
    test_groups = set(test_df["citizen_group_id"])

    assert train_groups.isdisjoint(validation_groups)
    assert train_groups.isdisjoint(test_groups)
    assert validation_groups.isdisjoint(test_groups)

    # ------------------------------------------------------------
    # 7) إرجاع النتيجة بصيغة Hugging Face DatasetDict
    # ------------------------------------------------------------

    return DatasetDict(
        train=Dataset.from_pandas(
            final_train_df,
            preserve_index=False,
        ),
        validation=Dataset.from_pandas(
            validation_df,
            preserve_index=False,
        ),
        test=Dataset.from_pandas(
            test_df,
            preserve_index=False,
        ),
    )


