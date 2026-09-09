"""Lab 6 starter: behavioural test generators/runners."""


# ------------------------------------------------------------
# 1) Invariance tests: swapping a value that should NOT change the
#    model's prediction (e.g. city name, reference number).
# ------------------------------------------------------------

INVARIANCE_TEMPLATES = [
    {
        "id": "INV-city-01",
        "base_text": "حفرة في الطريق أمام حي العليا منذ 5 أيام",
        "variant_text": "حفرة في الطريق أمام حي النرجس منذ 5 أيام",
        "note": "Swapping the neighbourhood name should not change the predicted topic.",
    },
    {
        "id": "INV-city-02",
        "base_text": "انقطاع المياه مستمر في جدة منذ الصباح",
        "variant_text": "انقطاع المياه مستمر في الرياض منذ الصباح",
        "note": "Swapping the city should not change the predicted topic.",
    },
    {
        "id": "INV-reference-01",
        "base_text": "لم تتغير حالة الطلب رقم BYN-2025-090006",
        "variant_text": "لم تتغير حالة الطلب رقم BYN-2025-090099",
        "note": "Swapping only the reference number should not change the predicted topic.",
    },
]


# ------------------------------------------------------------
# 2) Directional tests: changing something that SHOULD flip the
#    prediction in a predictable direction.
# ------------------------------------------------------------

DIRECTIONAL_TEMPLATES = [
    {
        "id": "DIR-topic-01",
        "base_text": "حفرة في الطريق أمام حي العليا منذ 5 أيام",
        "base_expected": "roads",
        "variant_text": "الشارع مظلم قرب حي العليا ونحتاج إصلاح الإنارة",
        "variant_expected": "lighting",
        "note": "Changing the complaint subject from roads to lighting should flip the predicted topic accordingly.",
    },
    {
        "id": "DIR-topic-02",
        "base_text": "دفعت الفاتورة لكن الحالة ما زالت غير مسددة",
        "base_expected": "billing",
        "variant_text": "نحتاج حاوية إضافية في حي الياسمين",
        "variant_expected": "waste",
        "note": "Changing the complaint subject from billing to waste should flip the predicted topic accordingly.",
    },
]


# ------------------------------------------------------------
# 3) Minimum Functionality Tests (MFT): simple, unambiguous cases
#    the model must get right.
# ------------------------------------------------------------

MFT_CASES = [
    {"id": "MFT-01", "text": "انقطاع المياه في حي النرجس", "expected": "water"},
    {"id": "MFT-02", "text": "عمود الإنارة معطل في الطريق", "expected": "lighting"},
    {"id": "MFT-03", "text": "حفرة كبيرة في الطريق السريع", "expected": "roads"},
    {"id": "MFT-04", "text": "الفاتورة غير صحيحة هذا الشهر", "expected": "billing"},
    {"id": "MFT-05", "text": "الحاوية ممتلئة ولم تُفرغ", "expected": "waste"},
    {"id": "MFT-06", "text": "طلب الترخيص ما زال قيد المراجعة", "expected": "licensing"},
    {"id": "MFT-07", "text": "الري متوقف في الحديقة العامة", "expected": "parks"},
    {"id": "MFT-08", "text": "التطبيق يتعطل عند رفع المستند", "expected": "digital_services"},
]


def run_behavioural_suite(predict_fn):
    """Run the invariance, directional, and MFT test suites against a
    given prediction function.

    predict_fn: callable(text: str) -> predicted_topic: str

    Returns a dict with per-suite pass/fail details and overall rates:
        {
            "invariance": {"pass_rate": float, "results": [...]},
            "directional": {"pass_rate": float, "results": [...]},
            "mft": {"pass_rate": float, "results": [...]},
        }
    """

    # ------------------------------------------------------------
    # Invariance: prediction must be IDENTICAL for base vs variant
    # ------------------------------------------------------------

    invariance_results = []
    for case in INVARIANCE_TEMPLATES:
        base_pred = predict_fn(case["base_text"])
        variant_pred = predict_fn(case["variant_text"])

        passed = base_pred == variant_pred

        invariance_results.append({
            "id": case["id"],
            "note": case["note"],
            "base_pred": base_pred,
            "variant_pred": variant_pred,
            "passed": passed,
        })

    invariance_pass_rate = (
        sum(r["passed"] for r in invariance_results) / len(invariance_results)
        if invariance_results else None
    )

    # ------------------------------------------------------------
    # Directional: base must predict base_expected, variant must
    # predict variant_expected (the prediction must "move" correctly)
    # ------------------------------------------------------------

    directional_results = []
    for case in DIRECTIONAL_TEMPLATES:
        base_pred = predict_fn(case["base_text"])
        variant_pred = predict_fn(case["variant_text"])

        passed = (
            base_pred == case["base_expected"]
            and variant_pred == case["variant_expected"]
        )

        directional_results.append({
            "id": case["id"],
            "note": case["note"],
            "base_pred": base_pred,
            "base_expected": case["base_expected"],
            "variant_pred": variant_pred,
            "variant_expected": case["variant_expected"],
            "passed": passed,
        })

    directional_pass_rate = (
        sum(r["passed"] for r in directional_results) / len(directional_results)
        if directional_results else None
    )

    # ------------------------------------------------------------
    # MFT: simple unambiguous cases, prediction must match expected
    # ------------------------------------------------------------

    mft_results = []
    for case in MFT_CASES:
        pred = predict_fn(case["text"])
        passed = pred == case["expected"]

        mft_results.append({
            "id": case["id"],
            "text": case["text"],
            "expected": case["expected"],
            "pred": pred,
            "passed": passed,
        })

    mft_pass_rate = (
        sum(r["passed"] for r in mft_results) / len(mft_results)
        if mft_results else None
    )

    return {
        "invariance": {
            "pass_rate": invariance_pass_rate,
            "results": invariance_results,
        },
        "directional": {
            "pass_rate": directional_pass_rate,
            "results": directional_results,
        },
        "mft": {
            "pass_rate": mft_pass_rate,
            "results": mft_results,
        },
    }