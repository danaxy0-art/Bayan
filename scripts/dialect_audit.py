"""Lab 4 starter: audit dialect mix and record the implication in NOTES.md."""

import pandas as pd


DATA_PATH = "data/raw/bayan_feedback.csv"


def main():
    df = pd.read_csv(DATA_PATH)

    print(f"Total rows: {len(df)}")

    # ------------------------------------------------------------
    # Overall dialect_region distribution
    # ------------------------------------------------------------

    print("\nDialect region distribution (all rows):")
    print(df["dialect_region"].value_counts())

    print("\nDialect region distribution (percentage):")
    print(
        (df["dialect_region"].value_counts(normalize=True) * 100)
        .round(2)
    )

    # ------------------------------------------------------------
    # Restrict to the Arabic slice only (lang == "ar")
    # ------------------------------------------------------------

    arabic_df = df[df["lang"] == "ar"]

    print(f"\nArabic-language rows: {len(arabic_df)}")

    print("\nDialect region distribution (Arabic rows only):")
    print(arabic_df["dialect_region"].value_counts())

    print("\nDialect region distribution, Arabic only (percentage):")
    print(
        (arabic_df["dialect_region"].value_counts(normalize=True) * 100)
        .round(2)
    )

    # ------------------------------------------------------------
    # Implication of evaluating only on MSA
    # ------------------------------------------------------------

    msa_count = (arabic_df["dialect_region"] == "MSA").sum()
    gulf_count = (arabic_df["dialect_region"] == "Gulf").sum()
    msa_share = msa_count / len(arabic_df) * 100

    print(
        "\nImplication: MSA makes up "
        f"{msa_share:.1f}% of Arabic rows "
        f"({msa_count} MSA vs {gulf_count} Gulf). "
        "Evaluating only on MSA would leave the majority of Arabic "
        "dialectal traffic (Gulf) unmeasured, risking an inflated "
        "sense of Arabic performance that does not reflect real "
        "citizen feedback, most of which is dialectal, not MSA."
    )


if __name__ == "__main__":
    main()