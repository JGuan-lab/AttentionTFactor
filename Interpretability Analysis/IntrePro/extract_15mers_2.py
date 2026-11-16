import pandas as pd

INPUT_FILE = "top_15mers_human_AttentionTFactor.csv"
OUTPUT_FILE = "highest_15mer_per_protein_human_AttentionTFactor.csv"


def extract_highest_fragment_per_protein():
    df = pd.read_csv(INPUT_FILE)

    required_columns = {"ProteinID", "DeltaLogitMean", "Peptide15mer", "Start", "End"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Input table is missing required columns: {sorted(missing)}")

    # Force the score column to numeric, then drop unscoreable rows.
    df["DeltaLogitMean"] = pd.to_numeric(df["DeltaLogitMean"], errors="coerce")
    df = df.dropna(subset=["ProteinID", "DeltaLogitMean"])

    # Sort per protein by descending score, keep only the top row per ProteinID.
    highest_df = (
        df.sort_values(by=["ProteinID", "DeltaLogitMean"], ascending=[True, False])
        .groupby("ProteinID", as_index=False)
        .head(1)
        .copy()
    )
    highest_df["Rank"] = 1

    # Global descending sort across proteins.
    highest_df = (
        highest_df.sort_values(by="DeltaLogitMean", ascending=False)
        .reset_index(drop=True)
    )

    # Add global rank.
    highest_df.insert(0, "GlobalRank", range(1, len(highest_df) + 1))

    highest_df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    print("Processing complete")
    print(f"Input fragments: {len(df)}")
    print(f"Proteins: {df['ProteinID'].nunique()}")
    print(f"Final retained fragments: {len(highest_df)}")
    print(f"Output file: {OUTPUT_FILE}")

    print("\nTop 10 fragments by global score:")
    print(highest_df[["GlobalRank", "ProteinID", "Start", "End", "Peptide15mer",
                      "DeltaLogitMean", "DeltaLogitSD", "OriginalProbMean"]]
          .head(10).to_string(index=False))


if __name__ == "__main__":
    extract_highest_fragment_per_protein()