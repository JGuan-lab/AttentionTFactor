#!/usr/bin/env python3
"""Filter the InterPro batch-annotated CSV down to the six-column paper Table 3.1.

Usage:
    python filter_interpro_to_paper_table.py
    python filter_interpro_to_paper_table.py --input interpro_dbd_results.csv --output interpro_dbd_paper_table.csv
    python filter_interpro_to_paper_table.py --include-smart

By default only InterPro-sourced records with a valid DBD interval are kept;
--include-smart also keeps valid DBD annotations from SMART and other sources.
Records with a DBD but no 15-mer overlap (0/15) are NOT removed.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


PAPER_COLUMNS = ["ProteinID", "Model_15mer", "Peptide_Position",
                 "InterPro_Domain", "Domain_Position", "Overlap"]
REQUIRED_COLUMNS = [
    "ProteinID", "Peptide15mer", "Start", "End", "SelectedDBD",
    "DBDStart", "DBDEnd", "DBDSource",
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Drop proteins without a valid DBD annotation and emit the exact six-column paper Table 3.1.")
    parser.add_argument(
        "--input", type=Path,
        default=Path(__file__).resolve().parent / "interpro_dbd_results_human_attentiontfactor.csv",
        help="CSV previously produced by online annotation",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Output CSV path; defaults to interpro_dbd_paper_table_human_attentiontfactor.csv next to the input",
    )
    parser.add_argument(
        "--include-smart", action="store_true",
        help="Also keep DBD annotations from SMART and other non-InterPro sources "
             "(column is still named 'InterPro_Domain')",
    )
    args = parser.parse_args()

    input_path = args.input.expanduser().resolve()
    output_path = (
        args.output.expanduser().resolve() if args.output is not None
        else input_path.with_name("interpro_dbd_paper_table_human_attentiontfactor.csv")
    )
    if not input_path.is_file():
        parser.error(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path, encoding="utf-8-sig", dtype={"ProteinID": str})
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        parser.error(f"Input file is missing required columns: {', '.join(missing)}")

    # Preserve contribution ranking order; if GlobalRank exists, sort ascending.
    if "GlobalRank" in df.columns:
        df = df.sort_values("GlobalRank", kind="stable").copy()

    dbd_name = df["SelectedDBD"].fillna("").astype(str).str.strip()
    dbd_source = df["DBDSource"].fillna("").astype(str).str.strip().str.lower()
    protein_id = df["ProteinID"].fillna("").astype(str).str.strip()

    # Coordinates are 1-based, inclusive, and must satisfy start <= end.
    numeric = {col: pd.to_numeric(df[col], errors="coerce")
               for col in ("Start", "End", "DBDStart", "DBDEnd")}
    peptide_ok = (numeric["Start"] >= 1) & (numeric["End"] >= numeric["Start"])
    domain_ok = (
        (dbd_name != "") & (dbd_source != "")
        & (numeric["DBDStart"] >= 1) & (numeric["DBDEnd"] >= numeric["DBDStart"]))
    annotated = protein_id.ne("") & peptide_ok & domain_ok
    source_ok = (pd.Series(True, index=df.index) if args.include_smart
                 else dbd_source.eq("interpro"))
    keep = annotated & source_ok

    chosen = df.loc[keep].copy()
    starts = numeric["Start"].loc[keep].astype(int)
    ends = numeric["End"].loc[keep].astype(int)
    dbd_starts = numeric["DBDStart"].loc[keep].astype(int)
    dbd_ends = numeric["DBDEnd"].loc[keep].astype(int)
    peptide = chosen["Peptide15mer"].fillna("").astype(str).str.strip()

    # Do not guess or auto-correct coordinates: raise if the peptide length disagrees.
    expected_length = (ends - starts + 1).to_numpy()
    observed_length = peptide.str.len().to_numpy()
    if not np.array_equal(expected_length, observed_length):
        bad = chosen.loc[expected_length != observed_length, "ProteinID"].head(10).tolist()
        raise ValueError(f"Peptide position and peptide length mismatch, example proteins: {bad}")

    overlap = np.maximum(
        0,
        np.minimum(ends.to_numpy(), dbd_ends.to_numpy())
        - np.maximum(starts.to_numpy(), dbd_starts.to_numpy()) + 1,
    ).astype(int)

    # Never use `+` on numpy string arrays (raises UFuncNoLoopError);
    # build strings element-wise with f-strings, safe across pandas/numpy versions.
    frag_positions = [f"{s}-{e}" for s, e in zip(starts.to_numpy(), ends.to_numpy())]
    domain_positions = [f"{s}-{e}" for s, e in zip(dbd_starts.to_numpy(), dbd_ends.to_numpy())]
    overlap_strs = [f"{o}/{n}" for o, n in zip(overlap, expected_length)]

    result = pd.DataFrame({
        "ProteinID": protein_id.loc[keep].to_numpy(),
        "Model_15mer": peptide.to_numpy(),
        "Peptide_Position": frag_positions,
        "InterPro_Domain": dbd_name.loc[keep].to_numpy(),
        "Domain_Position": domain_positions,
        "Overlap": overlap_strs,
    }, columns=PAPER_COLUMNS)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False, encoding="utf-8-sig")

    total_annotated = int(annotated.sum())
    no_annotation = len(df) - total_annotated
    excluded_by_source = int((annotated & ~source_ok).sum())
    total = len(result)
    print(f"Input protein records: {len(df)}")
    print(f"Records without a valid DBD annotation or coordinates: {no_annotation}")
    print(f"Records with a valid DBD annotation (all sources): {total_annotated}")
    if excluded_by_source:
        print(f"Additionally excluded by annotation source: {excluded_by_source} "
              "(use --include-smart to keep)")
    print(f"Retained records: {total}")
    print(f"Full overlap: {int(np.sum(overlap == expected_length))}")
    print(f"Partial overlap: {int(np.sum((overlap > 0) & (overlap < expected_length)))}")
    print(f"No overlap (has DBD but 0/15): {int(np.sum(overlap == 0))}")
    print(f"Output file: {output_path}")


if __name__ == "__main__":
    main()