# -*- coding: utf-8 -*-
"""
Length-normalized IG statistics script.
Reads the three-model CSVs produced in the three_model_dbd_results directory.
"""

import argparse
from itertools import combinations
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

MODEL_NAMES = ("AttentionTFactorCNN", "StrucTFactor", "DeepTFactor")
SPLITS = ("OOF", "TEST")


def cohen_dz(diff):
    """Paired effect size Cohen's dz."""
    diff = np.asarray(diff, dtype=float)
    diff = diff[~np.isnan(diff)]
    if len(diff) < 2 or diff.std(ddof=1) == 0:
        return np.nan
    return float(diff.mean() / diff.std(ddof=1))


def holm_bonferroni(pvals):
    """Holm-Bonferroni correction."""
    pvals = np.asarray(pvals, dtype=float)
    n = len(pvals)
    order = np.argsort(pvals)
    adjusted = np.empty(n)
    running = 0.0
    for rank, idx in enumerate(order):
        val = min(1.0, (n - rank) * pvals[idx])
        running = max(running, val)
        adjusted[idx] = running
    return adjusted


def run_paired_test(x, y):
    """Paired t-test and Wilcoxon signed-rank test."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]
    diff = x - y
    n = len(diff)

    res = {
        "N": n, "Mean_DBD": float(x.mean()), "Mean_nonDBD": float(y.mean()),
        "Mean_Diff": float(diff.mean()), "Effect_Size_dz": cohen_dz(diff),
    }

    if n >= 2:
        t_stat, p_t = stats.ttest_rel(x, y)
        res["t_stat"] = float(t_stat)
        res["p_ttest"] = float(p_t)
        if np.any(diff != 0):
            w_stat, p_w = stats.wilcoxon(x, y)
            res["wilcoxon_stat"] = float(w_stat)
            res["p_wilcoxon"] = float(p_w)
        else:
            res["wilcoxon_stat"] = np.nan
            res["p_wilcoxon"] = np.nan
    else:
        res["t_stat"] = res["p_ttest"] = np.nan
        res["wilcoxon_stat"] = res["p_wilcoxon"] = np.nan
    return res


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="three_model_dbd_results")
    args = parser.parse_args()

    input_dir = Path(args.input).resolve()
    output_dir = input_dir / "stats_normalized"
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = input_dir / "protein_DBD_IG_scores_three_models.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Not found: {csv_path}")

    df = pd.read_csv(csv_path, dtype={"Protein_ID": str})

    # ========== 1. Within-model: DBD vs non-DBD (paired tests) ==========
    within_results = []
    for split in SPLITS:
        sub_split = df[df["Split"] == split]
        if sub_split.empty:
            continue
        for model in MODEL_NAMES:
            sub_model = sub_split[sub_split["Model"] == model]
            if sub_model.empty:
                continue
            res = run_paired_test(sub_model["DBD_IG_mean"], sub_model["nonDBD_IG_mean"])
            res.update({"Split": split, "Model": model})
            within_results.append(res)

    df_within = pd.DataFrame(within_results)
    if not df_within.empty:
        df_within["p_ttest_holm"] = holm_bonferroni(df_within["p_ttest"].fillna(1.0))
        df_within["p_wilcoxon_holm"] = holm_bonferroni(df_within["p_wilcoxon"].fillna(1.0))
        cols = ["Split", "Model", "N", "Mean_DBD", "Mean_nonDBD", "Mean_Diff",
                "t_stat", "p_ttest", "p_ttest_holm",
                "wilcoxon_stat", "p_wilcoxon", "p_wilcoxon_holm", "Effect_Size_dz"]
        df_within = df_within[cols]
        df_within.to_csv(output_dir / "stats_normalized_within_model.csv",
                         index=False, encoding="utf-8-sig")

    # ========== 2. Between-model: DBD_minus_nonDBD differences ==========
    between_results = []
    for split in SPLITS:
        sub_split = df[df["Split"] == split]
        if sub_split.empty:
            continue
        wide = sub_split.pivot(index="Protein_ID", columns="Model",
                               values="DBD_minus_nonDBD_mean")
        for m1, m2 in combinations(MODEL_NAMES, 2):
            if m1 in wide.columns and m2 in wide.columns:
                res = run_paired_test(wide[m1], wide[m2])
                res.update({"Split": split, "Comparison": f"{m1} vs {m2}"})
                between_results.append(res)

    df_between = pd.DataFrame(between_results)
    if not df_between.empty:
        df_between["p_ttest_holm"] = holm_bonferroni(df_between["p_ttest"].fillna(1.0))
        df_between["p_wilcoxon_holm"] = holm_bonferroni(df_between["p_wilcoxon"].fillna(1.0))
        cols = ["Split", "Comparison", "N", "Mean_Diff",
                "t_stat", "p_ttest", "p_ttest_holm",
                "wilcoxon_stat", "p_wilcoxon", "p_wilcoxon_holm", "Effect_Size_dz"]
        df_between = df_between[cols]
        df_between.to_csv(output_dir / "stats_normalized_between_models.csv",
                          index=False, encoding="utf-8-sig")

    # ========== Console output (ready for paper) ==========
    print("\n" + "=" * 50)
    print("1. Within-model: DBD vs non-DBD (paired t-test)")
    print("=" * 50)
    if not df_within.empty:
        print(df_within[["Split", "Model", "N", "Mean_Diff", "t_stat",
                         "p_ttest_holm", "Effect_Size_dz"]].to_string(index=False))

    print("\n" + "=" * 50)
    print("2. Between-model: DBD_minus_nonDBD differences (paired t-test)")
    print("=" * 50)
    if not df_between.empty:
        print(df_between[["Split", "Comparison", "N", "Mean_Diff", "t_stat",
                          "p_ttest_holm", "Effect_Size_dz"]].to_string(index=False))

    print(f"\nStatistics saved to: {output_dir}")


if __name__ == "__main__":
    main()