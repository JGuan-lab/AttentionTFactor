# -*- coding: utf-8 -*-
"""AttentionTFactor - Rate4Site conservation comparison (Windows / PyCharm).

Input:
    Unzipped folder of Rate4Site_Conservation_Comparison_Results.zip.
    (Default: <script_dir>/Rate4Site_Conservation_Comparison_Results; otherwise
     recursively search the script directory for the required CSV.)

Output:
    Main paper table, overall statistics, significance test, paper figures
    (including the conservation violin plot).
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

# Global font: Arial
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["Arial"]
matplotlib.rcParams["axes.unicode_minus"] = False
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["svg.fonttype"] = "none"


# ========== 1. File paths ==========
ROOT = Path(__file__).resolve().parent
UNZIP_DIR_NAME = "Rate4Site_Conservation_Comparison_Results"
CSV_ALL_STATUS = "Rate4Site_All_Statuses.csv"
OUT = ROOT / "Rate4Site_PyCharm_Analysis"
OUT.mkdir(parents=True, exist_ok=True)
TF_ONLY = False   # False: keep all samples; True: keep only Label=1


# ========== Fixed axis limits (for cross-model comparison) ==========
# Fill the combined (min, max) of all three models so figures align.
# None = fall back to matplotlib auto limits.
FIXED_AXES = {
    "panel_a_ylim": (-1.50, 1.00),
    "panel_a_ytick": 0.25,
    "panel_b_ylim": (-1.10, 1.05),
    "panel_b_ytick": 0.25,
}


def get_fixed_ylim(key):
    val = FIXED_AXES.get(key, None)
    return None if val is None else val


def apply_fixed_axes(ax, ylim_key, ytick_key):
    ylim = get_fixed_ylim(ylim_key)
    if ylim is not None:
        ax.set_ylim(ylim)
    ytick = FIXED_AXES.get(ytick_key, None)
    if ytick is not None:
        ax.yaxis.set_major_locator(MultipleLocator(ytick))


# ========== 2. Locate input directory and read CSV ==========
def locate_input_dir():
    """Locate the unzipped input directory: prefer <ROOT>/UNZIP_DIR_NAME,
    else recursively search ROOT for the target CSV."""
    candidate = ROOT / UNZIP_DIR_NAME
    if candidate.is_dir() and any(candidate.rglob(CSV_ALL_STATUS)):
        return candidate

    matches = list(ROOT.rglob(CSV_ALL_STATUS))
    if not matches:
        raise FileNotFoundError(
            f"{CSV_ALL_STATUS} not found.\n"
            f"Unzip the archive and place the folder "
            f"(default name: {UNZIP_DIR_NAME}) under:\n{ROOT}")
    return matches[0].parent


INPUT_DIR = locate_input_dir()


def read_csv_file(filename):
    """Recursively find a unique match under INPUT_DIR and read it."""
    matches = [p for p in INPUT_DIR.rglob(filename) if p.is_file()]
    if not matches:
        raise FileNotFoundError(f"{filename} not found under {INPUT_DIR}")
    if len(matches) > 1:
        matches.sort(key=lambda p: len(str(p)))
        print(f"[Note] Multiple {filename} found, using: {matches[0]}")
    return pd.read_csv(matches[0], encoding="utf-8-sig", dtype={"ProteinID": str})


# ========== 3. Relative conservation analysis ==========
def analyze(df, region_col, label):
    result = df.copy()
    result[region_col] = pd.to_numeric(result[region_col], errors="coerce")

    bg_col = "DBD Background Mean Rate (z-score)"
    result[bg_col] = pd.to_numeric(result[bg_col], errors="coerce")
    result = result.dropna(subset=[region_col, bg_col]).copy()

    # Lower Rate4Site rate = stronger conservation;
    # positive difference = high-contribution region more conserved.
    result["Conservation Difference"] = result[bg_col] - result[region_col]
    result["More Conserved"] = np.where(
        result["Conservation Difference"] > 0, "Yes", "No")

    delta = result["Conservation Difference"].to_numpy(float)
    if len(delta) >= 2 and np.any(delta != 0):
        p_value = wilcoxon(delta, alternative="greater").pvalue
    else:
        p_value = np.nan

    summary = {
        "Comparison": label,
        "Valid Proteins": len(delta),
        "More Conserved in High-Contribution Region (n)": int(np.sum(delta > 0)),
        "More Conserved in High-Contribution Region (%)": (
            round(100 * np.mean(delta > 0), 2) if len(delta) else np.nan),
        "Mean Difference": float(np.mean(delta)) if len(delta) else np.nan,
        "Median Difference": float(np.median(delta)) if len(delta) else np.nan,
        "One-Sided Wilcoxon P-Value": p_value,
    }
    return result, summary


# ========== 4. Difference distribution figure ==========
def plot_distribution(result):
    delta = result["Conservation Difference"].to_numpy(float)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(delta, bins=26, edgecolor="white")
    ax.axvline(0, linestyle="--", color="black")
    ax.axvline(delta.mean(), linestyle=":", color="gray",
               label=f"Mean = {delta.mean():.3f}")
    ax.set_xlabel("DBD background - 15-mer mean rate (z-score)", fontname="Arial")
    ax.set_ylabel("Number of proteins", fontname="Arial")
    ax.set_title(f"Rate4Site conservation difference (n={len(delta)})", fontname="Arial")
    ax.legend(prop={"family": "Arial"})
    fig.tight_layout()
    fig.savefig(OUT / "Conservation_Difference_Distribution.png", dpi=300)
    fig.savefig(OUT / "Conservation_Difference_Distribution.svg")
    plt.close(fig)


# ========== 5. Paired comparison figure ==========
def plot_paired(result):
    hit = result["15-mer Mean Rate (z-score)"].to_numpy(float)
    background = result["DBD Background Mean Rate (z-score)"].to_numpy(float)

    fig, ax = plt.subplots(figsize=(7, 5))
    for h, b in zip(hit, background):
        ax.plot([1, 2], [h, b], color="gray", alpha=0.15, linewidth=0.6)
    ax.boxplot([hit, background], positions=[1, 2], widths=0.4, showfliers=False)
    ax.set_xticks([1, 2], ["Highest-contribution 15-mer", "Other DBD residues"])
    ax.set_ylabel("Mean normalized evolutionary rate (z-score)", fontname="Arial")
    ax.set_title(f"Paired Rate4Site comparison (n={len(hit)})", fontname="Arial")
    apply_fixed_axes(ax, "panel_a_ylim", "panel_a_ytick")
    fig.tight_layout()
    fig.savefig(OUT / "Conservation_Paired_Comparison.png", dpi=300)
    fig.savefig(OUT / "Conservation_Paired_Comparison.svg")
    plt.close(fig)


# ========== 6. Violin figure ==========
def plot_violin(result):
    """Left panel: side-by-side violins + jittered points + paired lines.
    Right panel: conservation-difference violin with zero and mean lines.
    FIXED_AXES is applied to both panels."""
    hit = result["15-mer Mean Rate (z-score)"].to_numpy(float)
    background = result["DBD Background Mean Rate (z-score)"].to_numpy(float)
    delta = result["Conservation Difference"].to_numpy(float)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5),
                             gridspec_kw={"width_ratios": [1, 1]})
    rng = np.random.default_rng(0)

    # ---------- Panel A ----------
    ax = axes[0]
    data, positions = [hit, background], [1, 2]
    parts = ax.violinplot(data, positions=positions, widths=0.7,
                          showmeans=True, showmedians=True, showextrema=False)
    for body, color in zip(parts["bodies"], ["#4C72B0", "#DD8452"]):
        body.set_facecolor(color)
        body.set_edgecolor("black")
        body.set_alpha(0.65)

    # 1) Build jittered X first so scatter and lines share the same coordinates.
    x_hit = rng.normal(1, 0.04, size=len(hit))
    x_bg = rng.normal(2, 0.04, size=len(background))

    # 2) Paired lines (bottom layer)
    for i in range(len(hit)):
        ax.plot([x_hit[i], x_bg[i]], [hit[i], background[i]],
                color="gray", alpha=0.15, linewidth=0.6, zorder=1)

    # 3) Jittered scatter (same coordinates)
    ax.scatter(x_hit, hit, s=10, color="black", alpha=0.35, zorder=2, linewidths=0)
    ax.scatter(x_bg, background, s=10, color="black", alpha=0.35, zorder=2, linewidths=0)

    ax.set_xticks(positions,
                  ["Highest-contribution\n15-mer", "Other DBD\nresidues"])
    ax.set_ylabel("Mean normalized evolutionary rate (z-score)", fontname="Arial")
    ax.set_title(f"A. Rate4Site scores (n={len(hit)})", fontname="Arial")
    apply_fixed_axes(ax, "panel_a_ylim", "panel_a_ytick")

    # ---------- Panel B ----------
    ax = axes[1]
    parts = ax.violinplot([delta], positions=[1], widths=0.7,
                          showmeans=True, showmedians=True, showextrema=False)
    body = parts["bodies"][0]
    body.set_facecolor("#55A868")
    body.set_edgecolor("black")
    body.set_alpha(0.65)

    x = rng.normal(1, 0.04, size=len(delta))
    ax.scatter(x, delta, s=10, color="black", alpha=0.35, zorder=2, linewidths=0)
    ax.axhline(0, linestyle="--", color="black", linewidth=1)
    ax.axhline(delta.mean(), linestyle=":", color="gray",
               label=f"Mean = {delta.mean():.3f}")
    ax.set_xticks([1], ["Difference"])
    ax.set_ylabel("Background - 15-mer (z-score)", fontname="Arial")
    ax.set_title(f"B. Conservation difference (n={len(delta)})", fontname="Arial")
    ax.legend(loc="upper right", fontsize=9, prop={"family": "Arial"})
    apply_fixed_axes(ax, "panel_b_ylim", "panel_b_ytick")

    fig.tight_layout()
    fig.savefig(OUT / "Conservation_Violin.png", dpi=300)
    fig.savefig(OUT / "Conservation_Violin.svg")
    plt.close(fig)


# ========== 7. Main ==========
def main():
    print(f"Input directory: {INPUT_DIR}")
    data = read_csv_file(CSV_ALL_STATUS)

    if TF_ONLY:
        data = data[pd.to_numeric(data["Label"], errors="coerce") == 1].copy()

    status = (data["Status"].value_counts()
              .rename_axis("Status").reset_index(name="Protein Count"))
    status.to_csv(OUT / "Status_Counts.csv", index=False, encoding="utf-8-sig")

    valid = data[data["Status"] == "OK"].copy()
    if valid.empty:
        raise ValueError("No analyzable Rate4Site results found")

    # Analysis 1: full 15-mer
    full, summary_full = analyze(
        valid, "15-mer Mean Rate (z-score)",
        "Full 15-mer vs. DBD Background")

    # Analysis 2: only the in-DBD intersection
    intersection, summary_intersection = analyze(
        valid, "Overlap Mean Rate (z-score)",
        "DBD Overlap vs. DBD Background")

    # Main paper table
    cols = ["ProteinID", "Model Fragment", "Fragment Positions",
            "InterPro Domain", "Domain Positions", "Overlap",
            "15-mer Mean Rate (z-score)", "DBD Background Mean Rate (z-score)",
            "Conservation Difference", "More Conserved"]
    full[cols].to_csv(OUT / "Paper_Table_Full_15mer.csv",
                      index=False, encoding="utf-8-sig")

    # Intersection sensitivity analysis
    cols2 = ["ProteinID", "Model Fragment", "Fragment Positions",
             "InterPro Domain", "Domain Positions", "Overlap",
             "Overlap Mean Rate (z-score)", "DBD Background Mean Rate (z-score)",
             "Conservation Difference", "More Conserved"]
    intersection[cols2].to_csv(OUT / "Supplementary_Table_DBD_Overlap.csv",
                               index=False, encoding="utf-8-sig")

    # Overall statistics
    summary = pd.DataFrame([summary_full, summary_intersection])
    summary.to_csv(OUT / "Overall_Statistics_and_Significance.csv",
                   index=False, encoding="utf-8-sig")

    # Figures
    plot_distribution(full)
    plot_paired(full)
    plot_violin(full)

    print("\n===== Analysis status =====")
    print(status.to_string(index=False))
    print("\n===== Overall statistics =====")
    print(summary.to_string(index=False))
    print("\nGenerated figures:")
    print("  - Conservation_Difference_Distribution.(png/svg)")
    print("  - Conservation_Paired_Comparison.(png/svg)")
    print("  - Conservation_Violin.(png/svg)")
    print("\nResults saved to:")
    print(OUT)


if __name__ == "__main__":
    main()