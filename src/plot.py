# file: heatmaps_from_csvs.py
import os
import glob
import argparse
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import os, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from textwrap import wrap

import os, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- core math helpers ---

def diagonal_average(df: pd.DataFrame) -> float:
    """Mean of the main diagonal (NaN-safe). Uses the largest possible square."""
    n = min(df.shape[0], df.shape[1])
    if n == 0:
        return float('nan')
    M = df.to_numpy()[:n, :n].astype(float)
    return float(np.nanmean(np.diag(M)))

def off_diagonal_average(df: pd.DataFrame) -> float:
    """
    Mean of all cells EXCEPT the main diagonal (NaN-safe).
    Uses the largest possible square (top-left n×n).
    """
    n = min(df.shape[0], df.shape[1])
    if n == 0:
        return float('nan')
    M = df.to_numpy()[:n, :n].astype(float)
    mask = ~np.eye(n, dtype=bool)
    vals = M[mask]
    # Handle case where everything except diag is NaN
    return float(np.nanmean(vals)) if np.isfinite(vals).any() else float('nan')

# --- batch loaders over a folder ---

def diag_offdiag_means_from_folder(folder: str, pattern: str = "*.csv", loader=None):
    """
    Load all CSVs in `folder`, compute both diagonal and off-diagonal means.
    `loader` should be your robust load_matrix(...). If None, a simple csv->numeric is used.
    Returns (names, diag_means, offdiag_means, summary_df)
    """
    if loader is None:
        def _fallback_loader(p):
            df = pd.read_csv(p, header=0, index_col=0)
            return df.apply(pd.to_numeric, errors="coerce")
        loader = _fallback_loader

    csv_paths = sorted(glob.glob(os.path.join(folder, pattern)))
    names, d_means, od_means = [], [], []
    for p in csv_paths:
        try:
            df = loader(p)
            d = diagonal_average(df)
            od = off_diagonal_average(df)
        except Exception as e:
            print(f"Skipping {p}: {e}")
            continue
        names.append(os.path.splitext(os.path.basename(p))[0])
        d_means.append(d)
        od_means.append(od)

    summary = pd.DataFrame({
        "model": names,
        "diagonal_mean": d_means,
        "offdiag_mean": od_means,
    })
    return names, d_means, od_means, summary

# --- plotting ---

def plot_offdiag_histogram(
    names,
    offdiag_means,
    out_path: str,
    title,
    ylim=(1, 5),
    palette="colorblind",     # good contrast & colorblind-safe
    context="paper"           # 'paper'/'talk'/'notebook'
):
    names=['Claude Sonnet 4.5','GPT-5','Gemini 2.5 Pro','llama3.1:8b','llama3.3:latest:70B','qwen3:8b','qwen3:30b']
    err= None
    sns.set_theme(style="whitegrid", context=context)
    plt.rcParams.update({
        "figure.dpi": 180,
        "axes.titleweight": "semibold",
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 9,
        "axes.labelweight": "regular",
        "font.family": "serif",          # prints well
        "pdf.fonttype": 42,              # better font embedding
        "ps.fonttype": 42
    })

    # Color-blind-safe (Okabe–Ito) palette with strong contrast
    okabe_ito = ["#0072B2", "#E69F00", "#009E73", "#D55E00",
                 "#CC79A7", "#56B4E9", "#F0E442", "#000000"]
    # If more bars than colors, Seaborn will cycle—this palette stays high-contrast
    palette = okabe_ito

    # DataFrame
    df = pd.DataFrame({"Model": names, "OffDiag": offdiag_means})
    if err is not None:
        df["Err"] = err

    # Figure
    fig, ax = plt.subplots(figsize=(7.4, 4.2), dpi=180)

    # Optional error bar kwargs
    eb_kwargs = dict(ci=None)
    if err is not None:
        eb_kwargs.update(dict(errorbar=None))  # disable Seaborn’s estimator CI
    bar = sns.barplot(
        data=df, x="Model", y="OffDiag",
        palette=palette, edgecolor="black", linewidth=0.8, ax=ax,
        **({} if err is None else {})
    )

    # If error bars provided, draw them explicitly (clear and print-friendly)
    if err is not None:
        xlocs = [p.get_x() + p.get_width() / 2 for p in ax.patches]
        ax.errorbar(
            xlocs, df["OffDiag"].values, yerr=df["Err"].values,
            fmt="none", elinewidth=1.0, capsize=3, capthick=1.0, ecolor="black", zorder=3
        )

    # Labels & title
    ax.set_ylabel("Code Paper Attribution Score")
    ax.set_xlabel("")
    #ax.set_title(title, pad=8)

    # Y-limits & grid
    if ylim:
        ax.set_ylim(*ylim)
    ax.grid(axis="y", linestyle=(0, (3, 3)), alpha=0.35)
    sns.despine(ax=ax, left=False, bottom=False)

    # Ticks: wrap long model names to keep rotation 0 (cleaner for print)
    wrapped = ['\n'.join(wrap(str(t), width=14)) for t in df["Model"]]
    ax.set_xticklabels(wrapped, rotation=0, ha="center")
    ax.tick_params(axis="x", pad=2)

    # Subtle baseline (optional; comment out if not wanted)
    # ax.axhline(0, linewidth=0.8, color="black", alpha=0.6)

    # Value labels with contrast box (prints well on grayscale too)
    for p in ax.patches:
        h = p.get_height()
        if not np.isnan(h):
            x = p.get_x() + p.get_width() / 2
            y = h
            ax.annotate(f"{h:.2f}",
                        (x, y),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha="center", va="bottom",
                        fontsize=8.5,
                        bbox=dict(boxstyle="round,pad=0.15",
                                  facecolor="white", edgecolor="none", alpha=0.85))

    fig.tight_layout()

    # Save PNG and a vector PDF (great for LaTeX/print)
    fig.savefig(out_path, bbox_inches="tight")
    root = out_path.rsplit(".", 1)[0]
    fig.savefig(f"{root}.pdf", bbox_inches="tight")

    plt.close(fig)

def plot_diag_vs_offdiag_grouped(
    names, diag_means, offdiag_means, out_path: str,
    title="",
    ylim=(1, 5), context="paper", errs=None
):
    """
    Grouped bar chart using Seaborn:
      - Two bars per model: Diagonal vs Off-Diagonal
      - Color-blind-safe palette, high contrast edges
      - Optional error bars via `errs={'diag': [...], 'offdiag': [...]}` (same length as names)
      - Saves both PNG and PDF (vector)
    """
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    # ------ Style & fonts (paper-friendly) ------
    sns.set_theme(style="whitegrid", context=context)
    # plt.rcParams.update({
    #     "figure.dpi": 180,
    #     "axes.titleweight": "semibold",
    #     "axes.titlesize": 12,
    #     "axes.labelsize": 11,
    #     "xtick.labelsize": 8.5,
    #     "ytick.labelsize": 9,
    #     "axes.labelweight": "regular",
    #     "font.family": "serif",
    #     "pdf.fonttype": 42,  # embed fonts better
    #     "ps.fonttype": 42
    # })
    plt.rcParams.update({
    "figure.dpi": 180,

    # Titles & labels
    "axes.titleweight": "bold",
    "axes.titlesize": 14,
    "axes.labelsize": 13,
    "axes.labelweight": "bold",

    # Tick labels
    "xtick.labelsize": 10.5,
    "ytick.labelsize": 11,
    "xtick.major.width": 1.2,
    "ytick.major.width": 1.2,

    # Axes spines
    "axes.linewidth": 1.2,

    # Font
    "font.family": "serif",

    # Vector-friendly fonts
    "pdf.fonttype": 42,
    "ps.fonttype": 42
})


    # Okabe–Ito (color-blind-safe) — pick two high-contrast colors
    diag_color = "#0072B2"      # blue
    offdiag_color = "#D55E00"   # vermillion
    palette = {
        "Code Paper Attribution Score": diag_color,
        "Misattribution  Score": offdiag_color
    }

    # ------ Data ------
    names=['Claude Sonnet 4.5','GPT-5 ','Gemini 2.5 Pro','llama3:8b','llama3.3:latest:70B','qwen3:8b','qwen3:30b']
    df = pd.DataFrame({
        "Model": names,
        "Code Paper Attribution Score": np.asarray(diag_means, dtype=float),
        "Misattribution  Score": np.asarray(offdiag_means, dtype=float),
    })
    df_long = df.melt(id_vars="Model", var_name="Type", value_name="Score")

    # Optional errors
    df_err = None
    if errs is not None and "diag" in errs and "offdiag" in errs:
        df_err = pd.DataFrame({
            "Model": names,
            "Code Paper Attribution Score": np.asarray(errs["diag"], dtype=float),
            "Misattribution Confidence Score": np.asarray(errs["offdiag"], dtype=float),
        }).melt(id_vars="Model", var_name="Type", value_name="Err")

    # ------ Figure ------
    fig, ax = plt.subplots(figsize=(9, 5), dpi=180)

    # Draw bars (we pass aggregated values; disable Seaborn's CI)
    bar = sns.barplot(
        data=df_long, x="Model", y="Score", hue="Type",
        palette=palette, edgecolor="black", linewidth=0.8,
        dodge=True, errorbar=None, ax=ax
    )

    # Manual error bars (clean, print-friendly)
    if df_err is not None:
        # Compute x positions for grouped bars
        # Grab positions by category order in the Axes
        # (Seaborn orders patches by each x, then each hue in order of legend)
        x_positions = {}
        for p in ax.patches:
            x_center = p.get_x() + p.get_width() / 2
            # p is in order: for each x tick, bars for each hue
            x_positions.setdefault(round(p.get_x(), 6), []).append((x_center, p))

        # Build mapping: (Model, Type) -> x_center
        model_to_x = {m: i for i, m in enumerate(df["Model"].tolist())}
        hue_order = ["Diagonal (i==j)", "Off-Diagonal (i≠j)"]
        width = ax.patches[0].get_width() if ax.patches else 0.35
        for i, m in enumerate(df["Model"].tolist()):
            # Left and right centers based on dodge width
            # Use the first bar width to estimate offset
            x_base = i
            offsets = np.linspace(-width/2, width/2, len(hue_order))
            centers = {h: x_base + off for h, off in zip(hue_order, offsets)}
            # plot error bars
            for h in hue_order:
                val = float(df_long[(df_long["Model"] == m) & (df_long["Type"] == h)]["Score"])
                err = float(df_err[(df_err["Model"] == m) & (df_err["Type"] == h)]["Err"])
                ax.errorbar(
                    centers[h], val, yerr=err,
                    fmt="none", elinewidth=1.0, capsize=3, capthick=1.0,
                    ecolor="black", zorder=3
                )

    # Labels & title
    ax.set_ylabel("Average score (1–5)")
    ax.set_xlabel("")
    ax.set_title(title, pad=8)

    # Y-limits & grid
    if ylim:
        ax.set_ylim(*ylim)
    ax.grid(axis="y", linestyle=(0, (3, 3)), alpha=0.35)
    sns.despine(ax=ax, left=False, bottom=False)

    # Wrap long model names to avoid rotation
    wrapped = ['\n'.join(wrap(str(t), width=14)) for t in df["Model"]]
    ax.set_xticklabels(wrapped, rotation=0, ha="center")
    ax.tick_params(axis="x", pad=2)

    # Legend on top, compact
    leg = ax.legend(title="", ncol=2, frameon=False, loc="upper center",
                    bbox_to_anchor=(0.5, 1.12), handlelength=1.2, columnspacing=1.2)
    for txt in leg.get_texts():
        txt.set_fontsize(9)

    # Value labels on bars
    for p in ax.patches:
        h = p.get_height()
        if np.isnan(h):
            continue
        x = p.get_x() + p.get_width() / 2
        ax.annotate(f"{h:.2f}",
                    (x, h),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center", va="bottom",
                    fontsize=8.5,
                    bbox=dict(boxstyle="round,pad=0.15",
                              facecolor="white", edgecolor="none", alpha=0.85))

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    root = out_path.rsplit(".", 1)[0]
    fig.savefig(f"{root}.pdf", bbox_inches="tight")
    plt.close(fig)

def save_diag_offdiag_summary(summary_df: pd.DataFrame, out_path: str):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    summary_df.to_csv(out_path, index=False)


def load_matrix(csv_path: str) -> pd.DataFrame:
    """
    Load a matrix CSV with header row and index column.
    Coerces to numeric where possible.
    """
    df = pd.read_csv(csv_path, header=0, index_col=0)
    # Make sure values are numeric
    df = df.apply(pd.to_numeric, errors="coerce")
    return df

def plot_heatmap(ax: plt.Axes, df: pd.DataFrame, title: str,
                 vmin: float = 1.0, vmax: float = 5.0,
                 annotate: bool = False) -> None:
    """
    Plot a single heatmap on the provided Axes.
    Uses a green=high, red=low colormap for 1..5 Likert scores.
    """
    # Colormap: red (1) -> yellow (3) -> green (5)
    cmap = plt.get_cmap("RdYlGn")
    im = ax.imshow(df.values, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto", interpolation="nearest")

    # Ticks & labels (keep it readable for 20×20)
    ax.set_xticks(np.arange(len(df.columns)))
    ax.set_yticks(np.arange(len(df.index)))
    ax.set_xticklabels(df.columns, fontsize=8, rotation=90)
    ax.set_yticklabels(df.index, fontsize=8)

    ax.set_title(title, fontsize=10, pad=8)

    # Gridlines (subtle)
    ax.set_xticks(np.arange(-.5, len(df.columns), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(df.index), 1), minor=True)
    ax.grid(which="minor", color="black", linestyle="-", linewidth=0.2)
    ax.tick_params(which="minor", bottom=False, left=False)

    if annotate:
        # Annotate integer scores
        for i in range(df.shape[0]):
            for j in range(df.shape[1]):
                val = df.iat[i, j]
                if pd.notna(val):
                    ax.text(j, i, f"{int(val)}", ha="center", va="center", fontsize=7, color="black")

    return im

def save_single_heatmaps(dfs: List[pd.DataFrame], names: List[str], outdir: str,
                         vmin=1.0, vmax=5.0, annotate=False) -> None:
    os.makedirs(outdir, exist_ok=True)
    for df, name in zip(dfs, names):
        fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
        im = plot_heatmap(ax, df, title=name, vmin=vmin, vmax=vmax, annotate=annotate)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Score (1–5)", rotation=90)
        fig.tight_layout()
        outpath = os.path.join(outdir, f"{name}_heatmap.png")
        fig.savefig(outpath, bbox_inches="tight")
        plt.close(fig)

def save_combined_grid(dfs: List[pd.DataFrame], names: List[str], outpath: str,
                       vmin=1.0, vmax=5.0, annotate=False) -> None:
    """
    Create a single figure with all heatmaps in a grid. For 5 files we’ll use 2×3.
    """
    n = len(dfs)
    rows, cols = (2, 3) if n <= 6 else (int(np.ceil(n/3)), 3)
    fig, axes = plt.subplots(rows, cols, figsize=(cols*4, rows*3.5), dpi=150)
    axes = np.array(axes).reshape(rows, cols)

    shared_im = None
    k = 0
    for r in range(rows):
        for c in range(cols):
            ax = axes[r, c]
            if k < n:
                shared_im = plot_heatmap(ax, dfs[k], names[k], vmin=vmin, vmax=vmax, annotate=annotate)
            else:
                ax.axis("off")
            k += 1

    if shared_im is not None:
        cbar = fig.colorbar(shared_im, ax=axes.ravel().tolist(), fraction=0.02, pad=0.01)
        cbar.set_label("Score (1–5)", rotation=90)

    fig.suptitle("Heatmaps", fontsize=12, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    os.makedirs(os.path.dirname(outpath) or ".", exist_ok=True)
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)

def save_average_heatmap(dfs: List[pd.DataFrame], names: List[str], outpath: str,
                         vmin=1.0, vmax=5.0, annotate=False) -> None:
    """
    Compute element-wise mean across aligned dataframes and plot.
    Aligns on shared row/column labels (inner join) to be safe.
    """
    # Align all to the intersection of indices/columns
    idx = set(dfs[0].index)
    cols = set(dfs[0].columns)
    for df in dfs[1:]:
        idx &= set(df.index)
        cols &= set(df.columns)
    idx = sorted(idx)
    cols = sorted(cols)

    if not idx or not cols:
        raise ValueError("No common row/column labels across CSVs to average.")

    aligned = [df.loc[idx, cols] for df in dfs]
    avg = sum(aligned) / len(aligned)

    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    im = plot_heatmap(ax, avg, title="Average Score (mean across CSVs)", vmin=vmin, vmax=vmax, annotate=annotate)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Average Score (1–5)", rotation=90)
    fig.tight_layout()
    os.makedirs(os.path.dirname(outpath) or ".", exist_ok=True)
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser(description="Generate heatmaps from multiple 20×20 CSVs.")
    parser.add_argument("--folder", type=str, default=".", help="Folder containing CSVs")
    parser.add_argument("--pattern", type=str, default="*.csv", help="Glob pattern for CSV files")
    parser.add_argument("--outdir", type=str, default="heatmaps_out", help="Output directory")
    parser.add_argument("--annotate", action="store_true", help="Draw the score number in each cell")
    parser.add_argument("--vmin", type=float, default=1.0, help="Min value for color scale")
    parser.add_argument("--vmax", type=float, default=5.0, help="Max value for color scale")
    args = parser.parse_args()

    csv_paths = sorted(glob.glob(os.path.join(args.folder, args.pattern)))
    if not csv_paths:
        raise SystemExit(f"No CSVs found in: {os.path.join(args.folder, args.pattern)}")

    dfs = []
    names = []
    for p in csv_paths:
        try:
            df = load_matrix(p)
        except Exception as e:
            print(f"Skipping {p}: {e}")
            continue
        dfs.append(df)
        # Use the file stem as title
        names.append(os.path.splitext(os.path.basename(p))[0])

    if not dfs:
        raise SystemExit("No valid CSVs to plot.")

    # Per-file heatmaps
    save_single_heatmaps(dfs, names, args.outdir, vmin=args.vmin, vmax=args.vmax, annotate=args.annotate)

    # Combined grid (e.g., 5 CSVs → 2×3 grid)
    save_combined_grid(dfs, names, os.path.join(args.outdir, "combined_heatmaps.png"),
                       vmin=args.vmin, vmax=args.vmax, annotate=args.annotate)

    # Average heatmap (common rows/cols)
    try:
        save_average_heatmap(dfs, names, os.path.join(args.outdir, "average_heatmap.png"),
                             vmin=args.vmin, vmax=args.vmax, annotate=args.annotate)
    except Exception as e:
        print(f"Average heatmap skipped: {e}")

    print(f"Done. Outputs in: {args.outdir}")



    names, d_means, od_means, summary = diag_offdiag_means_from_folder(args.folder, loader=load_matrix)
    save_diag_offdiag_summary(summary, "heatmaps_out/diag_offdiag_summary.csv")
    plot_offdiag_histogram(names, od_means, "heatmaps_out/offdiag_histogram.png","Off Diagonal")
    plot_offdiag_histogram(names, d_means, "heatmaps_out/diag_histogram.png","Diagonal")
    plot_diag_vs_offdiag_grouped(names, d_means, od_means, "heatmaps_out/diag_vs_offdiag.png")


if __name__ == "__main__":
    main()
