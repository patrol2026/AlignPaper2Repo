# file: compare_to_groundtruth.py
import os, re, glob, io
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from textwrap import wrap
# ------------------------
# Parsing & path utilities
# ------------------------

def _read_csv_any(csv_path: str) -> pd.DataFrame:
    """Robust CSV loader (auto-delimiter, BOM-safe)."""
    with open(csv_path, "rb") as f:
        raw = f.read()
    txt = raw.decode("utf-8-sig", errors="replace")
    return pd.read_csv(io.StringIO(txt), sep=None, engine="python")

def _norm_path(p: str) -> str:
    """Normalize path for comparison: slashes, strip quotes/spaces, lowercase."""
    if pd.isna(p):
        return ""
    p = str(p).strip().strip('"').strip("'")
    p = p.replace("\\", "/")
    # Drop URL schemes if any accidentally present
    p = re.sub(r"^[a-zA-Z]+://", "", p)
    # Remove duplicate slashes
    p = re.sub(r"/+", "/", p)
    return p.lower()

def _strip_prefix(path: str, prefix: str) -> str:
    """Remove prefix if path starts with it (after normalization)."""
    P, X = _norm_path(path), _norm_path(prefix)
    if X and P.startswith(X):
        return P[len(X):].lstrip("/")
    return P

def _tail_match(gt_rel: str, pred_hit: str) -> bool:
    """
    Tail-compare: returns True if one path endswith the other by segments.
    We compare normalized strings on segment boundaries.
    """
    a = [s for s in _norm_path(gt_rel).split("/") if s]
    b = [s for s in _norm_path(pred_hit).split("/") if s]
    if not a or not b:
        return False
    # Try matching last k segments where k = min(len(a), len(b))
    k = min(len(a), len(b))
    return a[-k:] == b[-k:]

def _parse_gt_interval(s: str):
    """
    Parse ground-truth 'original result' like '[12,34]' or ' [ 12 , 34 ] '.
    Returns (start, end) as ints, or (None, None) if missing.
    """
    if pd.isna(s):
        return (None, None)
    m = re.search(r"\[\s*(\d+)\s*,\s*(\d+)\s*\]", str(s))
    if not m:
        return (None, None)
    a, b = int(m.group(1)), int(m.group(2))
    if a > b: a, b = b, a
    return (a, b)

def _parse_pred_interval(row: pd.Series):
    """Parse startline/endline from model CSV row."""
    try:
        a = int(row.get("startLine"))
        b = int(row.get("endLine"))
        if a > b: a, b = b, a
        return (a, b)
    except Exception:
        return (None, None)

def _overlap_accurate(iv1, iv2) -> bool:
    """Closed-interval overlap: [a1,b1] intersects [a2,b2]."""
    a1, b1 = iv1
    a2, b2 = iv2
    if None in (a1, b1, a2, b2):
        return False
    return (a1 <= a2) and (b2 <= b1) or  (a2 <= a1) and (b1 <= b2)
def _overlap_fraction(iv1, iv2) -> bool:
    """Closed-interval overlap: [a1,b1] intersects [a2,b2]."""
    a1, b1 = iv1
    a2, b2 = iv2
    if None in (a1, b1, a2, b2):
        return False
    return (a1<a2)and (b1>a2) or (a2<a1) and (b2>a1)

# ------------------------
# Core evaluation
# ------------------------

def evaluate_models(ground_truth_csv: str, model_csv_folder: str, pattern: str = "*.csv"):
    """
    Returns summary DataFrame with:
      model, total_gt, path_matches, path_match_rate, correct, acc_among_matches, overall_acc
    and a dict: model_name -> details DataFrame with per-GT-row flags.
    """
    gt = _read_csv_any(ground_truth_csv)

    # # Normalize important GT fields
    # gt["gt_repopath"] = gt.get("repopath", "").apply(_norm_path) if "repopath" in gt.columns else ""
    # gt["gt_filepath"] = gt.get("filepath", "").apply(_norm_path)
    # gt["gt_relpath"]  = [
    #     _strip_prefix(fp, rp) if isinstance(rp, str) else fp
    #     for fp, rp in zip(gt["gt_filepath"], gt.get("gt_repopath", ""))
    # ]
    # gt[["gt_start", "gt_end"]] = gt.get("original result", "").apply(lambda s: pd.Series(_parse_gt_interval(s)))
    # # Collect model CSVs (exclude the GT one if in same folder)
    all_csvs = sorted(glob.glob(os.path.join(model_csv_folder, pattern)))
    model_csvs = [p for p in all_csvs if os.path.abspath(p) != os.path.abspath(ground_truth_csv)]
    # if not model_csvs:
    #     raise SystemExit("No model CSVs found (did you set the folder/pattern correctly?)")

    summaries = []
    per_model_details = {}

    for p in model_csvs:
        print(p)
        name = os.path.splitext(os.path.basename(p))[0]
        df = _read_csv_any(p)
        # Normalize model paths
        # df["repopath"] = df.get("repopath", "").apply(_norm_path)
        # df["hitpath"]  = df.get("hitpath", "").apply(_norm_path)
        # # Base path derived from hitpath: strip repopath prefix if present
        # df["basepath"] = [
        #     _strip_prefix(hp, rp) if isinstance(rp, str) else hp
        #     for hp, rp in zip(df["hitpath"], df.get("repopath", ""))
        # ]

        # # Pre-index predictions by filename tail to speed matches
        # # Map from filename -> rows with that filename
        # df["filename_tail"] = df["basepath"].apply(lambda x: os.path.basename(x) if isinstance(x, str) else "")
        # pred_by_file = {}
        count=0
        overlapped_count=0
        details=[]
        accurate_count=0
        for i, r in df.iterrows():
            # print(r)
            gt_row=gt.loc[i]
            #print(os.path.basename(r["hitPath"]), os.path.basename(gt_row["Filename"]))
            try:
                if os.path.basename(r["hitPath"]).strip()==os.path.basename(gt_row["Filename"]).strip() :
                    count+=1
                    print(_parse_gt_interval(gt_row["Original Result"]),r)
                    if _overlap_accurate(_parse_gt_interval(gt_row["Original Result"]), _parse_pred_interval(r)):
                        accurate_count+=1
                    elif _overlap_fraction(_parse_gt_interval(gt_row["Original Result"]), _parse_pred_interval(r)):
                        overlapped_count+=1
            except:
                pass
            #pred_by_file.setdefault(r["filename_tail"], []).append(i)
        total_gt = len(gt)
        path_match_rate = count / total_gt if total_gt else 0.0
        acc_among_matches = ((overlapped_count+accurate_count) / count) if count else 0.0
        overall_acc = (overlapped_count+accurate_count) / total_gt if total_gt else 0.0

        summaries.append({
            "model": name,
            "total_gt": total_gt,
            "path_matches": count,
            "path_match_rate": path_match_rate,
            "correct": overlapped_count+accurate_count,
            "accurate_correct": accurate_count,
            "partial_correct": overlapped_count,
            "acc_among_matches": acc_among_matches,
            "overall_acc": overall_acc,
        })
        details.append({
            "model": name,
            "total_gt": len(gt),
            "file_matches": count,
            "accurate_count": accurate_count,
            "overlapped_count": overlapped_count,
            "total_count":accurate_count+overlapped_count
        })
        per_model_details[name] = pd.DataFrame(details)
        print(count)
    summary_df = pd.DataFrame(summaries).sort_values("model").reset_index(drop=True)

    return summary_df, per_model_details
    #     # Evaluate per GT row: path match? and if match, any interval overlap?
    #     details = []
    #     path_match_count = 0
    #     correct_count = 0

    #     for idx, g in gt.iterrows():
    #         gt_rel = g["gt_relpath"]
    #         gt_file = os.path.basename(gt_rel)
    #         gt_iv = (g["gt_start"], g["gt_end"])

    #         candidate_idx = pred_by_file.get(gt_file, [])
    #         matched_rows = []
    #         for pi in candidate_idx:
    #             prow = df.loc[pi]
    #             if _tail_match(gt_rel, prow["basepath"]):
    #                 matched_rows.append(prow)
    #                 #print(prow)

    #         path_matched = len(matched_rows) > 0
    #         overlap_ok = False
    #         chosen_pred = None

    #         if path_matched:
    #             path_match_count += 1
    #             # If multiple predictions match the path, mark correct if ANY overlaps GT interval
    #             for prow in matched_rows:
    #                 pred_iv = _parse_pred_interval(prow)
    #                 if _overlap(gt_iv, pred_iv):
    #                     overlap_ok = True
    #                     chosen_pred = prow
    #                     break
    #             if overlap_ok:
    #                 correct_count += 1

    #         details.append({
    #             "gt_index": idx,
    #             "gt_relpath": gt_rel,
    #             "gt_interval": gt_iv,
    #             "path_matched": bool(path_matched),
    #             "overlap_correct": bool(overlap_ok),
    #             "picked_pred_start": None if chosen_pred is None else chosen_pred.get("startline"),
    #             "picked_pred_end":   None if chosen_pred is None else chosen_pred.get("endline"),
    #             "picked_pred_basepath": None if chosen_pred is None else chosen_pred.get("basepath"),
    #         })

    #     total_gt = len(gt)
    #     path_match_rate = path_match_count / total_gt if total_gt else 0.0
    #     acc_among_matches = (correct_count / path_match_count) if path_match_count else 0.0
    #     overall_acc = correct_count / total_gt if total_gt else 0.0

    #     summaries.append({
    #         "model": name,
    #         "total_gt": total_gt,
    #         "path_matches": path_match_count,
    #         "path_match_rate": path_match_rate,
    #         "correct": correct_count,
    #         "acc_among_matches": acc_among_matches,
    #         "overall_acc": overall_acc,
    #     })
    #     per_model_details[name] = pd.DataFrame(details)

    # summary_df = pd.DataFrame(summaries).sort_values("model").reset_index(drop=True)
    # return summary_df, per_model_details

# ------------------------
# Plotting
# ------------------------
# def plot_path_match_histogram(summary_df: pd.DataFrame, out_png: str):
#     """Bar chart of path matches per model (and non-matches stacked)."""
#     os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
#     names = summary_df["model"].tolist()
#     matches = summary_df["path_matches"].to_numpy()
#     totals = summary_df["total_gt"].to_numpy()
#     non_matches = totals - matches

#     x = np.arange(len(names))
#     fig, ax = plt.subplots(figsize=(9,5), dpi=150)
#     ax.bar(x, matches, label="Path matches")
#     ax.bar(x, non_matches, bottom=matches, label="Path non-matches", alpha=0.4)
#     ax.set_xticks(x)
#     ax.set_xticklabels(names, rotation=45, ha="right")
#     ax.set_ylabel("Count")
#     ax.set_title("Path match vs non-match per model")
#     ax.legend(frameon=False)
#     ax.grid(axis="y", linestyle="--", alpha=0.3)
#     fig.tight_layout()
#     fig.savefig(out_png, bbox_inches="tight")
#     plt.close(fig)
def plot_path_match_histogram(summary_df: pd.DataFrame, out_png: str,
                                  title="",
                                  context="paper", normalize=False):
    """
    Publication-ready stacked bar chart with Seaborn styling.
      - Color-blind-safe palette (Okabe–Ito)
      - Strong edges for print; serif fonts
      - Optional normalization to 100% stacked bars (proportions)
      - Saves PNG + vector PDF
    """
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)

    # --- Style ---
    sns.set_theme(style="whitegrid", context=context)
    plt.rcParams.update({
        "figure.dpi": 180,
        "axes.titleweight": "semibold",
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 9,
        "axes.labelweight": "regular",
        "font.family": "serif",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    # Okabe–Ito: high contrast & color-blind safe
    match_color = "#0072B2"     # blue
    nomatch_color = "#D55E00"   # vermillion

    names = summary_df["model"].tolist()
    names=['Claude Sonnet 4.5','Gemini 2.5 Pro','GPT-5','llama3.3:latest:70B','llama3.1:8b','qwen3:30b','qwen3:8b']

    matches = summary_df["path_matches"].to_numpy(dtype=float)
    totals  = summary_df["total_gt"].to_numpy(dtype=float)
    non_matches = totals - matches

    if normalize:
        # Avoid division by zero
        denom = np.where(totals == 0, 1.0, totals)
        matches = matches / denom
        non_matches = non_matches / denom
        y_label = "Proportion"
        y_lim = (0, 1)
    else:
        y_label = "Count"
        y_lim = None

    x = np.arange(len(names))
    width = 0.75  # wider bars look good in stacked plots

    fig, ax = plt.subplots(figsize=(9, 5), dpi=180)

    # Seaborn styling + Matplotlib stacking for clarity/contrast
    b1 = ax.bar(
        x, matches, width=width, label="matched paths",
        edgecolor="black", linewidth=0.8, color=match_color
    )
    b2 = ax.bar(
        x, non_matches, bottom=matches, width=width, label="non-matched paths",
        edgecolor="black", linewidth=0.8, color=nomatch_color, alpha=0.95
    )

    # Labels & title
    ax.set_ylabel(y_label)
    ax.set_xlabel("")
    ax.set_title(title, pad=8)

    # Grid & spine cleanup
    if y_lim: ax.set_ylim(*y_lim)
    ax.grid(axis="y", linestyle=(0, (3, 3)), alpha=0.35)
    sns.despine(ax=ax, left=False, bottom=False)

    # Wrap long model names, keep rotation 0 for print legibility
    wrapped = ['\n'.join(wrap(str(t), width=14)) for t in names]
    ax.set_xticks(x)
    ax.set_xticklabels(wrapped, rotation=0, ha="center")
    ax.tick_params(axis="x", pad=2)

    # Legend: compact, on top
    leg = ax.legend(title="", ncol=2, frameon=False, loc="upper center",
                    bbox_to_anchor=(0.5, 1.10), handlelength=1.2, columnspacing=1.2)
    for txt in leg.get_texts():
        txt.set_fontsize(9)

    # Value labels (counts or percentages) centered within each segment
    def _annotate_stack(bars, values_bottom, total_values):
        for rect, bottom_val, total_val in zip(bars, values_bottom, total_values):
            h = rect.get_height()
            if h <= 0:
                continue
            x = rect.get_x() + rect.get_width()/2
            y = rect.get_y() + h/2  # mid of the segment
            if normalize:
                txt = f"{h*100:.0f}%"
            else:
                # For the top segment, print its value; for bottom, print its own value
                txt = f"{int(round(h))}"
            ax.annotate(txt, (x, y),
                        ha="center", va="center",
                        fontsize=8.5,
                        bbox=dict(boxstyle="round,pad=0.15",
                                  facecolor="white", edgecolor="none", alpha=0.85))

    _annotate_stack(b1, np.zeros_like(matches), totals)
    _annotate_stack(b2, matches, totals)

    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    root = out_png.rsplit(".", 1)[0]
    fig.savefig(f"{root}.pdf", bbox_inches="tight")
    plt.close(fig)
def plot_accuracy_among_matches(summary_df: pd.DataFrame, out_png: str,
                                    title="",
                                    context="paper",
                                    show_total=True):
    """
    Publication-quality grouped bars for accuracy among path matches.
      - Bars: Partial, Accurate (proportions in [0,1])
      - Optional overlay: Total accuracy (black line+markers)
      - Color-blind safe (Okabe–Ito), black edges for print
      - Wrapped x tick labels, percent annotations
      - Exports PNG + vector PDF
    """
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)

    # --- Style ---
    sns.set_theme(style="whitegrid", context=context)
    plt.rcParams.update({
        "figure.dpi": 180,
        "axes.titleweight": "semibold",
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 9,
        "axes.labelweight": "regular",
        "font.family": "serif",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    # Okabe–Ito (color-blind safe)
    partial_color  = "#E69F00"  # orange
    accurate_color = "#009E73"  # green
    total_color    = "#000000"  # black for overlay
    palette = {"Partial": partial_color, "Accurate": accurate_color}

    # --- Data prep ---
    names = summary_df["model"].tolist()
    names=['Claude Sonnet 4.5','Gemini 2.5 Pro','GPT-5','llama3.3:latest:70B','llama3.1:8b','qwen3:30b','qwen3:8b']

    path_matches = summary_df["path_matches"].to_numpy(dtype=float)
    total_accuracy = summary_df["acc_among_matches"].to_numpy(dtype=float)

    # Avoid div-by-zero
    denom = np.where(path_matches == 0, np.nan, path_matches)
    partial_accuracy  = (summary_df["partial_correct"].to_numpy(dtype=float)  / denom)
    accurate_accuracy = (summary_df["accurate_correct"].to_numpy(dtype=float) / denom)

    # Replace NaNs with 0 for plotting
    partial_accuracy  = np.nan_to_num(partial_accuracy,  nan=0.0)
    accurate_accuracy = np.nan_to_num(accurate_accuracy, nan=0.0)
    total_accuracy    = np.nan_to_num(total_accuracy,    nan=0.0)

    df = pd.DataFrame({
        "Model": names,
        "Partial": partial_accuracy,
        "Accurate": accurate_accuracy,
        "Total": total_accuracy
    })
    df_long = df.melt(id_vars="Model", value_vars=["Partial", "Accurate"],
                      var_name="Type", value_name="Accuracy")

    # --- Figure ---
    fig, ax = plt.subplots(figsize=(12, 4.4), dpi=180)

    # Grouped bars (disable CI because values are precomputed)
    sns.barplot(
        data=df_long, x="Model", y="Accuracy", hue="Type",
        palette=palette, edgecolor="black", linewidth=0.8,
        dodge=True, errorbar=None, ax=ax
    )

    # Optional overlay: Total as line+markers centered on each group
    if show_total:
        x_positions = np.arange(len(names))
        # Get centers of grouped bars: use tick positions
        ax.plot(x_positions, df["Total"].values, marker="o", linestyle="-",
                linewidth=1.2, markersize=4, color=total_color, label="Total")

    # Labels & title
    ax.set_ylabel("Accuracy (0–1)")
    ax.set_xlabel("")
    ax.set_title(title, pad=8)

    # Limits & grid
    ax.set_ylim(0, 1)
    ax.grid(axis="y", linestyle=(0, (3, 3)), alpha=0.35)
    sns.despine(ax=ax, left=False, bottom=False)

    # Wrap long model names; keep rotation 0 for print legibility
    wrapped = ['\n'.join(wrap(str(t), width=14)) for t in names]
    ax.set_xticklabels(wrapped, rotation=0, ha="center")
    ax.tick_params(axis="x", pad=2)

    # Legend: compact, on top
    handles, labels = ax.get_legend_handles_labels()
    # Ensure unique legend entries (bars + optional total)
    seen = {}
    uniq_handles, uniq_labels = [], []
    for h, l in zip(handles, labels):
        if l not in seen:
            seen[l] = True
            uniq_handles.append(h)
            uniq_labels.append(l)
    leg = ax.legend(uniq_handles, uniq_labels, title="", ncol=3 if show_total else 2,
                    frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.12),
                    handlelength=1.2, columnspacing=1.2)
    for txt in leg.get_texts():
        txt.set_fontsize(9)

    # Annotate bar values as percentages
    for p in ax.patches:
        # Skip line markers (only bars are patches with width > 0)
        if p.get_width() <= 0:
            continue
        h = p.get_height()
        if np.isnan(h):
            continue
        x = p.get_x() + p.get_width() / 2
        ax.annotate(f"{h*100:.0f}%",
                    (x, h),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center", va="bottom",
                    fontsize=8.5,
                    bbox=dict(boxstyle="round,pad=0.15",
                              facecolor="white", edgecolor="none", alpha=0.85))

    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    root = out_png.rsplit(".", 1)[0]
    fig.savefig(f"{root}.pdf", bbox_inches="tight")
    plt.close(fig)

def plot_overall_accuracy(summary_df: pd.DataFrame, out_png: str,
                              title="Overall accuracy",
                              context="paper"):
    """
    Publication-quality bar chart of overall accuracy in [0,1].
    - Seaborn styling, Okabe–Ito color, black edges for print
    - Wrapped x-tick labels, value labels as percentages
    - Saves PNG + vector PDF
    """
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)

    # --- Style ---
    sns.set_theme(style="whitegrid", context=context)
    plt.rcParams.update({
        "figure.dpi": 180,
        "axes.titleweight": "semibold",
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 9,
        "axes.labelweight": "regular",
        "font.family": "serif",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    # Okabe–Ito single hue (high contrast); add black edges for print
    bar_color = "#0072B2"  # blue

    names = summary_df["model"].tolist()
    names=['Claude Sonnet 4.5','Gemini 2.5 Pro','GPT-5','llama3.3:latest:70B','llama3.1:8b','qwen3:30b','qwen3:8b']

    acc = summary_df["overall_acc"].to_numpy(dtype=float)

    # --- Figure ---
    fig, ax = plt.subplots(figsize=(8, 5), dpi=180)
    sns.barplot(
        x=names, y=acc,
        color=bar_color, edgecolor="black", linewidth=0.8,
        ax=ax, errorbar=None
    )

    # Labels & title
    ax.set_ylabel("Overall accuracy")
    ax.set_xlabel("")
    ax.set_title(title, pad=8)

    # Limits & grid
    ax.set_ylim(0, 1)
    ax.grid(axis="y", linestyle=(0, (3, 3)), alpha=0.35)
    sns.despine(ax=ax, left=False, bottom=False)

    # Wrap long model names; keep rotation 0 for legibility
    wrapped = ['\n'.join(wrap(str(t), width=14)) for t in names]
    ax.set_xticklabels(wrapped, rotation=0, ha="center")
    ax.tick_params(axis="x", pad=2)

    # Annotate percentages on bars
    for p in ax.patches:
        h = p.get_height()
        if np.isnan(h): 
            continue
        x = p.get_x() + p.get_width() / 2
        ax.annotate(f"{h*100:.1f}%",
                    (x, h),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center", va="bottom",
                    fontsize=8.5,
                    bbox=dict(boxstyle="round,pad=0.15",
                              facecolor="white", edgecolor="none", alpha=0.85))

    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    root = out_png.rsplit(".", 1)[0]
    fig.savefig(f"{root}.pdf", bbox_inches="tight")
    plt.close(fig)

# ------------------------
# CLI
# ------------------------

def main():
    ap = argparse.ArgumentParser(description="Compare five model CSVs to a ground-truth CSV.")
    ap.add_argument("--gt", required=True, help="Ground truth CSV path")
    ap.add_argument("--models_folder", required=True, help="Folder with the 5 model CSVs")
    ap.add_argument("--pattern", default="*.csv", help="Glob for model CSVs (default: *.csv)")
    ap.add_argument("--outdir", default="compare_out", help="Output directory")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    summary_df, details = evaluate_models(args.gt, args.models_folder, args.pattern)
    #print(count)
    summary_csv = os.path.join(args.outdir, "summary.csv")
    summary_df.to_csv(summary_csv, index=False)
    print(f"Saved summary: {summary_csv}")
    # Save per-model details too
    for name, df in details.items():
        df.to_csv(os.path.join(args.outdir, f"details_{name}.csv"), index=False)

    # Plots
    plot_path_match_histogram(summary_df, os.path.join(args.outdir, "path_match_histogram.png"))
    plot_accuracy_among_matches(summary_df, os.path.join(args.outdir, "accuracy_among_matches.png"))
    plot_overall_accuracy(summary_df, os.path.join(args.outdir, "overall_accuracy.png"))
    print(f"Done. Plots in {args.outdir}/")

if __name__ == "__main__":
    main()
