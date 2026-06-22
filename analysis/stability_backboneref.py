"""Analyze whether BackboneRef (BR) training improves ProteinGym zero-shot
performance on *stability*-category assays, relative to the UniRef50 baseline.

Inputs (all local, under data_local/):
  - Per-mutant zero-shot scores (used to recompute the ensemble Spearman robustly,
    avoiding malformed/glued columns in the rank-summary CSVs):
      data_local/proteingym_analysis_path/reformatted/<model>/<assay>.csv
      (cols: mutant/mutated_sequence, DMS_score, assay, ..., <model>_score)
    The ensemble "en_spearman" = spearman(<model>_score, DMS_score).
  - Assay -> Selection Type map (also used to assign each assay to subs/indels):
      data_local/proteingym_benchmarks/DMS_zero_shot/<dms>/AUC/DMS_<dms>_AUC_DMS_level.csv
      (cols include: 'DMS ID', 'Selection Type')

Outputs:
  - CSV  : data_local/proteingym_analysis_path/stability_backboneref_summary.csv
  - CSV  : data_local/proteingym_analysis_path/selection_type_summary.csv
  - Table: printed to stdout
  - Plot : data_local/proteingym_analysis_path/selection_types_boxplot.png
"""
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import seaborn as sns
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_ROOT = "data_local/proteingym_analysis_path"
BENCH_ROOT = "data_local/proteingym_benchmarks/DMS_zero_shot"
REFORMATTED = os.path.join(DATA_ROOT, "reformatted")
DMSS = ["substitutions", "indels"]

# 1novelty dropped: in the manuscript `bothfilter` is the novelty (BRn) model.
MODELS = [
    "dayhoff-170m-uniref50",      # baseline (UniRef50 only)
    "dayhoff-170m-nofilter",      # BR unfiltered  (BRu)
    "dayhoff-170m-rmsd",          # BR quality     (BRq)
    "dayhoff-170m-bothfilter",    # BR novelty     (BRn)
    "dayhoff-170m-uniref90",
    "dayhoff-170m-gigaref",
    "dayhoff-3b-uniref90",
    "dayhoff-3b-msa-gigaref",
    "dayhoff-3b-msa-uniref90-cooldown",
]
BASELINE = "dayhoff-170m-uniref50"

# Human-readable role tags (purely for the output table)
ROLE = {
    "dayhoff-170m-uniref50": "baseline (UniRef50)",
    "dayhoff-170m-nofilter": "BR unfiltered (BRu)",
    "dayhoff-170m-rmsd": "BR quality (BRq)",
    "dayhoff-170m-bothfilter": "BR novelty (BRn)",
}

# Canonical display names + colors, mirroring analysis/plot_metrics.py model_dict.
# BR variants get distinct blue shades; the two UR90 models distinct red shades.
_pal3b = sns.color_palette()
_pal170m = sns.color_palette("deep")
_blues = sns.color_palette("Blues", 6)
_reds = sns.color_palette("Reds", 6)
DISPLAY = {
    "dayhoff-170m-uniref50": "170m-UR50",
    "dayhoff-170m-uniref90": "170m-UR90",
    "dayhoff-170m-gigaref": "170m-GR",
    "dayhoff-170m-nofilter": "170m-UR50-BRu",
    "dayhoff-170m-rmsd": "170m-UR50-BRq",
    "dayhoff-170m-bothfilter": "170m-UR50-BRn",
    "dayhoff-3b-uniref90": "3b-UR90",
    "dayhoff-3b-msa-gigaref": "3b-GR-HM",
    "dayhoff-3b-msa-uniref90-cooldown": "3b-GR-HM-c",
}
COLORS = {
    "dayhoff-170m-uniref50": _pal170m[7],
    "dayhoff-170m-gigaref": _pal170m[4],
    # BackboneRef variants: light -> dark blue
    "dayhoff-170m-nofilter": _blues[2],   # BRu
    "dayhoff-170m-rmsd": _blues[3],       # BRq
    "dayhoff-170m-bothfilter": _blues[5], # BRn
    # UR90 models: light -> dark red
    "dayhoff-170m-uniref90": _reds[3],
    "dayhoff-3b-uniref90": _reds[5],
    "dayhoff-3b-msa-gigaref": _pal3b[1],
    "dayhoff-3b-msa-uniref90-cooldown": sns.color_palette("pastel")[1],
}


def load_selection_types() -> pd.DataFrame:
    """Return DataFrame[assay, dms, selection_type] from the AUC DMS_level files."""
    frames = []
    for dms in DMSS:
        f = os.path.join(BENCH_ROOT, dms, "AUC", f"DMS_{dms}_AUC_DMS_level.csv")
        df = pd.read_csv(f)[["DMS ID", "Selection Type"]].copy()
        df.columns = ["assay", "selection_type"]
        df["dms"] = dms
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out["assay"] = out["assay"].str.strip()
    return out


def load_model_spearman(model: str) -> pd.DataFrame:
    """Return DataFrame[assay, en_spearman] for a model by recomputing the
    ensemble Spearman from the reformatted per-mutant CSVs.

    en_spearman = spearman(<model>_score, DMS_score). The '<model>_score' column
    is the model's final ensemble score (fw+bw for seq models; seq+indel for MSA
    models), so this reproduces the 'en_spearman' metric without relying on the
    malformed rank-summary CSVs.
    """
    model_dir = os.path.join(REFORMATTED, model)
    if not os.path.isdir(model_dir):
        return pd.DataFrame(columns=["assay", "en_spearman"])
    score_col = f"{model}_score"
    recs = []
    for fname in os.listdir(model_dir):
        if not fname.endswith(".csv") or fname[0].isdigit():
            continue  # skip rank-summary files like 0.csv
        assay = fname[:-4]
        df = pd.read_csv(os.path.join(model_dir, fname))
        if score_col not in df.columns or "DMS_score" not in df.columns:
            continue
        x = pd.to_numeric(df[score_col], errors="coerce")
        y = pd.to_numeric(df["DMS_score"], errors="coerce")
        m = x.notna() & y.notna()
        if m.sum() < 2:
            continue
        rho = spearmanr(x[m], y[m]).statistic
        recs.append({"assay": assay.strip(), "en_spearman": rho})
    return pd.DataFrame(recs)


def build_long() -> pd.DataFrame:
    """Long table: model, dms, assay, en_spearman, selection_type, is_stability."""
    sel = load_selection_types()
    rows = []
    for model in MODELS:
        sp = load_model_spearman(model)
        if sp.empty:
            continue
        sp["model"] = model
        rows.append(sp)
    long = pd.concat(rows, ignore_index=True)
    # dms + selection_type both come from the reference map (assay ids are
    # disjoint between substitutions and indels).
    long = long.merge(sel, on="assay", how="left")
    long["is_stability"] = long["selection_type"].str.contains(
        "Stability", case=False, na=False)
    return long


def summarize(long: pd.DataFrame) -> pd.DataFrame:
    """One row per (model, dms, subset) with mean/median/n and delta-vs-baseline."""
    recs = []
    for dms in DMSS + ["pooled"]:
        sub = long if dms == "pooled" else long[long["dms"] == dms]
        for subset, mask in [
            ("stability", sub["is_stability"]),
            ("non_stability", ~sub["is_stability"] & sub["selection_type"].notna()),
            ("all", sub["selection_type"].notna()),
        ]:
            grp = sub[mask]
            for model in MODELS:
                g = grp[grp["model"] == model]
                if g.empty:
                    continue
                recs.append({
                    "model": model,
                    "role": ROLE.get(model, ""),
                    "dms": dms,
                    "subset": subset,
                    "n_assays": len(g),
                    "mean_spearman": g["en_spearman"].mean(),
                    "median_spearman": g["en_spearman"].median(),
                })
    res = pd.DataFrame(recs)
    # delta vs baseline within each (dms, subset)
    base = (res[res["model"] == BASELINE]
            .set_index(["dms", "subset"])["mean_spearman"])
    res["baseline_mean"] = res.apply(
        lambda r: base.get((r["dms"], r["subset"]), np.nan), axis=1)
    res["delta_vs_baseline"] = res["mean_spearman"] - res["baseline_mean"]
    return res


def print_table(res: pd.DataFrame) -> None:
    pd.set_option("display.width", 160)
    pd.set_option("display.max_rows", None)
    for dms in DMSS + ["pooled"]:
        print(f"\n{'='*78}\nStability subset -- {dms}\n{'='*78}")
        t = (res[(res["dms"] == dms) & (res["subset"] == "stability")]
             .sort_values("mean_spearman", ascending=False)
             [["model", "role", "n_assays", "mean_spearman",
               "median_spearman", "delta_vs_baseline"]])
        print(t.to_string(index=False,
                          float_format=lambda x: f"{x:0.4f}"))


SELECTION_TYPES = ["Stability", "Activity", "Binding", "Expression",
                   "OrganismalFitness"]


def summarize_by_type(long: pd.DataFrame) -> pd.DataFrame:
    """One row per (model, selection_type) pooled over dms, with mean Spearman,
    n_assays, and delta-vs-baseline."""
    recs = []
    valid = long[long["selection_type"].notna()]
    for st in SELECTION_TYPES:
        grp = valid[valid["selection_type"] == st]
        for model in MODELS:
            g = grp[grp["model"] == model]
            if g.empty:
                continue
            recs.append({
                "model": model,
                "role": ROLE.get(model, ""),
                "selection_type": st,
                "n_assays": len(g),
                "mean_spearman": g["en_spearman"].mean(),
                "std_spearman": g["en_spearman"].std(ddof=1),
            })
    res = pd.DataFrame(recs)
    base = (res[res["model"] == BASELINE]
            .set_index("selection_type")["mean_spearman"])
    res["baseline_mean"] = res["selection_type"].map(base)
    res["delta_vs_baseline"] = res["mean_spearman"] - res["baseline_mean"]
    return res


def _boxstrip(ax, df, order):
    """Draw the box+strip overlay for one dms split onto `ax`."""
    hue_order = [DISPLAY[m] for m in MODELS]
    palette = {DISPLAY[m]: COLORS[m] for m in MODELS}
    sns.boxplot(data=df, x="selection_type", y="en_spearman", hue="Model",
                order=order, hue_order=hue_order, palette=palette,
                showfliers=False, linewidth=0.8, ax=ax,
                boxprops={"alpha": 0.45})
    sns.stripplot(data=df, x="selection_type", y="en_spearman", hue="Model",
                  order=order, hue_order=hue_order, palette=palette,
                  dodge=True, size=2.5, alpha=0.7, linewidth=0,
                  jitter=0.15, ax=ax, legend=False)
    n = df.groupby("selection_type")["assay"].nunique()
    ax.set_xticklabels([f"{st}\n(n={int(n[st])})" for st in order], fontsize=14)
    ax.tick_params(axis="y", labelsize=14)
    ax.set_xlabel("")
    ax.set_ylabel("Spearman correlation (per assay)", fontsize=15)
    ax.axhline(0, color="0.6", lw=0.8, ls="--")
    ax.grid(axis="y", ls=":", alpha=0.4)


def make_boxplot(long: pd.DataFrame, out_png: str) -> None:
    """Box-and-whisker plot of the per-assay Spearman distribution: x = selection
    type, one box per model within each group, with substitutions and indels in
    separate panels. Center line = median; box = IQR (Q1-Q3); whiskers = 1.5*IQR;
    individual per-assay points overlaid."""
    df = long[long["selection_type"].notna()].copy()
    df["Model"] = df["model"].map(DISPLAY)

    fig, axes = plt.subplots(
        len(DMSS), 1, figsize=(2.4 * len(SELECTION_TYPES) + 4, 6.5 * len(DMSS)))
    for ax, dms in zip(np.atleast_1d(axes), DMSS):
        sub = df[df["dms"] == dms]
        # only show categories that actually have assays in this split
        order = [st for st in SELECTION_TYPES
                 if (sub["selection_type"] == st).any()]
        _boxstrip(ax, sub, order)
        ax.set_title(f"{dms} (n={sub['assay'].nunique()} assays)", fontsize=15)

    # single shared legend (one entry per model) on the top panel
    top = np.atleast_1d(axes)[0]
    handles, labels = top.get_legend_handles_labels()
    top.legend(handles[:len(MODELS)], labels[:len(MODELS)],
               fontsize=10, ncol=2, loc="upper right", framealpha=0.9, title=None)
    for ax in np.atleast_1d(axes)[1:]:
        leg = ax.get_legend()
        if leg is not None:
            leg.remove()

    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    print(f"Saved boxplot -> {out_png}")


def main() -> None:
    long = build_long()
    res = summarize(long)
    out_csv = os.path.join(DATA_ROOT, "stability_backboneref_summary.csv")
    res.to_csv(out_csv, index=False)
    print(f"Saved summary CSV -> {out_csv}")
    print_table(res)

    by_type = summarize_by_type(long)
    by_type_csv = os.path.join(DATA_ROOT, "selection_type_summary.csv")
    by_type.to_csv(by_type_csv, index=False)
    print(f"Saved selection-type CSV -> {by_type_csv}")
    make_boxplot(long, os.path.join(DATA_ROOT, "selection_types_boxplot.png"))


if __name__ == "__main__":
    main()
