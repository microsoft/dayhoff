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
import itertools

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
    "dayhoff-170m-grs",           # GigaRef singletons
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
    "dayhoff-170m-grs": "170m-GR-s",
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
    "dayhoff-170m-grs": _pal3b[6],  # GigaRef-singletons pink (matches FPD figure)
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
    """Return DataFrame[assay, dms, selection_type, uniprot_id] from the AUC
    DMS_level reference files (ProteinGym's DMS->UniProt/function mapping)."""
    frames = []
    for dms in DMSS:
        f = os.path.join(BENCH_ROOT, dms, "AUC", f"DMS_{dms}_AUC_DMS_level.csv")
        df = pd.read_csv(f)[["DMS ID", "Selection Type", "UniProt ID"]].copy()
        df.columns = ["assay", "selection_type", "uniprot_id"]
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


def proteingym_average(long: pd.DataFrame) -> pd.DataFrame:
    """Replicate ProteinGym's official aggregation of per-assay Spearman into a
    single headline number per (model, dms), matching the paper's Table.

    Method (proteingym/performance_DMS_benchmarks.py, lines ~297-309):
      1. per-DMS Spearman  (already in `long['en_spearman']`, which for these
         single-sequence models is spearman((fw+bw)/2 score, DMS_score))
      2. average within each UniProt ID
      3. average those within each Selection Type (5 functional categories)
      4. average the 5 category means -> ProteinGym 'Average Spearman'
    A 'pooled' row applies the same recipe over substitutions + indels together.
    """
    recs = []
    valid = long[long["selection_type"].notna() & long["uniprot_id"].notna()]
    for dms in DMSS + ["pooled"]:
        sub = valid if dms == "pooled" else valid[valid["dms"] == dms]
        for model in MODELS:
            g = sub[(sub["model"] == model) & sub["en_spearman"].notna()]
            if g.empty:
                continue
            uni = g.groupby(["uniprot_id", "selection_type"])["en_spearman"].mean()
            per_cat = uni.groupby("selection_type").mean()
            recs.append({
                "model": model,
                "display": DISPLAY.get(model, model),
                "role": ROLE.get(model, ""),
                "dms": dms,
                "n_assays": len(g),
                "proteingym_spearman": per_cat.mean(),
            })
    res = pd.DataFrame(recs)
    base = (res[res["model"] == BASELINE].set_index("dms")["proteingym_spearman"])
    res["baseline"] = res["dms"].map(base)
    res["delta_vs_baseline"] = res["proteingym_spearman"] - res["baseline"]
    return res


def print_proteingym(res: pd.DataFrame) -> None:
    pd.set_option("display.width", 160)
    pd.set_option("display.max_rows", None)
    for dms in DMSS + ["pooled"]:
        print(f"\n{'='*70}\nProteinGym Spearman (official aggregation) -- {dms}\n{'='*70}")
        t = (res[res["dms"] == dms]
             .sort_values("proteingym_spearman", ascending=False)
             [["display", "role", "n_assays", "proteingym_spearman",
               "delta_vs_baseline"]])
        print(t.to_string(index=False, float_format=lambda x: f"{x:0.3f}"))


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


def _pivot_rho(long, dms, st):
    """assay x model matrix of per-assay Spearman for one (dms, selection_type)."""
    d = long[(long["dms"] == dms) & (long["selection_type"] == st)]
    return d.pivot_table(index="assay", columns="model",
                         values="en_spearman", aggfunc="mean")


def pairwise_stats(long: pd.DataFrame, min_pairs: int = 6) -> pd.DataFrame:
    """All pairwise paired Wilcoxon signed-rank tests between models within each
    (dms, selection_type), over their common assays, with BH-FDR correction
    applied within each (dms, selection_type) family."""
    from scipy.stats import wilcoxon
    try:
        from statsmodels.stats.multitest import multipletests
        def fdr(p):
            p = np.asarray(p, float)
            ok = ~np.isnan(p)
            out = np.full_like(p, np.nan)
            if ok.sum():
                out[ok] = multipletests(p[ok], method="fdr_bh")[1]
            return out
    except Exception:                       # BH-FDR fallback (no statsmodels)
        def fdr(p):
            p = np.asarray(p, float); out = np.full_like(p, np.nan)
            idx = np.where(~np.isnan(p))[0]
            if len(idx) == 0:
                return out
            order = idx[np.argsort(p[idx])]
            m = len(order); prev = 1.0
            for rank, i in enumerate(order[::-1]):
                k = m - rank
                prev = min(prev, p[i] * m / k)
                out[i] = prev
            return out

    present = list(MODELS)
    recs = []
    for dms in DMSS:
        for st in SELECTION_TYPES:
            mat = _pivot_rho(long, dms, st)
            if mat.empty:
                continue
            local = []
            for a, b in itertools.combinations(present, 2):
                if a not in mat or b not in mat:
                    continue
                pair = mat[[a, b]].dropna()
                n = len(pair)
                if n < min_pairs or np.allclose(pair[a], pair[b]):
                    stat, p = np.nan, np.nan
                else:
                    stat, p = wilcoxon(pair[a], pair[b])
                local.append({
                    "dms": dms, "selection_type": st,
                    "model_a": DISPLAY.get(a, a), "model_b": DISPLAY.get(b, b),
                    "n_pairs": n,
                    "median_diff": float((pair[a] - pair[b]).median()) if n else np.nan,
                    "wilcoxon_stat": stat, "p_raw": p,
                })
            if local:
                ps = [r["p_raw"] for r in local]
                for r, q in zip(local, fdr(ps)):
                    r["p_fdr"] = q
                recs.extend(local)
    return pd.DataFrame(recs)


def make_pairwise_heatmaps(stats: pd.DataFrame, out_prefix: str):
    """One N x N model grid per (dms, selection_type) showing DISCRETE signed
    significance tiers from the pairwise paired-Wilcoxon tests.

    Cell (row i, col j): tier = sign(median rho_i - rho_j) * T, where T is
      3 (p<0.001, '***'), 2 (p<0.01, '**'), 1 (p<0.05, '*'), 0 (n.s.). So
      BLUE shades = row model significantly BETTER than column model,
      RED  shades = row model significantly WORSE,
      WHITE = not significant (p>=0.05) or diagonal.
    Uncorrected (raw) paired-Wilcoxon p-values. Darker = smaller p (stronger
    evidence). One PNG per dms."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap

    order = [DISPLAY[m] for m in MODELS]
    idx = {m: k for k, m in enumerate(order)}

    def tier(q):
        if not np.isfinite(q) or q >= 0.05:
            return 0
        if q < 0.001:
            return 3
        if q < 0.01:
            return 2
        return 1

    # 7 discrete colors for signed tiers -3..+3 sampled from RdBu
    # (red=worse, white=n.s., blue=better).
    base = plt.cm.RdBu(np.linspace(0, 1, 7))
    base[3] = (1.0, 1.0, 1.0, 1.0)          # n.s. tier -> pure white (match diagonal)
    cmap = ListedColormap(base)
    cmap.set_bad("white")
    bounds = np.arange(-3.5, 4.5, 1.0)
    norm = BoundaryNorm(bounds, cmap.N)

    for dms in DMSS:
        sd = stats[stats["dms"] == dms]
        sts = [s for s in SELECTION_TYPES if s in set(sd["selection_type"])]
        if not sts:
            continue
        ncol = min(3, len(sts))
        nrow = int(np.ceil(len(sts) / ncol))
        fig, axes = plt.subplots(nrow, ncol,
                                 figsize=(4.6 * ncol, 4.4 * nrow),
                                 squeeze=False,
                                 gridspec_kw={"hspace": 0.85, "wspace": 0.35})
        im = None
        for a_i, st in enumerate(sts):
            ax = axes[a_i // ncol][a_i % ncol]
            M = np.full((len(order), len(order)), np.nan)
            g = sd[sd["selection_type"] == st]
            for _, r in g.iterrows():
                a, b = r["model_a"], r["model_b"]
                if a not in idx or b not in idx:
                    continue
                q, md = r["p_raw"], r["median_diff"]
                if not np.isfinite(q) or not np.isfinite(md):
                    continue
                val = np.sign(md) * tier(q)
                M[idx[a], idx[b]] = val          # row a vs col b
                M[idx[b], idx[a]] = -val          # symmetric, flipped sign
            im = ax.imshow(M, cmap=cmap, norm=norm, aspect="auto")
            npairs = int(g["n_pairs"].max()) if len(g) else 0
            ax.set_title(f"{st} (n={npairs})", fontsize=11, pad=6)
            ax.set_xticks(range(len(order)))
            ax.set_yticks(range(len(order)))
            ax.set_xticklabels(order, rotation=90, fontsize=7)
            ax.set_yticklabels(order, fontsize=7)
            ax.set_xticks(np.arange(-.5, len(order), 1), minor=True)
            ax.set_yticks(np.arange(-.5, len(order), 1), minor=True)
            ax.grid(which="minor", color="0.85", lw=0.5)
            ax.tick_params(which="minor", length=0)
        for a_i in range(len(sts), nrow * ncol):
            axes[a_i // ncol][a_i % ncol].axis("off")
        if im is not None:
            cax = fig.add_axes([0.93, 0.30, 0.014, 0.40])
            cbar = fig.colorbar(im, cax=cax, ticks=range(-3, 4))
            cbar.set_ticklabels(["*** worse", "** worse", "* worse", "n.s.",
                                 "* better", "** better", "*** better"])
            cbar.ax.tick_params(labelsize=8)
            cbar.set_label("row-vs-col significance", fontsize=9)
        fig.suptitle(f"Pairwise paired-Wilcoxon significance - {dms}",
                     fontsize=13, y=0.94)
        fig.subplots_adjust(top=0.88, right=0.90)
        out_png = f"{out_prefix}_{dms}.png"
        fig.savefig(out_png, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved pairwise heatmap -> {out_png}")



def _compact_letters(models, sig_pairs):
    """Compact letter display: assign letters so two models share a letter iff they
    are NOT significantly different. `sig_pairs` = set of frozenset({a,b}) that ARE
    significant. Piepho (2004) insert-absorb sweep over the significant pairs."""
    cols = [set(models)]
    for pair in sorted([tuple(sorted(p)) for p in sig_pairs]):
        a, b = pair
        new = []
        for c in cols:
            if a in c and b in c:
                new.append(c - {a})
                new.append(c - {b})
            else:
                new.append(c)
        # absorb: drop any column that is a subset of another
        new = [c for c in new if c and not any(
            c < o for o in new if c is not o)]
        # dedupe
        uniq = []
        for c in new:
            if c not in uniq:
                uniq.append(c)
        cols = uniq
    cols.sort(key=lambda c: (-len(c), sorted(models.index(m) for m in c)))
    letters = {m: "" for m in models}
    for k, c in enumerate(cols):
        ch = chr(ord("a") + k)
        for m in c:
            letters[m] += ch
    return letters


def _boxstrip(ax, df, order, letters_by_cat=None):
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

    # Compact-letter-display annotations above each box.
    if letters_by_cat:
        k = len(MODELS)
        width = 0.8
        for i, st in enumerate(order):
            letters = letters_by_cat.get(st, {})
            for j, m in enumerate(MODELS):
                lab = letters.get(m, "")
                if not lab:
                    continue
                g = df[(df["selection_type"] == st) & (df["model"] == m)]
                if g.empty:
                    continue
                x = i - width / 2 + (j + 0.5) * (width / k)
                y = g["en_spearman"].max() + 0.02
                ax.text(x, y, lab, ha="center", va="bottom", fontsize=7,
                        color="0.15", rotation=90)


def make_boxplot(long: pd.DataFrame, out_png: str,
                 stats: pd.DataFrame = None) -> None:
    """Box-and-whisker plot of the per-assay Spearman distribution: x = selection
    type, one box per model within each group, with substitutions and indels in
    separate panels. Center line = median; box = IQR (Q1-Q3); whiskers = 1.5*IQR;
    individual per-assay points overlaid. If `stats` is provided, boxes are
    annotated with a compact letter display (shared letter => not significantly
    different, paired Wilcoxon + BH-FDR, alpha=0.05)."""
    df = long[long["selection_type"].notna()].copy()
    df["Model"] = df["model"].map(DISPLAY)
    disp2model = {DISPLAY[m]: m for m in MODELS}

    def letters_for(dms, st):
        if stats is None:
            return None
        s = stats[(stats["dms"] == dms) & (stats["selection_type"] == st)]
        if s.empty:
            return None
        sig = {frozenset({disp2model[r.model_a], disp2model[r.model_b]})
               for r in s.itertuples() if pd.notna(r.p_fdr) and r.p_fdr < 0.05}
        return _compact_letters(list(MODELS), sig)

    fig, axes = plt.subplots(
        len(DMSS), 1, figsize=(2.4 * len(SELECTION_TYPES) + 4, 6.5 * len(DMSS)))
    for ax, dms in zip(np.atleast_1d(axes), DMSS):
        sub = df[df["dms"] == dms]
        # only show categories that actually have assays in this split
        order = [st for st in SELECTION_TYPES
                 if (sub["selection_type"] == st).any()]
        _boxstrip(ax, sub, order, letters_by_cat=None)
        ax.set_title(f"{dms} (n={sub['assay'].nunique()} assays)", fontsize=15)

    # single shared legend (one entry per model), placed OUTSIDE the top panel
    top = np.atleast_1d(axes)[0]
    handles, labels = top.get_legend_handles_labels()
    top.legend(handles[:len(MODELS)], labels[:len(MODELS)],
               fontsize=10, ncol=1, loc="upper left",
               bbox_to_anchor=(1.01, 1.0), framealpha=0.9, title=None,
               borderaxespad=0.0)
    for ax in np.atleast_1d(axes)[1:]:
        leg = ax.get_legend()
        if leg is not None:
            leg.remove()

    fig.tight_layout(rect=(0, 0, 0.86, 1))
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"Saved boxplot -> {out_png}")


def summarize_overall(long: pd.DataFrame) -> pd.DataFrame:
    """One row per (model, dms) over ALL assays (no selection-type filter), plus a
    'pooled' row, reporting mean/median Spearman, n_assays, and delta-vs-baseline.
    This is the overall substitution- and indel-level performance summary."""
    recs = []
    for dms in DMSS + ["pooled"]:
        sub = long if dms == "pooled" else long[long["dms"] == dms]
        for model in MODELS:
            g = sub[sub["model"] == model]
            g = g[g["en_spearman"].notna()]
            if g.empty:
                continue
            recs.append({
                "model": model,
                "display": DISPLAY.get(model, model),
                "role": ROLE.get(model, ""),
                "dms": dms,
                "n_assays": len(g),
                "mean_spearman": g["en_spearman"].mean(),
                "median_spearman": g["en_spearman"].median(),
            })
    res = pd.DataFrame(recs)
    base = (res[res["model"] == BASELINE].set_index("dms")["mean_spearman"])
    res["baseline_mean"] = res["dms"].map(base)
    res["delta_vs_baseline"] = res["mean_spearman"] - res["baseline_mean"]
    return res


def print_overall(res: pd.DataFrame) -> None:
    pd.set_option("display.width", 160)
    pd.set_option("display.max_rows", None)
    for dms in DMSS + ["pooled"]:
        print(f"\n{'='*78}\nOverall Spearman -- {dms} (all assays)\n{'='*78}")
        t = (res[res["dms"] == dms]
             .sort_values("mean_spearman", ascending=False)
             [["display", "role", "n_assays", "mean_spearman",
               "median_spearman", "delta_vs_baseline"]])
        print(t.to_string(index=False,
                          float_format=lambda x: f"{x:0.4f}"))


def main() -> None:
    long = build_long()
    res = summarize(long)
    out_csv = os.path.join(DATA_ROOT, "stability_backboneref_summary.csv")
    res.to_csv(out_csv, index=False)
    print(f"Saved summary CSV -> {out_csv}")
    print_table(res)

    overall = summarize_overall(long)
    overall_csv = os.path.join(DATA_ROOT, "overall_spearman_summary.csv")
    overall.to_csv(overall_csv, index=False)
    print(f"Saved overall subs/indels CSV -> {overall_csv}")
    print_overall(overall)

    pg = proteingym_average(long)
    pg_csv = os.path.join(DATA_ROOT, "proteingym_spearman_summary.csv")
    pg.to_csv(pg_csv, index=False)
    print(f"Saved ProteinGym-aggregated CSV -> {pg_csv}")
    print_proteingym(pg)

    by_type = summarize_by_type(long)
    by_type_csv = os.path.join(DATA_ROOT, "selection_type_summary.csv")
    by_type.to_csv(by_type_csv, index=False)
    print(f"Saved selection-type CSV -> {by_type_csv}")

    stats = pairwise_stats(long)
    stats_csv = os.path.join(DATA_ROOT, "pairwise_wilcoxon_by_category.csv")
    stats.to_csv(stats_csv, index=False)
    print(f"Saved pairwise Wilcoxon CSV -> {stats_csv}")

    make_pairwise_heatmaps(
        stats, os.path.join(DATA_ROOT, "pairwise_wilcoxon_heatmap"))

    make_boxplot(long, os.path.join(DATA_ROOT, "selection_types_boxplot.png"),
                 stats=stats)


if __name__ == "__main__":
    main()
