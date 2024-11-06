import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import pandas as pd
import os

def plot_perplexity_vs_plddt(df,
                             temp_col='if_temp',
                             perp_col='proteinmpnnperplexity',
                             plddt_col='esmfoldplddt',
                             hue_col=None,
                             palette='viridis',
                             facet=False,
                             fold='esmfold'):
   
    # use white style to remove grid
    sns.set(style='white', font_scale=1.1)

    if facet:
        # swap axes: pLDDT on x, perplexity on y; remove grid via white style
        g = sns.FacetGrid(df, col=temp_col, sharex=True, sharey=True, palette=palette)
        g.map_dataframe(sns.scatterplot, x=plddt_col, y=perp_col)
        g.set_axis_labels(f"{fold.capitalize()} pLDDT", "Protein MPNN Perplexity")
        g.fig.suptitle(f"{fold.capitalize()} pLDDT vs MPNN Perplexity by IF Temperature", y=1.02)
        return g
    else:
        plt.figure(figsize=(8,6))
        # color points by hue_col if provided, no legend
        kwargs = {
            'data': df,
            'y': perp_col,
            'x': plddt_col,
            'edgecolor': 'w',
            'linewidth': 0.5,
            'palette': palette,
            'legend': False
        }
        if hue_col:
            kwargs['hue'] = hue_col
        ax = sns.scatterplot(**kwargs)
        ax.set_title(f"{fold.capitalize()} pLDDT vs MPNN Perplexity")
        ax.set_ylabel("Protein MPNN Perplexity")
        ax.set_xlabel(f"{fold.capitalize()} pLDDT")
        plt.tight_layout()
        return ax.get_figure()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot proteinmpnnperplexity vs esmfoldplddt"
    )
    parser.add_argument(
        "--input_file",
        type=str,
        default="dcp_merged/0/esmfold_proteinmpnn_merge_data.csv",
        help="Path to input CSV (will be updated based on --fold)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="dcp_merged/0",
        help="Directory to save plot"
    )
    parser.add_argument(
        "--facet",
        action="store_true",
        help="Use FacetGrid by if_temp"
    )
    parser.add_argument(
        "--only_if_temp",
        action="store_true",
        help="Only include rows where if_temp == 1.0"
    )
    parser.add_argument(
        "--merge_all",
        action="store_true",
        help="Merge CSVs in all subfolders under input_dir"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default=None,
        help="Directory containing subfolders with CSVs (defaults to parent of parent of input_file when --merge_all)"
    )
    parser.add_argument(
        "--hue_col",
        type=str,
        default=None,
        help="Column name to color points by (no legend will be shown)"
    )
    parser.add_argument(
        "--fold",
        choices=["esmfold", "omegafold"],
        default="esmfold",
        help="Which fold dataset to use (controls CSV name and pLDDT column)"
    )
    args = parser.parse_args()

    # adjust input_file to use selected fold
    base_dir = os.path.dirname(args.input_file)
    filename = f"{args.fold}_proteinmpnn_merge_data.csv"
    args.input_file = os.path.join(base_dir, filename)

    # Load data: either a single CSV or merge all subfolder CSVs
    if args.merge_all:
        # determine input directory if not provided
        if not args.input_dir:
            # parent of parent of input_file
            args.input_dir = os.path.dirname(os.path.dirname(args.input_file))
        filename = os.path.basename(args.input_file)
        dfs = []
        # only consider subdirectories (e.g., 0,1,2,3 folders)
        for sub in sorted(os.listdir(args.input_dir)):
            sub_dir = os.path.join(args.input_dir, sub)
            if not os.path.isdir(sub_dir):
                continue
            path = os.path.join(sub_dir, filename)
            if os.path.isfile(path):
                tmp = pd.read_csv(path)
                tmp['source'] = sub
                dfs.append(tmp)
            else:
                print(f"Warning: {path} not found, skipping.")
        if not dfs:
            raise FileNotFoundError(f"No CSV files found in subfolders of {args.input_dir}")
        df = pd.concat(dfs, ignore_index=True)
    else:
        df = pd.read_csv(args.input_file)
    # Optionally filter for only if_temp == 1.0
    if args.only_if_temp:
        df = df[df.get('if_temp', None) == 1.0]
    # compute and output mean values
    print(f"Rows evaluated {len(df)}")
    plddt_col = f"{args.fold}plddt"
    mean_plddt = df[plddt_col].mean()
    mean_mpnn = df['proteinmpnnperplexity'].mean()
    print(f"Mean {args.fold.capitalize()} pLDDT: {mean_plddt:.3f}")
    print(f"Mean Protein MPNN Perplexity: {mean_mpnn:.3f}")
    # plot and save (pass plddt_col and fold for dynamic labeling)
    fig = plot_perplexity_vs_plddt(
        df,
        hue_col=args.hue_col,
        facet=args.facet,
        plddt_col=plddt_col,
        fold=args.fold
    )
    os.makedirs(args.output_dir, exist_ok=True)
    # include fold method in output filename
    out_filename = f"perplexity_vs_plddt_{args.fold}.png"
    out_path = os.path.join(args.output_dir, out_filename)
    fig.savefig(out_path, dpi=150)