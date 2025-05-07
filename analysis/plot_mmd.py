import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
import pandas as pd
from glob import glob
from collections import defaultdict
import argparse
import os


sns.set_theme(font_scale=1.2)
sns.set_style('white')

FNAME_RUNNING_STATS_PDF         = 'mmd_results_running_stats.pdf'
FNAME_BASELINES_STATS_PDF       = 'mmd_results_baselines_running_stats.pdf'
FNAME_MEDIAN_HEURISTIC_PDF      = 'median_heuristic_running_stats.pdf'
FNAME_FINAL_RESULTS_PDF         = 'mmd_results.pdf'
FNAME_RESULTS_PARQUET           = 'mmd_results.parquet'
NAME_MAPPING = {
                "GIGAREF_CLUSTERED_10M_second_half_GIGAREF_CLUSTERED_10M_first_half": "GGR vs. GGR",
                "UNIREF50_10M_second_half_UNIREF50_10M_first_half": "UR50 vs. UR50",
                "GIGAREF_SINGLETONS_10M": "GGR-s",
                "GIGAREF_CLUSTERED_10M":"GGR",
                "UNIREF50_10M":"UR50",
                    "RFDIFFUSION_UNFILTERED":"BBR-u",
                "RFDIFFUSION_BOTH_FILTER":"BBR-n",
                "RFDIFFUSION_SCRMSD":"BBR-s",
                "DAYHOFF":"DR",
                "_": " vs. "
                }


def result_to_historic(mmd_results):
    results_historic_df = []
    for scenario, stats in mmd_results.items():
        c=1
        for mean,std_error in zip(stats['means'],stats['std_errors']):
            results_historic_df.append({"scenario":scenario,"mean":mean,"std_error":std_error,'sample':c}) 
            c+=1
        
    results_historic_df = pd.DataFrame(results_historic_df).replace(NAME_MAPPING,
                            regex=True
    )
    results_historic_df['dataset'] = results_historic_df['scenario'].apply(lambda x: x.split(' ')[-1])
    results_historic_df['model'] = results_historic_df['scenario'].apply(lambda x: x.split(' ')[0])

    return results_historic_df

def main(args):
    mmd_results = defaultdict(dict)
    for filename in glob(args.mmd_pattern):
        name = filename.split('/')[-1].split('.')[0].replace('mmd_','').replace('GENERATIONS_','')
        if "means" in name:
            mmd_results[name.replace('means_','')]['means'] = np.load(filename)
        elif "std_errors" in name:
            mmd_results[name.replace('std_errors_','')]['std_errors'] = np.load(filename)



    sigma_results = defaultdict(dict)
    for filename in glob(args.median_pattern):
        name = filename.split('/')[-1].split('.')[0].replace('median_dist_','').replace('GENERATIONS_','')
        if "means" in name:
            sigma_results[name.replace('means_','')]['means'] = np.load(filename)
        elif "std_errors" in name:
            sigma_results[name.replace('std_errors_','')]['std_errors'] = np.load(filename)

    mmd_results_historic_df = result_to_historic(mmd_results)
    sigma_results_historic_df = result_to_historic(sigma_results)

    ## Plotting the MMD results ##

    data = mmd_results_historic_df

    palette = sns.color_palette("tab10")
    models = data['model'].unique()
    model_color_map = dict(zip(models, palette))

    def plot_facet(data, color, **kwargs):
        ax = plt.gca()
        for model, group in data.groupby("model"):
            ax.plot(group['sample'], group['mean'], label=model, color=model_color_map[model])
            ax.fill_between(
                group['sample'],
                group['mean'] - group['std_error'] * 1.96,
                group['mean'] + group['std_error'] * 1.96,
                alpha=0.3,
                color=model_color_map[model]
            )
        ax.set_xlabel("Trial")
        ax.set_ylabel("MMD")

    g = sns.FacetGrid(data, col="dataset", height=4, aspect=1)
    g.map_dataframe(plot_facet)
    g.set_titles(col_template="Source B: {col_name}")
    plt.subplots_adjust(top=0.85)

    legend_elements = [
        Line2D([0], [0], color=model_color_map[model], lw=2, label=model)
        for model in models
    ]
    g.fig.legend(
        handles=legend_elements,
        title="Source A",
        loc='upper center',
        bbox_to_anchor=(0.5, -0.02),
        ncol=len(models)
    )

    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, FNAME_RUNNING_STATS_PDF), bbox_inches='tight', dpi=300)
    plt.show()



    ## Plotting the MMD results ##

    data = mmd_results_historic_df.query("scenario == 'GGR vs. GGR' or scenario == 'UR50 vs. UR50'")

    palette = sns.color_palette("tab10")
    models = data['model'].unique()
    model_color_map = dict(zip(models, palette))

    ax = plt.gca()  
    for model, group in data.groupby("model"):
        ax.plot(group['sample'], group['mean'], label=group['scenario'].unique()[0], color=model_color_map[model] )
        ax.fill_between(
            group['sample'],
            group['mean'] - group['std_error'] * 1.96,
            group['mean'] + group['std_error'] * 1.96,
            alpha=0.3,
            color=model_color_map[model]
        )
    ax.set_xlabel("Trial")
    ax.set_ylabel("MMD")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, FNAME_BASELINES_STATS_PDF), bbox_inches='tight', dpi=300)
    plt.show()



    ## Plotting the median distance estimates ##
    data = sigma_results_historic_df
    palette = sns.color_palette("tab10")
    models = data['model'].unique()
    model_color_map = dict(zip(models, palette))
    def plot_facet(data, color, **kwargs):
        ax = plt.gca() 
        for model, group in data.groupby("model"):
            ax.plot(group['sample'], group['mean'], label=model, color=model_color_map[model])
            ax.fill_between(
                group['sample'],
                group['mean'] - group['std_error'] * 1.96,
                group['mean'] + group['std_error'] * 1.96,
                alpha=0.3,
                color=model_color_map[model]
            )
        ax.set_xlabel("Trial")
        ax.set_ylabel("Median distance")

    g = sns.FacetGrid(data, col="dataset", height=4, aspect=1)
    g.map_dataframe(plot_facet)
    g.set_titles(col_template="Source B: {col_name}")
    plt.subplots_adjust(top=0.85)
    legend_elements = [
        Line2D([0], [0], color=model_color_map[model], lw=2, label=model)
        for model in models
    ]
    g.fig.legend(
        handles=legend_elements,
        title="Source A",
        loc='upper center',
        bbox_to_anchor=(0.5, -0.02),
        ncol=len(models)
    )
    plt.xlim(-50,1_000)
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, FNAME_MEDIAN_HEURISTIC_PDF), bbox_inches='tight', dpi=300)
    plt.show()

    ## Plot Final MMD Results Bar Charts ##

    mmd_results_df = pd.DataFrame([{'scenario':d,'mean':mmd_results[d]['means'][-1], 'std_error':mmd_results[d]['std_errors'][-1]} for d in mmd_results.keys()])
    mmd_results_df['95% CI'] = mmd_results_df.apply(lambda x: (x['mean'] - 1.96*x['std_error'], x['mean'] + 1.96*x['std_error']), axis=1)

    mmd_results_df = mmd_results_df.replace(NAME_MAPPING,
                            regex=True
    ).sort_values(by='mean', ascending=True)

    mmd_results_df.to_parquet(os.path.join(args.out_dir, FNAME_RESULTS_PARQUET),index=False)

    plt.figure(figsize=(13, 5))
    sns.barplot(x='scenario', y='mean', data=mmd_results_df, color="grey", errorbar=None)
    plt.errorbar(mmd_results_df['scenario'],mmd_results_df['mean'], yerr=mmd_results_df['std_error']*1.96, fmt='.', capsize=5,markersize=0, ecolor='black', elinewidth=0.5)
    plt.xticks(rotation=30)
    plt.title("MMD between BBR-{u,n,s} and DR vs. UR50 and GGR")
    plt.ylabel("MMD")
    plt.xlabel("")
    plt.savefig(os.path.join(args.out_dir, FNAME_FINAL_RESULTS_PDF), bbox_inches='tight', dpi=300)
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Load MMD/σ .npy files, generate plots, and save to disk"
    )
    parser.add_argument(
        '--input_dir', '-i',
        help='Base directory containing per-scenario subfolders of .npy results',
        required=True
    )
    parser.add_argument(
        '--mmd_pattern', '-m',
        help='Glob pattern for loading MMD result .npy files',
        required=True
    )
    parser.add_argument(
        '--median_pattern', '-d',
        help='Glob pattern for loading median-distance .npy files',
        required=True
    )
    parser.add_argument(
        '--out_dir', '-o',
        help='Directory where all output files (PDFs, parquet) will be written',
        default='mmd_results'
    )

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Derive your final patterns in case you want them relative to input_dir
    if not args.mmd_pattern.startswith(args.input_dir):
        args.mmd_pattern = os.path.join(args.input_dir, args.mmd_pattern)
    if not args.median_pattern.startswith(args.input_dir):
        args.median_pattern = os.path.join(args.input_dir, args.median_pattern)

    # Finally call your main logic
    main(args)
