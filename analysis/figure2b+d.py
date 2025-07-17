import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

sns.set(font_scale=1.7)
sns.set_style("whitegrid")
plt.rcParams['axes.grid'] = False
plt.rc('axes',edgecolor='black')

plt.rc("text", usetex=False)
plt.rc(
    "text.latex",
    preamble=r"\usepackage{newpxtext}\usepackage{newpxmath}\usepackage{commath}\usepackage{mathtools}",
)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rc("savefig", dpi=500)
plt.rc("legend", loc="best", fontsize="medium", fancybox=True, framealpha=0.5)
plt.rc("lines", linewidth=2.5, markersize=10, markeredgewidth=2.5)
plt.rc("axes", titlepad=10)
colors = ["#BBBBBB", "#33BBEE", "#EE3377", "#009988", "#CC3311", "#0077BB"]
colors = list(reversed(colors))
sns.set_palette(sns.color_palette(colors))
# set mpl palette
plt.rcParams["axes.prop_cycle"] = plt.cycler(color=colors)

# this should be on the zenodo
df = pd.read_parquet('backbone_novelty_quality_statistics.parquet')

# figure 2d
sns.histplot(df,
             x='length',
             y='max_search_tm')
plt.ylabel('TM-score (AFDB/UniProt)')
plt.xlabel('Backbone length (AA)')
plt.xlim(40, 512)
_ = plt.xticks([100, 200, 300, 400, 500, ])
plt.ylim(0, 1.0)
plt.title('Max. TM-score of BR structures')
# make correlation txt
corr_df = df[['length', 'max_search_tm']]
corr_df.dropna(inplace=True)
res = stats.pearsonr(corr_df['length'], corr_df['max_search_tm'])
r = res.correlation.item()
_ = plt.text(350, 0.85, f'R = {r:.2f}')


# figure 2b
plt.figure()
sns.ecdfplot(df['avg_scrmsd'], stat='proportion', 
             complementary=False, linewidth=2, color='#0D96C9')
plt.xlabel('Average scRMSD')
plt.ylabel('Percentile')
plt.title('Designability of BR backbones')
plt.axvline(x=2, linestyle='--', color='black',zorder=-1, alpha=0.5)




