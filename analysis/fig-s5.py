#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pathlib
import pandas as pd
import pathlib
import pandas as pd
import joblib
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import re

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
# plt.rc("font", family="serif", size=16.0, weight="medium")
plt.rc("savefig", dpi=500)
plt.rc("legend", loc="best", fontsize="medium", fancybox=True, framealpha=0.5)
plt.rc("lines", linewidth=2.5, markersize=10, markeredgewidth=2.5)
plt.rc("axes", titlepad=10)

colors = ["#BBBBBB", "#33BBEE", "#EE3377", "#009988", "#CC3311", "#0077BB"]
colors = list(reversed(colors))
sns.set_palette(sns.color_palette(colors))
# set mpl palette
plt.rcParams["axes.prop_cycle"] = plt.cycler(color=colors)


# In[2]:


df = pd.read_parquet('temperature_scRMSD_gridsearch_results.parquet')


# In[3]:


averages = df.groupby(["temperature", 
                      'backbone_pdb'])[
    ["scRMSD", "TM", "aa_length"]
].mean()
averages = averages.reset_index()

averages["pass"] = averages["scRMSD"] <= 2
averages["pass_tm"] = averages["TM"] >= 0.5
bins = np.arange(0, 1100, 100)


# In[4]:


plt.figure() 
averages['pass_pct'] = averages['pass'] * 100
sns.lineplot(data=averages, 
           x='temperature', 
           y='pass_pct',
           marker='o',
           color='#0D96C9', 
           errorbar=None)

plt.xlabel('Temperature')
plt.ylabel('% backbones with\n(scRMSD < 2Å)')


# Force x-axis tick labels to show
plt.gca().set_xticks(averages['temperature'].unique())  # Set ticks at each temperature value
plt.gca().set_xticklabels(averages['temperature'].unique())  # Force labels to show

# Remove top and right spines
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

plt.tight_layout()
plt.title('BackboneRef sample designability\nby temperature')

