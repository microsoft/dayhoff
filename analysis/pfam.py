import os

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr
from tqdm import tqdm

_ = sns.set_theme(font_scale=1.2)
_ = sns.set_style('white')

in_dir = "/home/kevyan/generations/pfam_annotations/"
pfam_outputs = os.listdir(in_dir)
with open(os.path.join(in_dir, "pfam_annotations.csv"), "w") as f_out:
    f_out.write("model,direction,temperature,subset,query,pfam\n")
    for file_name in pfam_outputs[::-1]:
        print(file_name)
        if file_name in ["sow2.txt", "uniref50.txt", "pfam_annotations.csv"]:
            continue
        elif "rfdiffusion" in file_name:
            model_name = 'rfdiffusion'
            subset = '_'.join(file_name.split(".")[0].split("_")[1:])
            direction = ""
            temperature = ""
        else:
            model_name = file_name.split("_")[0]
            direction = file_name.split("_")[1]
            temperature = file_name.split("_")[2][1:-4]
            subset = ""
        hit_line = 1e12
        with open(os.path.join(in_dir, file_name)) as hmmers:
            for i, line in enumerate(tqdm(hmmers)):
                if line[:5] == "Query" and "L=" in line:
                    hit_line = i + 5
                    current_query = line.split()[1]
                if i == hit_line:
                    if "inclusion threshold" in line:
                        continue
                    elif line == "\n":
                        continue
                    elif line == '[ok]\n':
                        continue
                    else:
                        pfam = line.split()[8]
                        hit_line += 1
                        f_out.write("{},{},{},{},{},{}\n".format(model_name, direction, temperature, subset, current_query, pfam))


in_dir = "/home/kevyan/generations/uniref_pfam_annotations/"
pfam_outputs = os.listdir(in_dir)
with open(os.path.join(in_dir, "uniref_pfam_annotations.csv"), "w") as f_out:
    f_out.write("ur_id,pfam\n")
    for file_name in pfam_outputs[::-1]:
        print(file_name)
        hit_line = 1e12
        with open(os.path.join(in_dir, file_name)) as hmmers:
            for i, line in enumerate(tqdm(hmmers)):
                if line[:5] == "Query" and "L=" in line:
                    hit_line = i + 5
                    current_query = line.split()[1]
                if i == hit_line:
                    if "inclusion threshold" in line:
                        continue
                    elif line == "\n":
                        continue
                    elif line == '[ok]\n':
                        continue
                    else:
                        pfam = line.split()[8]
                        hit_line += 1
                        f_out.write("{},{}\n".format(current_query, pfam))

in_dir = "/home/kevyan/generations/pfam_annotations/"
pfam_outputs = os.listdir(in_dir)
with open(os.path.join(in_dir, "bbr_unfiltered_pfam_annotations.csv"), "w") as f_out:
    f_out.write("ur_id,pfam\n")
    for file_name in pfam_outputs[::-1]:
        if "rfdiffusion_unfiltered" not in file_name:
            continue
        print(file_name)
        hit_line = 1e12
        with open(os.path.join(in_dir, file_name)) as hmmers:
            for i, line in enumerate(tqdm(hmmers)):
                if line[:5] == "Query" and "L=" in line:
                    hit_line = i + 5
                    current_query = line.split()[1]
                if i == hit_line:
                    if "inclusion threshold" in line:
                        continue
                    elif line == "\n":
                        continue
                    elif line == '[ok]\n':
                        continue
                    else:
                        pfam = line.split()[8]
                        hit_line += 1
                        f_out.write("{},{}\n".format(current_query, pfam))

in_dir = "/home/kevyan/generations/ggr_pfam_annotations/"
pfam_outputs = os.listdir(in_dir)
with open(os.path.join(in_dir, "ggr_singles_pfam_annotations.csv"), "w") as single_out, open(os.path.join(in_dir, "ggr_reps_pfam_annotations.csv"), "w") as rep_out:
    single_out.write("ur_id,pfam\n")
    rep_out.write("ur_id,pfam\n")
    for file_name in pfam_outputs[::-1]:
        print(file_name)
        if "reps" in file_name:
            f_out = rep_out
        else:
            f_out = single_out
        hit_line = 1e12
        with open(os.path.join(in_dir, file_name)) as hmmers:
            for i, line in enumerate(tqdm(hmmers)):
                if line[:5] == "Query" and "L=" in line:
                    hit_line = i + 5
                    current_query = line.split()[1]
                if i == hit_line:
                    if "inclusion threshold" in line:
                        continue
                    elif line == "\n":
                        continue
                    elif line == '[ok]\n':
                        continue
                    else:
                        pfam = line.split()[8]
                        hit_line += 1
                        f_out.write("{},{}\n".format(current_query, pfam))


in_dir = "/home/kevyan/generations/"

df_dr = pd.read_csv(os.path.join(in_dir, "pfam_annotations", "pfam_annotations.csv"))
df_sc = pd.read_csv(os.path.join(in_dir, "pfam_annotations", "bbr_sc_pfam_annotations.csv"))
df_n = pd.read_csv(os.path.join(in_dir, "pfam_annotations", "bbr_novel_pfam_annotations.csv"))
df_scn = pd.read_csv(os.path.join(in_dir, "pfam_annotations", "bbr_sc_novel_pfam_annotations.csv"))
df_un = pd.read_csv(os.path.join(in_dir, "pfam_annotations", "bbr_unfiltered_pfam_annotations.csv"))

df_ur = pd.read_csv(os.path.join(in_dir, "uniref_pfam_annotations", "uniref_pfam_annotations.csv"))
df_s = pd.read_csv(os.path.join(in_dir, "ggr_pfam_annotations", "ggr_singles_pfam_annotations.csv"))
df_r = pd.read_csv(os.path.join(in_dir, "ggr_pfam_annotations", "ggr_reps_pfam_annotations.csv"))

df_ur = df_ur[df_ur['pfam'] != "--------"]
df_sc = df_sc[df_sc['pfam'] != "--------"]
df_n = df_n[df_n['pfam'] != "--------"]
df_scn = df_scn[df_scn['pfam'] != "--------"]
df_un = df_un[df_un['pfam'] != "--------"]

models = list(set(df_dr["model"]))
for model in models:
    print(model, df_dr[df_dr["model"] == model].shape)


dr_domains = sorted(set(df_dr["pfam"]))
ur_domains = sorted(set(df_ur["pfam"]))
s_domains = sorted(set(df_s["pfam"]))
r_domains = sorted(set(df_r["pfam"]))
sc_domains = sorted(set(df_sc["pfam"]))
un_domains = sorted(set(df_un["pfam"]))
scn_domains = sorted(set(df_scn["pfam"]))
n_domains = sorted(set(df_n["pfam"]))
len(dr_domains), len(ur_domains)
set(ur_domains) - set(dr_domains)
set(dr_domains) - set(ur_domains)
set(r_domains) - set(ur_domains)
len(set(ur_domains) - set(sc_domains))
set(n_domains) - set(ur_domains)

all_domains = sorted(set(dr_domains).union(set(ur_domains)).union(set(r_domains)).union(set(s_domains)).union(set(sc_domains)))
dr_counter = {d: 0 for d in all_domains}
ur_counter = {d: 0 for d in all_domains}
s_counter = {d: 0 for d in all_domains}
r_counter = {d: 0 for d in all_domains}
sc_counter = {d: 0 for d in all_domains}
n_counter = {d: 0 for d in all_domains}
scn_counter = {d: 0 for d in all_domains}
un_counter = {d: 0 for d in all_domains}

for row in df_dr.itertuples():
    dr_counter[row.pfam] += 1
for row in df_ur.itertuples():
    ur_counter[row.pfam] += 1
for row in df_r.itertuples():
    r_counter[row.pfam] += 1
for row in df_s.itertuples():
    s_counter[row.pfam] += 1

for row in df_sc.itertuples():
    sc_counter[row.pfam] += 1
for row in df_n.itertuples():
    n_counter[row.pfam] += 1
for row in df_scn.itertuples():
    scn_counter[row.pfam] += 1
for row in df_un.itertuples():
    un_counter[row.pfam] += 1
dr_vector = np.array([dr_counter[d] for d in all_domains])
ur_vector = np.array([ur_counter[d] for d in all_domains])
s_vector = np.array([s_counter[d] for d in all_domains])
r_vector = np.array([r_counter[d] for d in all_domains])
sc_vector = np.array([sc_counter[d] for d in all_domains])
n_vector = np.array([n_counter[d] for d in all_domains])
scn_vector = np.array([scn_counter[d] for d in all_domains])
un_vector = np.array([un_counter[d] for d in all_domains])
input_dict = {
    "UniRef50": ur_vector,
    "GR": r_vector,
    "GR-singles": s_vector,
    "BRq": sc_vector,
    "BRn": scn_vector,
    "BRu": un_vector,
    "DayhoffRef": dr_vector,
}
np.savez_compressed(os.path.join(in_dir, "pfam_annotations", "counts.npz"), **input_dict)

d = np.load(os.path.join(in_dir, "pfam_annotations", "counts.npz"), allow_pickle=True)

# fixed = {
# "UniRef50": d['UniRef50'],
#     "GR": d['GGR'],
#     "GR-singles": d['GGR-singles'],
#     "BRq": d['BBR-sc'],
#     "BRn": d['BBR-n'],
#     "BRu": d['BBR-u'],
#     "DayhoffRef": d['DayhoffRef'],
# }

input_dict = {k: d[k] for k in d.keys()}
keys = list(input_dict.keys())
for i, k1 in enumerate(keys):
    v1 = input_dict[k1]
    sort_idx = v1.argsort()
    print(k1, v1.sum())
    for k2 in keys[i + 1:]:
        v2 = input_dict[k2]
        fig, ax = plt.subplots(1, 1)
        _ = ax.plot(v1[sort_idx][::10], v2[sort_idx][::10], '.', alpha=0.3, color='gray')
        _ = ax.set_xlabel("Occurences in %s" %k1)
        _ = ax.set_ylabel("Occurences in %s" %k2)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(10000))

        _ = fig.savefig(os.path.join(in_dir, "pfam_annotations", "pfam_counts_%s_%s.pdf" %(k1, k2)), dpi=300, bbox_inches='tight')
        r = pearsonr(v1, v2)[0]
        print(k1, k2, r)
