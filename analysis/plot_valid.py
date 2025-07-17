import os
from tqdm import tqdm

import torch
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd

_ = sns.set_style('white')

world_size = 8
models = ['jamba-3b-seq-sam-biar-fsdp-tok90k', 'jamba-170m-seqsam-36w']
checkpoints = {
    'jamba-3b-seq-sam-biar-fsdp-tok90k': [10000, 25000, 43300 ],
    'jamba-170m-seqsam-36w': [10000, 40000, 76000]
}
checkpoint_to_name = {
    'jamba-3b-seq-sam-biar-fsdp-tok90k': '3b-uniref90',
    'jamba-170m-seqsam-36w': '170m-uniref90',
}
directions = ['forward', 'reverse']
out_fpath = '/home/kevyan/generations/validation/'
pal = sns.color_palette()
for model in models:
    for direction in directions:
        fig1, ax1 = plt.subplots(1, 1)
        for i, checkpoint in enumerate(checkpoints[model]):
            print(model, direction, checkpoint)
            ces = []
            for rank in range(world_size):
                out_file = os.path.join(out_fpath, "valid_" + model + '_' + str(checkpoint) + "_" + "uniref" + "_" + direction + "_%d.pt" %rank)
                try:
                    dat = torch.load(out_file)
                    if '3b' in model and direction == 'forward' and checkpoint == 43300:
                        ces.append(dat["ce"][::1000])
                    else:
                        ces.append(dat["ce"])
                    print(rank, torch.cat(ces).shape)
                except(EOFError):
                    continue
            ces = torch.cat(ces)
            ces = np.array(ces)
            ces[ces == 0] = np.nan
            ces = ces[:, :-1]
            ce_by_pos = np.nanmean(ces, axis=0)
            x = np.arange(len(ce_by_pos))
            std_by_pos = np.nanstd(ces, axis=0)
            se_by_pos = std_by_pos / np.sqrt(np.isfinite(ces).sum(axis=0))
            _ = ax1.plot(x, ce_by_pos, "-", label=str(checkpoint), color=pal[i], alpha=0.7)
            # _ = ax1.fill_between(x, ce_by_pos + se_by_pos, ce_by_pos - se_by_pos, alpha=0.3, color=pal[i])
        _ = ax1.set_xlabel('position')
        _ = ax1.set_ylabel('cross-entropy')
        _ = ax1.set_ylim(0.5, 3)
        _ = ax1.legend()
        ax2 = ax1.twinx()
        n = np.isfinite(ces).sum(axis=0)
        _ = ax2.plot(x, n, "-", color="gray")
        _ = ax2.set_ylabel('# of val sequences', rotation=270, labelpad=15)
        _ = fig1.savefig(os.path.join(out_fpath, model + "_" + "uniref" + "_" + direction + ".pdf"), dpi=300, bbox_inches="tight")
#
#
models = ['jamba-170m-gigaclust-36w']
checkpoints = {
    'jamba-170m-gigaclust-36w': [10000, 40000, 76000]
}
directions = ['forward']
out_fpath = '/home/kevyan/generations/validation/'
pal = sns.color_palette()
for model in models:
    for direction in directions:
        fig1, ax1 = plt.subplots(1, 1)
        for i, checkpoint in enumerate(checkpoints[model]):
            ces = []
            for rank in range(world_size):
                out_file = os.path.join(out_fpath, "valid_" + model + '_' + str(checkpoint) + "_" + "gigaref" + "_" + direction + "_%d.pt" %rank)
                dat = torch.load(out_file)
                ces.append(dat["ce"])
            ces = torch.cat(ces)
            ces = np.array(ces)
            ces[ces == 0] = np.nan
            ces = ces[:, :-1]
            ce_by_pos = np.nanmean(ces, axis=0)
            x = np.arange(len(ce_by_pos))
            std_by_pos = np.nanstd(ces, axis=0)
            se_by_pos = std_by_pos / np.sqrt(np.isfinite(ces).sum(axis=0))
            _ = ax1.plot(x, ce_by_pos, "-", label=str(checkpoint), color=pal[i], alpha=0.7)
            # _ = ax1.fill_between(x, ce_by_pos + se_by_pos, ce_by_pos - se_by_pos, alpha=0.3, color=pal[i])
        _ = ax1.set_xlabel('position')
        _ = ax1.set_ylabel('cross-entropy')
        _ = ax1.legend()
        ax2 = ax1.twinx()
        n = np.isfinite(ces).sum(axis=0)
        _ = ax2.plot(x, n, "-", color="gray")
        _ = ax2.set_ylabel('n')
        _ = fig1.savefig(os.path.join(out_fpath, model + "_" + "gigaref" + "_" + direction + ".pdf"), dpi=300, bbox_inches="tight")
#
#
models = ['jamba-3b-indel-gigaclust-120k-2', 'jamba-3b-cooldown', 'jamba-3b-cooldown7']
checkpoints = {
    'jamba-3b-indel-gigaclust-120k-2': [10000, 25000, 52000 ],
    'jamba-3b-cooldown': [12000],
    'jamba-3b-cooldown7': [25000]
}
model_name = {
    'jamba-3b-indel-gigaclust-120k-2': '3b-msa-gigaclust',
    'jamba-3b-cooldown': '3b-msa-uniref',
    'jamba-3b-cooldown7': '3b-msa-uniref'
}
total_steps = 0
direction = 'forward'
out_fpath = '/home/kevyan/generations/validation'
pal = sns.color_palette()
fig1, ax1 = plt.subplots(1, 1)
pal_counter = 0
for model in models:
    for i, checkpoint in enumerate(checkpoints[model]):
        ces = []
        if model == 'jamba-3b-indel-gigaclust-120k-2':
            total_steps = checkpoint
        else:
            total_steps += checkpoint
        for rank in range(world_size):
            out_file = os.path.join(out_fpath, "valid_" + model + '_' + str(checkpoint) + "_" + "uniref" + "_" + direction + "_%d.pt" %rank)
            dat = torch.load(out_file)
            ces.append(dat["ce"])
        ces = torch.cat(ces)
        ces = np.array(ces)
        ces[ces == 0] = np.nan
        ces = ces[:, :-1]
        ce_by_pos = np.nanmean(ces, axis=0)
        x = np.arange(len(ce_by_pos))
        std_by_pos = np.nanstd(ces, axis=0)
        se_by_pos = std_by_pos / np.sqrt(np.isfinite(ces).sum(axis=0))
        _ = ax1.plot(x, ce_by_pos, "-", label=model_name[model] + '_' + str(total_steps), color=pal[pal_counter], alpha=0.7)
        pal_counter += 1
        # _ = ax1.fill_between(x, ce_by_pos + se_by_pos, ce_by_pos - se_by_pos, alpha=0.3, color=pal[i])
    _ = ax1.set_xlabel('position')
    _ = ax1.set_ylabel('cross-entropy')
    _ = ax1.legend()
    _ = ax1.set_ylim([1.5, 2.9])
    ax2 = ax1.twinx()
    n = np.isfinite(ces).sum(axis=0)
    _ = ax2.plot(x, n, "-", color="gray")
    _ = ax2.set_ylabel('n')
    _ = fig1.savefig(os.path.join(out_fpath, "jamba-3b-combined_" + "uniref" + "_" + direction + ".pdf"), dpi=300, bbox_inches="tight")
#
#
models = ['jamba-3b-indel-gigaclust-120k-2', 'jamba-3b-cooldown', 'jamba-3b-cooldown7']
checkpoints = {
    'jamba-3b-indel-gigaclust-120k-2': [10000, 25000, 52000 ],
    'jamba-3b-cooldown': [12000],
    'jamba-3b-cooldown7': [25000]
}
model_name = {
    'jamba-3b-indel-gigaclust-120k-2': 'msa-gigaclust',
    'jamba-3b-cooldown': 'msa-uniref',
    'jamba-3b-cooldown7': 'msa-uniref'
}
total_steps = 0
direction = 'forward'
out_fpath = '/home/kevyan/generations/validation'
pal = sns.color_palette()
fig1, ax1 = plt.subplots(1, 1)
pal_counter = 0
for model in models:
    for i, checkpoint in enumerate(checkpoints[model]):
        ces = []
        if model == 'jamba-3b-indel-gigaclust-120k-2':
            total_steps = checkpoint
        else:
            total_steps += checkpoint
        for rank in range(world_size):
            out_file = os.path.join(out_fpath, "valid_" + model + '_' + str(checkpoint) + "_" + "gigaref" + "_" + direction + "_%d.pt" %rank)
            dat = torch.load(out_file)
            ces.append(dat["ce"])
        ces = torch.cat(ces)
        ces = np.array(ces)
        ces[ces == 0] = np.nan
        ces = ces[:, :-1]
        ce_by_pos = np.nanmean(ces, axis=0)
        x = np.arange(len(ce_by_pos))
        std_by_pos = np.nanstd(ces, axis=0)
        se_by_pos = std_by_pos / np.sqrt(np.isfinite(ces).sum(axis=0))
        _ = ax1.plot(x, ce_by_pos, "-", label=model_name[model] + '_' + str(total_steps), color=pal[pal_counter], alpha=0.7)
        pal_counter += 1
        # _ = ax1.fill_between(x, ce_by_pos + se_by_pos, ce_by_pos - se_by_pos, alpha=0.3, color=pal[i])
    _ = ax1.set_xlabel('position')
    _ = ax1.set_ylabel('cross-entropy')
    _ = ax1.legend()
    _ = ax1.set_ylim([0.5, 5])
    ax2 = ax1.twinx()
    n = np.isfinite(ces).sum(axis=0)
    _ = ax2.plot(x, n, "-", color="gray")
    _ = ax2.set_ylabel('n')
    _ = fig1.savefig(os.path.join(out_fpath, "jamba-3b-combined_" + "gigaref" + "_" + direction + ".png"), dpi=300, bbox_inches="tight")
#
#
model = 'jamba-3b-cooldown7'
checkpoints = {
    'jamba-3b-cooldown7': [25000],
}
model_name = {
    'jamba-3b-cooldown7': 'Dayhoff-3B'
}
tasks = ["indel", "gap"]
out_fpath = '/home/kevyan/generations/validation'


checkpoint = checkpoints[model][0]
direction = "forward"
df = pd.DataFrame()
current_row = 0
current_msa_id = 0
for task in ["gap", "indel"]:
    for rank in range(8):
    # for rank in range(7):
        out_file = os.path.join(out_fpath, "valid_long_" + model + '_' + str(
            checkpoint) + "_" + task + "_" + direction + "_%d.pt" % rank)
        print(out_file)
        dat = torch.load(out_file)
        msas = dat['sequence']
        ces = dat['ce']
        for msa, ce in tqdm(zip(msas, ces), total=len(msas)):
            msa = msa.replace('!', '')
            msa = msa[1:]
            if task == 'indel':
                current_pos = 0
                current_seqs = 0
                break_idx = 0
                while break_idx > -1:
                    break_idx = msa.find('/', current_pos)
                    current_ce = ce[current_pos:break_idx].mean().item()
                    if current_seqs == 0:
                        first_ce = current_ce
                        first_perplexity = np.exp(current_ce)
                    if current_seqs < 65:
                        df.loc[current_row, 'msa_id'] = current_msa_id
                        df.loc[current_row, 'n_conditioning'] = current_seqs
                        df.loc[current_row, 'ce'] = current_ce
                        df.loc[current_row, 'perplexity'] = np.exp(current_ce)
                        df.loc[current_row, 'perplexity_diff'] = np.exp(current_ce) - first_perplexity
                        df.loc[current_row, 'ce_diff'] = current_ce - first_ce
                        df.loc[current_row, 'task'] = task
                        current_row += 1
                    current_seqs += 1
                    current_pos = break_idx + 1
            elif task == 'gap':
                ce = ce.numpy()
                if "*" not in msa:
                    break_idx = msa.find('/', 0)
                    n_seqs = len(msa.split('/'))
                    if len(msa) < (break_idx + 1) * n_seqs:
                        n_seqs -= 1
                    reshaped_ce = ce[: (break_idx + 1) * n_seqs].reshape(-1, break_idx + 1)
                    ce_mean = reshaped_ce.mean(axis=1)
                else:
                    if msa[-1] == "*":
                        break_idx = msa.find('/', 0)
                        if break_idx == -1:
                            continue
                        reshaped_ce = ce[:len(msa) - 3 - break_idx].reshape(-1, break_idx + 1)
                        ce_mean = reshaped_ce.mean(axis=1)
                        ce_mean = np.concatenate([ce_mean, ce[-break_idx-3:].mean(keepdims=True)])
                    else:
                        break_idx = msa.find('*', 0)
                        first_mean = ce[:break_idx + 1].mean(keepdims=True)
                        n_seqs = len(msa.split('/')) - 1
                        ell = break_idx - 2
                        n_aligned = n_seqs * ell
                        if n_aligned + break_idx + 2 > len(msa):
                            n_aligned -= ell
                        reshaped_ce = ce[break_idx + 2: break_idx + 2 + n_aligned].reshape(-1, ell)
                        ce_mean = reshaped_ce.mean(axis=1)
                        ce_mean = np.concatenate([first_mean, ce_mean])
                ce_mean_norm = ce_mean - ce_mean[0]
                perplexity_mean = np.exp(ce_mean)
                perplexity_mean_norm = perplexity_mean - perplexity_mean[0]
                df_current = pd.DataFrame()
                df_current['msa_id'] = current_msa_id
                df_current['n_conditioning'] = np.arange(len(ce_mean_norm))
                df_current['ce'] = ce_mean
                df_current['perplexity'] = perplexity_mean
                df_current['perplexity_diff'] = perplexity_mean_norm
                df_current['ce_diff'] = ce_mean_norm
                df_current['task'] = task
                df = pd.concat([df, df_current], ignore_index=True)
            current_msa_id += 1
        df.to_csv(os.path.join(out_fpath, "valid_by_conditioning_%s_%d.csv" %(task, rank)), index=False)

dfs = []
current_id = 0
tasks = ['indel', 'gap']
for task in tasks:
    for rank in range(world_size):
        if task == "gap" and rank == 7:
            continue
        df = pd.read_csv(os.path.join(out_fpath, "valid_by_conditioning_%s_%d.csv" %(task, rank)))
        df['msa_id'] = current_id + df['msa_id']
        current_id = max(df['msa_id'])
        dfs.append(df)
df = pd.concat(dfs, ignore_index=True)
pal = sns.color_palette()
fig1, ax1 = plt.subplots(1, 1)
# ax2 = ax1.twinx()
_ = sns.lineplot(data=df, x='n_conditioning', y='ce_diff', hue='task', ax=ax1, palette=pal, errorbar=None)
_ = fig1.savefig(os.path.join(out_fpath, "jamba-3b-combined" + "_long_msas" + "_" + direction + "_conditioning.pdf"), dpi=300, bbox_inches="tight")
fig1, ax1 = plt.subplots(1, 1)
# ax2 = ax1.twinx()
_ = sns.lineplot(data=df[df['n_conditioning'] < 64], x='n_conditioning', y='ce_diff', hue='task', ax=ax1, palette=pal)
_ = fig1.savefig(os.path.join(out_fpath, "jamba-3b-combined" + "_long_msas" + "_" + direction + "_conditioning64.pdf"), dpi=300, bbox_inches="tight")

world_size = 8
models = ['jamba-3b-seq-sam-biar-fsdp-tok90k']
checkpoints = [10000, 25000, 43300 ]
direction = 'forward'
out_fpath = '/home/kevyan/generations/validation/'
pal = sns.color_palette()
for model in models:
    fig1, ax1 = plt.subplots(1, 1)
    ces_by_step = []
    for i, checkpoint in enumerate(checkpoints):
        ces = []
        for rank in range(world_size):
            if rank == 2 and direction == 'forward':
                continue
            out_file = os.path.join(out_fpath, "valid_" + model + '_' + str(checkpoint) + "_" + "uniref" + "_" + direction + "_%d.pt" %rank)
            dat = torch.load(out_file)
            ces.append(dat["ce"])
        ces = torch.cat(ces)
        ces_by_step.append(ces)
    ces_by_step = torch.stack(ces_by_step)
    # normalize by position and step
ces_by_step = np.array(ces_by_step)
ces = ces_by_step[:, :, 1:-1].transpose(1, 2, 0).reshape(-1, 3)
m = np.isnan(ces)
m = m.sum(axis=1) == 0
ces_sel = ces[m]
idx = np.random.choice(len(ces_sel), 200, replace=False)
fig, ax = plt.subplots(1, 1)
for i in idx:
    _ = ax.plot(checkpoints, ces_sel[i], color=pal[3], alpha=0.2)
_ = ax.set_xlabel("step")
_ = ax.set_ylabel("cross-entropy")
_ = fig.savefig(os.path.join(out_fpath, model + "_" + "uniref" + "_" + direction + "_step.pdf"), dpi=300, bbox_inches="tight")
