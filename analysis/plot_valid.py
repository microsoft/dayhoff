import os

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
directions = ['forward', 'reverse']
out_fpath = '/home/kevyan/generations/'
pal = sns.color_palette()
for model in models:
    for direction in directions:
        fig1, ax1 = plt.subplots(1, 1)
        for i, checkpoint in enumerate(checkpoints[model]):
            ces = []
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
            _ = ax1.plot(x, ce_by_pos, "-", label=str(checkpoint), color=pal[i], alpha=0.7)
            # _ = ax1.fill_between(x, ce_by_pos + se_by_pos, ce_by_pos - se_by_pos, alpha=0.3, color=pal[i])
        _ = ax1.set_xlabel('position')
        _ = ax1.set_ylabel('cross-entropy')
        _ = ax1.legend()
        ax2 = ax1.twinx()
        n = np.isfinite(ces).sum(axis=0)
        _ = ax2.plot(x, n, "-", color="gray")
        _ = ax2.set_ylabel('n')
        _ = fig1.savefig(os.path.join(out_fpath, model + "_" + "uniref" + "_" + direction + ".png"), dpi=300, bbox_inches="tight")


models = ['jamba-170m-gigaclust-36w']
checkpoints = {
    'jamba-170m-gigaclust-36w': [10000, 40000, 76000]
}
directions = ['forward']
out_fpath = '/home/kevyan/generations/'
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
        _ = fig1.savefig(os.path.join(out_fpath, model + "_" + "gigaref" + "_" + direction + ".png"), dpi=300, bbox_inches="tight")


models = ['jamba-3b-indel-gigaclust-120k-2', 'jamba-3b-cooldown', 'jamba-3b-cooldown7']
checkpoints = {
    'jamba-3b-indel-gigaclust-120k-2': [10000, 25000, 52000 ],
    'jamba-3b-cooldown': [12000],
    'jamba-3b-cooldown7': [25000]
}
model_name = {
    'jamba-3b-indel-gigaclust-120k-2': 'indel-gigaclust',
    'jamba-3b-cooldown': 'indel-uniref-cooldown',
    'jamba-3b-cooldown7': 'indel-uniref-cooldown'
}
total_steps = 0
direction = 'forward'
out_fpath = '/home/kevyan/generations/'
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
    _ = fig1.savefig(os.path.join(out_fpath, "jamba-3b-combined_" + "uniref" + "_" + direction + ".png"), dpi=300, bbox_inches="tight")


models = ['jamba-3b-indel-gigaclust-120k-2', 'jamba-3b-cooldown', 'jamba-3b-cooldown7']
checkpoints = {
    'jamba-3b-indel-gigaclust-120k-2': [10000, 25000, 52000 ],
    'jamba-3b-cooldown': [12000],
    'jamba-3b-cooldown7': [25000]
}
model_name = {
    'jamba-3b-indel-gigaclust-120k-2': 'indel-gigaclust',
    'jamba-3b-cooldown': 'indel-uniref-cooldown',
    'jamba-3b-cooldown7': 'indel-uniref-cooldown'
}
total_steps = 0
direction = 'forward'
out_fpath = '/home/kevyan/generations/'
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


model = 'jamba-3b-cooldown7'
checkpoints = {
    'jamba-3b-cooldown7': [25000],
}
model_name = {
    'jamba-3b-cooldown7': 'Dayhoff-3B'
}
tasks = ["indel", "gap"]
total_steps = 0
direction = 'forward'
out_fpath = '/home/kevyan/generations/'
pal = sns.color_palette()
fig1, ax1 = plt.subplots(1, 1)
ax2 = ax1.twinx()
pal_counter = 0
for task in tasks:
    for i, checkpoint in enumerate(checkpoints[model]):
        ces = []
        for rank in range(world_size):
            out_file = os.path.join(out_fpath, "valid_" + model + '_' + str(checkpoint) + "_" + task + "_" + direction + "_%d.pt" %rank)
            dat = torch.load(out_file)
            ces.append(dat["ce"])
        ces = torch.cat(ces)
        ces = np.array(ces)
        ces[ces == 0] = np.nan
        ces = ces[:, :65000]
        ce_by_pos = np.nanmean(ces, axis=0)
        x = np.arange(len(ce_by_pos))
        _ = ax1.plot(x, ce_by_pos, "-", label=task, color=pal[pal_counter], alpha=0.7)
        pal_counter += 1
        # _ = ax1.fill_between(x, ce_by_pos + se_by_pos, ce_by_pos - se_by_pos, alpha=0.3, color=pal[i])
    n = np.isfinite(ces).sum(axis=0)
    _ = ax2.plot(x, n, "-", color="gray", alpha=0.7)
_ = ax1.set_xlabel('position')
_ = ax1.set_ylabel('cross-entropy')
_ = ax1.legend()
_ = ax2.set_ylabel('n')
_ = fig1.savefig(os.path.join(out_fpath, "jamba-3b-combined" + "msas" + "_" + direction + ".png"), dpi=300, bbox_inches="tight")

model = 'jamba-3b-cooldown7'
checkpoints = {
    'jamba-3b-cooldown7': [25000],
}
model_name = {
    'jamba-3b-cooldown7': 'Dayhoff-3B'
}
tasks = ["indel", "gap"]
total_steps = 0
direction = 'forward'
out_fpath = '/home/kevyan/generations/'
pal = sns.color_palette()
fig1, ax1 = plt.subplots(1, 1)
ax2 = ax1.twinx()
pal_counter = 0
for task in tasks:
    for i, checkpoint in enumerate(checkpoints[model]):
        ces = []
        for rank in range(world_size):
            out_file = os.path.join(out_fpath, "valid_long_" + model + '_' + str(checkpoint) + "_" + task + "_" + direction + "_%d.pt" %rank)
            dat = torch.load(out_file)
            ces.append(dat["ce"])
        ces = torch.cat(ces)
        ces = np.array(ces)
        ces[ces == 0] = np.nan
        ces = ces[:, :130000]
        ce_by_pos = np.nanmean(ces, axis=0)
        x = np.arange(len(ce_by_pos))
        _ = ax1.plot(x, ce_by_pos, "-", label=task, color=pal[pal_counter], alpha=0.7)
        pal_counter += 1
        # _ = ax1.fill_between(x, ce_by_pos + se_by_pos, ce_by_pos - se_by_pos, alpha=0.3, color=pal[i])
    n = np.isfinite(ces).sum(axis=0)
    _ = ax2.plot(x, n, "-", color="gray", alpha=0.7)
_ = ax1.set_xlabel('position')
_ = ax1.set_ylabel('cross-entropy')
_ = ax1.legend()
_ = ax2.set_ylabel('n')
_ = fig1.savefig(os.path.join(out_fpath, "jamba-3b-combined" + "_long_msas" + "_" + direction + ".png"), dpi=300, bbox_inches="tight")



ces.shape
np.nanmax(ces)
np.nanmin(ces)
np.exp(14) / 1e6
ces_by_seq = np.nanmean(ces[1:-1], axis=1)
ces_by_seq.shape
ces_by_seq.argmax()
ces_by_seq.min()

world_size = 8
models = ['jamba-3b-seq-sam-biar-fsdp-tok90k']
checkpoints = [10000, 25000, 43300 ]
direction = 'forward'
out_fpath = '/home/kevyan/generations/'
pal = sns.color_palette()
for model in models:
    fig1, ax1 = plt.subplots(1, 1)
    ces_by_step = []
    for i, checkpoint in enumerate(checkpoints):
        ces = []
        for rank in range(world_size):
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
_ = fig.savefig(os.path.join(out_fpath, model + "_" + "uniref" + "_" + direction + "_step.png"), dpi=300, bbox_inches="tight")



ces[10]
