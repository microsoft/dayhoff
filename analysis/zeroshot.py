import argparse
import os

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn as nn
from scipy.stats import spearmanr
from sequence_models.utils import parse_fasta
from torch.utils.data import DataLoader
from tqdm import tqdm

from dayhoff.collators import MSAARCollator
from dayhoff.datasets import ListDataset
from dayhoff.model import OTHER_METRICS_KEY
from dayhoff.utils import load_checkpoint, load_msa_config_and_model, seed_everything

# default to a single-GPU setup if not present
RANK = int(os.environ["RANK"])
LOCAL_RANK = int(os.environ["LOCAL_RANK"])
WORLD_SIZE = int(os.environ["WORLD_SIZE"])
OFFSET = int(os.getenv("OFFSET", 0))
DEVICE = torch.device(f"cuda:{LOCAL_RANK + OFFSET}")


def is_amlt() -> bool:
    return os.environ.get("AMLT_OUTPUT_DIR", None) is not None


def zero_shot(
        dms_dir,
        out_dir,
        msa_dir,
        model: nn.Module,
        tokenizer,
        args,
):
    subst_files = os.listdir(dms_dir)
    fw_collator = MSAARCollator(tokenizer, flip_prob=0.0)
    bw_collator = MSAARCollator(tokenizer, flip_prob=1.0)
    os.makedirs(out_dir, exist_ok=True)
    summary_path = os.path.join(out_dir, args.model_name + '_%d.csv' %RANK)
    if not os.path.exists(summary_path):
        with open(summary_path, 'w') as f:
            if args.no_seq:
                f.write(','.join(['assay', 'indel_spearman']) + '\n')
            elif not args.msa:
                f.write(','.join(['assay', 'fw_spearman', 'bw_spearman', 'en_spearman']) + '\n')
            else:
                f.write(','.join(['assay', 'fw_spearman', 'bw_spearman', 'seq_spearman',
                                 'indel_spearman', 'en_spearman']) + '\n')
    for j, file in enumerate(tqdm(subst_files)):
        if j % WORLD_SIZE != RANK:
            continue
        assay_name = file.split('.csv')[0]
        if args.no_seq:
            if args.gap:
                df_out_file = os.path.join(out_dir, args.model_name + '_gap4_' + assay_name + '.csv')
            else:
                df_out_file = os.path.join(out_dir, args.model_name + '_indel4_' + assay_name + '.csv')
        else:
            df_out_file = os.path.join(out_dir, args.model_name + '_' + assay_name + '.csv')
        if os.path.exists(df_out_file):
            continue
        df_in = pd.read_csv(os.path.join(dms_dir, file))
        df_out = pd.DataFrame()
        if 'mutant' in df_in:
            df_out['mutant'] = df_in['mutant']
        else:
            df_out['mutated_sequence'] = df_in['mutated_sequence']
        df_out['DMS_score'] = df_in['DMS_score']
        df_out['assay'] = assay_name
        sequences = list(df_in["mutated_sequence"])
        ds = ListDataset(sequences)
        if not args.no_seq:
            dl = DataLoader(ds, batch_size=1, collate_fn=fw_collator, num_workers=4, shuffle=False)
            for i, batch in enumerate(dl):
                src, tgt = batch
                src = src.to(DEVICE)
                tgt = tgt.to(DEVICE)
                outputs = model(src, tgt)
                with torch.no_grad():
                    ce = outputs[OTHER_METRICS_KEY]['ce_loss'].item()
                df_out.loc[i, args.model_name + '_fw_score'] = -ce
            dl = DataLoader(ds, batch_size=1, collate_fn=bw_collator, num_workers=8, shuffle=False)
            for i, batch in enumerate(dl):
                src, tgt = batch
                src = src.to(DEVICE)
                tgt = tgt.to(DEVICE)
                outputs = model(src, tgt)
                with torch.no_grad():
                    ce = outputs[OTHER_METRICS_KEY]['ce_loss'].item()
                df_out.loc[i, args.model_name + '_bw_score'] = -ce
            df_out[args.model_name + '_seq_score'] = (df_out[args.model_name + '_fw_score'] + df_out[
                args.model_name + '_bw_score']) / 2
            fw_spearman = spearmanr(df_out[args.model_name + '_fw_score'], df_out['DMS_score']).statistic
            bw_spearman = spearmanr(df_out[args.model_name + '_bw_score'], df_out['DMS_score']).statistic
            seq_spearman = spearmanr(df_out[args.model_name + '_seq_score'], df_out['DMS_score']).statistic
            print(assay_name, fw_spearman, bw_spearman, seq_spearman)
            with open(summary_path, 'a') as f:
                f.write(','.join([assay_name, str(fw_spearman), str(bw_spearman), str(seq_spearman)]))
            if not args.msa:
                df_out[args.model_name + '_score'] = df_out[args.model_name + '_seq_score']
                df_out.to_csv(df_out_file, index=False)
                with open(summary_path, 'a') as f:
                    f.write('\n')
        else:
            msa_files = os.listdir(msa_dir)
            protein_name = '_'.join(assay_name.split('_')[:2])
            if protein_name == 'ANCSZ_Hobbs':
                protein_name = 'ANCSZ'
            for msa_file in msa_files:
                if msa_file.startswith(protein_name):
                    break
            else:
                print(protein_name)
            seqs = parse_fasta(os.path.join(msa_dir, msa_file))
            collator = MSAARCollator(tokenizer, flip_prob=0.0)
            if args.no_seq:
                replicates = 4
            else:
                replicates = 1
            msas = []
            for rep in range(replicates):
                msa_idx = np.random.choice(len(seqs) - 1, size=63) + 1
                if args.gap:
                    msa = [seqs[i].replace('.', '-').upper() for i in msa_idx]
                else:
                    msa = [seqs[i].replace('-', '').replace('.', '').upper() for i in msa_idx]
                msa_src, msa_tgt = collator([[None, msa]])
                msa_src = msa_src.to(DEVICE)
                msas.append(msa_src)
            dl = DataLoader(ds, batch_size=1, collate_fn=fw_collator, num_workers=4, shuffle=False)
            for i, batch in enumerate(dl):
                src, tgt = batch
                src = src.to(DEVICE)
                tgt = tgt.to(DEVICE)
                tgt = tgt[0, 1:]
                n, ell = src.shape
                df_out.loc[i, args.model_name + '_indel_score'] = 0  # have to create it first
                for rep, msa_src in enumerate(msas):
                    combined_src = torch.cat([msa_src, src], dim=1)
                    with torch.no_grad():
                        outputs = model.module(combined_src)['logits'][0, -ell:-1]
                        ce = torch.nn.functional.cross_entropy(outputs, tgt).item()
                    if replicates > 1:
                        df_out.loc[i, args.model_name + '_indel_score%d' %rep] = -ce
                    df_out.loc[i, args.model_name + '_indel_score' ] += -ce / len(msas)
                    # print(i, rep, df_out.loc[i, args.model_name + '_indel_score' ], flush=True)
            indel_spearman = spearmanr(df_out[args.model_name + '_indel_score'], df_out['DMS_score']).statistic
            if not args.no_seq:
                df_out[args.model_name + '_score'] = (df_out[args.model_name + '_seq_score'] + df_out[
                    args.model_name + '_indel_score']) / 2
                en_spearman = spearmanr(df_out[args.model_name + '_score'], df_out['DMS_score']).statistic
                print(assay_name, indel_spearman, en_spearman)
                with open(summary_path, 'a') as f:
                    f.write(',' + ','.join([str(indel_spearman), str(en_spearman)]) + '\n')
            else:
                print(assay_name, indel_spearman)
                with open(summary_path, 'a') as f:
                    f.write(','.join([assay_name, str(indel_spearman)]) + '\n')
            df_out.to_csv(df_out_file, index=False)


def train(args: argparse.Namespace) -> None:
    print(f"Starting job on rank {RANK} with local rank {LOCAL_RANK} and world size {WORLD_SIZE}")
    seed_everything(0)

    dist.init_process_group(backend="nccl")
    # get the config, tokenizer, and model
    torch.cuda.set_device(LOCAL_RANK)
    config, tokenizer, model, block = load_msa_config_and_model(os.path.join(args.model_path, "config.json"),
                                                                use_flash_attention_2=(not args.no_fa2))
    print("Done initializing model.", RANK)

    # Load model and optimizer onto CPU
    initial_epoch, total_steps, total_tokens, total_seqs, _ = load_checkpoint(
        model, None, None, args.model_path, args.checkpoint_step, rank=RANK
    )
    # Move only model to GPU
    model = model.to(DEVICE)
    model = model.to(torch.bfloat16)
    model = model.eval()

    padding_idx = tokenizer.pad_id  # PROTEIN_ALPHABET.index(PAD)
    print("Using {} as padding index".format(padding_idx))
    print("Using {} as masking index".format(tokenizer.mask_id))
    print(f"Model has {sum(p.numel() for p in model.parameters())} trainable parameters.")


    # Get files
    subst_dir = os.path.join(args.data_root, "DMS_ProteinGym_substitutions")
    indel_dir = os.path.join(args.data_root, "DMS_ProteinGym_indels")
    msa_dir = os.path.join(args.data_root, "DMS_msa_files")
    if not args.gap:
        zero_shot(indel_dir, os.path.join(args.out_fpath, 'indels'), msa_dir, model, tokenizer, args)
    zero_shot(subst_dir, os.path.join(args.out_fpath, 'substitutions'), msa_dir, model, tokenizer, args)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("out_fpath", type=str)
    parser.add_argument("data_root", type=str)
    parser.add_argument("model_path", type=str)
    parser.add_argument("model_name", type=str)
    parser.add_argument("checkpoint_step", type=int)
    parser.add_argument("--no_fa2", action="store_true")
    parser.add_argument("--msa", action="store_true")
    parser.add_argument("--gap", action="store_true")
    parser.add_argument("--no_seq", action="store_true")

    model_name_dict = {
        'jamba-3b-indel-gigaclust-120k-2': 'dayhoff-3b-msa-gigaref',
        'jamba-3b-cooldown': 'dayhoff-3b-msa-uniref90-cooldown',
        'jamba-3b-cooldown7': 'dayhoff-3b-msa-uniref90-cooldown',
        'jamba-170m-10mnovelty-36w': 'dayhoff-170m-1novelty',
        'jamba-170m-seq-36w': 'dayhoff-170m-uniref50',
        'jamba-170m-10mrmsd-36w': 'dayhoff-170m-rmsd',
        'jamba-170m-10mbothfilter-36w': 'dayhoff-170m-bothfilter',
        'jamba-3b-seq-sam-biar-fsdp-tok90k': 'dayhoff-3b-uniref90',
        'jamba-170m-10mnofilter-36w': 'dayhoff-170m-nofilter',
        'jamba-170m-seqsam-36w': 'dayhoff-170m-uniref90',
        'jamba-170m-gigaclust-36w': 'dayhoff-170m-gigaref'
    }

    args = parser.parse_args()
    if args.model_name in model_name_dict:
        args.model_name = model_name_dict[args.model_name]
    train(args)


if __name__ == "__main__":
    main()




