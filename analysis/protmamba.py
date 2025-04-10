import argparse
import os
from tqdm import tqdm


import numpy as np

import torch

from ProtMamba_ssm.dataloaders import *
from ProtMamba_ssm.utils import *
from ProtMamba_ssm.modules import *
from protxlstm.dataloaders import ProteinMemmapDataset



def generate(args: argparse.Namespace) -> None:
    torch.random.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    device = torch.device("cuda:%d" %args.device)

    # load model parameters from config file
    checkpoint = "/home/kevyan/Prot-xLSTM/checkpoints/protxlstm_102M_60B"
    config_update_kwargs = {
        "mlstm_backend": "chunkwise_variable",
        "mlstm_chunksize": 1024,
        "mlstm_return_last_state": True}

    model = load_model(checkpoint,
                       model_class=xLSTMLMHeadModel,
                       device=device,
                       dtype=torch.bfloat16,
                       **config_update_kwargs,
                       )
    model = model.eval()


    os.makedirs(args.out_fpath, exist_ok=True)
    out_file = os.path.join(args.out_fpath, 'xlstm.fasta')
    msa_files = os.listdir(args.msas_fpath)
    with open(out_file, 'w') as f:
        for num, msa_filename in tqdm(enumerate(msa_files)):
            msa_sequences = load_sequences_from_msa_file(os.path.join(args.msas_fpath, msa_filename))
            if len(msa_sequences) == 1:
                print(msa_filename)
                continue
            protein_list = [msa.upper() for msa in msa_sequences[1:57]]
            # tokenize context sequences
            tokens = tokenizer(protein_list, concatenate=True)
            # load data class
            data_class = ProteinMemmapDataset(
                sample=False,
                max_msa_len=-1,
                reverse=False,
                seed=0,
                troubleshoot=False,
                fim_strategy="multiple_span",
                always_mask=False,
                max_position_embeddings=2048,
                max_seq_position_embeddings=768,
                add_position_ids="1d",
                mask_fraction=0.0,
                max_patches=5
            )

            # get number of context sequences
            num_context_sequences = len(protein_list)
            input_ids, pos_ids = data_class.sample_sequences(tokens.numpy()[0], num_sequences=num_context_sequences)
            input_ids.append(AA_TO_ID["<cls>"])
            input_ids = torch.asarray(input_ids, dtype=torch.int64)[None, :].to(device)
            pos_ids.append(0)
            pos_ids = torch.asarray(pos_ids, dtype=torch.int64)[None, :].to(device)

            # generate sequences
            output = generate_sequence(model,
                                       input_ids,
                                       position_ids=pos_ids,
                                       is_fim={},
                                       max_length=(input_ids.shape[1] + 1000),
                                       temperature=1.0,
                                       top_k=20,
                                       top_p=0.9,
                                       return_dict_in_generate=True,
                                       output_scores=True,
                                       eos_token_id=torch.tensor([AA_TO_ID["<cls>"]]).to(device),
                                       chunk_chunk_size=2 ** 15,
                                       device=device)
            new_seq = reorder_masked_sequence(output["generated"][0])
            f.write(">" + msa_filename[:-6] + "\n")
            f.write(new_seq + "\n")
            f.flush()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("msas_fpath", type=str)  # location of msas
    parser.add_argument("out_fpath", type=str)  # location to write to
    parser.add_argument("--random_seed", type=int, default=0)  #
    parser.add_argument("--device", type=int, default=0)  #


    args = parser.parse_args()
    generate(args)


if __name__ == "__main__":
    main()

model = load_model(checkpoint, model_class=MambaLMHeadModelwithPosids, device=device, dtype=torch.bfloat16)