import argparse
import os
import torch
import torch.distributed as dist
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed


# Constants
MSA_ALPHABET_PLUS = "ACDEFGHIKLMNPQRSTVWYBZXJOU-*#@!/_()[]"
END_AL = "]"

# default to a single-GPU setup if not present
if "RANK" not in os.environ and "WORLD_SIZE" not in os.environ:
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "8889"

RANK = int(os.environ["RANK"])
WORLD_SIZE = int(os.environ["WORLD_SIZE"])

DEVICE = torch.device(f"cuda:{RANK}")
print("device", DEVICE)


def generate(args: argparse.Namespace) -> None:

    # Create output directory
    os.makedirs(args.out_fpath, exist_ok=True)

    set_seed(args.random_seed)
    dist.init_process_group(backend="nccl")
    
    tokenizer = tokenizer = AutoTokenizer.from_pretrained(
        args.repo_id,
        trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(
        args.repo_id,
        subfolder=args.model_variant)
    
    # Move only model to GPU
    model = model.to(DEVICE)
    model.eval()

    if args.task == "sequence":
        if args.start_rev:
            start = tokenizer.eos_token_id
            stop = tokenizer.bos_token_id
        else:
            start = tokenizer.bos_token_id
            stop = tokenizer.eos_token_id
        max_len = args.max_len
    elif args.task == "msa":
        start = tokenizer.bos_token_id
        stop = tokenizer.encode(END_AL)
        max_len = args.msa_n_seqs * args.msa_max_seq_len

    untokenized_out = []

    for s in tqdm(range(args.n_generations)):
        if args.verbose:
            print(MSA_ALPHABET_PLUS)
            print(tokenizer.a_to_i)
            print(tokenizer.i_to_a)
        # Start from START token
        batch_size = 1
        sample = torch.full((batch_size, 1), start, dtype=torch.long).to(DEVICE)
        
        # Iterate over each residue until STOP or max length
        reach_stop = False  # initialize
        for _ in tqdm(range(max_len + 1)): # Adding 1 to account for the start token
            if reach_stop == False:  # Add residues until it predicts STOP token or hits max seq len
                with torch.inference_mode():
                    prediction = model(sample)["logits"]

                p = prediction[:, -1, : len(MSA_ALPHABET_PLUS)]  # predict next token
                p = torch.nn.functional.softmax(p / args.temp, dim=1)  # exp
                p_sample = torch.multinomial(p, num_samples=1).to(DEVICE)
                sample = torch.cat((sample, p_sample), dim=1)
                if args.verbose:
                    print(tokenizer.decode(sample[0]))
                if p_sample == stop:
                    reach_stop = True
            else:
                break
        # print(sample)
        untokenized = tokenizer.decode(sample[0])
        print("final sequence: ", untokenized)
        if args.start_rev:
            untokenized_out.append(untokenized[::-1])  # append forward sequence
            # print("fixed", untokenized[::-1])
        else:
            untokenized_out.append(untokenized)
        if args.task == "sequence":
            with open(args.out_fpath + "/generated_samples.fasta", "a") as f:
                f.write(">3BCOOLED_SEQUENCE_" + str(s) + "\n" + str(untokenized[1:-1]) + "\n")
    
    #destroy process group
    dist.destroy_process_group()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", type=str, help="Hugging Face repo id e.g., microsoft/dayhoff ")  # location of checkpoint
    parser.add_argument("--model-variant", type=str, help="Model variant to use -- equivalent to subfolder in the repo. Example: jamba-170m-seqsam-36w")  # location of checkpoint
    parser.add_argument("--out_fpath", type=str)  # location to write to
    parser.add_argument("--max-len", type=int, default=2048)
    parser.add_argument("--msa-max-seq-len", type=int, default=512)
    parser.add_argument("--msa-n-seqs", type=int, default=64)
    parser.add_argument("--n_generations", type=int, default=100)
    parser.add_argument("--task", type=str, default="sequence")  # 'sequence' or 'msa'
    parser.add_argument("--temp", type=float, default=1.0)  #
    parser.add_argument("--random_seed", type=int, default=0)  #
    parser.add_argument("--start_rev", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    generate(args)


if __name__ == "__main__":
    main()