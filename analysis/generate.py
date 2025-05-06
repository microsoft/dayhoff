import argparse
import os
from tqdm import tqdm
from transformers import SuppressTokensLogitsProcessor
import torch
from sequence_models.constants import START, STOP, CAN_AAS, SEP, GAP, MSA_PAD
from dayhoff.constants import UL_ALPHABET_PLUS, END_AL, END_UL, START_AL, START_UL
from dayhoff.utils import (load_msa_config_and_model,
                           load_checkpoint, seed_everything)
import torch.distributed as dist

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
    seed_everything(args.random_seed + RANK)
    dist.init_process_group(backend="nccl")
    
    # load model parameters from config file
    config, tokenizer, model, block = load_msa_config_and_model(os.path.join(args.in_fpath, "config.json"),
                                                                use_flash_attention_2=args.use_flash_attention_2)
    if args.verbose:
        print("Done initializing model.", RANK)

    # Load model and optimizer onto CPU
    _, total_steps, _, _, _ = load_checkpoint(
        model, None, None, args.in_fpath, args.checkpoint_step, rank=RANK
    )
    # Move only model to GPU
    model = model.to(DEVICE)
    model = model.to(torch.bfloat16)
    all_tokens = list(range(40))
    allowed_tokens = [UL_ALPHABET_PLUS.index(aa) for aa in CAN_AAS]
    if args.task == "sequence":
        if args.start_rev:
            start_seq = tokenizer.stop_id
            eos_id = int(tokenizer.start_id)
        else:
            start_seq = tokenizer.start_id
            eos_id = int(tokenizer.stop_id)
        max_len = config["max_len"]
    elif args.task == "gap_no_query":
        if args.start_rev:
            start_seq = UL_ALPHABET_PLUS.index(END_AL)
            eos_id = UL_ALPHABET_PLUS.index(START_AL)
        else:
            start_seq = UL_ALPHABET_PLUS.index(START_AL)
            eos_id = UL_ALPHABET_PLUS.index(END_AL)
        max_len = config["n_sequences"] * config["max_seq_len"]
    elif args.task == "indel_no_query":
        if args.start_rev:
            start_seq = UL_ALPHABET_PLUS.index(END_UL)
            eos_id = UL_ALPHABET_PLUS.index(START_UL)
        else:
            start_seq = UL_ALPHABET_PLUS.index(START_UL)
            eos_id = UL_ALPHABET_PLUS.index(END_UL)
        max_len = config["n_sequences"] * config["max_seq_len"]
    elif args.task == "gap_query":
        max_len = config["n_sequences"] * config["max_seq_len"]
        if args.start_rev:
            start_seq = UL_ALPHABET_PLUS.index(STOP)
            eos_id = UL_ALPHABET_PLUS.index(START_AL)
            all_tokens += [UL_ALPHABET_PLUS.index(START), UL_ALPHABET_PLUS.index(END_AL)]
        else:
            start_seq = UL_ALPHABET_PLUS.index(START)
            eos_id = UL_ALPHABET_PLUS.index(END_AL)
            all_tokens += [UL_ALPHABET_PLUS.index(STOP), UL_ALPHABET_PLUS.index(START_AL)]
    elif args.task == "indel_query":
        max_len = config["n_sequences"] * config["max_seq_len"]
        if args.start_rev:
            start_seq = UL_ALPHABET_PLUS.index(STOP)
            eos_id = UL_ALPHABET_PLUS.index(START_UL)
            all_tokens += [UL_ALPHABET_PLUS.index(START), UL_ALPHABET_PLUS.index(END_UL)]
        else:
            start_seq = UL_ALPHABET_PLUS.index(START)
            eos_id = UL_ALPHABET_PLUS.index(END_UL)
            all_tokens += [UL_ALPHABET_PLUS.index(STOP), UL_ALPHABET_PLUS.index(START_UL)]
    if "gap" in args.task or "indel" in args.task:
        allowed_tokens += [UL_ALPHABET_PLUS.index(SEP)]
    if "gap" in args.task:
        allowed_tokens += [UL_ALPHABET_PLUS.index(GAP)]
    allowed_tokens += [eos_id]
    seps = [SEP, START, STOP, END_UL, START_UL, END_AL, START_AL]
    start = torch.tensor([[start_seq]]).to(DEVICE)
    start = torch.repeat_interleave(start, args.batch_size, dim=0)
    model.module.generation_config.eos_token_id = eos_id
    sup = SuppressTokensLogitsProcessor([t for t in all_tokens if not t in allowed_tokens], device=DEVICE)
    if args.start_rev:
        task = args.task + ".rev"
    else:
        task = args.task + ".fwd"
    out_dir = os.path.join(args.out_fpath, args.model_name + '_' + str(total_steps) + "_" + task + '_t%.1f' %args.temp)
    if RANK == 0:
        os.makedirs(out_dir, exist_ok=True)
    unwritten_generations = []
    unwritten_ns = []
    for s in tqdm(range(args.n_generations // args.batch_size)):
        generated = model.module.generate(start, do_sample=True, logits_processor=[sup],
                                                 temperature=args.temp, num_beams=1, max_new_tokens=max_len,
                                          use_cache=True)
        untokenized = [tokenizer.untokenize(g) for g in generated]
        if args.task == "sequence":

            for n, unt in enumerate(untokenized):
                n_gen = s * args.batch_size + n
                if args.start_rev:
                    unt = unt[::-1]
                unwritten_generations.append(unt)
                unwritten_ns.append(n_gen)
                # save every 100 generations or last in case n_generations is less than 100 or not a multiple of 100
                if len(unwritten_generations) == 100 or s == args.n_generations // args.batch_size - 1: 
                    with open(os.path.join(out_dir, 'rank%d_seed%d.fasta' % (RANK, args.random_seed)), "a") as f:
                        for uwg, nwn in zip(unwritten_generations, unwritten_ns):
                            f.write(">%d_%d_%d\n" % (RANK, nwn, args.random_seed))
                            f.write(uwg.replace(START, "").replace(STOP, "").replace(MSA_PAD, "") + "\n")
                        unwritten_generations = []
                        unwritten_ns = []
        else:
            for n, unt in enumerate(untokenized):
                n_gen = s * args.batch_size + n
                with open(os.path.join(out_dir, 'rank%d_%d.fasta' %(RANK, n_gen)), "w") as f:
                    unt = unt.replace(MSA_PAD, "")[1:-1] # Strip out whatever stop and start
                    # Replace all things in the middle with new lines
                    for sep in seps:
                        unt = unt.replace(sep, " ")
                    unt = unt.split()
                    for i, seq in enumerate(unt):
                        f.write(">%d\n" %i)
                        if args.start_rev:
                            seq = seq[::-1]
                        f.write(seq + "\n")
                        print(">%d" %i)
                        print(seq, flush=True)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("in_fpath", type=str)  # location of checkpoint
    parser.add_argument("out_fpath", type=str)  # location to write to
    parser.add_argument("model_name", type=str)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--checkpoint_step", type=int, default=-1)
    parser.add_argument("--n_generations", type=int, default=100)
    parser.add_argument("--task", type=str, default="sequence")  # 'sequence' or 'msa'
    parser.add_argument("--temp", type=float, default=1.0)  #
    parser.add_argument("--random_seed", type=int, default=0)  #
    parser.add_argument("--start_rev", action="store_true")
    parser.add_argument("--dir", type=str, default="")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--use_flash_attention_2", action="store_true",default=False)

    args = parser.parse_args()
    if args.dir == "fwd":
        args.start_rev = False
    elif args.dir == "rev":
        args.start_rev = True
    generate(args)


if __name__ == "__main__":
    main()