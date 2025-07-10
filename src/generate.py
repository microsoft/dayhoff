import argparse
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate text using a pre-trained model.")
    parser.add_argument("--out-dir", type=str, required=True, help="The output directory for generated text.")
    parser.add_argument("--model", type=str, required=True, help="The model name.")
    parser.add_argument("--repo-id", type=str, default="microsoft/dayhoff", help="The repository ID of the model.")
    parser.add_argument("--max-length", type=int, default=2048, help="The maximum length of the generated text.")
    parser.add_argument("--n-generations", type=int, default=100, help="The number of generations to produce.")
    parser.add_argument("--temp", type=float, default=1.0, help="The temperature for sampling.")
    parser.add_argument("--min-p", type=float, default=0.0, help="Minimum probability for sampling.")
    parser.add_argument("--device", type=int, default=0, help="The device to use for computation.")
    parser.add_argument("--random-seed", type=int, default=0, help="Random seed for reproducibility.")
    parser.add_argument("--no-fa2", action="store_true",help="Disable FlashAttention 2")
    args = parser.parse_args()
    
    set_seed(args.random_seed)
    torch.set_default_device("cuda:%d" %args.device)

    model = AutoModelForCausalLM.from_pretrained(args.repo_id, subfolder = args.model,torch_dtype=torch.bfloat16,use_flash_attention_2=not args.no_fa2)
    tokenizer = AutoTokenizer.from_pretrained(args.repo_id, trust_remote_code=True)
    
    inputs = tokenizer(tokenizer.bos_token, return_tensors="pt", return_token_type_ids=False)
    outputs = model.generate(inputs['input_ids'],
                             max_length=args.max_length,
                             do_sample=True,
                             num_return_sequences=args.n_generations,
                             temperature=args.temp,
                             min_p = args.min_p
                             )
    generated = tokenizer.batch_decode(outputs,skip_special_tokens=True)

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, 'device%d_seed%d_temp%.2f_minp%.2f.fasta' % (args.device, args.random_seed, args.temp, args.min_p))
    print("Writing to %s" % out_path)
    # Write the generated sequences to a file
    with open(out_path, "a") as f:
        for idx, seq in enumerate(generated):
            f.write(">%d\n" % (idx))
            f.write(seq + "\n")
    