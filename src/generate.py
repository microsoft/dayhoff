import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
import argparse



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate text using a pre-trained model.")
    parser.add_argument("--repo-id", type=str, required=True, help="The repository ID of the model.")
    parser.add_argument("--model", type=str, required=True, help="The model name.")
    parser.add_argument("--max-length", type=int, default=50, help="The maximum length of the generated text.")
    args = parser.parse_args()
    

    set_seed(0)
    torch.set_default_device("cuda")

    model = AutoModelForCausalLM.from_pretrained(args.repo_id, subfolder = args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.repo_id, trust_remote_code=True)

    inputs = tokenizer(tokenizer.bos_token, return_tensors="pt", return_token_type_ids=False)

    outputs = model.generate(inputs['input_ids'],max_length=args.max_length,do_sample=True)
    sequence = tokenizer.batch_decode(outputs,skip_special_tokens=True)
    print(sequence)