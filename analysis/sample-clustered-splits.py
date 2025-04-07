import json
import random
import argparse
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument(
        "--clustered_splits",
        type=str,
        required=True,
        help="Path to the clustered splits JSON file."
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to the output JSON file."
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        help="Number of samples to take from the dataset.",
        required=True
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility."
    )

    args = parser.parse_args()

    with open(args.clustered_splits, "rb") as f:
        data = json.loads(f.read())

    flattened_clustered = {}
    idxs = []
    for v in tqdm(data['train']):
        idxs.extend(v)

    # Set seed
    random.seed(args.seed)
    flattened_clustered['idxs'] = random.sample(idxs, args.num_samples)
    with open(args.output_path, "w") as f:
        json.dump(flattened_clustered, f)
