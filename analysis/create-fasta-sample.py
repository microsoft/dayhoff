import argparse
import json
import os
from multiprocessing import cpu_count

import numpy as np
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from datasets import load_from_disk
from tqdm import tqdm

"""
Sample usage: python src/create-fasta-sample.py --input_hf_path data/uniref50_202401/arrow/train/ --output_fasta_file data/generated_proteins/uniref50_202401_10M.fasta --n 10000000
              python src/create-fasta-sample.py --input_hf_path /mnt/blob/dayhoff/data/gigaref_full/full/arrow/ --output_fasta_file data/generated_proteins/gigaref_clustered_15M.fasta --indexes_path /mnt/blob/dayhoff/data/gigaref_full/no_singletons/train_random_idxs_15M.json

"""
COMMON_AMINOACIDS = set([
                    "A",
                    "C",
                    "D",
                    "E",
                    "F",
                    "G",
                    "H",
                    "I",
                    "K",
                    "L",
                    "M",
                    "N",
                    "P",
                    "Q",
                    "R",
                    "S",
                    "T",
                    "V",
                    "W",
                    "Y",
                ])

def save_to_fasta(sequence_id_labels_tuples, output_file):
    """
    Save a list of tuples in the form (sequence, accession ,[labels]) to a FASTA file.

    :param sequence_label_tuples: List of tuples containing sequences and labels
    :param output_file: Path to the output FASTA file
    """
    records = []
    for _, (
        sequence,
        id,
        labels,
    ) in enumerate(tqdm(sequence_id_labels_tuples)):
        # Create a description from labels, joined by space
        description = " ".join(labels)

        record = SeqRecord(Seq(sequence), id=id, description=description)
        records.append(record)

    # Write the SeqRecord objects to a FASTA file
    with open(output_file, "w") as output_handle:
        SeqIO.write(records, output_handle, "fasta")
        print("Saved FASTA file to " + output_file)


def valid_sequence(example):
    # Check that every character in the sequence is allowed
    return set(example["formatted"][0]).issubset(COMMON_AMINOACIDS) 

def get_hf_ds_sample(ds, n, indexes_path = None):
    """Get a sample of n sequences from the dataset."""
    
    if n is not None and indexes_path is None:
        sampled_idxs = np.random.choice(
            len(ds), size = n, replace = False
        )
    elif n is None and indexes_path is not None:
        with open(indexes_path, "r") as f:
            sampled_idxs = json.load(f)["idxs"]
    elif n is None and indexes_path is None:
        raise ValueError("You must specify either n or indexes_path.")
    elif n is not None and indexes_path is not None:
        raise ValueError("You must specify only one of n or indexes_path.")
    
    
    return ds.select(sampled_idxs)

def create_fasta_sample_from_hf(input_hf_path, output_fasta_file, n, num_proc = cpu_count(), filter_rare_aa = False, indexes_path = None):
    """Create a FASTA sample
    from a Hugging Face dataset."""


    print("Loading dataset from disk...")
    ds = load_from_disk(input_hf_path)
    
    print("Generating HF sample...")
    sample_df = get_hf_ds_sample(ds = ds,
                                 n = n,
                                 indexes_path=indexes_path
        ).map(
        lambda record: {"formatted": (record["sequence"], record["accession"], record["description"])},
        num_proc = num_proc
    )

    if filter_rare_aa:
        sample_df = sample_df.filter(valid_sequence)
        
    print("Formatting sample as list")
    sample_df = sample_df["formatted"]

    print("Saving sample to FASTA file...")
    save_to_fasta(
        sample_df,
        output_fasta_file
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create a FASTA sample from a Hugging Face dataset."
    )
    parser.add_argument(
        "--input_hf_path",
        type=str,
        required=True,
        help="Path to the Hugging Face dataset. This should be a path to a directory containing the dataset files."
    )
    parser.add_argument(
        "--output_fasta_file",
        type=str,
        required=True,
        help="Path to the output FASTA file."
    )
    parser.add_argument(
        "--n",
        type=int,
        required=False,
        help="Number of sequences to sample from the dataset."
    )
    parser.add_argument(
        "--indexes_path",
        type=str,
        required=False,
        help="Path to the JSON file containing the indexes of the desired sample."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility."
    )

    parser.add_argument(
        "--filter_rare_aa",
        action="store_true",
        help="Filter out sequences with rare amino acids.",
        default=False
    )

    parser.add_argument(
        "--num_proc",
        type=int,
        default=cpu_count(),
        help="Number of processes to use for the conversion. Default is the number of CPU cores available"
    )

    args = parser.parse_args()


    #Validate args
    if args.n is None and args.indexes_path is None:
        raise ValueError("You must specify either n or indexes_path.")
    elif args.n is not None and args.indexes_path is not None:
        raise ValueError("You must specify only one of n or indexes_path.")
    
    is_amlt = os.environ.get("AMLT_OUTPUT_DIR", None) is not None

    if is_amlt:
        print("Output dir: ", os.environ["AMLT_OUTPUT_DIR"])
        args.input_hf_path = os.path.join(os.environ["AMLT_DATA_DIR"], args.input_hf_path)
        if args.indexes_path is not None:
            args.indexes_path = os.path.join(os.environ["AMLT_DATA_DIR"], args.indexes_path)
        if args.output_fasta_file is not None:
            args.output_fasta_file = os.path.join(os.environ["AMLT_OUTPUT_DIR"], args.output_fasta_file)

    # Set the random seed for reproducibility
    np.random.seed(args.seed)
    create_fasta_sample_from_hf(
        input_hf_path=args.input_hf_path,
        output_fasta_file=args.output_fasta_file,
        n=args.n,
        filter_rare_aa = args.filter_rare_aa,
        indexes_path=args.indexes_path,
        num_proc=args.num_proc
        )