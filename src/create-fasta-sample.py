from datasets import load_from_disk
import numpy as np
import pyfastx
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from tqdm import tqdm
import argparse
from multiprocessing import cpu_count

"""
Sample usage: python src/create-fasta-sample.py --input_hf_path data/uniref50_202401/arrow/train/ --output_fasta_file data/generated_proteins/uniref50_202401_10M.fasta --n 10000000
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

def get_hf_ds_sample(ds, n):
    """Get a sample of n sequences from the dataset."""
    sampled_idxs = np.random.choice(
        len(ds), size = n, replace = False
    )
    return ds.select(sampled_idxs)

def create_fasta_sample_from_hf(input_hf_path, output_fasta_file, n, num_proc = cpu_count(), filter_rare_aa = False):
    """Create a FASTA sample
    from a Hugging Face dataset."""

    print("Loading dataset from disk...")
    ds = load_from_disk(input_hf_path)
    
    print("Generating HF sample...")
    sample_df = get_hf_ds_sample(ds, n).map(
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
        required=True,
        help="Number of sequences to sample from the dataset."
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

    args = parser.parse_args()

    # Set the random seed for reproducibility
    np.random.seed(args.seed)
    create_fasta_sample_from_hf(
        input_hf_path=args.input_hf_path,
        output_fasta_file=args.output_fasta_file,
        n=args.n,
        filter_rare_aa = args.filter_rare_aa,
        )