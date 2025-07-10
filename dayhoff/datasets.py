import json
import os
import os.path as osp

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sequence_models.constants import GAP, MSA_ALPHABET, START, STOP
from sequence_models.utils import parse_fasta
from torch.utils.data import Dataset


def parse_msa(path):
    parsed_msa = parse_fasta(path)
    parsed_msa = list(
        filter(None, parsed_msa)
    )  # get rid of any empty entries from commented inputs
    # Convert to array for easy selection
    msa_array = [np.array(list(m)) for m in parsed_msa]
    # Compute a bunch of masks
    is_lower = [np.char.greater(m, "Z") for m in msa_array]
    is_gap = [np.char.equal(m, GAP) for m in msa_array]
    is_period = [np.char.equal(m, ".") for m in msa_array]
    is_aligned = [
        ~np.logical_or(lower, period) for period, lower in zip(is_lower, is_period)
    ]
    is_aa = [~np.logical_or(g, p) for g, p in zip(is_gap, is_period)]
    # Get the aligned positions
    aligned_msa = [m[is_a] for m, is_a in zip(msa_array, is_aligned)]
    aligned_msa = ["".join(seq) for seq in aligned_msa]
    # Get all the residues
    unaligned_msa = [np.char.upper(m[aa]) for m, aa in zip(msa_array, is_aa)]
    unaligned_msa = ["".join(seq) for seq in unaligned_msa]
    # Get the aligned index for each residue in the unaligned MSA
    corrected_indices = [
        (np.cumsum(is_a) - 1)[aa] for is_a, aa in zip(is_aligned, is_aa)
    ]
    return aligned_msa, unaligned_msa, corrected_indices


def msa_subsampling(sliced_msa, n_sequences, selection_type):
    """
    :param sliced_msa: msa sequences with query sliced out
    :param n_sequences: int, number of sequences in MSA to subsample to
    :return: constructed msa
    """
    if selection_type == "random":
        msa_depth = len(sliced_msa)
        random_idx = np.random.choice(msa_depth, size=n_sequences, replace=False)
        msa_sequences = [list(sliced_msa[int(i)]) for i in random_idx]
    elif selection_type == "max_hamming":
        msa_sequences = []
        msa_subset = sliced_msa
        msa_subset_ind = np.arange(len(msa_subset))
        # start with rand seq to initialize for maxhamming subsampling
        random_ind = np.random.choice(msa_subset_ind)
        # Keep track of selected indices
        random_idx = [random_ind]
        # Delete selected index from remaining indices
        msa_subset_ind = msa_subset_ind[msa_subset_ind != random_ind]
        random_seq = msa_subset[random_ind]
        msa_sequences.append(list(random_seq))
        random_seq = np.expand_dims(random_seq, axis=0)
        msa_subset = np.delete(msa_subset, random_ind, axis=0)
        m = len(msa_subset)
        distance_matrix = np.ones((n_sequences - 1, m))
        # subsample new seqs using max distance between min(hamming) array
        for i in range(n_sequences - 1):
            curr_dist = cdist(random_seq, msa_subset, metric="hamming")
            distance_matrix[i] = curr_dist
            col_min = np.min(distance_matrix, axis=0)  # (1,num_choices)
            max_ind = np.argmax(col_min)
            random_ind = max_ind
            # The corrected index is simply the original index in that position
            corrected_random_ind = msa_subset_ind[random_ind]
            random_idx += [corrected_random_ind]
            # Delete selected index
            msa_subset_ind = msa_subset_ind[msa_subset_ind != corrected_random_ind]
            random_seq = msa_subset[random_ind]
            msa_sequences.append(list(random_seq))
            random_seq = np.expand_dims(
                random_seq, axis=0
            )  # KKY: not really sure why we need this?
            msa_subset = np.delete(msa_subset, random_ind, axis=0)
            distance_matrix = np.delete(distance_matrix, random_ind, axis=1)
        random_idx = np.array(random_idx)
    else:
        raise Exception("Invalid selection type; choose from 'random' or 'max_hamming'")
    return msa_sequences, random_idx  # Returns aligned sequences and their indices


class ListDataset(Dataset):

    def __init__(self, data):
        super().__init__()
        self.data = data

    def __getitem__(self, item):
        return (self.data[item], )

    def __len__(self):
        return len(self.data)


class UniRefDataset(Dataset):
    """
    Dataset that pulls from UniRef/Uniclust downloads.

    The data folder should contain the following:
    - 'consensus.fasta': consensus sequences, no line breaks in sequences
    - 'splits.json': a dict with keys 'train', 'valid', and 'test' mapping to lists of indices
    - 'lengths_and_offsets.npz': byte offsets for the 'consensus.fasta' and sequence lengths
    """

    def __init__(self, data_dir: str, split: str, max_len=2048, split_file=None):
        self.data_dir = data_dir
        self.split = split
        if split_file is None:
            split_file = osp.join(data_dir, "splits.json")
        with open(split_file, "r") as f:
            self.indices = json.load(f)[self.split]
        if os.path.exists(os.path.join(self.data_dir, "offsets.dat")):
            metadata = {"seq_offsets": np.memmap(os.path.join(self.data_dir, "offsets.dat"), mode="r", dtype="uint64")}
        else:
            metadata = np.load(osp.join(self.data_dir, "lengths_and_offsets.npz"))
        self.offsets = metadata["seq_offsets"]
        self.max_len = max_len
        self.file = None

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        idx = self.indices[idx]
        offset = self.offsets[idx]
        if self.file is None:
            self.file = open(self.data_dir + "consensus.fasta")

        self.file.seek(offset)
        consensus = self.file.readline()[:-1]
        if len(consensus) - self.max_len > 0:
            start = np.random.choice(len(consensus) - self.max_len)
            stop = start + self.max_len
        else:
            start = 0
            stop = len(consensus)
        consensus = consensus[start:stop]
        return (consensus,)


class OpenProteinDataset(Dataset):
    """
    Dataset that pulls from OpenFold OpenProteinSet https://registry.opendata.aws/openfold/
    Training data has been pre preprocessed into train/val/test sets via homology filtering against uniref sets

    The data folder should contain the following:
    - 'rtest_index.csv', 'test_index.csv', 'val_index.csv', 'train_index.csv', 4 csv files corresponding to the
       filepaths for MSAs in each split, MSA depths, and MSA sequence lengths

    Will return a subsampled MSA, with START/STOP tokens added to each seq in the MSA
    """

    def __init__(
        self,
        data_dir: str,
        split: str,
        selection_type: str,
        n_sequences: int,
        max_seq_len: int,
        min_depth=None,
        gap_fraction=4,
        is_amlt=False,
        indel_frac=0.0,
        no_query_frac=0.0,
    ):
        """
        :param data_dir: str, path to directory containing openfold dataset
        :param split: str, split using for evaluation 'train', 'val', 'test', or 'rtest'
        :param selection_type: str, subsampling approach 'max_hamming' or 'random'
        :param n_sequences: int, number of sequences in MSA to subsample to
        :param max_seq_len: int, maximmum sequence length
        :param min_depth: minimum number of sequences needed to sample MSA
        :param gap_fraction: fraction of gap content to filter out (e.g 4 = filter out sequences with more than 1/4 (25%) gap content)
            (e.g 4 = filter out sequences with more than 1/4 (25%) gap content)
        :param is_amlt:
        :param indels: If True, return full sequences without gaps but with insertions.
        """
        alphabet = list("".join(MSA_ALPHABET))
        self.a_to_i = {u: i for i, u in enumerate(alphabet)}
        self.i_to_a = np.array(list(alphabet))
        self.gap_id = self.a_to_i[GAP]
        self.start_id = self.a_to_i[START]
        self.stop_id = self.a_to_i[STOP]
        self.gap_fraction = gap_fraction
        self.data_dir = data_dir
        self.is_amlt = is_amlt
        self.indel_frac = indel_frac
        self.no_query_frac = no_query_frac
        print("IS AMLT", self.is_amlt)

        # load in files from correct split
        split_path = os.path.join(data_dir, "out", split + "_index_processed.csv")
        metadata = pd.read_csv(split_path, usecols=["path", "depth", "length"])
        # filter depths
        if min_depth is not None:
            print("filtering sequences less than", min_depth)
            metadata = metadata[metadata["depth"] >= min_depth]

        self.filenames = metadata["path"].values.tolist()
        self.indices = metadata.index.values.tolist()
        self.depths = metadata["depth"].values.tolist()
        self.lengths = metadata["length"].values.tolist()
        if split == "rtest":
            exclude = [
                "alignments_41/B8KZ16/merged.a3m",
                "alignments_41/A0A1V6BYG1/merged.a3m",
                "alignments_41/A3T2H5/merged.a3m",
                "alignments_41/C0CU71/merged.a3m",
                "alignments_41/U6L0S1/merged.a3m",
                "alignments_41/A0A2D3P9P4/merged.a3m",
                "alignments_41/C4YSB1/merged.a3m",
                "alignments_41/W2Z7Y7/merged.a3m",
                "alignments_41/A0A1G2B126/merged.a3m",
                "alignments_41/A0A233HTX8/merged.a3m"
            ]
            keep_idx = []
            for i, fn in enumerate(self.filenames):
                for exc in exclude:
                    if exc in fn:
                        break
                else:
                    keep_idx.append(i)
            self.filenames = [self.filenames[i] for i in keep_idx]
            self.indices = [self.indices[i] for i in keep_idx]
            self.depths = [self.depths[i] for i in keep_idx]
            self.lengths = [self.lengths[i] for i in keep_idx]
        self.n_sequences = n_sequences
        self.max_seq_len = max_seq_len
        self.selection_type = selection_type
        self.min_depth = min_depth

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        path = self.filenames[idx]  # grab str from nested list
        if self.is_amlt:
            path = os.path.join(self.data_dir, path[15:])
        msa_seq_len = self.lengths[idx]
        parsed_msa, unaligned_msa, corrected_idx = parse_msa(path)
        tokenized_msa = np.vstack(
            [np.array([self.a_to_i[a] for a in seq]) for seq in parsed_msa]
        )
        assert self.depths[idx] == len(
            parsed_msa
        ), "msa depth does not match index file, indexing might be wrong"
        assert msa_seq_len == len(
            parsed_msa[0]
        ), "msa does not match index file, indexing might be wrong"
        if msa_seq_len > self.max_seq_len:
            slice_start = np.random.choice(msa_seq_len - self.max_seq_len + 1)
            seq_len = self.max_seq_len
        else:
            slice_start = 0
            seq_len = msa_seq_len
        # Slice sequence of max_seq_len
        sliced_msa = tokenized_msa[:, slice_start : slice_start + seq_len]
        # Reduce high-gap content in sliced sequences
        idx = [
            i
            for i, seq in enumerate(sliced_msa)
            if np.count_nonzero(seq == self.gap_id) < len(seq) / self.gap_fraction
        ]

        # Filter the aligned, unaligned, and corrected_idx lists
        sliced_msa = [sliced_msa[i] for i in idx]
        unaligned_msa = [unaligned_msa[i] for i in idx]
        corrected_idx = [corrected_idx[i] for i in idx]
        msa_depth = len(sliced_msa)
        indel = np.random.random() < self.indel_frac
        no_query = indel and np.random.random() < self.no_query_frac
        # The number of sequences to randomly select, the choices to select from
        if no_query:
            n_sequences = self.n_sequences
            msa_choices = sliced_msa
        else:
            n_sequences = self.n_sequences - 1
            msa_choices = sliced_msa[1:]

        if msa_depth <= self.n_sequences:
            msa_sequences = [list(seq) for seq in msa_choices]
            msa_idx = np.arange(len(msa_sequences))
            np.random.shuffle(msa_idx)
            msa_sequences = [msa_sequences[i] for i in msa_idx]
        else:
            msa_sequences, msa_idx = msa_subsampling(
                msa_choices, n_sequences, selection_type=self.selection_type
            )
        if indel:
            if not no_query:  # put the query back in and correct the indices
                msa_idx = msa_idx + 1
            msa_sequences = [unaligned_msa[i] for i in msa_idx]
            msa_sequences = [np.array(list(s)) for s in msa_sequences]
            slice_idx = [corrected_idx[i] for i in msa_idx]
            slice_idx_bool = [
                np.logical_and(s >= slice_start, s < slice_start + seq_len)
                for s in slice_idx
            ]
            total_len = sum([s.sum() for s in slice_idx_bool])
            slice_len = seq_len
            while total_len > self.max_seq_len * self.n_sequences:
                slice_len = slice_len // 2
                slice_idx_bool = [
                    np.logical_and(s >= slice_start, s < slice_start + slice_len)
                    for s in slice_idx
                ]
                total_len = sum([s.sum() for s in slice_idx_bool])
            homologs = ["".join(s[i]) for s, i in zip(msa_sequences, slice_idx_bool)]
            if not no_query:
                anchor_seq = unaligned_msa[0]
            else:
                anchor_seq = None
        else:
            homologs = ["".join(self.i_to_a[list(seq)]) for seq in msa_sequences]
            anchor_seq = "".join(self.i_to_a[sliced_msa[0]])  # query seq
        return anchor_seq, homologs