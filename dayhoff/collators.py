from typing import Callable, Literal, Optional, Sequence, Tuple

import numpy as np
import torch
from sequence_models.constants import GAP, SEP, START, STOP

from dayhoff.constants import (
    END_AL,
    END_UL,
    FIM_MIDDLE,
    FIM_PREFIX,
    FIM_SUFFIX,
    START_AL,
    START_UL,
)
from evodiff.utils import Tokenizer


def pad_to_mult(max_len, pad_to_mult):
    "helper function to pad to multiple of pad_to_mult"
    max_len = (
        max_len
        if pad_to_mult is None
        else pad_to_mult * torch.ceil(max_len / pad_to_mult).to(dtype=torch.int)
    )
    return max_len


class OAMaskCollator:
    def __init__(
        self, tokenizer: Callable, pad_to_multiple_of: Optional[int] = None
    ) -> None:
        self.tokenizer = tokenizer
        self.pad_to_mult = pad_to_multiple_of

    def __call__(
        self, sequences: Sequence[tuple]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Tokenize the input sequences, generate random masks, and convert into a tensor batch.

        Parameters:
        -----------
        sequences: Sequence[tuple]
            A sequence of tuples containing the input sequences as the first elem in each tuple.

        Returns:
        --------
        input_tokens: torch.Tensor
            The input tokens with the mask tokens in place.
        targets: torch.Tensor
            The target tokens.
        masks: torch.Tensor
            The mask tensor.
        timesteps: torch.Tensor
            The number of timesteps in the sequence.
        """
        # tokenize the samples (tokenizer accepts tuples of (seq,))
        tokenized = [torch.tensor(self.tokenizer.tokenize(s)) for s in sequences]

        # pad the max length if needed
        lens = torch.tensor([len(t) for t in tokenized])
        max_len = lens.max()
        max_len = pad_to_mult(max_len, self.pad_to_mult)

        # allocate the  output to fill
        input_tokens = torch.full(
            (len(tokenized), max_len), self.tokenizer.pad_id, dtype=torch.long
        )
        targets = input_tokens.clone()
        masks = torch.zeros(len(tokenized), max_len, dtype=torch.bool)

        # D - t + 1 where D is the length of the sequence and t is a random int in [1, D)
        timesteps = (
            lens - torch.tensor(np.random.randint(1, [max(2, lt) for lt in lens])) + 1
        )
        for i, (length, ts, toks) in enumerate(zip(lens, timesteps, tokenized)):
            input_tokens[i, : len(toks)] = toks
            targets[i, : len(toks)] = toks

            # generate the mask (num_timestep samples between [0, D-1])
            mask_idx = np.random.choice(length.item(), ts.numpy(), replace=False)
            masks[i, mask_idx] = True
            input_tokens[i, mask_idx] = self.tokenizer.mask_id

        return input_tokens, targets, masks, timesteps


class LMCollator:
    def __init__(
        self,
        tokenizer: Tokenizer,
        *,
        flip_prob: float = 0.0,
        fim_prob: float = 0.0,
        min_fim_prefix_len: int = 0,
        min_fim_suffix_len: int = 0,
        fim_mode: Literal["psm", "spm", "both"] = "both",
        simple_spm: bool = False,
        pad_to_multiple_of: Optional[int] = None,
        swap_bos_eos_on_flip: bool = True,
    ) -> None:
        """A collator which randomly converts a subset of samples into FIM samples.

        Parameters:
        -----------
        tokenizer: Callable
            A callable which tokenizes a string into a sequence of integers.
        fim_prob: float
            The probability of converting a sample into a FIM sample. Default is 0.5.
        min_fim_prefix_len: int
            The minimum length of the prefix for the FIM sample. Default is 0.
        min_fim_suffix_len: int
            The minimum length of the suffix for the FIM sample. Default is 0.
        fim_mode: Literal["psm", "spm", "both"]
            The mode of FIM to use. "psm" presents prefix-suffix-middle. "spm" presents suffix-prefix-middle.
            "both" presents both, switching between the two with equal probability. Default is both.
        simple_spm: bool
            If True, SPM samples are presented in the form <suffix>suffix-aa's<prefix>prefix-aa's<middle>middle-aa's.
            If False, SPM samples are presented in the form <prefix><suffix>suffix-aa's<middle>prefix-aa's middle-aa's.
            Default is False.
        flip_prob: float
            The probability of flipping the sample (always prior to FIM). Default is 0.5.
        pad_to_multiple_of: Optional[int]
            If not None, the length of the sequence will be padded to a multiple of this value.
        swap_bos_eos_on_flip: bool
            If True, the the sequence will be preceded by EOS (rather than BOS) when flipped. Default is True.
        """
        assert 0 <= fim_prob <= 1, "FIM probability must be in [0, 1]"
        assert 0 <= flip_prob <= 1, "Flip probability must be in [0, 1]"

        self.tokenizer = tokenizer
        self.fim_prob = fim_prob
        self.flip_prob = flip_prob
        self.pad_to_mult = pad_to_multiple_of
        self.fim_mode = fim_mode
        self.simple_spm = simple_spm
        self.swap_bos_eos_on_flip = swap_bos_eos_on_flip
        self.splitter = self.make_splitter(min_fim_prefix_len, min_fim_suffix_len)

        # intentionally keep them as arrays so we can concat later
        self.fim_pid = self.tokenizer.tokenize([FIM_PREFIX])
        self.fim_sid = self.tokenizer.tokenize([FIM_SUFFIX])
        self.fim_mid = self.tokenizer.tokenize([FIM_MIDDLE])
        self.start_id = self.tokenizer.tokenize([START])
        self.stop_id = self.tokenizer.tokenize([STOP])

    @staticmethod
    def make_splitter(min_prefix_len: int, min_suffix_len: int) -> Callable:
        def splitter(sequence: str) -> Tuple[str, str, str]:
            prefix_len = np.random.randint(
                min_prefix_len, len(sequence) - min_suffix_len
            )
            suffix_len = np.random.randint(min_suffix_len, len(sequence) - prefix_len)
            prefix = sequence[:prefix_len]
            suffix = sequence[-suffix_len:]
            middle = sequence[prefix_len:-suffix_len]
            return prefix, middle, suffix

        return splitter

    def _wrap(self, *args, flipped: bool):
        if flipped and self.swap_bos_eos_on_flip:
            return np.concatenate([self.stop_id, *args, self.start_id])
        return np.concatenate([self.start_id, *args, self.stop_id])

    def __call__(
        self, data: Sequence[str]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # tokenize
        tokenized = [self.tokenizer.tokenize(s) for s in data]

        # flip as needed
        if self.flip_prob > 0:
            maybe_flip = np.random.choice(
                [-1, 1], (len(tokenized),), p=[self.flip_prob, 1 - self.flip_prob]
            )
            tokenized = [s[::f] for s, f in zip(tokenized, maybe_flip)]

        is_flipped = (
            lambda i_: self.flip_prob > 0 and maybe_flip[i_] == -1
        )  # noqa: E731

        # FIM as needed
        maybe_fim = np.random.choice(
            [False, True], len(tokenized), p=[1 - self.fim_prob, self.fim_prob]
        )
        if self.fim_mode == "both":
            fim_mode = np.random.choice(["psm", "spm"], len(tokenized))
        else:
            fim_mode = [self.fim_mode] * len(tokenized)

        for i, fim in enumerate(maybe_fim):
            if fim:
                # randomly split the sample
                prefix, middle, suffix = self.splitter(tokenized[i])

                # select the fim type
                if fim_mode[i] == "psm":
                    tokenized[i] = self._wrap(
                        self.fim_pid,
                        prefix,
                        self.fim_sid,
                        suffix,
                        self.fim_mid,
                        middle,
                        flipped=is_flipped(i),
                    )
                else:
                    if self.simple_spm:
                        tokenized[i] = self._wrap(
                            self.fim_sid,
                            suffix,
                            self.fim_pid,
                            prefix,
                            self.fim_mid,
                            middle,
                            flipped=is_flipped(i),
                        )
                    else:
                        tokenized[i] = self._wrap(
                            self.fim_pid,
                            self.fim_sid,
                            suffix,
                            self.fim_mid,
                            prefix,
                            middle,
                            flipped=is_flipped(i),
                        )
            else:
                tokenized[i] = self._wrap(tokenized[i], flipped=is_flipped(i))

        # pad to a multiple of pad_to_mult
        max_len = max(len(s) for s in tokenized)

        # inflate to a mult of pad_to_mult
        if self.pad_to_mult is not None:
            max_len = (
                self.pad_to_mult * np.ceil(max_len / self.pad_to_mult).astype(int)
            ).item()

        out = torch.full(
            (len(tokenized), max_len), self.tokenizer.pad_id, dtype=torch.long
        )
        for i, s in enumerate(tokenized):
            out[i, : len(s)] = torch.tensor(s, device=out.device)

        lbls = out.clone()

        # no penalty for not predicting padding or FIM tokens
        lbls[lbls == self.tokenizer.pad_id] = -100
        lbls[lbls == self.fim_pid[0]] = -100
        lbls[lbls == self.fim_sid[0]] = -100
        lbls[lbls == self.fim_mid[0]] = -100

        return out, lbls


class MSAOAMaskCollator:
    def __init__(
        self, tokenizer: Callable, pad_to_multiple_of: Optional[int] = None
    ) -> None:
        self.tokenizer = tokenizer
        self.pad_to_mult = pad_to_multiple_of

    def __call__(
        self, batch_msa: "list"
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        depths = torch.tensor([len(msa) for msa in batch_msa])
        lens = torch.tensor([len(msa[0]) for msa in batch_msa])
        max_len = (
            lens.max() + 2
        )  # add start/stop token in lens for pre-pad array creation
        max_len = pad_to_mult(max_len, self.pad_to_mult)
        max_depth = depths.max()
        max_depth = pad_to_mult(max_depth, self.pad_to_mult)

        tokenized = [
            torch.tensor(np.vstack([self.tokenizer.tokenizeMSA(s) for s in msa]))
            for msa in batch_msa
        ]
        d = torch.tensor(
            [len(msa.flatten()) for msa in tokenized]
        )  # flattened msa shapes, excluding START/STOP tokens

        # allocate the  output to fill
        src = torch.full(
            (len(tokenized), max_depth, max_len),
            self.tokenizer.pad_id,
            dtype=torch.long,
        )
        targets = src.clone()
        masks = torch.zeros(len(tokenized), max_depth, max_len, dtype=torch.bool)

        # D - t + 1 where D is the length of the sequence and t is a random int in [1, D)
        timesteps = d - torch.tensor(np.random.randint(1, [max(2, lt) for lt in d])) + 1
        for i, (length, ts, msa) in enumerate(zip(d, timesteps, tokenized)):
            # save targets with start/stop
            targets[i, : depths[i], 1 : lens[i] + 1] = msa
            targets[i, : depths[i], 0] = self.tokenizer.start_id
            targets[i, : depths[i], lens[i] + 1] = self.tokenizer.stop_id

            # generate the mask on flattened MSAs
            input_tokens = msa.flatten()
            mask_arr = torch.zeros(depths[i] * lens[i], dtype=torch.bool)
            mask_idx = np.random.choice(length.item(), ts.numpy(), replace=False)
            mask_arr[mask_idx] = True
            mask_arr = mask_arr.reshape(depths[i], lens[i])
            masks[i, : depths[i], 1 : lens[i] + 1] = mask_arr

            # save masked inputs with start/stop
            input_tokens[mask_idx] = self.tokenizer.mask_id
            input_tokens = input_tokens.reshape(depths[i], lens[i])
            src[i, : depths[i], 1 : lens[i] + 1] = input_tokens
            src[i, : depths[i], 0] = self.tokenizer.start_id  # add START
            src[i, : depths[i], lens[i] + 1] = self.tokenizer.stop_id  # add STOP
        return src, timesteps, targets, masks


class MSAARCollator:
    """
    src: START + input + padding
    tgt: input + STOP + padding
    mask : 1 where tgt is not padding (non-pad locations)
    """

    def __init__(
        self,
        tokenizer: Callable,
        pad_to_multiple_of: Optional[int] = None,
        flip_prob: int = 0.5,
        query_last_prob: int = 0.5,
            trim_to: int = None,
            trim_to2: int = None
    ) -> None:
        self.tokenizer = tokenizer
        self.pad_to_mult = pad_to_multiple_of
        self.al_start_id = self.tokenizer.tokenize(START_AL)[0]
        self.al_stop_id = self.tokenizer.tokenize(END_AL)[0]
        self.ul_start_id = self.tokenizer.tokenize(START_UL)[0]
        self.ul_stop_id = self.tokenizer.tokenize(END_UL)[0]
        self.gap_id = self.tokenizer.tokenize(GAP)[0]
        self.sep_id = self.tokenizer.tokenize(SEP)[0]
        self.sep_tensor = torch.tensor([self.sep_id])
        self.al_start_tensor = torch.tensor([self.al_start_id])
        self.al_stop_tensor = torch.tensor([self.al_stop_id])
        self.ul_start_tensor = torch.tensor([self.ul_start_id])
        self.ul_stop_tensor = torch.tensor([self.ul_stop_id])
        self.start_tensor = torch.tensor([self.tokenizer.start_id])
        self.stop_tensor = torch.tensor([self.tokenizer.stop_id])
        self.flip_prob = flip_prob
        self.query_last_prob = query_last_prob
        self.trim_to = trim_to
        self.trim_to2 = trim_to2

    def __call__(self, batch_msa: "list") -> Tuple[torch.Tensor, torch.Tensor]:


        # Determine if sequence only
        if len(batch_msa[0]) == 1:
            tokenized_queries = [torch.tensor(self.tokenizer.tokenizeMSA(msa[0])) for msa in batch_msa]
            len_queries = [len(query) + 2 for query in tokenized_queries]
            tokenized_homologs = None
            max_len = max(len_queries)
        else:
            tokenized_homologs = [
                [torch.tensor(self.tokenizer.tokenizeMSA(s)) for s in msa[1]]
                for msa in batch_msa
            ]
            len_homologs = [sum([len(s) for s in msa]) + 2 + len(msa) for msa in tokenized_homologs]
            tokenized_queries = []
            len_queries = []
            for item in batch_msa:
                if item[0] is None:
                    tokenized_queries.append(None)
                    len_queries.append(0)
                else:
                    tokenized_queries.append(torch.tensor(self.tokenizer.tokenizeMSA(item[0])))
                    len_queries.append(len(item[0]) + 2)
            max_len = max([lq + lh for lq, lh in zip(len_queries, len_homologs)])

        # pre-pad final array
        n_items = len(tokenized_queries)
        src = torch.full(
            (n_items, max_len), self.tokenizer.pad_id, dtype=torch.long
        )
        # flip as needed
        is_flipped = np.random.rand(n_items) < self.flip_prob
        is_query_last = np.random.rand(n_items) < self.query_last_prob

        for i, tq in enumerate(tokenized_queries):
            if tq is not None:
                query = torch.cat((self.start_tensor, tq, self.stop_tensor, ), 0)
            else:
                query = torch.tensor([])
            if is_flipped[i]:
                query = query.flip(0)
            if tokenized_homologs is not None:
                th = tokenized_homologs[i]
                if len(th) > 0:
                    homologs = torch.cat([torch.cat([seq, self.sep_tensor], 0) for seq in th])
                    gapped = torch.any(homologs == self.gap_id)
                    if gapped:
                        h_start_tensor = self.al_start_tensor
                        h_stop_tensor = self.al_stop_tensor
                    else:
                        h_start_tensor = self.ul_start_tensor
                        h_stop_tensor = self.ul_stop_tensor
                    homologs = torch.cat(
                        (
                            h_start_tensor,
                            homologs,
                            h_stop_tensor,
                        ),
                        0
                    )
                else:
                    homologs = torch.tensor([])
                if is_flipped[i]:
                    homologs = homologs.flip(0)
                if is_query_last[i]:
                    src_i = torch.cat([homologs, query], 0)
                else:
                    src_i = torch.cat([query, homologs], 0)
            else:
                src_i = query
            src[i, :len(src_i)] = src_i
        tgt = src.clone()
        tgt[tgt == self.tokenizer.pad_id] = -100
        # throw away the ends of batches that are too big because lengths aren't accurate when there are indels
        if self.trim_to is not None:
            n, ell = src.shape
            if n > self.trim_to // ell:
                src = src[: self.trim_to // ell]
                tgt = tgt[: self.trim_to // ell]
            if ell > 512 * 64:
                src = src[:, :512 * 64]
                tgt = tgt[:, :512 * 64]
        if self.trim_to2 is not None:
            src = src[:, :self.trim_to2]
            tgt = tgt[:, :self.trim_to2]
        return src, tgt