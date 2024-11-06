import os
from typing import List, Optional, Union

from transformers.tokenization_utils import AddedToken, PreTrainedTokenizer

MASK = "#"
MSA_PAD = "!"
UL_ALPHABET_PLUS = "ACDEFGHIKLMNPQRSTVWYBZXJOU-*#@!/[]{}"
MSA_AAS = "ACDEFGHIKLMNPQRSTVWYBZXJOU-"
GAP = "-"
START = "@"
STOP = "*"
SEP = "/"
END_AL = "]"
END_UL = "}"
START_AL = "["
START_UL = "{"

class ProteinTokenizer(PreTrainedTokenizer):

    def __init__(
        self,
        protein_alphabet: str = UL_ALPHABET_PLUS,
        model_max_length: int = 2048,
        pad_token=MSA_PAD,
        mask_token=MASK,
        all_aas=MSA_AAS,
        gap_token=GAP,
        bos_token=START,
        eos_token=STOP,
        sep_token=SEP,
        **kwargs
    ):
        """Character tokenizer for Hugging Face transformers.

        model_max_length (int): Model maximum sequence length.
        """
        self.alphabet = list("".join(protein_alphabet))
        self.all_aas = list("".join(all_aas))
        self.a_to_i = {u: i for i, u in enumerate(self.alphabet)}
        self.i_to_a = {i: u for i, u in enumerate(self.alphabet)}
        self.gap_token = gap_token

        
        bos_token = AddedToken(bos_token, lstrip=False, rstrip=False) if isinstance(bos_token, str) else bos_token
        eos_token = AddedToken(eos_token, lstrip=False, rstrip=False) if isinstance(eos_token, str) else eos_token
        sep_token = AddedToken(sep_token, lstrip=False, rstrip=False) if isinstance(sep_token, str) else sep_token
        mask_token = AddedToken(mask_token, lstrip=False, rstrip=False) if isinstance(mask_token, str) else mask_token 
        pad_token = AddedToken(pad_token, lstrip=False, rstrip=False) if isinstance(pad_token, str) else pad_token
        gap_token = AddedToken(gap_token, lstrip=False, rstrip=False) if isinstance(gap_token, str) else gap_token

        super().__init__(
            pad_token=pad_token,
            mask_token=mask_token,
            eos_token=eos_token,
            bos_token=bos_token,
            sep_token=sep_token,
            model_max_length=model_max_length,
            **kwargs
        )

    @property
    def vocab_size(self):
        return len(self.alphabet)
    
    @property
    def gap_token_id(self):
        return self.convert_tokens_to_ids(self.gap_token)

    def get_vocab(self):
        return self.a_to_i

    def _tokenize(self, text: str) -> List[str]:
        return list(text)

    def _convert_token_to_id(self, token) -> int:
        return self.a_to_i[token]

    def _convert_id_to_token(self, index) -> str:
        return self.i_to_a[index]

    def convert_tokens_to_string(self, tokens):
        return "".join(tokens)

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        result = token_ids_0
        if token_ids_1 is not None:
            raise NotImplementedError("This tokenizer does not support two sequences")
        return result

    def get_special_tokens_mask(
        self,
        token_ids_0: List[int],
        token_ids_1: Optional[List[int]] = None,
        already_has_special_tokens: bool = False,
    ) -> List[int]:
        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0,
                token_ids_1=token_ids_1,
                already_has_special_tokens=True,
            )

        result = [0] * len(token_ids_0)
        if token_ids_1 is not None:
            raise NotImplementedError("This tokenizer does not support two sequences")

        return result

    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Identifies the type of token. 0 for the first sentence, 1 for the second sentence if it exists
        """

        result = len(token_ids_0) * [0]

        if token_ids_1 is not None:
            raise NotImplementedError("This tokenizer does not support two sequences")
        return result

    def save_pretrained(self, save_directory: Union[str, os.PathLike], **kwargs):
        super().save_pretrained(save_directory, **kwargs)
    
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None):
        return ()