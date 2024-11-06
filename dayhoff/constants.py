import enum

import sequence_models.constants as constants


class TaskType(enum.Enum):
    LM = "lm"
    OADM = "oadm"


FIM_MIDDLE = "_"
FIM_PREFIX = "("
FIM_SUFFIX = ")"
START_AL = "["
END_AL = "]"
START_UL = "{"
END_UL = "}"
MSA_ALPHABET_PLUS = (
    constants.MSA_ALPHABET_PLUS
    + FIM_MIDDLE
    + FIM_PREFIX
    + FIM_SUFFIX
    + START_AL
    + END_AL
)
UL_ALPHABET_PLUS = (
    constants.MSA_ALPHABET_PLUS
    + START_AL
    + END_AL
    + START_UL
    + END_UL
)
# ensure there are no unintentional character overlaps
assert len(MSA_ALPHABET_PLUS) == len(set(MSA_ALPHABET_PLUS))
assert len(UL_ALPHABET_PLUS) == len(set(UL_ALPHABET_PLUS))

