"""Constants for byte-level tokenization and encoding."""

# Token type IDs (0-3 are special tokens, OFFSET=4 starts byte values)
BOE_ID: int = 0  # Beginning of entropy
PAD_ID: int = 0  # Padding (shares ID with BOE)
BOS_ID: int = 1  # Beginning of sequence
EOS_ID: int = 2  # End of sequence
SEP_ID: int = 3  # Separator (shares ID with BPE marker, matching BLT layout)
BPE_ID: int = 3  # BPE marker (same as SEP_ID, matching BLT reference)

# Vocabulary constants
BYTE_UNITS: int = 256  # Number of byte values (0-255)
OFFSET: int = 4  # Offset for special tokens (same as original BLT)
