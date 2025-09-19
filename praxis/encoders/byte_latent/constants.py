"""Constants for byte-level tokenization and encoding."""

# Token type IDs
BOE_ID: int = 0  # Beginning of entity
PAD_ID: int = 0  # Padding (shares ID with BOE)
BOS_ID: int = 1  # Beginning of sequence
EOS_ID: int = 2  # End of sequence
SEP_ID: int = 3  # Separator
BPE_ID: int = 4  # BPE marker

# Vocabulary constants
BYTE_UNITS: int = 256  # Number of byte values (0-255)
OFFSET: int = 4  # Offset for special tokens (same as original BLT)