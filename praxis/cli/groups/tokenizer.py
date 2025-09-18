"""Tokenizer training CLI arguments."""


class TokenizerGroup:
    """Tokenizer training configuration arguments."""

    name = "tokenizer"

    @classmethod
    def add_arguments(cls, parser):
        """Add tokenizer arguments to the parser."""
        group = parser.add_argument_group(cls.name)

        group.add_argument(
            "--train-tokenizer",
            action="store_true",
            default=False,
            help="Train a new tokenizer and exit (shortcut mode)",
        )

        group.add_argument(
            "--tokenizer-train-type",
            type=str,
            choices=["bpe", "unigram"],
            default="unigram",
            help="The type of tokenizer to train",
        )

        group.add_argument(
            "--tokenizer-num-examples",
            type=int,
            default=5_000_000,
            help="The number of examples to use for tokenizer training",
        )

        group.add_argument(
            "--tokenizer-train-vocab-size",
            type=int,
            choices=[1024, 2048, 4096, 8192, 16384, 32768, 65536],
            default=16384,
            help="The vocab size for tokenizer training",
        )
