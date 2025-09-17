"""Architecture-related CLI arguments."""

import argparse

from praxis import (
    ACTIVATION_REGISTRY,
    ATTENTION_REGISTRY,
    BLOCK_REGISTRY,
    COMPRESSION_REGISTRY,
    CONTROLLER_REGISTRY,
    DECODER_REGISTRY,
    ENCODER_REGISTRY,
    ENCODING_REGISTRY,
    EXPERT_REGISTRY,
    HEAD_REGISTRY,
    NORMALIZATION_REGISTRY,
    RESIDUAL_REGISTRY,
    ROUTER_REGISTRY,
    SORTING_REGISTRY,
)


class ArchitectureGroup:
    """Model architecture configuration arguments."""

    name = "architecture"

    @classmethod
    def add_arguments(cls, parser):
        """Add architecture arguments to the parser."""
        group = parser.add_argument_group(cls.name)

        group.add_argument(
            "--encoder-type",
            type=str,
            choices=list(ENCODER_REGISTRY.keys()),
            default=None,
            help="Encoder integration to use",
        )

        group.add_argument(
            "--decoder-type",
            type=str,
            choices=list(DECODER_REGISTRY.keys()),
            default="sequential",
            help="How to process layers in the decoder",
        )

        group.add_argument(
            "--block-type",
            type=str,
            choices=BLOCK_REGISTRY.keys(),
            default="transformer",
            help="The type of block to use for every intermediate decoder layer",
        )

        group.add_argument(
            "--expert-type",
            type=str,
            choices=EXPERT_REGISTRY.keys(),
            default="glu",
            help="The integration to use for feedforward networks",
        )

        group.add_argument(
            "--attention-type",
            type=str,
            choices=ATTENTION_REGISTRY.keys(),
            default="standard",
            help="The base attention implementation to use",
        )

        group.add_argument(
            "--encoding-type",
            type=str,
            choices=ENCODING_REGISTRY.keys(),
            default="rope",
            help="The positional encoding to use for sequence length extrapolation",
        )

        group.add_argument(
            "--controller-type",
            type=str,
            choices=CONTROLLER_REGISTRY.keys(),
            default="base",
            help="Various methods used to route inputs through experts in the decoder",
        )

        group.add_argument(
            "--router-type",
            type=str,
            choices=ROUTER_REGISTRY.keys(),
            default=None,
            help="How to route tokens at every layer",
        )

        group.add_argument(
            "--residual-type",
            type=str,
            choices=RESIDUAL_REGISTRY.keys(),
            default="standard",
            help="The style of residual connection to use",
        )

        group.add_argument(
            "--compression-type",
            type=str,
            choices=COMPRESSION_REGISTRY.keys(),
            default="none",
            help="The type of sequence compression to use",
        )

        group.add_argument(
            "--sorting-type",
            type=str,
            choices=SORTING_REGISTRY.keys(),
            default="none",
            help="The type of feature sorting to use",
        )

        group.add_argument(
            "--activation",
            type=str,
            choices=ACTIVATION_REGISTRY.keys(),
            default="mish",
            help="The primary activation function to use",
        )

        group.add_argument(
            "--norm-type",
            type=str,
            choices=NORMALIZATION_REGISTRY.keys(),
            default="rms_norm",
            help="The type of normalization to use",
        )

        group.add_argument(
            "--head-type",
            type=str,
            choices=HEAD_REGISTRY.keys(),
            default="forward",
            help="The type of language modeling head to use",
        )

        group.add_argument(
            "--target-batch-size",
            type=int,
            default=256,
            help="The actual batch size to use, including accumulation steps",
        )

        group.add_argument(
            "--block-size",
            type=int,
            default=512,
            help="The base sequence length to train with",
        )

        group.add_argument(
            "--vocab-size",
            type=int,
            choices=[1024, 2048, 4096, 8192, 16384, 32768, 65536],
            default=16384,
            help="The absolute vocab size to use, though some architectures might scale it differently",
        )

        group.add_argument(
            "--depth",
            type=int,
            default=None,
            help="The max number of experts to route through (defaults to num_layers)",
        )

        group.add_argument(
            "--num-experts",
            type=int,
            default=1,
            help="Number of experts per layer (1 = no MoE)",
        )

        group.add_argument(
            "--num-layers",
            type=int,
            default=2,
            help="Number of layer components for controllers",
        )

        group.add_argument(
            "--hidden-size",
            type=int,
            default=256,
            help="The size of the model's hidden dimensions",
        )

        group.add_argument(
            "--embed-size",
            type=int,
            default=192,
            help="The size of the model's embedding dimension (if applicable)",
        )

        group.add_argument(
            "--dropout",
            type=int,
            default=0.1,
            help="The percentage of neurons to drop-out during training",
        )

        group.add_argument(
            "--num-heads",
            type=cls._validate_num_heads,
            default="4:2",
            help="The ratio of heads to queries per-head. (example: '4:2' is equal to 3 heads, with 2 queries per head)",
        )

        group.add_argument(
            "--head-size",
            type=int,
            default=None,
            help="Specify the inner head dimension",
        )

        group.add_argument(
            "--k-heads",
            type=int,
            default=None,
            help="A sparse MoE, controlling the number of heads to sample. Should be smaller than num_heads to enable.",
        )

        group.add_argument(
            "--kv-rank",
            type=int,
            default=None,
            help="Set this value to factorize key/value projections, making them low-rank. A value of 1 is lowest.",
        )

        # Boolean architecture flags
        group.add_argument(
            "--linear",
            action="store_true",
            default=False,
            help="Use a Linear (O(n)) attention mechanism",
        )

        group.add_argument(
            "--differential",
            action="store_true",
            default=False,
            help="Use a Differential Attention mechanism",
        )

        group.add_argument(
            "--stickbreaking",
            action="store_true",
            default=False,
            help="Use a Stickbreaking Attention mechanism",
        )

        group.add_argument(
            "--memory",
            action="store_true",
            default=False,
            help="Use a long-term episodic memory module",
        )

        group.add_argument(
            "--mla",
            action="store_true",
            default=False,
            help="Use Multi-Head Latent Attention (MLA)",
        )

        group.add_argument(
            "--mta",
            action="store_true",
            default=False,
            help="Use Multi-Token Attention (MTA)",
        )

        group.add_argument(
            "--mega",
            action="store_true",
            default=False,
            help="Equip the attention mechanism with exponentially-moving average-based gating",
        )

        group.add_argument(
            "--gated",
            action="store_true",
            default=False,
            help="Add a gating network to attention outputs",
        )

        group.add_argument(
            "--evolve",
            action="store_true",
            default=False,
            help="Use a genomic bottleneck",
        )

        group.add_argument(
            "--scaled",
            action="store_true",
            default=False,
            help="Scale the output of each layer by the inverse square root of its depth",
        )

        group.add_argument(
            "--bidirectional",
            action="store_true",
            default=False,
            help="Enable bidirectional language modeling (forward and backward prediction)",
        )

        group.add_argument(
            "--tie-weights",
            action="store_true",
            default=False,
            help="Tie embedding and output projection weights to reduce parameters",
        )

    @staticmethod
    def _validate_num_heads(x):
        """Validate num_heads format 'X:Y' where X and Y are positive integers."""
        if ":" in x:
            parts = x.split(":")
            if (
                len(parts) == 2
                and all(p.isdigit() for p in parts)
                and all(int(p) > 0 for p in parts)
            ):
                return x
        raise argparse.ArgumentTypeError(
            f"'{x}' is not in format 'X:Y', where X and Y are positive integers"
        )

    @classmethod
    def process_args(cls, args):
        """Process architecture arguments after parsing."""
        # Parse num_heads format - handle both string "X:Y" and integer formats
        if hasattr(args, "num_heads") and args.num_heads:
            # Only process if it's a string with the "X:Y" format
            if isinstance(args.num_heads, str) and ":" in args.num_heads:
                parts = args.num_heads.split(":")
                # Note: We don't override the original num_heads string here
                # That's handled in processors/config.py