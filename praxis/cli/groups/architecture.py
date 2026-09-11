"""Architecture-related CLI arguments."""

import argparse

from praxis import registry


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
            registry="encoders",
            default=None,
            help="Encoder integration to use",
        )

        group.add_argument(
            "--decoder-type",
            type=str,
            registry="decoders",
            default="sequential",
            help="How to process layers in the decoder",
        )

        group.add_argument(
            "--block-type",
            type=str,
            registry="blocks",
            default="transformer",
            help="The type of block to use for every intermediate decoder layer",
        )

        group.add_argument(
            "--ffn-type",
            type=str,
            registry="dense",
            default="glu",
            help="The feedforward-network implementation to use within each block",
        )

        group.add_argument(
            "--attention-type",
            type=str,
            registry="attention",
            default="modular",
            help="The base attention implementation to use",
        )

        group.add_argument(
            "--memory-type",
            type=str,
            registry="memory",
            default="none",
            help="Titans-style long-term memory profile",
        )

        group.add_argument(
            "--encoding-type",
            type=str,
            registry="encoding",
            default="rope",
            help="The positional encoding to use for sequence length extrapolation",
        )

        group.add_argument(
            "--controller-type",
            type=str,
            registry="controllers",
            default="base",
            help="Various methods used to route inputs through experts in the decoder",
        )

        group.add_argument(
            "--orchestration-type",
            type=str,
            registry="orchestration",
            default="none",
            help="Remote-expert pool profile: backend sidecar of tiny experts "
            "(joinable from the web Stage tab) + a mixing strategy",
        )

        group.add_argument(
            "--router-type",
            type=str,
            registry="routers",
            default=None,
            help="How to route tokens at every layer",
        )

        group.add_argument(
            "--halting-type",
            type=str,
            registry="halting",
            default=None,
            help="Halting strategy for recurrent depth loops",
        )

        group.add_argument(
            "--width-type",
            type=str,
            registry="width",
            default=None,
            help="Mixture-of-widths policy: deflate each recurrent step's inner "
            "rank to a helically-precessing slice. Presets tune the floor/peak "
            "of the arch (none = full width)",
        )

        group.add_argument(
            "--transform-type",
            type=str,
            choices=["none", *registry.namespace("transforms").keys()],
            registry="transforms",
            default="none",
            help="Model-transform profile: walk the module tree and rewrite the "
            "matched parameters in place. `tie_*` stores 1/d of a weight and "
            "derives the rest by fixed signed permutation",
        )

        group.add_argument(
            "--residual-type",
            type=str,
            registry="residuals",
            default="standard",
            help="The style of residual connection to use",
        )

        group.add_argument(
            "--compression-type",
            type=str,
            registry="compression",
            default="none",
            help="The type of sequence compression to use",
        )

        group.add_argument(
            "--sorting-type",
            type=str,
            registry="sorting",
            default="none",
            help="The type of feature sorting to use",
        )

        group.add_argument(
            "--activation-type",
            type=str,
            choices=registry.namespace("activations"),
            registry=("activations", "activation_types"),
            default="mish",
            help=(
                "The activation function to use. A bare name here; an experiment "
                "config may instead give a `{type, values}` mixture - see "
                "praxis/activations"
            ),
        )

        group.add_argument(
            "--norm-type",
            type=str,
            registry="normalization",
            default="rms_norm",
            help="The type of normalization to use",
        )

        group.add_argument(
            "--head-type",
            type=str,
            registry="heads",
            default="forward",
            help="The type of language modeling head to use",
        )

        group.add_argument(
            "--target-batch-size",
            type=int,
            default=256,
            help="The actual batch size to use, including accumulation steps",
            doc=(
                "Rows per optimizer step. When it exceeds --batch-size, gradients "
                "accumulate over ceil(target / batch_size) microbatches. It also sets "
                "the default warmup (4x this value) and scales the validation cadence. "
                "Under --governor gns_batch it becomes the ceiling the governor may "
                "grow the effective batch to."
            ),
        )

        group.add_argument(
            "--block-size",
            type=int,
            default=512,
            help="The base sequence length to train with",
            doc=(
                "Tokens per training sequence. When --batch-size is large enough, some "
                "batches trade rows for length: a multiplier of 2, 4 or 8 stretches "
                "the sequence and divides the row count by its square, keeping "
                "attention cost flat. --seq-curriculum chooses how often each "
                "multiplier is drawn."
            ),
        )

        group.add_argument(
            "--max-position-embeddings",
            type=int,
            default=None,
            help="Maximum positional capacity (unset = the model config's own default)",
            doc=(
                "Positional capacity for the modules that keep a table or count of "
                "positions (learned position embeddings, byte-latent and "
                "abstractinator encoders). An explicit value too small for the longest "
                "sequence-multiplied batch is raised to fit, with a notice."
            ),
        )

        from praxis.tokenizers import VOCAB_SIZE_CHOICES

        group.add_argument(
            "--vocab-size",
            type=int,
            choices=VOCAB_SIZE_CHOICES,
            default=16384,
            help="The absolute vocab size to use, though some architectures might scale it differently",
        )

        group.add_argument(
            "--hash-buckets",
            type=int,
            nargs="+",
            default=None,
            help="Buckets per n-gram hash table in the byte-latent input "
            "embedding: one value, or one per window size (e.g. 1024 1024 "
            "2048 for 3-, 4- and 5-byte windows). Independent of vocab_size, "
            "which under a byte tokenizer is a constant 256. Defaults to "
            "vocab_size when unset",
        )

        group.add_argument(
            "--codebook-size",
            type=int,
            default=None,
            help="Entries in the encoder's VQ codebook (abstractinator "
            "bottleneck). Defaults to vocab_size when unset",
        )

        group.add_argument(
            "--depth",
            type=int,
            default=None,
            help="The max number of experts to route through (defaults to num_layers)",
            doc=(
                "Block calls per forward pass. With depth above --num-layers the pass "
                "cycles through the same blocks again (recurrent depth); below it, "
                "only the first depth blocks run. Per-depth modules (residuals, router "
                "biases) are sized by it, and --halting-type may stop a pass before "
                "it."
            ),
        )

        group.add_argument(
            "--num-experts",
            type=int,
            default=1,
            help="Number of experts per layer (1 = no MoE)",
            doc=(
                "Not a feedforward mixture-of-experts. It is read by the routers: "
                "SMEAR-style routers reuse one block at every position and give each "
                "weight num_experts low-rank deviations to merge; the prismatic router "
                "keeps num_experts full copies of the block. Without a --router-type "
                "it has almost no effect."
            ),
        )

        group.add_argument(
            "--num-layers",
            type=int,
            default=2,
            help="Number of distinct blocks in the decoder stack",
            doc=(
                "Distinct blocks built. --depth sets how many calls a forward pass "
                "makes; when it is larger, the pass reuses these blocks in order, so "
                "num_layers is the unique-parameter count and depth the compute."
            ),
        )

        group.add_argument(
            "--hidden-size",
            type=int,
            default=256,
            help="The size of the model's hidden dimensions",
            doc=(
                "Width of the residual stream every block reads and writes. When "
                "--embed-size differs, the embedding and the tied head add a "
                "projection between the two."
            ),
        )

        group.add_argument(
            "--embed-size",
            type=int,
            default=192,
            help="The size of the model's embedding dimension (if applicable)",
            doc=(
                "Width of the token embedding. Equal to --hidden-size, the projection "
                "between them is left out; the byte-latent encoders and byte-level MTP "
                "also work at this width."
            ),
        )

        group.add_argument(
            "--dropout",
            type=float,
            default=0.0,
            help="The percentage of neurons to drop-out during training",
        )

        group.add_argument(
            "--num-heads",
            type=int,
            default=4,
            help="Number of attention heads",
        )

        group.add_argument(
            "--num-queries",
            type=int,
            default=2,
            help="Number of queries per attention head (for GQA/MQA)",
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

        group.add_argument(
            "--window-size",
            type=int,
            default=None,
            help="Sliding window size for attention (None = full attention). Only used with hex attention.",
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

        group.add_argument(
            "--regularizers",
            type=str,
            nargs="*",
            registry="regularizers",
            default=["contrastive_isotropy"],
            help=(
                "Additive representation-shaping losses to apply (space-separated; "
                "pass with no values to disable all)"
            ),
        )

        group.add_argument(
            "--mtp-type",
            type=str,
            # Bank types (one module owns all depths) are special-cased in
            # MultiTokenPrediction rather than per-depth registry modules:
            # "vear" = shared harmonic-expert pool, sliding-window merged;
            # "serpent_rnn" = one shared gated serpent cell unrolled K times;
            # "per_depth" = K independent light harmonic transforms, nothing
            # shared (the DeepSeek shape with a pointwise transform).
            choices=list(registry.namespace("mtp").keys())
            + ["vear", "serpent_rnn", "per_depth"],
            registry="mtp",
            default=None,
            help="MTP module type (omit to disable MTP)",
        )

        group.add_argument(
            "--mtp-depth",
            type=int,
            default=1,
            help="Number of Multi-Token Prediction depths",
        )

        group.add_argument(
            "--mono-type",
            type=str,
            registry="mono",
            default=None,
            help=(
                "Mono-forward graph cutting in the sequential decoder: detach "
                "hidden states on a cut schedule and train each segment from a "
                "local goodness score (omit to disable)"
            ),
        )
