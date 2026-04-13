"""Training-related CLI arguments."""

from praxis.trainers import TRAINER_REGISTRY


class TrainingGroup:
    """Training configuration arguments."""

    name = "training"

    @classmethod
    def add_arguments(cls, parser):
        """Add training arguments to the parser."""
        group = parser.add_argument_group(cls.name)

        group.add_argument(
            "--trainer-type",
            type=str,
            choices=list(TRAINER_REGISTRY.keys()),
            default="backpropagation",
            help=(
                "Training strategy to use. 'backpropagation' is standard "
                "gradient descent; 'mono_forward' is the pipelined "
                "Mono-Forward trainer (each layer trains locally against "
                "a shared output head, O(1) activation memory in depth, "
                "Ray-backed workers today)."
            ),
        )

        group.add_argument(
            "--max-steps",
            type=int,
            default=None,
            help="Maximum number of training steps (None for infinite training)",
        )

        group.add_argument(
            "--val-every",
            type=int,
            default=1024,
            help=(
                "Run validation every N effective steps (multiplied by "
                "target_batch_size / batch_size internally to get the "
                "raw-batch cadence). Default 1024 matches the historical "
                "Lightning default. Lower values like 64 or 128 give "
                "faster feedback at the cost of more validation overhead. "
                "Can also be set in experiment YAML as 'val_every'."
            ),
        )

        # Ray Mono-Forward specific arguments. See PROJECT_PLAN.md for the
        # full design; PLAN.md D1-D8 for the decision rationale.
        group.add_argument(
            "--ray-address",
            type=str,
            default=None,
            help=(
                "Ray cluster address for --trainer-type mono_forward. "
                "Default None means 'start a fresh in-process Ray cluster' - "
                "correct for both the single-host Phase 2/3 path and the "
                "Phase 4 multi-raylet compose test (which exports "
                "RAY_ADDRESS=127.0.0.1:6379 in the environment, taking "
                "precedence over this flag). Pass an explicit address like "
                "'127.0.0.1:6379' to join a pre-existing cluster. Do NOT "
                "pass 'auto' unless a cluster definitely exists - 'auto' "
                "errors loud if nothing is running."
            ),
        )

        group.add_argument(
            "--ray-num-replicas-per-layer",
            type=int,
            default=1,
            help=(
                "Number of LayerActor replicas to spawn per Mono-Forward "
                "layer. Phase 3 keeps this at 1; replica-level data "
                "parallelism is a Phase 5+ concern."
            ),
        )

        group.add_argument(
            "--ray-head-sync-every",
            type=int,
            default=50,
            help=(
                "Number of batches between shared-head synchronization "
                "rounds across actors (only meaningful for "
                "--trainer-type mono_forward)."
            ),
        )

        group.add_argument(
            "--ray-pipeline-api",
            type=str,
            choices=["compiled", "manual"],
            default="manual",
            help=(
                "Whether to drive the Mono-Forward pipeline with Ray's "
                "experimental_compile() DAG API ('compiled') or with a "
                "manual ray.wait()-based driver loop ('manual'). Default is "
                "'manual' because it only uses stable Ray APIs."
            ),
        )
