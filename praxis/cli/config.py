"""Typed view over the processed CLI arguments.

``RunConfig`` replaces the long block of ``processed_args.get(...)`` reads
that main.py used to carry. Each training-relevant flag becomes a typed
field with the same default it had inline, so the call sites downstream
read ``cfg.batch_size`` instead of digging through a dict.
"""

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from praxis.utils import coerce_to_list


def _resolve_optimizer_wrappers(get) -> List[str]:
    """Ordered optimizer-wrapper keys. An explicit ``optimizer_wrappers`` list
    wins; otherwise the legacy --trac/--ortho/--lookahead/--schedule-free
    booleans are translated (deprecated, for un-migrated configs)."""
    wrappers = list(coerce_to_list(get("optimizer_wrappers", [])) or [])
    if wrappers:
        return wrappers
    for legacy in ("trac", "ortho", "lookahead", "schedule_free"):
        if get(legacy, False):
            wrappers.append(legacy)
    return wrappers


@dataclass
class RunConfig:
    """All per-run settings, resolved from processed CLI arguments."""

    # Identity / IO
    seed: int
    vocab_size: int
    cache_dir: str  # base build dir; the namespaced run dir lives on RunContext
    optimizer: str
    batch_size: int
    block_size: int
    device: str
    target_batch_size: int

    encoder_type: Optional[str] = None
    tokenizer_type: Optional[str] = None
    no_docs: bool = False

    # Shortcut modes
    list_runs: bool = False
    train_tokenizer: bool = False
    tokenizer_train_type: str = "unigram"
    tokenizer_num_examples: int = 5_000_000
    tokenizer_train_vocab_size: int = 16384

    # Optimizer wrappers / schedule. optimizer_wrappers is an ordered list of
    # WRAPPER_REGISTRY keys (e.g. ["schedule_free"]); the old --trac/--ortho/
    # --lookahead/--schedule-free flags collapsed into it.
    fixed_schedule: bool = False
    optimizer_wrappers: List[str] = field(default_factory=list)
    disable_schedule: bool = False
    # Global gradient-norm clip threshold (norm mode). Applied by trainers that
    # support clipping; ignored by the ones that don't (mono_forward/pipeline).
    gradient_clip_val: float = 10.0

    # Training loop
    max_steps: Optional[int] = None
    val_every: int = 1024
    use_dashboard: bool = False
    headless: bool = False
    quiet: bool = False
    reset: bool = False
    reset_after: int = 0
    preserve: bool = False
    no_checkpoints: bool = False
    save_every: int = 256
    no_compile: bool = False
    dropout: float = 0.0
    strategy: Optional[str] = None
    trainer_type: str = "backpropagation"
    pipeline_depth: int = 4
    byte_level: bool = False

    # Data
    train_datasets: List[str] = field(default_factory=lambda: ["base"])
    validation_datasets: List[str] = field(default_factory=lambda: ["validation"])
    sampler_mode: str = "novelty"
    seq_curriculum: str = "fixed"
    data_path: List[str] = field(default_factory=list)
    rl_type: Optional[Any] = None  # name, comma-string, or list of RL tasks

    # Evaluation / inference display
    eval_every: Optional[int] = None
    eval_tasks: Optional[Any] = None
    infer_every: int = 3
    terminal_output_length: int = 0
    debug: bool = False

    # Memory profiling (diagnostic; off by default)
    profile_memory: bool = False
    profile_memory_start: int = 0
    profile_memory_steps: int = 50
    profile_memory_max_entries: int = 5_000_000

    # Web / API server
    host_name: str = "localhost"
    port: int = 2100

    # Distributed
    local_rank: int = 0
    num_nodes: int = 1
    node_rank: int = 0
    master_addr: str = "localhost"
    master_port: int = 29500

    # Ray Mono-Forward knobs (ignored by non-mono_forward trainers)
    ray_address: Optional[str] = None
    ray_num_replicas_per_layer: int = 1
    ray_head_sync_every: int = 50
    ray_pipeline_api: str = "manual"

    # Escape hatches for the rare consumer that needs the raw inputs
    args: Any = None
    raw: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_args(cls, processed_args: Dict[str, Any], args: Any = None) -> "RunConfig":
        """Build a RunConfig from the dict returned by ``get_processed_args``."""
        get = processed_args.get
        batch_size = processed_args["batch_size"]
        block_size = processed_args["block_size"]
        encoder_type = get("encoder_type")

        # The flag drives byte-token-specific behavior (bits-per-byte metric
        # and friends). Pure tokenizer concern; byte-latent encoders force
        # tokenizer_type='byte_level' upstream so this catches them too.
        tokenizer_type = get("tokenizer_type")
        byte_level = get("byte_level", False) or (tokenizer_type == "byte_level")

        return cls(
            seed=processed_args["seed"],
            vocab_size=processed_args["vocab_size"],
            cache_dir=processed_args["cache_dir"],
            optimizer=processed_args["optimizer"],
            batch_size=batch_size,
            block_size=block_size,
            device=processed_args["device"],
            target_batch_size=get("target_batch_size", batch_size),
            encoder_type=encoder_type,
            tokenizer_type=get("tokenizer_type"),
            no_docs=get("no_docs", False),
            list_runs=get("list_runs", False),
            train_tokenizer=get("train_tokenizer", False),
            tokenizer_train_type=get("tokenizer_train_type", "unigram"),
            tokenizer_num_examples=get("tokenizer_num_examples", 5_000_000),
            tokenizer_train_vocab_size=get("tokenizer_train_vocab_size", 16384),
            fixed_schedule=get("fixed_schedule", False),
            optimizer_wrappers=_resolve_optimizer_wrappers(get),
            disable_schedule=get("disable_schedule", False),
            gradient_clip_val=float(get("gradient_clip_val", 10.0)),
            max_steps=get("max_steps"),
            val_every=get("val_every", 1024),
            use_dashboard=get("use_dashboard", False),
            headless=get("headless", False),
            quiet=get("quiet", False),
            reset=get("reset", False),
            reset_after=int(get("reset_after", 0) or 0),
            preserve=get("preserve", False),
            no_checkpoints=get("no_checkpoints", False),
            save_every=int(get("save_every", 256)),
            no_compile=get("no_compile", False),
            dropout=get("dropout", 0.0),
            strategy=get("strategy"),
            trainer_type=get("trainer_type", "backpropagation"),
            pipeline_depth=get("pipeline_depth", 4),
            byte_level=byte_level,
            train_datasets=coerce_to_list(get("train_datasets") or ["base"]),
            validation_datasets=coerce_to_list(
                get("validation_datasets") or ["validation"]
            ),
            sampler_mode=get("sampler_mode", "novelty"),
            seq_curriculum=get("seq_curriculum", "fixed"),
            data_path=coerce_to_list(get("data_path")),
            rl_type=get("rl_type"),
            eval_every=get("eval_every"),
            eval_tasks=get("eval_tasks"),
            infer_every=get("infer_every", 3),
            terminal_output_length=get("terminal_output_length", block_size * 2),
            debug=get("debug", False),
            profile_memory=get("profile_memory", False),
            profile_memory_start=get("profile_memory_start", 0),
            profile_memory_steps=get("profile_memory_steps", 50),
            profile_memory_max_entries=get("profile_memory_max_entries", 5_000_000),
            host_name=get("host_name", "localhost"),
            port=get("port", 2100),
            local_rank=get("local_rank", 0),
            num_nodes=get("num_nodes", 1),
            node_rank=get("node_rank", 0),
            master_addr=get("master_addr", "localhost"),
            master_port=get("master_port", 29500),
            ray_address=get("ray_address"),
            ray_num_replicas_per_layer=get("ray_num_replicas_per_layer", 1),
            ray_head_sync_every=get("ray_head_sync_every", 50),
            ray_pipeline_api=get("ray_pipeline_api", "manual"),
            args=args,
            raw=processed_args,
        )

    def apply_distributed_env(self) -> None:
        """Export distributed env vars before Lightning builds the process
        group. Only touched for multi-node runs so single-node is untouched.
        """
        if self.num_nodes > 1:
            os.environ["MASTER_ADDR"] = self.master_addr
            os.environ["MASTER_PORT"] = str(self.master_port)
            os.environ["NODE_RANK"] = str(self.node_rank)
