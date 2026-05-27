from typing import List, Optional, Union

from transformers import PretrainedConfig


class PraxisConfig(PretrainedConfig):
    model_type = "praxis"

    def __init__(
        self,
        embed_size: int = 192,
        hidden_size: int = 256,
        num_heads: int = 4,
        num_queries: int = 2,
        head_size: Optional[int] = None,
        k_heads: Optional[int] = None,
        kv_rank: Optional[int] = None,
        depth: Optional[int] = None,
        num_experts: int = 1,
        num_layers: int = 2,
        dropout: float = 0.0,
        epsilon: float = 1e-5,
        vocab_size: int = 8192,
        max_position_embeddings: int = 32768,
        activation: str = "mish",
        block_type: str = "transformer",
        expert: str = "glu",
        encoding: str = "rope",
        router_type: Optional[str] = None,
        controller_type: str = "base",
        attention_type: str = "modular",
        memory_type: str = "none",
        encoder_type: Optional[str] = None,
        decoder_type: str = "sequential",
        residual_type: str = "standard",
        compression_type: str = "none",
        sorting_type: str = "none",
        norm_type: str = "rms_norm",
        head_type: str = "forward",
        halting_type: Optional[str] = None,
        mtp_type: Optional[str] = None,
        mtp_depth: int = 1,
        rl_type: Optional[str] = None,  # "reinforce", "grpo", "ppo", or None
        rl_weight: float = 0.1,
        grpo_group_size: int = 8,
        forward_weight: float = 0.666666,
        task_weights: Optional[dict] = None,
        no_mask_prompts: bool = False,
        linear: bool = False,
        differential: bool = False,
        stickbreaking: bool = False,
        memory: bool = False,
        mta: bool = False,
        mega: bool = False,
        gated: bool = False,
        evolve: bool = False,
        scaled: bool = False,
        mla: bool = False,
        hivemind: bool = False,
        tie_weights: bool = False,
        contrastive_isotropy: bool = True,
        window_size: Optional[int] = None,
        bidirectional: bool = False,
        initial_peers: List[str] = [],
        checkpoint_every: int = 0,
        loss_func: str = "cross_entropy",
        strategy: str = "naive",
        device_map: Union[str, dict] = "cpu",
        pad_token_id: int = 0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        sep_token_id: int = 3,
        seed: int = 42,
        debug: bool = False,
        meta: List[str] = [],
        **kwargs,
    ):
        # Snapshot the declared arguments so each can be assigned automatically.
        declared = dict(locals())

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            sep_token_id=sep_token_id,
            tie_word_embeddings=tie_weights,  # HF's name for tie_weights
            **kwargs,
        )

        # Every declared argument becomes a same-named attribute.
        skip = {"self", "kwargs", "__class__", "tie_weights"}
        for name, value in declared.items():
            if name not in skip:
                setattr(self, name, value)

        # Derived / constant attributes that aren't direct argument copies.
        self.depth = depth if depth is not None else num_layers
        self.num_hidden_layers = self.depth  # HF cache expects this name
        self.causal = False
