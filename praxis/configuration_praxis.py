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
        depth: int = 9,
        num_experts: int = 9,
        dropout: float = 0.0,
        epsilon: float = 1e-5,
        vocab_size: int = 8192,
        max_length: int = 4096,
        activation: str = "mish",
        block: str = "transformer",
        expert: str = "glu",
        encoding: str = "rope",
        router_type: Optional[str] = None,
        controller_type: str = "base",
        attention_type: str = "standard",
        encoder_type: Optional[str] = None,
        decoder_type: str = "sequential",
        residual_type: str = "standard",
        compression_type: str = "none",
        sorting_type: str = "none",
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
        initial_peers: List[str] = None,
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
        meta: List[str] = None,
        **kwargs,
    ):
        if initial_peers is None:
            initial_peers = []
        if meta is None:
            meta = []

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            sep_token_id=sep_token_id,
            **kwargs,
        )

        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_queries = num_queries
        self.head_size = head_size
        self.k_heads = k_heads
        self.kv_rank = kv_rank
        self.depth = depth
        self.num_experts = num_experts
        self.attention_type = attention_type
        self.encoder_type = encoder_type
        self.decoder_type = decoder_type
        self.linear = linear
        self.differential = differential
        self.stickbreaking = stickbreaking
        self.dropout = dropout
        self.epsilon = epsilon
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.activation = activation
        self.block_type = block
        self.expert = expert
        self.encoding = encoding
        self.router_type = router_type
        self.controller_type = controller_type
        self.residual_type = residual_type
        self.compression_type = compression_type
        self.sorting_type = sorting_type
        self.memory = memory
        self.mla = mla
        self.mta = mta
        self.mega = mega
        self.gated = gated
        self.evolve = evolve
        self.scaled = scaled
        self.hivemind = hivemind
        self.initial_peers = initial_peers
        self.checkpoint_every = checkpoint_every
        self.loss_func = loss_func
        self.strategy = strategy
        self.device_map = device_map
        self.seed = seed
        self.debug = debug
        self.meta = meta
        self.causal = False
