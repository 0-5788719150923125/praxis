from transformers import PretrainedConfig


class PraxisConfig(PretrainedConfig):
    model_type = "praxis"

    def __init__(
        self,
        embed_size=192,
        hidden_size=256,
        num_heads=4,
        num_queries=2,
        head_size=None,
        k_heads=None,
        kv_rank=None,
        depth=7,
        num_experts=7,
        dropout=0,
        epsilon=1e-5,
        capacity=0.125,
        vocab_size=8192,
        context_length=4096,
        activation="mish",
        block="transformer",
        expert="glu",
        encoding="rope",
        mod=None,
        shuffle=False,
        autopilot=False,
        graph=False,
        router=False,
        attention_type="standard",
        linear=False,
        differential=False,
        stickbreaking=False,
        compression=False,
        memory=False,
        mta=False,
        mega=False,
        gated=False,
        evolve=False,
        byte_latent=False,
        hyper=False,
        scaling=False,
        hivemind=False,
        initial_peers=[],
        checkpoint_every=0,
        loss_func="cross_entropy",
        device_map="cpu",
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        seed=42,
        debug=False,
        meta=[],
        **kwargs,
    ):
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
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
        self.linear = linear
        self.differential = differential
        self.stickbreaking = stickbreaking
        self.dropout = dropout
        self.epsilon = epsilon
        self.capacity = capacity
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.activation = activation
        self.block_type = block
        self.expert = expert
        self.encoding = encoding
        self.mod = mod
        self.shuffle = shuffle
        self.autopilot = autopilot
        self.graph = graph
        self.router = router
        self.compression = compression
        self.memory = memory
        self.mta = mta
        self.mega = mega
        self.gated = gated
        self.evolve = evolve
        self.byte_latent = byte_latent
        self.hyper = hyper
        self.scaling = scaling
        self.hivemind = hivemind
        self.initial_peers = initial_peers
        self.checkpoint_every = checkpoint_every
        self.loss_func = loss_func
        self.device_map = device_map
        self.seed = seed
        self.debug = debug
        self.meta = meta
        self.causal = False
