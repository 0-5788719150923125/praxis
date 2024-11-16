from transformers import PretrainedConfig

from praxis.modules import EXPERT_CONFIGS


class PraxisConfig(PretrainedConfig):
    model_type = "praxis"

    def __init__(
        self,
        num_embeds=720,
        num_dims=360,
        num_heads=4,
        depth=7,
        num_experts=7,
        dropout=0,
        epsilon=1e-5,
        capacity=0.125,
        vocab_size=4096,
        context_length=2048,
        activation="serf",
        block="transformer",
        expert="glu",
        encoding="rope",
        sparse=False,
        shuffle=False,
        autopilot=False,
        differential=False,
        stickbreaking=False,
        compression=False,
        memory=False,
        hivemind=False,
        initial_peers=[],
        memory_profile="speed",
        device_map="cpu",
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        unk_token_id=4,
        seed=42,
        debug=False,
        **kwargs,
    ):
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            unk_token_id=unk_token_id,
            **kwargs,
        )

        if sparse:
            assert (
                depth % 2 != 0
            ), "When using `sparse=True`, `depth` should be an odd number."

        if autopilot:
            assert (
                autopilot == shuffle
            ), "To use `autopilot`, you must also use `shuffle`."

        assert (
            num_experts >= depth
        ), "`num_experts` should be at least as large as `depth`."

        if not shuffle:
            assert (
                num_experts == depth
            ), "There is no point in making `num_experts` greater than or less than `depth`, when `shuffle != True`. The additional experts would never be used."

        # Praxis args
        self.num_embeds = num_embeds
        self.num_dims = num_dims
        self.num_heads = num_heads
        self.depth = depth
        self.num_experts = num_experts
        self.differential = differential
        self.stickbreaking = stickbreaking
        self.dropout = dropout
        self.epsilon = epsilon
        self.capacity = capacity
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.activation = activation
        self.block_type = block
        self.expert = self._register_experts(expert)
        self.encoding = encoding
        self.sparse = sparse
        self.shuffle = shuffle
        self.autopilot = autopilot
        self.compression = compression
        self.memory = memory
        self.hivemind = hivemind
        self.initial_peers = initial_peers
        self.memory_profile = memory_profile
        self.device_map = device_map
        self.seed = seed
        self.debug = debug
        self.causal = False

    def _register_experts(self, expert: str or dict):
        # Handle expert configuration
        if isinstance(expert, str):
            if expert not in EXPERT_CONFIGS:
                raise ValueError(f"Unknown expert type: {expert}")
            return {"type": expert, **EXPERT_CONFIGS[expert]}
        elif isinstance(expert, dict):
            return expert
        else:
            raise ValueError("Expert must be either a string or a dictionary")
