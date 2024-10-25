from transformers import PretrainedConfig

from praxis.modules import DEFAULT_EXPERT_CONFIGS


class PraxisConfig(PretrainedConfig):
    model_type = "praxis"

    def __init__(
        self,
        num_embeds=720,
        num_dims=360,
        num_heads=8,
        depth=7,
        dropout=0,
        epsilon=1e-5,
        capacity=0.125,
        vocab_size=4096,
        context_length=4096,
        activation="serf",
        expert="glu",
        sparse=False,
        shuffle=False,
        autopilot=False,
        differential=False,
        compression=False,
        hivemind=False,
        initial_peers=None,
        memory_profile="speed",
        device_map="cpu",
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        unk_token_id=4,
        **kwargs,
    ):
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            unk_token_id=unk_token_id,
            **kwargs,
        )

        if autopilot:
            assert (
                autopilot == shuffle
            ), "To use `autopilot`, you must also use `shuffle`."

        # Praxis args
        self.num_embeds = num_embeds
        self.num_dims = num_dims
        self.num_heads = num_heads
        self.depth = depth
        self.differential = differential
        self.dropout = dropout
        self.epsilon = epsilon
        self.capacity = capacity
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.activation = activation
        self.expert = self._register_experts(expert)
        self.sparse = sparse
        self.shuffle = shuffle
        self.autopilot = autopilot
        self.compression = compression
        self.hivemind = hivemind
        self.initial_peers = initial_peers
        self.memory_profile = memory_profile
        self.device_map = device_map
        self.causal = False

    def _register_experts(self, expert: str or dict):
        # Handle expert configuration
        if isinstance(expert, str):
            if expert not in DEFAULT_EXPERT_CONFIGS:
                raise ValueError(f"Unknown expert type: {expert}")
            return {"type": expert, **DEFAULT_EXPERT_CONFIGS[expert]}
        elif isinstance(expert, dict):
            return expert
        else:
            raise ValueError("Expert must be either a string or a dictionary")
