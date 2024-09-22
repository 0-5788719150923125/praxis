from transformers import PretrainedConfig


class PraxisConfig(PretrainedConfig):
    model_type = "praxis"

    def __init__(
        self,
        n_dim=768,
        n_layer=12,
        n_head=12,
        activation="mish",
        epsilon=1e-6,
        capacity=0.125,
        vocab_size=32000,
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

        self.vocab_size = vocab_size
        self.n_dim = n_dim
        self.n_layer = n_layer
        self.n_head = n_head
        self.activation = activation
        self.epsilon = epsilon
        self.capacity = capacity
        self.use_cache = False
        self.causal = False
