from transformers import PretrainedConfig


class PraxisConfig(PretrainedConfig):
    model_type = "praxis"

    def __init__(
        self,
        n_dim=768,
        n_layer=12,
        n_head=12,
        activation_function="mish",
        rms_norm_epsilon=1e-6,
        initializer_range=0.02,
        capacity=0.5,
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
        self.activation_function = activation_function
        self.rms_norm_epsilon = rms_norm_epsilon
        self.initializer_range = initializer_range
        self.capacity = capacity
        self.use_cache = False
        self.causal = False
