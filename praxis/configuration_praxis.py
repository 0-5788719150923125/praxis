from transformers import PretrainedConfig


class PraxisConfig(PretrainedConfig):
    model_type = "praxis"

    def __init__(
        self,
        n_dim=384,
        n_emb=512,
        n_layer=12,
        n_head=8,
        activation="mish",
        epsilon=1e-6,
        capacity=0.125,
        vocab_size=8192,
        context_length=1024,
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
        self.context_length = context_length
        self.n_dim = n_dim
        self.n_emb = n_emb
        self.n_layer = n_layer
        self.n_head = n_head
        self.activation = activation
        self.capacity = capacity
        self.epsilon = epsilon
        self.causal = False
