from transformers import PretrainedConfig


class PraxisConfig(PretrainedConfig):
    model_type = "praxis"

    def __init__(
        self,
        num_embeds=1024,
        num_dims=256,
        num_layers=7,
        num_heads=8,
        differential_heads=0,
        dropout=0,
        epsilon=1e-5,
        capacity=0.125,
        vocab_size=4096,
        context_length=4096,
        activation="mish",
        expert_type="peer",
        expert=dict(
            num_experts=40**2,
            num_heads=2,
            k=8,
            key_dims=128,
            offset_heads=True,
        ),
        sparse=False,
        shuffle=False,
        reclaim_memory="speed",
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

        # Praxis args
        self.num_embeds = num_embeds
        self.num_dims = num_dims
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.differential_heads = differential_heads
        self.dropout = dropout
        self.epsilon = epsilon
        self.capacity = capacity
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.activation = activation
        self.expert_type = expert_type
        self.expert = expert
        self.sparse = sparse
        self.shuffle = shuffle
        self.reclaim_memory = reclaim_memory
        self.causal = False

        # Huggingface args
        self.is_decoder = True
        self.is_encoder_decoder = False
        self.tie_word_embeddings = False
