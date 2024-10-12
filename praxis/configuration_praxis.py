from transformers import PretrainedConfig


class PraxisConfig(PretrainedConfig):
    model_type = "praxis"

    def __init__(
        self,
        n_emb=512,
        n_dim=384,
        n_layer=7,
        n_head=8,
        differential_heads=0,
        dropout=0,
        epsilon=1e-5,
        capacity=0.125,
        vocab_size=4096,
        context_length=4096,
        activation="mish",
        peer_heads=4,
        peer_experts=16**2,
        peer_experts_per_head=8,
        sparse=False,
        shuffle=False,
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
        self.n_emb = n_emb
        self.n_dim = n_dim
        self.n_layer = n_layer
        self.n_head = n_head
        self.differential_heads = differential_heads
        self.dropout = dropout
        self.epsilon = epsilon
        self.capacity = capacity
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.activation = activation
        self.peer_heads = peer_heads
        self.peer_experts = peer_experts
        self.peer_experts_per_head = peer_experts_per_head
        self.sparse = sparse
        self.shuffle = shuffle
        self.causal = False

        # Huggingface args
        self.is_decoder = True
        self.is_encoder_decoder = False
        self.tie_word_embeddings = False
