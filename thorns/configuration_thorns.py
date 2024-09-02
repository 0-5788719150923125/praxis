from transformers import PretrainedConfig


class ThornsConfig(PretrainedConfig):
    model_type = "thorns"

    def __init__(
        self,
        vocab_size=32000,
        n_positions=1024,
        n_embd=768,
        n_layer=12,
        n_head=12,
        activation_function="gelu",
        attn_pdrop=0.1,
        resid_pdrop=0.1,
        embd_pdrop=0.1,
        rms_norm_epsilon=1e-6,
        initializer_range=0.02,
        use_cache=False,
        causal=False,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        **kwargs
    ):
        super().__init__(
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            **kwargs
        )

        self.vocab_size = vocab_size
        self.n_positions = n_positions
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.activation_function = activation_function
        self.attn_pdrop = attn_pdrop
        self.resid_pdrop = resid_pdrop
        self.embd_pdrop = embd_pdrop
        self.rms_norm_epsilon = rms_norm_epsilon
        self.initializer_range = initializer_range
        self.use_cache = use_cache
        self.causal = causal
