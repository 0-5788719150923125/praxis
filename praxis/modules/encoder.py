from bytelatent.model.blt import (
    ByteLatentTransformerArgs,
    create_local_decoder,
    create_local_encoder,
)


def ByteLatentEncoder(config: "AutoConfig"):
    return create_local_encoder(
        ByteLatentTransformerArgs(
            dim=config.hidden_size,
            vocab_size=config.vocab_size,
            n_layers_local_encoder=1,
            n_heads_local_encoder=1,
            # dim_token_emb: int = config.vocab_size
            # dim_patch_emb: int = config.vocab_size // 2
            cross_attn_decoder=False,
            attn_bias_type="local_block_causal",
            max_encoder_seq_length=config.context_length,
            efficient_attn="sdpa",
            # sliding_window: int = 64
        )
    )
