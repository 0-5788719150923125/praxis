from bytelatent.model.blt import (
    ByteLatentTransformerArgs,
    EmbeddingType,
    compute_hash_embeddings,
    create_local_decoder,
    create_local_encoder,
    get_blt_input,
    init_embeddings,
)
from torch import nn


class PraxisByteLatentEncoder(nn.Module):
    def __init__(self, config: "AutoConfig"):
        super().__init__()

        max_seq_len = 512
        self.args = create_args(cross_attention=False)
        self.args.encoder_hash_byte_group_vocab = config.vocab_size
        self.args.dim = config.hidden_size
        self.args.dim_token = config.hidden_size
        self.args.dim_local_encoder = max_seq_len
        self.args.dim_local_decoder = config.hidden_size
        # self.args.dim_token_emb = config.hidden_size
        # self.args.dim_patch_emb = config.hidden_size
        self.args.vocab_size = config.vocab_size
        self.args.n_layers_local_encoder = 1
        self.args.n_layers_local_decoder = 1
        self.args.n_heads_local_encoder = 1
        self.args.n_heads_local_decoder = 1
        self.args.cross_attn_decoder = False
        self.args.attn_bias_type = "local_block_causal"
        self.args.max_encoder_seq_length = max_seq_len
        self.args.efficient_attn = "sdpa"
        self.args.sliding_window = (
            config.hidden_size  # basically required, else encoder dim is equal to max_seq_len
        )
        self.args.share_encoder_decoder_emb = False
        self.args.max_length = config.hidden_size
        self.args.max_seqlen = max_seq_len

        self.encoder = create_local_encoder(self.args)
        self.embeds = init_embeddings(
            self.args,
            EmbeddingType.HASH_TOK,
            local_encoder_dim=self.args.dim_local_encoder,
            encoder_hash_byte_group_size=[4],
        )
        self.decoder = create_local_decoder(self.args)

    def create_tokens(self, input_ids):
        local_encoder_tokens, _, _ = get_blt_input(
            tokens=input_ids,
            enforce_patch_size_multiple=False,
            nb_boe=0,
            patch_size=self.encoder.patch_size,
            boe_id=self.encoder.boe_id,
        )
        return local_encoder_tokens

    def compute_embeds(self, tokens):
        return compute_hash_embeddings(
            local_encoder_tokens=tokens,
            local_encoder=self.encoder,
            encoder_hash_tok_embedding=self.embeds,
            encoder_hash_byte_group_nb_functions=self.args.encoder_hash_byte_group_nb_functions,
            encoder_hash_byte_group_size=self.args.encoder_hash_byte_group_size,
            encoder_hash_byte_group_vocab=self.args.encoder_hash_byte_group_vocab,
        )

    def encode(self, tokens, embeds):
        return self.encoder(tokens, embeds)

    def decode(self, tokens, embeds):
        return self.encoder(tokens, embeds)


def create_args(cross_attention=False):
    transformer_args = ByteLatentTransformerArgs(
        # Base args provided
        n_heads=8,
        dim=512,
        vocab_size=260,
        # Additional args from command line
        dim_token=256,
        patch_size=6,
        tokenization_mode="bytes",
        patching_mode="space",
        tie_local_encoder_decoder_logits=False,
        data_loader_patching=True,
        max_encoder_seq_length=12288,
        pad_to_max_length=True,
        encoder_lm_loss=False,
        patching_threshold=3.1439168453216553,
        encoder_hash_byte_group_size=[4],
        encoder_hash_byte_group_vocab=50002,
        encoder_hash_byte_group_nb_functions=3,
        cross_attn_encoder=cross_attention,  # True,
        cross_attn_decoder=cross_attention,  # True,
        cross_attn_window_encoder=512,
        cross_attn_window_decoder=512,
        dim_local_encoder=256,
        dim_local_decoder=256,
        cross_attn_k=8,
        cross_attn_nheads=4,
        cross_attn_all_layers_decoder=True,
        cross_attn_all_layers_encoder=True,
        cross_attn_use_flex_attention=True,
        cross_attn_init_by_pooling=True,
        log_patch_lengths=True,
        non_linearity="swiglu",
        use_rope=True,
        recompute_fc1_out=False,
        recompute_fc3_out=False,
        recompute_attn=False,
        custom_bwd=False,
        layer_ckpt="none",
        efficient_attn="sdpa",
        patch_only_encoder=False,
        patch_only_decoder=False,
        use_local_encoder_transformer=True,
        init_use_gaussian=True,
        init_use_depth="current",
        attn_bias_type="block_causal",
        alpha_depth="disabled",
        max_length=256,
        local_attention_window_len=512,
        max_seqlen=12288,
        downsampling_by_pooling="max",
    )
    return transformer_args
