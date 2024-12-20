from typing import Optional, Union

import bytelatent
import torch
import torch.nn.functional as F
from bytelatent.base_transformer import flex_attention_comp, repeat_kv
from bytelatent.model.transformer import CrossAttention
from torch import nn


class SimpleCrossAttention(CrossAttention):
    """
    A monkey-patch for CrossAttention, which doesn't use FlexAttention.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_sdpa = True

    def forward(self, x, kv, mask=None) -> torch.Tensor:
        # B S D
        bsz, seq_len, _ = x.shape
        _, slen_kv, _ = kv.shape

        # without detaching x, there is unbounded memory growth; only required in the encoder
        # x = x.detach()

        x = self.cross_attn_norm_q(x)
        kv = self.cross_attn_norm_kv(kv)

        xq = self.wq(x)
        xk = self.wk(kv)
        xv = self.wv(kv)

        output_shape = xq.shape
        # B S D -> B S H D
        xq = xq.view(bsz, seq_len, self.n_heads, self.head_dim)
        xk = xk.view(bsz, slen_kv, self.n_kv_heads, self.head_dim)
        xv = xv.view(bsz, slen_kv, self.n_kv_heads, self.head_dim)

        xk = repeat_kv(xk, self.heads_per_group, dim=2)
        xv = repeat_kv(xv, self.heads_per_group, dim=2)

        xq, xk, xv = map(lambda e: e.transpose(1, 2), (xq, xk, xv))
        if self.use_sdpa:
            # Convert BlockMask to attention mask if needed
            if not torch.is_tensor(mask):
                attn_mask = (
                    mask.materialize((seq_len, slen_kv)) if mask is not None else None
                )
            else:
                attn_mask = mask

            # Apply scaled dot-product attention
            output = F.scaled_dot_product_attention(
                xq, xk, xv, attn_mask=attn_mask, dropout_p=0.0, is_causal=False
            )
        else:
            assert mask is None or isinstance(mask, BlockMask)
            # Use original flex attention
            output = flex_attention_comp(xq, xk, xv, block_mask=mask)
        output = output.transpose(1, 2).contiguous()  # B H S D -> B S H D

        output = self.wo(output.reshape(output_shape))

        return x + output


# Monkey patch the import
bytelatent.model.transformer.CrossAttention = SimpleCrossAttention

from bytelatent.data.patcher import Patcher, PatcherArgs
from bytelatent.model.blt import (
    ByteLatentTransformerArgs,
    EmbeddingType,
    compute_hash_embeddings,
    create_local_decoder,
    create_local_encoder,
    cross_attn_mask,
    decoder_patch_ids_from_lengths,
    get_blt_input,
    get_global_dim_patch_emb,
    init_embeddings,
    patch_ids_from_lengths,
)
from bytelatent.model.utils import downsample


class PraxisByteLatentEncoder(nn.Module):
    """
    An implementation of the Byte Latent Encoder/Decoder, from:
    https://arxiv.org/abs/2412.09871
    """

    def __init__(self, config: "AutoConfig"):
        super().__init__()
        self.args = create_args(config)
        self.patcher = Patcher(
            PatcherArgs(
                patch_size=self.args.patch_size,
                patching_mode=self.args.patching_mode,
                threshold=self.args.patching_threshold,
                threshold_add=self.args.patching_threshold_add,
                monotonicity=self.args.monotonicity,
                max_patch_length=self.args.max_patch_length,
            )
        )
        self.embeds = init_embeddings(
            self.args,
            EmbeddingType.HASH_TOK,
            local_encoder_dim=self.args.dim_local_encoder,
            encoder_hash_byte_group_size=self.args.encoder_hash_byte_group_size,
        )
        self.encoder = create_local_encoder(self.args)
        self.decoder = create_local_decoder(self.args)

    def __repr__(self):
        return f"PraxisByteLatentEncoder(in_features={self.args.vocab_size}, out_features={self.args.dim_global})"

    def encode(self, input_ids):
        encoder_tokens, _, decoder_tokens = get_blt_input(
            tokens=input_ids,
            enforce_patch_size_multiple=False,
            nb_boe=0,
            patch_size=self.encoder.patch_size,
            boe_id=self.encoder.boe_id,
        )

        # Patching
        patch_lengths, tok_scores = self.patcher.patch(
            encoder_tokens,
            include_next_token=True,
            threshold=self.patcher.threshold,
        )
        patch_ids = patch_ids_from_lengths(patch_lengths, encoder_tokens.shape[-1])

        # Create cross attention mask if needed
        cross_attn_mask_enc = None
        if self.args.cross_attn_encoder:
            cross_attn_mask_enc = cross_attn_mask(
                patch_ids,
                patch_lengths,
                encoder_tokens.shape[-1],  # N
                patches_as_queries=True,
                cross_attn_k=self.args.cross_attn_k,
                window=self.args.cross_attn_window_encoder,
                block_mask=False,
            )

        # Compute embeddings
        embeds = compute_hash_embeddings(
            local_encoder_tokens=encoder_tokens,
            local_encoder=self.encoder,
            encoder_hash_tok_embedding=self.embeds,
            encoder_hash_byte_group_nb_functions=self.args.encoder_hash_byte_group_nb_functions,
            encoder_hash_byte_group_size=self.args.encoder_hash_byte_group_size,
            encoder_hash_byte_group_vocab=self.args.encoder_hash_byte_group_vocab,
        )

        # Local encoder with cross attention
        h_cross = None
        (h_encoder, h_cross), _ = self.encoder(
            tokens=encoder_tokens,
            embeds=embeds,
            patch_embeds=h_cross if self.args.cross_attn_encoder else None,
            cross_mask=cross_attn_mask_enc,
            num_patches=patch_lengths.shape[1],
            patch_ids=patch_ids,
        )

        # Downsampling
        if not self.args.cross_attn_encoder:
            h = downsample(
                h_encoder,
                patch_lengths.shape[1],
                patch_lengths,
                patch_ids,
                downsampling_by_pooling=self.args.downsampling_by_pooling,
                patch_size=self.args.patch_size,
            )
        else:
            h = h_cross.view(h_encoder.shape[0], -1, h_encoder.shape[-1])

        return h, h_encoder, decoder_tokens, patch_lengths

    def decode(self, encoder_tokens, h_encoder, decoder_tokens, patch_lengths):
        nb_boe = 0 if self.args.patching_mode != "" else self.args.patch_size - 1
        N = decoder_tokens.shape[-1]

        dec_embeds = h_encoder[:, nb_boe : nb_boe + N, :]
        decoder_patch_ids = decoder_patch_ids_from_lengths(
            patch_lengths, nb_boe, decoder_tokens.shape[-1]
        )

        # Upsampling
        if self.args.cross_attn_decoder:
            h = (
                encoder_tokens
                if self.args.cross_attn_encoder
                else encoder_tokens.repeat_interleave(self.args.cross_attn_k, dim=1)
            )
            cross_mask = cross_attn_mask(
                decoder_patch_ids,
                patch_lengths,
                decoder_tokens.shape[-1],
                patches_as_queries=False,
                cross_attn_k=self.args.cross_attn_k,
                window=self.args.cross_attn_window_decoder,
                block_mask=False,
            )
        else:
            h = torch.gather(
                encoder_tokens,
                1,
                decoder_patch_ids.unsqueeze(-1).expand(
                    -1, -1, encoder_tokens.shape[-1]
                ),
            )
            cross_mask = None
            assert (
                h.shape[1] == decoder_tokens.shape[1]
            ), "Sequence length mismatch after gathering"

        output, _ = self.decoder(
            tokens=decoder_tokens,
            patch_embeds=h,
            embeds=dec_embeds,
            cross_mask=cross_mask,
        )

        return output


def create_args(config):
    """
    Defaults from the original Facebook code.
    https://github.com/facebookresearch/blt/blob/main/bytelatent/test_blt.py
    """

    hidden_size = config.hidden_size
    # embed_size = config.embed_size

    return ByteLatentTransformerArgs(
        vocab_size=260,
        norm_eps=config.epsilon,
        n_heads=1,
        # dim=hidden_size,
        # dim_token_emb=hidden_size,
        dim_token=hidden_size,  # must be set, else creates an unused module called self.token_embedding_projection
        # dim_patch_emb=hidden_size,
        dim_global=hidden_size,
        dim_local_encoder=hidden_size,
        dim_local_decoder=hidden_size,
        # tie_local_encoder_decoder_logits=False,
        data_loader_patching=False,
        # max_seqlen=config.context_length,
        max_encoder_seq_length=config.context_length,
        # max_length=hidden_size,
        # pad_to_max_length=True,
        # encoder_lm_loss=False,
        patching_threshold=3.1439168453216553,  # use this for "space" patch_mode
        # patching_threshold=1.335442066192627, # use this for "entropy" patch_mode
        patch_size=6,  # use this for "space" patch_mode
        # patch_size=4.5, # use this for "entropy" patch_mode
        tokenization_mode="bytes",
        patching_mode="space",  # space patching [is] a very close competitor to dynamic entropy based patching.
        # patching_mode="bpe",
        # patching_mode="entropy",
        encoder_hash_byte_group_size=[4],
        encoder_hash_byte_group_vocab=config.vocab_size,
        encoder_hash_byte_group_nb_functions=3,
        cross_attn_encoder=False,  # the authors found that using cross-attention in the decoder is most effective.
        cross_attn_decoder=True,
        cross_attn_window_encoder=512,
        cross_attn_window_decoder=512,
        cross_attn_k=4,
        cross_attn_nheads=2,
        n_layers_local_encoder=1,
        n_layers_local_decoder=1,
        n_heads_local_encoder=1,
        n_heads_local_decoder=1,
        cross_attn_all_layers_encoder=True,
        cross_attn_all_layers_decoder=True,
        cross_attn_use_flex_attention=False,  # not supported on CPU and older GPUs
        cross_attn_init_by_pooling=True,
        log_patch_lengths=True,
        # non_linearity="swiglu", # not implemented
        use_rope=True,
        # recompute_fc1_out=False, # I don't think these do anything
        # recompute_fc3_out=False,
        # recompute_attn=False,
        # custom_bwd=False,
        # layer_ckpt="none",
        efficient_attn="sdpa",
        # patch_only_encoder=False, # doesn't do anything
        # patch_only_decoder=False,
        # use_local_encoder_transformer=True,
        # init_use_gaussian=True,
        # init_use_depth="current",
        # attn_bias_type="local_block_causal",
        # alpha_depth="disabled",
        local_attention_window_len=512,
        sliding_window=256,  # basically required, else encoder dim is equal to max_seq_len
        downsampling_by_pooling="max",
        share_encoder_decoder_emb=False,
    )


if __name__ == "__main__":
    import torch
    from transformers import AutoConfig

    # Initialize test configuration
    class Dummy:
        vocab_size = 260
        hidden_size = 360
        # embed_size = 512
        context_length = 2048
        epsilon = 1e-5

    config = Dummy()

    # Initialize model
    model = PraxisByteLatentEncoder(config)

    def run_test():
        try:
            # Create sample input
            batch_size = 2
            seq_len = 128  # Should be less than max_seq_len (512)
            input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
            print(f"Input shape: {input_ids.shape}")

            # Step 1: Encode
            h, h_encoder, decoder_tokens, patch_lengths = model.encode(
                input_ids=input_ids
            )
            print(f"Encoder output (h) shape: {h.shape}")

            # Step 2: Decode
            decoder_output = model.decode(
                encoder_tokens=h,
                h_encoder=h_encoder,
                decoder_tokens=decoder_tokens,
                patch_lengths=patch_lengths,
            )
            print(f"Decoder output shape: {decoder_output.shape}")

            # Basic shape assertions
            assert len(decoder_output.shape) == 3, "Expected 3D output from decoder"
            assert (
                decoder_output.shape[0] == batch_size
            ), "Batch size mismatch in output"

            print("\nTest completed successfully!")
            return True

        except Exception as e:
            print(f"\nTest failed with error: {str(e)}")
            print(f"Error type: {type(e)}")
            import traceback

            print(traceback.format_exc())
            return False

    # Run the test
    print("Starting ByteLatent test...")
    run_test()
