import torch
from bytelatent.data.patcher import Patcher, PatcherArgs
from bytelatent.model.blt import (
    ByteLatentTransformerArgs,
    EmbeddingType,
    compute_hash_embeddings,
    create_local_decoder,
    create_local_encoder,
    decoder_patch_ids_from_lengths,
    get_blt_input,
    get_global_dim_patch_emb,
    init_embeddings,
    patch_ids_from_lengths,
)
from torch import nn


class PraxisByteLatentEncoder(nn.Module):
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
            encoder_hash_byte_group_size=[4],
        )
        self.encoder = create_local_encoder(self.args)
        self.decoder = create_local_decoder(self.args)

    def __repr__(self):
        return f"PraxisByteLatentEncoder({self.args.vocab_size, self.args.dim_global})"

    def encode(self, input_ids):
        encoder_tokens, _, decoder_tokens = get_blt_input(
            tokens=input_ids,
            enforce_patch_size_multiple=False,
            nb_boe=0,
            patch_size=self.encoder.patch_size,
            boe_id=self.encoder.boe_id,
        )
        patch_lengths, tok_scores = self.patcher.patch(
            encoder_tokens,
            include_next_token=True,
            threshold=self.patcher.threshold,
        )
        patch_ids = patch_ids_from_lengths(patch_lengths, encoder_tokens.shape[-1])
        embeds = compute_hash_embeddings(
            local_encoder_tokens=encoder_tokens,
            local_encoder=self.encoder,
            encoder_hash_tok_embedding=self.embeds,
            encoder_hash_byte_group_nb_functions=self.args.encoder_hash_byte_group_nb_functions,
            encoder_hash_byte_group_size=self.args.encoder_hash_byte_group_size,
            encoder_hash_byte_group_vocab=self.args.encoder_hash_byte_group_vocab,
        )
        (encoder_output, _), _ = self.encoder(
            encoder_tokens,
            embeds,
            num_patches=patch_lengths.shape[1],
            patch_ids=patch_ids,
        )
        return encoder_output, decoder_tokens, embeds, patch_lengths

    def decode(self, encoder_tokens, decoder_tokens, embeds, patch_lengths):
        nb_boe = 0 if self.args.patching_mode != "" else self.args.patch_size - 1

        # Create decoder patch ids to align tokens
        decoder_patch_ids = decoder_patch_ids_from_lengths(
            patch_lengths, nb_boe, decoder_tokens.shape[-1]
        )

        # Gather encoded tokens according to patch ids
        gathered_tokens = torch.gather(
            encoder_tokens,
            1,
            decoder_patch_ids.unsqueeze(-1).expand(-1, -1, encoder_tokens.shape[-1]),
        )

        # Run through local decoder
        output, _ = self.decoder(
            tokens=decoder_tokens, patch_embeds=gathered_tokens, embeds=embeds
        )

        return output


def create_args(config):
    """
    Defaults from the original Facebook code.
    https://github.com/facebookresearch/blt/blob/main/bytelatent/test_blt.py
    """

    return ByteLatentTransformerArgs(
        # Base args provided
        n_heads=8,
        dim=config.hidden_size,
        vocab_size=config.vocab_size,
        # Additional args from command line
        dim_token=config.hidden_size,
        dim_patch_emb=config.hidden_size,
        dim_global=config.hidden_size,
        tie_local_encoder_decoder_logits=False,
        data_loader_patching=True,
        max_encoder_seq_length=config.context_length,
        pad_to_max_length=True,
        encoder_lm_loss=False,
        patching_threshold=3.1439168453216553,
        # patching_threshold=1.335442066192627,
        patch_size=6,
        # patch_size=4.5,
        tokenization_mode="bytes",
        patching_mode="space",
        # patching_mode="bpe",
        encoder_hash_byte_group_size=[4],
        encoder_hash_byte_group_vocab=config.vocab_size,
        encoder_hash_byte_group_nb_functions=3,
        cross_attn_encoder=False,
        cross_attn_decoder=False,
        cross_attn_window_encoder=512,
        cross_attn_window_decoder=512,
        dim_local_encoder=config.hidden_size,
        dim_local_decoder=config.hidden_size,
        cross_attn_k=1,
        cross_attn_nheads=1,
        n_layers_local_encoder=1,
        n_layers_local_decoder=1,
        n_heads_local_encoder=1,
        n_heads_local_decoder=1,
        cross_attn_all_layers_decoder=True,
        cross_attn_all_layers_encoder=True,
        cross_attn_use_flex_attention=False,  # not supported on CPU and many GPUs
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
        attn_bias_type="local_block_causal",
        alpha_depth="disabled",
        max_length=config.hidden_size,
        local_attention_window_len=512,
        sliding_window=config.hidden_size,  # basically required, else encoder dim is equal to max_seq_len
        max_seqlen=config.context_length,
        downsampling_by_pooling="max",
        share_encoder_decoder_emb=False,
    )


if __name__ == "__main__":
    import torch
    from transformers import AutoConfig

    # Initialize test configuration
    class Dummy:
        vocab_size = 50002
        hidden_size = 256
        context_length = 2048

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
            print("\nStep 1: Encoding...")
            encoder_output, decoder_tokens, embeds, patch_lengths = model.encode(
                input_ids=input_ids
            )
            print(f"Encoder output shape: {encoder_output.shape}")

            # Step 4: Decode
            print("\nStep 2: Decoding...")
            decoder_output = model.decode(
                encoder_tokens=encoder_output,
                decoder_tokens=decoder_tokens,
                embeds=embeds,
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
