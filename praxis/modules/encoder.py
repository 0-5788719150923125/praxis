import math
import os
import random
from typing import Optional, Union

os.environ["BLT_SUPPRESS_ATTN_ERROR"] = "1"
os.environ["BLT_ALLOW_MISSING_FLEX_ATTENTION"] = "1"

from contextlib import nullcontext

import bytelatent
import torch
import torch.nn.functional as F
from bytelatent import base_transformer
from bytelatent.data.patcher import Patcher, PatcherArgs, calculate_entropies
from bytelatent.model.blt import (
    ByteLatentTransformerArgs,
    EmbeddingType,
    compute_hash_embeddings,
    create_local_decoder,
    create_local_encoder,
    cross_attn_mask,
    decoder_patch_ids_from_lengths,
    get_blt_input,
    get_decoder_dim_token_emb,
    get_encoder_dim_patch_emb,
    get_encoder_dim_token_emb,
    init_embeddings,
    patch_ids_from_lengths,
)
from bytelatent.model.local_models import LocalDecoder, LocalEncoder, LocalModelArgs
from bytelatent.model.utils import downsample
from bytelatent.transformer import LMTransformer, LMTransformerArgs
from torch import nn

from praxis.modules.recurrent import minGRU


class PraxisEncoder(nn.Module):
    """
    An implementation of the Byte Latent Encoder/Decoder, from:
    https://arxiv.org/abs/2412.09871
    """

    def __init__(self, config: "AutoConfig"):
        super().__init__()
        self.debug = config.debug
        self.log_rate = 0.005
        self.device_map = config.device_map
        self.args = create_base_args(config)

        realtime_patching = False
        self.entropy = None
        self.args.patching_mode = "entropy" if "entropy" in config.meta else "space"
        if self.args.patching_mode == "entropy":
            realtime_patching = True
            self.args.patching_threshold = 1.335442066192627
            self.entropy = EntropyModel(
                vocab_size=260, channels=config.hidden_size, n_layers=1, kernel_size=3
            )

            # Threshold optimization parameters
            self.target_len = 0.125  # 1/8th of original length
            self.min_threshold = 0.1
            self.max_threshold = 10.0
            self.absolute_max_attempts = 50

            # Register buffers for both current and EMA thresholds
            self.register_buffer(
                "optimal_threshold",
                torch.tensor(self.args.patching_threshold, dtype=torch.float32),
            )
            self.register_buffer(
                "ema_threshold",
                torch.tensor(self.args.patching_threshold, dtype=torch.float32),
            )

            # Parameters for stability vs exploration
            self.ema_alpha = 0.001  # Weight for new values (0.9 for history)
            self.decay_rate = 0.999  # Slight decay each forward pass
            self.random_scale = 0  # Random perturbations in update step

        self.patcher = Patcher(
            PatcherArgs(
                realtime_patching=realtime_patching,
                entropy_model=self.entropy,
                device=self.device_map,
                patch_size=self.args.patch_size,
                patching_mode=self.args.patching_mode,
                threshold=self.args.patching_threshold,
                threshold_add=self.args.patching_threshold_add,
                monotonicity=self.args.monotonicity,
            )
        )
        self.embeds = init_embeddings(
            self.args,
            EmbeddingType.HASH_TOK,
            local_encoder_dim=self.args.dim_local_encoder,
            encoder_hash_byte_group_size=self.args.encoder_hash_byte_group_size,
        )
        self.encoder = RecurrentEncoder(create_local_encoder_args(self.args))
        self.decoder = RecurrentDecoder(create_local_decoder_args(self.args))

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            + f"type='byte_latent', "
            + f"n_encoders={len(self.encoder.layers)}, "
            + f"n_decoders={len(self.decoder.layers)})"
        )

    def update_threshold(self, new_threshold):
        """Update threshold using EMA and small perturbations"""
        # Apply decay
        decayed_threshold = new_threshold * self.decay_rate

        # Update EMA
        current_ema = float(self.ema_threshold.item())
        new_ema = (
            current_ema * (1 - self.ema_alpha) + decayed_threshold * self.ema_alpha
        )

        # Add tiny random perturbation to EMA
        random_factor = 1.0 + (torch.rand(1).item() - 0.5) * self.random_scale
        final_threshold = new_ema * random_factor

        # Update buffers with bounds checking
        final_threshold = max(
            self.min_threshold, min(self.max_threshold, final_threshold)
        )
        self.optimal_threshold.fill_(float(final_threshold))
        self.ema_threshold.fill_(float(new_ema))

        return float(final_threshold)

    def adjust_threshold(self, original_len, reduced_len, current_threshold, attempt):
        """Compute next threshold based on current length"""
        target_len = int(original_len * self.target_len)
        if reduced_len <= target_len:
            if reduced_len < 0.5 * target_len:
                # Too small, careful decrease
                return current_threshold * 0.98
            return current_threshold  # Keep current if in good range

        # We're above target, increase based on how far we are
        ratio = reduced_len / target_len
        adjustment = max(0.001, current_threshold * 0.05 * min(ratio, 2.0))
        return current_threshold + adjustment

    def encode(self, input_ids):
        if self.entropy is None:
            # Space patching mode
            patch_lengths, tok_scores = self.patcher.patch(
                input_ids,
                include_next_token=True,
                threshold=self.patcher.threshold,
            )
        else:
            # Entropy patching mode
            entropy_scores, entropy_preds = calculate_entropies(
                tokens=input_ids,
                entropy_model=self.entropy,
                patching_batch_size=input_ids.size(0),
                device=self.device_map,
                enable_grad=True,
            )
            if self.training:
                # Start from current EMA with small perturbation
                current_threshold = self.update_threshold(
                    float(self.optimal_threshold.item())
                )
                best_threshold = current_threshold
                best_length = float("inf")

                attempt = 0
                consecutive_failures = 0

                original_len = input_ids.size(1)
                target_len = int(original_len * self.target_len)

                while attempt < self.absolute_max_attempts:
                    attempt += 1
                    patch_lengths, tok_scores = self.patcher.patch(
                        input_ids,
                        include_next_token=True,
                        threshold=current_threshold,
                        entropies=entropy_scores,
                    )
                    reduced_len = patch_lengths.shape[1]

                    # Update best if this gives valid length
                    if reduced_len <= target_len:
                        if best_length == float("inf") or reduced_len > best_length:
                            best_length = reduced_len
                            best_threshold = current_threshold
                            consecutive_failures = 0

                            # Update with this good threshold
                            current_threshold = self.update_threshold(best_threshold)

                            # If we're close enough to target, we can stop
                            if reduced_len >= 0.5 * target_len:
                                break

                    else:
                        consecutive_failures += 1

                    # Break if we're stuck
                    if consecutive_failures >= 10:
                        break

                    # Get next threshold and update
                    next_threshold = self.adjust_threshold(
                        original_len, reduced_len, current_threshold, attempt
                    )
                    current_threshold = self.update_threshold(next_threshold)

                # Use best threshold found
                if best_length <= target_len:
                    current_threshold = self.update_threshold(best_threshold)

                # Final patch with current threshold
                patch_lengths, tok_scores = self.patcher.patch(
                    input_ids,
                    include_next_token=True,
                    threshold=current_threshold,
                    entropies=entropy_scores,
                )

                # Emergency adjustment if still over target
                while patch_lengths.shape[1] > target_len:
                    current_threshold *= 1.1
                    current_threshold = self.update_threshold(current_threshold)
                    patch_lengths, tok_scores = self.patcher.patch(
                        input_ids,
                        include_next_token=True,
                        threshold=current_threshold,
                        entropies=entropy_scores,
                    )

                if self.debug and random.random() < self.log_rate:
                    print(
                        f"DEBUG: original length={original_len}, reduced length={patch_lengths.shape[1]}, patching threshold={current_threshold:.10f}"
                    )

            else:
                # During inference, use stored optimal threshold
                patch_lengths, tok_scores = self.patcher.patch(
                    input_ids,
                    include_next_token=True,
                    threshold=float(self.ema_threshold.item()),
                    entropies=entropy_scores,
                )

        patch_ids = patch_ids_from_lengths(patch_lengths, input_ids.shape[-1])

        # Create cross attention mask if needed
        cross_attn_mask_enc = None
        if self.args.cross_attn_encoder:
            cross_attn_mask_enc = cross_attn_mask(
                patch_ids,
                patch_lengths,
                input_ids.shape[-1],  # N
                patches_as_queries=True,
                cross_attn_k=self.args.cross_attn_k,
                window=self.args.cross_attn_window_encoder,
                block_mask=False,
            )

        # Compute embeddings
        embeds = compute_hash_embeddings(
            local_encoder_tokens=input_ids,
            local_encoder=self.encoder,
            encoder_hash_tok_embedding=self.embeds,
            encoder_hash_byte_group_nb_functions=self.args.encoder_hash_byte_group_nb_functions,
            encoder_hash_byte_group_size=self.args.encoder_hash_byte_group_size,
            encoder_hash_byte_group_vocab=self.args.encoder_hash_byte_group_vocab,
        )

        # Local encoder with cross attention
        (h_encoder, h_cross), _ = self.encoder(
            tokens=input_ids,
            embeds=embeds,
            patch_embeds=None,
            cross_mask=cross_attn_mask_enc,
            num_patches=patch_lengths.shape[1],
            patch_ids=patch_ids,
        )

        aux_loss = 0
        if self.training and self.entropy is not None:
            # Compute cross entropy loss
            aux_loss = F.cross_entropy(
                entropy_preds[:, :-1].reshape(-1, entropy_preds.size(-1)),
                input_ids[:, 1:].reshape(-1),
            )

        # Downsampling
        if self.args.cross_attn_encoder:
            h = h_cross.view(h_encoder.shape[0], -1, h_encoder.shape[-1])
        else:
            h = downsample(
                h_encoder,
                patch_lengths.shape[1],
                patch_lengths,
                patch_ids,
                downsampling_by_pooling=self.args.downsampling_by_pooling,
                patch_size=self.args.patch_size,
            )

        return h, h_encoder, patch_lengths, aux_loss

    def decode(self, h, h_encoder, input_ids, patch_lengths):
        nb_boe = 0

        decoder_patch_ids = decoder_patch_ids_from_lengths(
            patch_lengths, nb_boe, input_ids.shape[-1]
        )

        # Upsampling
        if self.args.cross_attn_decoder:
            h = (
                h
                if self.args.cross_attn_encoder
                else h.repeat_interleave(self.args.cross_attn_k, dim=1)
            )
            cross_mask = cross_attn_mask(
                decoder_patch_ids,
                patch_lengths,
                input_ids.shape[-1],
                patches_as_queries=False,
                cross_attn_k=self.args.cross_attn_k,
                window=self.args.cross_attn_window_decoder,
                block_mask=False,
            )
        else:
            h = torch.gather(
                h,
                1,
                decoder_patch_ids.unsqueeze(-1).expand(-1, -1, h.shape[-1]),
            )
            cross_mask = None
            assert (
                h.shape[1] == input_ids.shape[1]
            ), "Sequence length mismatch after gathering"

        output, _ = self.decoder(
            tokens=input_ids,
            patch_embeds=h,
            embeds=h_encoder,
            cross_mask=cross_mask,
        )

        return output


class RecurrentBlock(minGRU):
    """
    We replace transformer blocks in the encoder/decoder with something
    that is more memory-efficient, and faster to compute.
    """

    def __init__(self, args):
        super().__init__(dim=args.dim, proj_out=True)
        self.norm = nn.RMSNorm(args.dim, eps=args.norm_eps)

    def forward(self, x: torch.Tensor, *args, **kwargs):
        out, _ = super().forward(self.norm(x))
        return out + x


class RecurrentEncoder(LocalEncoder):
    def __init__(self, args: LocalModelArgs):
        super().__init__(args)
        self.layers = nn.ModuleList(
            [RecurrentBlock(args) for _ in range(args.n_layers)]
        )


class RecurrentDecoder(LocalDecoder):
    def __init__(self, args: LocalModelArgs):
        super().__init__(args)
        self.layers = nn.ModuleList(
            [RecurrentBlock(args) for _ in range(args.n_layers)]
        )


class EntropyModel(nn.Module):
    def __init__(self, vocab_size=260, channels=256, n_layers=1, kernel_size=3):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, channels)  # byte embedding

        # Stack of dilated convolutions
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(
                    channels,
                    channels,
                    kernel_size=kernel_size,
                    padding=(kernel_size - 1) * (2**i),
                    dilation=2**i,
                )
                for i in range(n_layers)
            ]
        )

        self.activation = nn.SiLU()  # Same as paper
        self.norm = nn.LayerNorm(channels)

        # Project to byte probabilities
        self.output = nn.Linear(channels, vocab_size)

    def forward(self, x: torch.Tensor, *args, **kwargs):
        # x: [batch, seq_len]
        x = self.embedding(x).transpose(1, 2)  # [batch, channels, seq_len]

        # Causal convolution stack
        for conv in self.convs:
            x = self.activation(conv(x))

        x = x.transpose(1, 2)  # [batch, seq_len, channels]
        x = self.norm(x)

        return self.output(x)  # [batch, seq_len, vocab_size]


# class EntropyModel(nn.Module):
#     def __init__(self, vocab_size=260, channels=256, n_layers=1, kernel_size=3):
#         super().__init__()

#         self.embedding = nn.Embedding(vocab_size, channels)  # byte embedding

#         class Config:
#             dim = channels
#             norm_eps = 1e-5

#         self.blocks = nn.ModuleList([RecurrentBlock(Config()) for i in range(n_layers)])

#     def forward(self, x: torch.Tensor, *args, **kwargs):
#         # x: [batch, seq_len]
#         x = self.embedding(x)  # [batch, seq_len, channels]

#         for block in self.blocks:
#             x = block(x)

#         return x


def create_base_args(config):
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
        patching_threshold=3.1439168453216553,  # not used with "space" patch_mode
        # patching_threshold=1.335442066192627, # use this for "entropy" patch_mode
        patch_size=6,  # use this for "space" patch_mode
        # patch_size=4.5, # use this for "entropy" patch_mode
        # tokenization_mode="bytes",
        patching_mode="space",  # space patching [is] a very close competitor to dynamic entropy based patching.
        # patching_mode="bpe",
        # patching_mode="entropy",
        encoder_hash_byte_group_nb_functions=1,
        encoder_hash_byte_group_size=[4],
        # encoder_hash_byte_group_vocab=config.vocab_size,
        encoder_hash_byte_group_vocab=config.vocab_size * 8,
        cross_attn_encoder=False,  # the authors found that using cross-attention in the decoder is most effective.
        cross_attn_decoder=False,
        cross_attn_window_encoder=512,
        cross_attn_window_decoder=512,
        cross_attn_k=4,  # is a multiplier for token and patch embedding
        cross_attn_nheads=2,  # num heads used in cross attn
        n_layers_local_encoder=1,
        n_layers_local_decoder=1,
        n_heads_local_encoder=1,
        n_heads_local_decoder=1,
        cross_attn_all_layers_encoder=False,
        cross_attn_all_layers_decoder=False,
        cross_attn_use_flex_attention=False,  # not supported on CPU and older GPUs
        cross_attn_init_by_pooling=True,
        # log_patch_lengths=True,
        # non_linearity="swiglu", # not implemented
        use_rope=True,
        # recompute_fc1_out=False, # I don't think these do anything
        # recompute_fc3_out=False,
        # recompute_attn=False,
        # custom_bwd=False,
        # layer_ckpt="none",
        # efficient_attn="sdpa",
        # patch_only_encoder=False, # doesn't do anything
        # patch_only_decoder=False,
        # use_local_encoder_transformer=True,
        # init_use_gaussian=True,
        # init_use_depth="current",
        # attn_bias_type="local_block_causal",
        # alpha_depth="disabled",
        # local_attention_window_len=512,
        # sliding_window=256,  # basically required, else encoder dim is equal to max_seq_len
        downsampling_by_pooling="max",
        share_encoder_decoder_emb=False,
        dropout=config.dropout,
        entropy_model_checkpoint_dir=None,
    )


def create_local_encoder_args(args: ByteLatentTransformerArgs) -> LocalModelArgs:
    return LocalModelArgs(
        # Updated args
        dim=args.dim_local_encoder,
        n_layers=args.n_layers_local_encoder,
        n_heads=args.n_heads_local_encoder,
        dim_token_emb=get_encoder_dim_token_emb(args),
        dim_patch_emb=get_encoder_dim_patch_emb(args),
        cross_attn_encoder=args.cross_attn_encoder,
        cross_attn_decoder=False,
        cross_attn_k=args.cross_attn_k if args.cross_attn_encoder else None,
        cross_attn_init_by_pooling=args.cross_attn_init_by_pooling,
        # Defaults
        head_dim=args.head_dim,
        max_seqlen=args.max_encoder_seq_length,
        dropout=args.dropout,
        vocab_size=args.vocab_size + args.pm_size,
        norm_eps=args.norm_eps,
        patch_size=args.patch_size,
        sliding_window=args.local_attention_window_len,
        use_rope=args.use_rope,
        rope_theta=args.rope_theta,
        init_base_std=args.init_base_std,
        init_std_factor=args.init_std_factor,
        n_kv_heads=args.n_kv_heads,
        attn_impl=args.attn_impl,
        attn_bias_type="local_block_causal",
        multiple_of=args.multiple_of,
        ffn_dim_multiplier=args.ffn_dim_multiplier,
        patching_mode=args.patching_mode,
        use_local_encoder_transformer=args.use_local_encoder_transformer,
        downsampling_by_pooling=args.downsampling_by_pooling,
        encoder_hash_byte_group_size=args.encoder_hash_byte_group_size,
        cross_attn_all_layers_encoder=args.cross_attn_all_layers_encoder,
        cross_attn_all_layers_decoder=args.cross_attn_all_layers_decoder,
        cross_attn_nheads=args.cross_attn_nheads,
        eos_id=args.eos_id,
    )


def create_local_decoder_args(args: ByteLatentTransformerArgs) -> LocalModelArgs:
    # First deep copy the original args
    return LocalModelArgs(
        dim=args.dim_local_decoder,
        n_layers=args.n_layers_local_decoder,
        n_heads=args.n_heads_local_decoder,
        dim_token_emb=get_decoder_dim_token_emb(args),
        dim_patch_emb=args.dim_global,
        cross_attn_encoder=False,
        cross_attn_decoder=args.cross_attn_decoder,
        cross_attn_init_by_pooling=False,  # states are already defined
        cross_attn_k=args.cross_attn_k if args.cross_attn_decoder else None,
        # Defaults
        head_dim=args.head_dim,
        max_seqlen=args.max_encoder_seq_length,
        dropout=args.dropout,
        vocab_size=args.vocab_size + args.pm_size,
        norm_eps=args.norm_eps,
        patch_size=args.patch_size,
        sliding_window=args.local_attention_window_len,
        use_rope=args.use_rope,
        rope_theta=args.rope_theta,
        init_base_std=args.init_base_std,
        init_std_factor=args.init_std_factor,
        n_kv_heads=args.n_kv_heads,
        attn_impl=args.attn_impl,
        attn_bias_type="local_block_causal",
        multiple_of=args.multiple_of,
        ffn_dim_multiplier=args.ffn_dim_multiplier,
        patching_mode=args.patching_mode,
        use_local_encoder_transformer=args.use_local_encoder_transformer,
        downsampling_by_pooling=args.downsampling_by_pooling,
        encoder_hash_byte_group_size=args.encoder_hash_byte_group_size,
        cross_attn_all_layers_encoder=args.cross_attn_all_layers_encoder,
        cross_attn_all_layers_decoder=args.cross_attn_all_layers_decoder,
        cross_attn_nheads=args.cross_attn_nheads,
        eos_id=args.eos_id,
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
    model = PraxisEncoder(config)

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
                h=h,
                h_encoder=h_encoder,
                input_ids=decoder_tokens,
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
    print(model)
    print("Done!")
