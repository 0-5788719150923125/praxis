import os
import random

os.environ["BLT_SUPPRESS_ATTN_ERROR"] = "1"
os.environ["BLT_ALLOW_MISSING_FLEX_ATTENTION"] = "1"

import bytelatent
import torch
import torch.nn.functional as F
from bytelatent.data.patcher import Patcher, PatcherArgs, calculate_entropies
from bytelatent.model.blt import (
    ByteLatentTransformerArgs,
    EmbeddingType,
    compute_hash_embeddings,
    cross_attn_mask,
    decoder_patch_ids_from_lengths,
    get_decoder_dim_token_emb,
    get_encoder_dim_patch_emb,
    get_encoder_dim_token_emb,
    init_embeddings,
    patch_ids_from_lengths,
)
from bytelatent.model.local_models import LocalDecoder, LocalEncoder, LocalModelArgs
from bytelatent.model.utils import downsample
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
        self.entropy_model = None
        self.args.patching_mode = "entropy" if "entropy" in config.meta else "space"
        if self.args.patching_mode == "entropy":
            realtime_patching = True
            self.args.patching_threshold = 1.335442066192627
            self.args.monotonicity = True
            self.entropy_model = EntropyModel(
                260, config.hidden_size * 2, config.dropout
            )
            self.loss_scale = 0.01

            # Threshold optimization parameters
            self.target_ratio = 0.125  # 1/8th of original length

            # Register buffers for both current and EMA thresholds
            self.register_buffer(
                "optimal_threshold",
                torch.tensor(self.args.patching_threshold, dtype=torch.float32),
            )

        self.patcher = Patcher(
            PatcherArgs(
                realtime_patching=realtime_patching,
                entropy_model=self.entropy_model,
                device=self.device_map,
                patch_size=self.args.patch_size,
                patching_mode=self.args.patching_mode,
                threshold=self.args.patching_threshold,
                # threshold_add=self.args.patching_threshold_add,
                threshold_add=None,
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

    def encode(self, input_ids):
        aux_loss = 0
        if self.entropy_model is None:
            # Space patching mode
            patch_lengths, _ = self.patcher.patch(input_ids, include_next_token=True)
        else:
            # Entropy patching mode
            entropy_scores, entropy_preds = calculate_entropies(
                tokens=input_ids,
                entropy_model=self.entropy_model,
                patching_batch_size=input_ids.size(0),
                device=self.device_map,
                enable_grad=True,
            )
            if self.training:
                # First, find a safe threshold that guarantees we're under target length
                safe_threshold = self._find_safe_threshold(input_ids, entropy_scores)

                # Now sample thresholds that are guaranteed to be safe
                n_samples = 10
                # Sample only multipliers >= 1.0 to stay above safe_threshold
                multipliers = 1.0 + torch.abs(
                    torch.normal(
                        mean=0.0,
                        std=0.1,
                        size=(n_samples - 1,),
                        device=input_ids.device,
                    )
                )

                # Always include the safe threshold itself
                multipliers = torch.cat(
                    [multipliers, torch.tensor([1.0], device=input_ids.device)]
                )

                candidates = safe_threshold * multipliers

                # Rest proceeds as before, but now we're guaranteed safe
                best_threshold = safe_threshold
                best_ratio = float("inf")
                best_patch_lengths = None
                target_ratio = self.target_ratio

                # Try each candidate
                for threshold in candidates:
                    patch_lengths, _ = self.patcher.patch(
                        input_ids,
                        include_next_token=True,
                        threshold=threshold,
                        entropies=entropy_scores,
                    )

                    actual_ratio = patch_lengths.shape[1] / input_ids.shape[1]
                    distance = abs(actual_ratio - target_ratio)

                    if distance < best_ratio and actual_ratio <= target_ratio:
                        best_ratio = distance
                        best_threshold = threshold
                        best_patch_lengths = patch_lengths

                # Use best values
                self.optimal_threshold.fill_(best_threshold)
                patch_lengths = best_patch_lengths

                # Compute cross entropy loss
                aux_loss = (
                    F.cross_entropy(
                        entropy_preds[:, :-1].reshape(-1, entropy_preds.size(-1)),
                        input_ids[:, 1:].reshape(-1),
                    )
                    * self.loss_scale
                )

                if self.debug and random.random() < self.log_rate:
                    print(
                        f"DEBUG: original length={input_ids.size(1)}, reduced length={patch_lengths.shape[1]}, patching threshold={best_threshold:.10f}"
                    )

            else:
                # During inference, use stored optimal threshold
                patch_lengths, _ = self.patcher.patch(
                    input_ids,
                    include_next_token=True,
                    threshold=float(self.optimal_threshold.item()),
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

    def _find_safe_threshold(self, input_ids, entropy_scores, target_ratio=0.125):
        # Start with current threshold
        threshold = float(self.optimal_threshold.item())
        target_len = int(input_ids.shape[1] * target_ratio)

        # First, find any working threshold by doubling until we succeed
        while True:
            patch_lengths, _ = self.patcher.patch(
                input_ids,
                include_next_token=True,
                threshold=threshold,
                entropies=entropy_scores,
            )

            if patch_lengths.shape[1] <= target_len:
                # Found a working threshold
                safe_threshold = threshold
                break

            # If still too long, double the threshold
            threshold *= 2.0

            # Safety check - if we've increased too much, something is wrong
            if threshold > 1000.0:  # Arbitrary large number
                raise ValueError("Could not find a working threshold")

        # Now we can do binary search to optimize it
        left = threshold / 4.0  # Go back a bit to find better threshold
        right = threshold

        for _ in range(10):  # Usually converges in < 10 steps
            mid = (left + right) / 2
            patch_lengths, _ = self.patcher.patch(
                input_ids,
                include_next_token=True,
                threshold=mid,
                entropies=entropy_scores,
            )

            current_len = patch_lengths.shape[1]
            if current_len > target_len:
                # Sequence too long, need higher threshold
                left = mid
            else:
                # Sequence acceptable, but might be able to go lower
                right = mid
                safe_threshold = mid  # Keep track of last working threshold

            if abs(right - left) < 1e-4:
                break

        return safe_threshold


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


# class EntropyModel(nn.Module):
#     def __init__(self, vocab_size=260, hidden_size=256, dropout=0.1, n_layers=1):
#         super().__init__()

#         self.embedding = nn.Embedding(vocab_size, hidden_size)  # byte embedding

#         class Config:
#             dim = hidden_size
#             norm_eps = 1e-5

#         self.blocks = nn.ModuleList([RecurrentBlock(Config()) for i in range(n_layers)])

#     def forward(self, x: torch.Tensor, *args, **kwargs):
#         # x: [batch, seq_len]
#         x = self.embedding(x)  # [batch, seq_len, hidden_size]

#         for block in self.blocks:
#             x = block(x)

#         return x


class EntropyModel(nn.Module):
    def __init__(
        self, vocab_size=260, channels=256, dropout=0.1, n_layers=2, kernel_size=3
    ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, channels)  # byte embedding

        # Stack of dilated convolutions
        self.convs = nn.ModuleList()
        for i in range(n_layers):
            dilation = 2**i
            padding = (kernel_size - 1) * dilation  # Causal padding
            self.convs.append(
                nn.Sequential(
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size=kernel_size,
                        padding=padding,
                        dilation=dilation,
                    ),
                    nn.Dropout(dropout),
                )
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
            out = self.activation(x)
            out = conv(out)
            out = out[..., : x.size(-1)]
            x = out + x  # Residual connection

        x = x.transpose(1, 2)  # [batch, seq_len, channels]
        x = self.norm(x)

        return self.output(x)  # [batch, seq_len, vocab_size]


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
        encoder_hash_byte_group_size=[4],  # [3,4,5,6,7,8]
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
