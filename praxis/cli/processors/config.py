"""Configuration building for PraxisConfig."""

import inspect


class ConfigBuilder:
    """Builds PraxisConfig from parsed arguments."""

    @staticmethod
    def create_praxis_config(args=None, tokenizer=None):
        """
        Create a PraxisConfig object from CLI arguments.
        Automatically maps CLI arguments to PraxisConfig parameters based on the config's signature.

        Args:
            args: Parsed arguments object
            tokenizer: Optional tokenizer for token IDs

        Returns:
            PraxisConfig: Configuration object ready for model initialization
        """
        from praxis import PraxisConfig

        # Get PraxisConfig's __init__ parameters to know what it accepts
        config_signature = inspect.signature(PraxisConfig.__init__)
        valid_config_params = set(config_signature.parameters.keys()) - {
            "self",
            "kwargs",
        }

        # Extract and transform arguments
        config_kwargs = {}

        # Mapping of CLI argument names to config parameter names (for renamed parameters)
        arg_to_config_mapping = {
            "ffn_type": "expert",
            "encoding_type": "encoding",
        }

        # Experiment/environment activator flags should not leak into PraxisConfig.
        from praxis.cli import get_loader_flag_attrs

        loader_flag_attrs = get_loader_flag_attrs()

        # Experimental kwargs (from experiment files, not in PraxisConfig signature)
        experimental_kwargs = {}

        # Process all arguments from CLI
        for arg_name in vars(args):
            if arg_name in loader_flag_attrs:
                continue

            arg_value = getattr(args, arg_name)

            # Skip None values
            if arg_value is None:
                continue

            # Check if this arg needs to be mapped to a different config param name
            config_param = arg_to_config_mapping.get(arg_name, arg_name)

            # Include parameters that PraxisConfig explicitly accepts
            if config_param in valid_config_params:
                config_kwargs[config_param] = arg_value
            else:
                # This might be an experimental parameter from an experiment file
                # Store it separately so we can pass it to PraxisConfig
                experimental_kwargs[config_param] = arg_value

        # Handle the new defaulting logic:
        # num_layers defaults to 2 (set in CLI)
        # num_experts defaults to 1 (set in CLI) - no MoE by default
        # depth defaults to num_layers if not specified

        # Get num_layers (should always be present from CLI defaults)
        num_layers = config_kwargs.get("num_layers", 2)

        # depth defaults to num_layers if not specified
        if "depth" not in config_kwargs or config_kwargs.get("depth") is None:
            config_kwargs["depth"] = num_layers

        # Gate byte-token-specific behavior (bits-per-byte metric, readable
        # terminal length, repetition defaults). Pure tokenizer concern;
        # byte-latent encoders force tokenizer_type='byte_level' upstream.
        tokenizer_type = getattr(args, "tokenizer_type", None)
        byte_level = tokenizer_type == "byte_level"
        if byte_level and "byte_level" in valid_config_params:
            config_kwargs["byte_level"] = True

        if "max_position_embeddings" in valid_config_params:
            explicit = getattr(args, "max_position_embeddings", None)
            if explicit is not None:
                config_kwargs["max_position_embeddings"] = explicit

        # Handle tokenizer IDs if tokenizer is provided
        if tokenizer is not None:
            token_id_mappings = [
                ("pad_token_id", "pad_token_id"),
                ("bos_token_id", "bos_token_id"),
                ("eos_token_id", "eos_token_id"),
                ("sep_token_id", "sep_token_id"),
            ]
            for tokenizer_attr, config_param in token_id_mappings:
                if (
                    hasattr(tokenizer, tokenizer_attr)
                    and config_param in valid_config_params
                ):
                    config_kwargs[config_param] = getattr(tokenizer, tokenizer_attr)

            # Sync the tokenizer's reported vocab_size into the model
            # config and args. For Standard tokenizers this captures the
            # actual trained vocab (may differ from the CLI target).
            # For byte/char tokenizers ``vocab_size`` is the model-facing
            # value already (the byte-level alphabet is exposed
            # separately as ``byte_alphabet_size``), so this is a no-op.
            tok_vocab = getattr(tokenizer, "vocab_size", None)
            if (
                isinstance(tok_vocab, int)
                and tok_vocab > 0
                and "vocab_size" in valid_config_params
            ):
                config_kwargs["vocab_size"] = tok_vocab
                if args is not None:
                    setattr(args, "vocab_size", tok_vocab)

            # Encoders that own their embeddings (e.g. CALM) size to the
            # tokenizer's true vocabulary, not the (hash-overloaded) vocab_size.
            # Byte/char tokenizers expose it as byte_alphabet_size.
            byte_vocab = getattr(tokenizer, "byte_alphabet_size", None)
            if (
                isinstance(byte_vocab, int)
                and byte_vocab > 0
                and "byte_vocab_size" in valid_config_params
            ):
                config_kwargs["byte_vocab_size"] = byte_vocab

        # Handle optimizer configuration
        if hasattr(args, "optimizer") and args.optimizer:
            from praxis.optimizers import get_optimizer_profile

            # Check for schedule-related flags
            disable_schedule = any(
                [
                    getattr(args, "fixed_schedule", False),
                    getattr(args, "schedule_free", False),
                ]
            )

            optimizer_config, _ = get_optimizer_profile(
                args.optimizer, disable_schedule
            )

            if "optimizer_config" in valid_config_params:
                config_kwargs["optimizer_config"] = optimizer_config

            # Add optimizer wrappers if they're valid config params
            if "optimizer_wrappers" in valid_config_params:
                config_kwargs["optimizer_wrappers"] = {
                    "trac": getattr(args, "trac", False),
                    "ortho": getattr(args, "ortho", False),
                    "lookahead": getattr(args, "lookahead", False),
                    "schedule_free": getattr(args, "schedule_free", False),
                }

        # Filter out any kwargs that aren't valid for PraxisConfig
        filtered_kwargs = {
            k: v for k, v in config_kwargs.items() if k in valid_config_params
        }

        # Merge experimental kwargs (from experiment files)
        # These will be caught by PraxisConfig's **kwargs and passed through
        all_kwargs = {**filtered_kwargs, **experimental_kwargs}

        return PraxisConfig(**all_kwargs)
