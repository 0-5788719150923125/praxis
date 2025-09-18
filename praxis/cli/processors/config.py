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
        valid_config_params = set(config_signature.parameters.keys()) - {"self", "kwargs"}

        # Extract and transform arguments
        config_kwargs = {}

        # Mapping of CLI argument names to config parameter names (for renamed parameters)
        arg_to_config_mapping = {
            "expert_type": "expert",
            "encoding_type": "encoding",
        }

        # Process all arguments from CLI
        for arg_name in vars(args):
            arg_value = getattr(args, arg_name)

            # Skip None values
            if arg_value is None:
                continue

            # Check if this arg needs to be mapped to a different config param name
            config_param = arg_to_config_mapping.get(arg_name, arg_name)

            # Only include parameters that PraxisConfig actually accepts
            if config_param in valid_config_params:
                config_kwargs[config_param] = arg_value

        # Special transformations that require custom logic

        # num_heads and num_queries from "num_heads:num_queries" format
        if hasattr(args, "num_heads") and args.num_heads:
            # Handle both string "X:Y" format and direct integer
            if isinstance(args.num_heads, str) and ":" in args.num_heads:
                parts = args.num_heads.split(":")
                config_kwargs["num_heads"] = int(parts[0])
                config_kwargs["num_queries"] = int(parts[1]) if len(parts) > 1 else int(parts[0])
            elif isinstance(args.num_heads, int):
                # If it's already an integer, use it directly
                config_kwargs["num_heads"] = args.num_heads
                # num_queries defaults to 1 if not specified separately
                if "num_queries" not in config_kwargs:
                    config_kwargs["num_queries"] = 1
            else:
                # It's a string but without ":", try to parse as integer
                config_kwargs["num_heads"] = int(args.num_heads)
                if "num_queries" not in config_kwargs:
                    config_kwargs["num_queries"] = 1

        # Handle the new defaulting logic:
        # num_layers defaults to 2 (set in CLI)
        # num_experts defaults to 1 (set in CLI) - no MoE by default
        # depth defaults to num_layers if not specified

        # Get num_layers (should always be present from CLI defaults)
        num_layers = config_kwargs.get("num_layers", 2)

        # depth defaults to num_layers if not specified
        if "depth" not in config_kwargs or config_kwargs.get("depth") is None:
            config_kwargs["depth"] = num_layers

        # Handle byte_latent encoding
        byte_latent = hasattr(args, "encoder_type") and args.encoder_type == "byte_latent"
        if byte_latent and "byte_latent" in valid_config_params:
            config_kwargs["byte_latent"] = True

        # Handle max_length based on block_size
        if hasattr(args, "block_size") and "max_length" in valid_config_params:
            block_size = args.block_size
            # Adjust for byte_latent if needed
            if byte_latent:
                block_size = block_size * 8
            config_kwargs["max_length"] = block_size * 8

        # Handle tokenizer IDs if tokenizer is provided
        if tokenizer is not None:
            token_id_mappings = [
                ("pad_token_id", "pad_token_id"),
                ("bos_token_id", "bos_token_id"),
                ("eos_token_id", "eos_token_id"),
                ("sep_token_id", "sep_token_id"),
            ]
            for tokenizer_attr, config_param in token_id_mappings:
                if hasattr(tokenizer, tokenizer_attr) and config_param in valid_config_params:
                    config_kwargs[config_param] = getattr(tokenizer, tokenizer_attr)

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

            optimizer_config, _ = get_optimizer_profile(args.optimizer, disable_schedule)

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
        filtered_kwargs = {k: v for k, v in config_kwargs.items() if k in valid_config_params}

        return PraxisConfig(**filtered_kwargs)