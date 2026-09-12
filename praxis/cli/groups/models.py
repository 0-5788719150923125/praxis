"""Foreign-model arguments: training a published checkpoint under Praxis."""


class ModelsGroup:
    """Loading a published ``transformers`` model instead of building one."""

    name = "models"

    @classmethod
    def add_arguments(cls, parser):
        group = parser.add_argument_group(cls.name)

        group.add_argument(
            "--model-name",
            type=str,
            default=None,
            doc=(
                "Load this published checkpoint (a Hugging Face repo id or a "
                "local directory) instead of building a model from the Praxis "
                "registries. The checkpoint brings its own architecture, "
                "config, vocabulary and chat template, all used verbatim - so "
                "the Praxis architecture flags do not apply and passing one is "
                "an error rather than a silent no-op. Everything about the RUN "
                "still applies: optimizer, batch, precision, losses, "
                "regularizers, task weights, RL policies and the whole data "
                "pipeline."
            ),
            help="Published model to load instead of building one",
        )

        group.add_argument(
            "--model-revision",
            type=str,
            default=None,
            doc=(
                "Git revision (branch, tag or commit) of ``--model-name`` to "
                "load. Part of the run's identity, because two revisions of "
                "one repo are two different models."
            ),
            help="Revision of --model-name to load",
        )

        group.add_argument(
            "--model-task",
            type=str,
            registry="model_tasks",
            default="causal_lm",
            help="Which task family --model-name is loaded as",
        )

        group.add_argument(
            "--model-kwarg",
            action="append",
            default=None,
            metavar="KEY=VALUE",
            doc=(
                "A keyword argument forwarded VERBATIM to the checkpoint's "
                "``from_pretrained`` (repeatable). This is the whole escape "
                "hatch for model-specific options - Praxis deliberately has no "
                "translation layer between its own names and a foreign model's, "
                "because an alias that disagrees with what lands in "
                "``config.json`` eventually becomes a lie about what ran. "
                "Values are read as YAML, so ``trust_remote_code=true`` and "
                "``num_labels=3`` arrive as the types transformers expects."
            ),
            help="key=value forwarded verbatim to from_pretrained (repeatable)",
        )

        group.add_argument(
            "--peft-type",
            type=str,
            registry="peft_profiles",
            default=None,
            help="Adapter profile to finetune --model-name with",
        )
