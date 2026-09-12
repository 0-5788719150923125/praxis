"""Inference-time arguments: what the model is told, and how it decodes.

Every flag here is excluded from the run hash. None of them changes the model
or what it trains on - two runs that differ only in how they were prompted at
inference are the same run, and forking a checkpoint directory over a
temperature change would be wrong.
"""


class InferenceGroup:
    """Standing instructions and decoding parameters for a run."""

    name = "inference"

    @classmethod
    def add_arguments(cls, parser):
        group = parser.add_argument_group(cls.name)

        group.add_argument(
            "--system-prompt",
            type=str,
            default=None,
            exclude_hash=True,
            doc=(
                "A ``system`` message prepended to every conversation this run "
                "serves - the web chat, Discord, and the raw API alike. A "
                "caller that sends its own system message wins, so this is a "
                "default rather than a lock. Unset sends none."
            ),
            help="System message prepended to every served conversation",
        )

        group.add_argument(
            "--developer-prompt",
            type=str,
            default=None,
            exclude_hash=True,
            doc=(
                "A ``developer`` message prepended to every conversation, and "
                "the default contents of the web app's editable developer "
                "prompt at the top of the chat - editing it there overrides "
                "this for that browser. Chat formats with no ``developer`` "
                "role (``hf_native``, a published checkpoint's own template) "
                "fold it into the system message instead of emitting a role "
                "their template cannot render."
            ),
            help="Developer message prepended to every served conversation",
        )

        group.add_argument(
            "--generation-kwargs",
            nargs="*",
            default=None,
            metavar="KEY=VALUE",
            exclude_hash=True,
            doc=(
                "Decoding parameters for served generation, as ``key=value`` "
                "pairs on the command line or a mapping in an experiment YAML. "
                "Any ``transformers`` ``GenerationConfig`` field works "
                "(``max_new_tokens``, ``temperature``, ``top_p``, ``top_k``, "
                "``min_p``, ``repetition_penalty``, ``no_repeat_ngram_size``, "
                "``do_sample``, ...), plus ``use_cache``, ``timeout``, "
                "``truncate_to`` and ``skip_special_tokens``. An unknown key is "
                "an error, because transformers would accept and silently drop "
                "it. These become the defaults the web app's Settings form is "
                "seeded with; edits there override them for that browser. The "
                "Terminal tab's rolling contexts are NOT affected - they are a "
                "fixed probe with their own arguments."
            ),
            help="key=value decoding parameters for served generation",
        )
