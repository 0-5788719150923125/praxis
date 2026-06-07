"""Networking-related CLI arguments."""

from praxis.pillars.thread import THREAD_REGISTRY


class NetworkingGroup:
    """Networking configuration arguments."""

    name = "networking"

    @classmethod
    def add_arguments(cls, parser):
        """Add networking arguments to the parser."""
        group = parser.add_argument_group(cls.name)

        group.add_argument(
            "--host-name",
            type=str,
            default="localhost",
            help="Serve the local API at this CNAME",
        )

        group.add_argument(
            "--port",
            type=int,
            default=2100,
            help="Serve the local API at this port",
        )

        group.add_argument(
            "--donations",
            type=str,
            default="https://buymeacoffee.com/vectorrent",
            help="URL the web app's donations icon links to (empty string hides it)",
        )

        group.add_argument(
            "--spider",
            type=str,
            nargs="*",
            default=None,
            metavar="KEY=VALUE",
            help=(
                "Enable the background web spider. Bare '--spider' uses the "
                "gentle profile; KEY=VALUE entries override it, e.g. "
                "'--spider profile=gentle tick_seconds=600 max_sites=8'. "
                "See SPIDER_REGISTRY for profiles."
            ),
        )

        group.add_argument(
            "--title",
            type=str,
            default=None,
            choices=sorted(THREAD_REGISTRY),
            help=(
                "Paper thread (layout) the living paper builds: the title block, "
                "master document, and which content generators run. "
                "See THREAD_REGISTRY in praxis/pillars/thread.py."
            ),
        )

        group.add_argument(
            "--author",
            type=str,
            nargs="*",
            default=["Ryan J. Brooks"],
            help=(
                "Authors written to the living paper's \\author block. The original author is always preserved and listed first. Pass additional names to append them."
            ),
        )
