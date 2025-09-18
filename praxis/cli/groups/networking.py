"""Networking-related CLI arguments."""


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