"""Persistence and storage CLI arguments."""


class PersistenceGroup:
    """Storage and persistence configuration arguments."""

    name = "persistence"

    @classmethod
    def add_arguments(cls, parser):
        """Add persistence arguments to the parser."""
        group = parser.add_argument_group(cls.name)

        group.add_argument(
            "--cache-dir",
            type=str,
            default="build",
            help="Paths to a directory where artifacts will be saved",
        )
