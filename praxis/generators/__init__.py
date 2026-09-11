from praxis import registry
from praxis.generators.energy import EnergyGenerator
from praxis.generators.flow import FlowGenerator
from praxis.generators.harmonic_latent import HarmonicLatentGenerator

registry.declare(
    "generators",
    title="Generators",
    doc=(
        "Continuous-output generators: each maps a conditioning hidden state to a "
        "sampled latent vector rather than a distribution over classes. All share "
        "one sample/forward surface; the CALM encoder picks one by the "
        "``generator_type`` baked into its profile."
    ),
    entries={
        "energy": EnergyGenerator,
        "flow": FlowGenerator,
        "harmonic": HarmonicLatentGenerator,
    },
)
