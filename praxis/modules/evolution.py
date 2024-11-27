import random

import torch
import torch.nn as nn
import torch.nn.functional as F


class GenomicBottleneck(nn.Module):
    """
    A DNA-like bottlenecking layer. Inspired by the following research:
    https://www.pnas.org/doi/10.1073/pnas.2409160121
    """

    __version__ = "0.1.0"

    def __init__(
        self,
        config,
        genome_dim: int = 23,
        population_size: int = 64,
        mutation_rate: float = 0.00001,
        tournament_size: int = 5,
        elite_size: int = 2,
    ):
        super().__init__()
        self.input_dim = config.num_dims
        self.output_dim = config.num_dims
        self.genome_dim = genome_dim
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size
        self.elite_size = elite_size

        # Initialize fitness scores to negative infinity
        self.fitness_scores = torch.full((population_size,), float("-inf"))
        self.best_fitness = float("-inf")

        # Trainable projections
        self.left = nn.Linear(self.input_dim, genome_dim)
        self.right = nn.Linear(genome_dim, self.output_dim)

        # Initialize population
        self.population = nn.Parameter(
            torch.randn(population_size, genome_dim, genome_dim) * 0.02,
            requires_grad=False,
        )

        self.active_genome = nn.Parameter(
            self.population[0].clone(), requires_grad=False
        )

        self.stored_inputs = []

    def forward(self, x):
        projected = self.left(x)

        # Store input state
        if self.training:
            self.stored_inputs.append(projected.detach())
            if len(self.stored_inputs) > 10:  # Keep last 10 states
                self.stored_inputs.pop(0)

        with torch.no_grad():
            weight = self.active_genome.clone()

        # Apply linear transformation with the genome
        x = F.linear(projected, weight)

        x = self.right(x)
        return x

    def compute_fitness(self, genome, num_trials=1):
        """Compute fitness based on reconstruction error"""
        if not self.stored_inputs:
            return float("-inf")

        fitness = 0.0
        for _ in range(num_trials):
            idx = random.randint(0, len(self.stored_inputs) - 1)
            input_sample = self.stored_inputs[idx]
            with torch.no_grad():
                x = F.linear(input_sample, genome)
                mse = F.mse_loss(x, input_sample)
                fitness -= mse.item()  # Lower MSE leads to higher fitness
        fitness /= num_trials
        return fitness

    def tournament_select(self):
        """Select individual using tournament selection"""
        indices = torch.randint(0, self.population_size, (self.tournament_size,))
        tournament_fitness = self.fitness_scores[indices]
        winner_idx = indices[torch.argmax(tournament_fitness)]
        return self.population[winner_idx]

    def mutate(self, genome):
        """Apply small random mutations"""
        with torch.no_grad():
            mutation = torch.randn_like(genome) * self.mutation_rate
            return genome + mutation

    def evolve_population(self, num_trials=10):
        """Evolve the population using tournament selection"""
        with torch.no_grad():
            # Evaluate current population
            for i in range(self.population_size):
                genome = self.population[i]
                fitness = self.compute_fitness(genome, num_trials=num_trials)
                self.fitness_scores[i] = fitness

            # Store best genome
            best_idx = torch.argmax(self.fitness_scores)
            if self.fitness_scores[best_idx] > self.best_fitness:
                self.best_fitness = self.fitness_scores[best_idx]
                self.active_genome.data.copy_(self.population[best_idx].clone())

            # Create new population
            new_population = torch.zeros_like(self.population)

            # Preserve elite individuals
            sorted_indices = torch.argsort(self.fitness_scores, descending=True)
            new_population[: self.elite_size] = self.population[
                sorted_indices[: self.elite_size]
            ]

            # Generate rest through tournament selection and mutation
            for i in range(self.elite_size, self.population_size):
                parent = self.tournament_select()
                child = self.mutate(parent)
                new_population[i] = child

            # Update population
            self.population.data.copy_(new_population)

    @property
    def current_best_fitness(self):
        return self.best_fitness.item()


if __name__ == "__main__":
    print("Running tests for GenomicBottleneck...")

    class MockConfig:
        num_dims = 512

    # Test parameters
    config = MockConfig()
    genome_dim = 30
    batch_size = 8
    sequence_length = 10

    # Create layer
    layer = GenomicBottleneck(
        config=config, genome_dim=genome_dim, population_size=50, mutation_rate=0.01
    )

    # Test forward pass
    test_input = torch.randn(batch_size, sequence_length, config.num_dims)
    output = layer(test_input)
    print(f"\nForward pass shape test:")
    print(f"Input shape: {test_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Genome shape: {layer.active_genome.shape}")
    assert output.shape == (
        batch_size,
        sequence_length,
        config.num_dims,
    ), "Shape mismatch!"

    # Test gradient flow
    test_input.requires_grad = True
    output = layer(test_input)
    loss = output.sum()
    loss.backward()

    # Verify gradient properties
    assert layer.active_genome.grad is None, "Genome should not have gradients!"
    assert layer.left.weight.grad is not None, "Input projection should have gradients!"
    assert (
        layer.right.weight.grad is not None
    ), "Output projection should have gradients!"
    assert test_input.grad is not None, "Input should have gradients!"
    print("\nGradient flow test passed!")

    print("\nTesting evolution over 100 generations...")
    layer.train()  # Enable training mode

    # Generate different test inputs for diversity
    test_inputs = [
        torch.randn(batch_size, sequence_length, config.num_dims) for _ in range(5)
    ]

    fitness_history = []
    num_trials = 5
    for generation in range(100):
        # Sample a test input
        test_input = random.choice(test_inputs)
        # Do forward pass to get stored inputs
        output = layer(test_input)
        # Evolve and print detailed stats
        layer.evolve_population(num_trials=num_trials)
        print(f"Generation {generation + 1}:")
        print(f"  Best Fitness = {layer.current_best_fitness:.6f}")
        print(f"  Avg Fitness = {layer.fitness_scores.mean().item():.6f}")
        print(f"  Min Fitness = {layer.fitness_scores.min().item():.6f}")
        fitness_history.append(layer.current_best_fitness)

    # Check if fitness improved
    assert (
        fitness_history[-1] > fitness_history[0]
    ), "Fitness should improve over generations!"

    # Report parameter counts
    print(
        f"\nTotal trainable parameters: {sum(p.numel() for p in layer.parameters() if p.requires_grad)}"
    )
    print(f"Genome parameters: {layer.active_genome.numel()}")
