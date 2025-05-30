import random
from typing import Any, Dict, List, Optional, TypeVar, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

ConfigType = TypeVar("ConfigType", bound="object")


class GenomicBottleneck(nn.Module):
    """
    Neural bottleneck with trainable weights and evolutionary selection.

    This module implements a genomic bottleneck that uses an evolutionary algorithm
    to adapt transformation matrices (genomes) during training. It allows the network
    to discover efficient transformations for information compression.
    """

    def __init__(
        self,
        config: ConfigType,
        genome_dim: int = 23,
        population_size: int = 64,
        mutation_rate: float = 0.00001,
        tournament_size: int = 5,
        elite_size: int = 2,
        evolve_every_n_steps: int = 10,
        num_trials: int = 5,
    ):
        """
        Initialize genomic bottleneck.

        Args:
            config: Model configuration with hidden_size attribute
            genome_dim: Dimension of the genome transformation matrix
            population_size: Number of genomes in the population
            mutation_rate: Rate of random mutations applied to genomes
            tournament_size: Number of individuals to select in tournament selection
            elite_size: Number of best individuals to preserve without mutation
            evolve_every_n_steps: Frequency of evolution steps
            num_trials: Number of trials for fitness evaluation
        """
        super().__init__()
        self.input_dim: int = config.hidden_size
        self.quarter_input_dim: int = self.input_dim // 4
        self.output_dim: int = config.hidden_size
        self.genome_dim: int = genome_dim
        self.population_size: int = population_size
        self.mutation_rate: float = mutation_rate
        self.tournament_size: int = tournament_size
        self.elite_size: int = elite_size
        self.evolve_every_n_steps: int = evolve_every_n_steps
        self.num_trials: int = num_trials
        self.step_counter: int = 0

        # Initialize fitness scores to negative infinity
        self.fitness_scores: torch.Tensor = torch.full(
            (population_size,), float("-inf")
        )
        self.best_fitness: float = float("-inf")

        # Single projection for quarter-sized inputs
        self.left = nn.Linear(self.quarter_input_dim, genome_dim)
        self.right = nn.Linear(genome_dim, self.quarter_input_dim)

        # Initialize population of genomes
        self.population = nn.Parameter(
            torch.randn(population_size, genome_dim, genome_dim) * 0.02,
            requires_grad=False,
        )

        self.active_genome = nn.Parameter(
            self.population[0].clone(), requires_grad=False
        )

        self.stored_inputs: List[torch.Tensor] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through genomic bottleneck.

        Splits the input into four parts, routes one part through a residual connection,
        and processes the other three through the genomic bottleneck.

        Args:
            x: Input tensor of shape [batch_size, seq_len, hidden_size]

        Returns:
            Output tensor of same shape as input
        """
        # Split the input tensor into four equal parts along the feature dimension
        A, T, C, G = torch.split(x, self.quarter_input_dim, dim=-1)
        splits = [A, T, C, G]

        # Randomly select one part to be residual, the rest go through bottleneck
        indices = [0, 1, 2, 3]
        residual_idx = random.choice(indices)
        bottleneck_indices = indices.copy()
        bottleneck_indices.remove(residual_idx)

        residual_part = splits[residual_idx]
        bottleneck_parts = [splits[i] for i in bottleneck_indices]

        # Process each bottleneck part separately
        processed_parts = []
        for part in bottleneck_parts:
            # Project the bottleneck input
            projected = self.left(part)

            # Store input state for fitness evaluation
            if self.training:
                self.stored_inputs.append(projected.detach())
                if len(self.stored_inputs) > 50:
                    self.stored_inputs.pop(0)

            # Use the active genome for transformation
            with torch.no_grad():
                weight = self.active_genome.clone()

            # Apply linear transformation with the genome
            x_bottleneck = F.linear(projected, weight)

            # Project back to the original quarter dimension
            x_bottleneck = self.right(x_bottleneck)
            processed_parts.append(x_bottleneck)

        if self.training:
            self.step_counter += 1
            if (
                self.step_counter % self.evolve_every_n_steps == 0
                and len(self.stored_inputs) >= self.num_trials
            ):
                self.evolve_population(num_trials=self.num_trials)

        # Reconstruct the output by inserting parts in their original positions
        output_splits = []
        bottleneck_idx = 0
        for i in range(4):
            if i == residual_idx:
                output_splits.append(residual_part)
            else:
                output_splits.append(processed_parts[bottleneck_idx])
                bottleneck_idx += 1

        x = torch.cat(output_splits, dim=-1)
        return x

    def compute_fitness(self, genome: torch.Tensor, num_trials: int = 1) -> float:
        """
        Compute fitness based on cosine similarity of the bottlenecked part.

        Args:
            genome: Genome to evaluate
            num_trials: Number of random input samples to use for evaluation

        Returns:
            Fitness score (average cosine similarity)
        """
        if not self.stored_inputs:
            return float("-inf")

        fitness = 0.0
        for _ in range(num_trials):
            idx = random.randint(0, len(self.stored_inputs) - 1)
            input_sample = self.stored_inputs[idx]
            with torch.no_grad():
                x = F.linear(input_sample, genome)
                cos_sim = F.cosine_similarity(x, input_sample, dim=-1).mean().item()
                fitness += cos_sim
        fitness /= num_trials
        return fitness

    def tournament_select(self) -> torch.Tensor:
        """
        Select an individual using tournament selection.

        Returns:
            Selected genome
        """
        indices = torch.randint(0, self.population_size, (self.tournament_size,))
        tournament_fitness = self.fitness_scores[indices]
        winner_idx = indices[torch.argmax(tournament_fitness)]
        return self.population[winner_idx]

    def mutate(self, genome: torch.Tensor) -> torch.Tensor:
        """
        Apply small random mutations to a genome.

        Args:
            genome: Genome to mutate

        Returns:
            Mutated genome
        """
        with torch.no_grad():
            mutation = torch.randn_like(genome) * self.mutation_rate
            return genome + mutation

    def evolve_population(self, num_trials: int = 10) -> None:
        """
        Evolve the population using tournament selection.

        Args:
            num_trials: Number of trials for fitness evaluation
        """
        with torch.no_grad():
            # Evaluate the current population
            for i in range(self.population_size):
                genome = self.population[i]
                fitness = self.compute_fitness(genome, num_trials=num_trials)
                self.fitness_scores[i] = fitness

            # Store the best genome
            best_idx = torch.argmax(self.fitness_scores)
            if self.fitness_scores[best_idx] > self.best_fitness:
                self.best_fitness = self.fitness_scores[best_idx]
                self.active_genome.data.copy_(self.population[best_idx].clone())

            # Create a new population
            new_population = torch.zeros_like(self.population)

            # Preserve elite individuals
            sorted_indices = torch.argsort(self.fitness_scores, descending=True)
            new_population[: self.elite_size] = self.population[
                sorted_indices[: self.elite_size]
            ]

            # Generate the rest through tournament selection and mutation
            for i in range(self.elite_size, self.population_size):
                parent = self.tournament_select()
                child = self.mutate(parent)
                new_population[i] = child

            # Update the population
            self.population.data.copy_(new_population)

    def get_metrics(self) -> Dict[str, float]:
        """
        Get metrics for monitoring evolution progress.

        Returns:
            Dictionary of metrics
        """
        return {"fitness": self.best_fitness}


if __name__ == "__main__":
    print("Running tests for GenomicBottleneck...")

    class MockConfig:
        hidden_size = 512  # Must be divisible by 4 for splitting

    # Test parameters
    config = MockConfig()
    genome_dim = 30
    batch_size = 8
    sequence_length = 10

    # Create the layer
    layer = GenomicBottleneck(
        config=config, genome_dim=genome_dim, population_size=50, mutation_rate=0.01
    )

    # Test forward pass
    test_input = torch.randn(batch_size, sequence_length, config.hidden_size)
    output = layer(test_input)
    print(f"\nForward pass shape test:")
    print(f"Input shape: {test_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Genome shape: {layer.active_genome.shape}")
    assert output.shape == (
        batch_size,
        sequence_length,
        config.hidden_size,
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
        torch.randn(batch_size, sequence_length, config.hidden_size) for _ in range(5)
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
        print(f"  Best Fitness = {layer.get_metrics()['fitness']:.6f}")
        print(f"  Avg Fitness = {layer.fitness_scores.mean().item():.6f}")
        print(f"  Min Fitness = {layer.fitness_scores.min().item():.6f}")
        fitness_history.append(layer.get_metrics()["fitness"])

    # Check if fitness improved
    assert (
        fitness_history[-1] > fitness_history[0]
    ), "Fitness should improve over generations!"

    # Report parameter counts
    print(
        f"\nTotal trainable parameters: {sum(p.numel() for p in layer.parameters() if p.requires_grad)}"
    )
    print(f"Genome parameters: {layer.active_genome.numel()}")
