"""Cellular automata visualizations for dashboard."""

import numpy as np


class ForestFireAutomata:
    """Forest fire cellular automaton simulation."""

    def __init__(self, width, height):
        """Initialize the forest fire simulation."""
        self.width = width
        self.height = height
        self.p_growth = 0.01
        self.p_lightning = 0.001

        # States: 0 = empty, 1 = tree, 2 = burning
        self.grid = np.zeros((height, width))
        self.grid = np.random.choice([0, 1], size=(height, width), p=[0.8, 0.2])

    def get_next_generation(self):
        new_grid = np.copy(self.grid)

        for i in range(self.height):
            for j in range(self.width):
                if self.grid[i, j] == 0:  # Empty
                    if np.random.random() < self.p_growth:
                        new_grid[i, j] = 1

                elif self.grid[i, j] == 1:  # Tree
                    neighbors = self.grid[
                        max(0, i - 1) : min(i + 2, self.height),
                        max(0, j - 1) : min(j + 2, self.width),
                    ]
                    if 2 in neighbors:
                        new_grid[i, j] = 2
                    elif np.random.random() < self.p_lightning:
                        new_grid[i, j] = 2

                elif self.grid[i, j] == 2:  # Burning
                    new_grid[i, j] = 0

        self.grid = new_grid
        return self.grid

    def to_ascii(self):
        """Convert the grid to ASCII art."""
        return [
            "".join(
                ["██" if cell == 1 else "░░" if cell == 2 else "  " for cell in row]
            )
            for row in self.grid
        ]
