"""
Simple math problems for bootstrapping RL training.

These easy problems help untrained models get some correct answers,
providing the variance in rewards needed for GRPO to work.
"""

import random
from typing import Dict, List


def generate_simple_arithmetic() -> Dict[str, str]:
    """Generate a simple arithmetic problem with ground truth."""
    operation = random.choice(['+', '-', '*'])
    
    if operation == '+':
        a = random.randint(1, 50)
        b = random.randint(1, 50)
        answer = a + b
        problem = f"What is {a} + {b}?"
    elif operation == '-':
        a = random.randint(10, 100)
        b = random.randint(1, a)  # Ensure positive result
        answer = a - b
        problem = f"What is {a} - {b}?"
    else:  # multiplication
        a = random.randint(2, 12)
        b = random.randint(2, 12)
        answer = a * b
        problem = f"What is {a} × {b}?"
    
    return {
        "prompt": problem,
        "ground_truth": str(answer),
        "difficulty": 0.9  # High solve rate for easy problems
    }


def generate_counting() -> Dict[str, str]:
    """Generate simple counting problems."""
    n = random.randint(3, 10)
    items = random.choice(["apples", "books", "cats", "dogs", "birds"])
    
    problem = f"If I have {n} {items}, how many {items} do I have?"
    
    return {
        "prompt": problem,
        "ground_truth": str(n),
        "difficulty": 0.95
    }


def generate_comparison() -> Dict[str, str]:
    """Generate number comparison problems."""
    a = random.randint(1, 100)
    b = random.randint(1, 100)
    
    if random.choice([True, False]):
        problem = f"Which is larger: {a} or {b}?"
        answer = str(max(a, b))
    else:
        problem = f"Which is smaller: {a} or {b}?"
        answer = str(min(a, b))
    
    return {
        "prompt": problem,
        "ground_truth": answer,
        "difficulty": 0.8
    }


class SimpleMathDataset:
    """Dataset of simple math problems for RL bootstrapping."""
    
    def __init__(self, mix_ratio: float = 0.2):
        """
        Args:
            mix_ratio: Fraction of simple problems to mix with hard problems
        """
        self.mix_ratio = mix_ratio
        self.generators = [
            generate_simple_arithmetic,
            generate_counting,
            generate_comparison,
        ]
    
    def should_use_simple(self) -> bool:
        """Decide whether to use a simple problem."""
        return random.random() < self.mix_ratio
    
    def generate(self) -> Dict[str, str]:
        """Generate a random simple math problem."""
        generator = random.choice(self.generators)
        return generator()
    
    def format_for_rl(self, problem: Dict[str, str]) -> Dict[str, any]:
        """Format problem for RL training."""
        return {
            "prompt": problem["prompt"],
            "verification_info": f'{{"ground_truth": "{problem["ground_truth"]}"}}',
            "solve_rate_qwen_r1_distill_7b": problem["difficulty"]
        }