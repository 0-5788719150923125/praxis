import re
from difflib import SequenceMatcher
from typing import Callable, List, Optional, Tuple


class TextPatternFinder:
    def __init__(self):
        """Initialize the text pattern finder with common transformations."""
        self.transformations = [
            ("Identity", lambda x: x),
            ("Uppercase", str.upper),
            ("Lowercase", str.lower),
            ("Capitalize", str.capitalize),
            ("Reverse", lambda x: x[::-1]),
            ("Remove Spaces", lambda x: x.replace(" ", "")),
            ("Double Spaces", lambda x: " ".join(x.split())),
            ("Title Case", str.title),
            ("Swap Case", str.swapcase),
        ]

    def _similarity_score(self, s1: str, s2: str) -> float:
        """Calculate similarity between two strings."""
        return SequenceMatcher(None, s1, s2).ratio()

    def _find_word_patterns(self, input_text: str, output_text: str) -> Optional[str]:
        """Look for word-level patterns."""
        input_words = input_text.split()
        output_words = output_text.split()

        if len(input_words) == len(output_words):
            mappings = []
            for i, o in zip(input_words, output_words):
                mappings.append(f"'{i}' → '{o}'")
            return "Word substitution: " + ", ".join(mappings)

        return None

    def find_pattern(
        self, input_text: str, output_text: str, threshold: float = 0.8
    ) -> List[str]:
        """
        Find possible patterns that transform input_text to output_text.

        Args:
            input_text: The input string
            output_text: The target output string
            threshold: Minimum similarity score to consider a match

        Returns:
            List of possible transformation descriptions
        """
        patterns = []

        # Try basic transformations
        for name, transform in self.transformations:
            try:
                result = transform(input_text)
                similarity = self._similarity_score(result, output_text)
                if similarity > threshold:
                    patterns.append(
                        f"{name} transformation (similarity: {similarity:.2f})"
                    )
            except Exception as e:
                continue

        # Check for word patterns
        word_pattern = self._find_word_patterns(input_text, output_text)
        if word_pattern:
            patterns.append(word_pattern)

        # Length-based patterns
        if len(input_text) == len(output_text):
            char_mappings = []
            for i, o in zip(input_text, output_text):
                if i != o:
                    char_mappings.append(f"'{i}' → '{o}'")
            if char_mappings:
                patterns.append("Character substitution: " + ", ".join(char_mappings))

        return patterns


def test_pattern_finder():
    """Test the pattern finder with various examples."""
    finder = TextPatternFinder()

    test_cases = [
        {
            "input": "the quick brown fox",
            "output": "jumped over a lazy dog",
            "description": "Complete phrase substitution",
        },
        {
            "input": "hello world",
            "output": "HELLO WORLD",
            "description": "Simple uppercase transformation",
        },
        {
            "input": "abc def",
            "output": "fed cba",
            "description": "Reverse transformation",
        },
    ]

    print("Text Pattern Finder Test Results")
    print("===============================")

    for case in test_cases:
        print(f"\nTest Case: {case['description']}")
        print(f"INPUT: {case['input']}")
        print(f"OUTPUT: {case['output']}")

        patterns = finder.find_pattern(case["input"], case["output"])

        print("\nPossible Patterns Found:")
        if patterns:
            for i, pattern in enumerate(patterns, 1):
                print(f"{i}. {pattern}")
        else:
            print("No clear patterns found")
        print("-" * 50)


if __name__ == "__main__":
    test_pattern_finder()
