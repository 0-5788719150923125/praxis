import pytest

from builders import text_formatter

# Define test cases as tuples of (input, expected_output, description)
TEST_CASES = [
    # Case 1: Basic paragraph separation (should convert)
    (
        "This is the first paragraph.\nThis is the second paragraph with a capital letter.",
        "This is the first paragraph.\n\nThis is the second paragraph with a capital letter.",
        "Basic paragraph test",
    ),
    # Case 2: Multiple paragraphs with different endings (should convert)
    (
        "First paragraph ends with period.\nSecond paragraph ends with exclamation!\nThird paragraph ends with question?\nFourth paragraph ends with a number 42.\nFifth paragraph.",
        "First paragraph ends with period.\n\nSecond paragraph ends with exclamation!\n\nThird paragraph ends with question?\n\nFourth paragraph ends with a number 42.\n\nFifth paragraph.",
        "Multiple paragraphs test",
    ),
    # Case 3: Paragraph ending with quotes (should convert)
    (
        'He said, "This is a quote."\nThe next paragraph begins here.',
        'He said, "This is a quote."\n\nThe next paragraph begins here.',
        "Quote ending test",
    ),
    # Case 4: Unordered list items (should not convert)
    (
        "List items:\n- Item one\n- Item two\n- Item three",
        "List items:\n\n- Item one\n- Item two\n- Item three",
        "Unordered list test",
    ),
    # Case 5: Ordered list with period format (should not convert)
    (
        "Ordered list:\n1. First item\n2. Second item\n3. Third item",
        "Ordered list:\n\n1. First item\n2. Second item\n3. Third item",
        "Ordered list period format test",
    ),
    # Case 6: Code block with indentation (should not convert)
    (
        "Python code:\ndef hello():\n    print('Hello')\n    return None",
        "Python code:\ndef hello():\n    print('Hello')\n    return None",
        "Code block test",
    ),
    # Case 7: Sentence not starting with capital (should not convert)
    (
        "This is a sentence.\nlowercase beginning shouldn't trigger a double newline.",
        "This is a sentence.\nlowercase beginning shouldn't trigger a double newline.",
        "Lowercase beginning test",
    ),
    # Case 8: Text already with double newlines (should not add more)
    (
        "This paragraph has proper formatting.\n\nThis one too.\n\nAnd this one.",
        "This paragraph has proper formatting.\n\nThis one too.\n\nAnd this one.",
        "Already formatted test",
    ),
    # Case 9: Mixed cases
    (
        "Regular paragraph.\nNew paragraph starts here.\n- List item 1\n- List item 2\nAnother paragraph after list.\ndef code():\n    return True\nFinal paragraph.",
        "Regular paragraph.\n\nNew paragraph starts here.\n\n- List item 1\n- List item 2\n\nAnother paragraph after list.\ndef code():\n    return True\n\nFinal paragraph.",
        "Mixed content test",
    ),
    # Case 10: List with capitalized items (should not convert)
    (
        "Important points:\n- The first point\n- Another critical point\n- The final consideration",
        "Important points:\n\n- The first point\n- Another critical point\n- The final consideration",
        "Capitalized list items test",
    ),
    # Case 11: Bullet points with different markers
    # (
    #     "Different markers:\n• First bullet\n* Second bullet\n+ Third bullet",
    #     "Different markers:\n\n• First bullet\n* Second bullet\n+ Third bullet",
    #     "Bullet points test",
    # ),
    # Case 12: Simple variable assignments (should not convert)
    (
        "Variable examples:\nmy_var = 100\ntotal = my_var + 50",
        "Variable examples:\nmy_var = 100\ntotal = my_var + 50",
        "Variable assignment test",
    ),
    # Case 13: After backticks - minimal handling
    (
        "Some code example:\n```\ndef hello():\n    print('Hello')\n```\nAnd here is more text.",
        "Some code example:\n```\ndef hello():\n    print('Hello')\n```\n\nAnd here is more text.",
        "After backticks test",
    ),
    # Case 14: Ordered list with parenthesis format (should not convert)
    (
        "Another list format:\n1) First item\n2) Second item\n3) Third item",
        "Another list format:\n\n1) First item\n2) Second item\n3) Third item",
        "Ordered list parenthesis format test",
    ),
    # Case 15: Subtle indentation (should not convert)
    (
        "Consider this code:\n var = 100\n  slightly_indented = 200",
        "Consider this code:\n var = 100\n  slightly_indented = 200",
        "Subtle indentation test",
    ),
    # Case 16: Triple newlines should be preserved
    (
        "First paragraph.\n\n\nSecond paragraph after triple newline.",
        "First paragraph.\n\n\nSecond paragraph after triple newline.",
        "Triple newline preservation test",
    ),
]


@pytest.mark.parametrize("input_text,expected,description", TEST_CASES)
def test_text_formatter(input_text, expected, description):
    """Test the text_formatter function with various test cases."""
    result = text_formatter(input_text)

    # For debugging failures
    if result != expected:
        print(f"\nTest failed: {description}")
        print(f"Expected:\n{repr(expected)}")
        print(f"Got:\n{repr(result)}")

        # Show differences character by character
        for i, (e, r) in enumerate(zip(expected, result)):
            if e != r:
                print(f"Position {i}: expected '{e}', got '{r}'")

        # Check for length differences
        if len(expected) != len(result):
            print(f"Length difference: expected {len(expected)}, got {len(result)}")

    assert result == expected, f"Test failed: {description}"
