import re


def text_formatter(text):
    """
    Convert single newlines to double newlines between paragraphs while preserving
    existing formatting with multiple newlines.

    A paragraph boundary is identified by:
    1. End of line is a letter, number, punctuation, or quote
    2. Start of next line is a capital letter
    3. Start of next line is NOT a list marker, indentation, or code-like content

    Args:
        text (str): The input text to reformat

    Returns:
        str: Reformatted text with appropriate double newlines
    """

    # First, preserve existing multiple newlines (2 or more)
    # Use regex to match and replace sequences of 2 or more newlines
    text = re.sub(
        r"\n{2,}",
        lambda m: "\n" + "__NEWLINE_" + str(len(m.group()) - 1) + "__" + "\n",
        text,
    )

    # Special case for lines ending with triple backticks
    # This specifically handles code block endings
    backtick_pattern = r"(```)\n(?![ \t]|[-*•+] |[0-9]+[\.\)] )([A-Z])"
    backtick_replacement = r"\1\n\n\2"
    text = re.sub(backtick_pattern, backtick_replacement, text)

    # Define the pattern for paragraph boundaries
    # Look for:
    # 1. One of these characters at the end: letter, number, common punctuation, quote, parenthesis
    # 2. Followed by a single newline
    # 3. NOT followed by indentation, list markers, or code keywords
    # 4. Followed by an uppercase letter
    pattern = r'([a-zA-Z0-9.,;:!?"\')])(\n)(?![ \t]|[-*•+] |[0-9]+[\.\)] |def |class |if |for |while |import |from |try |except |finally |with |async |await )([A-Z])'

    # Replace with the same characters but with double newline
    replacement = r"\1\n\n\3"

    # Perform the replacement
    reformatted_text = re.sub(pattern, replacement, text)

    # Restore original multiple newlines
    reformatted_text = re.sub(
        r"\n__NEWLINE_(\d+)__\n",
        lambda m: "\n" * (int(m.group(1)) + 1),
        reformatted_text,
    )

    return reformatted_text


if __name__ == "__main__":
    # Test cases

    # Case 1: Basic paragraph separation (should convert)
    test1 = "This is the first paragraph.\nThis is the second paragraph with a capital letter."
    expected1 = "This is the first paragraph.\n\nThis is the second paragraph with a capital letter."

    # Case 2: Multiple paragraphs with different endings (should convert)
    test2 = "First paragraph ends with period.\nSecond paragraph ends with exclamation!\nThird paragraph ends with question?\nFourth paragraph ends with a number 42.\nFifth paragraph."
    expected2 = "First paragraph ends with period.\n\nSecond paragraph ends with exclamation!\n\nThird paragraph ends with question?\n\nFourth paragraph ends with a number 42.\n\nFifth paragraph."

    # Case 3: Paragraph ending with quotes (should convert)
    test3 = 'He said, "This is a quote."\nThe next paragraph begins here.'
    expected3 = 'He said, "This is a quote."\n\nThe next paragraph begins here.'

    # Case 4: Unordered list items (should not convert)
    test4 = "List items:\n- Item one\n- Item two\n- Item three"
    expected4 = test4  # No change expected

    # Case 5: Ordered list with period format (should not convert)
    test5 = "Ordered list:\n1. First item\n2. Second item\n3. Third item"
    expected5 = test5  # No change expected

    # Case 6: Code block with indentation (should not convert)
    test6 = "Python code:\ndef hello():\n    print('Hello')\n    return None"
    expected6 = test6  # No change expected

    # Case 7: Sentence not starting with capital (should not convert)
    test7 = (
        "This is a sentence.\nlowercase beginning shouldn't trigger a double newline."
    )
    expected7 = test7  # No change expected

    # Case 8: Text already with double newlines (should not add more)
    test8 = "This paragraph has proper formatting.\n\nThis one too.\n\nAnd this one."
    expected8 = test8  # No change expected

    # Case 9: Mixed cases
    test9 = "Regular paragraph.\nNew paragraph starts here.\n- List item 1\n- List item 2\nAnother paragraph after list.\ndef code():\n    return True\nFinal paragraph."
    expected9 = "Regular paragraph.\n\nNew paragraph starts here.\n- List item 1\n- List item 2\n\nAnother paragraph after list.\ndef code():\n    return True\n\nFinal paragraph."

    # Case 10: List with capitalized items (should not convert)
    test10 = "Important points:\n- The first point\n- Another critical point\n- The final consideration"
    expected10 = test10  # No change expected

    # Case 11: Bullet points with different markers
    test11 = "Different markers:\n• First bullet\n* Second bullet\n+ Third bullet"
    expected11 = test11  # No change expected

    # Case 12: Simple variable assignments (should not convert)
    test12 = "Variable examples:\nmy_var = 100\ntotal = my_var + 50"
    expected12 = test12  # No change expected

    # Case 13: After backticks - minimal handling
    test13 = "Some code example:\n```\ndef hello():\n    print('Hello')\n```\nAnd here is more text."
    expected13 = "Some code example:\n```\ndef hello():\n    print('Hello')\n```\n\nAnd here is more text."

    # Case 14: Ordered list with parenthesis format (should not convert)
    test14 = "Another list format:\n1) First item\n2) Second item\n3) Third item"
    expected14 = test14  # No change expected

    # Case 15: Subtle indentation (should not convert)
    test15 = "Consider this code:\n var = 100\n  slightly_indented = 200"
    expected15 = test15  # No change expected

    # Case 16: Triple newlines should be preserved
    test16 = "First paragraph.\n\n\nSecond paragraph after triple newline."
    expected16 = "First paragraph.\n\n\nSecond paragraph after triple newline."

    # Case 17: Mix of single, double, triple, and quad newlines
    test17 = "Line one.\nLine two with single newline.\n\nLine three after double.\n\n\nLine four after triple.\n\n\n\nLine five after quad."
    expected17 = "Line one.\n\nLine two with single newline.\n\nLine three after double.\n\n\nLine four after triple.\n\n\n\nLine five after quad."

    # Run tests
    tests = [
        (test1, expected1, "Basic paragraph test"),
        (test2, expected2, "Multiple paragraphs test"),
        (test3, expected3, "Quote ending test"),
        (test4, expected4, "Unordered list test"),
        (test5, expected5, "Ordered list period format test"),
        (test6, expected6, "Code block test"),
        (test7, expected7, "Lowercase beginning test"),
        (test8, expected8, "Already formatted test"),
        (test9, expected9, "Mixed content test"),
        (test10, expected10, "Capitalized list items test"),
        (test11, expected11, "Bullet points test"),
        (test12, expected12, "Variable assignment test"),
        (test13, expected13, "After backticks test"),
        (test14, expected14, "Ordered list parenthesis format test"),
        (test15, expected15, "Subtle indentation test"),
        (test16, expected16, "Triple newline preservation test"),
        (test17, expected17, "Mixed newline sequences test"),
    ]

    for i, (test, expected, name) in enumerate(tests, 1):
        result = text_formatter(test)
        success = result == expected
        print(f"Test {i} ({name}): {'PASS' if success else 'FAIL'}")
        if not success:
            print(f"Expected:\n{repr(expected)}")
            print(f"Got:\n{repr(result)}")
            print("Differences:")
            for j, (e, r) in enumerate(zip(expected, result)):
                if e != r:
                    print(f"Position {j}: expected '{e}', got '{r}'")
            if len(expected) != len(result):
                print(f"Length difference: expected {len(expected)}, got {len(result)}")
