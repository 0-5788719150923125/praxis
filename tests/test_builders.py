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
        "Regular paragraph.\n\nNew paragraph starts here.\n\n- List item 1\n- List item 2\nAnother paragraph after list.\ndef code():\n    return True\nFinal paragraph.",
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
    # Case 16: 3 or more newlines should be collapsed into 2
    (
        "First paragraph.\n\n\nSecond paragraph after triple newline.",
        "First paragraph.\n\nSecond paragraph after triple newline.",
        "Triple newline collapsed test",
    ),
    # Case 17: Mix of single, double, triple, and quad newlines
    (
        "Line one.\nLine two with single newline.\n\nLine three after double.\n\n\nLine four after triple.\n\n\n\nLine five after quad.",
        "Line one.\n\nLine two with single newline.\n\nLine three after double.\n\nLine four after triple.\n\nLine five after quad.",
        "Mixed newline sequences test",
    ),
    # Case 18: Paragraph starting with quotation marks (should convert)
    (
        'SmolLM2 is a family of compact language models available in three size: 135M, 360M, and 1.7B parameters. They are capable of solving a wide range of tasks while being lightweight enough to run on-device.\n"SmolLM2 demonstrates significant advances over its predecessor SmolLM1, particularly in instruction following, knowledge, reasoning."',
        'SmolLM2 is a family of compact language models available in three size: 135M, 360M, and 1.7B parameters. They are capable of solving a wide range of tasks while being lightweight enough to run on-device.\n\n"SmolLM2 demonstrates significant advances over its predecessor SmolLM1, particularly in instruction following, knowledge, reasoning."',
        "Paragraph starting with quotation marks test",
    ),
    # Case 19: Multiple paragraphs with quotation marks (should convert)
    (
        "First paragraph ends normally.\n\"Second paragraph starts with quotes.\" And continues.\n'Third paragraph uses single quotes.' And also continues.",
        "First paragraph ends normally.\n\n\"Second paragraph starts with quotes.\" And continues.\n\n'Third paragraph uses single quotes.' And also continues.",
        "Multiple paragraphs with quotation marks test",
    ),
    # Case 20: Paragraphs following a quote (should convert)
    (
        '"This is a test."\nBut this is also a test.',
        '"This is a test."\n\nBut this is also a test.',
        "Paragraph following a quote test",
    ),
    # Case 21: Tag squeezing - remove blank lines between tags and content
    (
        "<thinking>\n\nThis is my thought process.\nI need to analyze this problem.\n\n</thinking>",
        "<thinking>\nThis is my thought process.\n\nI need to analyze this problem.\n</thinking>",
        "Tag squeezing test",
    ),
    # Case 22: Multiple tag squeezing with different tags
    (
        "<output>\n\nThis is the final answer.\nIt should be formatted properly.\n\n</output>\n\n<step_by_step>\n\nStep 1: Identify the problem\nStep 2: Solve it\n\n</step_by_step>",
        "<output>\nThis is the final answer.\n\nIt should be formatted properly.\n</output>\n\n<step_by_step>\nStep 1: Identify the problem\nStep 2: Solve it\n</step_by_step>",
        "Multiple tag squeezing test",
    ),
    # Case 23: Nested tags should be handled correctly
    (
        "<thinking>\n\nI need to think about this.\n\n<step_by_step>\n\nStep 1: First step\nStep 2: Second step\n\n</step_by_step>\n\nNow I have my answer.\n\n</thinking>",
        "<thinking>\nI need to think about this.\n\n<step_by_step>\nStep 1: First step\nStep 2: Second step\n</step_by_step>\n\nNow I have my answer.\n</thinking>",
        "Nested tag squeezing test",
    ),
    # Case 24: Structured data with colons should not get paragraph breaks
    (
        "Feeling: Good\nEmotional Response: Satisfied\nConfidence: High",
        "Feeling: Good\nEmotional Response: Satisfied\nConfidence: High",
        "Structured data with colons test",
    ),
    # Case 25: Mixed structured data and paragraphs
    (
        "Here is some introduction text. This explains the concept.\nName: John Doe\nAge: 30\nStatus: Active\nThis is a concluding paragraph. It wraps up the discussion.",
        "Here is some introduction text. This explains the concept.\n\nName: John Doe\nAge: 30\nStatus: Active\nThis is a concluding paragraph. It wraps up the discussion.",
        "Mixed structured data and paragraphs test",
    ),
    # Case 26: Numbers and scores should not get paragraph breaks
    (
        "Final results are in. Here are the scores.\nPlayer 1: 95\nPlayer 2: 87\nGame Mode: Tournament\nThe competition was fierce! Everyone played well.",
        "Final results are in. Here are the scores.\n\nPlayer 1: 95\nPlayer 2: 87\nGame Mode: Tournament\nThe competition was fierce! Everyone played well.",
        "Numbers and scores test",
    ),
    # Case 27: Lines ending without sentence punctuation should stay together
    (
        "Configuration Settings\nDatabase Host: localhost\nPort Number: 5432\nUsername: admin\nConnection Timeout: 30\nThese settings are important for the application.",
        "Configuration Settings\nDatabase Host: localhost\nPort Number: 5432\nUsername: admin\nConnection Timeout: 30\nThese settings are important for the application.",
        "Configuration without punctuation test",
    ),
    # Case 28: Consecutive tags should be squished together
    (
        "<conscious_thought>\nI am writing a test.\n</conscious_thought>\n<step_by_step>\n(the steps)\n</step_by_step>",
        "<conscious_thought>\nI am writing a test.\n</conscious_thought>\n<step_by_step>\n(the steps)\n</step_by_step>",
        "Consecutive tags squishing test",
    ),
    # Case 29: Multiple consecutive tags should all stay together
    (
        "<thinking>\nFirst thought\n</thinking>\n<analysis>\nAnalysis here\n</analysis>\n<conclusion>\nFinal thought\n</conclusion>",
        "<thinking>\nFirst thought\n</thinking>\n<analysis>\nAnalysis here\n</analysis>\n<conclusion>\nFinal thought\n</conclusion>",
        "Multiple consecutive tags test",
    ),
    # Case 30: Tags followed by regular content should get proper spacing
    (
        "<conscious_thought>\nI am thinking.\n</conscious_thought>\n<step_by_step>\nStep 1: Do this\n</step_by_step>\nThis is regular content after tags.",
        "<conscious_thought>\nI am thinking.\n</conscious_thought>\n<step_by_step>\nStep 1: Do this\n</step_by_step>\n\nThis is regular content after tags.",
        "Tags followed by content test",
    ),
    # Case 31: Mixed tags and structured data
    (
        "<analysis>\nFeeling: Good\nConfidence: High\n</analysis>\n<next_steps>\nAction: Review data\nDeadline: Tomorrow\n</next_steps>\nThe analysis is complete. We can proceed.",
        "<analysis>\nFeeling: Good\nConfidence: High\n</analysis>\n<next_steps>\nAction: Review data\nDeadline: Tomorrow\n</next_steps>\n\nThe analysis is complete. We can proceed.",
        "Mixed tags and structured data test",
    ),
    # Case 32: Curly quotes after sentence (double quotes)
    (
        'This is a regular sentence.\n"This starts with curly left double quote."',
        'This is a regular sentence.\n\n"This starts with curly left double quote."',
        "Curly left double quote after sentence test",
    ),
    # Case 33: Curly quotes after sentence (single quotes)
    (
        "This is a regular sentence.\n'This starts with curly left single quote.'",
        "This is a regular sentence.\n\n'This starts with curly left single quote.'",
        "Curly left single quote after sentence test",
    ),
    # Case 34: Curly quotes at end of sentence
    (
        'He said "this is with curly quotes."\nNext paragraph should have double newline.',
        'He said "this is with curly quotes."\n\nNext paragraph should have double newline.',
        "Curly right double quote at end test",
    ),
    # Case 35: Curly single quotes at end
    (
        "He said 'this is with curly single quotes.'\nNext paragraph here.",
        "He said 'this is with curly single quotes.'\n\nNext paragraph here.",
        "Curly right single quote at end test",
    ),
    # Case 36: Mixed curly and straight quotes
    (
        """First with straight quotes.
"Second with curly quotes."
'Third with straight single.'
'Fourth with curly single.'""",
        """First with straight quotes.

"Second with curly quotes."

'Third with straight single.'

'Fourth with curly single.'""",
        "Mixed quote types test",
    ),
    # Case 37: Curly quotes after code blocks
    (
        '```\ncode block\n```\n"Text starting with curly quote after code."',
        '```\ncode block\n```\n\n"Text starting with curly quote after code."',
        "Curly quote after code block test",
    ),
    # Case 38: Curly quotes with parentheses/brackets
    (
        'End of quote.")\nNext paragraph starts here.',
        'End of quote.")\n\nNext paragraph starts here.',
        "Curly quote with closing parenthesis test",
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
