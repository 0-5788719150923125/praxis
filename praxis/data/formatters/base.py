"""Base formatting utilities used by all formatters."""

import re
from typing import Dict, List
from transformers import PreTrainedTokenizer

from praxis.data.config import SYSTEM_PROMPT, DEVELOPER_PROMPTS


def text_formatter(text):
    """
    Convert single newlines to double newlines between paragraphs while preserving
    existing formatting with multiple newlines.

    A paragraph boundary is identified by:
    1. End of line is a letter, number, punctuation, or quote
    2. Start of next line is a capital letter (possibly preceded by quotes)
    3. Start of next line is NOT a list marker, indentation, or code-like content

    Special handling for tags:
    - Tags should "squeeze" their content (no double newlines between tags and content)
    - Pattern: <tag>\n becomes <tag> and \n</tag> becomes </tag>

    Args:
        text (str): The input text to reformat

    Returns:
        str: Reformatted text with appropriate double newlines
    """

    text = add_newline_before_lists(text)

    # First, preserve existing multiple newlines (2 or more)
    # Use regex to match and replace sequences of 2 or more newlines
    text = re.sub(
        r"\n{2,}",
        lambda m: "\n" + "__NEWLINE_" + str(len(m.group()) - 1) + "__" + "\n",
        text,
    )

    # Special case for lines ending with triple backticks
    # This specifically handles code block endings
    backtick_pattern = r"(```)\n(?![ \t]|[-*•+] |[0-9]+[.\)] )([\"\'" "'']*[A-Z])"
    backtick_replacement = r"\1\n\n\2"
    text = re.sub(backtick_pattern, backtick_replacement, text)

    # Define the pattern for paragraph boundaries
    # Look for:
    # 1. Lines ending with sentence-ending punctuation (., !, ?) - these are likely complete thoughts
    # 2. Followed by a single newline
    # 3. NOT followed by indentation, list markers, or code keywords
    # 4. Followed by an optional quotation mark and then an uppercase letter
    pattern_basic = (
        r"([.!?][\"\'"
        "'']*[)\\]]*)(\n)(?![ \t]|[-*•+] |[0-9]+[.\\)] |def |class |if |for |while |import |from |try |except |finally |with |async |await )([\"'"
        "'']*[A-Z])"
    )

    # Separate pattern for colons: Include them but exclude structured data patterns (word: value)
    pattern_colon = (
        r"(:[\"\'"
        "'']*[)\\]]*)(\n)(?![ \t]|[-*•+] |[0-9]+[.\\)] |def |class |if |for |while |import |from |try |except |finally |with |async |await |[A-Za-z][^:\n]*: )([\"'"
        "'']*[A-Z])"
    )

    # Replace with the same characters but with double newline
    replacement = r"\1\n\n\3"

    # Perform the replacements - first basic punctuation, then colons
    reformatted_text = re.sub(pattern_basic, replacement, text)
    reformatted_text = re.sub(pattern_colon, replacement, reformatted_text)

    # Restore original multiple newlines, but collapse 3+ newlines to 2
    reformatted_text = re.sub(
        r"\n__NEWLINE_(\d+)__\n",
        lambda m: "\n\n" if int(m.group(1)) >= 2 else "\n" * (int(m.group(1)) + 1),
        reformatted_text,
    )

    # Handle tag squeezing AFTER paragraph formatting: remove blank lines between tags and content
    # Tags remain on their own lines, but extra spacing is removed

    # Remove blank lines after opening tags: <tag>\n\ncontent becomes <tag>\ncontent
    tag_squeeze_pattern = r"(<[^/>]+>)\n\n+"  # Opening tag followed by blank lines
    reformatted_text = re.sub(tag_squeeze_pattern, r"\1\n", reformatted_text)

    # Remove blank lines before closing tags: content\n\n</tag> becomes content\n</tag>
    tag_squeeze_pattern_close = r"\n\n+(</[^>]+>)"  # Blank lines before closing tag
    reformatted_text = re.sub(tag_squeeze_pattern_close, r"\n\1", reformatted_text)

    # Ensure closing tags are followed by double newlines when there's content after them
    # </tag>\ncontent becomes </tag>\n\ncontent (but preserve existing double newlines)
    # BUT: Keep consecutive tags together (don't add space between </tag> and <tag>)
    tag_after_pattern = r"(</[^>]+>)\n(?!\n|<|$)"  # Closing tag followed by single newline and content (not another tag or end)
    reformatted_text = re.sub(tag_after_pattern, r"\1\n\n", reformatted_text)

    return reformatted_text


def add_newline_before_lists(text):
    """Add a newline before list items if they aren't already preceded by one."""
    # Define patterns for list items
    list_patterns = [
        r"^\s*- ",  # Bullet points
        r"^\s*\d+\.\s",  # Numbered lists with dot
        r"^\s*\d+\)\s*",  # Numbered lists with parenthesis
        r"^\s*\d+#\s*",  # Hash notation as mentioned in example
    ]

    # Function to check if a line is a list item
    def is_list_item(line):
        return any(re.match(pattern, line) for pattern in list_patterns)

    # Split the text into lines
    lines = text.split("\n")
    if not lines:  # Handle empty text
        return ""

    # Process the text
    result = []
    i = 0
    while i < len(lines):
        result.append(lines[i])

        # Check if we need to add a newline before a list item
        if i < len(lines) - 1:
            current = lines[i]
            next_line = lines[i + 1]

            # If current line has content and is not a list item,
            # and next line is a list item
            if (
                current.strip()
                and not is_list_item(current)
                and is_list_item(next_line)
            ):

                # Check if there's already a blank line between them
                if next_line.strip():  # This means there's only one newline separator
                    result.append("")  # Add a blank line

        i += 1

    return "\n".join(result)


def repair_text_punctuation(text: str) -> str:
    """First pass: Fix punctuation and spacing issues."""
    # Fix common typos/mangled words from the dataset
    # "ll" at start of sentence is often "lol" that got mangled
    text = re.sub(r"^ll\b", "lol", text, flags=re.IGNORECASE)
    text = re.sub(r"([.!?]\s+)ll\b", r"\1lol", text, flags=re.IGNORECASE)

    # Special case: Fix ". . . ? d" pattern (common in dataset)
    # This becomes "...? :D"
    text = re.sub(r"\.\s+\.\s+\.\s+\?\s+d$", r"...? :D", text, flags=re.IGNORECASE)

    # Pass 1: Fix broken emoticons (before any other punctuation fixes)
    # Common patterns where emoticons got split with spaces
    text = re.sub(r":\s+([dDpPsS)])", r":\1", text)  # ": d" -> ":D"
    text = re.sub(r";\s+([dDpPsS)])", r";\1", text)  # "; d" -> ";D"
    text = re.sub(r"x\s+d\b", r"xD", text, flags=re.IGNORECASE)  # "x d" -> "xD"
    # Fix standalone "? d" or ". d" at end
    text = re.sub(r"\?\s+d$", r"? :D", text, flags=re.IGNORECASE)
    text = re.sub(r"\.\s+d$", r". :D", text, flags=re.IGNORECASE)

    # Pass 2: Collapse spaced punctuation sequences
    text = re.sub(r"([.!?])\s+([.!?])", r"\1\2", text)  # ". ." -> ".."
    text = re.sub(r"([.!?])\1{3,}", r"\1\1\1", text)  # Limit to max 3 repetitions

    # Pass 3: Fix spacing around punctuation (but preserve emoticons)
    # Don't collapse spaces before : or ; that are part of emoticons
    text = re.sub(r"\s+([,.])", r"\1", text)  # Remove spaces before comma and period
    text = re.sub(
        r"\s+([!?])(?!\s*:)", r"\1", text
    )  # Remove before ! and ? unless followed by :
    text = re.sub(
        r"\s+([:;])(?![DPSdps)(/])", r"\1", text
    )  # Remove before : and ; unless emoticon
    text = re.sub(r"([,;:])(?=[^\s])", r"\1 ", text)  # Add space after punctuation
    text = re.sub(r"\s+", " ", text)  # Collapse multiple spaces

    return text


def repair_broken_emoticons(text: str) -> str:
    """Second pass: Fix emoticons that got mangled by first pass."""
    # Fix patterns where emoticons got merged with punctuation
    # "...?:D" should be "...? :D"
    text = re.sub(r"([.!?]):([DPSdps)])", r"\1 :\2", text)
    text = re.sub(r"([.!?]);([DPSdps)])", r"\1 ;\2", text)

    # Uppercase emoticon letters (with optional space before)
    text = re.sub(r":\s*d\b", ":D", text, flags=re.IGNORECASE)
    text = re.sub(r":\s*p\b", ":P", text, flags=re.IGNORECASE)
    text = re.sub(r":\s*s\b", ":S", text, flags=re.IGNORECASE)
    text = re.sub(r";\s*d\b", ";D", text, flags=re.IGNORECASE)
    text = re.sub(r";\s*p\b", ";P", text, flags=re.IGNORECASE)
    text = re.sub(r"x\s*d\b", "xD", text, flags=re.IGNORECASE)

    # Fix isolated ": d" patterns that might remain
    text = re.sub(
        r":\s+([dps])\b", lambda m: ":" + m.group(1).upper(), text, flags=re.IGNORECASE
    )
    text = re.sub(
        r";\s+([dps])\b", lambda m: ";" + m.group(1).upper(), text, flags=re.IGNORECASE
    )

    return text


def simple_truecase(text: str) -> str:
    """Simple truecasing to capitalize sentence starts."""
    if not text:
        return text

    # Capitalize first letter
    result = text[0].upper() + text[1:] if text else text

    # Capitalize after sentence-ending punctuation
    result = re.sub(
        r"([.!?])\s+([a-z])", lambda m: m.group(1) + " " + m.group(2).upper(), result
    )

    # Common words that should stay lowercase mid-sentence
    lowercase_words = {
        "a",
        "an",
        "the",
        "and",
        "but",
        "or",
        "for",
        "nor",
        "on",
        "at",
        "to",
        "by",
        "up",
        "in",
        "out",
        "if",
        "is",
        "it",
        "of",
        "as",
        "was",
        "with",
        "be",
        "are",
        "been",
        "were",
        "that",
        "this",
        "these",
        "those",
        "some",
        "all",
        "no",
        "not",
        "can",
        "will",
        "may",
        "would",
        "could",
        "should",
        "might",
        "must",
        "shall",
        "has",
        "have",
        "had",
        "do",
        "does",
        "did",
        "so",
        "yet",
        "from",
    }

    # Split into sentences and process each
    sentences = re.split(r"([.!?]\s+)", result)
    processed = []

    for i, part in enumerate(sentences):
        if i % 2 == 0 and part:  # This is a sentence, not punctuation
            words = part.split()
            if words:
                # First word stays capitalized
                new_words = [words[0]]
                # Process remaining words
                for word in words[1:]:
                    if word.lower() in lowercase_words and not word[0].isupper():
                        new_words.append(word.lower())
                    else:
                        new_words.append(word)
                processed.append(" ".join(new_words))
        else:
            processed.append(part)

    return "".join(processed)
