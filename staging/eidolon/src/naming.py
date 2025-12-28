#!/usr/bin/env python3
"""
Experiment naming utilities for Eidolon.

Generates experiment numbers and random titles for YouTube upload naming.
"""

import hashlib
from pathlib import Path
from faker import Faker


def get_counter_file() -> str:
    """Get path to experiment counter file."""
    return "outputs/experiment_counter.txt"


def get_next_experiment_number() -> int:
    """
    Read and increment experiment counter.

    Returns:
        Next experiment number (starting from 1)
    """
    counter_file = get_counter_file()

    # Read current number or default to 1
    try:
        with open(counter_file, 'r') as f:
            current = int(f.read().strip())
    except (FileNotFoundError, ValueError):
        current = 0

    # Increment
    next_num = current + 1

    # Save back to file
    Path(counter_file).parent.mkdir(parents=True, exist_ok=True)
    with open(counter_file, 'w') as f:
        f.write(str(next_num))

    return next_num


def generate_experiment_title(seed_text: str) -> str:
    """
    Generate random but deterministic title using source text as seed.
    Always starts with "ASMR".

    Args:
        seed_text: Source video filename (stem) to use as seed

    Returns:
        Random semi-coherent title (e.g., "ASMR Quantum Butterfly Manifesto")
    """
    # Create deterministic seed from source text
    seed_hash = hashlib.md5(seed_text.encode()).hexdigest()
    seed_int = int(seed_hash[:8], 16)  # Use first 8 hex chars as integer seed

    # Initialize Faker with seed for deterministic output
    fake = Faker()
    Faker.seed(seed_int)

    # Generate semi-coherent title patterns
    # Mix and match different Faker providers for variety
    patterns = [
        lambda: f"{fake.catch_phrase()}",  # "Innovative zero tolerance implementation"
        lambda: f"{fake.bs()}".title(),  # "Harness Real-Time Eyeballs"
        lambda: f"{fake.word().title()} {fake.word().title()} {fake.word().title()}",
        lambda: f"The {fake.word().title()} {fake.word().title()}",
        lambda: f"{fake.job().replace(',', '')}",  # Remove commas from job titles
    ]

    # Use seed to pick pattern deterministically
    pattern_idx = seed_int % len(patterns)
    random_part = patterns[pattern_idx]()

    # Sanitize random part for filesystem (remove special chars, limit length)
    random_part = random_part.replace('/', '-').replace('\\', '-')
    random_part = ''.join(c for c in random_part if c.isalnum() or c in ' -')
    random_part = ' '.join(random_part.split())  # Normalize whitespace
    random_part = random_part[:50]  # Limit length for filesystem compatibility

    # Prepend "ASMR" to every title
    title = f"ASMR {random_part}"

    return title


def format_experiment_filename(exp_number: int, exp_title: str) -> str:
    """
    Format experiment filename with number and title.
    Filename matches exact YouTube title format.

    Args:
        exp_number: Experiment number
        exp_title: Random generated title (already includes "ASMR" prefix)

    Returns:
        Formatted filename: "Experiment #N: Title.mlt"
    """
    return f"Experiment #{exp_number}: {exp_title}.mlt"


def get_experiment_path(video_name: str) -> str:
    """
    Generate full experiment path for MLT project.
    Filename = exact YouTube upload title.

    If a project with the same title already exists (ignoring experiment number),
    modifies the seed to generate a new deterministic title.

    Args:
        video_name: Source video filename (stem)

    Returns:
        Full path: "outputs/projects/Experiment #N: ASMR Title.mlt"
    """
    import os

    projects_dir = "outputs/projects"

    # Ensure directory exists
    Path(projects_dir).mkdir(parents=True, exist_ok=True)

    # Get existing project titles (without the "Experiment #N:" prefix)
    existing_titles = set()
    if os.path.exists(projects_dir):
        for filename in os.listdir(projects_dir):
            if filename.endswith('.mlt'):
                # Extract title part after "Experiment #N: "
                if ': ' in filename:
                    title_part = filename.split(': ', 1)[1]  # Get everything after first ": "
                    # Remove .mlt extension
                    title_part = title_part.rsplit('.mlt', 1)[0]
                    existing_titles.add(title_part)

    # Generate unique title by appending suffix to seed if needed
    seed = video_name
    suffix_counter = 0

    while True:
        # Generate title with current seed
        exp_title = generate_experiment_title(seed)

        # Check if this title already exists
        if exp_title not in existing_titles:
            # Unique title found!
            break

        # Title exists, modify seed and try again
        suffix_counter += 1
        seed = f"{video_name}_{suffix_counter}"

    # Get experiment number
    exp_num = get_next_experiment_number()

    # Format filename
    filename = format_experiment_filename(exp_num, exp_title)

    return f"{projects_dir}/{filename}"
