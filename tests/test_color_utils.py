"""Test color utility functions for agent freshness gradient."""


def test_step_based_freshness():
    """Test that freshness is calculated based on commit steps, not temporal distance."""

    print("\n" + "=" * 80)
    print("STEP-BASED FRESHNESS TEST")
    print("=" * 80)

    # Simulate 5 commits with real timestamps from git log
    commits = [
        {"hash": "1b62827", "timestamp": 1761875740},  # oldest
        {"hash": "404243e", "timestamp": 1761886566},
        {"hash": "032c57f", "timestamp": 1761890856},
        {"hash": "351b549", "timestamp": 1761946583},
        {
            "hash": "dd83544",
            "timestamp": 1761947377,
        },  # newest (only 13 min after previous!)
    ]

    # Get unique timestamps and sort
    unique_timestamps = sorted(set(c["timestamp"] for c in commits))

    print(f"\nFound {len(unique_timestamps)} unique commits")
    print("\nStep-based freshness calculation:")

    for commit in commits:
        position = unique_timestamps.index(commit["timestamp"])
        freshness = (
            position / (len(unique_timestamps) - 1)
            if len(unique_timestamps) > 1
            else 1.0
        )

        print(
            f"  {commit['hash']}: position {position}/{len(unique_timestamps)-1} → freshness = {freshness:.3f}"
        )

    # Verify evenly-spaced distribution
    assert len(unique_timestamps) == 5, "Should have 5 unique commits"

    # Check oldest
    oldest_pos = unique_timestamps.index(commits[0]["timestamp"])
    oldest_freshness = oldest_pos / 4
    assert oldest_freshness == 0.0, f"Oldest should be 0.0, got {oldest_freshness}"

    # Check newest
    newest_pos = unique_timestamps.index(commits[-1]["timestamp"])
    newest_freshness = newest_pos / 4
    assert newest_freshness == 1.0, f"Newest should be 1.0, got {newest_freshness}"

    # Check middle commit (should be exactly 0.5)
    middle_pos = unique_timestamps.index(commits[2]["timestamp"])
    middle_freshness = middle_pos / 4
    assert (
        middle_freshness == 0.5
    ), f"Middle commit should be 0.5, got {middle_freshness}"

    print("\n✓ All commits evenly distributed regardless of temporal distance")
    print("=" * 80 + "\n")


def test_color_interpolation():
    """Test that color interpolation works correctly."""

    # Simulate the color functions from state.js
    def rgb_to_greyscale(r, g, b):
        grey = round(0.2126 * r + 0.7152 * g + 0.0722 * b)
        return [grey, grey, grey]

    def lerp_color(color1, color2, t):
        t = max(0, min(1, t))
        return [round(c1 + (color2[i] - c1) * t) for i, c1 in enumerate(color1)]

    def hex_to_rgb(hex_str):
        cleaned = hex_str.replace("#", "")
        r = int(cleaned[0:2], 16)
        g = int(cleaned[2:4], 16)
        b = int(cleaned[4:6], 16)
        return [r, g, b]

    # Test with online status (green)
    primary_color = hex_to_rgb("#0B9A6D")  # [11, 154, 109]
    grey_color = rgb_to_greyscale(*primary_color)

    print("\n" + "=" * 80)
    print("COLOR INTERPOLATION TEST")
    print("=" * 80)

    print(
        f"\nBase color (green): rgb({primary_color[0]}, {primary_color[1]}, {primary_color[2]})"
    )
    print(f"Greyscale version: rgb({grey_color[0]}, {grey_color[1]}, {grey_color[2]})")

    # Test at different freshness values
    test_cases = [
        (0.0, "oldest commit - full greyscale"),
        (0.5, "middle commit - 50% blend"),
        (0.989, "near-newest commit (351b549)"),
        (1.0, "newest commit (dd83544) - full color"),
    ]

    print("\nInterpolation tests:")
    for freshness, description in test_cases:
        result = lerp_color(grey_color, primary_color, freshness)
        print(f"  freshness={freshness:.3f} ({description})")
        print(f"    → rgb({result[0]}, {result[1]}, {result[2]})")

    # Test with archived status (blue) in dark theme
    print("\n" + "-" * 80)
    archived_color_dark = hex_to_rgb("#5b8fc9")  # [91, 143, 201]
    archived_grey = rgb_to_greyscale(*archived_color_dark)

    print(
        f"\nArchived color (dark theme, blue): rgb({archived_color_dark[0]}, {archived_color_dark[1]}, {archived_color_dark[2]})"
    )
    print(
        f"Greyscale version: rgb({archived_grey[0]}, {archived_grey[1]}, {archived_grey[2]})"
    )

    print("\nInterpolation tests for archived:")
    for freshness, description in test_cases:
        result = lerp_color(archived_grey, archived_color_dark, freshness)
        print(f"  freshness={freshness:.3f} ({description})")
        print(f"    → rgb({result[0]}, {result[1]}, {result[2]})")

    # Verify that freshness=1.0 gives us back the original color
    full_color_online = lerp_color(grey_color, primary_color, 1.0)
    assert (
        full_color_online == primary_color
    ), f"Freshness 1.0 should return original color, got {full_color_online} != {primary_color}"

    full_color_archived = lerp_color(archived_grey, archived_color_dark, 1.0)
    assert (
        full_color_archived == archived_color_dark
    ), f"Freshness 1.0 should return original archived color, got {full_color_archived} != {archived_color_dark}"

    print("\n" + "=" * 80)
    print("✓ All interpolation tests passed")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    test_step_based_freshness()
    test_color_interpolation()
