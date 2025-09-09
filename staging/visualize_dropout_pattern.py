"""Visualize chaotic oscillating dropout pattern for SMEAR experts."""

import numpy as np
import matplotlib.pyplot as plt
from praxis.distributions import chaotic_dropout_pattern, DISTRIBUTION_PROFILES

def plot_dropout_patterns():
    """Create visualization of the dropout pattern using registered profiles."""
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    
    # Generate patterns with different profiles
    steps = 10000
    
    # Pattern 1: Gentle waves profile
    profile1 = DISTRIBUTION_PROFILES["gentle_waves"]
    pattern1 = chaotic_dropout_pattern(
        steps, 
        base_frequency=profile1["base_frequency"], 
        chaos_factor=profile1["chaos_factor"]
    ).numpy()
    axes[0].plot(pattern1, 'b-', linewidth=0.8, alpha=0.8)
    axes[0].set_title(f'Gentle Waves: {profile1["description"]}', fontsize=12)
    axes[0].set_ylabel('Dropout Rate', fontsize=10)
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(y=0.5, color='r', linestyle='--', alpha=0.3, label='Center (0.5)')
    axes[0].axhline(y=0.25, color='g', linestyle='--', alpha=0.3, label='Min (0.25)')
    axes[0].axhline(y=0.75, color='g', linestyle='--', alpha=0.3, label='Max (0.75)')
    axes[0].set_ylim(0.2, 0.8)
    axes[0].legend(loc='upper right', fontsize=8)
    
    # Pattern 2: Balanced chaos (recommended)
    profile2 = DISTRIBUTION_PROFILES["balanced_chaos"]
    pattern2 = chaotic_dropout_pattern(
        steps,
        base_frequency=profile2["base_frequency"],
        chaos_factor=profile2["chaos_factor"]
    ).numpy()
    axes[1].plot(pattern2, 'r-', linewidth=0.8, alpha=0.8)
    axes[1].set_title(f'Balanced Chaos (DEFAULT): {profile2["description"]}', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Dropout Rate', fontsize=10)
    axes[1].grid(True, alpha=0.3)
    axes[1].axhline(y=0.5, color='r', linestyle='--', alpha=0.3)
    axes[1].axhline(y=0.25, color='g', linestyle='--', alpha=0.3)
    axes[1].axhline(y=0.75, color='g', linestyle='--', alpha=0.3)
    axes[1].set_ylim(0.2, 0.8)
    
    # Pattern 3: Dynamic bursts
    profile3 = DISTRIBUTION_PROFILES["dynamic_bursts"]
    pattern3 = chaotic_dropout_pattern(
        steps,
        base_frequency=profile3["base_frequency"],
        chaos_factor=profile3["chaos_factor"]
    ).numpy()
    axes[2].plot(pattern3, 'g-', linewidth=0.8, alpha=0.8)
    axes[2].set_title(f'Dynamic Bursts: {profile3["description"]}', fontsize=12)
    axes[2].set_ylabel('Dropout Rate', fontsize=10)
    axes[2].set_xlabel('Training Steps', fontsize=10)
    axes[2].grid(True, alpha=0.3)
    axes[2].axhline(y=0.5, color='r', linestyle='--', alpha=0.3)
    axes[2].axhline(y=0.25, color='g', linestyle='--', alpha=0.3)
    axes[2].axhline(y=0.75, color='g', linestyle='--', alpha=0.3)
    axes[2].set_ylim(0.2, 0.8)
    
    plt.suptitle('Chaotic Oscillating Dropout Patterns for SMEAR Experts', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Add zoom-in plots
    fig2, axes2 = plt.subplots(2, 1, figsize=(14, 8))
    
    # Zoom in on first 1000 steps using balanced_chaos profile
    zoom_steps = 1000
    profile_zoom = DISTRIBUTION_PROFILES["balanced_chaos"]
    pattern_zoom = chaotic_dropout_pattern(
        steps,
        base_frequency=profile_zoom["base_frequency"],
        chaos_factor=profile_zoom["chaos_factor"]
    ).numpy()
    
    axes2[0].plot(pattern_zoom[:zoom_steps], 'r-', linewidth=1.2)
    axes2[0].set_title('Zoom: First 1000 Steps (Medium Chaos)', fontsize=12)
    axes2[0].set_ylabel('Dropout Rate', fontsize=10)
    axes2[0].grid(True, alpha=0.3)
    axes2[0].axhline(y=0.5, color='k', linestyle='--', alpha=0.3)
    axes2[0].fill_between(range(zoom_steps), 0.25, 0.75, alpha=0.1, color='green', label='Active Range')
    axes2[0].set_ylim(0.2, 0.8)
    axes2[0].legend(loc='upper right')
    
    # Show a "burst" region
    burst_region = slice(4000, 5000)
    axes2[1].plot(range(1000), pattern_zoom[burst_region], 'b-', linewidth=1.2)
    axes2[1].set_title('Zoom: Steps 4000-5000 (Showing Burst Behavior)', fontsize=12)
    axes2[1].set_ylabel('Dropout Rate', fontsize=10)
    axes2[1].set_xlabel('Relative Step', fontsize=10)
    axes2[1].grid(True, alpha=0.3)
    axes2[1].axhline(y=0.5, color='k', linestyle='--', alpha=0.3)
    axes2[1].fill_between(range(1000), 0.25, 0.75, alpha=0.1, color='green')
    axes2[1].set_ylim(0.2, 0.8)
    
    plt.tight_layout()
    
    # Show statistics
    print("\nPattern Statistics (Medium Chaos, 10000 steps):")
    print(f"  Mean dropout rate: {pattern2.mean():.3f}")
    print(f"  Std deviation: {pattern2.std():.3f}")
    print(f"  Min dropout rate: {pattern2.min():.3f}")
    print(f"  Max dropout rate: {pattern2.max():.3f}")
    print(f"  Median dropout rate: {np.median(pattern2):.3f}")
    
    # Calculate autocorrelation to show non-repetitive nature
    from scipy import signal
    autocorr = signal.correlate(pattern2 - pattern2.mean(), pattern2 - pattern2.mean(), mode='same')
    autocorr = autocorr / autocorr[len(autocorr)//2]
    
    # Find peaks in autocorrelation
    peaks, _ = signal.find_peaks(autocorr[len(autocorr)//2:], height=0.1)
    if len(peaks) > 0:
        print(f"  First significant autocorrelation peak at: {peaks[0]} steps")
    else:
        print(f"  No significant repetition detected (good!)")
    
    plt.show()

if __name__ == "__main__":
    print("Generating chaotic oscillating dropout patterns for SMEAR expert guidance...")
    print("=" * 70)
    print("Available Profiles:")
    for name, profile in DISTRIBUTION_PROFILES.items():
        default_marker = " (DEFAULT)" if name == "balanced_chaos" else ""
        print(f"  - {name}{default_marker}: {profile['description']}")
    print()
    print("Pattern characteristics:")
    print("  - Base range: 0.25 to 0.75 (25% to 75% dropout)")
    print("  - Combines multiple oscillation frequencies")
    print("  - Includes chaotic modulation to prevent repetition")
    print("  - Features occasional 'bursts' of rapid change")
    print("  - Slow drift prevents exact periodicity")
    print("=" * 70)
    
    plot_dropout_patterns()