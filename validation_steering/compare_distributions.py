#!/usr/bin/env python3
"""Compare stability distributions across different steering conditions.

This script loads stability predictions and creates visualizations and statistics.
"""

import argparse
from pathlib import Path

import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


def load_stabilities(csv_path: Path) -> list[float]:
    """Load stability values from a CSV file.
    
    Parameters
    ----------
    csv_path : Path
        Path to CSV file with 'stability' column.
        
    Returns
    -------
    list[float]
        List of stability values.
    """
    stabilities = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                stability = float(row["stability"])
                stabilities.append(stability)
            except (ValueError, KeyError):
                continue
    return stabilities


def compute_statistics(stabilities: list[float]) -> dict:
    """Compute statistics for a list of stability values.
    
    Parameters
    ----------
    stabilities : list[float]
        List of stability values.
        
    Returns
    -------
    dict
        Dictionary with statistics.
    """
    if not stabilities:
        return {}
    
    arr = np.array(stabilities)
    return {
        "count": len(stabilities),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "q25": float(np.percentile(arr, 25)),
        "q75": float(np.percentile(arr, 75)),
    }


def create_visualizations(
    no_steering: list[float],
    high_stability: list[float],
    low_stability: list[float],
    output_path: Path,
):
    """Create visualization comparing the three distributions.
    
    Parameters
    ----------
    no_steering : list[float]
        Stability values from no-steering run.
    high_stability : list[float]
        Stability values from high-stability steering run.
    low_stability : list[float]
        Stability values from low-stability steering run.
    output_path : Path
        Path to save the plot.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Histogram overlay
    ax = axes[0]
    ax.hist(
        no_steering,
        bins=20,
        alpha=0.6,
        label="No Steering",
        color="blue",
        density=True,
    )
    ax.hist(
        high_stability,
        bins=20,
        alpha=0.6,
        label="High Stability",
        color="green",
        density=True,
    )
    ax.hist(
        low_stability,
        bins=20,
        alpha=0.6,
        label="Low Stability",
        color="red",
        density=True,
    )
    ax.set_xlabel("Stability")
    ax.set_ylabel("Density")
    ax.set_title("Stability Distribution Comparison")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Box plot
    ax = axes[1]
    data = [no_steering, high_stability, low_stability]
    labels = ["No Steering", "High Stability", "Low Stability"]
    bp = ax.boxplot(data, labels=labels, patch_artist=True)
    colors = ["blue", "green", "red"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.set_ylabel("Stability")
    ax.set_title("Stability Distribution Box Plot")
    ax.grid(True, alpha=0.3, axis="y")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved visualization to {output_path}")


def perform_statistical_tests(
    no_steering: list[float],
    high_stability: list[float],
    low_stability: list[float],
) -> dict:
    """Perform statistical tests to compare distributions.
    
    Parameters
    ----------
    no_steering : list[float]
        Stability values from no-steering run.
    high_stability : list[float]
        Stability values from high-stability steering run.
    low_stability : list[float]
        Stability values from low-stability steering run.
        
    Returns
    -------
    dict
        Dictionary with test results.
    """
    results = {}
    
    # High stability vs no steering
    if high_stability and no_steering:
        t_stat, p_value_t = stats.ttest_ind(high_stability, no_steering)
        u_stat, p_value_u = stats.mannwhitneyu(high_stability, no_steering)
        results["high_vs_baseline"] = {
            "t_test": {"statistic": float(t_stat), "p_value": float(p_value_t)},
            "mannwhitneyu": {"statistic": float(u_stat), "p_value": float(p_value_u)},
        }
    
    # Low stability vs no steering
    if low_stability and no_steering:
        t_stat, p_value_t = stats.ttest_ind(low_stability, no_steering)
        u_stat, p_value_u = stats.mannwhitneyu(low_stability, no_steering)
        results["low_vs_baseline"] = {
            "t_test": {"statistic": float(t_stat), "p_value": float(p_value_t)},
            "mannwhitneyu": {"statistic": float(u_stat), "p_value": float(p_value_u)},
        }
    
    # High vs low
    if high_stability and low_stability:
        t_stat, p_value_t = stats.ttest_ind(high_stability, low_stability)
        u_stat, p_value_u = stats.mannwhitneyu(high_stability, low_stability)
        results["high_vs_low"] = {
            "t_test": {"statistic": float(t_stat), "p_value": float(p_value_t)},
            "mannwhitneyu": {"statistic": float(u_stat), "p_value": float(p_value_u)},
        }
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Compare stability distributions across steering conditions"
    )
    parser.add_argument(
        "--no-steering",
        type=Path,
        required=True,
        help="CSV file with no-steering stability predictions",
    )
    parser.add_argument(
        "--high-stability",
        type=Path,
        required=True,
        help="CSV file with high-stability steering predictions",
    )
    parser.add_argument(
        "--low-stability",
        type=Path,
        required=True,
        help="CSV file with low-stability steering predictions",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("validation_steering/stability_comparison.png"),
        help="Output path for visualization",
    )
    parser.add_argument(
        "--stats-output",
        type=Path,
        default=Path("validation_steering/stability_stats.txt"),
        help="Output path for statistics text file",
    )
    
    args = parser.parse_args()
    
    # Load data
    print("Loading stability predictions...")
    no_steering = load_stabilities(args.no_steering)
    high_stability = load_stabilities(args.high_stability)
    low_stability = load_stabilities(args.low_stability)
    
    print(f"No steering: {len(no_steering)} samples")
    print(f"High stability: {len(high_stability)} samples")
    print(f"Low stability: {len(low_stability)} samples")
    
    # Compute statistics
    stats_no = compute_statistics(no_steering)
    stats_high = compute_statistics(high_stability)
    stats_low = compute_statistics(low_stability)
    
    # Perform statistical tests
    test_results = perform_statistical_tests(
        no_steering, high_stability, low_stability
    )
    
    # Print summary
    print("\n" + "=" * 60)
    print("STATISTICS SUMMARY")
    print("=" * 60)
    
    print("\nNo Steering (Baseline):")
    if stats_no:
        print(f"  Count: {stats_no['count']}")
        print(f"  Mean: {stats_no['mean']:.4f}")
        print(f"  Median: {stats_no['median']:.4f}")
        print(f"  Std: {stats_no['std']:.4f}")
        print(f"  Range: [{stats_no['min']:.4f}, {stats_no['max']:.4f}]")
    
    print("\nHigh Stability Steering:")
    if stats_high:
        print(f"  Count: {stats_high['count']}")
        print(f"  Mean: {stats_high['mean']:.4f}")
        print(f"  Median: {stats_high['median']:.4f}")
        print(f"  Std: {stats_high['std']:.4f}")
        print(f"  Range: [{stats_high['min']:.4f}, {stats_high['max']:.4f}]")
        if stats_no:
            mean_diff = stats_high["mean"] - stats_no["mean"]
            print(f"  Mean difference from baseline: {mean_diff:+.4f}")
    
    print("\nLow Stability Steering:")
    if stats_low:
        print(f"  Count: {stats_low['count']}")
        print(f"  Mean: {stats_low['mean']:.4f}")
        print(f"  Median: {stats_low['median']:.4f}")
        print(f"  Std: {stats_low['std']:.4f}")
        print(f"  Range: [{stats_low['min']:.4f}, {stats_low['max']:.4f}]")
        if stats_no:
            mean_diff = stats_low["mean"] - stats_no["mean"]
            print(f"  Mean difference from baseline: {mean_diff:+.4f}")
    
    print("\n" + "=" * 60)
    print("STATISTICAL TESTS")
    print("=" * 60)
    
    for comparison, results in test_results.items():
        print(f"\n{comparison.replace('_', ' ').title()}:")
        if "t_test" in results:
            print(f"  t-test: p = {results['t_test']['p_value']:.4f}")
        if "mannwhitneyu" in results:
            print(f"  Mann-Whitney U: p = {results['mannwhitneyu']['p_value']:.4f}")
    
    # Save statistics to file
    with open(args.stats_output, "w") as f:
        f.write("STABILITY DISTRIBUTION STATISTICS\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("No Steering (Baseline):\n")
        for key, value in stats_no.items():
            f.write(f"  {key}: {value}\n")
        
        f.write("\nHigh Stability Steering:\n")
        for key, value in stats_high.items():
            f.write(f"  {key}: {value}\n")
        
        f.write("\nLow Stability Steering:\n")
        for key, value in stats_low.items():
            f.write(f"  {key}: {value}\n")
        
        f.write("\n" + "=" * 60 + "\n")
        f.write("STATISTICAL TESTS\n")
        f.write("=" * 60 + "\n")
        for comparison, results in test_results.items():
            f.write(f"\n{comparison}:\n")
            for test_name, test_result in results.items():
                f.write(f"  {test_name}: {test_result}\n")
    
    print(f"\nSaved statistics to {args.stats_output}")
    
    # Create visualization
    create_visualizations(
        no_steering, high_stability, low_stability, args.output
    )
    
    # Check success criteria
    print("\n" + "=" * 60)
    print("SUCCESS CRITERIA CHECK")
    print("=" * 60)
    
    if stats_high and stats_no:
        high_success = stats_high["mean"] > stats_no["mean"]
        print(f"✓ High stability > baseline: {high_success} "
              f"({stats_high['mean']:.4f} > {stats_no['mean']:.4f})")
    
    if stats_low and stats_no:
        low_success = stats_low["mean"] < stats_no["mean"]
        print(f"✓ Low stability < baseline: {low_success} "
              f"({stats_low['mean']:.4f} < {stats_no['mean']:.4f})")
    
    if test_results.get("high_vs_baseline"):
        p_val = test_results["high_vs_baseline"]["t_test"]["p_value"]
        sig = p_val < 0.05
        print(f"✓ Statistical significance (high vs baseline): {sig} (p = {p_val:.4f})")
    
    if test_results.get("low_vs_baseline"):
        p_val = test_results["low_vs_baseline"]["t_test"]["p_value"]
        sig = p_val < 0.05
        print(f"✓ Statistical significance (low vs baseline): {sig} (p = {p_val:.4f})")


if __name__ == "__main__":
    main()

