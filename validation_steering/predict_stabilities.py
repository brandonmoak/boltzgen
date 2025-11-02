#!/usr/bin/env python3
"""Predict stability for sequences extracted from designs.

This script loads sequences from CSV files and runs TAPE stability predictions.
"""

import argparse
import csv
from pathlib import Path

import torch

from boltzgen.model.modules.stability_predictor import TAPEStabilityPredictor


def predict_stabilities(input_csv: Path, output_csv: Path):
    """Predict stability for sequences in a CSV file.
    
    Parameters
    ----------
    input_csv : Path
        Input CSV with sequences (must have 'sequence' column).
    output_csv : Path
        Output CSV with stability predictions.
    """
    # Load sequences
    sequences = []
    with open(input_csv, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sequences.append(row)
    
    if not sequences:
        print(f"Warning: No sequences found in {input_csv}")
        return
    
    print(f"Predicting stability for {len(sequences)} sequences...")
    
    # Initialize predictor
    predictor = TAPEStabilityPredictor(device="cpu")
    
    # Predict stability for each sequence
    results = []
    for i, row in enumerate(sequences):
        sequence = row["sequence"]
        try:
            stability = predictor.forward(sequence)
            results.append({
                "file": row.get("file", ""),
                "sequence": sequence,
                "length": row.get("length", len(sequence)),
                "stability": stability,
            })
            if (i + 1) % 10 == 0:
                print(f"  Processed {i + 1}/{len(sequences)} sequences...")
        except Exception as e:
            print(f"Warning: Failed to predict stability for sequence {i+1}: {e}")
            continue
    
    # Write results
    if results:
        with open(output_csv, "w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["file", "sequence", "length", "stability"]
            )
            writer.writeheader()
            writer.writerows(results)
        print(f"Saved {len(results)} stability predictions to {output_csv}")
    else:
        print(f"Warning: No successful predictions")


def main():
    parser = argparse.ArgumentParser(
        description="Predict stability for sequences using TAPE"
    )
    parser.add_argument(
        "input_csv",
        type=Path,
        help="Input CSV file with sequences",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help="Output CSV file path",
    )
    
    args = parser.parse_args()
    
    if not args.input_csv.exists():
        raise FileNotFoundError(f"File not found: {args.input_csv}")
    
    predict_stabilities(args.input_csv, args.output)


if __name__ == "__main__":
    main()

