#!/bin/bash
# Run the complete validation experiment for property steering

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

# Activate virtual environment if it exists
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
    echo "Activated virtual environment"
elif [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
    echo "Activated virtual environment"
fi

# Parse command line arguments
START_STEP=1
if [ "$1" = "--start-at" ] || [ "$1" = "-s" ]; then
    START_STEP="$2"
    if [ -z "$START_STEP" ]; then
        echo "Error: --start-at requires a step number"
        echo "Usage: $0 [--start-at STEP_NUMBER]"
        exit 1
    fi
    shift 2
fi

echo "=========================================="
echo "Property Steering Validation Experiment"
echo "=========================================="
echo ""
if [ "$START_STEP" -gt 1 ]; then
    echo "Starting at step $START_STEP"
    echo ""
fi

# Configuration
DESIGN_SPEC="validation_steering/design_spec.yaml"
NUM_DESIGNS=50

# Step 1: Generate samples with no steering
if [ "$START_STEP" -le 1 ]; then
    echo "Step 1: Generating samples with NO steering (baseline)..."
    boltzgen run "$DESIGN_SPEC" \
        --output validation_steering/no_steering \
        --num_designs $NUM_DESIGNS \
        --protocol protein-anything \
        --steps design \
        --no_subprocess \
        --devices 1 \
        --config design trainer.accelerator=gpu \
        --config design trainer.devices=1 \
        --config design override.diffusion_process_args.enable_property_steering=False
fi

# Step 2: Generate samples with high stability steering
if [ "$START_STEP" -le 2 ]; then
    echo ""
    echo "Step 2: Generating samples with HIGH stability steering..."
    boltzgen run "$DESIGN_SPEC" \
        --output validation_steering/high_stability \
        --num_designs $NUM_DESIGNS \
        --protocol protein-anything \
        --steps design \
        --no_subprocess \
        --devices 1 \
        --config_dir validation_steering \
        --config design trainer.accelerator=gpu \
        --config design trainer.devices=1 \
        --config design override.diffusion_process_args.target_stability=0.8
fi

# Step 3: Generate samples with low stability steering
if [ "$START_STEP" -le 3 ]; then
    echo ""
    echo "Step 3: Generating samples with LOW stability steering..."
    boltzgen run "$DESIGN_SPEC" \
        --output validation_steering/low_stability \
        --num_designs $NUM_DESIGNS \
        --protocol protein-anything \
        --steps design \
        --no_subprocess \
        --devices 1 \
        --config_dir validation_steering \
        --config design trainer.accelerator=gpu \
        --config design trainer.devices=1 \
        --config design override.diffusion_process_args.target_stability=-0.5
fi

# Step 4: Extract sequences
if [ "$START_STEP" -le 4 ]; then
    echo ""
    echo "Step 4: Extracting sequences from generated designs..."

    python validation_steering/extract_sequences.py \
        validation_steering/no_steering \
        -o validation_steering/no_steering_sequences.csv

    python validation_steering/extract_sequences.py \
        validation_steering/high_stability \
        -o validation_steering/high_stability_sequences.csv

    python validation_steering/extract_sequences.py \
        validation_steering/low_stability \
        -o validation_steering/low_stability_sequences.csv
fi

# Step 5: Predict stabilities
if [ "$START_STEP" -le 5 ]; then
    echo ""
    echo "Step 5: Predicting stabilities using TAPE..."

    python validation_steering/predict_stabilities.py \
        validation_steering/no_steering_sequences.csv \
        -o validation_steering/no_steering_stabilities.csv

    python validation_steering/predict_stabilities.py \
        validation_steering/high_stability_sequences.csv \
        -o validation_steering/high_stability_stabilities.csv

    python validation_steering/predict_stabilities.py \
        validation_steering/low_stability_sequences.csv \
        -o validation_steering/low_stability_stabilities.csv
fi

# Step 6: Compare distributions
if [ "$START_STEP" -le 6 ]; then
    echo ""
    echo "Step 6: Comparing stability distributions..."

    python validation_steering/compare_distributions.py \
        --no-steering validation_steering/no_steering_stabilities.csv \
        --high-stability validation_steering/high_stability_stabilities.csv \
        --low-stability validation_steering/low_stability_stabilities.csv \
        -o validation_steering/stability_comparison.png \
        --stats-output validation_steering/stability_stats.txt
fi

echo ""
echo "=========================================="
echo "Experiment Complete!"
echo "=========================================="
echo ""
echo "Results saved to:"
echo "  - validation_steering/stability_comparison.png"
echo "  - validation_steering/stability_stats.txt"
echo ""

