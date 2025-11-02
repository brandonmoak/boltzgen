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

echo "=========================================="
echo "Property Steering Validation Experiment"
echo "=========================================="
echo ""

# Configuration
DESIGN_SPEC="validation_steering/design_spec.yaml"
NUM_DESIGNS=50

# Step 1: Generate samples with no steering
echo "Step 1: Generating samples with NO steering (baseline)..."
boltzgen run "$DESIGN_SPEC" \
    --output validation_steering/no_steering \
    --num_designs $NUM_DESIGNS \
    --protocol protein-anything \
    --steps design \
    --no_subprocess \
    --devices 1 \
    --config design trainer.accelerator=gpu \
    --config design trainer.devices=1

# Step 2: Generate samples with high stability steering
echo ""
echo "Step 2: Generating samples with HIGH stability steering..."
boltzgen run "$DESIGN_SPEC" \
    --output validation_steering/high_stability \
    --num_designs $NUM_DESIGNS \
    --protocol protein-anything \
    --steps design \
    --no_subprocess \
    --devices 1 \
    --config design trainer.accelerator=gpu \
    --config design trainer.devices=1 \
    --config diffusion_process_args.enable_property_steering=true \
    --config diffusion_process_args.target_stability=0.8 \
    --config diffusion_process_args.stability_bias_weight=1.0 \
    --config diffusion_process_args.steering_update_freq=5

# Step 3: Generate samples with low stability steering
echo ""
echo "Step 3: Generating samples with LOW stability steering..."
boltzgen run "$DESIGN_SPEC" \
    --output validation_steering/low_stability \
    --num_designs $NUM_DESIGNS \
    --protocol protein-anything \
    --steps design \
    --no_subprocess \
    --devices 1 \
    --config design trainer.accelerator=gpu \
    --config design trainer.devices=1 \
    --config diffusion_process_args.enable_property_steering=true \
    --config diffusion_process_args.target_stability=-0.5 \
    --config diffusion_process_args.stability_bias_weight=1.0 \
    --config diffusion_process_args.steering_update_freq=5

# Step 4: Extract sequences
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

# Step 5: Predict stabilities
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

# Step 6: Compare distributions
echo ""
echo "Step 6: Comparing stability distributions..."

python validation_steering/compare_distributions.py \
    --no-steering validation_steering/no_steering_stabilities.csv \
    --high-stability validation_steering/high_stability_stabilities.csv \
    --low-stability validation_steering/low_stability_stabilities.csv \
    -o validation_steering/stability_comparison.png \
    --stats-output validation_steering/stability_stats.txt

echo ""
echo "=========================================="
echo "Experiment Complete!"
echo "=========================================="
echo ""
echo "Results saved to:"
echo "  - validation_steering/stability_comparison.png"
echo "  - validation_steering/stability_stats.txt"
echo ""

