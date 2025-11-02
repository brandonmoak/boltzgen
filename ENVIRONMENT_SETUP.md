# Environment Setup for Property Steering

Since you already have a BoltzGen environment, you just need to add TAPE for stability prediction.

## Quick Setup

### 1. Activate your existing BoltzGen environment

```bash
conda activate bg  # or whatever your environment is called
```

### 2. Install TAPE

```bash
pip install tape-proteins
```

### 3. Verify installation

```bash
python -c "import tape; print('TAPE installed successfully')"
```

### 4. Install BoltzGen in editable mode (if you haven't already)

Since you're working on the source code, make sure BoltzGen is installed in editable mode:

```bash
pip install -e .
```

This ensures your code changes are immediately available.

---

## Running Tests

Once TAPE is installed, you can run the test scripts:

```bash
# Test sequence extraction (no TAPE needed)
python test_sequence_extraction.py

# Test property steering (no TAPE needed)
python test_property_steering.py

# Test diffusion initialization (TAPE optional)
python test_diffusion_init.py

# Test TAPE stability predictor (requires TAPE)
python test_stability_predictor.py

# Test full pipeline (requires TAPE)
python test_integration_pipeline.py
```

---

## Troubleshooting

**If TAPE installation fails:**
```bash
# Try installing from source
pip install git+https://github.com/songlab-cal/tape.git
```

**If you get import errors:**
- Make sure you're in the correct conda environment
- Verify BoltzGen is installed in editable mode: `pip install -e .`
- Check Python version: `python --version` (should be >= 3.9, preferably 3.11+)

**To test without TAPE first:**
Run `test_sequence_extraction.py` and `test_property_steering.py` - these don't require TAPE.

