# Quick Start Guide

Get up and running with adversarial forensic testing in 5 minutes.

## Prerequisites

- Python 3.8+
- pip

## Installation

### Step 1: Install Dependencies

```bash
cd adversarial-testing
pip install -r requirements.txt
```

This installs:
- PyTorch (CPU version)
- NumPy, SciPy
- scikit-learn
- matplotlib

### Step 2: Generate Test Data

```bash
cd src
python data_generators.py
```

This creates synthetic:
- System trajectories
- Causal graphs
- System metrics

Output location: `adversarial-testing/data/`

## Running Your First Test

### Simple Demo (2 minutes)

```bash
cd examples
python simple_demo.py
```

This runs a minimal demonstration showing:
- Multi-scale encoding
- Coherence analysis
- Random perturbations
- Adversarial attacks
- Tamper detection

**Expected Output**:
```
========================================================
ADVERSARIAL FORENSIC TESTING - SIMPLE DEMO
========================================================

✓ Dependencies loaded successfully
✓ Modules imported successfully

1. Creating Multi-Manifold Forensic Engine...
   ✓ Engine created
   - Input dimension: 20
   - Fast latent dimension: 10
   - Slow latent dimension: 5

...

✅ DEMO COMPLETED SUCCESSFULLY
```

### Full Test Suite (10-15 minutes)

```bash
cd examples
python run_full_test_suite.py --output-dir ../test_results
```

This runs comprehensive tests:
1. Forensic engine validation
2. Random perturbation tests (15 noise levels)
3. Adversarial attack tests (15 step sizes)
4. Tamper detection on clean vs. tampered trajectories
5. ACP planner integration

**Output Files**:
```
test_results/
├── adversarial_results.png              # Robustness plots
├── adversarial_test_results.json        # Raw metrics
├── tamper_detection_results.png         # Detection visualization
├── tamper_detection_results.json        # Detection metrics
├── acp_integration_results.json         # Defense plans
└── SUMMARY_REPORT.md                    # Executive summary
```

## Understanding the Results

### Coherence Degradation Curves

**What to look for**:
- **Y-axis**: Coherence loss (lower is better)
- **X-axis**: Perturbation strength
- **Interpretation**:
  - Flat curve = robust system
  - Steep curve = vulnerable to attacks
  - Adversarial > Random = expected behavior

### Mutual Information Plots

**What to look for**:
- **Y-axis**: MI (higher is better)
- **X-axis**: Perturbation strength
- **Interpretation**:
  - High MI = information preserved
  - Low MI = information lost
  - Slow decay = resilient system

### Tamper Detection

**What to look for**:
- **Clean trajectory**: High mean coherence (>0.7)
- **Tampered trajectory**: Low mean coherence (<0.5)
- **Detection**: Suspicious windows identified

**Success criteria**:
- Clean trajectory: `tampered = False`
- Tampered trajectory: `tampered = True`
- Detected indices overlap with actual tamper region

## Using the Makefile

For convenience, use the Makefile:

```bash
# Install everything
make install

# Generate fixtures
make fixtures

# Run simple demo
make demo

# Run unit tests
make test

# Run full suite
make full-test

# Clean up
make clean

# See all commands
make help
```

## Common Issues

### Issue: ModuleNotFoundError: torch

**Solution**: Install PyTorch
```bash
pip install torch
```

### Issue: "No test fixtures found"

**Solution**: Generate fixtures first
```bash
cd src
python data_generators.py
```

### Issue: Memory error during full test

**Solution**: Reduce batch size in config
```python
# In run_full_test_suite.py
batch_data = data_gen.generate_batch_trajectories(
    batch_size=8,  # Reduce from 16
    seq_len=10,
    input_dim=50
)
```

## Next Steps

1. **Read the README**: Comprehensive documentation
2. **Explore ARCHITECTURE.md**: Deep technical dive
3. **Customize parameters**: Edit configs in example scripts
4. **Integrate with your system**: Use the APIs

## Example Integration

```python
from multi_manifold_forensic_engine import create_forensic_engine

# Create engine
engine = create_forensic_engine(
    input_dim=YOUR_FEATURE_DIM,
    fast_latent_dim=32,
    slow_latent_dim=16
)

# Your system data
your_trajectory = load_system_logs()  # Shape: (timesteps, features)

# Detect tampering
result = engine.detect_tampering_via_trajectory_consistency(
    your_trajectory,
    window_size=10,
    threshold=0.5
)

if result['tampered']:
    print(f"⚠️  Tampering detected at indices: {result['tampered_indices']}")
    print(f"   Mean coherence: {result['mean_coherence']:.4f}")
else:
    print("✅ No tampering detected")
```

## Getting Help

- **Issues**: Check GitHub issues
- **Documentation**: See README.md and ARCHITECTURE.md
- **Examples**: Browse `examples/` directory

---

**Happy Testing! 🛡️**
