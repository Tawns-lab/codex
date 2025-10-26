# Adversarial Forensic Testing Framework

A neuroscientifically-grounded forensic integrity engine that shifts from discrete event logs to continuous state-space trajectories, enabling robust tamper detection through multi-scale coherence analysis.

## 🎯 Overview

This framework implements a **Multi-Manifold Forensic Engine** that uses distributed memory principles to detect system tampering and anomalies. Unlike traditional discrete event-based logging, this approach models system state as continuous trajectories across multiple timescales (fast/slow dynamics), making it significantly more robust against sophisticated attacks.

### Key Features

- **Multi-timescale Representation**: Dual encoder architecture capturing both rapid state changes and gradual trends
- **Variational Encoding**: Robust latent representations using VAE-based manifold learning
- **Cross-scale Coherence**: Integrity validation through coherence constraints across timescales
- **Adversarial Testing**: Comprehensive robustness evaluation with gradient-based attacks
- **Advanced Attack Strategies**: 5 sophisticated attack methods (gradient, manifold, ensemble, temporal, false memory)
- **Statistical Robustness**: Bootstrap confidence intervals for mutual information estimation
- **Security Assessment**: Automated risk analysis and mitigation recommendations
- **ACP Integration**: Seamless integration with Adversarial Counterfactual Planning for proactive defense

### 🆕 Advanced Adversarial Testing

The framework now includes **AdvancedAdversarialTester** with:

- **5 Attack Strategies**: Gradient-based, manifold projection, ensemble disagreement, temporal coherence, false memory induction
- **Bootstrap MI Estimation**: Statistical confidence intervals for robustness metrics
- **Comprehensive Reporting**: Security assessment with operational implications and mitigation strategies
- **FNR Calibration**: Integration with false negative rate thresholds
- **Advanced Visualization**: Multi-panel plots showing evasion rates, degradation curves, and risk assessment

See [ADVANCED_TESTING.md](ADVANCED_TESTING.md) for detailed documentation.

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│         Multi-Manifold Forensic Engine                  │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  ┌──────────────┐                  ┌──────────────┐     │
│  │ Fast-Scale   │──────┐           │ Slow-Scale   │     │
│  │ Encoder      │      │           │ Encoder      │     │
│  │ (Hippocampal)│      ▼           │ (Cortical)   │     │
│  └──────────────┘   Coherence      └──────────────┘     │
│                     Analysis                             │
│                        │                                 │
│                        ▼                                 │
│                  ┌──────────┐                            │
│                  │ Decoder  │                            │
│                  └──────────┘                            │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

### Neuroscientific Inspiration

The architecture is inspired by distributed memory systems in the brain:

- **Fast Encoder** (Hippocampal): Rapid encoding of new system states
- **Slow Encoder** (Cortical): Consolidation of long-term patterns
- **Coherence Network**: Cross-scale validation analogous to hippocampal-neocortical replay

## 📦 Installation

```bash
# Clone the repository
cd adversarial-testing

# Install dependencies
pip install -r requirements.txt
```

## 🚀 Quick Start

### 1. Generate Test Fixtures

```bash
cd src
python data_generators.py
```

This creates:
- Causal graphs for system modeling
- Healthy and stressed system metrics
- Clean and tampered trajectories

### 2. Run the Full Test Suite

```bash
cd examples
python run_full_test_suite.py --output-dir ../test_results
```

This executes:
1. **Forensic Engine Tests**: Validates core encoding/decoding
2. **Adversarial Perturbation Tests**: Random and gradient-based attacks
3. **Tamper Detection Tests**: Coherence-based anomaly detection
4. **ACP Integration**: Defense planning with counterfactual reasoning

### 3. View Results

Results are saved to `test_results/`:
```
test_results/
├── adversarial_results.png          # Robustness visualizations
├── adversarial_test_results.json    # Detailed metrics
├── tamper_detection_results.png     # Detection performance
├── acp_integration_results.json     # Defense plan outputs
└── SUMMARY_REPORT.md                # Executive summary
```

## 📚 Core Components

### Multi-Manifold Forensic Engine

The core engine implementing continuous state-space forensic analysis.

```python
from multi_manifold_forensic_engine import create_forensic_engine

# Create engine
engine = create_forensic_engine(
    input_dim=50,
    fast_latent_dim=32,
    slow_latent_dim=16
)

# Encode system state
latent = engine.encode(system_trajectory)

# Detect tampering
result = engine.detect_tampering_via_trajectory_consistency(
    trajectory,
    window_size=10,
    threshold=0.5
)
```

### Adversarial Perturbation Generator

Tools for generating adversarial attacks to test robustness.

```python
from adversarial_perturbations import PerturbationGenerator

generator = PerturbationGenerator(engine)

# Random perturbation
perturbed = generator.generate_random_perturbation(
    latent, scale='fast', noise_level=1.0
)

# Gradient-based adversarial attack
adversarial = generator.generate_adversarial_perturbation(
    latent, scale='fast', step_size=0.1
)
```

### Adversarial Test Suite

Comprehensive robustness evaluation framework.

```python
from adversarial_perturbations import AdversarialTestSuite

suite = AdversarialTestSuite(engine)
results = suite.run_full_suite(test_data)

# Visualize results
suite.visualize_results(save_path='results.png')
```

### Enhanced ACP Planner

Real-time adversarial counterfactual planning for system defense.

```python
from enhanced_acp_planner import RealTimeACPPlanner

planner = RealTimeACPPlanner(
    causal_graph_path='causal_graph.json',
    system_components=['load_balancer', 'gateway']
)

# Analyze system health
health = planner.analyze_system_health(current_metrics)

# Generate defense plans
plans = planner.generate_defense_plans(current_metrics, health)
```

## 🧪 Testing Methodology

### 1. Random Perturbation Tests

Evaluate robustness against non-targeted noise by injecting Gaussian perturbations at various scales.

**Metrics**:
- Coherence loss degradation
- Mutual information preservation
- Reconstruction error

### 2. Adversarial Perturbation Tests

Test resilience against gradient-based attacks that maximize coherence loss.

**Attack Methods**:
- Fast Gradient Sign Method (FGSM)
- Iterative gradient ascent
- Targeted latent manipulation

### 3. Tamper Detection Validation

Verify that the engine can detect various tampering techniques:
- Spike injection
- Gradual drift
- State replacement

### 4. Cross-Scale Consistency

Validate that fast and slow representations maintain coherence even under perturbation.

## 📊 Interpretation of Results

### Coherence Degradation Curves

Shows how coherence loss increases with perturbation strength:
- **Steeper curve**: More fragile system
- **Adversarial > Random**: Expected, shows targeted attacks are more effective
- **Saturation point**: Maximum degradation level

### Mutual Information Resilience

Quantifies information preservation after perturbation:
- **High MI**: System recovers well from attacks
- **Low MI**: Information loss indicates vulnerability
- **MI decay rate**: Speed of degradation under attack

### Detection Performance

Measured by:
- **True Positive Rate**: Detecting actual tampering
- **False Positive Rate**: False alarms on clean data
- **Coherence threshold**: Tunable for precision/recall tradeoff

## 🔬 Advanced Usage

### Custom Forensic Engine Configuration

```python
from multi_manifold_forensic_engine import ForensicConfig, MultiManifoldForensicEngine

config = ForensicConfig(
    input_dim=128,
    fast_latent_dim=64,
    slow_latent_dim=32,
    hidden_dim=256,
    coherence_weight=1.5,  # Increase coherence importance
    reconstruction_weight=1.0,
    kl_weight=0.1
)

engine = MultiManifoldForensicEngine(config)
```

### Training the Engine

```python
import torch.optim as optim

optimizer = optim.Adam(engine.parameters(), lr=1e-3)

for epoch in range(num_epochs):
    for batch in dataloader:
        optimizer.zero_grad()
        output = engine(batch)
        output['total_loss'].backward()
        optimizer.step()
```

### Custom Perturbation Strategies

```python
from adversarial_perturbations import PerturbationGenerator

class CustomAttack(PerturbationGenerator):
    def generate_custom_attack(self, latent, params):
        # Implement custom attack logic
        pass
```

## 📈 Performance Considerations

### Computational Complexity

- **Encoding**: O(n × d × h) where n=batch, d=dims, h=hidden
- **Coherence**: O(n × (d_fast + d_slow))
- **MI Estimation**: O(n² × k) for k-NN method

### Optimization Tips

1. **Batch Processing**: Use larger batches for better GPU utilization
2. **Caching**: Pre-encode static trajectories
3. **Sparse Representations**: Consider sparse autoencoders for high-dimensional data
4. **Mixed Precision**: Use `torch.amp` for faster training

## 🤝 Contributing

This is a research prototype. Contributions welcome for:

- Additional attack methods
- Improved MI estimation
- Real-world system integration
- Performance optimizations

## 📄 License

See LICENSE file in the parent directory.

## 🔗 References

1. **Distributed Memory Systems**: O'Reilly, R. C., & McClelland, J. L. (1994). Hippocampal conjunctive encoding
2. **VAE for Robustness**: Kingma, D. P., & Welling, M. (2013). Auto-Encoding Variational Bayes
3. **Adversarial Robustness**: Goodfellow, I. J., et al. (2014). Explaining and Harnessing Adversarial Examples
4. **Mutual Information Estimation**: Kraskov, A., et al. (2004). Estimating mutual information

## 📧 Contact

For questions or collaboration: See main repository README

---

**Built with Claude Code** 🤖
