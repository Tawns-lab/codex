# Advanced Adversarial Testing Guide

This guide describes the advanced adversarial testing capabilities added to the Multi-Manifold Forensic Engine framework.

## Overview

The `AdvancedAdversarialTester` extends the basic adversarial testing framework with:

1. **5 Sophisticated Attack Strategies**
2. **Bootstrap Confidence Intervals for MI Estimation**
3. **Comprehensive Security Risk Assessment**
4. **FNR-Calibrated Detection Thresholds**
5. **Operational Impact Analysis**
6. **Automated Mitigation Recommendations**

## Advanced Attack Strategies

### 1. Gradient-Based Attack

**Type**: White-box iterative optimization

**Method**: Projected Gradient Descent (PGD) to maximize coherence loss

**Implementation**:
```python
def _gradient_based_attack(latent, target_scale, epsilon, num_iterations):
    for i in range(num_iterations):
        coherence_loss = model.compute_coherence_loss(latent)
        coherence_loss.backward()
        latent[target_scale] += epsilon * gradient.sign()  # Gradient ascent
```

**Threat Model**:
- Attacker has full knowledge of model architecture
- Can compute gradients
- Iteratively optimizes perturbation

**Defense Difficulty**: HIGH
- Specifically targets model weaknesses
- More effective than random noise
- Harder to detect than simple perturbations

---

### 2. Manifold Projection Attack

**Type**: Data-manifold constrained attack

**Method**: Projects perturbations onto the learned data manifold

**Implementation**:
```python
def _manifold_projection_attack(latent, epsilon):
    reconstructed = model.decode(latent)
    manifold_projection = model.encode(reconstructed)
    manifold_direction = manifold_projection - latent
    return latent + epsilon * manifold_direction
```

**Threat Model**:
- Attacker wants perturbations to look "natural"
- Stays on manifold of valid latent codes
- Bypasses out-of-distribution detection

**Defense Difficulty**: MEDIUM-HIGH
- Harder to detect via distributional checks
- Perturbations appear valid
- Exploits manifold structure

---

### 3. Ensemble Disagreement Attack

**Type**: Component diversity exploitation

**Method**: Maximizes disagreement between fast and slow scales

**Implementation**:
```python
def _ensemble_disagreement_attack(latent, epsilon):
    # Get gradients for each scale
    grad_fast = compute_gradient(latent['fast'])
    grad_slow = compute_gradient(latent['slow'])

    # Move scales in opposite directions
    direction = grad_fast - grad_slow
    latent['fast'] += epsilon * direction
    latent['slow'] -= epsilon * direction
```

**Threat Model**:
- Attacks cross-scale coherence mechanism
- Exploits multi-timescale architecture
- Creates maximum internal disagreement

**Defense Difficulty**: CRITICAL
- Directly targets core defense mechanism
- Can severely degrade cross-scale coherence
- Hardest to defend against

---

### 4. Temporal Coherence Attack

**Type**: Sequence-based attack

**Method**: Amplifies temporal inconsistencies

**Implementation**:
```python
def _temporal_coherence_attack(sequence, epsilon):
    for i, latent in enumerate(sequence):
        if i > 0:
            temporal_gap = latent - sequence[i-1]
            latent += epsilon * temporal_gap  # Amplify difference
```

**Threat Model**:
- Attacks temporal consistency
- Exploits sequential dependencies
- Creates abrupt transitions

**Defense Difficulty**: MEDIUM
- Detectable via temporal smoothness checks
- Can be mitigated with sequence modeling
- Requires access to multiple timesteps

---

### 5. False Memory Induction

**Type**: State forgery attack

**Method**: Blends current state with reference trajectory

**Implementation**:
```python
def _false_memory_induction(latent, reference, epsilon):
    blend_ratio = epsilon
    return (1 - blend_ratio) * latent + blend_ratio * reference
```

**Threat Model**:
- Attacker tries to implant forged state
- Mimics legitimate system behavior
- Creates plausible but false history

**Defense Difficulty**: MEDIUM-HIGH
- Hard to distinguish from real states
- May pass individual checks
- Requires context verification

---

## Enhanced Mutual Information Estimation

### Bootstrap Confidence Intervals

The advanced tester provides statistical confidence intervals for MI estimates:

```python
mi_result = tester.estimate_mutual_information_robustness(
    original_data, perturbed_data,
    k=5, n_bootstraps=100
)

print(f"MI: {mi_result['mutual_information']:.3f}")
print(f"95% CI: [{mi_result['confidence_interval'][0]:.3f}, "
      f"{mi_result['confidence_interval'][1]:.3f}]")
print(f"Bootstrap std: {mi_result['bootstrap_std']:.3f}")
```

**Benefits**:
- Statistical significance testing
- Uncertainty quantification
- More reliable robustness assessment

**Method**:
1. Compute point estimate using Kraskov k-NN estimator
2. Bootstrap resample n times
3. Compute MI for each bootstrap sample
4. Extract 2.5% and 97.5% percentiles for 95% CI

---

## Security Risk Assessment

### Risk Levels

The tester automatically assigns risk levels based on attack effectiveness:

| Risk Level | Criteria |
|------------|----------|
| **CRITICAL** | Evasion rate > 70% OR degradation > 5.0 |
| **HIGH** | Evasion rate > 40% OR degradation > 2.0 |
| **MEDIUM** | Evasion rate > 20% OR degradation > 1.0 |
| **LOW** | Evasion rate ≤ 20% AND degradation ≤ 1.0 |

### Assessment Metrics

For each attack type:

```json
{
  "attack_name": {
    "evasion_success_rate": 0.35,
    "max_coherence_degradation": 2.14,
    "avg_coherence_degradation": 1.42,
    "min_mutual_information": 0.18,
    "avg_mutual_information": 0.67,
    "risk_level": "HIGH"
  }
}
```

**Metrics Explained**:
- **Evasion Success Rate**: Fraction of attacks that bypassed detection
- **Coherence Degradation**: How much coherence loss increased
- **Mutual Information**: Information preservation after attack
- **Risk Level**: Overall threat assessment

---

## Comprehensive Reporting

### Security Assessment Report

The tester generates a multi-section report:

#### 1. Security Assessment
Per-attack detailed metrics and risk levels

#### 2. Vulnerability Analysis
- Overall system risk
- Most dangerous attack vector
- Maximum observed evasion rate

#### 3. Operational Implications
- FNR increase risk analysis
- Alert fatigue potential
- Defense evasion scenarios
- Recommended operational actions

#### 4. Mitigation Recommendations
Targeted strategies for each high-risk attack

**Example Mitigations**:
- Gradient attacks → Gradient masking or defensive distillation
- Manifold attacks → Out-of-distribution detection
- Ensemble attacks → Enhanced consensus mechanisms
- Temporal attacks → Sequence modeling and forecasting
- False memory → Cryptographic integrity verification

---

## Usage Example

### Basic Usage

```python
from advanced_adversarial_tester import AdvancedAdversarialTester
from multi_manifold_forensic_engine import create_forensic_engine

# Create engine
engine = create_forensic_engine(input_dim=50)

# Initialize tester with calibrated threshold
tester = AdvancedAdversarialTester(
    forensic_engine=engine,
    calibration_threshold=1.5125  # From FNR calibration
)

# Prepare test data (latent representations)
latent_dataset = [...]  # List of {'fast': tensor, 'slow': tensor}

# Define attack intensities
attack_intensities = np.linspace(0.05, 1.5, 12)

# Run comprehensive suite
results = tester.run_comprehensive_adversarial_suite(
    test_dataset=latent_dataset,
    attack_intensities=attack_intensities.tolist()
)

# Generate report
report = tester.generate_adversarial_report()

# Visualize results
tester.visualize_adversarial_results(save_path='results.png')
```

### Advanced Usage with Custom Attacks

```python
# Access perturbation generator
generator = tester.EnhancedPerturbationGenerator(engine)

# Apply specific attack
perturbed = generator._gradient_based_attack(
    latent_trajectory=my_latent,
    target_scale='both',  # Attack both scales
    epsilon=0.1,
    num_iterations=20
)

# Evaluate impact
original_scores = tester.compute_detection_scores(my_latent)
perturbed_scores = tester.compute_detection_scores(perturbed)

robustness = tester.compute_robustness_metrics(
    original_scores, perturbed_scores
)

print(f"Coherence degradation: {robustness['coherence_degradation']}")
print(f"Detection evasion: {robustness['detection_evasion']}")
```

---

## Interpreting Results

### Visualization Output

The visualization includes 4 plots:

#### Plot 1: Evasion Success Rates
- **X-axis**: Attack types
- **Y-axis**: Success rate (0-1)
- **Colors**: Green (safe) → Orange → Red (critical)
- **Interpretation**: Higher bars = more successful attacks

#### Plot 2: Coherence Degradation Curves
- **X-axis**: Attack intensity
- **Y-axis**: Coherence loss increase
- **Lines**: One per attack type
- **Interpretation**: Steeper curves = more fragile system

#### Plot 3: Mutual Information Resilience
- **X-axis**: Attack intensity
- **Y-axis**: MI (nats)
- **Lines**: One per attack type
- **Interpretation**: Higher MI = better information preservation

#### Plot 4: Risk Assessment
- **Horizontal bars**: Composite risk score per attack
- **Colors**: Risk-level coded (green/yellow/orange/red)
- **Labels**: Risk level text
- **Interpretation**: Longer red/orange bars = higher priority threats

### Key Questions to Answer

1. **Which attack is most dangerous?**
   - Check "Most dangerous attack" in vulnerability analysis
   - Look for highest evasion success rate

2. **Is the system robust enough?**
   - Check overall risk level
   - If CRITICAL or HIGH → Implement mitigations urgently

3. **What should we fix first?**
   - Prioritize CRITICAL risk attacks
   - Review mitigation recommendations for those attacks

4. **How effective are perturbations?**
   - Compare adversarial vs random degradation curves
   - Higher ratio = more effective gradient-based attacks

5. **Is information preserved under attack?**
   - Check MI resilience plots
   - Higher MI at high intensity = robust encoding

---

## Integration with FNR Calibration

The advanced tester integrates with FNR-calibrated detection thresholds:

```python
# Threshold from FNR calibration (e.g., 1.5125 for 1% FNR)
calibration_threshold = 1.5125

tester = AdvancedAdversarialTester(
    forensic_engine=engine,
    calibration_threshold=calibration_threshold
)
```

**Benefits**:
- Evaluates attacks against operational detection threshold
- Quantifies FNR degradation risk
- Assesses real-world evasion potential

**Detection Logic**:
```python
detection_score = coherence_loss.item()
detected = detection_score >= calibration_threshold

# Attack successful if:
# - Original: detected = True
# - Perturbed: detected = False
evasion_success = (orig_detected and not pert_detected)
```

---

## Best Practices

### Testing Strategy

1. **Start with Low Intensities**
   - Use `np.linspace(0.05, 0.5, 10)` initially
   - Increase range if system is robust

2. **Test Multiple Samples**
   - Don't rely on single data point
   - Use diverse test dataset (20+ samples)

3. **Monitor All Metrics**
   - Don't focus only on evasion rate
   - Check coherence degradation and MI jointly

4. **Regular Testing**
   - Run tests after model updates
   - Establish baseline security metrics
   - Track changes over time

### Mitigation Workflow

1. **Identify Critical Risks**
   - Check overall risk level
   - List all CRITICAL/HIGH risk attacks

2. **Prioritize by Impact**
   - Consider evasion rate × operational impact
   - Address attacks that affect FNR most

3. **Implement Targeted Mitigations**
   - Use recommendations from report
   - Test effectiveness after implementation

4. **Re-test**
   - Run adversarial suite again
   - Verify risk reduction
   - Iterate until acceptable

---

## Performance Considerations

### Computational Cost

- **Gradient attacks**: Most expensive (requires backprop)
- **Manifold attacks**: Moderate (encoder-decoder cycle)
- **Others**: Lightweight

**Optimization tips**:
- Reduce `num_iterations` for gradient attacks (default: 10)
- Use fewer attack intensities for quick tests
- Run on GPU for large-scale testing

### Memory Usage

- **Per sample**: ~100-500 MB (depends on latent dim)
- **Bootstrap**: Increases with `n_bootstraps`

**Tips**:
- Reduce `n_bootstraps` for faster testing (50 vs 100)
- Process samples in batches if memory-constrained
- Clear GPU cache between attacks

---

## Example Output

### Console Output

```
==================================================================
RUNNING COMPREHENSIVE ADVERSARIAL TEST SUITE
==================================================================

🔍 Testing gradient_attack...
  ✓ gradient_attack: 42.5% evasion success

🔍 Testing manifold_attack...
  ✓ manifold_attack: 28.3% evasion success

🔍 Testing ensemble_attack...
  ✓ ensemble_attack: 65.8% evasion success

...

📋 SECURITY ASSESSMENT SUMMARY
------------------------------------------------------------------

gradient_attack:
  Evasion success rate: 42.5%
  Max coherence degradation: 2.1456
  Min mutual information: 0.2341
  Risk level: HIGH

...

OVERALL RISK: HIGH
Most dangerous attack: ensemble_attack
```

### Report JSON

```json
{
  "security_assessment": {
    "ensemble_attack": {
      "evasion_success_rate": 0.658,
      "max_coherence_degradation": 3.42,
      "risk_level": "CRITICAL"
    }
  },
  "vulnerability_analysis": {
    "overall_risk": "HIGH",
    "most_dangerous_attack": "ensemble_attack"
  },
  "mitigation_recommendations": [
    "Enhance ensemble consensus mechanisms...",
    "Regular adversarial testing...",
    ...
  ]
}
```

---

## Troubleshooting

### Issue: Low MI estimates

**Cause**: Insufficient sample size or high noise

**Solution**:
- Increase test dataset size (20+ samples)
- Use higher k for k-NN (default: 5, try: 7-10)
- Check data preprocessing/scaling

### Issue: All attacks show 0% evasion

**Cause**: Threshold too low or attacks too weak

**Solution**:
- Verify calibration threshold is correct
- Increase attack intensities
- Check that detection scoring is working

### Issue: Memory errors

**Cause**: Large latent dimensions or too many bootstraps

**Solution**:
- Reduce `n_bootstraps` (50 instead of 100)
- Process fewer samples at once
- Use CPU instead of GPU for testing

### Issue: Gradient attacks fail

**Cause**: Graph computation issues

**Solution**:
- Ensure model in eval mode but allows gradients
- Check that latent tensors have `requires_grad=True`
- Verify coherence loss returns scalar tensor

---

## Future Enhancements

Potential extensions:

1. **Adaptive Attacks**: Attacks that learn from detection feedback
2. **Certified Robustness**: Provable bounds on attack effectiveness
3. **Transfer Attacks**: Test attacks from similar models
4. **Physical Attacks**: Simulate real-world attack scenarios
5. **Compositional Attacks**: Combine multiple attack strategies

---

## References

1. **Goodfellow et al. (2014)**: Explaining and Harnessing Adversarial Examples
2. **Madry et al. (2017)**: Towards Deep Learning Models Resistant to Adversarial Attacks
3. **Kraskov et al. (2004)**: Estimating Mutual Information
4. **Carlini & Wagner (2017)**: Towards Evaluating the Robustness of Neural Networks

---

**Document Version**: 1.0
**Last Updated**: 2025-10-25
**Maintainer**: Claude Code Team
