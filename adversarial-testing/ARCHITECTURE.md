# Architecture Overview

## Adversarial Forensic Testing Framework

This document provides a detailed technical overview of the Multi-Manifold Forensic Engine architecture and its adversarial testing framework.

## Table of Contents

1. [System Architecture](#system-architecture)
2. [Core Components](#core-components)
3. [Adversarial Testing Methodology](#adversarial-testing-methodology)
4. [Integration with ACP](#integration-with-acp)
5. [Data Flow](#data-flow)
6. [Performance Characteristics](#performance-characteristics)

---

## System Architecture

### High-Level Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                   System Input (Trajectories)                    │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│              Multi-Manifold Forensic Engine                      │
│                                                                  │
│  ┌──────────────┐          ┌──────────────┐                     │
│  │ Fast-Scale   │          │ Slow-Scale   │                     │
│  │ Encoder      ├──────┐   │ Encoder      │                     │
│  │ (μ, σ²)      │      │   │ (μ, σ²)      │                     │
│  └──────────────┘      │   └──────────────┘                     │
│         │              │           │                             │
│         │   Reparameterize         │                             │
│         ▼              │           ▼                             │
│   ┌──────────┐         │     ┌──────────┐                       │
│   │ z_fast   │         │     │ z_slow   │                       │
│   └─────┬────┘         │     └─────┬────┘                       │
│         │              │           │                             │
│         └──────────────┼───────────┘                             │
│                        │                                         │
│                        ▼                                         │
│              ┌──────────────────┐                                │
│              │ Coherence Network│                                │
│              │  (Cross-scale    │                                │
│              │   validation)    │                                │
│              └──────────────────┘                                │
│                        │                                         │
│                        ▼                                         │
│              ┌──────────────────┐                                │
│              │     Decoder      │                                │
│              └──────────────────┘                                │
└─────────────────────────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                  Adversarial Test Suite                          │
│                                                                  │
│  ┌───────────────┐  ┌───────────────┐  ┌──────────────┐        │
│  │ Random        │  │ Adversarial   │  │ Targeted     │        │
│  │ Perturbations │  │ Attacks       │  │ Forgery      │        │
│  └───────────────┘  └───────────────┘  └──────────────┘        │
│                                                                  │
│  ┌──────────────────────────────────────────────────────┐       │
│  │ Metrics: Coherence, MI, Reconstruction Error         │       │
│  └──────────────────────────────────────────────────────┘       │
└─────────────────────────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    ACP Defense Planner                           │
│                                                                  │
│  Causal Graph → Counterfactual Reasoning → Defense Plans        │
└─────────────────────────────────────────────────────────────────┘
```

---

## Core Components

### 1. Multi-Manifold Forensic Engine

#### Purpose
Detect system tampering by encoding system state into multiple temporal scales and validating cross-scale coherence.

#### Components

##### Fast-Scale Encoder
- **Architecture**: MLP with variational bottleneck
- **Input**: Single timestep state (last observation)
- **Output**: (μ_fast, σ²_fast) → z_fast via reparameterization
- **Biological Analogy**: Hippocampal rapid encoding
- **Dimensions**: input_dim → hidden_dim → fast_latent_dim

```python
class FastDynamicsEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim):
        self.encoder = Sequential(
            Linear(input_dim, hidden_dim),
            ReLU(),
            Dropout(0.1),
            Linear(hidden_dim, hidden_dim // 2),
            ReLU()
        )
        self.mu = Linear(hidden_dim // 2, latent_dim)
        self.logvar = Linear(hidden_dim // 2, latent_dim)
```

**Key Features**:
- Captures rapid state changes
- Low-dimensional bottleneck enforces compression
- Variational formulation provides robustness

##### Slow-Scale Encoder
- **Architecture**: 1D Conv + MLP with variational bottleneck
- **Input**: Full sequence window
- **Output**: (μ_slow, σ²_slow) → z_slow via reparameterization
- **Biological Analogy**: Cortical consolidation
- **Dimensions**: (input_dim, seq_len) → hidden_dim → slow_latent_dim

```python
class SlowDynamicsEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim):
        self.temporal_conv = Conv1d(input_dim, hidden_dim,
                                     kernel_size=5, padding=2)
        # Global average pooling over time
        # Then MLP to μ, σ²
```

**Key Features**:
- Temporal convolution captures trends
- Global pooling aggregates long-term patterns
- Slower to respond to transient changes

##### Coherence Network
- **Purpose**: Validate consistency between fast and slow representations
- **Architecture**: MLP classifier
- **Input**: [z_fast; z_slow]
- **Output**: Coherence score ∈ [0, 1]

```python
coherence_net = Sequential(
    Linear(fast_dim + slow_dim, 128),
    ReLU(),
    Linear(128, 64),
    ReLU(),
    Linear(64, 1),
    Sigmoid()
)
```

**Coherence Loss**:
```python
L_coherence = 1 - coherence_score + λ * ||normalize(z_fast) - normalize(z_slow_proj)||²
```

This dual-term loss ensures:
1. High coherence score from network
2. Low L2 distance after normalization

##### Decoder
- **Purpose**: Reconstruct input from combined latent codes
- **Architecture**: MLP
- **Input**: [z_fast; z_slow]
- **Output**: Reconstructed state

**Total Loss**:
```python
L_total = λ_recon * L_recon + λ_coherence * L_coherence + λ_kl * L_kl
```

Where:
- L_recon: MSE between input and reconstruction
- L_coherence: Cross-scale consistency
- L_kl: KL divergence from N(0, I) for both fast and slow

---

### 2. Adversarial Perturbation Framework

#### Random Perturbations

**Purpose**: Evaluate robustness against non-targeted noise

**Method**:
```python
z_perturbed = z_original + ε * N(0, I)
```

Where ε ∈ [0, 5] is noise level

**Metrics**:
- Coherence degradation: Δ(L_coherence)
- Reconstruction error: ||x_recon - x_original||²
- Mutual information: MI(z_original, z_reconstructed)

#### Adversarial Attacks

**Purpose**: Test against gradient-based attacks

**Method**: Fast Gradient Sign Method (FGSM)
```python
# Compute gradient of coherence loss
∇_z L_coherence(z)

# Attack: maximize coherence loss
z_adversarial = z + α * sign(∇_z L_coherence)
```

**Iterative variant**:
```python
for i in range(num_iterations):
    ∇ = ∇_z L_coherence(z_i)
    z_{i+1} = z_i + α * sign(∇)
```

#### Targeted Forgery

**Purpose**: Test against state forgery attacks

**Method**:
```python
# Attacker tries to forge z_target
for i in range(num_iterations):
    L = ||z_i - z_target||²
    ∇ = ∇_z L
    z_{i+1} = z_i - α * sign(∇)
```

---

### 3. Mutual Information Estimation

#### Kraskov Estimator

**Purpose**: Quantify information preserved after perturbation

**Method**:
```python
MI(X; Y) = ψ(k) - <ψ(n_x + 1) + ψ(n_y + 1)> + ψ(N)
```

Where:
- ψ: Digamma function
- k: Number of nearest neighbors
- n_x, n_y: Neighbor counts in marginal spaces
- N: Sample size

**Implementation**:
1. Find k-th nearest neighbor in joint space (X, Y)
2. Count neighbors within ε/2 in marginal spaces
3. Apply Kraskov formula

**Simplified Alternative**:
Uses CCA (Canonical Correlation Analysis) as proxy:
```python
MI ≈ -0.5 * Σ log(1 - ρ_i²)
```

Where ρ_i are canonical correlations

---

### 4. ACP Integration

#### Causal Graph Structure

**Nodes**: System component metrics
- component_load
- component_latency
- component_throughput
- component_capacity
- component_resilience

**Edges**: Causal relationships with weights

**Example**:
```
load → latency (weight: 0.8)
load → throughput (weight: -0.6)
capacity → throughput (weight: 0.7)
```

#### Counterfactual Reasoning

**Process**:
1. **Abduction**: Infer noise from current state
   ```python
   ε = X_observed - W @ X_parents
   ```

2. **Action**: Intervene on system
   ```python
   do(X_component = new_value)
   ```

3. **Prediction**: Compute counterfactual
   ```python
   X_cf = W @ X_cf + ε
   # With intervened nodes fixed
   ```

#### Defense Plan Generation

**Steps**:
1. Detect anomalies (high load, latency)
2. For each anomaly:
   - Identify available interventions
   - Simulate counterfactual outcomes
   - Estimate efficacy = Δ(metric)
3. Rank by cost-effectiveness
4. Return top-k plans

**Intervention Types**:
- REINFORCE: Add capacity
- ISOLATE: Disconnect dependencies
- REROUTE: Change traffic flow
- HARDEN: Improve security
- THROTTLE: Rate limit

---

## Data Flow

### Training Flow

```
Raw System Logs
      ↓
Feature Engineering
      ↓
Trajectories (batch, seq_len, input_dim)
      ↓
Multi-Manifold Engine
      ↓
Latent Codes (z_fast, z_slow)
      ↓
Reconstruction + Coherence Loss
      ↓
Backprop + Optimization
```

### Inference Flow

```
System State Trajectory
      ↓
Encode to Latents
      ↓
Compute Coherence
      ↓
Threshold Check
      ↓
Tamper Detection Result
```

### Adversarial Testing Flow

```
Test Data
      ↓
Encode to Latents
      ↓
┌───────────────┬─────────────────┐
│ Random Perturb│ Adversarial     │
└───────────────┴─────────────────┘
      ↓               ↓
Perturbed Latents
      ↓
Decode + Re-encode
      ↓
Metrics (Coherence, MI, Error)
```

---

## Performance Characteristics

### Computational Complexity

| Operation | Complexity | Notes |
|-----------|-----------|--------|
| Fast Encoding | O(b × d × h) | b=batch, d=input_dim, h=hidden |
| Slow Encoding | O(b × s × d × h) | s=seq_len |
| Coherence | O(b × (d_f + d_s)) | d_f, d_s = latent dims |
| MI Estimation | O(n² × k) | n=samples, k=neighbors |
| ACP Counterfactual | O(n² + n × e) | n=nodes, e=edges |

### Memory Requirements

- **Forensic Engine**: ~10-50 MB (depending on hidden_dim)
- **Batch Storage**: batch_size × seq_len × input_dim × 4 bytes
- **Gradient Storage**: 2× model params during backprop

### Scalability

**Parallelization**:
- Batch encoding: Fully parallel
- MI estimation: Embarrassingly parallel across samples
- ACP simulation: Parallel across interventions

**Bottlenecks**:
- MI k-NN search: Can use approximate methods (Annoy, FAISS)
- Slow encoder convolution: Can reduce kernel size
- ACP graph traversal: Cache topological order

---

## Security Considerations

### Threat Model

**Attacker Capabilities**:
1. **White-box**: Full knowledge of forensic model
2. **Gradient access**: Can compute ∇L for attacks
3. **State manipulation**: Can modify latent codes

**Attacker Goals**:
- Evade tamper detection
- Forge coherent fake states
- Degrade system integrity

### Defense Mechanisms

1. **Multi-scale Redundancy**: Must attack both scales coherently
2. **Variational Robustness**: Stochastic encoding resists point attacks
3. **Coherence Constraints**: Hard to maintain across scales
4. **Mutual Information**: Detects information loss from attacks

### Limitations

- **Adaptive Attacks**: Attacker who trains against defenses
- **Transferability**: Attacks from similar models may transfer
- **Detection Threshold**: Trade-off between FP/FN rates

---

## Future Directions

1. **Attention Mechanisms**: Replace MLP with transformers for better temporal modeling
2. **Hierarchical Scales**: Extend to 3+ timescales
3. **Online Learning**: Adapt to distribution shift
4. **Differential Privacy**: Add DP guarantees to latent codes
5. **Certified Robustness**: Provable bounds on attack effectiveness

---

**Document Version**: 1.0
**Last Updated**: 2025-10-25
**Maintainer**: Claude Code Team
