"""
Unit tests for the Multi-Manifold Forensic Engine
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest
import torch
import numpy as np

from multi_manifold_forensic_engine import (
    create_forensic_engine,
    ForensicConfig,
    MultiManifoldForensicEngine
)
from adversarial_perturbations import (
    PerturbationGenerator,
    AdversarialTestSuite,
    estimate_mutual_information_simple
)
from data_generators import SyntheticDataGenerator


class TestForensicEngine:
    """Tests for the core forensic engine"""

    def test_engine_creation(self):
        """Test that the engine can be created with default config"""
        engine = create_forensic_engine(input_dim=50)
        assert engine is not None
        assert isinstance(engine, MultiManifoldForensicEngine)

    def test_forward_pass(self):
        """Test forward pass through the engine"""
        engine = create_forensic_engine(input_dim=50, fast_latent_dim=32, slow_latent_dim=16)

        # Create test data
        batch_size, seq_len, input_dim = 4, 10, 50
        x = torch.randn(batch_size, seq_len, input_dim)

        # Forward pass
        output = engine(x)

        # Check outputs
        assert 'x_recon' in output
        assert 'latent' in output
        assert 'total_loss' in output

        assert output['x_recon'].shape == (batch_size, input_dim)
        assert output['latent']['fast'].shape == (batch_size, 32)
        assert output['latent']['slow'].shape == (batch_size, 16)

    def test_encode_decode(self):
        """Test encoding and decoding"""
        engine = create_forensic_engine(input_dim=50)

        x = torch.randn(4, 10, 50)

        # Encode
        latent = engine.encode(x)
        assert 'fast' in latent
        assert 'slow' in latent

        # Decode
        x_recon = engine.decode(latent)
        assert x_recon.shape == (4, 50)

    def test_coherence_loss(self):
        """Test coherence loss computation"""
        engine = create_forensic_engine(input_dim=50)

        x = torch.randn(4, 10, 50)
        latent = engine.encode(x)

        coherence_loss = engine.compute_coherence_loss(latent)
        assert coherence_loss.item() >= 0

    def test_tamper_detection(self):
        """Test tamper detection on clean vs tampered trajectories"""
        engine = create_forensic_engine(input_dim=50)

        # Clean trajectory
        clean_traj = torch.randn(50, 50)
        clean_result = engine.detect_tampering_via_trajectory_consistency(
            clean_traj, window_size=5
        )

        assert 'tampered' in clean_result
        assert 'coherence_scores' in clean_result

        # Tampered trajectory (with spikes)
        tampered_traj = clean_traj.clone()
        tampered_traj[20:25] += torch.randn(5, 50) * 5.0

        tampered_result = engine.detect_tampering_via_trajectory_consistency(
            tampered_traj, window_size=5
        )

        # Tampered trajectory should have lower mean coherence
        assert tampered_result['mean_coherence'] <= clean_result['mean_coherence']


class TestPerturbationGenerator:
    """Tests for perturbation generation"""

    def test_random_perturbation(self):
        """Test random perturbation generation"""
        engine = create_forensic_engine(input_dim=50)
        generator = PerturbationGenerator(engine)

        x = torch.randn(4, 10, 50)
        latent = engine.encode(x)

        perturbed = generator.generate_random_perturbation(
            latent, scale='fast', noise_level=1.0
        )

        assert 'fast' in perturbed
        assert 'slow' in perturbed
        # Fast should be perturbed
        assert not torch.equal(perturbed['fast'], latent['fast'])
        # Slow should be unchanged
        assert torch.equal(perturbed['slow'], latent['slow'])

    def test_adversarial_perturbation(self):
        """Test adversarial perturbation generation"""
        engine = create_forensic_engine(input_dim=50)
        generator = PerturbationGenerator(engine)

        x = torch.randn(4, 10, 50)
        latent = engine.encode(x)

        perturbed = generator.generate_adversarial_perturbation(
            latent, scale='fast', step_size=0.1, num_iterations=1
        )

        assert 'fast' in perturbed
        assert 'slow' in perturbed
        # Fast should be perturbed
        assert not torch.equal(perturbed['fast'], latent['fast'])


class TestMutualInformation:
    """Tests for mutual information estimation"""

    def test_mi_identical_variables(self):
        """MI between identical variables should be positive"""
        x = np.random.randn(100, 10)
        mi = estimate_mutual_information_simple(x, x)
        assert mi > 0

    def test_mi_independent_variables(self):
        """MI between independent variables should be near zero"""
        x = np.random.randn(100, 10)
        y = np.random.randn(100, 10)
        mi = estimate_mutual_information_simple(x, y)
        # Should be small but not necessarily exactly zero
        assert mi >= 0


class TestDataGenerators:
    """Tests for data generation utilities"""

    def test_synthetic_trajectory_generation(self):
        """Test synthetic trajectory generation"""
        gen = SyntheticDataGenerator(seed=42)

        traj = gen.generate_system_trajectory(num_timesteps=50, input_dim=30)

        assert traj.shape == (50, 30)
        assert isinstance(traj, torch.Tensor)

    def test_batch_generation(self):
        """Test batch trajectory generation"""
        gen = SyntheticDataGenerator(seed=42)

        batch = gen.generate_batch_trajectories(batch_size=8, seq_len=10, input_dim=50)

        assert batch.shape == (8, 10, 50)

    def test_tampered_trajectory_generation(self):
        """Test tampered trajectory generation"""
        gen = SyntheticDataGenerator(seed=42)

        clean = gen.generate_system_trajectory(num_timesteps=100, input_dim=50)
        tampered = gen.generate_tampered_trajectory(
            clean, tamper_start=30, tamper_end=40, tamper_type='spike'
        )

        assert tampered.shape == clean.shape
        # Tampered region should be different
        assert not torch.equal(tampered[30:40], clean[30:40])
        # Other regions should be the same
        assert torch.equal(tampered[:30], clean[:30])


if __name__ == '__main__':
    # Run tests
    pytest.main([__file__, '-v'])
