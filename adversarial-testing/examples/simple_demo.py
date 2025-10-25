#!/usr/bin/env python3
"""
Simple Demo of Adversarial Forensic Testing

A minimal demonstration that shows the core concepts without heavy computation.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

print("=" * 70)
print("ADVERSARIAL FORENSIC TESTING - SIMPLE DEMO")
print("=" * 70)

try:
    import torch
    import numpy as np

    print("\n✓ Dependencies loaded successfully")

    # Import our modules
    from multi_manifold_forensic_engine import create_forensic_engine
    from adversarial_perturbations import PerturbationGenerator
    from data_generators import SyntheticDataGenerator

    print("✓ Modules imported successfully")

    # Create a small forensic engine
    print("\n1. Creating Multi-Manifold Forensic Engine...")
    engine = create_forensic_engine(input_dim=20, fast_latent_dim=10, slow_latent_dim=5)
    print("   ✓ Engine created")
    print(f"   - Input dimension: 20")
    print(f"   - Fast latent dimension: 10")
    print(f"   - Slow latent dimension: 5")

    # Generate synthetic data
    print("\n2. Generating synthetic system trajectory...")
    data_gen = SyntheticDataGenerator(seed=42)
    trajectory = data_gen.generate_system_trajectory(num_timesteps=30, input_dim=20)
    print(f"   ✓ Trajectory generated: {trajectory.shape}")

    # Encode to latent space
    print("\n3. Encoding to multi-scale latent representation...")
    latent = engine.encode(trajectory.unsqueeze(0))
    print(f"   ✓ Fast latent: {latent['fast'].shape}")
    print(f"   ✓ Slow latent: {latent['slow'].shape}")

    # Test coherence
    print("\n4. Computing cross-scale coherence...")
    coherence_loss = engine.compute_coherence_loss(latent)
    print(f"   ✓ Coherence loss: {coherence_loss.item():.4f}")
    print(f"   ✓ Coherence score: {1.0 - coherence_loss.item():.4f}")

    # Apply random perturbation
    print("\n5. Testing robustness with random perturbation...")
    generator = PerturbationGenerator(engine)
    perturbed = generator.generate_random_perturbation(latent, scale='fast', noise_level=0.5)

    perturbed_coherence = engine.compute_coherence_loss(perturbed)
    print(f"   ✓ Perturbed coherence loss: {perturbed_coherence.item():.4f}")
    print(f"   ✓ Degradation: {perturbed_coherence.item() - coherence_loss.item():.4f}")

    # Apply adversarial perturbation
    print("\n6. Testing robustness with adversarial attack...")
    adversarial = generator.generate_adversarial_perturbation(
        latent, scale='fast', step_size=0.1, num_iterations=3
    )

    adversarial_coherence = engine.compute_coherence_loss(adversarial)
    print(f"   ✓ Adversarial coherence loss: {adversarial_coherence.item():.4f}")
    print(f"   ✓ Degradation: {adversarial_coherence.item() - coherence_loss.item():.4f}")

    # Tamper detection
    print("\n7. Testing tamper detection...")

    # Clean trajectory
    clean_result = engine.detect_tampering_via_trajectory_consistency(
        trajectory, window_size=5, threshold=0.5
    )
    print(f"   Clean trajectory:")
    print(f"     - Tampered: {clean_result['tampered']}")
    print(f"     - Mean coherence: {clean_result['mean_coherence']:.4f}")

    # Create tampered trajectory
    tampered_traj = data_gen.generate_tampered_trajectory(
        trajectory, tamper_start=15, tamper_end=20, tamper_type='spike'
    )

    tampered_result = engine.detect_tampering_via_trajectory_consistency(
        tampered_traj, window_size=5, threshold=0.5
    )
    print(f"   Tampered trajectory:")
    print(f"     - Tampered: {tampered_result['tampered']}")
    print(f"     - Mean coherence: {tampered_result['mean_coherence']:.4f}")
    print(f"     - Suspicious windows: {len(tampered_result['tampered_indices'])}")

    print("\n" + "=" * 70)
    print("✅ DEMO COMPLETED SUCCESSFULLY")
    print("=" * 70)

    print("\nKey Findings:")
    print(f"  • Random perturbation degrades coherence by {perturbed_coherence.item() - coherence_loss.item():.4f}")
    print(f"  • Adversarial attack degrades coherence by {adversarial_coherence.item() - coherence_loss.item():.4f}")
    print(f"  • Adversarial attacks are {(adversarial_coherence.item() - coherence_loss.item()) / (perturbed_coherence.item() - coherence_loss.item() + 1e-6):.2f}x more effective")
    print(f"  • Tamper detection successfully identified {len(tampered_result['tampered_indices'])} suspicious windows")

    print("\nNext Steps:")
    print("  • Run 'python run_full_test_suite.py' for comprehensive testing")
    print("  • See README.md for detailed documentation")

except ImportError as e:
    print(f"\n❌ Missing dependency: {e}")
    print("\nPlease install dependencies:")
    print("  pip install -r requirements.txt")
    sys.exit(1)

except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
