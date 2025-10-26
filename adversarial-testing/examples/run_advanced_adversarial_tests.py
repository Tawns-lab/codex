#!/usr/bin/env python3
"""
Advanced Adversarial Testing Example

This script demonstrates the advanced adversarial testing capabilities,
including sophisticated attack strategies and comprehensive security assessment.

Usage:
    python run_advanced_adversarial_tests.py [--output-dir OUTPUT_DIR] [--threshold THRESHOLD]
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import numpy as np
import argparse
import json
from pathlib import Path
from datetime import datetime

from multi_manifold_forensic_engine import create_forensic_engine
from advanced_adversarial_tester import AdvancedAdversarialTester
from data_generators import SyntheticDataGenerator


def print_section(title: str):
    """Print a formatted section header."""
    print("\n" + "=" * 70)
    print(f"{title}")
    print("=" * 70)


def generate_test_dataset(n_samples: int = 20, seq_len: int = 15, input_dim: int = 50):
    """Generate synthetic test dataset."""
    print_section("GENERATING TEST DATASET")

    data_gen = SyntheticDataGenerator(seed=42)

    # Generate diverse samples
    test_dataset = []
    for i in range(n_samples):
        # Generate trajectory
        trajectory = data_gen.generate_system_trajectory(
            num_timesteps=seq_len,
            input_dim=input_dim,
            anomaly_probability=0.05  # 5% anomaly probability
        )

        test_dataset.append(trajectory)

    print(f"✓ Generated {n_samples} test samples")
    print(f"  Sample shape: ({seq_len}, {input_dim})")

    return test_dataset


def create_latent_dataset(forensic_engine, trajectories):
    """Encode trajectories to latent space."""
    latent_dataset = []

    with torch.no_grad():
        for trajectory in trajectories:
            # Add batch dimension
            traj_batch = trajectory.unsqueeze(0)

            # Encode to latent space
            latent = forensic_engine.encode(traj_batch)

            # Store latent codes
            latent_dataset.append({
                'fast': latent['fast'].squeeze(0),
                'slow': latent['slow'].squeeze(0)
            })

    return latent_dataset


def run_advanced_tests(output_dir: Path, calibration_threshold: float, device: str = 'cpu'):
    """Run advanced adversarial testing suite."""

    # Create forensic engine
    print_section("INITIALIZING MULTI-MANIFOLD FORENSIC ENGINE")

    engine = create_forensic_engine(
        input_dim=50,
        fast_latent_dim=32,
        slow_latent_dim=16,
        device=device
    )

    print(f"✓ Forensic engine created")
    print(f"  Input dim: 50")
    print(f"  Fast latent dim: 32")
    print(f"  Slow latent dim: 16")
    print(f"  Device: {device}")

    # Generate test dataset
    trajectories = generate_test_dataset(n_samples=20, seq_len=15, input_dim=50)

    # Encode to latent space
    print("\n🔄 Encoding trajectories to latent space...")
    latent_dataset = create_latent_dataset(engine, trajectories)
    print(f"✓ Encoded {len(latent_dataset)} trajectories")

    # Initialize advanced adversarial tester
    print_section("INITIALIZING ADVANCED ADVERSARIAL TESTER")

    tester = AdvancedAdversarialTester(
        forensic_engine=engine,
        calibration_threshold=calibration_threshold
    )

    print(f"✓ Tester initialized")
    print(f"  Calibration threshold: {calibration_threshold}")
    print(f"  Available attacks:")
    generator = tester.EnhancedPerturbationGenerator(engine)
    for attack_name in generator.perturbation_strategies.keys():
        print(f"    - {attack_name}")

    # Define attack intensities
    attack_intensities = np.linspace(0.05, 1.5, 12)

    print(f"\n  Attack intensities: {len(attack_intensities)} levels")
    print(f"  Range: [{attack_intensities[0]:.2f}, {attack_intensities[-1]:.2f}]")

    # Run comprehensive adversarial suite
    print_section("RUNNING COMPREHENSIVE ADVERSARIAL TEST SUITE")
    print("This may take several minutes...")

    results = tester.run_comprehensive_adversarial_suite(
        test_dataset=latent_dataset,
        attack_intensities=attack_intensities.tolist(),
        verbose=True
    )

    # Save raw results
    results_path = output_dir / 'advanced_adversarial_results.json'

    # Convert to serializable format
    serializable_results = {}
    for attack_name, attack_results in results.items():
        serializable_results[attack_name] = [
            {k: (v.tolist() if isinstance(v, np.ndarray) else
                float(v) if isinstance(v, (np.float32, np.float64, np.int64)) else v)
             for k, v in result.items()}
            for result in attack_results
        ]

    with open(results_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)

    print(f"\n✓ Raw results saved to {results_path}")

    # Generate comprehensive report
    print_section("GENERATING SECURITY ASSESSMENT REPORT")

    report = tester.generate_adversarial_report()

    # Print report summary
    print("\n📋 SECURITY ASSESSMENT SUMMARY")
    print("-" * 70)

    for attack_name, assessment in report['security_assessment'].items():
        print(f"\n{attack_name}:")
        print(f"  Evasion success rate: {assessment['evasion_success_rate']:.1%}")
        print(f"  Max coherence degradation: {assessment['max_coherence_degradation']:.4f}")
        print(f"  Min mutual information: {assessment['min_mutual_information']:.4f}")
        print(f"  Risk level: {assessment['risk_level']}")

    print("\n" + "-" * 70)
    print(f"OVERALL RISK: {report['vulnerability_analysis']['overall_risk']}")
    print(f"Most dangerous attack: {report['vulnerability_analysis']['most_dangerous_attack']}")

    # Save report
    report_path = output_dir / 'security_assessment_report.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\n✓ Full report saved to {report_path}")

    # Generate visualizations
    print_section("GENERATING VISUALIZATIONS")

    viz_path = output_dir / 'advanced_adversarial_visualization.png'
    tester.visualize_adversarial_results(save_path=str(viz_path))

    print(f"✓ Visualization saved to {viz_path}")

    # Generate markdown summary report
    print_section("GENERATING MARKDOWN SUMMARY")

    markdown_report = generate_markdown_summary(report, results)
    md_path = output_dir / 'ADVERSARIAL_TESTING_REPORT.md'

    with open(md_path, 'w') as f:
        f.write(markdown_report)

    print(f"✓ Markdown report saved to {md_path}")

    return results, report


def generate_markdown_summary(report: dict, results: dict) -> str:
    """Generate a markdown-formatted summary report."""

    lines = []
    lines.append("# Advanced Adversarial Testing Report\n")
    lines.append(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    lines.append("---\n")

    # Executive Summary
    lines.append("\n## Executive Summary\n")
    lines.append(f"**Overall Risk Level**: {report['vulnerability_analysis']['overall_risk']}\n\n")
    lines.append(f"**Most Dangerous Attack**: {report['vulnerability_analysis']['most_dangerous_attack']}\n\n")

    max_evasion = report['vulnerability_analysis'].get('max_evasion_rate', 0)
    lines.append(f"**Maximum Evasion Rate**: {max_evasion:.1%}\n\n")

    # Detailed Attack Analysis
    lines.append("\n## Detailed Attack Analysis\n")

    for attack_name, assessment in report['security_assessment'].items():
        lines.append(f"\n### {attack_name.replace('_', ' ').title()}\n")
        lines.append(f"- **Risk Level**: {assessment['risk_level']}\n")
        lines.append(f"- **Evasion Success Rate**: {assessment['evasion_success_rate']:.1%}\n")
        lines.append(f"- **Max Coherence Degradation**: {assessment['max_coherence_degradation']:.4f}\n")
        lines.append(f"- **Avg Coherence Degradation**: {assessment['avg_coherence_degradation']:.4f}\n")
        lines.append(f"- **Min Mutual Information**: {assessment['min_mutual_information']:.4f}\n")
        lines.append(f"- **Avg Mutual Information**: {assessment['avg_mutual_information']:.4f}\n")

    # Operational Implications
    lines.append("\n## Operational Implications\n")

    ops = report['operational_implications']
    lines.append(f"\n**FNR Risk**: {ops['fnr_increase_risk']}\n")
    lines.append(f"\n**Alert Fatigue**: {ops['alert_fatigue_exploitation']}\n")
    lines.append(f"\n**Defense Evasion**: {ops['defense_evasion']}\n")

    lines.append("\n### Recommended Actions\n")
    for action in ops['recommended_actions']:
        lines.append(f"- {action}\n")

    # Mitigation Strategies
    lines.append("\n## Mitigation Recommendations\n")

    for i, mitigation in enumerate(report['mitigation_recommendations'], 1):
        lines.append(f"{i}. {mitigation}\n")

    # Technical Details
    lines.append("\n## Technical Details\n")
    lines.append(f"\n**Total Attack Types Tested**: {len(results)}\n")

    total_tests = sum(len(attack_results) for attack_results in results.values())
    lines.append(f"**Total Test Configurations**: {total_tests}\n")

    # Key Findings
    lines.append("\n## Key Findings\n")

    # Find most vulnerable attack
    most_vuln = max(
        report['security_assessment'].items(),
        key=lambda x: x[1]['evasion_success_rate']
    )

    lines.append(f"1. **Most Vulnerable to**: {most_vuln[0]} ({most_vuln[1]['evasion_success_rate']:.1%} evasion rate)\n")

    # Find most robust against
    least_vuln = min(
        report['security_assessment'].items(),
        key=lambda x: x[1]['evasion_success_rate']
    )

    lines.append(f"2. **Most Robust Against**: {least_vuln[0]} ({least_vuln[1]['evasion_success_rate']:.1%} evasion rate)\n")

    # Average metrics
    avg_evasion = np.mean([a['evasion_success_rate'] for a in report['security_assessment'].values()])
    lines.append(f"3. **Average Evasion Rate**: {avg_evasion:.1%}\n")

    lines.append("\n---\n")
    lines.append("\n*This report was generated by the Advanced Adversarial Testing Framework*\n")

    return ''.join(lines)


def main():
    parser = argparse.ArgumentParser(
        description='Run advanced adversarial testing on Multi-Manifold Forensic Engine'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='../test_results/advanced',
        help='Directory for output files'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=1.5125,
        help='FNR-calibrated detection threshold'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        choices=['cpu', 'cuda'],
        help='Device to run tests on'
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("ADVANCED ADVERSARIAL TESTING SUITE")
    print("=" * 70)
    print(f"\nOutput directory: {output_dir}")
    print(f"Calibration threshold: {args.threshold}")
    print(f"Device: {args.device}")

    try:
        # Run tests
        results, report = run_advanced_tests(output_dir, args.threshold, args.device)

        # Final summary
        print_section("TEST SUITE COMPLETE")

        print("\n✅ All tests completed successfully!")
        print(f"\n📊 Results Summary:")
        print(f"   - Tests run: {sum(len(r) for r in results.values())} configurations")
        print(f"   - Overall risk: {report['vulnerability_analysis']['overall_risk']}")
        print(f"   - Most dangerous: {report['vulnerability_analysis']['most_dangerous_attack']}")

        print(f"\n📁 Output files:")
        print(f"   - {output_dir / 'advanced_adversarial_results.json'}")
        print(f"   - {output_dir / 'security_assessment_report.json'}")
        print(f"   - {output_dir / 'advanced_adversarial_visualization.png'}")
        print(f"   - {output_dir / 'ADVERSARIAL_TESTING_REPORT.md'}")

        print("\n" + "=" * 70)

    except Exception as e:
        print(f"\n❌ TEST SUITE FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == '__main__':
    exit(main())
