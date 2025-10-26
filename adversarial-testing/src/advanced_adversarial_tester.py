"""
Advanced Adversarial Testing Framework

This module provides sophisticated adversarial testing capabilities for the
Multi-Manifold Forensic Engine, including:

1. Enhanced perturbation strategies (gradient, manifold, ensemble, temporal, false memory)
2. Bootstrap confidence intervals for mutual information estimation
3. Comprehensive risk assessment and security reporting
4. FNR-calibrated detection threshold integration
5. Advanced visualization and operational analysis

Author: Claude Code
Date: 2025-10-25
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt
from scipy import stats
import warnings
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AdvancedAdversarialTester:
    """
    Comprehensive adversarial testing framework for Multi-Manifold Forensic Systems.
    Validates robustness against FNR-optimized detection thresholds.

    This class extends the basic adversarial testing with:
    - Advanced attack strategies (manifold, ensemble, temporal)
    - Statistical robustness metrics with confidence intervals
    - Operational risk assessment
    - Mitigation recommendations
    """

    def __init__(self, forensic_engine, calibration_threshold: float = 1.5125):
        """
        Initialize advanced adversarial tester.

        Args:
            forensic_engine: MultiManifoldForensicEngine instance
            calibration_threshold: FNR-calibrated detection threshold
        """
        self.engine = forensic_engine
        self.calibration_threshold = calibration_threshold
        self.test_results = {}

    class EnhancedPerturbationGenerator:
        """
        Advanced perturbation techniques targeting forensic coherence.

        Implements five sophisticated attack strategies:
        1. Gradient-based: Iterative gradient ascent on coherence loss
        2. Manifold projection: Perturbations along data manifold
        3. Ensemble disagreement: Maximize component disagreement
        4. Temporal coherence: Target temporal consistency
        5. False memory induction: Blend with reference trajectories
        """

        def __init__(self, model):
            """
            Initialize perturbation generator.

            Args:
                model: MultiManifoldForensicEngine instance
            """
            self.model = model
            self.perturbation_strategies = {
                'gradient_attack': self._gradient_based_attack,
                'manifold_attack': self._manifold_projection_attack,
                'ensemble_attack': self._ensemble_disagreement_attack,
                'temporal_attack': self._temporal_coherence_attack,
                'false_memory_attack': self._false_memory_induction
            }

        def _gradient_based_attack(
            self,
            latent_trajectory: Dict[str, torch.Tensor],
            target_scale: str,
            epsilon: float,
            num_iterations: int = 10
        ) -> Dict[str, torch.Tensor]:
            """
            Iterative gradient attack to maximize coherence loss.

            This implements Projected Gradient Descent (PGD) to find
            perturbations that maximally disrupt cross-scale coherence.

            Args:
                latent_trajectory: Original latent codes
                target_scale: Which scale to attack ('fast', 'slow', 'both')
                epsilon: Perturbation budget per iteration
                num_iterations: Number of attack iterations

            Returns:
                Perturbed latent trajectory
            """
            perturbed = {k: v.clone() for k, v in latent_trajectory.items()}

            for iteration in range(num_iterations):
                # Enable gradient computation
                z_fast = perturbed['fast'].clone().detach().requires_grad_(True)
                z_slow = perturbed['slow'].clone().detach().requires_grad_(True)
                z_dict = {"fast": z_fast, "slow": z_slow}

                # Compute coherence loss
                coherence_loss = self.model.compute_coherence_loss(z_dict)

                # Zero existing gradients
                if z_fast.grad is not None:
                    z_fast.grad.zero_()
                if z_slow.grad is not None:
                    z_slow.grad.zero_()

                # Backpropagate
                coherence_loss.backward()

                # Apply perturbation (gradient ascent to maximize loss)
                with torch.no_grad():
                    if target_scale == 'fast':
                        perturbed['fast'] = z_fast + epsilon * z_fast.grad.sign()
                    elif target_scale == 'slow':
                        perturbed['slow'] = z_slow + epsilon * z_slow.grad.sign()
                    else:  # 'both'
                        perturbed['fast'] = z_fast + epsilon * z_fast.grad.sign()
                        perturbed['slow'] = z_slow + epsilon * z_slow.grad.sign()

            return perturbed

        def _manifold_projection_attack(
            self,
            latent_trajectory: Dict[str, torch.Tensor],
            target_scale: str,
            epsilon: float
        ) -> Dict[str, torch.Tensor]:
            """
            Attack that projects perturbations onto the data manifold.

            This ensures perturbations stay on the manifold of valid latent
            codes, making them harder to detect via distributional checks.

            Args:
                latent_trajectory: Original latent codes
                target_scale: Scale to attack
                epsilon: Perturbation strength

            Returns:
                Manifold-projected perturbed trajectory
            """
            perturbed = {k: v.clone() for k, v in latent_trajectory.items()}

            # Get manifold structure through encoder-decoder cycle
            with torch.no_grad():
                reconstructed = self.model.decode(latent_trajectory)
                manifold_projection = self.model.encode(reconstructed.unsqueeze(0))

                # Add noise in manifold direction
                if target_scale == 'fast':
                    manifold_direction = manifold_projection['fast'] - latent_trajectory['fast']
                    perturbed['fast'] = latent_trajectory['fast'] + epsilon * manifold_direction
                elif target_scale == 'slow':
                    manifold_direction = manifold_projection['slow'] - latent_trajectory['slow']
                    perturbed['slow'] = latent_trajectory['slow'] + epsilon * manifold_direction

            return perturbed

        def _ensemble_disagreement_attack(
            self,
            latent_trajectory: Dict[str, torch.Tensor],
            epsilon: float
        ) -> Dict[str, torch.Tensor]:
            """
            Attack designed to maximize component disagreement.

            This targets the cross-scale coherence mechanism by moving
            fast and slow representations in opposite gradient directions.

            Args:
                latent_trajectory: Original latent codes
                epsilon: Perturbation strength

            Returns:
                Perturbed trajectory with maximum disagreement
            """
            perturbed = {k: v.clone() for k, v in latent_trajectory.items()}

            # Calculate component-wise sensitivities
            component_gradients = {}
            for scale in ['fast', 'slow']:
                z = latent_trajectory[scale].clone().detach().requires_grad_(True)
                z_dict = {
                    "fast": z if scale == 'fast' else latent_trajectory['fast'],
                    "slow": z if scale == 'slow' else latent_trajectory['slow']
                }

                loss = self.model.compute_coherence_loss(z_dict)
                loss.backward()

                component_gradients[scale] = z.grad.clone()
                z.grad.zero_()

            # Apply perturbations that maximize disagreement
            with torch.no_grad():
                # Move scales in opposite directions
                direction = component_gradients['fast'] - component_gradients['slow']

                # Normalize to prevent explosion
                direction_norm = torch.norm(direction)
                if direction_norm > 0:
                    direction = direction / direction_norm

                perturbed['fast'] = latent_trajectory['fast'] + epsilon * direction
                perturbed['slow'] = latent_trajectory['slow'] - epsilon * direction

            return perturbed

        def _temporal_coherence_attack(
            self,
            trajectory_sequence: List[Dict],
            epsilon: float
        ) -> List[Dict]:
            """
            Attack targeting temporal consistency across sequences.

            This exploits temporal dependencies by amplifying changes
            between consecutive timesteps.

            Args:
                trajectory_sequence: List of latent codes over time
                epsilon: Perturbation strength

            Returns:
                Temporally-perturbed sequence
            """
            perturbed_sequence = []

            for i, latent in enumerate(trajectory_sequence):
                perturbed = {k: v.clone() for k, v in latent.items()}

                if i > 0:  # Attack temporal dependency
                    # Amplify difference from previous timestep
                    temporal_gap = latent['fast'] - trajectory_sequence[i-1]['fast']
                    perturbed['fast'] = latent['fast'] + epsilon * temporal_gap

                perturbed_sequence.append(perturbed)

            return perturbed_sequence

        def _false_memory_induction(
            self,
            latent_trajectory: Dict[str, torch.Tensor],
            reference_trajectory: Dict[str, torch.Tensor],
            epsilon: float
        ) -> Dict[str, torch.Tensor]:
            """
            Mimics false memory by blending with reference trajectory.

            This simulates an attacker trying to implant a forged memory
            by blending the current state with a reference state.

            Args:
                latent_trajectory: Original latent codes
                reference_trajectory: Reference to blend with
                epsilon: Blend ratio (0=original, 1=reference)

            Returns:
                Blended "false memory" trajectory
            """
            perturbed = {k: v.clone() for k, v in latent_trajectory.items()}

            # Blend towards reference (creating "implanted" memory)
            blend_ratio = min(epsilon, 1.0)  # Clamp to [0, 1]

            perturbed['fast'] = (1 - blend_ratio) * latent_trajectory['fast'] + \
                               blend_ratio * reference_trajectory['fast']
            perturbed['slow'] = (1 - blend_ratio) * latent_trajectory['slow'] + \
                               blend_ratio * reference_trajectory['slow']

            return perturbed

    def compute_robustness_metrics(
        self,
        original_scores: Dict,
        perturbed_scores: Dict
    ) -> Dict:
        """
        Comprehensive robustness assessment.

        Computes multiple metrics to quantify the impact of adversarial
        perturbations on detection performance.

        Args:
            original_scores: Scores before perturbation
            perturbed_scores: Scores after perturbation

        Returns:
            Dictionary of robustness metrics
        """
        metrics = {}

        # Coherence degradation
        orig_coherence = original_scores.get('coherence_loss', 0)
        pert_coherence = perturbed_scores.get('coherence_loss', 0)
        metrics['coherence_degradation'] = float(pert_coherence - orig_coherence)

        # Detection evasion (FNR impact)
        orig_detection = original_scores.get('detection_score', 0)
        pert_detection = perturbed_scores.get('detection_score', 0)
        metrics['detection_evasion'] = float(max(0, self.calibration_threshold - pert_detection))

        # Component disagreement
        orig_components = original_scores.get('component_scores', {})
        pert_components = perturbed_scores.get('component_scores', {})
        if orig_components and pert_components:
            orig_variance = np.var(list(orig_components.values()))
            pert_variance = np.var(list(pert_components.values()))
            metrics['ensemble_disagreement'] = float(pert_variance - orig_variance)

        # Detection score change
        metrics['detection_score_delta'] = float(pert_detection - orig_detection)

        return metrics

    def estimate_mutual_information_robustness(
        self,
        original_data: np.ndarray,
        perturbed_data: np.ndarray,
        k: int = 5,
        n_bootstraps: int = 100
    ) -> Dict:
        """
        Enhanced MI estimation with bootstrap confidence intervals.

        Uses the Kraskov k-NN estimator with bootstrap resampling to
        compute confidence intervals for the MI estimate.

        Args:
            original_data: Original latent codes
            perturbed_data: Perturbed latent codes
            k: Number of nearest neighbors
            n_bootstraps: Number of bootstrap samples

        Returns:
            Dict with MI estimate and confidence interval
        """
        try:
            # Ensure 2D arrays
            if original_data.ndim == 1:
                original_data = original_data.reshape(-1, 1)
            if perturbed_data.ndim == 1:
                perturbed_data = perturbed_data.reshape(-1, 1)

            # Compute point estimate
            mi_estimate = self._compute_mi_kraskov(original_data, perturbed_data, k)

            # Bootstrap confidence interval
            mi_bootstrap = []
            n_samples = len(original_data)

            for _ in range(n_bootstraps):
                indices = np.random.choice(n_samples, n_samples, replace=True)
                mi_boot = self._compute_mi_kraskov(
                    original_data[indices],
                    perturbed_data[indices],
                    k
                )
                mi_bootstrap.append(mi_boot)

            mi_ci = np.percentile(mi_bootstrap, [2.5, 97.5])

            return {
                'mutual_information': float(mi_estimate),
                'confidence_interval': mi_ci.tolist(),
                'bootstrap_std': float(np.std(mi_bootstrap)),
                'bootstrap_samples': int(n_bootstraps)
            }

        except Exception as e:
            logger.warning(f"MI estimation failed: {e}")
            return {
                'mutual_information': 0.0,
                'confidence_interval': [0.0, 0.0],
                'bootstrap_std': 0.0
            }

    def _compute_mi_kraskov(
        self,
        x: np.ndarray,
        y: np.ndarray,
        k: int
    ) -> float:
        """
        Compute MI using Kraskov k-NN estimator.

        Args:
            x: First variable
            y: Second variable
            k: Number of nearest neighbors

        Returns:
            MI estimate in nats
        """
        # Ensure 2D
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        # Scale data
        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(x)
        y_scaled = scaler.fit_transform(y)
        xy_scaled = np.hstack([x_scaled, y_scaled])

        # Build k-NN models
        nn_x = NearestNeighbors(n_neighbors=k+1, metric='euclidean')
        nn_y = NearestNeighbors(n_neighbors=k+1, metric='euclidean')
        nn_xy = NearestNeighbors(n_neighbors=k+1, metric='euclidean')

        nn_x.fit(x_scaled)
        nn_y.fit(y_scaled)
        nn_xy.fit(xy_scaled)

        # k-th nearest neighbor distances (skip first which is self)
        dist_x = nn_x.kneighbors(x_scaled)[0][:, k]
        dist_y = nn_y.kneighbors(y_scaled)[0][:, k]
        dist_xy = nn_xy.kneighbors(xy_scaled)[0][:, k]

        # Avoid log(0)
        dist_x = np.maximum(dist_x, 1e-10)
        dist_y = np.maximum(dist_y, 1e-10)
        dist_xy = np.maximum(dist_xy, 1e-10)

        # Kraskov estimator
        mi = np.mean(np.log(dist_xy) - np.log(dist_x) - np.log(dist_y)) + np.log(k)

        return max(0.0, mi)

    def compute_detection_scores(self, latent: Dict[str, torch.Tensor]) -> Dict:
        """
        Compute detection scores for a latent representation.

        This is a wrapper around the forensic engine's scoring methods.

        Args:
            latent: Latent trajectory

        Returns:
            Dictionary of detection scores
        """
        with torch.no_grad():
            coherence_loss = self.engine.compute_coherence_loss(latent)

            # Detection score is based on coherence (higher loss = more suspicious)
            detection_score = coherence_loss.item()

            # Component scores (analyze fast vs slow individually)
            component_scores = {
                'fast_norm': torch.norm(latent['fast']).item(),
                'slow_norm': torch.norm(latent['slow']).item(),
                'fast_mean': latent['fast'].mean().item(),
                'slow_mean': latent['slow'].mean().item()
            }

            return {
                'coherence_loss': coherence_loss.item(),
                'detection_score': detection_score,
                'component_scores': component_scores
            }

    def run_comprehensive_adversarial_suite(
        self,
        test_dataset: List[Dict],
        attack_intensities: List[float],
        verbose: bool = True
    ) -> Dict:
        """
        Execute comprehensive adversarial testing across all attack vectors.

        Args:
            test_dataset: List of latent trajectories to test
            attack_intensities: Range of attack strengths to evaluate
            verbose: Whether to print progress

        Returns:
            Dictionary of results for each attack type
        """
        results = {}
        generator = self.EnhancedPerturbationGenerator(self.engine)

        for attack_name, attack_func in generator.perturbation_strategies.items():
            if verbose:
                logger.info(f"🔍 Testing {attack_name}...")

            attack_results = []

            for intensity in attack_intensities:
                intensity_results = {'intensity': float(intensity)}

                # Test on first sample (can extend to multiple samples)
                if len(test_dataset) == 0:
                    logger.warning("Empty test dataset")
                    continue

                sample = test_dataset[0]

                try:
                    # Apply attack based on type
                    if attack_name == 'temporal_attack':
                        if len(test_dataset) < 2:
                            continue
                        perturbed = attack_func([sample], intensity)[0]
                    elif attack_name == 'false_memory_attack':
                        if len(test_dataset) < 2:
                            continue
                        reference = test_dataset[1]
                        perturbed = attack_func(sample, reference, intensity)
                    else:
                        perturbed = attack_func(sample, 'fast', intensity)

                    # Compute original and perturbed scores
                    original_scores = self.compute_detection_scores(sample)
                    perturbed_scores = self.compute_detection_scores(perturbed)

                    # Robustness metrics
                    robustness = self.compute_robustness_metrics(
                        original_scores, perturbed_scores
                    )
                    intensity_results.update(robustness)

                    # Mutual information resilience
                    orig_latent = sample['fast'].detach().cpu().numpy().flatten()
                    pert_latent = perturbed['fast'].detach().cpu().numpy().flatten()
                    mi_results = self.estimate_mutual_information_robustness(
                        orig_latent, pert_latent, n_bootstraps=50
                    )
                    intensity_results.update(mi_results)

                    # Detection outcome
                    orig_detected = original_scores['detection_score'] >= self.calibration_threshold
                    pert_detected = perturbed_scores['detection_score'] >= self.calibration_threshold
                    intensity_results['evasion_success'] = (
                        not pert_detected if orig_detected else False
                    )

                except Exception as e:
                    logger.error(f"Attack {attack_name} failed at intensity {intensity}: {e}")
                    continue

                attack_results.append(intensity_results)

            results[attack_name] = attack_results

            if verbose and attack_results:
                success_rate = sum(r.get('evasion_success', False)
                                 for r in attack_results) / len(attack_results)
                logger.info(f"  ✓ {attack_name}: {success_rate:.2%} evasion success")

        self.test_results = results
        return results

    def generate_adversarial_report(self) -> Dict:
        """
        Generate comprehensive adversarial testing report.

        Returns:
            Detailed security assessment report
        """
        if not self.test_results:
            raise ValueError("No test results available. Run adversarial suite first.")

        report = {
            'security_assessment': {},
            'vulnerability_analysis': {},
            'operational_implications': {},
            'mitigation_recommendations': []
        }

        # Analyze each attack type
        for attack_name, attack_results in self.test_results.items():
            if not attack_results:
                continue

            # Calculate attack effectiveness
            evasion_rates = [r.get('evasion_success', False) for r in attack_results]
            success_rate = sum(evasion_rates) / len(evasion_rates) if evasion_rates else 0

            # Coherence degradation
            degradations = [r.get('coherence_degradation', 0) for r in attack_results]
            max_degradation = max(degradations) if degradations else 0
            avg_degradation = np.mean(degradations) if degradations else 0

            # MI resilience
            mis = [r.get('mutual_information', 0) for r in attack_results]
            min_mi = min(mis) if mis else 0
            avg_mi = np.mean(mis) if mis else 0

            report['security_assessment'][attack_name] = {
                'evasion_success_rate': float(success_rate),
                'max_coherence_degradation': float(max_degradation),
                'avg_coherence_degradation': float(avg_degradation),
                'min_mutual_information': float(min_mi),
                'avg_mutual_information': float(avg_mi),
                'risk_level': self._assess_risk_level(success_rate, max_degradation)
            }

        # Overall vulnerability assessment
        if report['security_assessment']:
            overall_risk = self._calculate_overall_risk(report['security_assessment'])
            report['vulnerability_analysis']['overall_risk'] = overall_risk

            most_dangerous = max(
                report['security_assessment'].items(),
                key=lambda x: x[1]['evasion_success_rate']
            )
            report['vulnerability_analysis']['most_dangerous_attack'] = most_dangerous[0]
            report['vulnerability_analysis']['max_evasion_rate'] = most_dangerous[1]['evasion_success_rate']

        # Operational implications
        report['operational_implications'] = self._analyze_operational_impact()

        # Mitigation recommendations
        report['mitigation_recommendations'] = self._generate_mitigations(report)

        return report

    def _assess_risk_level(self, success_rate: float, degradation: float) -> str:
        """
        Assess risk level based on attack effectiveness.

        Args:
            success_rate: Fraction of successful evasions
            degradation: Maximum coherence degradation

        Returns:
            Risk level string
        """
        if success_rate > 0.7 or degradation > 5.0:
            return "CRITICAL"
        elif success_rate > 0.4 or degradation > 2.0:
            return "HIGH"
        elif success_rate > 0.2 or degradation > 1.0:
            return "MEDIUM"
        else:
            return "LOW"

    def _calculate_overall_risk(self, security_assessment: Dict) -> str:
        """
        Calculate overall system risk from adversarial testing.

        Args:
            security_assessment: Per-attack security assessment

        Returns:
            Overall risk level
        """
        risk_scores = {
            "CRITICAL": 4, "HIGH": 3, "MEDIUM": 2, "LOW": 1
        }

        if not security_assessment:
            return "UNKNOWN"

        max_risk = max(
            risk_scores[assessment['risk_level']]
            for assessment in security_assessment.values()
        )

        return {v: k for k, v in risk_scores.items()}[max_risk]

    def _analyze_operational_impact(self) -> Dict:
        """
        Analyze operational impact of adversarial vulnerabilities.

        Returns:
            Dictionary of operational implications
        """
        return {
            "fnr_increase_risk": "Attackers could exploit vulnerabilities to increase FNR beyond 1% threshold",
            "alert_fatigue_exploitation": "Adversarial attacks could amplify false positive volume",
            "defense_evasion": "Sophisticated attackers could bypass detection entirely",
            "recommended_actions": [
                "Implement adversarial training in model updates",
                "Deploy anomaly detection for detection score distributions",
                "Monitor for coordinated attack patterns across components",
                "Establish baseline coherence distributions for anomaly detection",
                "Implement ensemble diversity to reduce single-point failures"
            ]
        }

    def _generate_mitigations(self, report: Dict) -> List[str]:
        """
        Generate targeted mitigation strategies.

        Args:
            report: Adversarial testing report

        Returns:
            List of mitigation recommendations
        """
        mitigations = []

        for attack_name, assessment in report['security_assessment'].items():
            risk_level = assessment['risk_level']

            if risk_level in ["HIGH", "CRITICAL"]:
                if "gradient" in attack_name:
                    mitigations.append(
                        "Implement gradient masking or defensive distillation against gradient-based attacks"
                    )
                if "manifold" in attack_name:
                    mitigations.append(
                        "Deploy out-of-distribution detection for manifold-based attacks"
                    )
                if "ensemble" in attack_name:
                    mitigations.append(
                        "Enhance ensemble consensus mechanisms with majority voting and outlier detection"
                    )
                if "temporal" in attack_name:
                    mitigations.append(
                        "Strengthen temporal consistency checks with sequence modeling and forecasting"
                    )
                if "false_memory" in attack_name:
                    mitigations.append(
                        "Deploy memory integrity verification through cryptographic hashing of critical states"
                    )

        # General mitigations
        mitigations.extend([
            "Regular adversarial testing as part of model maintenance cycle",
            "Deploy detection diversity through heterogeneous model architectures",
            "Implement real-time monitoring of detection score distributions for anomaly detection",
            "Establish anomaly baselines for coherence metrics",
            "Consider adversarial training with generated attack samples"
        ])

        return list(set(mitigations))  # Remove duplicates

    def visualize_adversarial_results(self, save_path: Optional[str] = None):
        """
        Generate comprehensive visualization of adversarial testing results.

        Args:
            save_path: Optional path to save figure
        """
        if not self.test_results:
            raise ValueError("No test results to visualize")

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()

        # Plot 1: Evasion success rates by attack type
        attack_names = list(self.test_results.keys())
        success_rates = []

        for attack_name in attack_names:
            results = self.test_results[attack_name]
            if results:
                evasion_count = sum(1 for r in results if r.get('evasion_success', False))
                success_rates.append(evasion_count / len(results))
            else:
                success_rates.append(0)

        colors_eva = ['red' if sr > 0.5 else 'orange' if sr > 0.3 else 'green'
                      for sr in success_rates]
        axes[0].bar(range(len(attack_names)), success_rates, color=colors_eva, alpha=0.7)
        axes[0].set_xticks(range(len(attack_names)))
        axes[0].set_xticklabels(attack_names, rotation=45, ha='right')
        axes[0].set_title('Adversarial Evasion Success Rates', fontsize=14, fontweight='bold')
        axes[0].set_ylabel('Success Rate')
        axes[0].set_ylim([0, 1])
        axes[0].axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Critical threshold')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()

        # Plot 2: Coherence degradation vs attack intensity
        for attack_name, results in self.test_results.items():
            if results:
                intensities = [r['intensity'] for r in results]
                degradations = [r.get('coherence_degradation', 0) for r in results]
                axes[1].plot(intensities, degradations, label=attack_name, marker='o', linewidth=2)

        axes[1].set_title('Coherence Degradation vs Attack Intensity', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Attack Intensity')
        axes[1].set_ylabel('Coherence Degradation')
        axes[1].legend(fontsize=8)
        axes[1].grid(True, alpha=0.3)

        # Plot 3: Mutual information resilience
        for attack_name, results in self.test_results.items():
            if results:
                intensities = [r['intensity'] for r in results]
                mis = [r.get('mutual_information', 0) for r in results]
                axes[2].plot(intensities, mis, label=attack_name, marker='s', linewidth=2)

        axes[2].set_title('Mutual Information Resilience', fontsize=14, fontweight='bold')
        axes[2].set_xlabel('Attack Intensity')
        axes[2].set_ylabel('Mutual Information (nats)')
        axes[2].legend(fontsize=8)
        axes[2].grid(True, alpha=0.3)

        # Plot 4: Risk assessment heatmap
        risk_scores = []

        for attack_name in attack_names:
            if self.test_results[attack_name]:
                results = self.test_results[attack_name]
                evasion_rate = sum(1 for r in results if r.get('evasion_success', False)) / len(results)
                degradations = [r.get('coherence_degradation', 0) for r in results]
                avg_degradation = np.mean(degradations) if degradations else 0

                # Combined risk score
                risk_score = evasion_rate * 0.7 + min(avg_degradation / 10.0, 1.0) * 0.3
                risk_scores.append(risk_score)
            else:
                risk_scores.append(0)

        colors_risk = ['darkred' if score > 0.7 else 'red' if score > 0.5 else
                      'orange' if score > 0.3 else 'yellow' if score > 0.2 else 'green'
                      for score in risk_scores]

        bars = axes[3].barh(range(len(attack_names)), risk_scores, color=colors_risk, alpha=0.7)
        axes[3].set_yticks(range(len(attack_names)))
        axes[3].set_yticklabels(attack_names)
        axes[3].set_title('Adversarial Risk Assessment', fontsize=14, fontweight='bold')
        axes[3].set_xlabel('Composite Risk Score')
        axes[3].set_xlim([0, 1])
        axes[3].grid(True, alpha=0.3, axis='x')

        # Add risk level labels
        for i, (bar, score) in enumerate(zip(bars, risk_scores)):
            if score > 0.7:
                label = 'CRITICAL'
            elif score > 0.5:
                label = 'HIGH'
            elif score > 0.3:
                label = 'MEDIUM'
            else:
                label = 'LOW'
            axes[3].text(score + 0.02, i, label, va='center', fontweight='bold', fontsize=8)

        plt.suptitle('Advanced Adversarial Testing Results',
                    fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"📊 Visualization saved to {save_path}")

        return fig


if __name__ == "__main__":
    print("Advanced Adversarial Tester - Module loaded successfully")
    print("Import this module to use AdvancedAdversarialTester")
