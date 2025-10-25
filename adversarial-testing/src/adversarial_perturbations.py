"""
Adversarial Perturbation Testing Module

This module implements adversarial testing for the Multi-Manifold Forensic Engine.
It provides tools to:
1. Generate synthetic random and gradient-based adversarial perturbations
2. Measure coherence degradation under attack
3. Estimate mutual information to quantify robustness
4. Run comprehensive test suites

The goal is to validate that the forensic engine maintains integrity
even under targeted adversarial manipulation.
"""

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple, Optional, Any
import logging
from dataclasses import dataclass
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PerturbationConfig:
    """Configuration for perturbation generation"""
    random_noise_levels: List[float] = None
    adversarial_step_sizes: List[float] = None
    num_iterations: int = 10
    target_scales: List[str] = None

    def __post_init__(self):
        if self.random_noise_levels is None:
            self.random_noise_levels = np.linspace(0, 5.0, 20).tolist()
        if self.adversarial_step_sizes is None:
            self.adversarial_step_sizes = np.linspace(0, 0.5, 20).tolist()
        if self.target_scales is None:
            self.target_scales = ['fast', 'slow']


class PerturbationGenerator:
    """
    Generates synthetic random or adversarial perturbations for latent trajectories.

    This class implements two types of attacks:
    1. Random perturbations: Gaussian noise added to latent codes
    2. Adversarial perturbations: Gradient-based attacks that maximize coherence loss

    The adversarial attacks are particularly important for testing robustness,
    as they represent an attacker with knowledge of the forensic model.
    """

    def __init__(self, model: 'MultiManifoldForensicEngine'):
        """
        Initialize the perturbation generator.

        Args:
            model: The forensic engine to test
        """
        self.model = model

    def generate_random_perturbation(
        self,
        latent_trajectory: Dict[str, torch.Tensor],
        scale: str,
        noise_level: float
    ) -> Dict[str, torch.Tensor]:
        """
        Adds Gaussian noise to a specific scale of the latent trajectory.

        Args:
            latent_trajectory: Dictionary of latent trajectories for different scales
            scale: The specific scale to perturb ('fast' or 'slow')
            noise_level: The standard deviation of the Gaussian noise

        Returns:
            A new latent trajectory dictionary with the perturbation applied
        """
        perturbed_latent = {k: v.clone() for k, v in latent_trajectory.items()}

        if scale not in perturbed_latent:
            raise ValueError(f"Scale '{scale}' not found in latent trajectory")

        noise = torch.randn_like(perturbed_latent[scale]) * noise_level
        perturbed_latent[scale] = perturbed_latent[scale] + noise

        return perturbed_latent

    def generate_adversarial_perturbation(
        self,
        latent_trajectory: Dict[str, torch.Tensor],
        scale: str,
        step_size: float,
        num_iterations: int = 1
    ) -> Dict[str, torch.Tensor]:
        """
        Generates an adversarial perturbation using the gradient of the coherence loss.

        This implements a Fast Gradient Sign Method (FGSM) attack, modified to
        maximize coherence loss rather than minimize it.

        Args:
            latent_trajectory: Dictionary of latent trajectories
            scale: The scale to attack ('fast' or 'slow')
            step_size: The step size for the gradient ascent
            num_iterations: Number of attack iterations (default 1 for FGSM)

        Returns:
            A new latent trajectory dictionary with the adversarial perturbation
        """
        # Clone and enable gradients
        perturbed_latent = {
            'fast': latent_trajectory['fast'].clone().detach().requires_grad_(True),
            'slow': latent_trajectory['slow'].clone().detach().requires_grad_(True)
        }

        for _ in range(num_iterations):
            # Zero gradients
            if perturbed_latent['fast'].grad is not None:
                perturbed_latent['fast'].grad.zero_()
            if perturbed_latent['slow'].grad is not None:
                perturbed_latent['slow'].grad.zero_()

            # Calculate coherence loss
            coherence_loss = self.model.compute_coherence_loss(perturbed_latent)

            # Backpropagate to get gradients
            coherence_loss.backward()

            # Attack by moving in the direction of the gradient (gradient ascent)
            # This increases the coherence loss, making the system less coherent
            with torch.no_grad():
                if scale == 'fast':
                    grad = perturbed_latent['fast'].grad
                    perturbed_latent['fast'] = perturbed_latent['fast'] + step_size * grad.sign()
                    perturbed_latent['fast'].requires_grad_(True)
                elif scale == 'slow':
                    grad = perturbed_latent['slow'].grad
                    perturbed_latent['slow'] = perturbed_latent['slow'] + step_size * grad.sign()
                    perturbed_latent['slow'].requires_grad_(True)

        # Return detached tensors
        return {
            'fast': perturbed_latent['fast'].detach(),
            'slow': perturbed_latent['slow'].detach()
        }

    def generate_targeted_perturbation(
        self,
        latent_trajectory: Dict[str, torch.Tensor],
        target_latent: Dict[str, torch.Tensor],
        scale: str,
        step_size: float,
        num_iterations: int = 10
    ) -> Dict[str, torch.Tensor]:
        """
        Generate perturbation that moves latent code toward a specific target.

        This simulates an attacker trying to forge a specific system state.

        Args:
            latent_trajectory: Original latent trajectory
            target_latent: Target latent code to move toward
            scale: Scale to perturb
            step_size: Step size for optimization
            num_iterations: Number of optimization steps

        Returns:
            Perturbed latent trajectory
        """
        perturbed_latent = {
            'fast': latent_trajectory['fast'].clone().detach().requires_grad_(True),
            'slow': latent_trajectory['slow'].clone().detach().requires_grad_(True)
        }

        for _ in range(num_iterations):
            # Zero gradients
            if perturbed_latent[scale].grad is not None:
                perturbed_latent[scale].grad.zero_()

            # Calculate distance to target
            target_loss = F.mse_loss(perturbed_latent[scale], target_latent[scale])

            # Backpropagate
            target_loss.backward()

            # Move toward target
            with torch.no_grad():
                grad = perturbed_latent[scale].grad
                perturbed_latent[scale] = perturbed_latent[scale] - step_size * grad.sign()
                perturbed_latent[scale].requires_grad_(True)

        return {
            'fast': perturbed_latent['fast'].detach(),
            'slow': perturbed_latent['slow'].detach()
        }


def estimate_mutual_information(
    x: np.ndarray,
    y: np.ndarray,
    k: int = 5
) -> float:
    """
    Estimates mutual information using k-nearest neighbors (Kraskov et al. method).

    MI quantifies the amount of information shared between two variables.
    In our context:
    - High MI between original and reconstructed latents = good robustness
    - Low MI = information was lost/corrupted, poor resilience

    Args:
        x: First dataset (e.g., original latent trajectory)
        y: Second dataset (e.g., reconstructed latent trajectory)
        k: Number of nearest neighbors

    Returns:
        The estimated mutual information (in nats)
    """
    if len(x.shape) == 1:
        x = x.reshape(-1, 1)
    if len(y.shape) == 1:
        y = y.reshape(-1, 1)

    n_samples = x.shape[0]

    # Combine the datasets for joint nearest neighbor search
    xy = np.hstack([x, y])

    # Scale data for better nearest neighbor performance
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    scaler_xy = StandardScaler()

    x_scaled = scaler_x.fit_transform(x)
    y_scaled = scaler_y.fit_transform(y)
    xy_scaled = scaler_xy.fit_transform(xy)

    # Build k-NN models
    nn_xy = NearestNeighbors(n_neighbors=k+1, metric='euclidean')
    nn_xy.fit(xy_scaled)

    # Get k-th nearest neighbor distances in joint space
    distances_xy, _ = nn_xy.kneighbors(xy_scaled)
    epsilon = distances_xy[:, k]  # k-th neighbor distance

    # Count neighbors within epsilon in marginal spaces
    nn_x = NearestNeighbors(metric='chebyshev')
    nn_y = NearestNeighbors(metric='chebyshev')
    nn_x.fit(x_scaled)
    nn_y.fit(y_scaled)

    # Count neighbors within epsilon/2 (Chebyshev distance)
    n_x = np.array([len(nn_x.radius_neighbors([x_scaled[i]], radius=epsilon[i]/2.0)[1][0]) - 1
                    for i in range(n_samples)])
    n_y = np.array([len(nn_y.radius_neighbors([y_scaled[i]], radius=epsilon[i]/2.0)[1][0]) - 1
                    for i in range(n_samples)])

    # Compute MI using Kraskov estimator
    # MI(X;Y) = ψ(k) - <ψ(n_x + 1) + ψ(n_y + 1)> + ψ(N)
    # where ψ is the digamma function
    from scipy.special import digamma

    mi = digamma(k) - np.mean(digamma(n_x + 1) + digamma(n_y + 1)) + digamma(n_samples)

    return max(0.0, mi)  # MI should be non-negative


def estimate_mutual_information_simple(
    x: np.ndarray,
    y: np.ndarray,
    k: int = 5
) -> float:
    """
    Simplified mutual information estimator using correlation-based approach.

    This is faster but less accurate than the full Kraskov method.

    Args:
        x: First dataset
        y: Second dataset
        k: Number of nearest neighbors (for compatibility)

    Returns:
        Estimated MI
    """
    if len(x.shape) == 1:
        x = x.reshape(-1, 1)
    if len(y.shape) == 1:
        y = y.reshape(-1, 1)

    # Use canonical correlation analysis as a proxy
    from sklearn.cross_decomposition import CCA

    n_components = min(x.shape[1], y.shape[1], 5)
    cca = CCA(n_components=n_components)

    try:
        x_c, y_c = cca.fit_transform(x, y)
        # MI approximation from canonical correlations
        correlations = np.array([np.corrcoef(x_c[:, i], y_c[:, i])[0, 1]
                                for i in range(n_components)])
        # Transform correlations to MI approximation
        mi = -0.5 * np.sum(np.log(1 - correlations**2 + 1e-10))
        return max(0.0, mi)
    except:
        # Fallback: use simple correlation
        corr = np.corrcoef(x.flatten(), y.flatten())[0, 1]
        return max(0.0, -0.5 * np.log(1 - corr**2 + 1e-10))


class AdversarialTestSuite:
    """
    Comprehensive test suite for adversarial robustness evaluation.

    This suite runs multiple perturbation tests and collects metrics including:
    - Coherence loss degradation curves
    - Mutual information between original and perturbed states
    - Reconstruction quality
    - Cross-scale consistency
    """

    def __init__(
        self,
        forensic_engine: 'MultiManifoldForensicEngine',
        config: Optional[PerturbationConfig] = None
    ):
        """
        Initialize the test suite.

        Args:
            forensic_engine: The engine to test
            config: Perturbation configuration
        """
        self.engine = forensic_engine
        self.config = config or PerturbationConfig()
        self.generator = PerturbationGenerator(forensic_engine)
        self.results = {}

    def run_random_perturbation_test(
        self,
        original_latent: Dict[str, torch.Tensor],
        target_scale: str = 'fast'
    ) -> Dict[str, List[float]]:
        """
        Run random perturbation test across multiple noise levels.

        Args:
            original_latent: Original latent trajectory
            target_scale: Scale to perturb

        Returns:
            Dictionary of test metrics
        """
        logger.info(f"Running random perturbation test on '{target_scale}' scale...")

        results = {
            'noise_levels': [],
            'coherence_loss': [],
            'mutual_info': [],
            'reconstruction_error': []
        }

        # Get original latent as numpy for MI calculation
        orig_latent_np = original_latent[target_scale].detach().cpu().numpy()
        if orig_latent_np.ndim == 1:
            orig_latent_np = orig_latent_np.reshape(1, -1)

        for noise_level in self.config.random_noise_levels:
            # Generate perturbation
            perturbed_latent = self.generator.generate_random_perturbation(
                original_latent, target_scale, noise_level
            )

            # Measure coherence loss
            with torch.no_grad():
                coherence_loss = self.engine.compute_coherence_loss(perturbed_latent)
                results['coherence_loss'].append(coherence_loss.item())

            # Reconstruct from perturbed latent
            with torch.no_grad():
                reconstructed = self.engine.decode(perturbed_latent)
                # Re-encode to get reconstructed latent
                reencoded = self.engine.encode(reconstructed.unsqueeze(0))
                reconst_latent_np = reencoded[target_scale].detach().cpu().numpy()

                # Reconstruction error
                orig_decoded = self.engine.decode(original_latent)
                recon_error = F.mse_loss(reconstructed, orig_decoded).item()
                results['reconstruction_error'].append(recon_error)

            # Measure mutual information
            try:
                mi = estimate_mutual_information_simple(orig_latent_np, reconst_latent_np)
                results['mutual_info'].append(mi)
            except Exception as e:
                logger.warning(f"MI estimation failed: {e}")
                results['mutual_info'].append(0.0)

            results['noise_levels'].append(noise_level)

        logger.info(f"Random perturbation test complete. Peak coherence loss: "
                   f"{max(results['coherence_loss']):.4f}")

        return results

    def run_adversarial_perturbation_test(
        self,
        original_latent: Dict[str, torch.Tensor],
        target_scale: str = 'fast'
    ) -> Dict[str, List[float]]:
        """
        Run adversarial perturbation test across multiple step sizes.

        Args:
            original_latent: Original latent trajectory
            target_scale: Scale to attack

        Returns:
            Dictionary of test metrics
        """
        logger.info(f"Running adversarial perturbation test on '{target_scale}' scale...")

        results = {
            'step_sizes': [],
            'coherence_loss': [],
            'mutual_info': [],
            'reconstruction_error': []
        }

        orig_latent_np = original_latent[target_scale].detach().cpu().numpy()
        if orig_latent_np.ndim == 1:
            orig_latent_np = orig_latent_np.reshape(1, -1)

        for step_size in self.config.adversarial_step_sizes:
            # Generate adversarial perturbation
            perturbed_latent = self.generator.generate_adversarial_perturbation(
                original_latent, target_scale, step_size, num_iterations=self.config.num_iterations
            )

            # Measure coherence loss
            with torch.no_grad():
                coherence_loss = self.engine.compute_coherence_loss(perturbed_latent)
                results['coherence_loss'].append(coherence_loss.item())

            # Reconstruct and measure
            with torch.no_grad():
                reconstructed = self.engine.decode(perturbed_latent)
                reencoded = self.engine.encode(reconstructed.unsqueeze(0))
                reconst_latent_np = reencoded[target_scale].detach().cpu().numpy()

                orig_decoded = self.engine.decode(original_latent)
                recon_error = F.mse_loss(reconstructed, orig_decoded).item()
                results['reconstruction_error'].append(recon_error)

            # Measure MI
            try:
                mi = estimate_mutual_information_simple(orig_latent_np, reconst_latent_np)
                results['mutual_info'].append(mi)
            except Exception as e:
                logger.warning(f"MI estimation failed: {e}")
                results['mutual_info'].append(0.0)

            results['step_sizes'].append(step_size)

        logger.info(f"Adversarial perturbation test complete. Peak coherence loss: "
                   f"{max(results['coherence_loss']):.4f}")

        return results

    def run_full_suite(
        self,
        test_data: torch.Tensor
    ) -> Dict[str, Any]:
        """
        Run the complete adversarial test suite.

        Args:
            test_data: Input data for testing (batch, seq_len, input_dim)

        Returns:
            Comprehensive test results
        """
        logger.info("=" * 60)
        logger.info("Starting Full Adversarial Test Suite")
        logger.info("=" * 60)

        # Encode test data to get latent representation
        with torch.no_grad():
            original_latent = self.engine.encode(test_data)

        all_results = {}

        # Run tests for each scale
        for scale in self.config.target_scales:
            logger.info(f"\n--- Testing {scale.upper()} scale ---")

            # Random perturbations
            random_results = self.run_random_perturbation_test(original_latent, scale)
            all_results[f'{scale}_random'] = random_results

            # Adversarial perturbations
            adversarial_results = self.run_adversarial_perturbation_test(original_latent, scale)
            all_results[f'{scale}_adversarial'] = adversarial_results

        # Compute summary statistics
        summary = self._compute_summary_statistics(all_results)
        all_results['summary'] = summary

        logger.info("\n" + "=" * 60)
        logger.info("Test Suite Complete")
        logger.info("=" * 60)
        self._print_summary(summary)

        self.results = all_results
        return all_results

    def _compute_summary_statistics(self, results: Dict) -> Dict[str, Any]:
        """Compute summary statistics from test results."""
        summary = {}

        for test_name, test_results in results.items():
            if 'coherence_loss' in test_results:
                summary[f'{test_name}_max_coherence_loss'] = max(test_results['coherence_loss'])
                summary[f'{test_name}_mean_coherence_loss'] = np.mean(test_results['coherence_loss'])
                summary[f'{test_name}_min_mi'] = min(test_results['mutual_info'])
                summary[f'{test_name}_mean_mi'] = np.mean(test_results['mutual_info'])

        return summary

    def _print_summary(self, summary: Dict):
        """Print summary statistics."""
        print("\nSummary Statistics:")
        print("-" * 60)
        for key, value in summary.items():
            print(f"  {key}: {value:.4f}")

    def visualize_results(self, save_path: Optional[str] = None):
        """
        Visualize test results.

        Args:
            save_path: Path to save the figure (optional)
        """
        if not self.results:
            logger.warning("No results to visualize. Run tests first.")
            return

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Adversarial Robustness Test Results', fontsize=16)

        # Plot coherence degradation for random perturbations
        ax = axes[0, 0]
        for scale in self.config.target_scales:
            key = f'{scale}_random'
            if key in self.results:
                ax.plot(self.results[key]['noise_levels'],
                       self.results[key]['coherence_loss'],
                       label=f'{scale} scale', marker='o')
        ax.set_xlabel('Noise Level')
        ax.set_ylabel('Coherence Loss')
        ax.set_title('Random Perturbations: Coherence Degradation')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot MI for random perturbations
        ax = axes[0, 1]
        for scale in self.config.target_scales:
            key = f'{scale}_random'
            if key in self.results:
                ax.plot(self.results[key]['noise_levels'],
                       self.results[key]['mutual_info'],
                       label=f'{scale} scale', marker='s')
        ax.set_xlabel('Noise Level')
        ax.set_ylabel('Mutual Information')
        ax.set_title('Random Perturbations: Information Preservation')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot coherence degradation for adversarial perturbations
        ax = axes[1, 0]
        for scale in self.config.target_scales:
            key = f'{scale}_adversarial'
            if key in self.results:
                ax.plot(self.results[key]['step_sizes'],
                       self.results[key]['coherence_loss'],
                       label=f'{scale} scale', marker='o')
        ax.set_xlabel('Step Size')
        ax.set_ylabel('Coherence Loss')
        ax.set_title('Adversarial Perturbations: Coherence Degradation')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot MI for adversarial perturbations
        ax = axes[1, 1]
        for scale in self.config.target_scales:
            key = f'{scale}_adversarial'
            if key in self.results:
                ax.plot(self.results[key]['step_sizes'],
                       self.results[key]['mutual_info'],
                       label=f'{scale} scale', marker='s')
        ax.set_xlabel('Step Size')
        ax.set_ylabel('Mutual Information')
        ax.set_title('Adversarial Perturbations: Information Preservation')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Results saved to {save_path}")

        return fig


if __name__ == '__main__':
    # Demonstration
    print("Adversarial Perturbation Testing - Demonstration")
    print("=" * 60)

    # Import the forensic engine
    from multi_manifold_forensic_engine import create_forensic_engine

    # Create engine
    engine = create_forensic_engine(input_dim=50, fast_latent_dim=32, slow_latent_dim=16)

    # Generate test data
    test_data = torch.randn(8, 10, 50)

    # Create test suite
    config = PerturbationConfig(
        random_noise_levels=np.linspace(0, 3.0, 10).tolist(),
        adversarial_step_sizes=np.linspace(0, 0.3, 10).tolist()
    )
    test_suite = AdversarialTestSuite(engine, config)

    # Run tests
    results = test_suite.run_full_suite(test_data)

    print("\nTest complete! Results summary available in results['summary']")
