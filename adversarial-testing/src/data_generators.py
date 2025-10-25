"""
Data Generators for Adversarial Testing

This module provides utilities for generating synthetic test data,
causal graphs, and system metrics for adversarial robustness testing.
"""

import numpy as np
import torch
import json
from typing import Dict, List, Tuple, Optional
from pathlib import Path


class SyntheticDataGenerator:
    """
    Generate synthetic system state trajectories for testing.
    """

    def __init__(self, seed: Optional[int] = 42):
        """
        Initialize the data generator.

        Args:
            seed: Random seed for reproducibility
        """
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)

    def generate_system_trajectory(
        self,
        num_timesteps: int = 100,
        input_dim: int = 50,
        anomaly_probability: float = 0.1
    ) -> torch.Tensor:
        """
        Generate a synthetic system state trajectory.

        Args:
            num_timesteps: Number of timesteps in the trajectory
            input_dim: Dimension of each state vector
            anomaly_probability: Probability of anomalous behavior

        Returns:
            Trajectory tensor of shape (num_timesteps, input_dim)
        """
        # Generate base trajectory with temporal correlation
        trajectory = torch.zeros(num_timesteps, input_dim)

        # Initialize with random state
        trajectory[0] = torch.randn(input_dim)

        # Generate correlated states
        for t in range(1, num_timesteps):
            # Autoregressive process with noise
            trajectory[t] = 0.7 * trajectory[t-1] + 0.3 * torch.randn(input_dim)

            # Inject anomalies
            if np.random.random() < anomaly_probability:
                # Spike anomaly
                trajectory[t] += torch.randn(input_dim) * 3.0

        return trajectory

    def generate_batch_trajectories(
        self,
        batch_size: int = 8,
        seq_len: int = 10,
        input_dim: int = 50
    ) -> torch.Tensor:
        """
        Generate a batch of trajectories.

        Args:
            batch_size: Number of trajectories
            seq_len: Length of each trajectory
            input_dim: Dimension of state vectors

        Returns:
            Batch tensor of shape (batch_size, seq_len, input_dim)
        """
        batch = torch.zeros(batch_size, seq_len, input_dim)

        for i in range(batch_size):
            trajectory = self.generate_system_trajectory(seq_len, input_dim)
            batch[i] = trajectory

        return batch

    def generate_tampered_trajectory(
        self,
        clean_trajectory: torch.Tensor,
        tamper_start: int,
        tamper_end: int,
        tamper_type: str = 'spike'
    ) -> torch.Tensor:
        """
        Generate a tampered version of a trajectory.

        Args:
            clean_trajectory: Original clean trajectory
            tamper_start: Start index of tampering
            tamper_end: End index of tampering
            tamper_type: Type of tampering ('spike', 'drift', 'replace')

        Returns:
            Tampered trajectory
        """
        tampered = clean_trajectory.clone()

        if tamper_type == 'spike':
            # Sudden spike in values
            tampered[tamper_start:tamper_end] += torch.randn_like(
                tampered[tamper_start:tamper_end]
            ) * 5.0
        elif tamper_type == 'drift':
            # Gradual drift
            drift = torch.linspace(0, 1, tamper_end - tamper_start).unsqueeze(-1)
            drift = drift * torch.randn(tampered.shape[-1]) * 3.0
            tampered[tamper_start:tamper_end] += drift
        elif tamper_type == 'replace':
            # Complete replacement
            tampered[tamper_start:tamper_end] = torch.randn_like(
                tampered[tamper_start:tamper_end]
            )

        return tampered


class CausalGraphGenerator:
    """
    Generate synthetic causal graphs for system modeling.
    """

    @staticmethod
    def generate_system_causal_graph(
        components: List[str] = None
    ) -> Dict:
        """
        Generate a causal graph for a system with multiple components.

        Args:
            components: List of component names

        Returns:
            Causal graph dictionary
        """
        if components is None:
            components = ['disk_cache', 'load_balancer', 'gateway', 'memory_controller']

        # Generate nodes for each component
        nodes = []
        for component in components:
            nodes.extend([
                f"{component}_load",
                f"{component}_latency",
                f"{component}_throughput",
                f"{component}_capacity",
                f"{component}_resilience"
            ])

        # Generate edges representing causal relationships
        edges = []

        for component in components:
            # Internal component relationships
            edges.append({
                'source': f"{component}_load",
                'target': f"{component}_latency",
                'weight': 0.8
            })
            edges.append({
                'source': f"{component}_load",
                'target': f"{component}_throughput",
                'weight': -0.6
            })
            edges.append({
                'source': f"{component}_capacity",
                'target': f"{component}_throughput",
                'weight': 0.7
            })
            edges.append({
                'source': f"{component}_resilience",
                'target': f"{component}_latency",
                'weight': -0.5
            })

        # Cross-component relationships
        if 'load_balancer' in components and 'gateway' in components:
            edges.append({
                'source': 'load_balancer_throughput',
                'target': 'gateway_load',
                'weight': 0.7
            })

        if 'disk_cache' in components and 'load_balancer' in components:
            edges.append({
                'source': 'disk_cache_latency',
                'target': 'load_balancer_latency',
                'weight': 0.5
            })

        return {
            'nodes': nodes,
            'edges': edges
        }

    @staticmethod
    def save_causal_graph(graph: Dict, path: str):
        """Save causal graph to JSON file."""
        with open(path, 'w') as f:
            json.dump(graph, f, indent=2)


class SystemMetricsGenerator:
    """
    Generate synthetic system metrics for testing.
    """

    @staticmethod
    def generate_healthy_metrics(
        components: List[str] = None
    ) -> Dict[str, float]:
        """
        Generate metrics for a healthy system state.

        Args:
            components: List of component names

        Returns:
            Dictionary of metrics
        """
        if components is None:
            components = ['disk_cache', 'load_balancer', 'gateway', 'memory_controller']

        metrics = {}

        for component in components:
            # Normal operating ranges
            metrics[f"{component}_load"] = np.random.uniform(0.3, 0.6)
            metrics[f"{component}_latency"] = np.random.uniform(0.1, 0.3)
            metrics[f"{component}_throughput"] = np.random.uniform(0.7, 0.9)
            metrics[f"{component}_capacity"] = np.random.uniform(0.8, 1.0)
            metrics[f"{component}_resilience"] = np.random.uniform(0.7, 0.9)

        return metrics

    @staticmethod
    def generate_stressed_metrics(
        components: List[str] = None,
        stressed_component: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Generate metrics for a stressed system state.

        Args:
            components: List of component names
            stressed_component: Specific component under stress

        Returns:
            Dictionary of metrics
        """
        if components is None:
            components = ['disk_cache', 'load_balancer', 'gateway', 'memory_controller']

        # Start with healthy metrics
        metrics = SystemMetricsGenerator.generate_healthy_metrics(components)

        # Stress a specific component or random one
        if stressed_component is None:
            stressed_component = np.random.choice(components)

        # Modify metrics to show stress
        metrics[f"{stressed_component}_load"] = np.random.uniform(0.85, 0.95)
        metrics[f"{stressed_component}_latency"] = np.random.uniform(0.6, 0.8)
        metrics[f"{stressed_component}_throughput"] = np.random.uniform(0.3, 0.5)

        return metrics

    @staticmethod
    def save_metrics(metrics: Dict[str, float], path: str):
        """Save metrics to JSON file."""
        with open(path, 'w') as f:
            json.dump(metrics, f, indent=2)


def generate_test_fixtures(output_dir: str = './data'):
    """
    Generate all test fixtures including graphs, metrics, and trajectories.

    Args:
        output_dir: Directory to save fixtures
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Generating test fixtures in {output_dir}...")

    # Generate causal graph
    components = ['disk_cache', 'load_balancer', 'gateway', 'memory_controller']
    graph_gen = CausalGraphGenerator()
    causal_graph = graph_gen.generate_system_causal_graph(components)
    graph_path = output_path / 'causal_graph.json'
    graph_gen.save_causal_graph(causal_graph, str(graph_path))
    print(f"  ✓ Causal graph: {graph_path}")

    # Generate healthy metrics
    metrics_gen = SystemMetricsGenerator()
    healthy_metrics = metrics_gen.generate_healthy_metrics(components)
    healthy_path = output_path / 'healthy_metrics.json'
    metrics_gen.save_metrics(healthy_metrics, str(healthy_path))
    print(f"  ✓ Healthy metrics: {healthy_path}")

    # Generate stressed metrics
    stressed_metrics = metrics_gen.generate_stressed_metrics(components)
    stressed_path = output_path / 'stressed_metrics.json'
    metrics_gen.save_metrics(stressed_metrics, str(stressed_path))
    print(f"  ✓ Stressed metrics: {stressed_path}")

    # Generate trajectories
    data_gen = SyntheticDataGenerator()

    # Clean trajectory
    clean_traj = data_gen.generate_system_trajectory(num_timesteps=200, input_dim=50)
    torch.save(clean_traj, str(output_path / 'clean_trajectory.pt'))
    print(f"  ✓ Clean trajectory: {output_path / 'clean_trajectory.pt'}")

    # Tampered trajectory
    tampered_traj = data_gen.generate_tampered_trajectory(
        clean_traj, tamper_start=100, tamper_end=120, tamper_type='spike'
    )
    torch.save(tampered_traj, str(output_path / 'tampered_trajectory.pt'))
    print(f"  ✓ Tampered trajectory: {output_path / 'tampered_trajectory.pt'}")

    # Batch of trajectories
    batch_traj = data_gen.generate_batch_trajectories(
        batch_size=16, seq_len=20, input_dim=50
    )
    torch.save(batch_traj, str(output_path / 'batch_trajectories.pt'))
    print(f"  ✓ Batch trajectories: {output_path / 'batch_trajectories.pt'}")

    print("\n✅ All test fixtures generated successfully!")
    return output_path


if __name__ == '__main__':
    # Generate test fixtures
    generate_test_fixtures('../data')
