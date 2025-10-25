"""
Multi-Manifold Forensic Engine

This module implements a neuroscientifically-grounded forensic integrity engine
that shifts from discrete event logs to continuous state-space trajectories.
Based on distributed memory principles, it enables robust tamper detection
through cross-scale coherence analysis.

Architecture:
- Multi-timescale latent representations (fast/slow dynamics)
- Continuous trajectory encoding with VAE-based manifold learning
- Cross-scale coherence constraints for integrity validation
- Adversarial robustness through redundant encoding

Reference: Distributed memory systems with hippocampal-neocortical interactions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import numpy as np
from dataclasses import dataclass


@dataclass
class ForensicConfig:
    """Configuration for the Multi-Manifold Forensic Engine"""
    input_dim: int = 128
    fast_latent_dim: int = 64
    slow_latent_dim: int = 32
    hidden_dim: int = 256
    num_timescales: int = 2
    coherence_weight: float = 1.0
    reconstruction_weight: float = 1.0
    kl_weight: float = 0.1
    device: str = 'cpu'


class FastDynamicsEncoder(nn.Module):
    """
    Fast-timescale encoder capturing rapid system state changes.
    Analogous to hippocampal rapid encoding in memory systems.
    """

    def __init__(self, input_dim: int, latent_dim: int, hidden_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
        )
        self.mu = nn.Linear(hidden_dim // 2, latent_dim)
        self.logvar = nn.Linear(hidden_dim // 2, latent_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode input to fast-timescale latent representation.

        Returns:
            mu: Mean of latent distribution
            logvar: Log variance of latent distribution
        """
        h = self.encoder(x)
        mu = self.mu(h)
        logvar = self.logvar(h)
        return mu, logvar


class SlowDynamicsEncoder(nn.Module):
    """
    Slow-timescale encoder capturing gradual system trends.
    Analogous to cortical consolidation in memory systems.
    """

    def __init__(self, input_dim: int, latent_dim: int, hidden_dim: int):
        super().__init__()
        # Use temporal convolution to capture slow trends
        self.temporal_conv = nn.Conv1d(input_dim, hidden_dim, kernel_size=5, padding=2)
        self.encoder = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
        )
        self.mu = nn.Linear(hidden_dim // 2, latent_dim)
        self.logvar = nn.Linear(hidden_dim // 2, latent_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode input sequence to slow-timescale latent representation.

        Args:
            x: Input tensor of shape (batch, seq_len, input_dim)

        Returns:
            mu: Mean of latent distribution
            logvar: Log variance of latent distribution
        """
        # x shape: (batch, seq_len, input_dim)
        x_t = x.transpose(1, 2)  # (batch, input_dim, seq_len)
        h = self.temporal_conv(x_t)
        h = h.mean(dim=2)  # Global average pooling over time
        h = self.encoder(h)
        mu = self.mu(h)
        logvar = self.logvar(h)
        return mu, logvar


class ManifoldDecoder(nn.Module):
    """
    Decoder that reconstructs input from multi-scale latent representations.
    Integrates fast and slow dynamics for coherent reconstruction.
    """

    def __init__(self, fast_latent_dim: int, slow_latent_dim: int,
                 hidden_dim: int, output_dim: int):
        super().__init__()
        combined_dim = fast_latent_dim + slow_latent_dim
        self.decoder = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, z_fast: torch.Tensor, z_slow: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct input from multi-scale latent codes.

        Args:
            z_fast: Fast-timescale latent code
            z_slow: Slow-timescale latent code

        Returns:
            Reconstructed input
        """
        z_combined = torch.cat([z_fast, z_slow], dim=-1)
        return self.decoder(z_combined)


class MultiManifoldForensicEngine(nn.Module):
    """
    Core forensic engine implementing multi-manifold trajectory analysis.

    Key Features:
    - Dual-timescale latent representation (fast/slow dynamics)
    - Variational encoding for robustness
    - Cross-scale coherence constraints
    - Trajectory-based tamper detection

    The engine maintains consistency across temporal scales, making it
    difficult for attackers to forge coherent trajectories at all scales.
    """

    def __init__(self, config: ForensicConfig):
        super().__init__()
        self.config = config

        # Multi-scale encoders
        self.fast_encoder = FastDynamicsEncoder(
            config.input_dim, config.fast_latent_dim, config.hidden_dim
        )
        self.slow_encoder = SlowDynamicsEncoder(
            config.input_dim, config.slow_latent_dim, config.hidden_dim
        )

        # Shared decoder
        self.decoder = ManifoldDecoder(
            config.fast_latent_dim, config.slow_latent_dim,
            config.hidden_dim, config.input_dim
        )

        # Cross-scale coherence network
        self.coherence_net = nn.Sequential(
            nn.Linear(config.fast_latent_dim + config.slow_latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

        self.to(config.device)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick for VAE.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Encode input to multi-scale latent representations.

        Args:
            x: Input tensor of shape (batch, seq_len, input_dim) or (batch, input_dim)

        Returns:
            Dictionary containing:
                - fast: Fast-timescale latent code
                - slow: Slow-timescale latent code
                - fast_mu, fast_logvar: Fast encoder parameters
                - slow_mu, slow_logvar: Slow encoder parameters
        """
        # Handle both single-step and sequential inputs
        if x.dim() == 2:
            x = x.unsqueeze(1)  # Add sequence dimension

        # Fast encoding (on last timestep)
        fast_mu, fast_logvar = self.fast_encoder(x[:, -1, :])
        z_fast = self.reparameterize(fast_mu, fast_logvar)

        # Slow encoding (on full sequence)
        slow_mu, slow_logvar = self.slow_encoder(x)
        z_slow = self.reparameterize(slow_mu, slow_logvar)

        return {
            'fast': z_fast,
            'slow': z_slow,
            'fast_mu': fast_mu,
            'fast_logvar': fast_logvar,
            'slow_mu': slow_mu,
            'slow_logvar': slow_logvar
        }

    def decode(self, latent: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Decode from latent representation to input space.

        Args:
            latent: Dictionary containing 'fast' and 'slow' latent codes

        Returns:
            Reconstructed input
        """
        return self.decoder(latent['fast'], latent['slow'])

    def compute_coherence_loss(self, latent: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute cross-scale coherence loss.

        This measures how well the fast and slow representations align.
        Low coherence indicates potential tampering or anomalous behavior.

        Args:
            latent: Dictionary containing latent codes

        Returns:
            Coherence loss (lower is more coherent)
        """
        z_combined = torch.cat([latent['fast'], latent['slow']], dim=-1)
        coherence_score = self.coherence_net(z_combined)

        # Loss is inverse of coherence (we want high coherence)
        coherence_loss = 1.0 - coherence_score.mean()

        # Additional term: L2 distance between fast and slow (after projection)
        fast_proj = F.normalize(latent['fast'], dim=-1)
        slow_proj = F.normalize(latent['slow'][:, :latent['fast'].shape[-1]], dim=-1) \
                    if latent['slow'].shape[-1] >= latent['fast'].shape[-1] \
                    else F.pad(F.normalize(latent['slow'], dim=-1),
                              (0, latent['fast'].shape[-1] - latent['slow'].shape[-1]))

        alignment_loss = F.mse_loss(fast_proj, slow_proj)

        return coherence_loss + 0.5 * alignment_loss

    def compute_kl_divergence(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Compute KL divergence from standard normal.
        """
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1).mean()

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Full forward pass through the forensic engine.

        Args:
            x: Input tensor

        Returns:
            Dictionary containing all outputs and intermediate representations
        """
        # Encode to multi-scale latents
        latent = self.encode(x)

        # Decode to reconstruction
        x_recon = self.decode(latent)

        # Compute losses
        recon_loss = F.mse_loss(x_recon, x[:, -1, :] if x.dim() == 3 else x)
        coherence_loss = self.compute_coherence_loss(latent)
        kl_loss_fast = self.compute_kl_divergence(latent['fast_mu'], latent['fast_logvar'])
        kl_loss_slow = self.compute_kl_divergence(latent['slow_mu'], latent['slow_logvar'])
        kl_loss = kl_loss_fast + kl_loss_slow

        # Total loss
        total_loss = (
            self.config.reconstruction_weight * recon_loss +
            self.config.coherence_weight * coherence_loss +
            self.config.kl_weight * kl_loss
        )

        return {
            'x_recon': x_recon,
            'latent': latent,
            'total_loss': total_loss,
            'recon_loss': recon_loss,
            'coherence_loss': coherence_loss,
            'kl_loss': kl_loss
        }

    def detect_tampering_via_trajectory_consistency(
        self,
        trajectory: torch.Tensor,
        window_size: int = 10,
        threshold: float = 0.5
    ) -> Dict[str, any]:
        """
        Detect tampering by analyzing trajectory coherence over time.

        Args:
            trajectory: Sequence of system states (seq_len, input_dim)
            window_size: Size of sliding window for analysis
            threshold: Coherence threshold for tamper detection

        Returns:
            Detection results including tampered indices and confidence scores
        """
        self.eval()
        tampered_indices = []
        coherence_scores = []

        with torch.no_grad():
            # Sliding window analysis
            for i in range(len(trajectory) - window_size + 1):
                window = trajectory[i:i+window_size].unsqueeze(0)

                # Encode window
                latent = self.encode(window)

                # Compute coherence
                coherence_loss = self.compute_coherence_loss(latent)
                coherence_score = 1.0 - coherence_loss.item()
                coherence_scores.append(coherence_score)

                # Flag if below threshold
                if coherence_score < threshold:
                    tampered_indices.append(i + window_size - 1)

        return {
            'tampered': len(tampered_indices) > 0,
            'tampered_indices': tampered_indices,
            'coherence_scores': coherence_scores,
            'mean_coherence': np.mean(coherence_scores) if coherence_scores else 0.0,
            'min_coherence': np.min(coherence_scores) if coherence_scores else 0.0
        }

    def get_latent_trajectory(self, data: torch.Tensor) -> Dict[str, np.ndarray]:
        """
        Extract latent trajectories for analysis.

        Args:
            data: Sequential data (batch, seq_len, input_dim)

        Returns:
            Dictionary of latent trajectories for each scale
        """
        self.eval()
        with torch.no_grad():
            latent = self.encode(data)
            return {
                'fast': latent['fast'].cpu().numpy(),
                'slow': latent['slow'].cpu().numpy()
            }


def create_forensic_engine(
    input_dim: int = 128,
    fast_latent_dim: int = 64,
    slow_latent_dim: int = 32,
    device: str = 'cpu'
) -> MultiManifoldForensicEngine:
    """
    Factory function to create a forensic engine with standard configuration.

    Args:
        input_dim: Dimension of input features
        fast_latent_dim: Dimension of fast-timescale latent space
        slow_latent_dim: Dimension of slow-timescale latent space
        device: Device to run on ('cpu' or 'cuda')

    Returns:
        Configured MultiManifoldForensicEngine instance
    """
    config = ForensicConfig(
        input_dim=input_dim,
        fast_latent_dim=fast_latent_dim,
        slow_latent_dim=slow_latent_dim,
        device=device
    )
    return MultiManifoldForensicEngine(config)


if __name__ == '__main__':
    # Demonstration
    print("Multi-Manifold Forensic Engine - Demonstration")
    print("=" * 60)

    # Create engine
    engine = create_forensic_engine(input_dim=50, fast_latent_dim=32, slow_latent_dim=16)

    # Generate synthetic data
    batch_size, seq_len, input_dim = 8, 10, 50
    x = torch.randn(batch_size, seq_len, input_dim)

    # Forward pass
    output = engine(x)

    print(f"Input shape: {x.shape}")
    print(f"Reconstruction shape: {output['x_recon'].shape}")
    print(f"Fast latent shape: {output['latent']['fast'].shape}")
    print(f"Slow latent shape: {output['latent']['slow'].shape}")
    print(f"\nLosses:")
    print(f"  Total: {output['total_loss'].item():.4f}")
    print(f"  Reconstruction: {output['recon_loss'].item():.4f}")
    print(f"  Coherence: {output['coherence_loss'].item():.4f}")
    print(f"  KL: {output['kl_loss'].item():.4f}")

    # Tamper detection demo
    print(f"\n{'=' * 60}")
    print("Tamper Detection Demo")
    trajectory = torch.randn(100, 50)
    detection_result = engine.detect_tampering_via_trajectory_consistency(trajectory)
    print(f"Tampered: {detection_result['tampered']}")
    print(f"Mean coherence: {detection_result['mean_coherence']:.4f}")
    print(f"Number of suspicious windows: {len(detection_result['tampered_indices'])}")
