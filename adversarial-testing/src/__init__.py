"""
Adversarial Forensic Testing Framework

A neuroscientifically-grounded integrity engine for detecting system tampering
through multi-scale coherence analysis.
"""

from .multi_manifold_forensic_engine import (
    MultiManifoldForensicEngine,
    ForensicConfig,
    create_forensic_engine
)

from .adversarial_perturbations import (
    PerturbationGenerator,
    PerturbationConfig,
    AdversarialTestSuite,
    estimate_mutual_information,
    estimate_mutual_information_simple
)

from .data_generators import (
    SyntheticDataGenerator,
    CausalGraphGenerator,
    SystemMetricsGenerator,
    generate_test_fixtures
)

from .enhanced_acp_planner import (
    RealTimeACPPlanner,
    EnhancedLinearSCM,
    SystemComponent,
    InterventionType
)

from .advanced_adversarial_tester import (
    AdvancedAdversarialTester
)

__version__ = '0.2.0'

__all__ = [
    # Forensic Engine
    'MultiManifoldForensicEngine',
    'ForensicConfig',
    'create_forensic_engine',

    # Adversarial Testing
    'PerturbationGenerator',
    'PerturbationConfig',
    'AdversarialTestSuite',
    'AdvancedAdversarialTester',
    'estimate_mutual_information',
    'estimate_mutual_information_simple',

    # Data Generation
    'SyntheticDataGenerator',
    'CausalGraphGenerator',
    'SystemMetricsGenerator',
    'generate_test_fixtures',

    # ACP Planner
    'RealTimeACPPlanner',
    'EnhancedLinearSCM',
    'SystemComponent',
    'InterventionType',
]
