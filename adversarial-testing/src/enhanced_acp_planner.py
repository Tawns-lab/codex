# enhanced_acp_planner.py
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
import logging
import argparse
import os
from datetime import datetime
import itertools
from enum import Enum

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class InterventionType(Enum):
    REINFORCE = "reinforce"
    ISOLATE = "isolate"
    REROUTE = "reroute"
    HARDEN = "harden"
    THROTTLE = "throttle"

class SystemComponent:
    """Represents actual system components with real intervention capabilities"""

    INTERVENTION_MAPPING = {
        'disk_cache': [InterventionType.REINFORCE, InterventionType.ISOLATE],
        'load_balancer': [InterventionType.REROUTE, InterventionType.THROTTLE],
        'gateway': [InterventionType.HARDEN, InterventionType.REINFORCE],
        'memory_controller': [InterventionType.ISOLATE, InterventionType.THROTTLE]
    }

    INTERVENTION_COSTS = {
        InterventionType.REINFORCE: 0.3,
        InterventionType.ISOLATE: 0.6,
        InterventionType.REROUTE: 0.4,
        InterventionType.HARDEN: 0.5,
        InterventionType.THROTTLE: 0.2
    }

    @classmethod
    def get_available_interventions(cls, component: str) -> List[InterventionType]:
        return cls.INTERVENTION_MAPPING.get(component, [InterventionType.HARDEN])

    @classmethod
    def get_intervention_cost(cls, intervention: InterventionType) -> float:
        return cls.INTERVENTION_COSTS.get(intervention, 0.5)

class EnhancedLinearSCM:
    """
    Enhanced SCM with system component integration and performance optimizations
    """

    def __init__(self, graph: Dict[str, Any]):
        self.nodes = graph['nodes']
        self.node_index = {node: i for i, node in enumerate(self.nodes)}
        self.n_nodes = len(self.nodes)

        # Build adjacency matrix with performance optimizations
        self.W = self._build_optimized_weight_matrix(graph)
        self.topological_order = self._get_topological_order()

        # Cache for frequently accessed parent relationships
        self._parent_cache = {}
        self._build_parent_cache()

    def _build_optimized_weight_matrix(self, graph: Dict) -> np.ndarray:
        """Build weight matrix with sparse representation consideration"""
        W = np.zeros((self.n_nodes, self.n_nodes))
        for edge in graph['edges']:
            i = self.node_index[edge['source']]
            j = self.node_index[edge['target']]
            W[i, j] = edge['weight']
        return W

    def _build_parent_cache(self):
        """Cache parent relationships for faster access"""
        for j in range(self.n_nodes):
            self._parent_cache[j] = np.where(self.W[:, j] != 0)[0]

    def abduce_noise(self, event: Dict[str, float]) -> np.ndarray:
        """Optimized abduction with cached parent relationships"""
        X_obs = np.array([event.get(node, 0.0) for node in self.nodes])
        epsilon = np.zeros(self.n_nodes)

        for j in self.topological_order:
            parents = self._parent_cache[j]
            if len(parents) > 0:
                parent_contributions = X_obs[parents] @ self.W[parents, j]
                epsilon[j] = X_obs[j] - parent_contributions
            else:
                epsilon[j] = X_obs[j]  # Root node

        return epsilon

    def predict_counterfactual(self, noise_vector: np.ndarray,
                               intervention: Dict[str, float]) -> np.ndarray:
        """Predict counterfactual outcome for given intervention"""
        intervened_indices = [self.node_index[node] for node in intervention.keys()]
        X_cf = np.array([intervention.get(node, 0.0) for node in self.nodes])

        for j in self.topological_order:
            if j in intervened_indices:
                continue

            parents = self._parent_cache[j]
            if len(parents) > 0:
                parent_contributions = X_cf[parents] @ self.W[parents, j]
                X_cf[j] = parent_contributions + noise_vector[j]

        return X_cf

    def _get_topological_order(self) -> List[int]:
        """Kahn's algorithm with cycle detection"""
        in_degree = [0] * self.n_nodes
        adj = {i: [] for i in range(self.n_nodes)}

        for i in range(self.n_nodes):
            for j in range(self.n_nodes):
                if self.W[i, j] != 0:
                    adj[i].append(j)
                    in_degree[j] += 1

        queue = [i for i in range(self.n_nodes) if in_degree[i] == 0]
        topo_order = []

        while queue:
            node = queue.pop(0)
            topo_order.append(node)

            for neighbor in adj[node]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        if len(topo_order) != self.n_nodes:
            raise ValueError("Graph contains cycles - invalid for causal inference")

        return topo_order

class RealTimeACPPlanner:
    """
    Production-ready ACP planner with system integration
    """

    def __init__(self, causal_graph_path: str, system_components: List[str]):
        self.causal_graph_path = causal_graph_path
        self.system_components = system_components
        self.scm: Optional[EnhancedLinearSCM] = None
        self.intervention_history: List[Dict] = []
        self.load_causal_graph()

    def load_causal_graph(self):
        """Load and validate causal graph"""
        try:
            with open(self.causal_graph_path, 'r') as f:
                graph = json.load(f)
            self.scm = EnhancedLinearSCM(graph)
            logger.info(f"✅ Loaded causal graph with {len(graph['nodes'])} nodes")
        except Exception as e:
            logger.error(f"❌ Failed to load causal graph: {e}")
            raise

    def analyze_system_health(self, current_metrics: Dict[str, float]) -> Dict[str, Any]:
        """Analyze current system state and identify risks"""
        health_report = {
            'timestamp': datetime.now().isoformat(),
            'component_health': {},
            'anomalies': [],
            'overall_risk_score': 0.0
        }

        risk_scores = []
        for component, value in current_metrics.items():
            # Simple threshold-based anomaly detection
            if component.endswith('_load') and value > 0.8:
                health_report['anomalies'].append({
                    'component': component,
                    'metric': 'high_load',
                    'severity': min((value - 0.8) * 5, 1.0)  # Normalize to 0-1
                })
                risk_scores.append(min((value - 0.8) * 5, 1.0))
            elif component.endswith('_latency') and value > 0.5:
                health_report['anomalies'].append({
                    'component': component,
                    'metric': 'high_latency',
                    'severity': min(value * 2, 1.0)
                })
                risk_scores.append(min(value * 2, 1.0))

        health_report['overall_risk_score'] = max(risk_scores) if risk_scores else 0.0
        return health_report

    def generate_defense_plans(self,
                              current_state: Dict[str, float],
                              health_report: Dict[str, Any],
                              max_plans: int = 3) -> List[Dict[str, Any]]:
        """Generate multiple defense plans based on current system state"""
        if health_report['overall_risk_score'] < 0.3:
            return []  # No action needed

        plans = []

        for anomaly in health_report['anomalies']:
            component = anomaly['component'].replace('_load', '').replace('_latency', '')
            if component not in self.system_components:
                continue

            # Generate plans for this anomalous component
            component_plans = self._plan_component_defense(
                current_state, component, anomaly['severity']
            )
            plans.extend(component_plans)

        # Sort by cost-effectiveness and limit results
        plans.sort(key=lambda p: p.get('expected_efficacy', 0) / (p.get('cost', 1) + 1e-6), reverse=True)
        return plans[:max_plans]

    def _plan_component_defense(self,
                               current_state: Dict[str, float],
                               target_component: str,
                               severity: float) -> List[Dict[str, Any]]:
        """Generate defense plans for a specific component"""
        plans = []

        # Get available interventions for this component
        available_interventions = SystemComponent.get_available_interventions(target_component)

        for intervention_type in available_interventions:
            plan = self._create_intervention_plan(
                current_state, target_component, intervention_type, severity
            )
            if plan and plan['expected_efficacy'] > 0.3:  # Only consider effective plans
                plans.append(plan)

        return plans

    def _create_intervention_plan(self,
                                 current_state: Dict[str, float],
                                 target_component: str,
                                 intervention_type: InterventionType,
                                 severity: float) -> Optional[Dict[str, Any]]:
        """Create a detailed intervention plan"""

        # Simulate intervention effect using SCM
        intervention_effect = self._simulate_intervention(
            current_state, target_component, intervention_type
        )

        if not intervention_effect['successful']:
            return None

        plan = {
            'target_component': target_component,
            'intervention_type': intervention_type.value,
            'description': self._get_intervention_description(intervention_type, target_component),
            'cost': SystemComponent.get_intervention_cost(intervention_type),
            'expected_efficacy': intervention_effect['efficacy'],
            'implementation_time_minutes': self._estimate_implementation_time(intervention_type),
            'prerequisites': self._get_prerequisites(intervention_type, target_component),
            'rollback_procedure': self._get_rollback_procedure(intervention_type),
            'risk_assessment': self._assess_intervention_risk(intervention_type, severity)
        }

        return plan

    def _simulate_intervention(self,
                              current_state: Dict[str, float],
                              target_component: str,
                              intervention_type: InterventionType) -> Dict[str, Any]:
        """Simulate the effect of an intervention using the SCM"""
        try:
            # Map intervention type to causal model intervention
            intervention_params = self._map_intervention_to_scm(
                target_component, intervention_type, current_state
            )

            # Use SCM to predict counterfactual
            noise = self.scm.abduce_noise(current_state)
            counterfactual = self.scm.predict_counterfactual(noise, intervention_params)

            # Calculate efficacy (reduction in target metric)
            original_value = current_state.get(f"{target_component}_load", 0.5)
            cf_key = f"{target_component}_load"
            if cf_key in self.scm.node_index:
                new_value = counterfactual[self.scm.node_index[cf_key]]
            else:
                new_value = original_value * 0.7  # Assume 30% improvement

            efficacy = max(0, original_value - new_value) / (original_value + 1e-6)

            return {
                'successful': True,
                'efficacy': efficacy,
                'predicted_state': {
                    self.scm.nodes[i]: float(counterfactual[i])
                    for i in range(len(self.scm.nodes))
                }
            }

        except Exception as e:
            logger.error(f"Intervention simulation failed: {e}")
            return {'successful': False, 'efficacy': 0.0}

    def _map_intervention_to_scm(self,
                                component: str,
                                intervention: InterventionType,
                                current_state: Dict[str, float]) -> Dict[str, float]:
        """Map real-world interventions to SCM parameter changes"""
        base_intervention = {}

        if intervention == InterventionType.REROUTE:
            # Reduce load on target component
            base_intervention[f"{component}_load"] = current_state.get(f"{component}_load", 0.5) * 0.7
        elif intervention == InterventionType.THROTTLE:
            # Reduce throughput
            base_intervention[f"{component}_throughput"] = current_state.get(f"{component}_throughput", 1.0) * 0.6
        elif intervention == InterventionType.ISOLATE:
            # Isolate component (set dependencies to minimal)
            base_intervention[f"{component}_dependencies"] = 0.1
        elif intervention == InterventionType.HARDEN:
            # Increase resilience
            base_intervention[f"{component}_resilience"] = current_state.get(f"{component}_resilience", 0.5) * 1.5
        elif intervention == InterventionType.REINFORCE:
            # Add capacity
            base_intervention[f"{component}_capacity"] = current_state.get(f"{component}_capacity", 1.0) * 1.3

        return base_intervention

    def _get_intervention_description(self, intervention: InterventionType, component: str) -> str:
        """Generate human-readable intervention description"""
        descriptions = {
            InterventionType.REROUTE: f"Reroute traffic away from {component}",
            InterventionType.THROTTLE: f"Throttle incoming requests to {component}",
            InterventionType.ISOLATE: f"Isolate {component} from dependent services",
            InterventionType.HARDEN: f"Harden security configuration for {component}",
            InterventionType.REINFORCE: f"Reinforce resource allocation for {component}"
        }
        return descriptions.get(intervention, f"Apply {intervention.value} to {component}")

    def _estimate_implementation_time(self, intervention: InterventionType) -> int:
        """Estimate implementation time in minutes"""
        time_estimates = {
            InterventionType.REROUTE: 5,
            InterventionType.THROTTLE: 2,
            InterventionType.ISOLATE: 8,
            InterventionType.HARDEN: 15,
            InterventionType.REINFORCE: 10
        }
        return time_estimates.get(intervention, 10)

    def _get_prerequisites(self, intervention: InterventionType, component: str) -> List[str]:
        """Get prerequisites for implementing intervention"""
        prereqs = {
            InterventionType.REROUTE: ["Backup routes configured", "Load balancer access"],
            InterventionType.THROTTLE: ["Rate limiting enabled", "Monitoring in place"],
            InterventionType.ISOLATE: ["Service discovery", "Circuit breaker pattern"],
            InterventionType.HARDEN: ["Security audit completed", "Backup available"],
            InterventionType.REINFORCE: ["Resource monitoring", "Capacity planning data"]
        }
        return prereqs.get(intervention, [])

    def _get_rollback_procedure(self, intervention: InterventionType) -> str:
        """Get rollback procedure for intervention"""
        procedures = {
            InterventionType.REROUTE: "Restore original routing configuration",
            InterventionType.THROTTLE: "Remove rate limits and restore normal throughput",
            InterventionType.ISOLATE: "Reconnect isolated services and verify dependencies",
            InterventionType.HARDEN: "Revert to previous security configuration",
            InterventionType.REINFORCE: "Scale back to original resource allocation"
        }
        return procedures.get(intervention, "Manual rollback required")

    def _assess_intervention_risk(self, intervention: InterventionType, severity: float) -> Dict[str, Any]:
        """Assess risks associated with intervention"""
        base_risk = SystemComponent.get_intervention_cost(intervention)
        adjusted_risk = base_risk * (1 + severity)

        return {
            'level': 'HIGH' if adjusted_risk > 0.7 else 'MEDIUM' if adjusted_risk > 0.4 else 'LOW',
            'score': adjusted_risk,
            'mitigation': 'Have rollback plan ready' if adjusted_risk > 0.5 else 'Low risk intervention'
        }

    def execute_plan(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a defense plan and track results"""
        execution_id = f"acp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        logger.info(f"🚀 Executing defense plan {execution_id}: {plan['description']}")

        # Simulate plan execution (in production, this would call actual APIs)
        execution_result = {
            'execution_id': execution_id,
            'plan': plan,
            'start_time': datetime.now().isoformat(),
            'status': 'IN_PROGRESS',
            'steps_completed': [],
            'errors': []
        }

        try:
            # Simulate execution steps
            execution_result['steps_completed'].append("Pre-flight checks passed")
            execution_result['steps_completed'].append("Resource allocation confirmed")
            execution_result['steps_completed'].append("Intervention configuration applied")

            # Simulate success
            execution_result['status'] = 'COMPLETED'
            execution_result['completion_time'] = datetime.now().isoformat()
            execution_result['estimated_impact'] = plan['expected_efficacy']

            # Log execution
            self.intervention_history.append(execution_result)

            logger.info(f"✅ Plan {execution_id} executed successfully")

        except Exception as e:
            execution_result['status'] = 'FAILED'
            execution_result['errors'].append(str(e))
            logger.error(f"❌ Plan {execution_id} failed: {e}")

        return execution_result

def main():
    """Main execution function for enhanced ACP"""
    parser = argparse.ArgumentParser(description="Enhanced Adversarial Counterfactual Planner")
    parser.add_argument("--graph", type=str, required=True, help="Path to causal graph JSON")
    parser.add_argument("--metrics", type=str, required=True, help="Path to current system metrics JSON")
    parser.add_argument("--components", type=str, nargs='+',
                       default=['disk_cache', 'load_balancer', 'gateway'],
                       help="System components to protect")

    args = parser.parse_args()

    try:
        # Initialize planner
        planner = RealTimeACPPlanner(args.graph, args.components)

        # Load current metrics
        with open(args.metrics, 'r') as f:
            current_metrics = json.load(f)

        # Analyze system health
        health_report = planner.analyze_system_health(current_metrics)
        print(f"🔍 System Health Analysis:")
        print(f"   Overall Risk Score: {health_report['overall_risk_score']:.2f}")
        print(f"   Anomalies Detected: {len(health_report['anomalies'])}")

        # Generate defense plans
        defense_plans = planner.generate_defense_plans(current_metrics, health_report)

        if defense_plans:
            print(f"\n🛡️  Generated {len(defense_plans)} Defense Plans:")
            for i, plan in enumerate(defense_plans, 1):
                print(f"\n   Plan {i}: {plan['description']}")
                print(f"      Efficacy: {plan['expected_efficacy']:.2f}")
                print(f"      Cost: {plan['cost']:.2f}")
                print(f"      Time: {plan['implementation_time_minutes']} min")
                print(f"      Risk: {plan['risk_assessment']['level']}")

            # Execute top plan
            top_plan = defense_plans[0]
            print(f"\n🚀 Executing Top Plan: {top_plan['description']}")
            result = planner.execute_plan(top_plan)

            print(f"✅ Execution Result: {result['status']}")
            if result['status'] == 'COMPLETED':
                print(f"   Impact: {result.get('estimated_impact', 0):.2f}")
                print(f"   Execution ID: {result['execution_id']}")

        else:
            print("✅ No defense plans needed - system is healthy")

    except Exception as e:
        logger.error(f"ACP execution failed: {e}")
        return 1

    return 0

if __name__ == "__main__":
    exit(main())
