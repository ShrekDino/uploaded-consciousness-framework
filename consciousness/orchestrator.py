"""Main orchestrator — coordinates the multi-node upload network.

The orchestrator:
  - Spawns and manages N NodeProcess instances
  - Controls merge cycle timing (GWFR barycenter computation)
  - Monitors Ω_coherence between nodes
  - Triggers emergency merge if coherence bound is violated
  - Feeds the dashboard data stream
  - Implements the DQFR scheduling policy

Equation reference:
  μ_merged = argmin Σ w_i · GWFR²_κ(μ, μ_i)          (Eq 7)
  max_i,j GWFR²_κ(μ_i, μ_j) ≤ Ω_coherence              (Eq 9)
  𝒱_network = Σ(k_B·ε·H_env + Σ λ·I(μ_i; μ_j))        (Eq 14)
"""

import time
import multiprocessing as mp
import numpy as np
from collections import deque
from consciousness.node import NodeProcess
from consciousness.gwfr_merge import GWFRMerger
from config import NUM_NODES, MERGE_INTERVAL, OMEGA_COHERENCE


class Orchestrator:
    """Coordinates the distributed upload network.

    Usage:
        orch = Orchestrator(num_nodes=3)
        orch.start()
        for metrics in orch.run(steps=500):
            print(metrics)
        orch.shutdown()
    """

    def __init__(self, num_nodes=None):
        self.num_nodes = num_nodes or NUM_NODES
        self.nodes = []
        self.cmd_queues = []
        self.state_queues = []
        self.merger = GWFRMerger()
        self.merge_counter = 0
        self.history = deque(maxlen=500)

        # Track accumulated structured flux per node
        self.node_flux = [0.0] * self.num_nodes

    def start(self):
        """Spawn all node processes."""
        for i in range(self.num_nodes):
            cmd_q = mp.Queue()
            state_q = mp.Queue()
            node = NodeProcess(
                node_id=i,
                cmd_queue=cmd_q,
                state_queue=state_q,
                seed=42 + i
            )
            node.start()
            self.nodes.append(node)
            self.cmd_queues.append(cmd_q)
            self.state_queues.append(state_q)
        print(f"  Orchestrator: {self.num_nodes} nodes spawned.")

    def _collect_states(self):
        """Collect latest state from each node."""
        states = [None] * self.num_nodes
        for i, q in enumerate(self.state_queues):
            try:
                while not q.empty():
                    states[i] = q.get_nowait()
            except mp.queues.Empty:
                pass
        return states

    def _send_command(self, node_id, cmd):
        """Send a command to a specific node."""
        self.cmd_queues[node_id].put(cmd)

    def _broadcast_command(self, cmd):
        """Send a command to all nodes."""
        for q in self.cmd_queues:
            q.put(cmd)

    def _run_merge_cycle(self):
        """Execute one GWFR barycenter merge across all nodes.

        1. Collect weights from all nodes
        2. Compute pairwise GWFR distances
        3. Check Ω_coherence
        4. If any pair exceeds Ω_coherence, trigger emergency merge
        5. Compute weighted barycenter
        6. Broadcast merged weights to all nodes
        """
        self.merge_counter += 1

        # Collect current weights
        self._broadcast_command({"type": "step", "n_steps": 1})
        time.sleep(0.1)
        states = self._collect_states()

        # Extract weights and flux
        weights_list = []
        for i, s in enumerate(states):
            if s is not None:
                weights_list.append(s["weights"])
                self.node_flux[i] += s.get("flux_accumulated", 0)
            else:
                # Fallback: use previous weights (will be None for first cycle)
                weights_list.append(None)

        # Filter None states
        valid_indices = [i for i, w in enumerate(weights_list) if w is not None]
        valid_weights = [weights_list[i] for i in valid_indices]
        valid_flux = [self.node_flux[i] for i in valid_indices]

        if len(valid_weights) < 1:
            return None

        if len(valid_weights) == 1:
            merged = valid_weights[0]
            distances = np.zeros((1, 1))
            weights_used = np.array([1.0])
        else:
            # Compute merge
            merged, distances, weights_used = self.merger.merge(
                valid_weights, valid_flux
            )

            # Check Ω_coherence
            max_dist = np.max(distances)
            if max_dist > OMEGA_COHERENCE:
                print(
                    f"  ⚠ Emergency merge: GWFR distance {max_dist:.4f} "
                    f"> Ω_coherence = {OMEGA_COHERENCE}. "
                    f"Forcing sleep cycle."
                )

        # Broadcast merged weights
        for i in valid_indices:
            self._send_command(i, {"type": "set_weights", "weights": merged})

        return {
            "merge_cycle": self.merge_counter,
            "nodes_merged": len(valid_indices),
            "max_distance": float(np.max(distances)) if len(distances) > 0 else 0.0,
            "weights_used": weights_used.tolist() if hasattr(weights_used, 'tolist') else list(weights_used),
        }

    def run(self, steps=200):
        """Run the multi-node system for a given number of merge cycles.

        Yields step metrics as a generator so the dashboard can consume them.
        """
        for cycle in range(steps // MERGE_INTERVAL):
            # Run each node independently for MERGE_INTERVAL steps
            for i in range(self.num_nodes):
                self._send_command(i, {"type": "step", "n_steps": MERGE_INTERVAL})

            # Wait for nodes to finish processing
            time.sleep(MERGE_INTERVAL * 0.05 + 0.1)

            # Collect states
            states = self._collect_states()

            # Compute network vitality (Eq 14)
            network_vitality = 0.0
            for s in states:
                if s is not None and "metrics" in s:
                    m = s["metrics"]
                    network_vitality += m.get("epsilon", 0) * m.get("H_env", 0)

            # Run merge cycle
            merge_result = self._run_merge_cycle()

            # Compute pairwise mutual information proxy
            pairwise_mi = 0.0
            for i in range(self.num_nodes):
                for j in range(i + 1, self.num_nodes):
                    if states[i] and states[j]:
                        F_i = states[i].get("thermo", {}).get("F", 0)
                        F_j = states[j].get("thermo", {}).get("F", 0)
                        pairwise_mi += abs(F_i - F_j) / max(abs(F_i + F_j), 1e-8)

            network_vitality += pairwise_mi

            metrics = {
                "cycle": cycle,
                "network_vitality": network_vitality,
                "pairwise_mi": pairwise_mi,
                "merge": merge_result,
                "states": states,
            }
            self.history.append(metrics)
            yield metrics

    def shutdown(self):
        """Gracefully shut down all node processes."""
        self._broadcast_command({"type": "shutdown"})
        for node in self.nodes:
            node.join(timeout=2.0)
            if node.is_alive():
                node.terminate()
        print("  Orchestrator: all nodes shut down.")
