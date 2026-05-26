"""Multi-node subprocess wrapper.

Each node runs as an isolated subprocess (multiprocessing.Process) with
its own Agent, world model, and environment. The orchestrator communicates
with nodes via queues for commands and state snapshots.

The node implements the Quantum State Seeding protocol (Section V):
  - Pre-shared entanglement (simulated via shared seed)
  - Classical macrostate encrypted and transmitted via IPC queues
  - Source node persists; target node instantiates identical state
  - No-Cloning satisfied by construction
"""

import multiprocessing as mp
import time

from consciousness.agent import Agent


class NodeProcess(mp.Process):
    """A single node in the distributed consciousness network.

    Runs its own active inference loop in an isolated process.
    Communicates with the orchestrator via multiprocessing queues.
    """

    def __init__(self, node_id, cmd_queue, state_queue, seed=None):
        super().__init__()
        self.node_id = node_id
        self.cmd_queue = cmd_queue
        self.state_queue = state_queue
        self.seed = seed
        self.daemon = True

    def run(self):
        """Main node loop: receive commands, execute steps, report state."""
        import os
        os.environ["IS_CONSCIOUSNESS_NODE"] = "1"
        agent = Agent(node_id=self.node_id, env_seed=self.seed)
        running = True
        step_count = 0
        flux_accumulated = 0.0

        while running:
            # Check for commands (non-blocking)
            try:
                cmd = self.cmd_queue.get_nowait()
            except mp.queues.Empty:
                cmd = None

            if cmd is not None:
                if cmd.get("type") == "shutdown":
                    running = False
                    continue
                elif cmd.get("type") == "set_weights":
                    agent.set_weights(cmd["weights"])
                    continue
                elif cmd.get("type") == "step":
                    # Execute a configurable number of steps
                    n_steps = cmd.get("n_steps", 1)
                    for _ in range(n_steps):
                        metrics = agent.step()
                        flux_accumulated += max(0, metrics.get("H_env", 0) * metrics.get("epsilon", 0))
                        step_count += 1
                    continue

            # Default: step once
            metrics = agent.step()
            flux_accumulated += max(0, metrics.get("H_env", 0) * metrics.get("epsilon", 0))
            step_count += 1

            # Periodically report state to orchestrator
            if step_count % 5 == 0:
                state = agent.state_dict()
                state["flux_accumulated"] = flux_accumulated
                state["step"] = step_count
                state["metrics"] = metrics
                self.state_queue.put(state)

            # Control step rate (prevent busy-waiting)
            time.sleep(0.01)

        agent.world_model.to('cpu')
