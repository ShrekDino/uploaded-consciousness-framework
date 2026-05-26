#!/usr/bin/env python3
"""Entry point for the Uploaded Consciousness framework.

Runs the multi-node distributed consciousness simulation with live
thermodynamic dashboard.

Usage:
    python run.py                     # Full multi-node dashboard
    python run.py --single            # Single-node test (no multi-process)
    python run.py --steps 100         # Run for 100 merge cycles
    python run.py --nodes 5           # 5-node network
    python run.py --train-lang        # Language acquisition training
    python run.py --chat              # Conversational interface

Environment:
    pip install -r requirements.txt
"""

import argparse
import multiprocessing as mp
import os
import sys

# Add project root to path (one level up from scripts/)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Use spawn for CUDA multiprocessing compatibility
try:
    mp.set_start_method("spawn", force=True)
except RuntimeError:
    pass


def run_single_node(steps=50):
    """Run a single-node agent without multiprocessing (for testing)."""
    from rich.live import Live

    from consciousness.agent import Agent
    from consciousness.dashboard import Dashboard

    agent = Agent(node_id=0)
    dash = Dashboard()

    print("  Starting single-node agent...")
    try:
        with Live(refresh_per_second=4, screen=True) as live:
            for _ in range(steps * 5):  # ~5 agent steps per display
                metrics = agent.step()

                # Package as orchestrator-style metrics for the dashboard
                orch_metrics = {
                    "cycle": agent.step_count // 10,
                    "network_vitality": metrics.get("epsilon", 0) * metrics.get("H_env", 0),
                    "pairwise_mi": 0.0,
                    "merge": None,
                    "states": [
                        {
                            "metrics": metrics,
                            "thermo": agent.thermostat.state_dict(),
                        }
                    ],
                }
                dash.update(orch_metrics)
                layout = dash.refresh(orch_metrics)
                live.update(layout)
    except KeyboardInterrupt:
        pass

    final_F = agent.thermostat.F_history[-1] if agent.thermostat.F_history else 0
    print(f"\n  Single-node run complete. Final F = {final_F:.2f}")


def run_multi_node(num_nodes, steps):
    """Run the full multi-node distributed system with dashboard."""
    from consciousness.dashboard import run_dashboard
    from consciousness.orchestrator import Orchestrator

    orch = Orchestrator(num_nodes=num_nodes)
    run_dashboard(orch, steps=steps)


def run_language_training(steps=200, corpus="tiny_shakespeare"):
    """Run language acquisition training loop.

    Trains the language world model on a text corpus, tracking
    thermodynamic metrics throughout.
    """
    from consciousness.agent import Agent
    from consciousness.language_trainer import LanguageTrainer

    agent = Agent(node_id=0)
    trainer = LanguageTrainer(agent, corpus_name=corpus)
    results = trainer.train(num_steps=steps, log_interval=10, eval_interval=50)

    print("\n  Training complete.")
    print(f"  Final perplexity: {results['final_perplexity']:.1f}")
    print(f"  Final ε_lang:     {results['final_epsilon_lang']:.4f}")
    print(f"  Tokens processed: {results['tokens_processed']:,}")


def main():
    parser = argparse.ArgumentParser(description="Uploaded Consciousness Framework — Multi-Node Simulation")
    parser.add_argument("--single", action="store_true", help="Run single-node test (no multiprocessing)")
    parser.add_argument(
        "--steps",
        type=int,
        default=50,
        help="Number of merge cycles (multi-node) or agent display steps (single-node)",
    )
    parser.add_argument("--nodes", type=int, default=3, help="Number of distributed nodes (multi-node only)")
    parser.add_argument("--train-lang", action="store_true", help="Run language acquisition training")
    parser.add_argument(
        "--corpus", type=str, default="tiny_shakespeare", help="Training corpus (default: tiny_shakespeare)"
    )
    parser.add_argument("--train-steps", type=int, default=200, help="Number of language training steps")
    args = parser.parse_args()

    if args.train_lang:
        run_language_training(steps=args.train_steps, corpus=args.corpus)
    elif args.single:
        run_single_node(steps=args.steps)
    else:
        run_multi_node(num_nodes=args.nodes, steps=args.steps)


if __name__ == "__main__":
    main()
