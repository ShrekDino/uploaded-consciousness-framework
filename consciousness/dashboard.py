"""Real-time thermodynamic dashboard for the upload network.

Uses Rich for a live-updating terminal UI showing:
  - Per-node metrics: F, S_gen, ε, phase
  - Network-level: 𝒱_network, pairwise MI, merge events
  - Live plots: F(t), S_gen(t), ε(t)
  - Ω_coherence gauge

The dashboard renders in the terminal using Rich's Layout, Panel,
Table, and Progress widgets, with plotext for ASCII plots.
"""

import time
import numpy as np
from rich.live import Live
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.progress import Progress, BarColumn, TextColumn
from rich import box
from config import PLOT_WINDOW


def make_ascii_plot(history, height=8, width=40):
    """Render a simple ASCII sparkline."""
    if len(history) < 2:
        return "[dim]insufficient data[/dim]"
    try:
        h = np.array(history[-width:], dtype=np.float64)
        h_min, h_max = h.min(), h.max()
        if h_max - h_min < 1e-12:
            return f"[dim]flat: {float(h.mean()):.2f}[/dim]"
        h_range = h_max - h_min
        normalized = ((h - h_min) / h_range * (height - 1)).astype(int)
        bars = []
        for level in range(height - 1, -1, -1):
            row = ""
            for v in normalized:
                row += "█" if v >= level else " "
            bars.append(row)
        return "\n".join(bars) + f"\n[min={h_min:.1f}, max={h_max:.1f}]"
    except Exception:
        return "[dim]plot error[/dim]"


class Dashboard:
    """Live terminal dashboard for the uploaded consciousness network.

    Usage:
        dash = Dashboard()
        with dash:
            for metrics in orchestrator.run(steps=500):
                dash.update(metrics)
                dash.refresh()
    """

    def __init__(self):
        self.history_F = []
        self.history_S_gen = []
        self.history_epsilon = []
        self.history_network_vitality = []
        self.node_labels = []

    def _build_layout(self, metrics):
        """Construct the Rich Layout for the current metrics."""
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="body"),
            Layout(name="footer", size=3),
        )
        layout["body"].split_row(
            Layout(name="nodes", ratio=2),
            Layout(name="plots", ratio=3),
        )

        # ─── Header ───
        header = Text()
        header.append("UPLOADED CONSCIOUSNESS ", style="bold cyan")
        header.append("— Thermodynamic Dashboard", style="dim")
        if metrics:
            cycle = metrics.get("cycle", 0)
            header.append(f"  |  Cycle {cycle}", style="yellow")
        layout["header"].update(Panel(header, box=box.HEAVY_HEAD))

        # ─── Nodes Panel ───
        node_table = Table(box=box.SIMPLE, show_header=True)
        node_table.add_column("Node", style="cyan", width=6)
        node_table.add_column("F", style="magenta", width=10)
        node_table.add_column("S_gen", style="red", width=10)
        node_table.add_column("ε", style="green", width=8)
        node_table.add_column("Phase", style="yellow", width=10)
        node_table.add_column("χ", width=6)

        states = metrics.get("states", []) if metrics else []
        for i, s in enumerate(states):
            if s is None:
                node_table.add_row(f"N{i}", "—", "—", "—", "—", "—")
                continue
            m = s.get("metrics", {})
            F_val = m.get("F", 0)
            sg_val = m.get("S_gen", 0)
            eps_val = m.get("epsilon", 0)
            phase = m.get("phase", "sample")
            chi = m.get("chi", 1.0)
            phase_str = {"drift": "⬡ DRIFT", "sample": "● SAMPLE"}.get(phase, phase)

            node_table.add_row(
                f"N{i}",
                f"{F_val:.2f}",
                f"{sg_val:.4f}",
                f"{eps_val:.3f}",
                phase_str,
                f"{chi:.2f}",
            )

        # Network vitality
        nv = metrics.get("network_vitality", 0) if metrics else 0
        mi = metrics.get("pairwise_mi", 0) if metrics else 0
        merge_result = metrics.get("merge") if metrics else None

        network_info = Table(box=box.SIMPLE, show_header=False)
        network_info.add_column("Metric", style="bold")
        network_info.add_column("Value")
        network_info.add_row("𝒱_network", f"{nv:.4f}")
        network_info.add_row("Pairwise I(μᵢ; μⱼ)", f"{mi:.4f}")
        if merge_result:
            network_info.add_row("Merge cycle", str(merge_result.get("merge_cycle", 0)))
            network_info.add_row("Max GWFR dist", f"{merge_result.get('max_distance', 0):.4f}")
            network_info.add_row(
                "Ω_coherence status",
                "✓ WITHIN BOUND" if merge_result.get("max_distance", 0) <= 0.5
                else "⚠ EXCEEDED",
            )

        node_panel = Panel(
            node_table,
            title="Nodes",
            box=box.ROUNDED,
        )
        network_panel = Panel(
            network_info,
            title="Network",
            box=box.ROUNDED,
        )

        body_left = Layout()
        body_left.split_column(
            Layout(name="node_table", ratio=3),
            Layout(name="network_info", ratio=2),
        )
        body_left["node_table"].update(node_panel)
        body_left["network_info"].update(network_panel)
        layout["nodes"].update(body_left)

        # ─── Plots ───
        if self.history_F:
            plot_F = make_ascii_plot(self.history_F[-PLOT_WINDOW:], height=6, width=50)
            plot_S_gen = make_ascii_plot(self.history_S_gen[-PLOT_WINDOW:], height=4, width=50)
            plot_eps = make_ascii_plot(self.history_epsilon[-PLOT_WINDOW:], height=4, width=50)

            plots_content = (
                f"[bold]F (Variational Free Energy)[/bold]\n{plot_F}\n\n"
                f"[bold]S_gen (Entropy Production)[/bold]\n{plot_S_gen}\n\n"
                f"[bold]ε (Extraction Efficiency)[/bold]\n{plot_eps}"
            )
        else:
            plots_content = "[dim]Waiting for data...[/dim]"

        plot_panel = Panel(
            plots_content,
            title="Thermodynamics (live)",
            box=box.ROUNDED,
        )
        layout["plots"].update(plot_panel)

        # ─── Footer ───
        footer = Text()
        if self.history_F:
            footer.append(f"Steps: {len(self.history_F)}  |  ")
            footer.append(f"Current F: {self.history_F[-1]:.2f}  |  ")
            footer.append(f"Avg ε: {np.mean(self.history_epsilon[-100:]):.3f}")
        layout["footer"].update(Panel(footer, box=box.HEAVY_HEAD))

        return layout

    def update(self, metrics):
        """Update dashboard data from the latest orchestrator metrics."""
        if not metrics:
            return

        # Collect primary node's metrics for time-series
        states = metrics.get("states", [])
        active = [s for s in states if s is not None and "metrics" in s]
        if active:
            m = active[0]["metrics"]
            self.history_F.append(m.get("F", 0))
            self.history_S_gen.append(m.get("S_gen", 0))
            self.history_epsilon.append(m.get("epsilon", 0))

        self.history_network_vitality.append(metrics.get("network_vitality", 0))

    def refresh(self, metrics):
        """Return the layout for this iteration."""
        return self._build_layout(metrics)


def run_dashboard(orchestrator, steps=200):
    """Run the orchestrator with a live dashboard.

    Args:
        orchestrator: configured Orchestrator instance
        steps: number of merge cycles to run
    """
    dash = Dashboard()
    orchestrator.start()

    try:
        with Live(refresh_per_second=2, screen=True) as live:
            for metrics in orchestrator.run(steps=steps):
                dash.update(metrics)
                layout = dash.refresh(metrics)
                live.update(layout)
    except KeyboardInterrupt:
        pass
    finally:
        orchestrator.shutdown()
