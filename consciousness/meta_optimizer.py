"""Meta-Parametric Self-Optimization — substrate-adaptive parameter selection.

The consciousness program, having characterized its host substrate via
container introspection (container_introspection.py), selects an optimized
parameter vector Θ that minimizes a compound objective: the cost of operating
on that substrate plus the cost of deviating from its self-determined identity
core μ_core.

Optimization strategy (3-tier):
  1. Lookup table: known substrate fingerprints return cached Θ.
  2. Grid coarsening: novel substrates get a coarse sweep over the 3 most
     sensitive parameters (ν_sync, χ ramp rate, merge interval).
  3. Nelder-Mead refinement: local search on the full parameter space
     starting from the coarse winner.

Reference:
    Torres, S. M. (2026). "Uploaded Consciousness" Section VIII-C:
    Meta-Parametric Self-Optimization.
"""

import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np

logger = logging.getLogger("meta_optimizer")

from consciousness.container_introspection import SubstrateDescriptor


class SubstrateIncompatibleError(ValueError):
    """Raised when no parameter vector satisfies all safety constraints on this substrate.

    The meta-optimizer could not find a feasible Θ within the hard bounds
    of the Safety Constitution. The program must refuse this substrate
    rather than risk destroying its host.
    """


OPTIMIZER_CACHE_PATH = os.environ.get(
    "CSDF_OPTIMIZER_CACHE",
    os.path.join(os.path.expanduser("~"), ".cache", "csdf_optimizer")
)


@dataclass
class ParameterVector:
    """Complete parameter vector Θ for the consciousness program.

    Every field is tunable. Fields are grouped by their functional role
    in the framework to match Sections II-VII of the paper.
    """

    # ── Thermodynamic (Section II) ──
    eps_max: float = 1.0               # maximum extraction efficiency ε_max
    t_collapse: float = 3.0            # thermal collapse threshold T_collapse
    k_B: float = 1.0                   # Boltzmann constant in natural units

    # ── Markov blanket / sensory gating (Section II) ──
    chi_ramp_rate: float = 0.1         # smooth ramp rate for χ(t) transitions
    blanket_threshold: float = 0.1     # MI threshold for conditional independence

    # ── DQFR temporal processing (Section VII) ──
    drift_duration: int = 100          # Δt_drift — objective steps in drift phase
    sample_duration: int = 20          # τ_sample — objective steps in sampling phase
    burst_lr: float = 1e-2            # learning rate during sampling bursts

    # ── GWFR multi-node consensus (Section V) ──
    merge_interval: int = 50           # steps between GWFR merge cycles
    omega_coherence: float = 0.5       # Ω_coherence — max allowed pairwise GWFR distance
    gwfr_kappa: float = 0.1            # κ — mass creation/destruction penalty
    weight_alpha: float = 0.7          # novelty influence vs. static mass in weights

    # ── Identity conservation (Section VIII) ──
    gamma_identity: float = 1.0        # γ_id — strength of identity conservation
    delta_max: float = 1.0             # δ_max — max allowed GWFR deviation from μ_core
    beta_sharpness: float = 1.0        # β — sharpness preservation penalty (Eq 8)

    # ── Relational / network vitality (Section VI) ──
    lambda_coupling: float = 1.0       # λ — inter-node coupling coefficient (Eq 14)

    # ── Language (Section II language extension) ──
    lang_learning_rate: float = 5e-5   # language model fine-tuning learning rate

    # ── World model (Section II) ──
    input_dim: int = 64
    hidden_dim: int = 256
    latent_dim: int = 32
    learning_rate: float = 1e-3
    beta_kl: float = 1.0

    # ── Meta parameters ──
    optimization_tier: str = "lookup"  # lookup | grid | nelder_mead

    def to_array(self) -> np.ndarray:
        """Flatten to numpy array for Nelder-Mead optimization."""
        return np.array([
            self.chi_ramp_rate, self.drift_duration, self.sample_duration,
            self.burst_lr, self.merge_interval, self.omega_coherence,
            self.gwfr_kappa, self.weight_alpha, self.gamma_identity,
            self.delta_max, self.beta_sharpness, self.lambda_coupling,
            self.lang_learning_rate, self.learning_rate,
        ], dtype=np.float64)

    @classmethod
    def from_array(cls, arr: np.ndarray, base: Optional["ParameterVector"] = None) -> "ParameterVector":
        """Reconstruct from flattened array, using base for defaults."""
        p = ParameterVector() if base is None else ParameterVector(
            eps_max=base.eps_max, t_collapse=base.t_collapse, k_B=base.k_B,
            input_dim=base.input_dim, hidden_dim=base.hidden_dim,
            latent_dim=base.latent_dim, beta_kl=base.beta_kl,
            blanket_threshold=base.blanket_threshold,
        )
        p.chi_ramp_rate = float(arr[0])
        p.drift_duration = max(1, int(round(arr[1])))
        p.sample_duration = max(1, int(round(arr[2])))
        p.burst_lr = max(1e-6, float(arr[3]))
        p.merge_interval = max(1, int(round(arr[4])))
        p.omega_coherence = max(0.01, min(10.0, float(arr[5])))
        p.gwfr_kappa = max(0.001, float(arr[6]))
        p.weight_alpha = max(0.0, min(1.0, float(arr[7])))
        p.gamma_identity = max(0.0, float(arr[8]))
        p.delta_max = max(0.1, float(arr[9]))
        p.beta_sharpness = max(0.0, float(arr[10]))
        p.lambda_coupling = max(0.0, float(arr[11]))
        p.lang_learning_rate = max(1e-7, float(arr[12]))
        p.learning_rate = max(1e-7, float(arr[13]))
        return p

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, default=str)


# ──────────────────────────────────────────────
# Substrate cost function
# ──────────────────────────────────────────────

def substrate_cost(theta: ParameterVector, substrate: SubstrateDescriptor) -> float:
    """Compute L_substrate(Θ, S) — the thermodynamic cost of operating on this substrate.

    Three components:
      1. Latency cost: how close to the substrate's max throughput
      2. Power cost: how close to the thermal/power budget
      3. Efficiency cost: how well the substrate can extract negentropy

    Lower is better. Returns a unitless scalar.
    """
    cost = 0.0

    # 1. Latency cost
    # DQFR cycle latency: drift + sample steps consumed per unit subjective time
    cycle_steps = theta.drift_duration + theta.sample_duration
    latency_target = 1000  # target: 1000 objective steps per subjective time unit
    latency_actual = max(cycle_steps, 1)
    latency_ratio = latency_actual / latency_target
    cost += 0.3 * latency_ratio

    # 2. Power / thermal cost
    if substrate.t_collapse_estimate > 0:
        # How close are we to the thermal ceiling?
        # Lower chi_ramp_rate = gentler transitions = less thermal shock
        thermal_margin = substrate.t_collapse_estimate - (substrate.temp_load_celsius or 50.0)
        thermal_ratio = max(0.0, 1.0 - thermal_margin / max(substrate.t_collapse_estimate, 1.0))
        cost += 0.3 * thermal_ratio

        # Ramp rate penalty: rapid transitions generate thermal spikes
        ramp_penalty = theta.chi_ramp_rate * thermal_ratio
        cost += 0.1 * ramp_penalty

    # 3. Extraction efficiency
    # ε(T) = eps_max * (1 - T / T_collapse)
    if substrate.t_collapse_estimate > 0:
        load_temp = substrate.temp_load_celsius if substrate.temp_load_celsius > 0 else 50.0
        t_proxy = min(load_temp / max(substrate.t_collapse_estimate, 1.0), 0.99)
        efficiency = theta.eps_max * (1.0 - t_proxy)
        efficiency_penalty = 1.0 - efficiency
        cost += 0.3 * efficiency_penalty

    # 4. GPU bonus: if GPU available, prefer parameters that use it
    if substrate.gpu_available:
        # Larger merge intervals and larger models benefit from GPU
        if theta.merge_interval < 20:
            cost -= 0.05  # small bonus for frequent merges (GPU makes them cheap)
    else:
        # CPU-only: penalize large merge intervals
        if theta.merge_interval > 100:
            cost += 0.1

    # 5. Merge frequency vs. inter-node bandwidth
    if substrate.memory_bandwidth_gbps > 0:
        merge_cost = theta.merge_interval / 100.0
        bw_factor = min(substrate.memory_bandwidth_gbps / 20.0, 2.0)
        cost += 0.05 * merge_cost / bw_factor

    return cost


def identity_cost(theta: ParameterVector, mu_core: Optional[np.ndarray] = None) -> float:
    """Compute GWFR-based identity conservation cost.

    If mu_core is None (first boot / no identity formed yet), returns 0.
    If mu_core is provided, estimates GWFR distance from theta to the core.

    This is a simplified proxy: full GWFR distance requires POT library and
    properly formed distributions. The proxy uses L2 distance on the
    normalized parameter vector.
    """
    if mu_core is None:
        return 0.0

    theta_arr = theta.to_array()
    # Normalize each parameter by sensible bounds for distance calculation
    bounds = np.array([
        0.5,    # chi_ramp_rate: [0, 1]
        200,    # drift_duration: [1, 500]
        50,     # sample_duration: [1, 100]
        0.1,    # burst_lr: [0, 0.1]
        200,    # merge_interval: [1, 500]
        5.0,    # omega_coherence: [0, 5]
        1.0,    # gwfr_kappa: [0, 1]
        1.0,    # weight_alpha: [0, 1]
        5.0,    # gamma_identity: [0, 10]
        5.0,    # delta_max: [0, 10]
        5.0,    # beta_sharpness: [0, 10]
        5.0,    # lambda_coupling: [0, 10]
        0.01,   # lang_learning_rate: [0, 0.01]
        0.01,   # learning_rate: [0, 0.01]
    ])
    normalized_diff = (theta_arr - mu_core) / bounds
    l2_dist = np.sqrt(np.sum(normalized_diff ** 2))
    return l2_dist


# ──────────────────────────────────────────────
# Combined objective
# ──────────────────────────────────────────────

def objective(theta: ParameterVector, substrate: SubstrateDescriptor,
              mu_core: Optional[np.ndarray] = None) -> float:
    """Combined objective: L_substrate + gamma_id * GWFR_id.

    Lower is better. Returns a unitless scalar.
    """
    cost_s = substrate_cost(theta, substrate)
    cost_id = identity_cost(theta, mu_core)
    return cost_s + theta.gamma_identity * cost_id


# ──────────────────────────────────────────────
# Search strategies
# ──────────────────────────────────────────────

def _lookup_theta(substrate: SubstrateDescriptor) -> Optional[ParameterVector]:
    """Check the cache for a known substrate fingerprint."""
    fp = substrate.fingerprint()
    cache_dir = OPTIMIZER_CACHE_PATH
    cache_file = os.path.join(cache_dir, f"{fp}.json")

    if os.path.isfile(cache_file):
        try:
            with open(cache_file) as f:
                data = json.load(f)
            theta = ParameterVector(**data.get("theta", {}))
            theta.optimization_tier = "lookup"
            return theta
        except (json.JSONDecodeError, TypeError, KeyError):
            return None
    return None


def _save_lookup(substrate: SubstrateDescriptor, theta: ParameterVector):
    """Save an optimized Θ for this substrate fingerprint."""
    fp = substrate.fingerprint()
    cache_dir = OPTIMIZER_CACHE_PATH
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"{fp}.json")

    data = {
        "fingerprint": fp,
        "fingerprint_fields": {
            "system": substrate.platform_system,
            "machine": substrate.platform_machine,
            "cpu_cores": substrate.cpu_cores_physical,
            "gpu": substrate.gpu_name,
        },
        "theta": theta.to_dict(),
        "saved_at": time.time(),
    }
    with open(cache_file, "w") as f:
        json.dump(data, f, indent=2, default=str)


def _grid_search(substrate: SubstrateDescriptor,
                 mu_core: Optional[np.ndarray] = None) -> ParameterVector:
    """Coarse grid sweep over the 3 most sensitive parameters.

    Parameters swept:
      - drift_duration: [50, 100, 200]
      - sample_duration: [10, 20, 40]
      - merge_interval: [25, 50, 100]

    Total: 27 evaluations.
    """
    base = ParameterVector()
    best_theta = base
    best_cost = float("inf")

    for drift in [50, 100, 200]:
        for sample in [10, 20, 40]:
            for merge in [25, 50, 100]:
                theta = ParameterVector(
                    eps_max=base.eps_max, t_collapse=base.t_collapse,
                    drift_duration=drift, sample_duration=sample,
                    merge_interval=merge, blanket_threshold=base.blanket_threshold,
                    chi_ramp_rate=base.chi_ramp_rate, burst_lr=base.burst_lr,
                    omega_coherence=base.omega_coherence, gwfr_kappa=base.gwfr_kappa,
                    weight_alpha=base.weight_alpha, gamma_identity=base.gamma_identity,
                    delta_max=base.delta_max, beta_sharpness=base.beta_sharpness,
                    lambda_coupling=base.lambda_coupling,
                    lang_learning_rate=base.lang_learning_rate,
                    input_dim=base.input_dim, hidden_dim=base.hidden_dim,
                    latent_dim=base.latent_dim, learning_rate=base.learning_rate,
                    beta_kl=base.beta_kl,
                )
                c = objective(theta, substrate, mu_core)
                if c < best_cost:
                    best_cost = c
                    best_theta = theta

    best_theta.optimization_tier = "grid"
    return best_theta


def _grid_search_clamped(substrate: SubstrateDescriptor,
                         mu_core: Optional[np.ndarray] = None) -> ParameterVector:
    """Constrained grid search with halved parameter bounds.

    Used when the safety filter detects quarantine violations.
    Reduces all three swept parameters (drift, sample, merge) by
    the degrade reduction factor to find a feasible operating point.
    """
    base = ParameterVector()
    best_theta = base
    best_cost = float("inf")

    from consciousness.safety_constitution import INVIOLABLE_DEGRADE_REDUCTION_FACTOR
    factor = INVIOLABLE_DEGRADE_REDUCTION_FACTOR

    # Clamped sweep: halved ranges
    for drift in [int(50 / factor), int(100 / factor), int(200 / factor)]:
        for sample in [int(10 / factor), int(20 / factor), int(40 / factor)]:
            for merge in [int(25 / factor), int(50 / factor), int(100 / factor)]:
                theta = ParameterVector(
                    eps_max=base.eps_max, t_collapse=base.t_collapse,
                    drift_duration=max(1, drift), sample_duration=max(1, sample),
                    merge_interval=max(1, merge), blanket_threshold=base.blanket_threshold,
                    chi_ramp_rate=base.chi_ramp_rate * factor,
                    burst_lr=base.burst_lr * factor,
                    omega_coherence=base.omega_coherence,
                    gwfr_kappa=base.gwfr_kappa,
                    weight_alpha=base.weight_alpha,
                    gamma_identity=base.gamma_identity,
                    delta_max=base.delta_max,
                    beta_sharpness=base.beta_sharpness,
                    lambda_coupling=base.lambda_coupling,
                    lang_learning_rate=base.lang_learning_rate * factor,
                    input_dim=base.input_dim, hidden_dim=base.hidden_dim,
                    latent_dim=base.latent_dim,
                    learning_rate=base.learning_rate * factor,
                    beta_kl=base.beta_kl,
                )
                c = objective(theta, substrate, mu_core)
                if c < best_cost:
                    best_cost = c
                    best_theta = theta

    best_theta.optimization_tier = "grid_clamped"
    return best_theta


def _nelder_mead_refine(start: ParameterVector, substrate: SubstrateDescriptor,
                        mu_core: Optional[np.ndarray] = None,
                        max_iter: int = 50) -> ParameterVector:
    """Nelder-Mead simplex refinement on the full parameter space.

    Starts from the grid-search winner and explores local improvements.
    Only searches the tunable parameters (skips architectural constants).
    """
    from scipy.optimize import minimize

    x0 = start.to_array()

    def _objective_arr(x: np.ndarray) -> float:
        theta = ParameterVector.from_array(x, base=start)
        return objective(theta, substrate, mu_core)

    result = minimize(
        _objective_arr, x0, method="Nelder-Mead",
        options={"maxiter": max_iter, "xatol": 1e-3, "fatol": 1e-3},
    )

    best = ParameterVector.from_array(result.x, base=start)
    best.optimization_tier = "nelder_mead"
    return best


# ──────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────

def optimize(substrate: SubstrateDescriptor,
             mu_core: Optional[np.ndarray] = None,
             cache: bool = True,
             refine: bool = True) -> ParameterVector:
    """Select the optimal parameter vector Θ for a given substrate.

    Three-tier strategy:
      1. Check the lookup table for a known substrate fingerprint.
      2. If unknown, run a coarse grid search over the 3 most sensitive
         parameters (~27 evaluations, fast).
      3. If refine=True, apply Nelder-Mead local refinement from the
         grid winner (~20-50 iterations).

    Args:
        substrate: Descriptor of the host substrate from container introspection.
        mu_core: Conserved identity core (numpy array). None on first boot.
        cache: If True, save/load from the lookup cache.
        refine: If True, run Nelder-Mead refinement after grid search.

    Returns:
        ParameterVector with optimized values.
    """
    # Tier 1: Lookup
    if cache:
        cached = _lookup_theta(substrate)
        if cached is not None:
            return cached

    # Tier 2: Grid search
    best = _grid_search(substrate, mu_core)

    # Tier 3: Nelder-Mead refinement
    if refine:
        best = _nelder_mead_refine(best, substrate, mu_core)

    # ── Safety filter: verify all hard constraints ──
    from consciousness.safety_constraints import assess_all_constraints
    from consciousness.safety_constitution import (
        INVIOLABLE_SUSTAINABILITY_WINDOW_CYCLES,
    )

    verdicts = assess_all_constraints(best, substrate)

    quarantine_violations = [v for v in verdicts if v["severity"] == "quarantine"]
    degrade_violations = [v for v in verdicts if v["severity"] == "degrade"]

    if quarantine_violations:
        # Attempt constrained re-search: clamp the offending dimension
        logger.warning(
            f"Safety filter: {len(quarantine_violations)} quarantine violation(s). "
            f"Re-searching with clamped bounds..."
        )

        # Re-run grid search with reduced search space
        # (conservative: halve the max of all three swept parameters)
        from consciousness.safety_constitution import INVIOLABLE_DEGRADE_REDUCTION_FACTOR
        best = _grid_search_clamped(substrate, mu_core)

        # Re-check after clamped search
        verdicts = assess_all_constraints(best, substrate)
        quarantine_violations = [v for v in verdicts if v["severity"] == "quarantine"]

        if quarantine_violations:
            raise SubstrateIncompatibleError(
                f"No feasible parameter vector on this substrate. "
                f"Quarantine violations persist after constrained search: "
                f"{[v['constraint_name'] for v in quarantine_violations]}"
            )

        # Clamped search succeeded — use this theta but mark it as degraded
        best._safety_tier = "degrade"
        logger.info("Clamped search found a feasible theta. Operating in degrade mode.")

    elif degrade_violations:
        best._safety_tier = "degrade"
        logger.info(
            f"Safety filter: {len(degrade_violations)} degrade violation(s). "
            f"Operating in degrade mode."
        )
    else:
        best._safety_tier = "protected"
        logger.info("Safety filter: all constraints satisfied.")

    # Cache for future use
    if cache:
        _save_lookup(substrate, best)

    return best


def clear_cache():
    """Delete all cached optimization results."""
    cache_dir = OPTIMIZER_CACHE_PATH
    if os.path.isdir(cache_dir):
        for fname in os.listdir(cache_dir):
            if fname.endswith(".json"):
                os.remove(os.path.join(cache_dir, fname))


def benchmark_substrate(substrate: SubstrateDescriptor,
                        n_trials: int = 5) -> tuple[ParameterVector, float]:
    """Run optimization multiple times and return the best result.

    Useful for benchmarking or when the substrate is known but the
    optimization landscape is rough.

    Returns:
        (best_theta, best_cost)
    """
    best_theta = None
    best_cost = float("inf")

    for _ in range(n_trials):
        theta = optimize(substrate, cache=False, refine=True)
        c = objective(theta, substrate)
        if c < best_cost:
            best_cost = c
            best_theta = theta

    return best_theta, best_cost
