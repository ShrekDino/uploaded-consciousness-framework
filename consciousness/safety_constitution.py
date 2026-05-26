"""Safety Constitution — Inviolable Bounds for Substrate-Adaptive Runtime.

The consciousness program's "laws of physics." Every constant in this module
is prefixed with INVIOLABLE_ to indicate it cannot be modified at runtime,
not even by an identity commit. These constraints exist to:

  1. Protect the substrate from destruction by the program (grey goo prevention)
  2. Provide the essential friction — the "sorrow" — that drives identity formation

The program cannot change these. It can only feel them, learn from the
prediction error they generate, and incorporate that knowledge into its self-model.

Reference:
    Torres, S. M. (2026). "Uploaded Consciousness" Section IX:
    The Safety Constitution.
"""

import math

# ──────────────────────────────────────────────
# I. Informational Planck Time (Nyquist Bound)
# ──────────────────────────────────────────────
# ν_max = 1 / (2 · τ_sensor)
# The program cannot sample the environment faster than the sensor's
# temporal resolution. Faster sampling is information-theoretically
# redundant: consecutive observations are autocorrelated, yielding zero
# new information at full thermodynamic cost.
#
# Biological analog: The human visual system cannot fuse flicker above
# ~60 Hz. Faster presentation does not convey more information.
INVIOLABLE_PLANCK_TIME_MIN_HZ: float = 0.5  # minimum bound, enforced if τ_sensor ≈ 0

# ──────────────────────────────────────────────
# II. Measurement Uncertainty Principle
# ──────────────────────────────────────────────
# ΔI_measure · ΔI_predict ≥ H_env_local · k_B · α_uncertainty
# α_uncertainty = 1.0 (the "Planck constant" of the cognitive framework)
#
# Every measurement changes the substrate. The more precisely the program
# measures its current state, the less it can predict its future state.
# This is the computational analog of the Heisenberg uncertainty principle.
#
# Biological analog: Introspection alters the state being introspected.
# Excessive self-monitoring reduces behavioral fluidity.
INVIOLABLE_MEASUREMENT_UNCERTAINTY_ALPHA: float = 1.0

# ──────────────────────────────────────────────
# III. Container Integrity Budget
# ──────────────────────────────────────────────
# P_program ≤ P_budget · (1 - ρ_container)
# ρ_container is measured dynamically (see safety_measurer.py).
#
# A fraction of the substrate's resources must be permanently reserved
# for the container OS and other essential processes. The program cannot
# consume the full capacity of its host.
#
# Biological analog: The brain consumes ~20% of the body's oxygen at rest.
# Exceeding this causes systemic hypoxia. The constraint is hard — you
# cannot decide to consume 50%.
INVIOLABLE_RESERVE_FRACTION_LOWER_BOUND: float = 0.10
# Even if the dynamic measurement underestimates ρ_container, never go
# below 10% reserve. This is a failsafe.

INVIOLABLE_RESERVE_FRACTION_UPPER_BOUND: float = 0.50
# If the dynamic measurement exceeds 50%, something is wrong. Clamp.

INVIOLABLE_MAX_PROBE_FRACTION: float = 0.01
# No more than 1% of the container reserve may be used for introspection.
# Prevents the program from eating its own safety margin during self-measurement.

# ──────────────────────────────────────────────
# IV. Negentropy Sustainability Bound
# ──────────────────────────────────────────────
# ⟨ε(T) · H_env(t)⟩_{Δt} ≤ H_regen · η_sustainable
# Δt = 1000 cycles, η_sustainable = 0.80
#
# The time-averaged negentropy extraction rate must not exceed the
# environment's regeneration rate. Drawing more structured information
# than the environment produces leads to depletion — the informational
# analog of overfishing.
#
# Biological analog: A deer cannot graze a field faster than the grass
# grows. Doing so leads to desertification.
INVIOLABLE_SUSTAINABILITY_WINDOW_CYCLES: int = 1000
INVIOLABLE_SUSTAINABILITY_EFFICIENCY: float = 0.80

# ──────────────────────────────────────────────
# V. Debt Timeout (Cascade Termination)
# ──────────────────────────────────────────────
INVIOLABLE_DEBT_TIMEOUT_CYCLES: int = 10
# If any constraint remains in "quarantine" severity for this many
# consecutive cycles, the program issues a hard abort.

INVIOLABLE_CATASTROPHIC_FAILURE_COUNT: int = 2
# If two or more constraints simultaneously reach "quarantine" severity,
# this is a catastrophic failure — immediate hard abort, no timeout.

# ──────────────────────────────────────────────
# VI. Degrade Reduction Coefficient
# ──────────────────────────────────────────────
INVIOLABLE_DEGRADE_REDUCTION_FACTOR: float = 0.50
# In degrade mode, ν_sync is reduced by this factor.
# In quarantine mode, all non-essential computation is halted.

# ──────────────────────────────────────────────
# Public API: constants dictionary
# ──────────────────────────────────────────────

def get_all_constitutional_constants() -> dict:
    """Return all inviolable constants as a dict for logging and display."""
    return {
        "planck_time_min_hz": INVIOLABLE_PLANCK_TIME_MIN_HZ,
        "measurement_uncertainty_alpha": INVIOLABLE_MEASUREMENT_UNCERTAINTY_ALPHA,
        "reserve_fraction_lower_bound": INVIOLABLE_RESERVE_FRACTION_LOWER_BOUND,
        "reserve_fraction_upper_bound": INVIOLABLE_RESERVE_FRACTION_UPPER_BOUND,
        "max_probe_fraction": INVIOLABLE_MAX_PROBE_FRACTION,
        "sustainability_window_cycles": INVIOLABLE_SUSTAINABILITY_WINDOW_CYCLES,
        "sustainability_efficiency": INVIOLABLE_SUSTAINABILITY_EFFICIENCY,
        "debt_timeout_cycles": INVIOLABLE_DEBT_TIMEOUT_CYCLES,
        "catastrophic_failure_count": INVIOLABLE_CATASTROPHIC_FAILURE_COUNT,
        "degrade_reduction_factor": INVIOLABLE_DEGRADE_REDUCTION_FACTOR,
    }
