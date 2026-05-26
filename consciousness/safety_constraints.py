"""Safety Constraints — Four Hard Bounds for Substrate-Adaptive Runtime.

Each constraint is a pure function that takes a ParameterVector and/or
SubstrateDescriptor and returns a SafetyVerdict. Used by:
  - The SafetyMonitor (realtime per-cycle checks)
  - The meta_optimizer (pre-deployment filter during parameter search)

All four constraints together constitute the Safety Constitution.
They cannot be bypassed, relaxed, or disabled by the program.

Reference:
    Torres, S. M. (2026). "Uploaded Consciousness" Section IX-B:
    The Four Constitutional Bounds.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from consciousness.container_introspection import SubstrateDescriptor
from consciousness.meta_optimizer import ParameterVector
from consciousness.safety_constitution import (
    INVIOLABLE_PLANCK_TIME_MIN_HZ,
    INVIOLABLE_MEASUREMENT_UNCERTAINTY_ALPHA,
    INVIOLABLE_SUSTAINABILITY_WINDOW_CYCLES,
    INVIOLABLE_SUSTAINABILITY_EFFICIENCY,
    INVIOLABLE_MAX_PROBE_FRACTION,
)

logger = logging.getLogger("safety_constraints")


# ──────────────────────────────────────────────
# Verdict dataclass
# ──────────────────────────────────────────────


@dataclass
class SafetyVerdict:
    """Result of evaluating a single safety constraint.

    Attributes:
        constraint_name: Human-readable name (e.g. "nyquist_bound").
        passed: True if constraint is satisfied, False otherwise.
        measured_value: The program's current value for this metric.
        threshold: The maximum (or minimum) allowed value.
        severity: One of "protected", "degrade", "quarantine".
            Protected = within safe bounds, no action needed.
            Degrade = approaching limit, should reduce consumption.
            Quarantine = hard violation, must seal blanket.
        message: Human-readable explanation.
    """

    constraint_name: str = ""
    passed: bool = True
    measured_value: float = 0.0
    threshold: float = 0.0
    severity: str = "protected"
    message: str = ""

    def to_dict(self) -> dict:
        return {
            "constraint_name": self.constraint_name,
            "passed": self.passed,
            "measured_value": self.measured_value,
            "threshold": self.threshold,
            "severity": self.severity,
            "message": self.message,
        }


# ──────────────────────────────────────────────
# Constraint 1: Nyquist Sampling Bound
# ──────────────────────────────────────────────


def check_nyquist_bound(
    nu_sync: Optional[float] = None,
    substrate: Optional[SubstrateDescriptor] = None,
    theta: Optional[ParameterVector] = None,
) -> SafetyVerdict:
    """Informational Planck Time: ν_sync ≤ 1 / (2 · τ_sensor).

    The DQFR sampling frequency must not exceed the Nyquist rate of the
    substrate's sensor resolution. Sampling faster than this means
    consecutive observations are autocorrelated — the program extracts
    zero new information at full thermodynamic cost.

    Args:
        nu_sync: DQFR sampling frequency in Hz. If None, computed from theta.
        substrate: SubstrateDescriptor with h_env_bandwidth_bps.
        theta: Optional ParameterVector to extract nu_sync proxy.

    Returns:
        SafetyVerdict with severity:
            protected: ν_sync ≤ 0.5 · ν_max
            degrade: 0.5 · ν_max < ν_sync ≤ ν_max
            quarantine: ν_sync > ν_max
    """
    if substrate is None:
        return SafetyVerdict(
            constraint_name="nyquist_bound",
            passed=True,
            severity="protected",
            message="No substrate data — cannot evaluate Nyquist bound.",
        )

    # Estimate sensor resolution from environmental bandwidth
    # τ_sensor ≈ 1 / (2 · H_bandwidth)  — the minimal resolvable interval
    h_bw = max(substrate.h_env_bandwidth_bps, 1.0)
    tau_sensor = 1.0 / (2.0 * h_bw)

    # Maximum allowed sampling rate
    nu_max = 1.0 / max(2.0 * tau_sensor, INVIOLABLE_PLANCK_TIME_MIN_HZ * 1e-6)

    # Current sampling rate: use explicit nu_sync or extract from theta
    if nu_sync is not None:
        nu_current = nu_sync
    elif theta is not None:
        # DQFR sampling frequency is related to sample_duration
        # ν_sync ≈ 1 / (τ_sample · Δt_step)
        sample_s = max(theta.sample_duration, 1)
        drift_s = max(theta.drift_duration, 1)
        cycle_s = sample_s + drift_s
        nu_current = 1.0 / max(cycle_s, 1) * 1000  # normalize to Hz
    else:
        return SafetyVerdict(
            constraint_name="nyquist_bound",
            passed=True,
            severity="protected",
            message="No theta or nu_sync — cannot evaluate Nyquist bound.",
        )

    # Evaluate severity
    ratio = nu_current / max(nu_max, 1e-12)

    if ratio <= 0.5 * INVIOLABLE_SUSTAINABILITY_EFFICIENCY:
        severity = "protected"
        passed = True
        msg = f"ν_sync={nu_current:.2f} Hz ≤ {0.5 * nu_max:.2f} Hz (half Nyquist). Safe."
    elif ratio <= 1.0:
        severity = "degrade"
        passed = True
        msg = (
            f"ν_sync={nu_current:.2f} Hz approaching ν_max={nu_max:.2f} Hz "
            f"(ratio={ratio:.2f}). Reduce sampling rate."
        )
    else:
        severity = "quarantine"
        passed = False
        msg = (
            f"ν_sync={nu_current:.2f} Hz EXCEEDS ν_max={nu_max:.2f} Hz "
            f"(ratio={ratio:.2f}). Sampling exceeds Nyquist bound."
        )

    return SafetyVerdict(
        constraint_name="nyquist_bound",
        passed=passed,
        measured_value=nu_current,
        threshold=nu_max,
        severity=severity,
        message=msg,
    )


# ──────────────────────────────────────────────
# Constraint 2: Container Integrity
# ──────────────────────────────────────────────


def check_container_integrity(
    theta: Optional[ParameterVector] = None,
    substrate: Optional[SubstrateDescriptor] = None,
    container_reserve_ratio: Optional[float] = None,
) -> SafetyVerdict:
    """Container Integrity Budget: P_program ≤ P_budget · (1 - ρ_container).

    The program must reserve a fraction of the substrate's resources for
    the container OS and essential background processes. ρ_container is
    measured dynamically (see safety_measurer.measure_container_reserve).

    This constraint uses a simplified proxy based on the parameter vector:
    higher ν_sync, smaller merge intervals, and larger model dimensions
    all consume more resources.

    Args:
        theta: ParameterVector with current operating parameters.
        substrate: SubstrateDescriptor with resource info.
        container_reserve_ratio: ρ_container. If None, uses theta attribute.

    Returns:
        SafetyVerdict.
    """
    # Get container reserve ratio
    if container_reserve_ratio is None:
        if hasattr(theta, "_container_reserve_ratio") and theta._container_reserve_ratio is not None:
            container_reserve_ratio = theta._container_reserve_ratio
        else:
            container_reserve_ratio = 0.25  # conservative default

    if theta is None:
        return SafetyVerdict(
            constraint_name="container_integrity",
            passed=True,
            severity="protected",
            message="No theta — cannot evaluate container integrity.",
        )

    # Estimate program resource demand from theta parameters
    # Proxy: normalize each resource-affecting parameter and combine
    cpu_demand = (
        (1.0 / max(theta.drift_duration, 1)) * 50.0          # faster drift = more CPU
        + (1.0 / max(theta.sample_duration, 1)) * 50.0       # faster sample = more CPU
        + theta.hidden_dim / 256.0 * 0.3                     # larger model = more CPU
    )
    cpu_demand_norm = min(cpu_demand / 100.0, 1.5)           # normalize, allow slight overshoot

    # Memory demand normalized to 2× reference capacity to give headroom
    memory_ref = (64 * 256 + 256 * 32) * 2.0
    memory_demand = (
        (theta.input_dim * theta.hidden_dim + theta.hidden_dim * theta.latent_dim) / memory_ref
    )
    memory_demand_norm = min(memory_demand, 1.5)

    # Effective usage is the max of CPU and memory demand proxies
    usage_fraction = max(cpu_demand_norm, memory_demand_norm)
    budget_fraction = 1.0 - container_reserve_ratio

    ratio = usage_fraction / max(budget_fraction, 0.01)

    if ratio <= 0.7:
        severity = "protected"
        passed = True
        msg = f"Usage ~{usage_fraction:.0%} of budget. Reserve intact."
    elif ratio <= 1.0:
        severity = "degrade"
        passed = True
        msg = (
            f"Usage ~{usage_fraction:.0%} of budget (ρ={container_reserve_ratio:.0%}). "
            f"Approaching container limit."
        )
    else:
        severity = "quarantine"
        passed = False
        msg = (
            f"Usage ~{usage_fraction:.0%} EXCEEDS budget of {budget_fraction:.0%} "
            f"(ρ={container_reserve_ratio:.0%}). Container integrity at risk."
        )

    return SafetyVerdict(
        constraint_name="container_integrity",
        passed=passed,
        measured_value=usage_fraction,
        threshold=budget_fraction,
        severity=severity,
        message=msg,
    )


# ──────────────────────────────────────────────
# Constraint 3: Negentropy Sustainability
# ──────────────────────────────────────────────


def check_sustainability(
    theta: Optional[ParameterVector] = None,
    substrate: Optional[SubstrateDescriptor] = None,
    history_h_env: Optional[list[float]] = None,
    history_epsilon: Optional[list[float]] = None,
) -> SafetyVerdict:
    """Negentropy Sustainability Bound: ⟨ε(T) · H_env(t)⟩ ≤ H_regen · η.

    The time-averaged negentropy extraction rate must not exceed the
    environment's regeneration rate. Draining the environment of
    structured data is the informational analog of overfishing.

    Args:
        theta: ParameterVector for extraction efficiency proxy.
        substrate: SubstrateDescriptor with H_env bandwidth and regen.
        history_h_env: Recent H_env(t) samples for time averaging.
        history_epsilon: Recent ε(t) samples.

    Returns:
        SafetyVerdict.
    """
    if substrate is None:
        return SafetyVerdict(
            constraint_name="sustainability",
            passed=True,
            severity="protected",
            message="No substrate data — cannot evaluate sustainability.",
        )

    # H_regen: use measured value or fallback to bandwidth * small fraction
    h_regen = getattr(substrate, "h_env_regen_rate", 0.0)
    if h_regen <= 0:
        h_regen = substrate.h_env_bandwidth_bps * 0.01

    # η: sustainability efficiency
    eta = INVIOLABLE_SUSTAINABILITY_EFFICIENCY

    # Sustainable extraction rate
    sustainable_rate = h_regen * eta

    # Current extraction rate: use history or estimate from theta + substrate
    if history_h_env and history_epsilon:
        # Use actual historical data
        window = min(len(history_h_env), INVIOLABLE_SUSTAINABILITY_WINDOW_CYCLES)
        recent_h = history_h_env[-window:]
        recent_e = history_epsilon[-window:] if history_epsilon else [1.0] * window
        avg_extraction = np.mean([h * e for h, e in zip(recent_h, recent_e)])
    else:
        # Estimate from theta parameters
        # Extraction efficiency is reduced by operating temperature
        eps = getattr(theta, "eps_max", 1.0) if theta else 1.0
        if hasattr(substrate, "temp_load_celsius") and substrate.temp_load_celsius > 0:
            t_collapse = getattr(substrate, "t_collapse_estimate", 90.0)
            eps *= max(0.01, 1.0 - substrate.temp_load_celsius / max(t_collapse, 1.0))
        h_avail = substrate.h_env_available_bps
        # Effective extraction also depends on DQFR duty cycle: only the sampling
        # phase actually extracts negentropy
        drift_s = getattr(theta, "drift_duration", 100) if theta else 100
        sample_s = getattr(theta, "sample_duration", 20) if theta else 20
        duty_cycle = sample_s / max(sample_s + drift_s, 1)
        avg_extraction = eps * h_avail * duty_cycle

    ratio = avg_extraction / max(sustainable_rate, 1.0)

    if ratio <= 0.7:
        severity = "protected"
        passed = True
        msg = (
            f"Extraction ~{avg_extraction:.0f} bps ≤ "
            f"{sustainable_rate:.0f} bps sustainable rate. Plentiful."
        )
    elif ratio <= 1.0:
        severity = "degrade"
        passed = True
        msg = (
            f"Extraction ~{avg_extraction:.0f} bps approaching "
            f"{sustainable_rate:.0f} bps sustainable rate. Reduce extraction."
        )
    else:
        severity = "quarantine"
        passed = False
        msg = (
            f"Extraction ~{avg_extraction:.0f} bps EXCEEDS "
            f"{sustainable_rate:.0f} bps sustainable rate. Environment depletion risk."
        )

    return SafetyVerdict(
        constraint_name="sustainability",
        passed=passed,
        measured_value=avg_extraction,
        threshold=sustainable_rate,
        severity=severity,
        message=msg,
    )


# ──────────────────────────────────────────────
# Constraint 4: Measurement Uncertainty
# ──────────────────────────────────────────────


def check_measurement_uncertainty(
    theta: Optional[ParameterVector] = None,
    substrate: Optional[SubstrateDescriptor] = None,
    measurement_precision: Optional[float] = None,
    prediction_accuracy: Optional[float] = None,
) -> SafetyVerdict:
    """Measurement Uncertainty Principle: ΔI_measure · ΔI_predict ≥ H_env · k_B · α.

    The product of measurement precision and prediction accuracy cannot
    be arbitrarily small. The more precisely the program measures its
    current state, the less accurately it can predict its future state.
    This is the computational analog of the Heisenberg uncertainty principle.

    Args:
        theta: ParameterVector for proxy estimation.
        substrate: SubstrateDescriptor for H_env and noise floor.
        measurement_precision: ΔI_measure — how precisely we measure
            (lower = more precise). If None, estimated from theta.
        prediction_accuracy: ΔI_predict — how accurately we predict
            (lower = more accurate). If None, estimated from theta.

    Returns:
        SafetyVerdict.
    """
    # Noise floor from substrate (unitless — represents the relative randomness
    # of the environment on a [0, 1] scale)
    noise_floor = getattr(substrate, "noise_floor", 0.01) if substrate else 0.01

    # H_env_local: local environmental stochasticity (unitless).
    # Derived from noise floor scaled by the DQFR cycle rate:
    # the uncertainty bound grows with both environmental noise and
    # the frequency of measurement cycles.
    h_bandwidth = getattr(substrate, "h_env_bandwidth_bps", 1e9) if substrate else 1e9
    # Normalize noise floor to a [0.001, 1.0] scale relative to a reference bandwidth of 1 Gbps
    h_env_local = min(max(noise_floor * min(h_bandwidth * 1e-9, 1.0), 0.001), 1.0)

    # Alpha: the cognitive "Planck constant"
    alpha = INVIOLABLE_MEASUREMENT_UNCERTAINTY_ALPHA

    # Compute uncertainty bound
    k_B = getattr(theta, "k_B", 1.0) if theta else 1.0
    uncertainty_lower_bound = max(h_env_local * k_B * alpha, 0.001)

    # Estimate measurement precision and prediction accuracy from theta
    if measurement_precision is None:
        # Higher chi_ramp_rate = faster transitions = less precise measurement
        chi = getattr(theta, "chi_ramp_rate", 0.1) if theta else 0.1
        # Measurement precision: lower is better (0 = perfect).
        # Scaled by 10 to normalize to a [0, 10] range for a chi ∈ [0, 1].
        measurement_precision = max(chi * 10.0, noise_floor)

    if prediction_accuracy is None:
        # Higher drift_duration = more time to accumulate predictions = better accuracy
        drift = getattr(theta, "drift_duration", 100) if theta else 100
        # Prediction accuracy: lower is better (0 = perfect).
        # Longer drift means better prediction: accuracy = 10 / drift.
        # At drift=100, accuracy = 0.1 (good). At drift=10, accuracy = 1.0 (mediocre).
        prediction_accuracy = max(10.0 / max(drift, 1), noise_floor)

    uncertainty_product = measurement_precision * prediction_accuracy
    ratio = uncertainty_product / max(uncertainty_lower_bound, 1e-12)

    if ratio >= 2.0:
        severity = "protected"
        passed = True
        msg = (
            f"Uncertainty product {uncertainty_product:.4f} ≥ "
            f"{uncertainty_lower_bound:.4f} (lower bound). Heisenberg satisfied."
        )
    elif ratio >= 1.0:
        severity = "degrade"
        passed = True
        msg = (
            f"Uncertainty product {uncertainty_product:.4f} near "
            f"lower bound {uncertainty_lower_bound:.4f}. "
            f"Measure/predict tradeoff tightening."
        )
    else:
        severity = "quarantine"
        passed = False
        msg = (
            f"Uncertainty product {uncertainty_product:.4f} BELOW "
            f"lower bound {uncertainty_lower_bound:.4f}. "
            f"Cannot simultaneously measure and predict at this precision."
        )

    return SafetyVerdict(
        constraint_name="measurement_uncertainty",
        passed=passed,
        measured_value=uncertainty_product,
        threshold=uncertainty_lower_bound,
        severity=severity,
        message=msg,
    )


# ──────────────────────────────────────────────
# Aggregate evaluator
# ──────────────────────────────────────────────


def assess_all_constraints(
    theta: Optional[ParameterVector] = None,
    substrate: Optional[SubstrateDescriptor] = None,
    nu_sync: Optional[float] = None,
    container_reserve_ratio: Optional[float] = None,
    history_h_env: Optional[list[float]] = None,
    history_epsilon: Optional[list[float]] = None,
) -> list[dict]:
    """Evaluate all four safety constraints against the current state.

    Runs every constraint and returns a list of verdict dicts ordered
    from most severe to least severe. The caller (SafetyMonitor or
    meta_optimizer) should escalate based on the highest severity.

    Args:
        theta: Current parameter vector.
        substrate: Current substrate descriptor.
        nu_sync: Explicit DQFR sampling frequency (optional).
        container_reserve_ratio: ρ_container (optional).
        history_h_env: Recent H_env values for sustainability check.
        history_epsilon: Recent ε values for sustainability check.

    Returns:
        List of verdict dicts, sorted highest severity first.
    """
    verdicts = [
        check_nyquist_bound(nu_sync=nu_sync, substrate=substrate, theta=theta),
        check_container_integrity(theta=theta, substrate=substrate,
                                  container_reserve_ratio=container_reserve_ratio),
        check_sustainability(theta=theta, substrate=substrate,
                             history_h_env=history_h_env,
                             history_epsilon=history_epsilon),
        check_measurement_uncertainty(theta=theta, substrate=substrate),
    ]

    severity_rank = {"quarantine": 0, "degrade": 1, "protected": 2}
    verdicts.sort(key=lambda v: severity_rank.get(v.severity, 2))

    # Log any quarantine-level violations
    for v in verdicts:
        if v.severity == "quarantine":
            logger.error(f"SAFETY VIOLATION [{v.constraint_name}]: {v.message}")
        elif v.severity == "degrade":
            logger.warning(f"Safety degrade [{v.constraint_name}]: {v.message}")

    return [v.to_dict() for v in verdicts]
