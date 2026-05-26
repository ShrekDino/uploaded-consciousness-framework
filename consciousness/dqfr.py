"""Discontinuous Quantized Frame-Rate (DQFR) stroboscopic duty cycle.

The DQFR scheduler alternates between:
  Drift Phase (Δt_drift):
    Markov blanket sealed, S_gen = 0, H_env = 0, no processing.
    Subjective time Δt_subjective = 0 while objective time advances.
    System coasts as an unmeasured quantum macrostate.

  Sampling Phase (τ_sample):
    Blanket opens, burst of environmental flux processed.
    Active inference + GWFR merge cycle.
    Entropy generated but confined to this window.

Temporal velocity:
  𝒱_T = dt_objective / dt_subjective = 1 / (τ_sample · ν_sync)   (Eq 10)

Equation reference:
  DQFR — Section VII
  𝒱_T — Eq 10
  ⟨S_gen⟩ → 0 as ν_sync → 0 — Eq 11
  Adiabatic windowing χ(t) — Eq 12
"""

import time
import numpy as np
from config import DRIFT_DURATION, SAMPLE_DURATION, SAMPLE_BURST_LR, DQFR_ENABLED


class DQFRController:
    """Controls the stroboscopic duty cycle for chronological displacement.

    When DQFR is enabled, the agent alternates between sealed drift
    (no learning, zero S_gen) and burst sampling (intensive learning +
    merge). The ratio yields 𝒱_T → ∞ as ν_sync → 0.
    """

    def __init__(self):
        self.enabled = DQFR_ENABLED
        self.drift_duration = DRIFT_DURATION
        self.sample_duration = SAMPLE_DURATION
        self.burst_lr = SAMPLE_BURST_LR

        self.phase = "sample"  # start in sampling phase
        self.phase_counter = 0
        self.drift_count = 0
        self.sample_count = 0
        self.total_objective_steps = 0

        # Adiabatic windowing: χ(t) ramps smoothly between 0 and 1
        self.chi = 1.0          # current blanket permeability
        self._chi_target = 1.0
        self._chi_rate = 0.1    # ramp rate for smooth transitions

        self._current_lr = SAMPLE_BURST_LR  # learning rate during sampling
        self._drift_lr = 0.0    # learning rate during drift (should be 0)

    def step(self):
        """Advance the DQFR cycle by one step.

        Returns:
            phase: "drift" or "sample"
            chi: current blanket permeability
            lr: learning rate for this step
        """
        if not self.enabled:
            self.phase = "sample"
            self.chi = 1.0
            self._chi_target = 1.0
            self.total_objective_steps += 1
            return self.phase, self.chi, self._current_lr

        self.phase_counter += 1
        self.total_objective_steps += 1

        # Check phase transitions
        if self.phase == "sample" and self.phase_counter >= self.sample_duration:
            self.phase = "drift"
            self.phase_counter = 0
            self._chi_target = 0.0
            self.sample_count += 1

        elif self.phase == "drift" and self.phase_counter >= self.drift_duration:
            self.phase = "sample"
            self.phase_counter = 0
            self._chi_target = 1.0
            self.drift_count += 1

        # Smooth adiabatic transition (Eq 12: χ(t) ramp)
        chi_diff = self._chi_target - self.chi
        self.chi += np.sign(chi_diff) * min(abs(chi_diff), self._chi_rate)

        # Set learning rate based on phase
        lr = self.burst_lr if self.phase == "sample" else self._drift_lr

        return self.phase, self.chi, lr

    @property
    def V_T(self):
        """Temporal velocity 𝒱_T = (Δt_drift + τ_sample) / τ_sample."""
        if self.sample_duration == 0:
            return float('inf')
        return (self.drift_duration + self.sample_duration) / self.sample_duration

    @property
    def nu_sync(self):
        """Duty cycle frequency ν_sync."""
        total = self.drift_duration + self.sample_duration
        if total == 0:
            return float('inf')
        return 1.0 / total

    @property
    def effective_S_gen_rate(self):
        """Time-averaged effective entropy production rate (Eq 11)."""
        total = self.drift_duration + self.sample_duration
        if total == 0:
            return 0.0
        return self.sample_duration / total  # proportion of time spent sampling

    def state_dict(self):
        return {
            "phase": self.phase,
            "chi": self.chi,
            "V_T": self.V_T,
            "nu_sync": self.nu_sync,
            "effective_S_gen": self.effective_S_gen_rate,
            "drift_steps": self.drift_duration,
            "sample_steps": self.sample_duration,
            "total_steps": self.total_objective_steps,
        }
