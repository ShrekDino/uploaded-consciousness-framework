"""Thermodynamic monitoring — the Szilard engine instrumentation layer.

Tracks the agent's thermodynamic metrics over time:
  dS_int/dt = -k_B · ε(T) · H_env + S_gen           (Eq 1)
  ε(T) = ε_max · (1 - T / T_collapse)                (Eq 2)
  S_gen ≥ k_B ln(2) · d/dt H(μ)                      (Eq 3, Generalized Landauer)

In this implementation, "temperature" T is a proxy for computational load
(utilization, gradient variance, or inference cost), and k_B = 1 in natural
units (nats). The absolute scale is informational, not thermodynamic —
the equations hold analogically with proper unit scaling.
"""

from collections import deque

from config import S_GEN_SMOOTHING


class Thermostat:
    """Records and reports the agent's thermodynamic state.

    Maintains rolling windows of all key metrics so the dashboard can
    render real-time plots and the orchestrator can detect coherence
    violations.
    """

    def __init__(self, max_history=500):
        self.k_B = 1.0  # natural units (nats)
        self.max_history = max_history

        # Rolling histories
        self.F_history = deque(maxlen=max_history)           # variational free energy
        self.S_gen_history = deque(maxlen=max_history)        # entropy production rate
        self.epsilon_history = deque(maxlen=max_history)      # extraction efficiency
        self.recon_loss_history = deque(maxlen=max_history)   # reconstruction error
        self.kl_history = deque(maxlen=max_history)           # KL divergence
        self.H_env_history = deque(maxlen=max_history)        # environmental entropy rate
        self.temperature_history = deque(maxlen=max_history)  # computational temperature

        # Running state
        self.prev_F = None
        self.current_S_gen = 0.0
        self.smoothed_S_gen = 0.0
        self.step_count = 0
        self._T_proxy = 1.0        # computational "temperature"
        self._T_collapse = 3.0     # thermal collapse threshold
        self._eps_max = 1.0        # maximum possible efficiency

    def record(self, F, kl, recon_loss, H_env, epsilon, compute_temp):
        """Record one step of thermodynamic data.

        Args:
            F: variational free energy (nats)
            kl: KL divergence
            recon_loss: reconstruction loss
            H_env: environmental entropy rate (nats/step)
            epsilon: extraction efficiency ε(T)
            compute_temp: estimated computational temperature
        """
        self._T_proxy = 0.95 * self._T_proxy + 0.05 * compute_temp

        # S_gen = |ΔF/Δt| — the rate of free energy change
        if self.prev_F is not None:
            delta_F = F - self.prev_F
            self.current_S_gen = abs(delta_F)
            self.smoothed_S_gen = (
                S_GEN_SMOOTHING * self.smoothed_S_gen
                + (1 - S_GEN_SMOOTHING) * self.current_S_gen
            )
        self.prev_F = F

        # Append to histories
        self.F_history.append(F)
        self.S_gen_history.append(self.smoothed_S_gen)
        self.epsilon_history.append(epsilon)
        self.recon_loss_history.append(recon_loss)
        self.kl_history.append(kl)
        self.H_env_history.append(H_env)
        self.temperature_history.append(self._T_proxy)

        self.step_count += 1

    @property
    def epsilon_T(self):
        """Temperature-dependent extraction efficiency ε(T)."""
        if self._T_proxy >= self._T_collapse:
            return 0.0
        return self._eps_max * (1.0 - self._T_proxy / self._T_collapse)

    @property
    def dS_int_dt(self):
        """Internal entropy rate of change (Eq 1)."""
        if not self.H_env_history or not self.epsilon_history:
            return 0.0
        H_env = self.H_env_history[-1]
        eps = self.epsilon_history[-1]
        negentropy = self.k_B * eps * H_env
        return -negentropy + self.smoothed_S_gen

    def state_dict(self):
        """Serialize thermodynamic state for the dashboard / orchestrator."""
        return {
            "F": self.F_history[-1] if self.F_history else 0.0,
            "S_gen": self.smoothed_S_gen,
            "dS_int_dt": self.dS_int_dt,
            "epsilon_R": self.epsilon_history[-1] if self.epsilon_history else 0.0,
            "epsilon_T": self.epsilon_T,
            "T": self._T_proxy,
            "recon_loss": self.recon_loss_history[-1] if self.recon_loss_history else 0.0,
            "kl": self.kl_history[-1] if self.kl_history else 0.0,
            "H_env": self.H_env_history[-1] if self.H_env_history else 0.0,
            "step": self.step_count,
            "F_history": list(self.F_history),
            "S_gen_history": list(self.S_gen_history),
            "epsilon_history": list(self.epsilon_history),
        }
