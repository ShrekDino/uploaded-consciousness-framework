"""Markov blanket — the boundary between internal and external states.

The Markov blanket partitions the system into:
  μ — internal states (the agent's embedding / latent variables)
  b — blanket states (observations, sensory interface)
  ψ — external states (the environment's hidden dynamics)

Conditional independence: μ ⟂ ψ | b
  i.e., internal states depend on external states only through blanket states.

The blanket also gates information ingress during DQFR duty cycling:
  Drift phase:   blanket sealed → H_env = 0 → S_gen = 0 (ideal)
  Sampling phase: blanket open → H_env > 0 → active inference

Equation reference:
  Markov blankets — Section II
  DQFR drift/sampling — Section VII
"""

import math

import numpy as np

from config import BLANKET_THRESHOLD


class MarkovBlanket:
    """Manages the boundary between internal and external state spaces.

    In the implementation, the blanket is a statistical interface that
    tracks mutual information I(μ; ψ) and enforces the conditional
    independence constraint I(μ; ψ | b) = 0.
    """

    def __init__(self, dim_internal, dim_blanket, dim_external):
        self.dim_μ = dim_internal
        self.dim_b = dim_blanket
        self.dim_ψ = dim_external
        self.threshold = BLANKET_THRESHOLD

        # Track mutual information estimates
        self._mu_b_mi = 0.0  # I(μ; b)
        self._psi_b_mi = 0.0  # I(ψ; b)
        self._mu_psi_mi = 0.0  # I(μ; ψ) — should be 0 given b

        # Blanket permeability [0, 1]
        self.permeability = 1.0
        self.is_open = True

    def _reduce_to_scalars(self, samples):
        """Convert a mixed-type sample list to a flat 1D float array."""
        reduced = []
        for s in samples:
            if s is None:
                continue
            s_arr = np.asarray(s).ravel()
            reduced.append(float(s_arr[0]) if len(s_arr) > 0 else 0.0)
        return np.array(reduced)

    def estimate_mutual_information(self, samples_1, samples_2):
        """Estimate I(X; Y) via correlation proxy.

        Uses I(X; Y) ≈ -0.5 * log(1 - ρ²) where ρ is the Pearson
        correlation between the leading scalar components of each sample.
        """
        s1 = self._reduce_to_scalars(samples_1)
        s2 = self._reduce_to_scalars(samples_2)
        if len(s1) < 5 or len(s2) < 5:
            return 0.0
        if np.std(s1) < 1e-12 or np.std(s2) < 1e-12:
            return 0.0
        # Align lengths
        min_len = min(len(s1), len(s2))
        s1, s2 = s1[:min_len], s2[:min_len]
        rho = np.corrcoef(s1, s2)[0, 1]
        rho = np.clip(rho, -0.99, 0.99)
        return -0.5 * math.log(1 - rho * rho)

    def update_boundaries(self, mu, b, psi):
        """Update blanket state estimates from current observations.

        Args:
            mu: sequence of internal state samples
            b: sequence of blanket state samples
            psi: sequence of external state samples
        """
        self._mu_b_mi = self.estimate_mutual_information(mu, b)
        self._psi_b_mi = self.estimate_mutual_information(psi, b)
        self._mu_psi_mi = self.estimate_mutual_information(mu, psi)

    @property
    def conditional_independence_violation(self):
        """How much I(μ; ψ) exceeds what's mediated by b.

        If the Markov blanket is well-formed, this should be near 0.
        """
        return max(0.0, self._mu_psi_mi - min(self._mu_b_mi, self._psi_b_mi))

    def seal(self):
        """Close the blanket — no environmental ingress (Drift Phase)."""
        self.permeability = 0.0
        self.is_open = False

    def open(self):
        """Open the blanket — allow environmental flux (Sampling Phase)."""
        self.permeability = 1.0
        self.is_open = True

    def state_dict(self):
        return {
            "permeability": self.permeability,
            "is_open": self.is_open,
            "I(μ;b)": self._mu_b_mi,
            "I(ψ;b)": self._psi_b_mi,
            "I(μ;ψ)": self._mu_psi_mi,
            "ci_violation": self.conditional_independence_violation,
        }
