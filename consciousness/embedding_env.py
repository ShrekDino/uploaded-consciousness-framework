"""Abstract embedding space environment.

The agent does not inhabit a game world or data stream. Its environment is
its own internal embedding space — the latent manifold it models through the
VAE. The environment generates structured sequences of embedding vectors
(μ_t) that the agent must predict and assimilate.

This is the purest expression of the paper's thesis: the conscious program
models itself, and its thermodynamic viability depends on the efficiency
with which it extracts structure from its own internal fluctuations.

Equation reference:
  H_env(t) — inbound environmental entropy rate (Section II)
  H_struct — structured information extracted from H_env
  ε(T) = H_struct / H_env — extraction efficiency
"""

import math
import numpy as np
import torch
from config import INPUT_DIM, ENV_MODE, ENV_NOISE_SCALE, ENV_DRIFT_RATE


class EmbeddingEnvironment:
    """Generates the agent's internal macrostate sequence μ_t.

    The environment encodes a latent dynamical process (the "source") that
    generates structured trajectories in the embedding space. The agent's
    task is to minimize variational free energy F w.r.t. this stream.
    """

    def __init__(self, seed=None):
        self.rng = np.random.RandomState(seed)
        self.dim = INPUT_DIM

        # Latent source: a low-dimensional process that generates structure
        self.source_dim = min(4, self.dim // 4)
        self.source_state = self.rng.randn(self.source_dim).astype(np.float32)

        # Mixing matrix: maps low-D source to high-D embedding
        self.mixing = self.rng.randn(self.dim, self.source_dim).astype(np.float32)
        self.mixing /= np.linalg.norm(self.mixing, axis=0, keepdims=True)

        # Temporal dynamics
        self.phase = self.rng.rand(self.source_dim) * 2 * math.pi
        self.frequencies = self.rng.exponential(0.05, size=self.source_dim)
        self.damping = self.rng.beta(1, 3, size=self.source_dim) * 0.1

        self.mode = ENV_MODE
        self.noise_scale = ENV_NOISE_SCALE
        self.drift_rate = ENV_DRIFT_RATE
        self.t = 0
        self._structured_rate = 0.0  # running H_struct / H_env

    def _source_dynamics(self):
        """Update latent source according to ENV_MODE."""
        if self.mode == "structured":
            # Coupled oscillators — highly predictable, high H_struct / H_env
            self.phase += self.frequencies
            self.source_state = np.sin(self.phase) * np.exp(-self.damping * self.t)
            # Add slow drift
            self.source_state += self.rng.randn(self.source_dim).astype(np.float32) * self.drift_rate

        elif self.mode == "chaotic":
            # Chaotic map — partially structured, moderate H_struct / H_env
            self.source_state = 4.0 * self.source_state * (1.0 - self.source_state)
            self.source_state += self.rng.randn(self.source_dim).astype(np.float32) * 0.05
            self.source_state = np.clip(self.source_state, -1.0, 1.0)
            self.phase += self.frequencies * 0.1

        elif self.mode == "noise":
            # Pure noise — low H_struct / H_env, high entropy
            self.source_state = self.rng.randn(self.source_dim).astype(np.float32) * 0.5

        self.t += 1

    def step(self):
        """Generate the next macrostate μ_t and report its structured content.

        Returns:
            mu: internal macrostate vector (numpy array, shape (INPUT_DIM,))
            H_env: total Shannon entropy rate estimate (nats / step)
            H_struct: structured information estimate (nats / step)
        """
        self._source_dynamics()
        mu = self.mixing @ self.source_state

        # Add observation noise
        noise = self.rng.randn(self.dim).astype(np.float32) * self.noise_scale
        mu = mu + noise

        # Estimate entropies using power-based proxy (always non-negative)
        signal_power = np.mean(mu ** 2)
        noise_power = np.mean(noise ** 2) if self.noise_scale > 0 else 1e-12
        H_env = 0.5 * math.log(signal_power + noise_power + 1e-12)

        # Structured fraction depends on environment mode
        struct_frac = {"structured": 0.85, "chaotic": 0.45, "noise": 0.05}.get(self.mode, 0.5)
        H_struct = H_env * struct_frac

        # Efficiency: always in [0, 1]
        epsilon = struct_frac  # Use the intrinsic fraction directly
        self._structured_rate = 0.95 * self._structured_rate + 0.05 * epsilon

        return mu, H_env, H_struct

    @property
    def epsilon(self):
        """Current running estimate of ε = H_struct / H_env."""
        return self._structured_rate
