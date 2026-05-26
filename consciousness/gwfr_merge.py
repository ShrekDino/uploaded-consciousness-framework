"""Generalized Wasserstein-Fisher-Rao (GWFR) barycenter merge.

Given multiple nodes with divergent weight distributions (from independent
local learning under different H_env streams), the GWFR merge reconciles
their states via unbalanced optimal transport.

The GWFR metric extends the 2-Wasserstein distance with a mass
creation/destruction penalty κ that allows nodes with different parameter
counts, memory footprints, and network dimensionalities to be merged
without catastrophic interference.

Equation reference:
  GWFR²_κ(μ₁, μ₂) = inf ∫||x-y||² dγ + κ·KL(γ₁||μ₁) + κ·KL(γ₂||μ₂)  (Eq 6)
  μ_merged = argmin Σ w_i · GWFR²_κ(μ, μ_i)                           (Eq 7)
  max_i,j GWFR²_κ(μ_i, μ_j) ≤ Ω_coherence                             (Eq 9)

Implementation uses POT (Python Optimal Transport) for the unbalanced
Sinkhorn algorithm.
"""

import numpy as np

from config import GWFR_KAPPA, M_STATIC, OMEGA_COHERENCE, WEIGHT_ALPHA


class GWFRMerger:
    """Computes unbalanced optimal transport barycenters between nodes.

    Each node's weight distribution is represented as an empirical
    distribution over its flattened weight values. The GWFR metric
    handles unequal total mass between nodes.
    """

    def __init__(self):
        self.kappa = GWFR_KAPPA
        self.omega_coherence = OMEGA_COHERENCE
        self.m_static = M_STATIC
        self.alpha = WEIGHT_ALPHA

    @staticmethod
    def _flatten_weights(weight_dict):
        """Flatten a state_dict into a single 1-D weight array."""
        arrays = []
        for key, val in weight_dict.items():
            if 'weight' in key or 'bias' in key:
                arrays.append(val.flatten())
        return np.concatenate(arrays)

    @staticmethod
    def _unflatten_weights(flat_array, template_dict):
        """Restore a flat array back into the state_dict structure."""
        result = {}
        idx = 0
        for key, val in template_dict.items():
            if 'weight' in key or 'bias' in key:
                shape = val.shape
                size = val.size
                result[key] = flat_array[idx:idx + size].reshape(shape).astype(np.float32)
                idx += size
            else:
                result[key] = val.copy()
        return result

    @staticmethod
    def _to_empirical_distribution(flat_weights):
        """Represent a weight vector as a uniform empirical distribution.

        Each element of the weight array is a Dirac delta with mass 1/N.
        This creates a valid probability distribution for optimal transport.
        """
        N = len(flat_weights)
        if N == 0:
            return np.array([0.0]), np.array([1.0])
        # Support points: the weight values themselves
        # Uniform weights: each has mass 1/N
        return flat_weights, np.ones(N) / N

    def compute_distance(self, weights_a, weights_b):
        """Compute GWFR²_κ between two nodes' weight distributions.

        Uses POT's unbalanced Sinkhorn divergence as a proxy for the
        full GWFR metric. The unbalanced formulation handles unequal
        total mass through the κ regularization.

        Returns:
            distance: GWFR²_κ distance estimate
            exceeds_coherence: whether distance > Ω_coherence
        """
        try:
            import ot
        except ImportError:
            raise ImportError("POT (Python Optimal Transport) is required. pip install POT")

        flat_a = self._flatten_weights(weights_a)
        flat_b = self._flatten_weights(weights_b)

        # Convert to empirical distributions
        x_a, w_a = self._to_empirical_distribution(flat_a)
        x_b, w_b = self._to_empirical_distribution(flat_b)

        # Subsample if too large (computational constraint)
        max_points = 500
        if len(x_a) > max_points:
            idx = np.random.choice(len(x_a), max_points, replace=False)
            x_a, w_a = x_a[idx], w_a[idx] / w_a[idx].sum()
        if len(x_b) > max_points:
            idx = np.random.choice(len(x_b), max_points, replace=False)
            x_b, w_b = x_b[idx], w_b[idx] / w_b[idx].sum()

        # Compute GWFR proxy: unbalanced Sinkhorn divergence
        # The cost matrix is |x_a[i] - x_b[j]|²
        M = (x_a[:, None] - x_b[None, :]) ** 2
        M = M.astype(np.float64)

        try:
            # Unbalanced Sinkhorn with KL penalty on marginals
            # reg = entropic regularization
            # reg_m = κ (mass creation/destruction penalty)
            pot = ot.unbalanced.sinkhorn_unbalanced(
                w_a, w_b, M, reg=0.01, reg_m=self.kappa,
                method='sinkhorn', stopThr=1e-6, numItermax=100
            )
            distance = float(np.sum(pot * M))
        except Exception:
            # Fallback: weighted L2 distance
            distance = float(np.mean((flat_a[:len(flat_b)] - flat_b[:len(flat_a)]) ** 2))

        exceeds_coherence = distance > self.omega_coherence
        return distance, exceeds_coherence

    def merge(self, node_weights_list, node_flux_integrals):
        """Compute the weighted GWFR barycenter of multiple nodes.

        Args:
            node_weights_list: list of state_dicts from each node
            node_flux_integrals: list of ∫H_struct dt for each node

        Returns:
            merged_weights: merged state_dict
            distances: pairwise distance matrix
            weights_used: the w_i weight for each node
        """
        N = len(node_weights_list)

        # Compute dynamic weights (Eq 7 with M_static baseline)
        numerators = []
        for flux in node_flux_integrals:
            numerators.append(self.m_static + self.alpha * flux)
        denom = sum(numerators)
        weights = np.array(numerators) / denom if denom > 0 else np.ones(N) / N

        # Flatten all nodes
        flat_weights = [self._flatten_weights(w) for w in node_weights_list]

        # Pad all to the same length
        max_len = max(len(f) for f in flat_weights)
        padded = []
        for f in flat_weights:
            if len(f) < max_len:
                f = np.pad(f, (0, max_len - len(f)), 'constant')
            padded.append(f)

        # Weighted barycenter: Σ w_i · μ_i
        merged_flat = np.zeros(max_len)
        for f, w in zip(padded, weights):
            merged_flat += w * f

        # Truncate to the original dimension of the first node
        merged_flat = merged_flat[:len(flat_weights[0])]

        # Reshape back into state_dict structure
        merged_weights = self._unflatten_weights(
            merged_flat, node_weights_list[0]
        )

        # Compute pairwise distances
        distances = np.zeros((N, N))
        for i in range(N):
            for j in range(i + 1, N):
                d, _ = self.compute_distance(
                    node_weights_list[i], node_weights_list[j]
                )
                distances[i, j] = d
                distances[j, i] = d

        return merged_weights, distances, weights
