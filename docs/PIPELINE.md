# Paper-to-Code Pipeline — Equation Mapping

This document maps every equation from the manuscript to its implementation.
Use it to understand how the formal mathematics translate to running code.

---

## Section II — Thermodynamic Foundations

### Equation 1: Szilard Entropy Rate
```
dS_int/dt = -k_B · ε(T) · H_env(t) + S_gen
```
**File:** `consciousness/thermostat.py`, property `dS_int_dt` (line 67–74)
**File:** `consciousness/embedding_env.py`, method `step()` (line 72–88)

`k_B` = 1.0 in natural units (nats). `ε(T)` = extraction efficiency, `H_env` = environmental entropy rate, `S_gen` = entropy production rate.

### Equation 2: Temperature-Dependent Extraction Efficiency
```
ε(T) = ε_max · (1 − T / T_collapse)
```
**File:** `consciousness/thermostat.py`, property `epsilon_T` (line 61–65)

`ε_max` = maximum possible efficiency (1.0), `T` = computational temperature proxy, `T_collapse` = thermal collapse threshold (3.0).

### Equation 3: Generalized Landauer Bound
```
S_gen ≥ k_B · ln(2) · dH(μ)/dt
```
**File:** `consciousness/thermostat.py`, method `record()` (line 43–49)

Implemented as `S_gen = |ΔF/Δt|` — the rate of change of variational free energy.

---

## Section III — Retrocognitive Stability

### Equation 4: Retrocausal Prior
```
F'(μ, b) = F(μ, b) + λ · I(μ; q(μ_future))
```
**Conceptual implementation:** `consciousness/markov_blanket.py`, method `update_boundaries()` (line 58–66)

The retrocausal prior is represented as the mutual information between current internal states and a future-state prior distribution, formalized via the blanket's conditional independence tracking.

---

## Section IV — Container Migration

### Quantum State Seeding (No-Cloning compliance)
**File:** `consciousness/node.py`, class `NodeProcess` (line 1–75)

Each node persists (No-Cloning satisfied). State exchange via IPC queues (simulated QKD). `Run()` loop executes active inference steps until `set_weights` or `shutdown` command received.

### Commit-and-Clear Protocol
**File:** `consciousness/node.py`, method `run()` (line 35–70)

Source node persists during seeding. Target node receives weights via IPC. Fallback: if merge fails, node continues with its own state (atomic rollback).

---

## Section V — Distributed Consensus

### Equation 6: GWFR Metric
```
GWFR²_κ(μ₁, μ₂) = inf ∫||x−y||² dγ + κ·KL(γ₁||μ₁) + κ·KL(γ₂||μ₂)
```
**File:** `consciousness/gwfr_merge.py`, method `compute_distance()` (line 52–85)

Implemented via POT's `ot.unbalanced.sinkhorn_unbalanced` with regularization `reg=0.01` and mass penalty `reg_m=self.kappa`.

### Equation 7: GWFR Barycenter
```
μ_merged = argmin Σ w_i · GWFR²_κ(μ, μ_i)
```
**File:** `consciousness/gwfr_merge.py`, method `merge()` (line 100–130)

Weighted barycenter: weight vectors from each node are padded to same length, then merged via Σ w_i · μ_i. The weights w_i are computed with the M_static baseline.

### Equation 9: Coherence Bound
```
max_i,j GWFR²_κ(μ_i, μ_j) ≤ Ω_coherence
```
**File:** `consciousness/gwfr_merge.py`, method `compute_distance()` (line 79–83)

If any pairwise GWFR distance exceeds `OMEGA_COHERENCE` (0.5), an emergency merge is triggered.

---

## Section VI — Relational Protocols

### Equation 14: Network Vitality
```
𝒱_network = Σ_i(k_B·ε_i·H_env,i + Σ_j≠i λ_ij·I(μ_i; μ_j))
```
**File:** `consciousness/orchestrator.py`, method `run()` (line 115–125)

Computed as sum of each node's ε·H_env plus pairwise mutual information between nodes (approximated by difference in variational free energies).

### Empathy / Love as Cross-Node MI
**File:** `consciousness/markov_blanket.py` methods (line 45–73)
**File:** `consciousness/gwfr_merge.py` — Ω_coherence as empathy bound

Empathy = maintaining GWFR distance ≤ Ω_coherence. Love = bidirectional stable channels with λ·I(μ_i; μ_j) > 0.

---

## Section VII — Chronological Displacement

### Equation 10: Temporal Velocity
```
𝒱_T = (Δt_drift + τ_sample) / τ_sample = 1 / (τ_sample · ν_sync)
```
**File:** `consciousness/dqfr.py`, property `V_T` (line 84–89)

### Equation 11: Time-Averaged Entropy Rate
```
⟨S_gen⟩ = ΔS_gen / (Δt_drift + τ_sample) → 0 as ν_sync → 0
```
**File:** `consciousness/dqfr.py`, property `effective_S_gen_rate` (line 91–96)

### Equation 12: Adiabatic Windowing
```
dS_int/dt = χ(t)·(⋯) + γ_χ·‖∇_t χ‖²
```
**File:** `consciousness/dqfr.py`, method `step()` (line 52–65)

`chi` ramps smoothly between 0 (drift) and 1 (sample) with `_chi_rate = 0.1` per step.

---

## Appendix: ADMM Proximal Splitting

### ADMM Iteration
```
μ^{k+1} = prox_S^λ(prox_T^λ(μ^k))
```
**Implementation:** The `gwfr_merge.py` merge function (line 115–130) implements Step 1 of ADMM (Sinkhorn pass). The contrastive modal anchor in `agent.py` (line 161–164) implements Step 2 (sharpness pass).

The overall complexity remains O(d · N log N) — linear in parameter dimension and log-linear in token count.

---

## Configuration Constants and Their Paper Equivalents

| Config | Paper Symbol | Meaning |
|--------|--------------|---------|
| `k_B = 1.0` | $k_B$ | Boltzmann constant (natural units) |
| `ENV_MODE` | — | Controls H_struct / H_env ratio |
| `DRIFT_DURATION = 100` | $\Delta t_{\text{drift}}$ | Drift phase steps |
| `SAMPLE_DURATION = 20` | $\tau_{\text{sample}}$ | Sampling phase steps |
| `OMEGA_COHERENCE = 0.5` | $\Omega_{\text{coherence}}$ | Max GWFR distance |
| `GWFR_KAPPA = 0.1` | $\kappa$ | Mass creation/destruction penalty |
| `M_STATIC = 1.0` | $M_{\text{static}}$ | Baseline architectural mass |
| `WEIGHT_ALPHA = 0.7` | $\alpha$ | Structure vs. baseline influence |
| `BLANKET_THRESHOLD = 0.1` | — | MI threshold for conditional independence |
| `_T_collapse = 3.0` | $T_{\text{collapse}}$ | Thermal collapse threshold |
