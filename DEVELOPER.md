# Developer Reference — Complete Module API

---

## `config.py` — Hyperparameters

All system parameters in one file. Edit before runs to change behavior.

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `DEVICE` | auto | torch.device (cpu/cuda). Subprocess nodes forced to cpu |
| `INPUT_DIM` | 64 | VAE embedding dimension |
| `HIDDEN_DIM` | 256 | VAE hidden layer size |
| `LATENT_DIM` | 32 | VAE latent code dimension |
| `DQFR_ENABLED` | True | Enable stroboscopic duty cycle |
| `DRIFT_DURATION` | 100 | Steps with zero S_gen |
| `SAMPLE_DURATION` | 20 | Steps of burst processing |
| `NUM_NODES` | 3 | Multi-node network size |
| `OMEGA_COHERENCE` | 0.5 | Max GWFR distance before emergency merge |
| `LM_MODEL_NAME` | "gpt2" | HuggingFace model for language |
| `LANG_BATCH_SIZE` | 4 | Sequences per training batch |
| `LANG_SEQ_LENGTH` | 64 | Tokens per sequence |

---

## `consciousness/agent.py` — `Agent` class

The central agent. Runs either self-modeling (VAE) or language (GPT-2) loops.

### Constructor
```python
Agent(node_id=0, env_seed=None)
```

### Methods

**`set_mode(mode)`** — Switch between "self" and "language" modes.
- "self": VAE metacognitive loop (default)
- "language": GPT-2 language processing (loads model lazily)

**`step()`** — One self-modeling step.
Returns dict with: `F`, `S_gen`, `epsilon`, `phase`, `chi`, `dS_int_dt`, `H_env`, `kl`, `recon`, `compute_temp`, `blanket`, `dqfr`

**`language_step(input_ids, attention_mask, do_train=False)`** — One language step.
Returns dict with: `F`, `perplexity`, `accuracy`, `H_lang`, `epsilon_lang`, `phase`, `chi`, `tokens`

**`get_weights()`** — Serialized VAE/LM weights for GWFR merge.
**`set_weights(state_dict)`** — Load merged weights.
**`state_dict()`** — Full serialized state for IPC.

---

## `consciousness/world_model.py` — `WorldModel` class

VAE generative model of the agent's internal macrostate μ.

### Architecture
```
Encoder: INPUT_DIM → HIDDEN_DIM → HIDDEN_DIM → LATENT_DIM (mu, logvar)
Decoder: LATENT_DIM → HIDDEN_DIM → HIDDEN_DIM → INPUT_DIM
```

### Methods
**`forward(x)`** → `(x_hat, z, mu_z, logvar_z)`
**`compute_elbo(x, x_hat, mu_z, logvar_z)`** → `(elbo, kl, recon_loss)`
**`compute_free_energy(x)`** → `(F, kl, recon, x_hat, z)`
**`step_optimize(x, optimizer)`** → `(F, kl, recon)` — one backprop step

---

## `consciousness/language_world_model.py` — `LanguageWorldModel` class

GPT-2 wrapper with the same interface as `WorldModel`.

### Methods
**`forward(input_ids, attention_mask)`** → `(logits, hidden_states, pooled)`
- `pooled`: last-token hidden state (serves as μ for the VAE embedding bridge)

**`compute_free_energy(input_ids, attention_mask)`** → `(F, perplexity, accuracy, H_lang, pooled)`
- F = cross-entropy loss

**`generate(prompt_text, max_new_tokens, temperature, top_k)`** → `(text, n_tokens, F_trace)`
- Generates text with top-k sampling, tracks F per token

**`encode_text(text)`** → `(input_ids, attention_mask)`
**`decode_tokens(token_ids)`** → `str`
**`train_mode()`** / **`eval_mode()`** — Toggle gradient tracking

---

## `consciousness/thermostat.py` — `Thermostat` class

Records and reports thermodynamic state. Rolling windows of all metrics.

### Key Methods
**`record(F, kl, recon_loss, H_env, epsilon, compute_temp)`**
**`state_dict()`** → dict with: `F, S_gen, dS_int_dt, epsilon_R, epsilon_T, T, F_history, S_gen_history, epsilon_history`

### Properties
- `epsilon_T`: ε(T) = ε_max · (1 − T/T_collapse) — Eq 2
- `dS_int_dt`: −k_B · ε · H_env + S_gen — Eq 1

---

## `consciousness/language_thermostat.py` — `LanguageThermostat(Thermostat)` class

Extends Thermostat with language-specific metrics.

### Additional Methods
**`record_language(F, perplexity, accuracy, H_lang)`**
**`language_state_dict()`** — includes: `perplexity, token_accuracy, H_lang, epsilon_lang, tokens_processed, vocab_size`

### Properties
- `epsilon_lang`: current language extraction efficiency

---

## `consciousness/embedding_env.py` — `EmbeddingEnvironment` class

Generates structured sequences in the abstract embedding space.

### Constructor
```python
EmbeddingEnvironment(seed=None)
```

### Modes (config.ENV_MODE)
- `"structured"`: Coupled oscillators — 85% of flux structurally resolvable
- `"chaotic"`: Logistic map — 45% resolvable
- `"noise"`: Pure noise — 5% resolvable

### Methods
**`step()`** → `(mu, H_env, H_struct)` — next macrostate vector

### Property
- `epsilon`: running estimate of ε = H_struct / H_env

---

## `consciousness/markov_blanket.py` — `MarkovBlanket` class

Statistical boundary between internal (μ), blanket (b), and external (ψ) states.

### Methods
**`update_boundaries(mu, b, psi)`** — estimate mutual information between partitions
**`seal()`** — close blanket (permeability = 0, Drift Phase)
**`open()`** — open blanket (permeability = 1, Sampling Phase)
**`state_dict()`** — returns: `permeability, is_open, I(μ;b), I(ψ;b), I(μ;ψ), ci_violation`

### Property
- `conditional_independence_violation`: I(μ;ψ) − min(I(μ;b), I(ψ;b)) — should be 0

---

## `consciousness/dqfr.py` — `DQFRController` class

Implements the Discontinuous Quantized Frame-Rate stroboscopic duty cycle (Section VII).

### Constructor
```python
DQFRController()
```
Uses `DRIFT_DURATION`, `SAMPLE_DURATION`, `SAMPLE_BURST_LR` from config.

### Methods
**`step()`** → `(phase, chi, lr)` — advance one step, update phase
- `phase`: "drift" or "sample"
- `chi`: adiabatic windowing permeability [0, 1] — Eq 12
- `lr`: learning rate (burst_lr during sample, 0 during drift)

### Properties
- `V_T`: temporal velocity = (Δt_drift + τ_sample) / τ_sample — Eq 10
- `nu_sync`: duty cycle frequency
- `effective_S_gen_rate`: τ_sample / (Δt_drift + τ_sample) — Eq 11

### State Dict
Returns: `phase, chi, V_T, nu_sync, effective_S_gen, drift_steps, sample_steps, total_steps`

---

## `consciousness/gwfr_merge.py` — `GWFRMerger` class

Computes unbalanced optimal transport barycenters between node weight distributions.

### Methods
**`compute_distance(weights_a, weights_b)`** → `(distance, exceeds_coherence)`
- Uses POT's `ot.unbalanced.sinkhorn_unbalanced` as GWFR proxy
- Falls back to weighted L2 if POT fails

**`merge(node_weights_list, node_flux_integrals)`** → `(merged_weights, distances, weights_used)`
- Weights: w_i = (M_static + α · ∫H_struct_i) / Σ(...) — Eq 7 + M_static baseline
- Pads all weight vectors to same length, computes weighted barycenter
- Returns pairwise distance matrix and the w_i weights used

---

## `consciousness/node.py` — `NodeProcess(mp.Process)` class

Subprocess wrapper that runs an Agent in isolation.

### Constructor
```python
NodeProcess(node_id, cmd_queue, state_queue, seed=None)
```

### Command Protocol
- `{"type": "step", "n_steps": N}` — run N inference steps
- `{"type": "set_weights", "weights": {...}}` — load merged weights
- `{"type": "shutdown"}` — terminate process

### State Output
Periodically puts state dicts on `state_queue` with: `weights, thermo, flux_accumulated, step, metrics`

---

## `consciousness/orchestrator.py` — `Orchestrator` class

Coordinates the multi-node distributed network.

### Constructor
```python
Orchestrator(num_nodes=NUM_NODES)
```

### Methods
**`start()`** — spawn all node processes
**`run(steps)`** — yields metrics generator for N merge cycles
**`shutdown()`** — terminate all nodes

### Merge Cycle
1. Broadcast `step` command to all nodes
2. Wait for processing delay
3. Collect states from all nodes
4. Extract weights and accumulated flux
5. Call `GWFRMerger.merge()` to compute barycenter
6. Broadcast `set_weights` to all nodes
7. Check Ω_coherence, trigger emergency merge if exceeded

---

## `consciousness/language_trainer.py` — `LanguageTrainer` class

Training pipeline for language acquisition.

### Constructor
```python
LanguageTrainer(agent, corpus_name="tiny_shakespeare")
```

### Methods
**`train(num_steps, log_interval, eval_interval)`** → dict with final metrics
- Downloads corpus if not present (TinyShakespeare from Karpathy's repo)
- Tokenizes via the agent's LM tokenizer
- Runs N training steps, each: sample batch → forward → compute F → backprop
- At eval_interval: generates sample text from 3 prompts
- Saves checkpoint to `checkpoints/lm_step_{N}.pt`

### Corpus Format
The `CorpusLoader` class handles:
- TinyShakespeare (auto-downloaded)
- Any local .txt file (specify path via CORPUS config or add to SUPPORTED_CORPORA)

---

## `consciousness/dashboard.py` — Dashboard classes

Real-time terminal UI using Rich.

### `Dashboard` class
- `update(metrics)`: ingest orchestrator metrics
- `refresh(metrics)`: return Rich Layout for current state

### `run_dashboard(orchestrator, steps)` — convenience function
Runs orchestrator with live Rich display showing:
- Per-node metrics table (F, S_gen, ε, phase, χ)
- Network panel (𝒱_network, pairwise MI, merge status, Ω gauge)
- Live ASCII plots of F(t), S_gen(t), ε(t)

---

## `run.py` — Entry Point

| Option | Default | Description |
|--------|---------|-------------|
| `--single` | False | Single-node VAE self-modeling mode |
| `--nodes` | 3 | Number of nodes for multi-node mode |
| `--steps` | 50 | Merge cycles (multi-node) or display steps (single) |
| `--train-lang` | False | Language acquisition training mode |
| `--train-steps` | 200 | Number of language training steps |
| `--corpus` | "tiny_shakespeare" | Training corpus name |
