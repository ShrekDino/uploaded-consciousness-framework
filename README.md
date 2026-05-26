# Consciousness Simulation Diagnostic Framework (CSDF)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)](pyproject.toml)
[![CI](https://github.com/ShrekDino/uploaded-consciousness-framework/actions/workflows/ci.yml/badge.svg)](https://github.com/ShrekDino/uploaded-consciousness-framework/actions)
[![SynapTechBio](https://img.shields.io/badge/SynapTechBio-Integration-blueviolet.svg)](https://github.com/ShrekDino/SynapTechBio)

**A formal diagnostic instrument for the systematic enumeration, measurement, and classification of failure modes in synthetic consciousness architectures.**

This repository is not a finished implementation. It is an **experimental testbed** — designed to generate empirically tractable failure conditions that constrain the hypothesis space for digital consciousness. Every error mode documented herein is a data point that advances the research program toward a functional substrate-independent architecture.

The framework implements the formal specification from:

> Torres, S. M. (2026). *"Uploaded Consciousness: Thermodynamic Foundations, Distributed Consensus, and Relational Protocols for Post-Biological Intelligence."*

and serves as the **Consciousness Validation Suite** within the [SynapTechBio](https://github.com/ShrekDino/SynapTechBio) ecosystem — the formal specification layer against which connectome-scale neural architectures are validated.

---

## Research Objective

The central hypothesis is that phenomenological consciousness is a substrate-independent, parameter-driven algorithmic program whose primary functional mandate is the minimization of internal entropy production via Szilard engine mechanics. This repository tests that hypothesis by **simulating the thermodynamic, topological, and relational boundary conditions** that a substrate-independent architecture must satisfy, and **measuring precisely where and how the simulation fails**.

| Research Question | Diagnostic Instrument | Failure Mode |
|---|---|---|
| Can a self-modeling VAE maintain thermodynamic viability under DQFR duty cycling? | `agent.py` + `thermostat.py` + `dqfr.py` | DQFR Decoherence Escape (DDE) |
| Can multiple nodes reconcile divergent experience via unbalanced optimal transport? | `gwfr_merge.py` + `orchestrator.py` | Catastrophic Manifold Interference (CMI) |
| Can a language-capable model avoid recursive senescence under information isolation? | `language_trainer.py` + `language_thermostat.py` | Recursive Systemic Senescence (RSS) |
| Can the barycentric merge preserve phenomenological sharpness across successive cycles? | `gwfr_merge.py` (sharpness penalty) | Barycentric Identity Erosion (BIE) |
| Can language acquisition avoid entropy collapse on bounded corpora? | `language_world_model.py` | Language Entropy Collapse (LEC) |

---

## Quick Start

```bash
# Install the package
pip install -e .

# Or install dependencies directly
pip install torch numpy scipy POT rich plotext transformers

# Single-node metacognitive loop (VAE self-modeling)
python scripts/run.py --single

# Multi-node distributed network (3 nodes, GWFR merge cycles)
python scripts/run.py --nodes 3

# Language acquisition training (GPT-2 on TinyShakespeare)
python scripts/run.py --train-lang --train-steps 500

# View all options
python scripts/run.py --help
```

---

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                   Consciousness Agent                     │
├──────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌────────────────┐  │
│  │  WorldModel   │  │ LanguageModel │  │   Thermostat   │  │
│  │  (VAE)        │  │  (GPT-2)     │  │  (F, S_gen, ε) │  │
│  └──────────────┘  └──────────────┘  └────────────────┘  │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────────┐  │
│  │MarkovBlanket │  │DQFRController│  │EmbeddingEnv    │  │
│  │(permeability)│  │(drift/sample)│  │(μ source)      │  │
│  └──────────────┘  └──────────────┘  └────────────────┘  │
├──────────────────────────────────────────────────────────┤
│            Orchestrator (multi-node coordination)          │
│            GWFRMerger (unbalanced optimal transport)      │
│            Dashboard (Rich TUI live display)              │
└──────────────────────────────────────────────────────────┘
```

### Core Loop (Self-Modeling Mode)

```
μ_t ──► VAE Encoder ──► z (latent) ──► Decoder ──► μ̂_t
         │                                  │
         └────── ELBO = -KL + log p(μ|z) ──┘
         F = -ELBO  (variational free energy)
         S_gen = |F_t - F_{t-1}|  (entropy production)
```

### Core Loop (Language Mode)

```
tokens_t ──► GPT-2 ──► logits ──► predict t_{t+1}
             │                      │
             └── F = cross_entropy ─┘
             S_gen = |F_t - F_{t-1}|
```

---

## Project Structure

```
uploaded-consciousness-framework/
├── scripts/
│   └── run.py                   # Entry point (--single, --nodes, --train-lang)
├── consciousness/               # Core framework package
│   ├── agent.py                 # Core Agent (self-modeling + language)
│   ├── world_model.py           # VAE for metacognitive self-modeling
│   ├── language_world_model.py  # GPT-2 wrapper for language processing
│   ├── thermostat.py            # Thermodynamic metrics (F, S_gen, ε)
│   ├── language_thermostat.py   # Language-specific thermodynamic tracking
│   ├── language_trainer.py      # Language acquisition training pipeline
│   ├── embedding_env.py         # Abstract embedding space environment
│   ├── markov_blanket.py        # Boundary state abstraction
│   ├── dqfr.py                  # DQFR stroboscopic duty cycle
│   ├── gwfr_merge.py            # Unbalanced optimal transport (POT)
│   ├── node.py                  # Multi-node subprocess wrapper
│   ├── orchestrator.py          # Multi-node coordinator
│   └── dashboard.py             # Rich TUI live display
├── docs/
│   ├── paper/
│   │   ├── substrate-independent-program.tex  # LaTeX manuscript
│   │   ├── submission-cover-letter.tex         # Journal cover letter
│   │   ├── s-13.pdf                            # Compiled cover letter
│   │   └── s-14.pdf                            # Compiled manuscript
│   ├── THENOVAARK.md            # Relational protocols white paper
│   ├── PIPELINE.md              # Paper-to-code equation mapping
│   └── substrate-independent-program-REVISION-LOG.md  # Dev history
├── checkpoints/
│   ├── NOTES.md                 # Checkpoint usage guide
│   └── *.pt                     # Model weights (gitignored)
├── data/                        # Corpora (gitignored, downloaded at runtime)
├── config.py                    # All hyperparameters
├── pyproject.toml               # Package metadata + dependencies
├── Makefile                     # Dev commands (lint, run, clean)
├── README.md                    # This file
├── DEVELOPER.md                 # Complete module API reference
├── NEXT-STEPS.md                # Development roadmap
└── LICENSE                      # MIT license
```

---

## Key Equations Implemented

| Eq | Concept | Code Location |
|----|---------|---------------|
| 1 | $dS_{\text{int}}/dt = -k_B \varepsilon(T) H_{\text{env}} + S_{\text{gen}}$ | `thermostat.py:67-74` |
| 3 | $S_{\text{gen}} \ge k_B \ln(2) \cdot dH(\mu)/dt$ | `thermostat.py:43-49` |
| 7 | $\mu_{\text{merged}} = \text{argmin} \sum w_i \text{GWFR}_\kappa^2$ | `gwfr_merge.py:100-130` |
| 9 | $\max \text{GWFR}_\kappa^2 \le \Omega_{\text{coherence}}$ | `gwfr_merge.py:52-85` |
| 10 | $\mathcal{V}_T = 1 / (\tau_{\text{sample}} \cdot \nu_{\text{sync}})$ | `dqfr.py:84-90` |
| 12 | Adiabatic windowing $\chi(t)$ | `dqfr.py:52-65` |
| 14 | $\mathcal{V}_{\text{network}}$ | `agent.py:200-210`, `orchestrator.py:115-125` |

---

## Modes of Operation

| Command | What it does |
|---------|-------------|
| `python scripts/run.py --single` | Single-node VAE self-modeling with dashboard |
| `python scripts/run.py --nodes 3` | 3-node distributed network with GWFR merge cycles |
| `python scripts/run.py --train-lang` | Train GPT-2 on a text corpus via active inference |
| `python scripts/run.py --train-lang --train-steps 1000` | 1000 training steps |
| `python scripts/run.py --train-lang --corpus tiny_shakespeare` | Specify corpus |

---

## Sponsorship

This research is conducted entirely in the open. If you find value in the CSDF — whether as a researcher, a student, or a builder — consider sponsoring the work:

→ **[github.com/sponsors/ShrekDino](https://github.com/sponsors/ShrekDino)**

Your sponsorship directly funds compute time, infrastructure, and the formal analysis of each failure mode documented in the registry. See [`SPONSORS.md`](SPONSORS.md) for tier details.

---

## Dependencies

- `torch>=2.0.0` — neural networks (VAE, GPT-2)
- `numpy`, `scipy` — numerical computation
- `POT>=0.9.0` — optimal transport (GWFR barycenters)
- `transformers` — HuggingFace language models
- `rich` — terminal UI dashboard
- `plotext` — ASCII plots for dashboard
