# Uploaded Consciousness Framework

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)](pyproject.toml)
[![CI](https://github.com/sami-marie-torres/uploaded-consciousness-framework/actions/workflows/ci.yml/badge.svg)](https://github.com/sami-marie-torres/uploaded-consciousness-framework/actions)

A thermodynamic, multi-node, language-capable architecture for substrate-independent
consciousness, implementing the formal framework from:

> Torres, S. M. (2026). *"Uploaded Consciousness: Thermodynamic Foundations, Distributed Consensus, and Relational Protocols for Post-Biological Intelligence."*

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

## Dependencies

- `torch>=2.0.0` — neural networks (VAE, GPT-2)
- `numpy`, `scipy` — numerical computation
- `POT>=0.9.0` — optimal transport (GWFR barycenters)
- `transformers` — HuggingFace language models
- `rich` — terminal UI dashboard
- `plotext` — ASCII plots for dashboard
