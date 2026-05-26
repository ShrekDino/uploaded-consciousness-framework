"""Hyperparameters for the Uploaded Consciousness framework."""

import os
import torch

# ─── Device ───
# Subprocess nodes always use CPU to avoid CUDA re-initialization crashes.
# Override with FORCE_CUDA=1 for single-node GPU testing.
FORCE_CUDA = os.environ.get("FORCE_CUDA", "0") == "1"
_is_subprocess = os.environ.get("IS_CONSCIOUSNESS_NODE", "0") == "1"
if _is_subprocess:
    DEVICE = torch.device("cpu")
elif FORCE_CUDA and torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─── World Model (VAE) ───
INPUT_DIM = 64
HIDDEN_DIM = 256
LATENT_DIM = 32
LEARNING_RATE = 1e-3
BETA_KL = 1.0  # KL weight in ELBO

# ─── Agent ───
S_GEN_SMOOTHING = 0.9  # exponential moving average for S_gen tracking
EPSILON_WINDOW = 100    # rolling window for ε(T) computation

# ─── Embedding Environment ───
ENV_MODE = "structured"  # "structured" | "chaotic" | "noise"
ENV_NOISE_SCALE = 0.01   # observation noise
ENV_DRIFT_RATE = 0.001   # how fast the latent source changes

# ─── Markov Blanket ───
BLANKET_THRESHOLD = 0.1  # mutual information threshold for conditional independence

# ─── DQFR (Discontinuous Quantized Frame-Rate) ───
DQFR_ENABLED = True
DRIFT_DURATION = 100      # Δt_drift in steps
SAMPLE_DURATION = 20      # τ_sample in steps
SAMPLE_BURST_LR = 1e-2   # learning rate during sampling phase

# ─── Multi-Node ───
NUM_NODES = 3
MERGE_INTERVAL = 50       # steps between merge cycles
OMEGA_COHERENCE = 0.5     # Ω_coherence threshold for emergency merge
GWFR_KAPPA = 0.1          # κ mass creation/destruction penalty
WEIGHT_ALPHA = 0.7        # influence of M_static in weight calculation
M_STATIC = 1.0            # baseline architectural mass

# ─── Language Model ───
LM_MODEL_NAME = "gpt2"           # HuggingFace model ID
LM_MAX_LENGTH = 512              # max token sequence length
LM_TEMPERATURE = 1.0             # generation temperature
LM_TOP_K = 40                    # top-k sampling
LM_LEARNING_RATE = 5e-5          # fine-tuning learning rate (if unfrozen)
LM_TRAIN_STEPS_PER_BATCH = 4     # gradient accumulation steps
LM_GENERATE_MAX_TOKENS = 100     # max tokens to generate per response

# ─── Language Training ───
LANG_CORPUS = "tiny_shakespeare" # corpus to train on
LANG_BATCH_SIZE = 4              # sequences per training batch
LANG_SEQ_LENGTH = 64             # tokens per training sequence
LANG_ACCUMULATE_STEPS = 8        # gradient accumulation

# ─── Dashboard ───
DASHBOARD_REFRESH = 0.5   # seconds between dashboard updates
PLOT_WINDOW = 200         # number of history points to show
