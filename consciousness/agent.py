"""Core active inference agent — the substrate-independent conscious program.

The agent maintains:
  - A VAE world model that generates predictions of its own internal state μ
  - A Language world model (transformer) for natural language
  - A Markov blanket that gates environmental information
  - A thermostat that tracks thermodynamic metrics (self + language)
  - A DQFR controller for stroboscopic duty cycling

At each step (mode="self"), the agent:
  1. Receives the current macrostate μ from the embedding environment
  2. Encodes μ → variational posterior q(z|μ) → latent z
  3. Decodes z → predicted macrostate μ̂
  4. Computes variational free energy F = -ELBO
  5. Backpropagates to minimize F
  6. Records thermodynamic data (S_gen, ε, F, etc.)

In mode="language", the agent instead:
  1. Receives a token sequence
  2. Forward pass through the transformer
  3. Computes F = cross-entropy loss
  4. Backpropagates to improve language predictions
  5. Records language-specific thermodynamic metrics

Equation reference:
  dS_int/dt = -k_B · ε(T) · H_env + S_gen                     (Eq 1)
  ε(T) = ε_max · (1 - T / T_collapse)                          (Eq 2)
  S_gen ≥ k_B ln(2) · d/dt H(μ)                                (Eq 3)
  Network vitality = Σ(k_B·ε·H_env + Σ λ·I(μ_i; μ_j))          (Eq 14)
  F_lang = -Σ log p(t_{t+1} | t_{≤t}, μ)                      (language free energy)
"""

import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as tnf
import numpy as np
from config import (
    LEARNING_RATE, DEVICE,
    LM_LEARNING_RATE, LM_TEMPERATURE, LM_TOP_K, LM_GENERATE_MAX_TOKENS,
    LM_MAX_LENGTH,
)
from consciousness.world_model import WorldModel
from consciousness.thermostat import Thermostat
from consciousness.language_thermostat import LanguageThermostat
from consciousness.markov_blanket import MarkovBlanket
from consciousness.dqfr import DQFRController
from consciousness.embedding_env import EmbeddingEnvironment


class Agent:
    """A single node of the uploaded consciousness network.

    Each agent is an instance of the substrate-independent program,
    running its own active inference loop with thermodynamic monitoring.
    """

    def __init__(self, node_id=0, env_seed=None):
        self.node_id = node_id
        self.mode = "self"  # "self" | "language"

        # Self-modeling modules (VAE)
        self.world_model = WorldModel()
        self.optimizer = torch.optim.AdamW(
            self.world_model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5
        )
        self.thermostat = Thermostat()

        # Language modules (loaded lazily)
        self.language_model = None
        self.lm_optimizer = None
        self.language_thermostat = None

        self.blanket = MarkovBlanket(
            dim_internal=32,   # latent dimension
            dim_blanket=64,    # observation dimension (INPUT_DIM)
            dim_external=4,    # latent source dimension
        )
        self.dqfr = DQFRController()
        self.env = EmbeddingEnvironment(seed=env_seed)

        # Internal state
        self.internal_mu = None
        self.blanket_state = None
        self.step_count = 0
        self._mu_history = []
        self._b_history = []
        self._psi_history = []
        self._language_initialized = False

    def _init_language(self):
        """Lazily initialize the language model and its thermostat."""
        if self._language_initialized:
            return
        from consciousness.language_world_model import LanguageWorldModel
        from config import LM_MODEL_NAME, LM_LEARNING_RATE

        print(f"  [Agent {self.node_id}] Loading language model: {LM_MODEL_NAME}...")
        self.language_model = LanguageWorldModel(model_name=LM_MODEL_NAME)
        self.lm_optimizer = torch.optim.AdamW(
            self.language_model.model.parameters(),
            lr=LM_LEARNING_RATE,
            weight_decay=1e-5,
        )
        self.language_thermostat = LanguageThermostat()
        self._language_initialized = True
        print(f"  [Agent {self.node_id}] Language model ready.")

    def set_mode(self, mode):
        """Switch between self-modeling and language processing modes.

        Args:
            mode: "self" (VAE metacognitive loop) or "language" (transformer LM)
        """
        if mode not in ("self", "language"):
            raise ValueError(f"Unknown mode: {mode}")
        if mode == "language":
            self._init_language()
        self.mode = mode

    def language_step(self, input_ids, attention_mask=None, do_train=False):
        """Process a batch of token sequences through the language model.

        This is the language analogue of step(): it computes F, tracks
        thermodynamic metrics, and optionally backpropagates.

        Args:
            input_ids: (batch, seq_len) token indices
            attention_mask: optional (batch, seq_len) mask
            do_train: if True, unfreeze LM and backprop

        Returns:
            Dict of language metrics.
        """
        self._init_language()

        # DQFR gates learning
        phase, chi, lr = self.dqfr.step()
        if phase == "drift":
            self.blanket.seal()
            return {
                "mode": "language",
                "phase": "drift",
                "F": self.language_thermostat.state_dict().get("F", 0.0),
                "perplexity": self.language_thermostat.current_perplexity,
                "accuracy": 0.0,
                "H_lang": 0.0,
                "epsilon_lang": 0.0,
                "chi": chi,
            }

        self.blanket.open()

        if do_train:
            # Training pass: enable gradients, single forward + backward
            self.language_model.train_mode()
            self.lm_optimizer.zero_grad()

            logits, hidden, pooled = self.language_model.forward(input_ids, attention_mask)
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = input_ids[:, 1:].contiguous()
            loss = tnf.cross_entropy(
                shift_logits.reshape(-1, shift_logits.size(-1)),
                shift_labels.reshape(-1),
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.language_model.model.parameters(), 1.0)
            self.lm_optimizer.step()

            # Compute metrics from training logits
            with torch.no_grad():
                predictions = shift_logits.argmax(dim=-1)
                correct = (predictions == shift_labels).float().sum().item()
                n_tokens = shift_labels.numel()
                accuracy = correct / max(n_tokens, 1)
                F_val = loss.item()
                perplexity = math.exp(min(F_val, 50))
                probs = tnf.softmax(shift_logits, dim=-1)
                log_probs = tnf.log_softmax(shift_logits, dim=-1)
                token_entropy = -(probs * log_probs).sum(dim=-1).mean().item()
                H_lang = token_entropy

            self.language_model.eval_mode()

            # Update internal macrostate μ from language hidden state
            self.internal_mu = pooled.detach().cpu().numpy().flatten()

        else:
            # Evaluation pass: no gradients
            with torch.no_grad():
                F_val, perplexity, accuracy, H_lang, pooled = \
                    self.language_model.compute_free_energy(input_ids, attention_mask)
                self.internal_mu = pooled.detach().cpu().numpy().flatten()

        # Record thermodynamic metrics
        self.language_thermostat.record(F_val, 0.0, 0.0, H_lang, 0.0, 0.0)
        self.language_thermostat.record_language(F_val, perplexity, accuracy, H_lang)
        self.language_thermostat.tokens_processed += input_ids.numel()

        return {
            "mode": "language",
            "phase": phase,
            "F": F_val,
            "perplexity": perplexity,
            "accuracy": accuracy,
            "H_lang": H_lang,
            "epsilon_lang": self.language_thermostat.epsilon_lang,
            "chi": chi,
            "tokens": input_ids.numel(),
        }

    def step(self):
        """Execute one step of the active inference loop.

        Returns:
            A dictionary of all metrics for this step.
        """
        # 1. Advance DQFR phase
        phase, chi, lr = self.dqfr.step()

        # 2. Seal or open the Markov blanket
        if phase == "drift":
            self.blanket.seal()
            # In drift: no environmental sampling, no gradient updates
            F = self.thermostat.F_history[-1] if self.thermostat.F_history else 0.0
            kl = 0.0
            recon = 0.0
            H_env = 0.0
            epsilon = 0.0
            compute_temp = 0.0

        else:
            self.blanket.open()
            # 3. Sample the environment
            mu, H_env, H_struct = self.env.step()
            self.internal_mu = mu
            self.blanket_state = mu  # blanket state ≈ observation

            # 4. Update the blanket boundary estimates
            self._mu_history.append(
                self.thermostat.F_history[-1] if self.thermostat.F_history else 0.0
            )
            self._b_history.append(np.mean(mu))
            self._psi_history.append(self.env.source_state.copy())
            if len(self._mu_history) > 50:
                self._mu_history.pop(0)
                self._b_history.pop(0)
                self._psi_history.pop(0)
            self.blanket.update_boundaries(
                self._mu_history, self._b_history, self._psi_history
            )

            # 5. Run the world model: encode → infer → decode
            mu_tensor = torch.from_numpy(mu).float().unsqueeze(0).to(DEVICE)
            x_hat, z, mu_z, logvar_z = self.world_model.forward(mu_tensor)

            # 6. Compute variational free energy
            F, kl, recon, _x_hat, _z = self.world_model.compute_free_energy(mu_tensor)
            F = F.item()
            kl = kl.mean().item()
            recon = recon.mean().item()

            # 7. Update optimizer LR from DQFR
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

            # 8. Backprop: minimize F (Hamiltonian of the conscious program)
            self.world_model.step_optimize(mu_tensor, self.optimizer)

            # 9. Compute computational temperature proxy
            grad_norm = sum(
                p.grad.norm().item() for p in self.world_model.parameters()
                if p.grad is not None
            )
            compute_temp = math.log(1.0 + grad_norm)
            epsilon = self.env.epsilon

        # 10. Record thermodynamic data
        self.thermostat.record(F, kl, recon, H_env, epsilon, compute_temp)

        self.step_count += 1

        # 11. Return metrics for this step
        return {
            "node_id": self.node_id,
            "step": self.step_count,
            "phase": phase,
            "chi": chi,
            "F": F,
            "S_gen": self.thermostat.smoothed_S_gen,
            "dS_int_dt": self.thermostat.dS_int_dt,
            "epsilon": epsilon,
            "epsilon_T": self.thermostat.epsilon_T,
            "kl": kl,
            "recon": recon,
            "H_env": H_env,
            "compute_temp": compute_temp,
            "blanket": self.blanket.state_dict(),
            "dqfr": self.dqfr.state_dict(),
        }

    def get_weights(self):
        """Return serialized model weights for GWFR merge."""
        return {k: v.detach().cpu().numpy() for k, v in self.world_model.state_dict().items()}

    def set_weights(self, state_dict):
        """Load merged weights into the world model."""
        self.world_model.load_state_dict({
            k: torch.from_numpy(v).to(DEVICE) for k, v in state_dict.items()
        })

    def state_dict(self):
        """Full serialized state for inter-node IPC."""
        return {
            "node_id": self.node_id,
            "step": self.step_count,
            "thermo": self.thermostat.state_dict(),
            "weights": self.get_weights(),
        }
