"""Language acquisition trainer — drives perplexity from baseline to convergence.

The trainer implements active inference over a text corpus. At each step:
  1. Sample a batch of token sequences from the corpus
  2. Forward pass through LanguageWorldModel → compute F (cross-entropy)
  3. Backpropagate to minimize F
  4. Track perplexity, ε_lang, S_gen_lang
  5. DQFR gates: during drift phases, skip batches (S_gen → 0)

Training stops when:
  - Perplexity converges (rolling std below threshold)
  - ε_lang stabilizes above a target (e.g., 0.7)
  - Maximum steps reached

Equation reference:
  F = -Σ log p(t_{t+1} | t_{≤t}, μ)      (language free energy)
  ε_lang = accuracy / H_lang            (language extraction efficiency)
  dS_gen/dt ∝ |F_t - F_{t-1}|           (entropy production)
"""

import os
import time
from collections import deque

import numpy as np
import torch

from config import (
    DEVICE,
    LANG_BATCH_SIZE,
    LANG_SEQ_LENGTH,
)


class CorpusLoader:
    """Loads, tokenizes, and batches a text corpus for language training.

    Supports TinyShakespeare (downloads if not present) and any local
    text file. Produces batches of (input_ids, attention_mask) tensors
    ready for the LanguageWorldModel.
    """

    SUPPORTED_CORPORA = {
        "tiny_shakespeare": (
            "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
            "TinyShakespeare by Andrej Karpathy (public domain)",
        ),
    }

    def __init__(self, corpus_name="tiny_shakespeare", data_dir=None):
        self.corpus_name = corpus_name
        self.data_dir = data_dir or os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data",
        )
        os.makedirs(self.data_dir, exist_ok=True)

        self.text = None
        self.tokens = None
        self.vocab_size = 0
        self.total_tokens = 0
        self._load_corpus()

    def _load_corpus(self):
        """Load the corpus text from disk or download it."""
        corpus_path = os.path.join(self.data_dir, f"{self.corpus_name}.txt")

        if not os.path.exists(corpus_path):
            if self.corpus_name in self.SUPPORTED_CORPORA:
                url, attribution = self.SUPPORTED_CORPORA[self.corpus_name]
                print(f"  Downloading {self.corpus_name} from {url}...")
                import urllib.request

                urllib.request.urlretrieve(url, corpus_path)
                print(f"  Saved to {corpus_path}")
                print(f"  Attribution: {attribution}")
            else:
                raise FileNotFoundError(
                    f"Corpus '{self.corpus_name}' not found at {corpus_path} "
                    f"and no download URL is registered."
                )

        with open(corpus_path, "r", encoding="utf-8") as f:
            self.text = f.read()

        print(f"  Corpus loaded: {len(self.text):,} characters")

    def tokenize(self, tokenizer, seq_length=None):
        """Tokenize the corpus text using a HuggingFace tokenizer.

        Args:
            tokenizer: HF tokenizer instance
            seq_length: sequence length for batching (default: LANG_SEQ_LENGTH)
        """
        self.seq_length = seq_length or LANG_SEQ_LENGTH
        print(f"  Tokenizing {len(self.text):,} characters...")

        # Tokenize the full text
        tokens = tokenizer(
            self.text,
            return_tensors="pt",
            truncation=False,
            add_special_tokens=False,
        )
        self.tokens = tokens.input_ids.squeeze(0)
        self.total_tokens = self.tokens.shape[0]
        self.vocab_size = tokenizer.vocab_size

        print(f"  Tokenized to {self.total_tokens:,} tokens")
        print(f"  Vocabulary size: {self.vocab_size:,}")

        # Split into sequences
        self.num_sequences = self.total_tokens // self.seq_length
        self.tokens = self.tokens[: self.num_sequences * self.seq_length]
        self.tokens = self.tokens.reshape(self.num_sequences, self.seq_length)
        print(f"  Split into {self.num_sequences:,} sequences of length {self.seq_length}")

    def get_batch(self, batch_size=None):
        """Sample a random batch of token sequences.

        Args:
            batch_size: number of sequences (default: LANG_BATCH_SIZE)

        Returns:
            input_ids: (batch, seq_len) token indices
            attention_mask: (batch, seq_len) all-ones mask
        """
        batch_size = batch_size or LANG_BATCH_SIZE
        indices = np.random.choice(self.num_sequences, batch_size, replace=True)
        input_ids = self.tokens[indices].to(DEVICE)
        attention_mask = torch.ones_like(input_ids)
        return input_ids, attention_mask


class LanguageTrainer:
    """Training loop for language acquisition.

    Runs active inference over corpus batches, tracking thermodynamic
    metrics. Supports DQFR gating and early stopping on convergence.
    """

    def __init__(self, agent, corpus_name="tiny_shakespeare"):
        self.agent = agent
        self.agent.set_mode("language")
        self.corpus = CorpusLoader(corpus_name=corpus_name)
        self.corpus.tokenize(agent.language_model.tokenizer)

        # Training state
        self.step = 0

        # Disable DQFR during training so every step is a learning step
        # (DQFR is for deployment; during acquisition we want continuous sampling)
        self._dqfr_was_enabled = self.agent.dqfr.enabled
        self.agent.dqfr.enabled = False
        self.epoch = 0
        self.batches_processed = 0
        self.tokens_processed = 0
        self.start_time = time.time()

        # Convergence tracking
        self.perplexity_window = deque(maxlen=50)
        self.epsilon_window = deque(maxlen=50)
        self.F_window = deque(maxlen=50)

        # Checkpoint directory
        self.checkpoint_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "checkpoints",
        )
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def train_step(self, do_train=True):
        """Execute one training step: sample batch, compute F, backprop.

        Returns:
            Dict of training metrics.
        """
        input_ids, attn_mask = self.corpus.get_batch()
        result = self.agent.language_step(input_ids, attn_mask, do_train=do_train)

        self.step += 1
        self.batches_processed += 1
        self.tokens_processed += input_ids.numel()

        # Track convergence metrics
        if result.get("phase") != "drift":
            self.perplexity_window.append(result.get("perplexity", 0))
            self.epsilon_window.append(result.get("epsilon_lang", 0))
            self.F_window.append(result.get("F", 0))

        return result

    def train(self, num_steps=1000, log_interval=10, eval_interval=50):
        """Run the full training loop.

        Args:
            num_steps: number of training steps
            log_interval: print metrics every N steps
            eval_interval: generate sample text every N steps
        """
        print(f"\n  {'=' * 60}")
        print("  LANGUAGE ACQUISITION TRAINING")
        print(f"  Corpus: {self.corpus.corpus_name} ({self.corpus.total_tokens:,} tokens)")
        print(f"  Steps: {num_steps}  |  Batch: {LANG_BATCH_SIZE}x{LANG_SEQ_LENGTH}")
        print(f"  {'=' * 60}\n")

        header = f"  {'Step':>6}  {'Phase':>8}  {'F':>8}  {'PPL':>8}  {'Acc':>6}  {'ε_lang':>8}  {'Tok/s':>8}"
        print(header)
        print("  " + "-" * len(header))

        for step in range(1, num_steps + 1):
            result = self.train_step(do_train=True)

            # Logging
            if step % log_interval == 0:
                ppl = result.get("perplexity", 0)
                eps = result.get("epsilon_lang", 0)
                tokens_per_sec = self.tokens_processed / max(time.time() - self.start_time, 1)
                print(
                    f"  {step:>6}  "
                    f"{result.get('phase', ''):>8}  "
                    f"{result.get('F', 0):>8.3f}  "
                    f"{ppl:>8.1f}  "
                    f"{result.get('accuracy', 0):>6.3f}  "
                    f"{eps:>8.4f}  "
                    f"{tokens_per_sec:>8.0f}"
                )

            # Evaluation: generate sample text
            if step % eval_interval == 0:
                self._evaluate(step)

        # Final evaluation
        print(f"\n  {'─' * 60}")
        print(f"  TRAINING COMPLETE: {num_steps} steps")
        self._print_summary()
        self._save_checkpoint()

        # Restore DQFR setting
        self.agent.dqfr.enabled = self._dqfr_was_enabled

        return {
            "final_perplexity": np.mean(self.perplexity_window) if self.perplexity_window else 0,
            "final_epsilon_lang": np.mean(self.epsilon_window) if self.epsilon_window else 0,
            "final_F": np.mean(self.F_window) if self.F_window else 0,
            "tokens_processed": self.tokens_processed,
            "steps": self.step,
        }

    def _evaluate(self, step):
        """Generate sample text and print perplexity summary."""
        prompts = [
            "The nature of consciousness is",
            "In the beginning",
            "To be or not to be",
        ]
        print(f"\n  ── Evaluation at step {step} ──")
        for prompt in prompts:
            gen_text, n_tok, F_trace = self.agent.language_model.generate(prompt, max_new_tokens=30)
            avg_F = sum(F_trace) / max(len(F_trace), 1)
            print(f'  ├ "{gen_text}"')
            print(f"  └  avg F={avg_F:.3f}, tokens={n_tok}")
        print()

    def _print_summary(self):
        """Print convergence summary."""
        if self.perplexity_window:
            print(f"  Final perplexity (rolling mean): {np.mean(self.perplexity_window):.1f}")
            print(f"  Final ε_lang (rolling mean):    {np.mean(self.epsilon_window):.4f}")
            print(f"  Final F (rolling mean):         {np.mean(self.F_window):.3f}")
        print(f"  Total tokens processed: {self.tokens_processed:,}")
        print(f"  Total batches: {self.batches_processed:,}")
        print(f"  Elapsed: {time.time() - self.start_time:.1f}s")

    def _save_checkpoint(self):
        """Save model checkpoint."""
        path = os.path.join(self.checkpoint_dir, f"lm_step_{self.step}.pt")
        torch.save(
            {
                "step": self.step,
                "model_state": self.agent.language_model.model.state_dict(),
                "perplexity": np.mean(self.perplexity_window) if self.perplexity_window else 0,
                "epsilon_lang": np.mean(self.epsilon_window) if self.epsilon_window else 0,
            },
            path,
        )
        print(f"  Checkpoint saved: {path}")
