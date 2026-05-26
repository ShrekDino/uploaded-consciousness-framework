"""Language-specific thermodynamic monitoring.

Extends the base Thermostat with language-centric metrics:
  F_lang = cross-entropy loss (nats per token)
  Perplexity = exp(F_lang)
  ε_lang = token_accuracy / H_lang  (language extraction efficiency)
  H_lang = token entropy rate
  S_gen_lang = |F_t - F_{t-1}|  (entropy production from language processing)

Equation reference:
  F = -Σ log p(t_{t+1} | t_{≤t}, μ)    (language variational free energy)
  ε_lang = accuracy / H_lang            (Section II, extended to language)
  S_gen ≥ k_B ln(2) · d/dt H(μ)         (Generalized Landauer, Eq 3)
"""

from collections import deque

from consciousness.thermostat import Thermostat


class LanguageThermostat(Thermostat):
    """Thermodynamic tracker specialized for language acquisition.

    Extends the base Thermostat with language-specific rolling windows
    and metrics that parallel the self-modeling thermodynamic quantities.
    """

    def __init__(self, max_history=500):
        super().__init__(max_history=max_history)

        # Language-specific histories
        self.perplexity_history = deque(maxlen=max_history)
        self.token_accuracy_history = deque(maxlen=max_history)
        self.H_lang_history = deque(maxlen=max_history)
        self.epsilon_lang_history = deque(maxlen=max_history)
        self.tokens_processed = 0
        self.vocabulary_seen = set()

    def record_language(self, F, perplexity, accuracy, H_lang):
        """Record one language processing step.

        Args:
            F: variational free energy (cross-entropy, nats/token)
            perplexity: exp(F)
            accuracy: next-token prediction accuracy [0, 1]
            H_lang: token entropy rate (nats/token)
        """
        self.perplexity_history.append(perplexity)
        self.token_accuracy_history.append(accuracy)
        self.H_lang_history.append(H_lang)
        self.epsilon_lang_history.append(accuracy / max(H_lang, 1e-12))

        # A single "step" in language processing corresponds to one
        # forward pass over a batch of tokens. The number of tokens
        # processed is batch_size * seq_length. We track tokens here
        # separately from the agent's step count.

    @property
    def epsilon_lang(self):
        """Current language extraction efficiency ε_lang.

        High ε_lang means the model is accurately predicting tokens
        with low uncertainty — it has extracted the structured
        information from the language stream.
        """
        if not self.epsilon_lang_history:
            return 0.0
        return self.epsilon_lang_history[-1]

    @property
    def current_perplexity(self):
        if not self.perplexity_history:
            return float('inf')
        return self.perplexity_history[-1]

    def language_state_dict(self):
        """Extended state dict including language metrics."""
        base = self.state_dict()
        base.update({
            "perplexity": self.current_perplexity,
            "token_accuracy": self.token_accuracy_history[-1] if self.token_accuracy_history else 0.0,
            "H_lang": self.H_lang_history[-1] if self.H_lang_history else 0.0,
            "epsilon_lang": self.epsilon_lang,
            "tokens_processed": self.tokens_processed,
            "vocab_size": len(self.vocabulary_seen),
            "perplexity_history": list(self.perplexity_history),
            "epsilon_lang_history": list(self.epsilon_lang_history),
        })
        return base
