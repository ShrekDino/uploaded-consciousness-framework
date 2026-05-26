"""Language world model — transformer-based generative model for natural language.

The language world model wraps a pretrained causal language model (GPT-2)
as a generative model p(t | μ) over token sequences. The variational free
energy F is the cross-entropy loss: the agent minimizes surprise by
predicting the next token in a sequence.

This mirrors the VAE WorldModel interface:
  forward(tokens) → logits, hidden_states
  compute_free_energy(tokens) → F, perplexity, accuracy, H_lang

The hidden state (pooled last layer) serves as the agent's internal
macrostate μ for the language modality, allowing the existing VAE to
model language representations as part of the self-model.

Equation reference:
  F = -log p(t_{t+1} | t_{≤t}, μ)      (language variational free energy)
  ε_lang = accuracy / H_lang            (language extraction efficiency)
  H_lang = -Σ p(t) log p(t)             (token entropy rate)
"""

import torch
import torch.nn as nn
import torch.nn.functional as tnf

from config import DEVICE


class LanguageWorldModel(nn.Module):
    """Wrapper around a pretrained causal language model.

    The transformer decoder serves as the generative model p(t | μ),
    where μ is the pooled hidden state after processing the sequence.
    """

    def __init__(self, model_name="gpt2"):
        super().__init__()
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_name = model_name
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
        )
        self.model.to(DEVICE)
        self.model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Freeze embeddings for efficiency (will unfreeze during training)
        for param in self.model.parameters():
            param.requires_grad = False

        self.hidden_dim = self.model.config.n_embd
        self.vocab_size = self.model.config.vocab_size
        self._device = DEVICE

    @property
    def device(self):
        return self._device

    def forward(self, input_ids, attention_mask=None):
        """Forward pass through the transformer.

        Args:
            input_ids: (batch, seq_len) token indices
            attention_mask: (batch, seq_len) optional mask

        Returns:
            logits: (batch, seq_len, vocab_size) prediction logits
            hidden_states: (batch, seq_len, hidden_dim) all hidden states
            pooled: (batch, hidden_dim) pooled last-token hidden state (μ)
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        logits = outputs.logits

        # Pooled hidden state = last non-padding token's hidden state
        hidden_states = outputs.hidden_states[-1]
        if attention_mask is not None:
            # Find the last token position for each sequence
            lengths = attention_mask.sum(dim=1) - 1
            pooled = hidden_states[torch.arange(hidden_states.size(0)), lengths]
        else:
            pooled = hidden_states[:, -1, :]

        return logits, hidden_states, pooled

    def compute_free_energy(self, input_ids, attention_mask=None):
        """Compute variational free energy F over a token sequence.

        F = cross-entropy loss = -Σ log p(t_{t+1} | t_{≤t}, μ)

        This is the language analogue of the VAE's F = -ELBO.

        Returns:
            F: scalar free energy (mean cross-entropy over tokens)
            perplexity: exp(F)
            accuracy: token prediction accuracy
            H_lang: token entropy rate of the predicted distribution
        """
        input_ids = input_ids.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        logits, hidden, pooled = self.forward(input_ids, attention_mask)

        # Shift logits and labels for next-token prediction
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        if attention_mask is not None:
            shift_mask = attention_mask[:, 1:].contiguous()
        else:
            shift_mask = torch.ones_like(shift_labels)

        # Cross-entropy loss = F
        loss_fn = nn.CrossEntropyLoss(reduction="none")
        token_losses = loss_fn(
            shift_logits.reshape(-1, self.vocab_size),
            shift_labels.reshape(-1),
        ).reshape(shift_labels.shape)

        # Mask out padding tokens
        token_losses = token_losses * shift_mask
        n_tokens = shift_mask.sum().clamp(min=1)
        F = token_losses.sum() / n_tokens

        # Perplexity
        perplexity = torch.exp(F).item()

        # Token prediction accuracy
        predictions = shift_logits.argmax(dim=-1)
        correct = (predictions == shift_labels).float() * shift_mask
        accuracy = (correct.sum() / n_tokens).item()

        # Token entropy rate H_lang
        probs = tnf.softmax(shift_logits, dim=-1)
        log_probs = tnf.log_softmax(shift_logits, dim=-1)
        token_entropy = -(probs * log_probs).sum(dim=-1)
        H_lang = (token_entropy * shift_mask).sum() / n_tokens
        H_lang = H_lang.item()

        return F.item(), perplexity, accuracy, H_lang, pooled

    def generate(self, prompt_text, max_new_tokens=50, temperature=1.0, top_k=40):
        """Generate text continuation from a prompt.

        Returns:
            generated_text: the full output including the prompt
            tokens_generated: number of tokens generated
            F_per_token: list of free energy values per generated token
        """
        inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.device)
        input_ids = inputs.input_ids
        attention_mask = inputs.attention_mask

        F_trace = []
        generated_ids = input_ids.clone()

        for _ in range(max_new_tokens):
            logits, _, _ = self.forward(generated_ids, attention_mask)
            next_logits = logits[:, -1, :] / temperature

            # Top-k filtering
            if top_k > 0:
                top_k_vals, top_k_idx = torch.topk(next_logits, top_k, dim=-1)
                next_logits = torch.full_like(next_logits, float("-inf"))
                next_logits.scatter_(-1, top_k_idx, top_k_vals)

            # Sample
            probs = tnf.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Compute F for this token
            token_F = tnf.cross_entropy(next_logits, next_token.squeeze(-1), reduction="none").item()
            F_trace.append(token_F)

            # Append token
            generated_ids = torch.cat([generated_ids, next_token], dim=-1)
            attn_append = torch.ones_like(next_token)
            attention_mask = torch.cat([attention_mask, attn_append], dim=-1)

            # Early stopping at EOS
            if next_token.item() == self.tokenizer.eos_token_id:
                break

        generated_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        return generated_text, len(F_trace), F_trace

    def encode_text(self, text):
        """Encode text to token IDs + attention mask."""
        encoded = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        return encoded.input_ids.to(self.device), encoded.attention_mask.to(self.device)

    def decode_tokens(self, token_ids):
        """Decode token IDs to text."""
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)

    def train_mode(self):
        """Unfreeze model parameters for training."""
        for param in self.model.parameters():
            param.requires_grad = True
        self.model.train()

    def eval_mode(self):
        """Freeze model parameters for inference."""
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()
