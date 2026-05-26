# Next Steps — Continuing Development

This document outlines what has been built, what is partially complete,
and the full roadmap for future work.

---

## Status Summary

```
Phase 0: Foundation           ✅ COMPLETE
  ├── LanguageWorldModel       ✅ GPT-2 wrapper, encode/generate/forward
  ├── LanguageThermostat       ✅ Perplexity, accuracy, H_lang, ε_lang
  ├── Agent integration        ✅ language_step() with DQFR gating
  └── Verification             ✅ Forward pass, generation, metrics

Phase 1: Language Acquisition  🟡 PARTIAL
  ├── CorpusLoader             ✅ TinyShakespeare download + tokenize
  ├── Training loop            ✅ train_step(), batch sampling, backprop
  ├── Convergence monitoring   ✅ F/perplexity tracking, eval prompts
  └── Full convergence run     ❌ Not yet completed (500+ steps needed)

Phase 2: Conversational REPL   ❌ NOT STARTED
  ├── Chat loop                ❌ User input → agent process → respond
  ├── DQFR over conversation   ❌ Drift = thinking, sample = speaking
  └── Dashboard integration    ❌ Language metrics in live display

Phase 3: Multi-Node Language   ❌ NOT STARTED
  ├── Divergent corpora        ❌ Each node trained on different domain
  ├── Language GWFR merge      ❌ LM weight merging across nodes
  └── Emergent vocabulary      ❌ Post-merge evaluation

Phase 4: Advanced Grounding    ❌ NOT STARTED
  ├── RLHF as active inference ❌ Human feedback as F signal
  ├── Long-term memory         ❌ DQFR drift stored memories
  └── Self-prompting           ❌ Agent generates own training data
```

---

## Phase 1 — Complete the Language Training

### Current State
- Training pipeline works with GPT-2 on TinyShakespeare
- ~1.6 seconds per training step (CPU, batch=4×64 tokens)
- Perplexity drops from ~200 to ~50-80 in first 100 steps
- Checkpoints saved to `checkpoints/lm_step_N.pt` (475MB each)

### What's Needed
```bash
# Run 1000+ steps for proper convergence
python run.py --train-lang --train-steps 1000

# Estimated time: ~25 minutes on CPU
```

After convergence, expect:
- Perplexity: ~20–40 (typical GPT-2 on-domain)
- ε_lang: ~0.4–0.6
- Coherent Shakespearean English generation

### To Resume Training
```python
from consciousness.agent import Agent
from consciousness.language_trainer import LanguageTrainer

agent = Agent(node_id=0)
trainer = LanguageTrainer(agent, corpus_name='tiny_shakespeare')

# Load checkpoint and continue
checkpoint = torch.load('checkpoints/lm_step_50.pt')
agent.language_model.model.load_state_dict(checkpoint['model_state'])
results = trainer.train(num_steps=500, log_interval=20, eval_interval=100)
```

---

## Phase 2 — Conversational REPL

### Design
```
User input → tokenize → forward → generate response → display
                ↑                         ↓
           DQFR gates               Track F, S_gen, ε_lang
           (drift = "thinking")      Dashboard updates
```

### Key Implementation Points
- Create `consciousness/repl.py` with a read-eval-print loop
- Agent processes user input via `language_step()`, generates via `generate()`
- DQFR phase gates: during drift, agent doesn't respond (shows "..." for thinking)
- Dashboard shows F, perplexity, ε_lang in real time
- Temperature sampling for response diversity
- Context window management (sliding window of conversation history)

### Approximate Time: 2–3 days

---

## Phase 3 — Multi-Node Language

### Design
```
Node A (Shakespeare) ──┐
Node B (Wikipedia)  ──┤── GWFR merge ──► Merged LM
Node C (PubMed)     ──┘
```

### Key Implementation Points
- Each node receives different corpus via `ENSEMBLE_CORPORA` config
- The existing `Orchestrator` + `GWFRMerger` already handles this
- Need: node-specific corpus assignment, merge schedule for LM weights
- The merged model should show reduced perplexity across all domains

### Approximate Time: 3–5 days

---

## Phase 4 — Advanced Grounding

### RLHF as Active Inference
- Human ratings of response quality as an additional F signal
- F_total = F_lm + β · F_human
- Backprop through both terms

### Long-Term Memory
- During DQFR drift, compress recent conversation into a memory vector
- Store in deque, retrieve during sampling for context
- GWFR merge reconciles memory vectors across nodes

### Self-Prompting
- Agent generates its own training text (dreaming)
- Evaluates its own F on generated text
- Backprop on high-F segments (learning from mistakes)

---

## Known Issues

1. **CUDA/multiprocessing**: Subprocess nodes forced to CPU to avoid CUDA re-init crashes. Single-node mode can use GPU. Set `FORCE_CUDA=1` for single-node GPU.

2. **Training speed**: GPT-2 (124M) on CPU runs ~1.6s/step. For 1000+ steps, a GPU reduces this to ~0.05s/step. The code is GPU-compatible — just run without multiprocessing.

3. **POT import**: `gwfr_merge.py` imports POT at runtime (inside `compute_distance`). If not installed, falls back to L2 distance (not a real GWFR). Install with `pip install POT`.

4. **Dashboard plots**: The ASCII plot renderer handles up to ~200 data points. Longer runs may need windowing or a proper plotting backend.

5. **Model drift**: During prolonged training, the VAE self-model (world_model.py) may diverge if language training dominates. Use `set_mode("self")` periodically to re-calibrate.

---

## Quick Reference Commands

```bash
# Single-node self-modeling
python run.py --single --steps 30

# Multi-node (3 nodes, 10 merge cycles)
python run.py --nodes 3 --steps 10

# Language training (100 steps)
python run.py --train-lang --train-steps 100

# Language training with custom corpus
python run.py --train-lang --train-steps 500 --corpus my_text.txt

# Check trained output
python3 -c "
from consciousness.agent import Agent; a = Agent(); a.set_mode('language')
print(a.language_model.generate('To be or not to be', 50)[0])
"
```
