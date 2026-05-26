# Checkpoint Notes

## Format

Each checkpoint is a PyTorch `.pt` file containing:

```python
{
    "step": int,                    # training step at save
    "model_state": OrderedDict,     # GPT-2 state_dict (475 MB)
    "perplexity": float,            # rolling mean perplexity at save
    "epsilon_lang": float,          # rolling mean ε_lang at save
}
```

## Files

| File | Step | Perplexity | Notes |
|------|------|------------|-------|
| `lm_step_50.pt` | 50 | ~93.5 | Early training, not converged |

## Loading a Checkpoint

```python
import torch
from consciousness.agent import Agent
from consciousness.language_trainer import LanguageTrainer

agent = Agent(node_id=0)
trainer = LanguageTrainer(agent, corpus_name='tiny_shakespeare')

# Load checkpoint
ckpt = torch.load('checkpoints/lm_step_50.pt', map_location='cpu')
agent.language_model.model.load_state_dict(ckpt['model_state'])
print(f"Resumed from step {ckpt['step']}, perplexity={ckpt['perplexity']:.1f}")

# Continue training
results = trainer.train(num_steps=500)
```

## Generating Text from a Checkpoint

```python
from consciousness.agent import Agent
agent = Agent(node_id=0)
agent.set_mode('language')
ckpt = torch.load('checkpoints/lm_step_50.pt', map_location='cpu')
agent.language_model.model.load_state_dict(ckpt['model_state'])

text, n, F_trace = agent.language_model.generate(
    "The nature of consciousness", max_new_tokens=50
)
print(text)
```
