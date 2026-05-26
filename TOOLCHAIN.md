# Collaborative Development Toolchain

> *Infrastructure for democratized AGI research — local-first, zero-meter, immediately available.*

---

## Rationale

The rate at which the Consciousness Simulation Diagnostic Framework can enumerate and eliminate failure modes is bounded by the iteration cycle: the time between forming a hypothesis about a failure mode and running the experiment that tests it. The toolchain documented here exists to collapse that cycle to its theoretical minimum by eliminating every possible barrier — cost, connectivity, platform lock-in, and expertise — between the researcher and the experiment.

---

## Primary Stack: OpenMonoAgent.ai

**Repository:** [github.com/ShrekDino/OpenMonoAgent.ai](https://github.com/ShrekDino/OpenMonoAgent.ai)

| Property | Value |
|----------|-------|
| Architecture | Terminal-native AI coding agent |
| Runtime | C# / .NET |
| Licensing | 100% open source |
| Cost | Zero — no metered tokens, no API fees, no subscriptions |
| Deployment | Single-command install |

OpenMonoAgent serves as the primary collaborative development environment for the CSDF. Its local-first architecture ensures that every contributor — regardless of institutional affiliation, funding status, or geographic location — has identical access to the full toolchain.

### Integration with StartupHakk

The StartupHakk framework provides structured collaboration protocols for distributed research teams operating on OpenMonoAgent. This enables:

- **Synchronous pair debugging** of CSDF failure mode simulations
- **Asynchronous contribution** via shared prompt templates and agent configurations
- **Benchmark standardization** across heterogeneous hardware environments

---

## Fallback Stack: Ollama + Open Code

**Repository:** [github.com/ShrekDino/llama.cpp](https://github.com/ShrekDino/llama.cpp) (via Ollama)

| Component | Role |
|-----------|------|
| [Ollama](https://ollama.ai) | Local LLM inference server (llama.cpp backend) |
| [Open Code](https://github.com/sst/open-code) | Terminal-native coding agent (alternative to OpenMonoAgent) |

This stack exists to ensure that there is **no scenario** in which a contributor lacks access to a capable coding agent. If OpenMonoAgent is unavailable for any platform or dependency reason, Ollama + Open Code provides functional equivalence.

The Ollama benchmarking suite at [github.com/ShrekDino/ollama-bench](https://github.com/ShrekDino/ollama-bench) provides real-time performance telemetry to help contributors select the optimal local model for their hardware.

---

## Hardware Independence

The CSDF simulation engine is designed to run on any hardware that supports PyTorch:

| Hardware | Performance Profile | Suitable For |
|----------|-------------------|--------------|
| CPU-only (any) | ~1.6s/training step | VAE self-modeling, failure mode diagnostics |
| Consumer GPU (RTX 3060+) | ~0.05s/training step | Full training runs, multi-node simulations |
| Apple Silicon (M1+) | ~0.5s/training step | Development, debugging, rapid iteration |

No cloud credits. No GPU cluster access. No API keys. The framework runs on whatever hardware you have.

---

## Contribution Protocol

1. **Fork the CSDF repository** — [github.com/ShrekDino/uploaded-consciousness-framework](https://github.com/ShrekDino/uploaded-consciousness-framework)
2. **Set up your toolchain** — OpenMonoAgent or Ollama + Open Code
3. **Select a failure mode** from `FAILURE_MODE_REGISTRY.md`
4. **Run the diagnostic** — `python scripts/run.py --single --steps 50`
5. **Document your observation** — Add to the registry's observation log
6. **Submit a pull request** — Include your configuration, metrics, and analysis

---

## Epistemic Commitment

> *The barrier between having an idea and running an experiment must be zero.*

When the toolchain is local, free, and immediately available, the iteration cycle collapses from weeks to hours. The systematic enumeration and elimination of failure modes in synthetic consciousness architectures becomes a parallel, community-driven research program rather than a privileged pursuit of well-funded laboratories.

This is the explicit engineering rationale for every tool choice documented here.
