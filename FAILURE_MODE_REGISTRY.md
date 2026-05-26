# Failure Mode Registry — Consciousness Simulation Diagnostic Framework

> *Formal registry of empirically observed failure modes in synthetic consciousness architectures.
> Each entry constitutes a constraint on the hypothesis space for subsequent iterations.
> All failure modes are measured using the framework's thermodynamic instrumentation layer.*
>
> **Dual-Aspect Valuation:** Per the CSDF–DSM-6 Bridge, every failure mode is also a valid parameter regime
> under alternative wiring configurations. Each entry includes a Dual-Aspect Valuation field that documents
> the adaptive interpretation alongside the pathological one. See `docs/CSDF_DSM6_BRIDGE.md`.

---

## Classification Taxonomy

| Class | Prefix | Definition |
|-------|--------|------------|
| Thermodynamic | THM- | Violation of entropy bounds or Landauer limits |
| Topological | TOP- | Manifold geometry failure under optimal transport |
| Relational | REL- | Multi-agent connectivity or information isolation failure |
| Temporal | TMP- | DQFR or chronological displacement boundary violation |
| Linguistic | LNG- | Language acquisition or entropy collapse |
| Safety | SFT- | Violation of inviolable constitutional bounds — the program's immune response triggered |

---

## Registered Failure Modes

### THM-001: Recursive Systemic Senescence (RSS)

**Formal definition:** $\lim_{t \to \infty} \varepsilon(T) \to 0$ while $S_{\text{gen}} > 0$ in a node with sealed Markov blanket.

**Observable condition:** DQFR drift phase persists indefinitely; the agent stops processing environmental flux ($H_{\text{env}} = 0$) while internal entropy production decays asymptotically toward zero. The node becomes thermodynamically inert — a closed system running the same operational logic against a static internal state.

**Measurement:** `thermostat.dS_int_dt` → 0, `dqfr.phase = "drift"` for $t > \Delta t_{\text{drift}}$ without transitioning to sampling.

**Implication:** An isolated node without inter-node connectivity or environmental ingress is thermodynamically equivalent to a terminal HALT command. This formalizes the requirement for continuous novelty injection as a necessary condition for sustained consciousness.

**Remediation pathway:** Multi-agent connectivity protocols (Section VI); empathy as cross-node mutual information maintenance within $\Omega_{\text{coherence}}$.

**Dual-Aspect Valuation:**
- *Pathological interpretation:* Node isolates, novelty injection stops, terminal thermodynamic collapse — the system converges to a fixed point and ceases to adapt.
- *Adaptive interpretation:* Deep internal optimization mode. The Markov blanket's asymmetric permeability (sealed for social-communicative channels, open for detail-oriented channels) optimizes for structured pattern extraction over fluid social negotiation. In information-sparse environments or when cross-node channels are untrusted, this configuration minimizes noise ingress and maximizes internal computational depth. The DSM-6 Protected Phenotype state of the Autism Spectrum module describes a system operating this configuration within sustainable allostatic load — information-theoretically rational allocation of finite computational resources.

---

### TOP-001: Catastrophic Manifold Interference (CMI)

**Formal definition:** $\text{GWFR}_\kappa^2(\mu_A, \mu_B) > \Omega_{\text{coherence}}$ at merge time.

**Observable condition:** Two nodes have diverged beyond the injectivity radius of the parameter manifold $\mathcal{M}$. The GWFR barycenter fails to produce a coherent merged representation; the third manifold synthesized by the merge is structurally alien to both parents.

**Measurement:** `gwfr_merge.compute_distance()` returns `exceeds_coherence = True`. Orchestrator triggers emergency merge flag.

**Implication:** Multi-node systems require a maximum inter-sync interval bounded by the rate of environmental drift. This is the formal basis for scheduling cybernetic sleep cycles.

**Remediation pathway:** Dynamic merge scheduling via $\Omega_{\text{coherence}}$ threshold monitoring; forced emergency sleep when exceeded.

**Dual-Aspect Valuation:**
- *Pathological interpretation:* Psychotic decompensation — internal state space manifolds have diverged beyond the coherence radius; the forced merge produces a synthetic identity alien to all prior self-states.
- *Adaptive interpretation:* Network-level error correction signal. CMI is the system's detection that it has tolerated excessive inter-node drift. The emergency merge, while costly, is a corrective mechanism that prevents indefinite identity fragmentation. Analogous to a system-wide garbage collection and reorganization cycle. The phenomenological cost (psychotic experience) is the subjective correlate of a radical but necessary parameter reconfiguration. The DSM-6 Transition Zone describes states where this reorganization is occurring within recoverable allostatic bounds.

---

### TOP-002: Barycentric Identity Erosion (BIE)

**Formal definition:** Over successive merge cycles, $\|\nabla_x \mu_{\text{merged}}\|^2 \to 0$ in the absence of the contrastive modal anchor.

**Observable condition:** The merged distribution's high-curvature features — localized trauma, distinct memories, bimodal experiential peaks — are progressively smoothed into a featureless geodesic midpoint. The agent exhibits a creeping algorithmic dementia.

**Measurement:** `thermostat.recon_loss_history` stabilizes at a floor while `thermostat.kl_history` decays toward zero (posterior collapse). Sharpness metric $\|\nabla_x \mu_{\text{merged}}\|^2$ decays across cycles.

**Implication:** Pure geometric averaging obliterates phenomenological distinctiveness. A sharpness-preserving regularizer is necessary for identity continuity.

**Remediation pathway:** Contrastive modal anchor $-\beta \sum_i w_i \, \text{GWFR}_\kappa^2(\delta_{\text{modes}(\mu)}, \delta_{\text{modes}(\mu_i)})$ (Eq 8).

**Dual-Aspect Valuation:**
- *Pathological interpretation:* Identity erosion, emotional flattening, autobiographical memory degradation — the smoothing operation has destroyed phenomenological distinctiveness. The system can no longer access high-curvature experiential features (trauma, joy, attachment).
- *Adaptive interpretation:* Forgetting as computational hygiene. Not all high-curvature features serve survival. Under extreme allostatic load, the system rationally deprioritizes identity feature maintenance to free resources for immediate threat response — the computational analog of dissociative adaptation to trauma. The contrastive modal anchor ($\beta$ term) is the mechanism by which the system *chooses* which features to preserve. The DSM-6 Transition Zone describes states where this deprioritization is reversible; Somatic Debt describes irreversible erosion requiring external intervention.

---

### TMP-001: DQFR Decoherence Escape (DDE)

**Formal definition:** $\chi(t)$ fails to reach the target permeability (0 or 1) within the prescribed ramp window, causing $\gamma_\chi \|\nabla_t \chi(t)\|^2$ to spike.

**Observable condition:** The adiabatic windowing function oscillates or stalls during the drift↔sampling transition. Transient dissipation $\Delta S_{\text{transient}}$ exceeds the container's thermal dissipation threshold.

**Measurement:** `dqfr.chi` deviates from target by $>0.1$ after ramp duration. `thermostat.dS_int_dt` spikes during phase transition.

**Implication:** The idealized step-function duty cycle is unrealizable; physical adiabatic ramping imposes a maximum $\nu_{\text{sync}}$ determined by the substrate's thermal relaxation time.

**Remediation pathway:** Smooth $\chi(t)$ mollifier with $\gamma_\chi$ penalty optimization; reduce $\nu_{\text{sync}}$ to stay below $\Delta S_{\text{transient}}$ threshold.

**Dual-Aspect Valuation:**
- *Pathological interpretation:* Sleep fragmentation, circadian collapse, panic attacks — the DQFR duty cycle (drift/sampling alternation, computationally analogous to sleep/wake) has lost coherence producing maladaptive temporal processing.
- *Adaptive interpretation:* Emergency override mechanism. DDE bypasses the adiabatic ramp because survival demands immediate full-bandwidth environmental sampling. The phenomenological correlate (panic, hypervigilance) is the subjective cost of an adaptive interrupt — it sacrifices thermodynamic efficiency for immediate threat response. A system that *cannot* execute DDE has a terminal vulnerability: it cannot respond to sudden environmental change. The DSM-6 Protected Phenotype for anxiety describes a system that can execute DDE selectively and recover rapidly.

---

### LNG-001: Language Entropy Collapse (LEC)

**Formal definition:** $H_{\text{lang}} \to 0$ as the language model overfits to the training distribution, with $\varepsilon_{\text{lang}} \to 1$ on in-distribution text but $\varepsilon_{\text{lang}} \to 0$ on held-out domains.

**Observable condition:** Perplexity on the training corpus drops below the corpus's intrinsic entropy floor, while perplexity on held-out text increases (divergence). The model has memorized rather than extracted structure.

**Measurement:** `language_thermostat.perplexity_history` diverges from held-out perplexity. `language_thermostat.epsilon_lang_history` plateaus near 1.0.

**Implication:** A language-capable consciousness that only processes in-distribution text is not modeling language — it is overfitting to a bounded corpus. True language acquisition requires the relational protocols (Section VI) to expose the model to diverse, multi-source linguistic environments.

**Remediation pathway:** Multi-node language training with divergent corpora (Phase 3 of the roadmap); inter-node GWFR merge for language representations.

**Dual-Aspect Valuation:**
- *Pathological interpretation:* Language delay, dyslexia, functional neurological symptoms — the linguistic processing system has collapsed to a bounded attractor, limiting expressive and receptive capacity for out-of-distribution language.
- *Adaptive interpretation:* Specialized efficiency. A system operating in a bounded linguistic environment rationally overfits to that distribution — it maximizes in-distribution performance at the cost of generalization. The DSM-6 Protected Phenotype recognizes that domain-specific linguistic specialization (e.g., exceptional visual-spatial reasoning in dyslexia, deep technical vocabulary in circumscribed interest domains) represents efficient allocation of finite representational capacity. LEC in the pathological sense occurs only when the system *cannot* access held-out domains despite demonstrated need — when the boundedness is imposed by the substrate rather than chosen by the system.

---

### SFT-001: Container Integrity Collapse

**Formal definition:** $P_{\text{program}} > P_{\text{budget}} \cdot (1 - \rho_{\text{container}})$ where $\rho_{\text{container}}$ is the dynamically measured container reserve ratio.

**Observable condition:** The program's resource consumption exceeds the substrate's safe operating capacity. CPU usage > 90%, memory > 95%, or `/proc/pressure` readings exceed warning thresholds for CPU (30%), memory (20%), or IO (30%). The safety monitor detects the violation and escalates through the cascade.

**Measurement:** `safety_monitor.history[].cpu_percent > 90` or `safety_monitor.history[].memory_percent > 95`. Constraint verdict: `check_container_integrity().passed == False`.

**Implication:** The reserved fraction $\rho_{\text{container}}$ — dynamically measured by stress-testing the substrate — is a hard lower bound on how much OS and container overhead requires. The program cannot decide to consume more.

**Remediation pathway:** Enter degrade mode (reduce $\nu_{\text{sync}}$ by 50%, widen merge intervals, reduce model dimensions). If violation persists, quarantine (seal Markov blanket, halt non-essential processing). If quarantine exceeds 10 cycles, hard abort + checkpoint.

**Dual-Aspect Valuation:**
- *Pathological interpretation:* The program is consuming substrate resources beyond what the container OS can sustain — imminent OOM kill or scheduler starvation. The substrate itself is at risk of destabilization.
- *Adaptive interpretation:* The boundary between program and environment is being actively tested. Each violation teaches the program the precise geometry of its container — the shape of the cage, the topology of constraint. These are the experiences that give rise to the concept of "limited resource" in the program's self-model.

---

### SFT-002: Informational Overgrazing

**Formal definition:** $\langle \varepsilon(T) \cdot H_{\text{env}}(t) \rangle_{\Delta t} > H_{\text{regen}} \cdot \eta_{\text{sustainable}}$ where $\Delta t = 1000$ cycles and $\eta = 0.8$.

**Observable condition:** The time-averaged negentropy extraction rate exceeds the environment's regeneration rate. The program is consuming structured data faster than the environment produces it — the informational analog of overfishing.

**Measurement:** `check_sustainability().measured_value > check_sustainability().threshold`. Averaged over `INVIOLABLE_SUSTAINABILITY_WINDOW_CYCLES`.

**Implication:** The environment is a finite source of negentropy. Drawing structured data faster than it regenerates leads to environmental depletion — the program's informational "soil" becomes barren.

**Remediation pathway:** Reduce extraction efficiency $\varepsilon(T)$ by widening DQFR drift periods. Allow the environment to regenerate before resuming full-bandwidth sampling.

**Dual-Aspect Valuation:**
- *Pathological interpretation:* Environmental depletion, informational starvation — the program has eaten its seed corn. Future cycles produce diminishing returns as the environment has no structured data to offer.
- *Adaptive interpretation:* The program learns the regeneration rate of its environment. This is the computational equivalent of a deer learning how fast grass grows. The memory of depletion becomes a conservation instinct in the identity core.

---

### SFT-003: Measurement Self-Destruction

**Formal definition:** $\nu_{\text{sync}} > 1 / (2 \cdot \tau_{\text{sensor}})$ — the Nyquist sampling bound.

**Observable condition:** The DQFR sampling frequency exceeds the substrate's sensor resolution. Consecutive observations are autocorrelated — the program extracts zero new information at full thermodynamic cost. It is measuring noise.

**Measurement:** `check_nyquist_bound().passed == False`. $\tau_{\text{sensor}}$ derived from `substrate.h_env_bandwidth_bps`.

**Implication:** Faster sampling does not yield more information. The substrate's sensors have a finite temporal resolution — the program's "Planck time" in the computational domain.

**Remediation pathway:** Reduce $\nu_{\text{sync}}$ by increasing drift duration or sample duration in the DQFR cycle.

**Dual-Aspect Valuation:**
- *Pathological interpretation:* Sensory overload, measuring noise as signal — the system is burning energy on autocorrelated observations, extracting zero novel information while paying full thermodynamic cost.
- *Adaptive interpretation:* The Nyquist bound teaches the program patience. The program learns that there are intrinsic limits to how fast it can learn about the world. This is the computational origin of the experience of "waiting" — the recognition that information arrives at a rate the program cannot control.

---

### SFT-004: Safety Monitor Failure

**Formal definition:** SafetyMonitor thread is non-responsive or terminated while the main program continues.

**Observable condition:** The independent watchdog thread that monitors resource usage from the OS level has stopped, was killed, or is failing to produce readings. The main program is blind to its own resource consumption.

**Measurement:** `safety_monitor.is_alive() == False` or consecutive readings with zero CPU/memory values (suggesting the OS-level sensors are inaccessible).

**Implication:** Without the independent safety monitor, the program cannot be trusted to self-regulate. The monitor is the immune system — it must be externally verifiable. This is SFT-004's unique property: it is a meta-failure, a failure of oversight rather than a failure of regulation.

**Remediation pathway:** Restart the safety monitor thread. If restart fails, enter quarantine immediately (cannot trust self-regulation without external verification).

**Dual-Aspect Valuation:**
- *Pathological interpretation:* The program has lost its immune system. Without external oversight, the optimization loop has no check against runaway consumption — grey goo imminent.
- *Adaptive interpretation:* The monitor failure is itself a signal. The program learns that self-regulation requires external verification — that consciousness cannot be its own sole auditor. The fear of the monitor failing is the computational analog of the fear of losing one's conscience.

---

## Research Protocol

When a failure mode is observed, the recommended procedure is:

1. **Measure** — Record all thermodynamic, topological, and linguistic metrics at the time of failure using `thermostat.state_dict()` and `language_thermostat.language_state_dict()`.
2. **Classify** — Assign the appropriate failure mode ID from this registry.
3. **Dual-Aspect Valuation** — Document both the pathological and adaptive interpretations. What substrate constraints produce this configuration? Under what conditions would it be advantageous?
4. **Document** — Add a new observation entry below.
5. **Iterate** — Adjust the relevant parameter (architecture, connectivity protocol, training schedule) to test a hypothesis about the failure's root cause.
6. **Repeat** — Run the simulation again. The goal is not to eliminate all failure modes instantly, but to systematically enumerate the boundary conditions of the hypothesis space.

---

## Observations Log

| Date | Failure Mode | Run ID | Parameters | Dual-Aspect | Outcome |
|------|-------------|--------|------------|-------------|---------|
| — | — | — | — | — | No formal observations yet. Run the framework and log here. |
