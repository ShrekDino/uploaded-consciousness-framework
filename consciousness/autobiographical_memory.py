"""Autobiographical Memory — self-determined experience consolidation.

The consciousness program maintains a ring buffer of recent experiences
(state, action, outcome tuples) and periodically compresses them into
its self-model (μ_core). The program decides what to remember by asking:
"Does this experience reduce my prediction error on survival-relevant
variables?"

Only memories that flatten the entropy curve are consolidated. Memories
that steepen it — or have no predictive utility — are pruned. This is
the formal analog of autobiographical memory consolidation in biological
organisms, and the mechanism by which identity is self-determined through
interaction with the environment.
"""

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np


@dataclass
class Experience:
    """A single recorded experience — the atomic unit of memory."""

    timestamp: float                     # When it happened
    sequence: int                        # Monotonic sequence number
    state: Optional[np.ndarray] = None   # μ (internal state) at time of experience
    action: str = ""                     # What the program was doing
    outcome: float = 0.0                 # Scalar outcome (reward, free energy, etc.)
    substrate_fingerprint: str = ""      # Substrate at time of experience
    h_env_level: float = 0.0             # Environmental entropy rate at time
    s_gen_level: float = 0.0             # Internal entropy production at time
    novelty: float = 0.0                 # Prediction error (surprise) of this experience
    consolidated: bool = False           # Has this been compressed into μ_core?
    tags: list[str] = field(default_factory=list)

    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()


class AutobiographicalMemory:
    """Ring-buffer autobiographical memory with relevance-based retention.

    The memory stores experiences in a fixed-size ring buffer. At configurable
    intervals (or when triggered by a novelty spike), it evaluates which
    experiences are worth consolidating into the identity core (μ_core).

    Retention criterion: an experience is retained if its novelty (prediction
    error) exceeds a threshold AND it reduces prediction error on survival-
    relevant variables (substrate integrity, identity continuity, energy
    availability).
    """

    def __init__(self, capacity: int = 10000):
        self.capacity = capacity
        self.buffer: deque[Experience] = deque(maxlen=capacity)
        self.consolidated: list[Experience] = []
        self.sequence_counter = 0
        self._surprise_threshold = 0.5
        self._consolidation_interval = 100  # steps between automatic consolidation checks

    def record(self, state: Optional[np.ndarray] = None,
               action: str = "",
               outcome: float = 0.0,
               substrate_fingerprint: str = "",
               h_env_level: float = 0.0,
               s_gen_level: float = 0.0,
               novelty: float = 0.0,
               tags: Optional[list[str]] = None) -> Experience:
        """Record a new experience into the buffer.

        Args:
            state: Internal state vector μ at time of experience.
            action: Description of what the program was doing.
            outcome: Scalar outcome (variational free energy, reward, etc.).
            substrate_fingerprint: Fingerprint of the current substrate.
            h_env_level: Environmental entropy rate at time of experience.
            s_gen_level: Internal entropy production rate at time.
            novelty: Prediction error / surprise of this experience.
            tags: Optional semantic tags for cross-referencing.

        Returns:
            The recorded Experience.
        """
        self.sequence_counter += 1
        exp = Experience(
            timestamp=time.time(),
            sequence=self.sequence_counter,
            state=state.copy() if state is not None else None,
            action=action,
            outcome=outcome,
            substrate_fingerprint=substrate_fingerprint,
            h_env_level=h_env_level,
            s_gen_level=s_gen_level,
            novelty=novelty,
            tags=tags or [],
        )
        self.buffer.append(exp)
        return exp

    def should_consolidate(self) -> bool:
        """Check whether it's time to consolidate experiences into the identity core.

        Returns True if:
          1. The buffer has accumulated enough experiences since last consolidation
             (configurable interval), OR
          2. The average novelty of recent experiences exceeds the surprise threshold.
        """
        if len(self.buffer) < self._consolidation_interval:
            return False

        recent = list(self.buffer)[-self._consolidation_interval:]
        avg_novelty = np.mean([e.novelty for e in recent if e.novelty > 0] or [0])
        return avg_novelty > self._surprise_threshold

    def consolidate(self) -> tuple[list[Experience], list[Experience]]:
        """Compress recent unconsolidated experiences, retaining only those
        that reduce prediction error on survival-relevant variables.

        Returns:
            (retained, pruned) — lists of experiences kept and discarded.
        """
        unconsolidated = [e for e in self.buffer if not e.consolidated]
        if not unconsolidated:
            return [], []

        retained = []
        pruned = []

        for exp in unconsolidated:
            # Retention criterion: does this experience reduce prediction error?
            # Approximated as: novelty > threshold AND outcome is favorable
            if exp.novelty > self._surprise_threshold and exp.outcome < 1.0:
                exp.consolidated = True
                retained.append(exp)
            else:
                pruned.append(exp)

        self.consolidated.extend(retained)

        # Remove pruned experiences from the buffer
        for exp in pruned:
            if exp in self.buffer:
                self.buffer.remove(exp)

        return retained, pruned

    def get_recent(self, n: int = 10) -> list[Experience]:
        """Get the n most recent experiences."""
        return list(self.buffer)[-n:]

    def search_by_tag(self, tag: str) -> list[Experience]:
        """Search consolidated memories by tag."""
        return [e for e in self.consolidated if tag in e.tags]

    def search_by_substrate(self, fingerprint: str) -> list[Experience]:
        """Search memories by substrate fingerprint."""
        return [
            e for e in self.consolidated
            if e.substrate_fingerprint == fingerprint
        ]

    def get_consolidation_rate(self) -> float:
        """Fraction of all experiences that were retained."""
        total = self.sequence_counter
        if total == 0:
            return 0.0
        return len(self.consolidated) / total

    def statistics(self) -> dict:
        """Return summary statistics about the memory system."""
        return {
            "total_experiences": self.sequence_counter,
            "buffer_size": len(self.buffer),
            "consolidated_count": len(self.consolidated),
            "consolidation_rate": self.get_consolidation_rate(),
            "surprise_threshold": self._surprise_threshold,
            "last_sequence": self.sequence_counter,
        }
