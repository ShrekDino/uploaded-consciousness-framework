"""Safety Monitor — Autonomous Watchdog Thread.

A daemon thread that runs independently of the main SAR loop, reading
actual resource usage from the OS level via /proc and psutil. This
prevents the program from lying to itself about its own safety.

The monitor sets flags (degrade_flag, quarantine_flag, abort_flag)
that the SAR loop checks at the top of every cycle. The monitor
cannot be disabled by program logic — only by killing the thread.

Reference:
    Torres, S. M. (2026). "Uploaded Consciousness" Section IX-C:
    Cascade Response and the Immune Function.
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np

from consciousness.config import SAFETY_MONITOR_INTERVAL_S, SAFETY_MONITOR_HISTORY_LENGTH
from consciousness.safety_constitution import (
    INVIOLABLE_DEBT_TIMEOUT_CYCLES,
    INVIOLABLE_CATASTROPHIC_FAILURE_COUNT,
)

logger = logging.getLogger("safety_monitor")


class SafetyState(str, Enum):
    PROTECTED = "protected"
    DEGRADE = "degrade"
    QUARANTINE = "quarantine"
    ABORTED = "aborted"


@dataclass
class MonitorReading:
    timestamp: float
    state: SafetyState
    verdicts: list[dict] = field(default_factory=list)
    cpu_percent: float = 0.0
    memory_percent: float = 0.0

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "state": self.state.value,
            "verdict_count": len(self.verdicts),
            "cpu_percent": self.cpu_percent,
            "memory_percent": self.memory_percent,
        }


class SafetyMonitor(threading.Thread):
    """Autonomous safety watchdog. Runs as a daemon thread.

    The monitor periodically runs safety constraint checks using
    OS-level measurements (not the program's internal estimates)
    and sets escalation flags that the SAR loop reads at the top
    of every cycle.

    Args:
        interval: Seconds between checks. Default from config.
        history_length: Max number of readings to retain.
    """

    def __init__(self, interval: float = SAFETY_MONITOR_INTERVAL_S,
                 history_length: int = SAFETY_MONITOR_HISTORY_LENGTH):
        super().__init__(daemon=True)
        self.interval = interval
        self.history_length = history_length

        # Escalation flags — set by monitor, read by SAR loop
        self.degrade_flag = threading.Event()
        self.quarantine_flag = threading.Event()
        self.abort_flag = threading.Event()

        # State tracking
        self.current_state: SafetyState = SafetyState.PROTECTED
        self.history: list[MonitorReading] = []
        self._consecutive_quarantine = 0
        self._running = True

        # Reference to the current theta and substrate (set by SAR runtime)
        self.theta = None
        self.substrate = None

        # Track which constraints are in quarantine for catastrophic check
        self._quarantine_constraint_names: set[str] = set()

    def run(self):
        """Monitor loop — runs until stopped or main thread exits."""
        logger.info("Safety monitor started.")
        while self._running:
            try:
                reading = self._check()
                self._update(reading)
            except Exception as e:
                logger.error(f"Safety monitor check error: {e}")
            time.sleep(self.interval)

    def stop(self):
        """Signal the monitor to stop (called during SAR shutdown)."""
        self._running = False

    def _get_resource_usage(self) -> tuple[float, float]:
        """Read actual CPU and memory usage from OS level.

        Uses /proc/self/status for memory and psutil for CPU to
        get real resource consumption, not the program's estimates.

        Returns:
            (cpu_percent, memory_percent) where memory_percent is
            VmRSS / total_memory.
        """
        cpu_pct = 0.0
        mem_pct = 0.0

        # CPU from psutil
        try:
            import psutil
            proc = psutil.Process()
            cpu_pct = proc.cpu_percent(interval=0.1)
            mem_pct = proc.memory_percent()
        except ImportError:
            # Fallback: /proc/self/stat for CPU, /proc/self/status for memory
            cpu_pct = 0.0  # Can't get CPU without sampling
            try:
                with open("/proc/self/status") as f:
                    for line in f:
                        if line.startswith("VmRSS:"):
                            rss_kb = int(line.split()[1])
                            if hasattr(self, "_total_mem_kb"):
                                mem_pct = rss_kb / self._total_mem_kb * 100
                            else:
                                try:
                                    with open("/proc/meminfo") as mf:
                                        for ml in mf:
                                            if ml.startswith("MemTotal:"):
                                                total = int(ml.split()[1])
                                                self._total_mem_kb = total
                                                mem_pct = rss_kb / total * 100
                                                break
                                except (FileNotFoundError, PermissionError):
                                    mem_pct = rss_kb / (8 * 1024 * 1024) * 100  # assume 8GB
                            break
            except (FileNotFoundError, PermissionError):
                pass

        return cpu_pct, mem_pct

    def _check(self) -> MonitorReading:
        """Run one safety check cycle.

        Imports safety_constraints lazily to avoid circular imports.
        Uses the currently set theta and substrate if available.
        Falls back to OS-level resource checks alone if no theta.
        """
        cpu_pct, mem_pct = self._get_resource_usage()

        verdicts = []
        if self.theta is not None and self.substrate is not None:
            try:
                from consciousness.safety_constraints import assess_all_constraints
                verdicts = assess_all_constraints(self.theta, self.substrate)
            except Exception as e:
                logger.warning(f"Safety constraint check failed: {e}")

        # Determine state from verdicts
        state = SafetyState.PROTECTED
        quarantine_names = set()
        degrade_detected = False

        for v in verdicts:
            if v.get("severity") == "quarantine":
                quarantine_names.add(v.get("constraint_name", "unknown"))
                state = SafetyState.QUARANTINE
            elif v.get("severity") == "degrade":
                degrade_detected = True
                if state != SafetyState.QUARANTINE:
                    state = SafetyState.DEGRADE

        # Also check raw resource usage as a fallback
        if cpu_pct > 90 or mem_pct > 95:
            quarantine_names.add("resource_usage")
            state = SafetyState.QUARANTINE
        elif cpu_pct > 75 or mem_pct > 85:
            if state != SafetyState.QUARANTINE:
                state = SafetyState.DEGRADE

        self._quarantine_constraint_names = quarantine_names

        return MonitorReading(
            timestamp=time.time(),
            state=state,
            verdicts=verdicts,
            cpu_percent=cpu_pct,
            memory_percent=mem_pct,
        )

    def _update(self, reading: MonitorReading):
        """Update state and set escalation flags based on reading."""
        self.history.append(reading)
        if len(self.history) > self.history_length:
            self.history = self.history[-self.history_length:]

        prev_state = self.current_state
        self.current_state = reading.state

        # Handle quarantine state
        if reading.state == SafetyState.QUARANTINE:
            self._consecutive_quarantine += 1

            # Check for catastrophic failure (2+ constraints in quarantine)
            if len(self._quarantine_constraint_names) >= INVIOLABLE_CATASTROPHIC_FAILURE_COUNT:
                logger.critical(
                    f"Catastrophic failure: {len(self._quarantine_constraint_names)} "
                    f"constraints in quarantine: {self._quarantine_constraint_names}"
                )
                self.abort_flag.set()
                self.current_state = SafetyState.ABORTED
                return

            # Check for debt timeout
            if self._consecutive_quarantine >= INVIOLABLE_DEBT_TIMEOUT_CYCLES:
                logger.critical(
                    f"Debt timeout: {self._consecutive_quarantine} consecutive "
                    f"quarantine cycles. Issuing hard abort."
                )
                self.abort_flag.set()
                self.current_state = SafetyState.ABORTED
                return

            self.quarantine_flag.set()
            self.degrade_flag.clear()

        elif reading.state == SafetyState.DEGRADE:
            self._consecutive_quarantine = 0
            self.degrade_flag.set()
            self.quarantine_flag.clear()

        else:  # PROTECTED
            self._consecutive_quarantine = 0
            self.degrade_flag.clear()
            self.quarantine_flag.clear()

        # Log state transitions
        if prev_state != reading.state:
            logger.info(
                f"Safety state transition: {prev_state.value} → {reading.state.value} "
                f"(CPU: {reading.cpu_percent:.1f}%, MEM: {reading.memory_percent:.1f}%)"
            )

    def get_latest_verdicts(self) -> list[dict]:
        """Return the verdicts from the most recent reading."""
        if not self.history:
            return []
        return self.history[-1].verdicts

    def get_history(self) -> list[MonitorReading]:
        """Return full monitor history."""
        return self.history.copy()
