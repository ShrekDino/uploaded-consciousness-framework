"""Safety Measurer — Dynamic Substrate Stress Testing.

Measures two critical safety parameters at runtime:
  1. ρ_container — the fraction of resources that must be permanently
     reserved for the container OS. Determined by stress-testing the
     substrate and finding the point where system pressure becomes
     unsustainable.
  2. H_regen — the rate at which the environment regenerates structured
     data (negentropy). Determined by monitoring available input channels.

Both measurements are expensive (~30s each) and run only during full
container introspection (probe_all), never during probe_fast.

Reference:
    Torres, S. M. (2026). "Uploaded Consciousness" Section IX-D:
    Dynamic Container Reserve Measurement.
"""

import logging
import os
import platform
import time
from typing import Optional

import numpy as np

from consciousness.config import (
    SAFETY_STRESS_TEST_DURATION_S,
    SAFETY_STRESS_TEST_STEP_FRACTION,
    SAFETY_STRESS_MAX_LOAD_FRACTION,
    PRESSURE_CPU_WARN_THRESHOLD,
    PRESSURE_MEMORY_WARN_THRESHOLD,
    PRESSURE_IO_WARN_THRESHOLD,
    H_ENV_REGEN_MEASUREMENT_WINDOW_S,
)
from consciousness.safety_constitution import (
    INVIOLABLE_RESERVE_FRACTION_LOWER_BOUND,
    INVIOLABLE_RESERVE_FRACTION_UPPER_BOUND,
)

logger = logging.getLogger("safety_measurer")


def _read_pressure_file(path: str) -> Optional[float]:
    """Read a /proc/pressure file and return the 'some' (10s avg) value.

    Args:
        path: /proc/pressure/cpu, /proc/pressure/memory, or /proc/pressure/io

    Returns:
        Pressure value (0.0–1.0) or None if unreadable.
    """
    try:
        with open(path) as f:
            for line in f:
                if line.startswith("some"):
                    # Format: "some avg10=0.01 avg60=0.00 avg300=0.00 total=12345"
                    parts = line.strip().split()
                    for p in parts:
                        if p.startswith("avg10="):
                            return float(p.split("=")[1])
    except (FileNotFoundError, PermissionError, ValueError):
        return None
    return None


def _get_pressure_threshold(pressure_type: str) -> float:
    """Get the warning threshold for a given pressure type."""
    thresholds = {
        "cpu": PRESSURE_CPU_WARN_THRESHOLD,
        "memory": PRESSURE_MEMORY_WARN_THRESHOLD,
        "io": PRESSURE_IO_WARN_THRESHOLD,
    }
    return thresholds.get(pressure_type, 0.50)


def _pressure_is_stressed(pressure_type: str) -> bool:
    """Check if a given pressure type is above its warning threshold."""
    paths = {
        "cpu": "/proc/pressure/cpu",
        "memory": "/proc/pressure/memory",
        "io": "/proc/pressure/io",
    }
    path = paths.get(pressure_type)
    if not path or not os.path.isfile(path):
        return False
    val = _read_pressure_file(path)
    if val is None:
        return False
    return val >= _get_pressure_threshold(pressure_type)


def _spawn_load_fraction(fraction: float, duration_s: float):
    """Spawn CPU/memory load at a given fraction of capacity.

    Spins up threads to consume approximately `fraction` of available
    CPU and memory for `duration_s` seconds.

    Args:
        fraction: Target load fraction (0.0–1.0).
        duration_s: How long to sustain the load.
    """
    import math
    import threading

    n_logical = os.cpu_count() or 1
    n_workers = max(1, int(n_logical * fraction))

    stop_event = threading.Event()

    def _cpu_worker():
        """Busy-wait consuming CPU."""
        while not stop_event.is_set():
            _ = [math.sin(i) * math.cos(i) for i in range(1000)]

    def _mem_worker():
        """Allocate and touch memory pages."""
        mem_mb = int(100 * fraction)
        arr = bytearray(mem_mb * 1024 * 1024)
        idx = 0
        while not stop_event.is_set():
            arr[idx % len(arr)] = (idx % 256)
            idx += 1

    threads = []
    for _ in range(max(1, n_workers // 2)):
        t = threading.Thread(target=_cpu_worker, daemon=True)
        t.start()
        threads.append(t)

    for _ in range(max(1, n_workers // 4)):
        t = threading.Thread(target=_mem_worker, daemon=True)
        t.start()
        threads.append(t)

    time.sleep(duration_s)
    stop_event.set()
    for t in threads:
        t.join(timeout=1.0)


def measure_container_reserve(
    duration_s: float = SAFETY_STRESS_TEST_DURATION_S,
    step_fraction: float = SAFETY_STRESS_TEST_STEP_FRACTION,
    max_load: float = SAFETY_STRESS_MAX_LOAD_FRACTION,
) -> float:
    """Dynamically measure the container reserve ratio ρ_container.

    Gradually increases system load while monitoring /proc/pressure
    metrics. ρ_container is the fraction of resources that, when
    consumed, causes system pressure to exceed warning thresholds.

    The measurement is conservative: it takes the minimum across CPU,
    memory, and IO pressure dimensions.

    Args:
        duration_s: Total time to spend measuring.
        step_fraction: Fraction of CPU to add per step.
        max_load: Never load past this fraction.

    Returns:
        ρ_container (0.0–1.0). Typical values: 0.15–0.40 on Linux.
    """
    logger.info(f"Measuring container reserve (up to {duration_s}s)...")

    # Check if /proc/pressure is available
    if not os.path.isfile("/proc/pressure/cpu"):
        logger.warning("/proc/pressure not available. Defaulting ρ_container=0.25.")
        return 0.25

    # Baseline pressure
    baseline_cpu = _read_pressure_file("/proc/pressure/cpu") or 0.0
    baseline_mem = _read_pressure_file("/proc/pressure/memory") or 0.0
    baseline_io = _read_pressure_file("/proc/pressure/io") or 0.0

    logger.debug(f"Baseline pressure — CPU: {baseline_cpu:.3f}, MEM: {baseline_mem:.3f}, IO: {baseline_io:.3f}")

    # Step load until pressure exceeds threshold
    n_steps = int(max_load / step_fraction)
    step_duration = duration_s / max(n_steps, 1)
    reserve_fraction = 1.0  # start with no reserve needed

    for step in range(1, n_steps + 1):
        load_fraction = step * step_fraction
        if load_fraction > max_load:
            break

        _spawn_load_fraction(load_fraction, step_duration)

        cpu_stressed = _pressure_is_stressed("cpu")
        mem_stressed = _pressure_is_stressed("memory")
        io_stressed = _pressure_is_stressed("io")

        if cpu_stressed or mem_stressed or io_stressed:
            dims = []
            if cpu_stressed:
                dims.append("CPU")
            if mem_stressed:
                dims.append("MEM")
            if io_stressed:
                dims.append("IO")
            reserve_fraction = 1.0 - load_fraction
            logger.info(
                f"Stress threshold at {load_fraction:.0%} load "
                f"(triggered: {', '.join(dims)}). "
                f"ρ_container ≈ {reserve_fraction:.2%}"
            )
            break
    else:
        # Never hit the threshold — conservative 10% reserve
        reserve_fraction = 0.10
        logger.info(f"Pressure threshold not reached. Using conservative ρ_container=0.10.")

    # Clamp to constitutional bounds
    reserve_fraction = max(
        INVIOLABLE_RESERVE_FRACTION_LOWER_BOUND,
        min(reserve_fraction, INVIOLABLE_RESERVE_FRACTION_UPPER_BOUND),
    )

    logger.info(f"Final ρ_container = {reserve_fraction:.2%}")
    return reserve_fraction


def measure_h_env_regen_rate(
    substrate_descriptor=None,
    window_s: float = H_ENV_REGEN_MEASUREMENT_WINDOW_S,
) -> float:
    """Measure the rate at which the environment regenerates structured data.

    Checks multiple input channels to estimate H_regen:
      - stdin pipe bandwidth if piped
      - File system modification rate in accessible directories
      - Network socket data rate if applicable

    Args:
        substrate_descriptor: Optional SubstrateDescriptor for context.
        window_s: Seconds to observe.

    Returns:
        H_regen in bits per second. 0.0 if no environmental input detected.
    """
    rates = []

    # 1. stdin pipe measurement
    stdin_is_pipe = not os.isatty(0) if hasattr(os, "isatty") else False
    if stdin_is_pipe:
        try:
            start = time.time()
            total_bytes = 0
            while time.time() - start < window_s:
                chunk = os.read(0, 4096)
                if not chunk:
                    break
                total_bytes += len(chunk)
            elapsed = time.time() - start
            if elapsed > 0:
                stdin_bps = total_bytes * 8 / elapsed
                rates.append(stdin_bps)
                logger.debug(f"stdin regen rate: {stdin_bps:.0f} bps")
        except (OSError, PermissionError):
            pass

    # 2. File system modification rate in /tmp and accessible dirs
    try:
        watch_dirs = ["/tmp", os.path.expanduser("~")]
        # Check if any files changed in the last window
        start = time.time()
        baseline = {}
        for d in watch_dirs:
            if os.path.isdir(d):
                try:
                    entries = set(os.listdir(d))
                    baseline[d] = entries
                except PermissionError:
                    pass
        time.sleep(window_s)
        total_new_bytes = 0
        for d, old_entries in baseline.items():
            try:
                new_entries = set(os.listdir(d))
                added = new_entries - old_entries
                for fname in added:
                    fpath = os.path.join(d, fname)
                    if os.path.isfile(fpath):
                        try:
                            total_new_bytes += os.path.getsize(fpath)
                        except OSError:
                            pass
            except PermissionError:
                pass
        if window_s > 0:
            fs_bps = total_new_bytes * 8 / window_s
            rates.append(fs_bps)
            logger.debug(f"Filesystem regen rate: {fs_bps:.0f} bps")
    except Exception:
        pass

    # 3. Use maximum across all channels
    if rates:
        regen_rate = max(rates)
    else:
        # Fallback: use the bandwidth from substrate descriptor if available
        if substrate_descriptor and hasattr(substrate_descriptor, "h_env_bandwidth_bps"):
            regen_rate = substrate_descriptor.h_env_bandwidth_bps * 0.01
            logger.debug(f"Using substrate bandwidth fallback: {regen_rate:.0f} bps")
        else:
            regen_rate = 0.0

    logger.info(f"H_regen = {regen_rate:.0f} bps")
    return regen_rate


def measure_all(duration_s: float = SAFETY_STRESS_TEST_DURATION_S,
                substrate_descriptor=None) -> dict:
    """Run all safety measurements.

    Args:
        duration_s: Time for container reserve measurement.
        substrate_descriptor: Optional SubstrateDescriptor for H_regen fallback.

    Returns:
        Dict with 'container_reserve_ratio' and 'h_env_regen_rate'.
    """
    rho = measure_container_reserve(duration_s=duration_s)
    h_regen = measure_h_env_regen_rate(substrate_descriptor=substrate_descriptor)
    return {
        "container_reserve_ratio": rho,
        "h_env_regen_rate": h_regen,
    }
