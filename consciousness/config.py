"""Configuration parameters for the consciousness program.

Safety monitor interval, stress test durations, and other tunable
operational parameters. These are NOT constitutional constants —
they can be modified between runs but not by the program at runtime.
"""

# ─── Safety Monitor ───
SAFETY_MONITOR_INTERVAL_S: float = 1.0      # seconds between monitor checks
SAFETY_MONITOR_HISTORY_LENGTH: int = 100    # recent verdicts to retain

# ─── Dynamic Stress Testing ───
SAFETY_STRESS_TEST_DURATION_S: float = 30.0  # seconds for container reserve measurement
SAFETY_STRESS_TEST_STEP_FRACTION: float = 0.10  # load increment per step (10%)
SAFETY_STRESS_MAX_LOAD_FRACTION: float = 0.90   # never load past 90% even in test

# ─── Degrade Mode Parameters ───
SAFETY_DEGRADE_DRIFT_MULTIPLIER: float = 2.0    # drift_duration × 2 in degrade
SAFETY_DEGRADE_MERGE_MULTIPLIER: float = 2.0    # merge_interval × 2 in degrade
SAFETY_DEGRADE_SAMPLE_DIVISOR: float = 2.0      # sample_duration ÷ 2 in degrade

# ─── Quarantine Mode ───
SAFETY_QUARANTINE_MAX_CYCLES: int = 100     # max cycles in quarantine before hard abort

# ─── Pressure Thresholds ───
# /proc/pressure values above which the system is considered stressed
PRESSURE_CPU_WARN_THRESHOLD: float = 0.30   # some (10s avg) pressure
PRESSURE_MEMORY_WARN_THRESHOLD: float = 0.20
PRESSURE_IO_WARN_THRESHOLD: float = 0.30

# ─── Environment Regeneration Measurement ───
H_ENV_REGEN_MEASUREMENT_WINDOW_S: float = 10.0  # seconds to measure input rate
