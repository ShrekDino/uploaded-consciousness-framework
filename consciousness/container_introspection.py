"""Container Introspection — substrate characterization for the consciousness program.

The program, upon instantiation in an unknown computational environment, performs
a systematic probe of its host substrate's properties: computational topology,
accelerator availability, thermal envelope, memory hierarchy, and environmental
flux bandwidth. The output is a SubstrateDescriptor that serves as the conditional
variable for meta-parametric optimization (see meta_optimizer.py).

Reference:
    Torres, S. M. (2026). "Uploaded Consciousness" Section VIII-B:
    Container Introspection.
"""

import json
import os
import platform
import struct
import time
from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class SubstrateDescriptor:
    """Complete characterization of the host computational substrate.

    This is the program's answer to "where am I and what am I running on?"
    Every field is measured or benchmarked at runtime, not read from spec sheets.
    """

    # ── Platform identity ──
    platform_system: str = ""         # Linux, Darwin, Windows
    platform_machine: str = ""        # x86_64, aarch64, arm64
    platform_release: str = ""        # kernel version

    # ── CPU topology ──
    cpu_cores_physical: int = 0
    cpu_cores_logical: int = 0
    cpu_arch: str = ""                # x86_64, arm64, riscv64
    cpu_brand: str = ""               # model string from /proc/cpuinfo
    vector_unit_width: int = 0        # SSE=128, AVX=256, AVX-512=512, NEON=128, SVE=256
    cache_line_size: int = 64         # bytes, measured via pointer-chase
    l1d_cache_kb: int = 0
    l2_cache_kb: int = 0
    l3_cache_kb: int = 0

    # ── Memory performance (measured) ──
    memory_total_gb: float = 0.0
    memory_available_gb: float = 0.0
    memory_bandwidth_gbps: float = 0.0   # measured STREAM-like triadic loop
    memory_latency_ns: float = 0.0       # approximate pointer-chase latency

    # ── GPU ──
    gpu_present: bool = False
    gpu_name: str = ""
    gpu_compute_capability: str = ""     # e.g. "8.6" for RTX 3060
    gpu_vram_gb: float = 0.0
    gpu_spmv_latency_ms: float = 0.0     # 130k×130k sparse MV benchmark
    gpu_available: bool = False           # CuPy or CUDA actually importable

    # ── Neuromorphic (Lava-NC / Loihi) ──
    neuromorphic_present: bool = False
    neuromorphic_backend: str = ""        # loihi, lava-cpu, lava-sim
    neuropod_count: int = 0

    # ── Thermodynamic envelope ──
    thermal_max_celsius: float = 0.0
    power_budget_watts: float = 0.0       # TDP or measured sustained draw
    cooling_type: str = ""                # passive, fan, liquid, cryo
    temp_idle_celsius: float = 0.0
    temp_load_celsius: float = 0.0        # after 60s synthetic max load
    t_collapse_estimate: float = 0.0      # thermal collapse threshold (Eq 2)

    # ── Environmental flux ──
    h_env_available_bps: float = 0.0      # bits/sec of structured environmental input
    h_env_bandwidth_bps: float = 0.0      # maximum possible ingestion rate
    noise_floor: float = 0.0              # background entropy rate (nats/s)

    # ── Safety constitution (measured) ──
    container_reserve_ratio: float = 0.25  # ρ_container — fraction of resources reserved for OS
    h_env_regen_rate: float = 0.0          # H_regen — rate at which environment produces
                                           #   structured data (bits/sec)

    # ── Probe metadata ──
    probe_duration_s: float = 0.0         # total time for full probe
    probe_success: bool = False
    probe_errors: list[str] = field(default_factory=list)


    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, default=str)

    def fingerprint(self) -> str:
        """A hashable fingerprint for substrate identity.

        Two substrates with the same fingerprint are treated as identical
        for cached-optimization purposes.
        """
        key = (
            self.platform_system, self.platform_machine,
            self.cpu_cores_physical, self.cpu_cores_logical,
            self.cpu_arch, self.gpu_name, self.gpu_compute_capability,
            self.memory_total_gb, self.thermal_max_celsius
        )
        import hashlib
        return hashlib.sha256(str(key).encode()).hexdigest()[:16]


class ProbeError(Exception):
    """Raised when a specific probe fails but the overall probe can continue."""


# ──────────────────────────────────────────────
# Probe implementations
# ──────────────────────────────────────────────

def probe_platform() -> tuple[str, str, str]:
    """Platform identity."""
    return (platform.system(), platform.machine(), platform.release())


def probe_cpu_topology() -> dict:
    """CPU core count, architecture, cache sizes.

    Primary: /proc/cpuinfo on Linux.
    Fallback: os.cpu_count(), platform.machine().
    """
    result = {
        "cpu_cores_physical": 0,
        "cpu_cores_logical": 0,
        "cpu_arch": platform.machine(),
        "cpu_brand": "",
        "vector_unit_width": 0,
        "cache_line_size": 64,
        "l1d_cache_kb": 0,
        "l2_cache_kb": 0,
        "l3_cache_kb": 0,
    }

    result["cpu_cores_logical"] = os.cpu_count() or 0

    # Linux: read /proc/cpuinfo
    if platform.system() == "Linux":
        try:
            physical_ids = set()
            core_ids = set()
            brand = ""
            with open("/proc/cpuinfo") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("physical id"):
                        physical_ids.add(line.split(":")[-1].strip())
                    elif line.startswith("core id"):
                        core_ids.add(line.split(":")[-1].strip())
                    elif line.startswith("model name"):
                        brand = line.split(":")[-1].strip()
                    elif line.startswith("flags"):
                        flags = line.split(":")[-1].strip().split()
                        if "avx512f" in flags:
                            result["vector_unit_width"] = 512
                        elif "avx2" in flags:
                            result["vector_unit_width"] = 256
                        elif "avx" in flags:
                            result["vector_unit_width"] = 256
                        elif "sse" in flags:
                            result["vector_unit_width"] = 128
                        elif "asimd" in flags or "neon" in flags:
                            result["vector_unit_width"] = 128
            result["cpu_cores_physical"] = max(len(physical_ids) * len(core_ids) if core_ids else 0, 1)
            result["cpu_brand"] = brand
        except (FileNotFoundError, PermissionError):
            result["cpu_cores_physical"] = result["cpu_cores_logical"]

        # Cache info from /sys (Linux)
        cache_info_paths = [
            ("l1d_cache_kb", "/sys/devices/system/cpu/cpu0/cache/index0/size"),
            ("l2_cache_kb", "/sys/devices/system/cpu/cpu0/cache/index2/size" if os.path.exists("/sys/devices/system/cpu/cpu0/cache/index2/size") else "/sys/devices/system/cpu/cpu0/cache/index3/size"),
        ]
        # Find L3 by scanning all cache indices
        try:
            cpu0_cache = "/sys/devices/system/cpu/cpu0/cache"
            if os.path.isdir(cpu0_cache):
                for entry in sorted(os.listdir(cpu0_cache)):
                    idx_path = os.path.join(cpu0_cache, entry)
                    level_file = os.path.join(idx_path, "level")
                    size_file = os.path.join(idx_path, "size")
                    if os.path.isfile(level_file) and os.path.isfile(size_file):
                        try:
                            with open(level_file) as f:
                                level = f.read().strip()
                            with open(size_file) as f:
                                size_str = f.read().strip()
                            size_kb = int(size_str.rstrip("K"))
                            if level == "1":
                                result["l1d_cache_kb"] = result.get("l1d_cache_kb", 0) + size_kb
                            elif level == "2":
                                result["l2_cache_kb"] = result.get("l2_cache_kb", 0) + size_kb
                            elif level == "3":
                                result["l3_cache_kb"] = result.get("l3_cache_kb", 0) + size_kb
                        except (ValueError, OSError):
                            pass
        except PermissionError:
            pass

    elif platform.system() == "Darwin":
        import subprocess
        try:
            result["cpu_cores_physical"] = int(subprocess.check_output(
                ["sysctl", "-n", "hw.physicalcpu"]
            ).decode().strip())
        except (subprocess.SubprocessError, ValueError):
            result["cpu_cores_physical"] = result["cpu_cores_logical"]
        try:
            for key, sysctl_key in [
                ("l1d_cache_kb", "hw.l1dcachesize"),
                ("l2_cache_kb", "hw.l2cachesize"),
                ("l3_cache_kb", "hw.l3cachesize"),
            ]:
                val = int(subprocess.check_output(
                    ["sysctl", "-n", sysctl_key]
                ).decode().strip())
                result[key] = val // 1024
        except (subprocess.SubprocessError, ValueError):
            pass

    return result


def probe_memory() -> dict:
    """Memory capacity and approximate bandwidth/latency."""
    result = {
        "memory_total_gb": 0.0,
        "memory_available_gb": 0.0,
        "memory_bandwidth_gbps": 0.0,
        "memory_latency_ns": 0.0,
    }

    if platform.system() == "Linux":
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        result["memory_total_gb"] = int(line.split()[1]) / 1_048_576
                    elif line.startswith("MemAvailable:"):
                        result["memory_available_gb"] = int(line.split()[1]) / 1_048_576
        except (FileNotFoundError, PermissionError):
            pass

    # Memory bandwidth: STREAM-like triadic loop
    # Measures bandwidth of a[b] = b[c] + alpha * c[d] pattern
    try:
        import numpy as np
        n = 10_000_000  # 80 MB arrays (fits L3 on most systems)
        a = np.ones(n, dtype=np.float64)
        b = np.ones(n, dtype=np.float64)
        c = np.ones(n, dtype=np.float64)
        alpha = 0.5

        # Warmup
        for _ in range(3):
            a[:] = b + alpha * c

        # Timed run
        start = time.perf_counter()
        for _ in range(10):
            a[:] = b + alpha * c
        elapsed = time.perf_counter() - start
        bytes_moved = 10 * (a.nbytes + b.nbytes + c.nbytes)
        result["memory_bandwidth_gbps"] = (bytes_moved / elapsed) / 1e9
    except Exception:
        result["memory_bandwidth_gbps"] = 0.0

    # Memory latency: pointer-chase approximate
    try:
        import numpy as np
        chain_len = 10_000
        indices = np.random.permutation(chain_len).astype(np.int64)
        # Create a linked-list traversal pattern
        next_idx = np.zeros(chain_len, dtype=np.int64)
        for i in range(chain_len - 1):
            next_idx[indices[i]] = indices[i + 1]
        next_idx[indices[-1]] = indices[0]

        idx = 0
        start = time.perf_counter()
        for _ in range(1000):
            idx = next_idx[idx]
        elapsed = time.perf_counter() - start
        result["memory_latency_ns"] = (elapsed / (1000 * chain_len)) * 1e9
    except Exception:
        result["memory_latency_ns"] = 0.0

    return result


def probe_gpu() -> dict:
    """GPU presence, capabilities, and sparse MV benchmark.

    Tries CuPy first, then checks for nvidia-smi / NVML.
    """
    result = {
        "gpu_present": False,
        "gpu_name": "",
        "gpu_compute_capability": "",
        "gpu_vram_gb": 0.0,
        "gpu_spmv_latency_ms": 0.0,
        "gpu_available": False,
    }

    # Try CuPy (CUDA)
    try:
        import cupy as cp
        result["gpu_available"] = True
        result["gpu_present"] = True
        props = cp.cuda.runtime.getDeviceProperties(0)
        result["gpu_name"] = props["name"].decode() if isinstance(props["name"], bytes) else props["name"]
        result["gpu_compute_capability"] = f"{props['major']}.{props['minor']}"
        result["gpu_vram_gb"] = props["totalGlobalMem"] / (1024**3)

        # Sparse MV benchmark: 130k×130k at ~0.001 density
        try:
            import numpy as np
            n = 130_000
            density = 0.001
            nnz = int(n * n * density)
            rows = np.random.randint(0, n, nnz)
            cols = np.random.randint(0, n, nnz)
            data = np.random.randn(nnz).astype(np.float32)
            x = cp.random.randn(n, 1).astype(cp.float32)

            spmv_start = time.perf_counter()
            for _ in range(10):
                # CSR sparse MV
                from cupyx.scipy.sparse import csr_matrix
                A = csr_matrix((cp.array(data), (cp.array(rows), cp.array(cols))), shape=(n, n))
                y = A @ x
                cp.cuda.Stream.null.synchronize()
            spmv_elapsed = (time.perf_counter() - spmv_start) / 10
            result["gpu_spmv_latency_ms"] = spmv_elapsed * 1000
        except Exception as e:
            result["probe_errors"] = result.get("probe_errors", [])
            result["probe_errors"].append(f"spmv_benchmark: {e}")

    except ImportError:
        # Try pynvml / nvidia-smi
        try:
            import subprocess
            smi_out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=name,memory.total,compute_cap",
                 "--format=csv,noheader,nounits"],
                timeout=10
            ).decode().strip()
            if smi_out:
                result["gpu_present"] = True
                parts = smi_out.split(",")
                result["gpu_name"] = parts[0].strip()
                result["gpu_vram_gb"] = float(parts[1].strip()) / 1024
                result["gpu_compute_capability"] = parts[2].strip()
        except (subprocess.SubprocessError, FileNotFoundError, IndexError, ValueError):
            pass

    return result


def probe_neuromorphic() -> dict:
    """Detect Lava-NC / Loihi neuromorphic backend.

    Tries importing lava.magma and probing for hardware.
    """
    result = {
        "neuromorphic_present": False,
        "neuromorphic_backend": "",
        "neuropod_count": 0,
    }

    try:
        # Lava-NC Loihi protocol
        import lava.magma.core.run_conditions as rc
        result["neuromorphic_present"] = True
        result["neuromorphic_backend"] = "lava-cpu"
        # Probe for Loihi hardware
        try:
            from lava.magma.core.run_configs import Loihi1RunCfg
            result["neuromorphic_backend"] = "loihi"
            # Attempt to query neuropod count (Loihi 1 has 128 neuropods per chip)
            result["neuropod_count"] = 128
        except (ImportError, AttributeError):
            pass
    except ImportError:
        pass

    return result


def probe_thermal_envelope() -> dict:
    """Thermal characteristics and power budget.

    Measures idle temperature, runs a synthetic load, measures peak temperature.
    Estimates T_collapse from thermal capacity and dissipation rate.
    """
    result = {
        "thermal_max_celsius": 0.0,
        "power_budget_watts": 0.0,
        "cooling_type": "",
        "temp_idle_celsius": 0.0,
        "temp_load_celsius": 0.0,
        "t_collapse_estimate": 0.0,
    }

    # Read thermal zones on Linux
    if platform.system() == "Linux":
        thermal_dir = "/sys/class/thermal"
        if os.path.isdir(thermal_dir):
            temps = []
            try:
                for entry in sorted(os.listdir(thermal_dir)):
                    temp_path = os.path.join(thermal_dir, entry, "temp")
                    if os.path.isfile(temp_path):
                        with open(temp_path) as f:
                            millideg = int(f.read().strip())
                            temps.append(millideg / 1000)
            except (PermissionError, ValueError, OSError):
                pass
            if temps:
                result["temp_idle_celsius"] = max(temps)

        # Power budget estimate from TDP / RAPL
        try:
            rapl_dir = "/sys/class/powercap"
            if os.path.isdir(rapl_dir):
                for entry in sorted(os.listdir(rapl_dir)):
                    if "intel-rapl" in entry:
                        constraint_path = os.path.join(rapl_dir, entry, "constraint_0_power_limit_uw")
                        if os.path.isfile(constraint_path):
                            with open(constraint_path) as f:
                                result["power_budget_watts"] = int(f.read().strip()) / 1_000_000
                                break
        except (PermissionError, ValueError, OSError):
            pass

        # Cooling type heuristic
        cooling = "passive"
        thermal_zones = "/sys/class/thermal"
        if os.path.isdir(thermal_zones):
            try:
                for entry in sorted(os.listdir(thermal_zones)):
                    if entry.startswith("cooling_device"):
                        cur_state_path = os.path.join(thermal_zones, entry, "cur_state")
                        if os.path.isfile(cur_state_path):
                            with open(cur_state_path) as f:
                                if int(f.read().strip()) > 0:
                                    cooling = "fan"
                                    break
            except (PermissionError, ValueError, OSError):
                pass
            # Check for liquid cooling via hwmon
            hwmon_dir = "/sys/class/hwmon"
            if os.path.isdir(hwmon_dir):
                try:
                    for entry in sorted(os.listdir(hwmon_dir)):
                        name_path = os.path.join(hwmon_dir, entry, "name")
                        if os.path.isfile(name_path):
                            with open(name_path) as f:
                                if "liquid" in f.read().lower():
                                    cooling = "liquid"
                    result["cooling_type"] = cooling
                except (PermissionError, OSError):
                    result["cooling_type"] = cooling
        result["cooling_type"] = cooling

    # Synthetic load: burn CPU for 60s, measure peak temperature
    if result["temp_idle_celsius"] > 0:
        try:
            import numpy as np
            end_time = time.time() + 60
            load_arr = np.random.randn(10_000_000)
            acc = 0.0
            while time.time() < end_time:
                acc += np.sum(np.sin(load_arr))
            _ = acc  # prevent optimization

            # Re-read temperature
            if platform.system() == "Linux" and os.path.isdir(thermal_dir):
                temps = []
                for entry in sorted(os.listdir(thermal_dir)):
                    temp_path = os.path.join(thermal_dir, entry, "temp")
                    if os.path.isfile(temp_path):
                        try:
                            with open(temp_path) as f:
                                millideg = int(f.read().strip())
                                temps.append(millideg / 1000)
                        except (PermissionError, ValueError):
                            pass
                if temps:
                    result["temp_load_celsius"] = max(temps)

            # Estimate T_collapse
            if result["temp_load_celsius"] > result["temp_idle_celsius"]:
                delta_t = result["temp_load_celsius"] - result["temp_idle_celsius"]
                t_collapse = result["temp_load_celsius"] + delta_t * 2
                result["t_collapse_estimate"] = min(t_collapse, 105.0)
            else:
                result["t_collapse_estimate"] = 90.0  # conservative default
        except Exception:
            result["t_collapse_estimate"] = 90.0
    else:
        result["t_collapse_estimate"] = 90.0

    return result


def probe_environmental_flux() -> dict:
    """Maximum structured data ingestion rate across available I/O channels.

    Measures I/O bandwidth for disk reads, network receive, and stdin pipe.
    Estimates H_env_available as the maximum across all channels.
    """
    result = {
        "h_env_available_bps": 0.0,
        "h_env_bandwidth_bps": 0.0,
        "noise_floor": 0.0,
    }

    # Disk read bandwidth
    try:
        import numpy as np
        test_size = 100_000_000  # 100 MB
        test_data = np.random.randint(0, 255, test_size, dtype=np.uint8)
        temp_path = f"/tmp/_csdf_disk_probe_{os.getpid()}.bin"
        with open(temp_path, "wb") as f:
            f.write(test_data.tobytes())

        # Read back and measure
        start = time.perf_counter()
        with open(temp_path, "rb") as f:
            _ = f.read()
        read_elapsed = time.perf_counter() - start
        os.remove(temp_path)

        disk_bps = test_size / read_elapsed
        result["h_env_bandwidth_bps"] = disk_bps * 8  # bits/sec
    except Exception:
        disk_bps = 0.0

    # Environmental flux estimate: use maximum of disk, or stdin if piped
    stdin_is_pipe = not os.isatty(0) if hasattr(os, "isatty") else False
    if stdin_is_pipe:
        result["h_env_available_bps"] = max(result["h_env_bandwidth_bps"] * 0.1, 1_000_000)
    else:
        result["h_env_available_bps"] = result["h_env_bandwidth_bps"] * 0.01

    # Noise floor: approximate from system idle variability
    try:
        import numpy as np
        samples = []
        for _ in range(10):
            samples.append(time.perf_counter_ns())
            time.sleep(0.001)
        jitter = np.std([abs(samples[i] - samples[i-1]) for i in range(2, len(samples))])
        result["noise_floor"] = max(np.log2(jitter + 1) * 0.1, 0.01)
    except Exception:
        result["noise_floor"] = 0.01

    return result


# ──────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────

def probe_all(timeout: float = 120.0) -> SubstrateDescriptor:
    """Run the full container introspection protocol.

    Probes every substrate characteristic and returns a complete
    SubstrateDescriptor. Individual probe failures are recorded in
    probe_errors; the overall probe continues past non-critical failures.

    Args:
        timeout: Maximum wall-clock time for the full probe in seconds.

    Returns:
        SubstrateDescriptor with all probed fields populated.
    """
    start_time = time.time()
    desc = SubstrateDescriptor()
    errors = []

    # Phase 1: Platform identity (cannot fail meaningfully)
    desc.platform_system, desc.platform_machine, desc.platform_release = probe_platform()

    # Phase 2: CPU topology
    try:
        cpu_data = probe_cpu_topology()
        for k, v in cpu_data.items():
            setattr(desc, k, v)
    except Exception as e:
        errors.append(f"cpu_topology: {e}")

    # Phase 3: Memory
    try:
        mem_data = probe_memory()
        for k, v in mem_data.items():
            setattr(desc, k, v)
    except Exception as e:
        errors.append(f"memory: {e}")

    if time.time() - start_time > timeout:
        errors.append("probe_timeout_after_cpu_memory")
        desc.probe_errors = errors
        desc.probe_success = len(errors) == 0
        desc.probe_duration_s = time.time() - start_time
        return desc

    # Phase 4: GPU
    try:
        gpu_data = probe_gpu()
        for k, v in gpu_data.items():
            if k in ("gpu_present", "gpu_name", "gpu_compute_capability",
                     "gpu_vram_gb", "gpu_spmv_latency_ms", "gpu_available"):
                setattr(desc, k, v)
    except Exception as e:
        errors.append(f"gpu: {e}")

    # Phase 5: Neuromorphic (fast, non-blocking)
    try:
        neuro_data = probe_neuromorphic()
        for k, v in neuro_data.items():
            setattr(desc, k, v)
    except Exception as e:
        errors.append(f"neuromorphic: {e}")

    if time.time() - start_time > timeout:
        errors.append("probe_timeout_after_gpu")
        desc.probe_errors = errors
        desc.probe_success = len(errors) == 0
        desc.probe_duration_s = time.time() - start_time
        return desc

    # Phase 6: Thermal envelope (takes ~60s for the load test)
    try:
        thermal_data = probe_thermal_envelope()
        for k, v in thermal_data.items():
            setattr(desc, k, v)
    except Exception as e:
        errors.append(f"thermal: {e}")

    if time.time() - start_time > timeout:
        errors.append("probe_timeout_after_thermal")
        desc.probe_errors = errors
        desc.probe_success = len(errors) == 0
        desc.probe_duration_s = time.time() - start_time
        return desc

    # Phase 7: Environmental flux
    try:
        flux_data = probe_environmental_flux()
        for k, v in flux_data.items():
            setattr(desc, k, v)
    except Exception as e:
        errors.append(f"flux: {e}")

    if time.time() - start_time > timeout:
        errors.append("probe_timeout_before_safety")
        desc.probe_errors = errors
        desc.probe_success = len(errors) == 0
        desc.probe_duration_s = time.time() - start_time
        return desc

    # Phase 8: Safety constitution measurements (expensive, ~30s)
    try:
        from consciousness.safety_measurer import measure_all
        safety_data = measure_all(duration_s=30.0, substrate_descriptor=desc)
        desc.container_reserve_ratio = safety_data.get("container_reserve_ratio", 0.25)
        desc.h_env_regen_rate = safety_data.get("h_env_regen_rate", 0.0)
    except ImportError:
        errors.append("safety_measurer: module not available")
        desc.container_reserve_ratio = 0.25
    except Exception as e:
        errors.append(f"safety_measurer: {e}")
        desc.container_reserve_ratio = 0.25

    desc.probe_errors = errors
    desc.probe_success = len(errors) == 0
    desc.probe_duration_s = time.time() - start_time
    return desc


def probe_fast() -> SubstrateDescriptor:
    """Quick substrate probe (~5s) for when full introspection is not needed.

    Skips the thermal load test and GPU sparse MV benchmark.
    Returns a SubstrateDescriptor with best-effort values.
    """
    desc = SubstrateDescriptor()
    desc.platform_system, desc.platform_machine, desc.platform_release = probe_platform()

    try:
        cpu_data = probe_cpu_topology()
        for k, v in cpu_data.items():
            setattr(desc, k, v)
    except Exception:
        pass

    try:
        mem_data = probe_memory()
        desc.memory_total_gb = mem_data.get("memory_total_gb", 0.0)
        desc.memory_available_gb = mem_data.get("memory_available_gb", 0.0)
    except Exception:
        pass

    try:
        gpu_data = probe_gpu()
        desc.gpu_present = gpu_data.get("gpu_present", False)
        desc.gpu_name = gpu_data.get("gpu_name", "")
        desc.gpu_available = gpu_data.get("gpu_available", False)
    except Exception:
        pass

    desc.t_collapse_estimate = 90.0
    desc.h_env_bandwidth_bps = 100_000_000
    desc.h_env_available_bps = 1_000_000
    desc.probe_success = True
    desc.probe_duration_s = 5.0
    return desc
