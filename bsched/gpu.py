"""Device synchronisation, device memory, and GPU/system telemetry.

* `Cudart` wraps the CUDA runtime that ONNX Runtime itself uses (the
  cudart64_12.dll from the nvidia-cuda-runtime wheel, already loaded by
  `bsched.models._preload_cuda_dlls`), so `synchronize()` acts on the same
  primary context as ORT's kernels.
* `Telemetry` runs `nvidia-smi -lms` in a separate *process* (not a Python
  thread, which would compete for the GIL with the timed loop) and parses its
  CSV afterwards.
* `system_snapshot()` records RAM/commit, AC power and power plan, so a run
  degraded by paging or battery power can be identified after the fact.
"""
from __future__ import annotations

import ctypes
import glob
import os
import subprocess
import sys
import time

import onnxruntime as ort


class Cudart:
    def __init__(self, device_id: int = 0):
        from bsched.models import _preload_cuda_dlls
        _preload_cuda_dlls()
        site = os.path.dirname(os.path.dirname(ort.__file__))
        cands = glob.glob(os.path.join(site, "nvidia", "cuda_runtime", "bin", "cudart64_*.dll"))
        name = cands[0] if cands else ("cudart64_12.dll" if sys.platform == "win32" else "libcudart.so.12")
        self.lib = ctypes.CDLL(name)
        self.path = name
        self._check(self.lib.cudaSetDevice(ctypes.c_int(device_id)), "cudaSetDevice")

    @staticmethod
    def _check(rc: int, what: str) -> None:
        if rc != 0:
            raise RuntimeError(f"{what} failed with cudaError {rc}")

    def synchronize(self) -> None:
        self._check(self.lib.cudaDeviceSynchronize(), "cudaDeviceSynchronize")

    def mem_info(self) -> tuple[int, int]:
        """(used_bytes, total_bytes), device-wide (includes other processes)."""
        free, total = ctypes.c_size_t(), ctypes.c_size_t()
        self._check(self.lib.cudaMemGetInfo(ctypes.byref(free), ctypes.byref(total)), "cudaMemGetInfo")
        return total.value - free.value, total.value


def nvsmi_used_mib() -> int | None:
    """Device memory used per nvidia-smi; usable before any CUDA context exists."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, check=True).stdout
        return int(out.strip().splitlines()[0])
    except Exception:  # noqa: BLE001
        return None


# ------------------------------------------------------------------ telemetry

_FIELDS = ["timestamp", "clocks.sm", "clocks.mem", "temperature.gpu", "power.draw",
           "utilization.gpu", "memory.used", "clocks_event_reasons.active", "pstate"]

# Bits of clocks_event_reasons (nvml.h, nvmlClocksEventReason*).
_REASONS = {0x1: "gpu_idle", 0x2: "app_clocks", 0x4: "sw_power_cap", 0x8: "hw_slowdown",
            0x20: "sync_boost", 0x40: "sw_thermal", 0x80: "hw_thermal", 0x100: "hw_power_brake",
            0x200: "display_clocks"}


class Telemetry:
    def __init__(self, csv_path: str, interval_ms: int = 100):
        self.csv_path, self.interval_ms, self.proc = csv_path, interval_ms, None

    def __enter__(self):
        self._f = open(self.csv_path, "w")
        self.proc = subprocess.Popen(
            ["nvidia-smi", f"--query-gpu={','.join(_FIELDS)}", "--format=csv,noheader,nounits",
             f"-lms={self.interval_ms}"], stdout=self._f, stderr=subprocess.DEVNULL)
        time.sleep(0.3)  # let the first sample land before the timed loop
        return self

    def __exit__(self, *exc):
        self.proc.terminate()
        self.proc.wait(timeout=5)
        self._f.close()

    def summary(self) -> dict:
        rows = []
        with open(self.csv_path) as f:
            for line in f:
                parts = [p.strip() for p in line.split(",")]
                if len(parts) == len(_FIELDS):
                    rows.append(dict(zip(_FIELDS, parts)))
        if not rows:
            return {"n_samples": 0}

        def col(k):
            vals = []
            for r in rows:
                try:
                    vals.append(float(r[k]))
                except ValueError:
                    pass
            return vals

        def stats(k):
            v = col(k)
            return {"min": min(v), "max": max(v), "mean": sum(v) / len(v)} if v else None

        reason_counts: dict[str, int] = {}
        for r in rows:
            try:
                bits = int(r["clocks_event_reasons.active"], 16)
            except ValueError:
                continue
            for bit, name in _REASONS.items():
                if bits & bit:
                    reason_counts[name] = reason_counts.get(name, 0) + 1
        return {"n_samples": len(rows), "interval_ms": self.interval_ms,
                "sm_clock_mhz": stats("clocks.sm"), "mem_clock_mhz": stats("clocks.mem"),
                "temp_c": stats("temperature.gpu"), "power_w": stats("power.draw"),
                "util_pct": stats("utilization.gpu"), "device_mem_used_mib": stats("memory.used"),
                "clock_event_reason_sample_counts": reason_counts,
                "pstates": sorted({r["pstate"] for r in rows})}


# --------------------------------------------------------------------- system

def system_snapshot() -> dict:
    snap: dict = {}
    if sys.platform == "win32":
        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
        m = MEMORYSTATUSEX(dwLength=ctypes.sizeof(MEMORYSTATUSEX))
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
        gib = 1 << 30
        snap["ram"] = {"total_gib": round(m.ullTotalPhys / gib, 2), "avail_gib": round(m.ullAvailPhys / gib, 2),
                       "commit_used_gib": round((m.ullTotalPageFile - m.ullAvailPageFile) / gib, 2),
                       "commit_limit_gib": round(m.ullTotalPageFile / gib, 2)}

        class SYSTEM_POWER_STATUS(ctypes.Structure):
            _fields_ = [("ACLineStatus", ctypes.c_ubyte), ("BatteryFlag", ctypes.c_ubyte),
                        ("BatteryLifePercent", ctypes.c_ubyte), ("SystemStatusFlag", ctypes.c_ubyte),
                        ("BatteryLifeTime", ctypes.c_ulong), ("BatteryFullLifeTime", ctypes.c_ulong)]
        p = SYSTEM_POWER_STATUS()
        ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(p))
        snap["on_ac_power"] = {0: False, 1: True}.get(p.ACLineStatus)
        snap["battery_pct"] = None if p.BatteryLifePercent == 255 else p.BatteryLifePercent
        try:
            snap["power_scheme"] = subprocess.run(["powercfg", "/getactivescheme"], capture_output=True,
                                                  text=True).stdout.strip()
        except Exception:  # noqa: BLE001
            snap["power_scheme"] = None
    return snap
