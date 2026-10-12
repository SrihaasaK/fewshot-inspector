"""Reproducibility metadata saved next to every results file (config, git SHA, device, versions)."""

from __future__ import annotations

import contextlib
import datetime as dt
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch


def device_name(device: torch.device) -> str:
    """Human-readable accelerator name, e.g. ``Tesla T4`` or ``Apple M4 Pro (MPS)``."""
    if device.type == "cuda":
        return torch.cuda.get_device_name(device)
    if device.type == "mps":
        chip = "Apple Silicon"
        if sys.platform == "darwin":
            with contextlib.suppress(OSError, subprocess.SubprocessError):
                chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True,
                                      text=True, check=True, timeout=5).stdout.strip() or chip
        return f"{chip} (MPS)"
    return f"CPU ({platform.processor() or platform.machine()})"


def git_sha(repo: str | Path = ".") -> str:
    """Current commit SHA (suffixed ``-dirty`` if the tree has uncommitted changes)."""
    try:
        sha = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True,
                             check=True, timeout=5).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=no"],
                               capture_output=True, text=True, check=True, timeout=5).stdout.strip()
        return sha + ("-dirty" if dirty else "")
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def run_metadata(device: torch.device, config: dict[str, Any] | None = None, **extra: Any) -> dict[str, Any]:
    return {
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_sha": git_sha(Path(__file__).resolve().parent.parent),
        "device": str(device),
        "device_name": device_name(device),
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "config": config or {},
        **extra,
    }
