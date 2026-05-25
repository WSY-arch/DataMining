from __future__ import annotations

from typing import TYPE_CHECKING, Any
import os

# Avoid loky subprocess attempts to probe physical cores on Windows which
# can raise "[WinError 2] 系统找不到指定的文件。" when wmic is unavailable.
# Prefer to default to the logical CPU count so loky emits no warning.
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))

from .clustering import ClusteringResult, cluster_time_series, k_medoids
from .perturbations import (
    add_gaussian_noise,
    random_global_shift,
    truncate_and_resample,
)

if TYPE_CHECKING:
    from .measures.isolation_kernel import IsolationKernel as IsolationKernelType


__all__ = [
    "ClusteringResult",
    "IsolationKernel",
    "add_gaussian_noise",
    "cluster_time_series",
    "k_medoids",
    "random_global_shift",
    "truncate_and_resample",
]


def __getattr__(name: str) -> Any:
    # Lazy import to avoid heavy optional dependency loading at package import time.
    if name == "IsolationKernel":
        from .measures.isolation_kernel import IsolationKernel

        return IsolationKernel
    raise AttributeError(f"module 'tsclust' has no attribute {name!r}")
