"""CPU/GPU memory preflight for a prepared training plan."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ResourcePreflight:
    """Actionable device fit decision shown before training starts."""

    ok: bool
    device: str
    required_bytes: int
    available_bytes: int
    usable_bytes: int
    recommended_retain_fraction: float
    reason: str


def training_preflight(
    required_bytes: int,
    *,
    device: str = "auto",
    reserve_fraction: float = 0.15,
    available_cuda_bytes: int | None = None,
) -> ResourcePreflight:
    """Check CUDA fit and recommend a point-retention ratio when oversized."""

    if required_bytes < 1:
        raise ValueError("required_bytes must be positive")
    if device not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be auto, cpu, or cuda")
    if not 0 <= reserve_fraction < 1:
        raise ValueError("reserve_fraction must be in [0, 1)")
    if device == "cpu":
        return ResourcePreflight(
            ok=True,
            device="cpu",
            required_bytes=required_bytes,
            available_bytes=0,
            usable_bytes=0,
            recommended_retain_fraction=1.0,
            reason="CPU selected; system RAM must be checked by the caller.",
        )

    if available_cuda_bytes is None:
        from naviertwin.utils.device import available_memory_mb

        memory = available_memory_mb()
        if bool(memory["cuda"]):
            available_cuda_bytes = int(float(memory["free"]) * 1024**2)
        else:
            available_cuda_bytes = 0
    if available_cuda_bytes <= 0:
        if device == "cuda":
            return ResourcePreflight(
                False,
                "cuda",
                required_bytes,
                0,
                0,
                0.0,
                "CUDA requested but no CUDA memory is available.",
            )
        return ResourcePreflight(
            True,
            "cpu",
            required_bytes,
            0,
            0,
            1.0,
            "CUDA unavailable; use CPU or install a compatible CUDA backend.",
        )

    usable = int(available_cuda_bytes * (1.0 - reserve_fraction))
    ok = required_bytes <= usable
    ratio = 1.0 if ok else max(0.01, min(1.0, usable / required_bytes))
    if ok:
        reason = "Estimated training tensors fit in available CUDA memory."
    else:
        reason = (
            "Estimated training memory exceeds CUDA headroom; reduce retained points "
            f"to approximately {ratio:.1%}, lower batch size, or enable gradient accumulation."
        )
    return ResourcePreflight(
        ok=ok,
        device="cuda",
        required_bytes=required_bytes,
        available_bytes=available_cuda_bytes,
        usable_bytes=usable,
        recommended_retain_fraction=ratio,
        reason=reason,
    )


__all__ = ["ResourcePreflight", "training_preflight"]
