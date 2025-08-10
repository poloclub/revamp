

import torch as ch
from typing import Union

__all__ = ["gpu_mem", "NvtxRange"]


def _fmt_mb(x: int) -> str:
    return f"{x / (1024**2):.1f} MB"


def gpu_mem(tag: str = "", device: Union[str, int] = 0):
    """Lightweight memory probe. Safe to call even if CUDA is unavailable."""
    try:
        ch.cuda.synchronize()
    except Exception:
        pass
    if not ch.cuda.is_available():
        print(f"[MEM] {tag} | CUDA not available")
        return
    dev = ch.device(device) if isinstance(device, (int, str)) else device
    alloc = ch.cuda.memory_allocated(dev)
    reserv = ch.cuda.memory_reserved(dev)
    peak = ch.cuda.max_memory_allocated(dev)
    print(f"[MEM] {tag} | allocated={_fmt_mb(alloc)} reserved={_fmt_mb(reserv)} peak={_fmt_mb(peak)}")


class NvtxRange:
    """Context manager to push/pop NVTX ranges if available; otherwise no-op."""

    def __init__(self, msg: str):
        self.msg = msg

    def __enter__(self):
        try:
            ch.cuda.nvtx.range_push(self.msg)
        except Exception:
            pass

    def __exit__(self, exc_type, exc, tb):
        try:
            ch.cuda.nvtx.range_pop()
        except Exception:
            pass