"""Backends. A backend casts inputs to its working dtype, runs the op's
(framework-agnostic) reference fn, and casts the output back. This is enough to
model precision differences today.

SEAM FOR REAL HARDWARE: `torch:<device>:<dtype>` is where the NVIDIA reference and
the Strix Halo / CUDA-Bridge candidate plug in. A TorchBackend would move tensors
to the device, dispatch a per-op torch kernel (keyed by op.name), and return — the
runner, fixtures, diff, tolerances, and coverage all stay identical. It is left
unimplemented on purpose (no torch on this machine); see README.md."""

from harness.precision import cast_tree


class Backend:
    def __init__(self, name, precision):
        self.name = name
        self.precision = precision

    def run(self, op, inputs):
        casted = {k: cast_tree(v, self.precision) for k, v in inputs.items()}
        return cast_tree(op.fn(casted), self.precision)


class PurePyBackend(Backend):
    """Stdlib-only backend. Computes in fp64, rounds I/O to `precision`."""


def get_backend(spec):
    """spec: 'purepy:fp32' | 'purepy:bf16' | 'torch:cuda:bf16' (seam) | ..."""
    parts = spec.split(":")
    kind = parts[0]
    if kind == "purepy":
        return PurePyBackend(spec, parts[1])
    if kind == "torch":
        return _make_torch_backend(spec, parts[1:])
    raise ValueError(f"unknown backend kind: {kind!r}")


class OpNotImplemented(Exception):
    """Raised when a TorchBackend has no impl registered for an op (a TODO)."""


class TorchBackend(Backend):
    """Runs each op via a torch kernel keyed by op.name (see harness/torch_ops.py).

    SEAM for real hardware — same code, different device string:
      torch:cpu:*   portable reference
      torch:mps:*   Apple GPU (this Mac, the dev/proxy path)
      torch:cuda:*  NVIDIA reference (capture) — or, with a ROCm torch build on
                    Strix Halo, the CUDA-Bridge candidate (verify)
    """

    def __init__(self, name, device, dtype_str):
        super().__init__(name, dtype_str)
        import torch
        self.device = device
        self.tdtype = {"fp32": torch.float32, "fp16": torch.float16,
                       "bf16": torch.bfloat16}[dtype_str]

    def run(self, op, inputs):
        from harness.torch_ops import TORCH_OPS
        impl = TORCH_OPS.get(op.name)
        if impl is None:
            raise OpNotImplemented(op.name)
        out = impl(inputs, self.tdtype, self.device)
        return out.detach().to("cpu").float().tolist()


def _make_torch_backend(spec, rest):
    try:
        import torch
    except ImportError:
        raise RuntimeError(
            f"backend '{spec}' needs torch, which is not installed.\n"
            "Install it into a project venv (see OVERNIGHT-PROMPT.md / tests/README.md):\n"
            "  python3 -m venv .venv && . .venv/bin/activate && pip install torch numpy"
        )
    if len(rest) != 2:
        raise ValueError(f"torch backend must be 'torch:<device>:<dtype>', got {spec!r}")
    device, dtype_str = rest
    if dtype_str not in ("fp32", "fp16", "bf16"):
        raise ValueError(f"unknown dtype {dtype_str!r} (use fp32|fp16|bf16)")
    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("device 'mps' requested but torch.backends.mps.is_available() is False")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("device 'cuda' requested but torch.cuda.is_available() is False")
    return TorchBackend(spec, device, dtype_str)
