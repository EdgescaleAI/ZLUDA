"""Dtype simulation. Python floats are fp64; we round to fp32 / bf16 to model the
precision a real backend would compute at. bf16 uses round-to-nearest-even on the
top 16 bits of the fp32 representation — the same thing the hardware does."""

import struct


def round_fp32(x):
    return struct.unpack("f", struct.pack("f", float(x)))[0]


def round_bf16(x):
    x = float(x)
    if x != x or x in (float("inf"), float("-inf")):
        return x
    u = struct.unpack(">I", struct.pack(">f", x))[0]
    bias = 0x7FFF + ((u >> 16) & 1)          # round-to-nearest-even
    u = (u + bias) & 0xFFFFFFFF
    u &= 0xFFFF0000                           # truncate low 16 bits
    return struct.unpack(">f", struct.pack(">I", u))[0]


CASTERS = {
    "fp64": float,
    "fp32": round_fp32,
    "bf16": round_bf16,
    "int": lambda x: float(int(round(x))),
}


def cast_tree(x, dtype):
    """Recursively cast a scalar / nested-list tensor to `dtype`."""
    if isinstance(x, list):
        return [cast_tree(e, dtype) for e in x]
    return CASTERS[dtype](x)
