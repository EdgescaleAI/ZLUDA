"""Golden-output fixtures + provenance. A fixture stores the reference output and a
manifest (op, case, seed, backend, dtype, capture time) so every comparison is
reproducible and traceable to the exact reference that produced it."""

import json
import os
import time

FIX_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "fixtures")


def _path(op_name, case_id):
    return os.path.join(FIX_DIR, f"{op_name}__{case_id}.json")


def save_fixture(op_name, case_id, output, manifest):
    os.makedirs(FIX_DIR, exist_ok=True)
    manifest = dict(manifest, captured_at=time.time())
    with open(_path(op_name, case_id), "w") as f:
        json.dump({"manifest": manifest, "output": output}, f)


def load_fixture(op_name, case_id):
    p = _path(op_name, case_id)
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)
