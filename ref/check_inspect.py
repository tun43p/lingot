"""Check that `lingot inspect` lists the same tensors as llama.cpp's GGUF reader.

Usage: python ref/check_inspect.py models/*.gguf   (after `cargo build --release`)
Compares name, type, dims, file offset and byte size of every tensor, in order.
"""

import subprocess
import sys
from pathlib import Path

from gguf import GGUFReader

LINGOT = Path(__file__).parent.parent / "target/release/lingot"

for path in sys.argv[1:]:
    out = subprocess.run(
        [LINGOT, "inspect", path], check=True, capture_output=True, text=True
    ).stdout
    ours = [
        tuple(line.strip().split("\t"))
        for line in out.split("\ntensors:\n")[1].splitlines()
    ]
    theirs = [
        (
            t.name,
            t.tensor_type.name,
            "x".join(str(int(d)) for d in t.shape),
            str(t.data_offset),
            str(t.n_bytes),
        )
        for t in GGUFReader(path).tensors
    ]
    assert len(ours) == len(theirs), f"{path}: {len(ours)} tensors vs {len(theirs)}"
    for a, b in zip(ours, theirs):
        assert a == b, f"{path}: {a} != {b}"
    print(f"{path}: {len(ours)} tensors match")
