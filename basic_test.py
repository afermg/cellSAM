"""Standalone smoke test for CellSAM.

Loads the model the same way ``server.py`` does and runs a forward pass on a
small synthetic input. Does NOT spin up the IPC server.

Run from the repo root:
    nix develop --impure --command python basic_test.py

Note: requires ``DEEPCELL_ACCESS_TOKEN`` in the environment. The model
weights are fetched from ``users.deepcell.org`` on first use and cached in
``~/.deepcell/models/``.
"""

import sys

# server.py reads sys.argv[1] at import time; inject a placeholder so
# importing it from this file doesn't crash.
if len(sys.argv) < 2:
    sys.argv.append("ipc:///tmp/cellsam_basic_test.ipc")

import numpy  # noqa: E402

from server import setup  # noqa: E402


def main() -> None:
    processor, info = setup()
    print(f"setup: {info}")
    assert "cuda" in info["device"], (
        f"Not on GPU! info['device']={info['device']!r}"
    )

    # 5-D NCZYX with a single 256x256 RGB image.
    numpy.random.seed(0)
    data = numpy.random.random_sample((1, 3, 1, 256, 256)).astype(numpy.float32)
    out = processor(data)
    print(f"process: {type(out).__name__} shape={out.shape} dtype={out.dtype}")


if __name__ == "__main__":
    main()
