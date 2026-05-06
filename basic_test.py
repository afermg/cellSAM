"""Standalone smoke test for CellSAM (ONNX edition).

Loads the three ONNX sessions the same way ``server.py`` does, downloads
weights from HuggingFace if needed (no auth), and runs a forward pass on a
small synthetic input. Does NOT spin up the IPC server.

Run from the repo root:
    nix develop --impure --command python basic_test.py
"""

import importlib.util
import os
import sys

# Drop the repo root from sys.path BEFORE importing cellSAM — otherwise the
# in-tree `cellSAM/` source dir shadows the nix-built package.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p) not in {_HERE, ""}]

# server.py reads sys.argv[1] at import time; inject a placeholder so
# importing it from this file doesn't crash.
if len(sys.argv) < 2:
    sys.argv.append("ipc:///tmp/cellsam_basic_test.ipc")

import numpy  # noqa: E402

# Load `server.py` directly via importlib so we never need to put _HERE back
# on sys.path (which would re-trigger the cellSAM source-dir shadow).
_spec = importlib.util.spec_from_file_location("server", os.path.join(_HERE, "server.py"))
server = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(server)
setup = server.setup


def main() -> None:
    processor, info = setup()
    print(f"setup: {info}")
    assert "device" in info, f"missing device in info: {info}"

    numpy.random.seed(0)
    # 5-D NCZYX with a single 256x256 1-channel image; replicated to 3 chans
    # internally by _to_chw_3.
    data = numpy.random.random_sample((1, 1, 1, 256, 256)).astype(numpy.float32)
    out = processor(data)
    print(f"process: {type(out).__name__} shape={out.shape} dtype={out.dtype} max_label={out.max()}")
    assert out.shape == (1, 256, 256), f"unexpected shape {out.shape}"
    assert out.dtype == numpy.int32, f"unexpected dtype {out.dtype}"


if __name__ == "__main__":
    main()
