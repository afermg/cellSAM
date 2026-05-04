# /usr/bin/env python
"""
This example uses a server within the environment defined on `https://github.com/afermg/cellSAM.git`.

Run `nix run github:afermg/cellSAM/nahual-wrap -- ipc:///tmp/cellsam.ipc` from any
directory, or `nix develop --impure --command bash -c "python server.py
ipc:///tmp/cellsam.ipc"` from the root of that repository.

Note
----
CellSAM weights are gated; the server requires the ``DEEPCELL_ACCESS_TOKEN``
environment variable to be set when ``setup`` is first called. Tokens are
issued at https://users.deepcell.org. Weights are then cached in
``~/.deepcell/models``.
"""

import numpy

from nahual.process import dispatch_setup_process

setup, process = dispatch_setup_process("cellsam", signature=("dict", "numpy"))
address = "ipc:///tmp/cellsam.ipc"

# %% Load model server-side
parameters = {
    "device": 0,
    # "model_name": "cellsam_general",  # or "cellsam_extra"
    # "bbox_threshold": 0.4,
    # "normalize": True,
    # "postprocess": False,
}
response = setup(parameters, address=address)
print(response)
# Expected: {'device': 'cuda:0', 'model_name': 'cellsam_general', ...}

# %% Define custom data — 5-D NCZYX. CellSAM is 2-D (Z is squeezed server-side).
tile_size = 256
numpy.random.seed(seed=42)
data = numpy.random.random_sample((1, 1, 1, tile_size, tile_size)).astype("float32")
result = process(data, address=address)
print(f"Shape: {result.shape}, Max: {result.max()}, dtype: {result.dtype}")
# Expected: (1, 256, 256) integer instance label map
