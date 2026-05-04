# /usr/bin/env python
"""CellSAM (ONNX edition) Nahual client example.

Server-side: this points at the ONNX rewrite of the cellSAM Nahual wrap
which uses the weights published at
https://huggingface.co/keejkrej/cellsam-onnx (image_encoder.onnx,
cellfinder.onnx, mask_decoder.onnx, image_pe.npy). Unlike the original
cellSAM PyTorch wrap, this server does not require ``DEEPCELL_ACCESS_TOKEN``
— weights are downloaded from HuggingFace on first ``setup()``.

Run:
    nix run github:afermg/cellSAM/nahual-wrap-onnx -- ipc:///tmp/cellsam.ipc
from the cellSAM nahual-wrap-onnx branch, then run this script.
"""

import numpy

from nahual.process import dispatch_setup_process

setup, process = dispatch_setup_process("cellsam", signature=("dict", "numpy"))
address = "ipc:///tmp/cellsam.ipc"

# %% Load the three ONNX sessions server-side. Weights are pulled from HF
# on first call into ~/.cache/cellsam-onnx if not already present.
parameters = {"device": 0, "bbox_threshold": 0.4, "mask_threshold": 0.5}
response = setup(parameters, address=address)
print(response)
# Loaded model with parameters {'setup': {'device': 'cuda:0', 'providers': "['CUDAExecutionProvider', 'CPUExecutionProvider']", ...}}

# %% Define custom data: 5-D NCZYX, single 256x256 image, 1 channel.
tile_size = 256
numpy.random.seed(42)
data = numpy.random.random_sample((1, 1, 1, tile_size, tile_size)).astype("float32")
result = process(data, address=address)
print(f"Shape: {result.shape}, dtype: {result.dtype}, Max label: {result.max()}")
# Shape: (1, 256, 256), dtype: int32, Max label: 0   # noise -> no cells
