"""Nahual server for CellSAM.

CellSAM is a Segment Anything-derived foundation model for cell segmentation.
``setup`` loads the pretrained CellSAM weights (downloaded on first use from
``users.deepcell.org``; requires ``DEEPCELL_ACCESS_TOKEN``) and moves the
model to the chosen GPU. ``process`` accepts a 5-D NCZYX numpy array, runs
per-image inference, and returns an instance label map of shape ``(N, H, W)``.

Run with:
    nix run --impure . -- ipc:///tmp/cellsam.ipc
or:
    python server.py ipc:///tmp/cellsam.ipc
"""

import sys
from functools import partial
from typing import Callable

import numpy
import pynng
import torch
import trio
from loguru import logger
from nahual.server import responder

from cellSAM import get_model, segment_cellular_image

# server.py captures argv[1] at import time; basic_test.py injects a
# placeholder before importing this module.
address = sys.argv[1]

logger.add(address.split("/")[-1])


def setup(
    device: int | None = None,
    model_name: str = "cellsam_general",
    version: str | None = None,
    bbox_threshold: float = 0.4,
    normalize: bool = True,
    postprocess: bool = False,
    remove_boundaries: bool = False,
    fast: bool = False,
) -> tuple[Callable, dict]:
    """Load CellSAM and return (processor_partial, info_dict).

    Parameters
    ----------
    device : int, optional
        CUDA device id; default 0. Falls back to CPU if CUDA is unavailable.
    model_name : str
        ``"cellsam_general"`` (paper-faithful) or ``"cellsam_extra"``
        (extended training data).
    version : str, optional
        Model release version. Defaults to latest (currently ``"1.2"``).
    bbox_threshold : float
        Confidence threshold for the bounding-box detector. Lower values are
        more permissive. Default 0.4.
    normalize : bool
        Apply CellSAM's percentile + CLAHE normalisation per image.
    postprocess : bool
        Apply opening/closing/dilation post-processing. Slower; recommended
        only for noisy images.
    remove_boundaries : bool
        Strip a one-pixel boundary from each segmented cell.
    fast : bool
        Use the alpha batched-inference path. May produce slightly different
        results.
    """
    if device is None:
        device = 0
    if torch.cuda.is_available():
        torch_device = torch.device(int(device))
        device_str = f"cuda:{int(device)}"
    else:
        torch_device = torch.device("cpu")
        device_str = "cpu"

    logger.info(f"Loading CellSAM model={model_name} version={version} -> {device_str}")
    model = get_model(model=model_name, version=version)
    model = model.to(torch_device).eval()
    model.bbox_threshold = bbox_threshold

    info = {
        "device": device_str,
        "model_name": model_name,
        "version": version or "1.2",
        "bbox_threshold": bbox_threshold,
        "normalize": normalize,
        "postprocess": postprocess,
        "remove_boundaries": remove_boundaries,
        "fast": fast,
    }

    processor = partial(
        process,
        model=model,
        device_str=device_str,
        normalize=normalize,
        postprocess=postprocess,
        remove_boundaries=remove_boundaries,
        bbox_threshold=bbox_threshold,
        fast=fast,
    )
    return processor, info


def process(
    pixels: numpy.ndarray,
    model,
    device_str: str,
    normalize: bool,
    postprocess: bool,
    remove_boundaries: bool,
    bbox_threshold: float,
    fast: bool,
) -> numpy.ndarray:
    """Run CellSAM segmentation on a 5-D NCZYX numpy array.

    Z is squeezed (CellSAM is 2-D). For each item in the batch (N) we call
    ``segment_cellular_image`` once. The returned array has shape
    ``(N, H, W)`` of integer instance labels (``0`` is background).
    """
    if pixels.ndim != 5:
        raise ValueError(f"Expected NCZYX (5D) array, got shape {pixels.shape}")

    n, c, z, h, w = pixels.shape
    if z != 1:
        # CellSAM is 2-D; reduce Z by taking the max projection.
        pixels = pixels.max(axis=2, keepdims=True)
    # Drop Z -> (N, C, H, W)
    arr = pixels[:, :, 0, :, :]

    # CellSAM expects (H, W, C) per image with C in {1, 3} (channel-last).
    arr = numpy.transpose(arr, (0, 2, 3, 1)).astype(numpy.float32)

    masks = []
    for i in range(n):
        img = arr[i]
        # Pad/trim to 1 or 3 channels.
        ch = img.shape[-1]
        if ch == 2:
            # Pad a leading blank channel so ordering matches the
            # (blank, nuclear, membrane) multiplexed convention.
            blank = numpy.zeros_like(img[..., :1])
            img = numpy.concatenate([blank, img], axis=-1)
        elif ch > 3:
            img = img[..., :3]
        # Now img is (H, W, 1) or (H, W, 3); CellSAM accepts both.

        mask, _, _ = segment_cellular_image(
            img,
            model=model,
            normalize=normalize,
            postprocess=postprocess,
            remove_boundaries=remove_boundaries,
            bbox_threshold=bbox_threshold,
            fast=fast,
            device=device_str,
        )
        masks.append(numpy.asarray(mask, dtype=numpy.int32))

    return numpy.stack(masks, axis=0)


async def main():
    with pynng.Rep0(listen=address, recv_timeout=300) as sock:
        print(f"CellSAM server listening on {address}", flush=True)
        async with trio.open_nursery() as nursery:
            nursery.start_soon(partial(responder, setup=setup), sock)


if __name__ == "__main__":
    try:
        trio.run(main)
    except KeyboardInterrupt:
        pass
