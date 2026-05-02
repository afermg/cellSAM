"""Nahual server for CellSAM (ONNX edition).

This wrap drops the upstream PyTorch path (which required
``DEEPCELL_ACCESS_TOKEN``) and runs CellSAM via the ONNX export hosted at
https://huggingface.co/keejkrej/cellsam-onnx — three onnxruntime sessions
chained together in the SAM-style:

    image_encoder(image_1024) -> image_embeddings (1, 256, 64, 64)
    cellfinder(image_1024)    -> pred_logits, pred_boxes (1, 3500, *)
    mask_decoder(emb, box, pe) -> low-res mask (1, 1, 256, 256), iou

Weights are downloaded on first ``setup()`` into ``~/.cache/cellsam-onnx/``
via ``huggingface_hub.snapshot_download``. No auth required.

Run with:
    nix run --impure . -- ipc:///tmp/cellsam.ipc
or:
    python server.py ipc:///tmp/cellsam.ipc
"""

import os
import sys
from functools import partial
from pathlib import Path
from typing import Callable

import cv2
import numpy
import onnxruntime as ort
import pynng
import trio
from loguru import logger
from nahual.server import responder

# server.py captures argv[1] at import time; basic_test.py injects a
# placeholder before importing this module.
address = sys.argv[1]

logger.add(address.split("/")[-1])


# ---------------------------------------------------------------------------
# Constants pulled from upstream cellSAM / cellsam-rs
# ---------------------------------------------------------------------------
HF_REPO_ID = "keejkrej/cellsam-onnx"
DEFAULT_WEIGHTS_DIR = Path(os.path.expanduser("~/.cache/cellsam-onnx"))
REQUIRED_FILES = (
    "image_encoder.onnx",
    "image_encoder.onnx.data",
    "cellfinder.onnx",
    "cellfinder.onnx.data",
    "mask_decoder.onnx",
    "mask_decoder.onnx.data",
    "image_pe.npy",
)
SAM_PIXEL_MEAN = numpy.array([123.675, 116.28, 103.53], dtype=numpy.float32) / 255.0
SAM_PIXEL_STD = numpy.array([58.395, 57.12, 57.375], dtype=numpy.float32) / 255.0
IMAGENET_MEAN = numpy.array([0.485, 0.456, 0.406], dtype=numpy.float32)
IMAGENET_STD = numpy.array([0.229, 0.224, 0.225], dtype=numpy.float32)
SAM_INPUT_SIZE = 1024
LOW_RES_MASK_SIZE = 256


# ---------------------------------------------------------------------------
# Weight management
# ---------------------------------------------------------------------------
def _ensure_weights(weights_dir: Path) -> Path:
    """Download the four ONNX assets (+ external .data buffers) if missing."""
    weights_dir.mkdir(parents=True, exist_ok=True)
    have_all = all((weights_dir / f).exists() for f in REQUIRED_FILES)
    if have_all:
        return weights_dir

    logger.info(f"Fetching {HF_REPO_ID} into {weights_dir}")
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=HF_REPO_ID,
        local_dir=str(weights_dir),
        allow_patterns=list(REQUIRED_FILES),
    )
    missing = [f for f in REQUIRED_FILES if not (weights_dir / f).exists()]
    if missing:
        raise RuntimeError(
            f"Download finished but {missing!r} are still missing under {weights_dir}"
        )
    return weights_dir


# ---------------------------------------------------------------------------
# Pre/post-processing
# ---------------------------------------------------------------------------
def _to_chw_3(img_hwc: numpy.ndarray) -> numpy.ndarray:
    """Convert an (H, W, C) numpy float image into a normalised (3, H, W) CHW.

    - Each channel is min-max normalised independently (matches CellSAM's
      ``normalize_image`` per-channel behaviour).
    - For C == 1 the channel is replicated 3 times.
    - For C == 2 a leading blank channel is prepended (blank, ch0, ch1).
    - For C >= 3 the first three channels are used; if either of the
      green/blue channels is all zero, it is filled from the other (matches
      CellSAM's ``ToRGB`` convention).
    """
    if img_hwc.ndim != 3:
        raise ValueError(f"Expected (H, W, C); got {img_hwc.shape!r}")
    h, w, c = img_hwc.shape
    img = img_hwc.astype(numpy.float32, copy=True)
    if c == 1:
        ch = img[..., 0]
        ch = _minmax(ch)
        return numpy.stack([ch, ch, ch], axis=0)
    if c == 2:
        zero = numpy.zeros((h, w), dtype=numpy.float32)
        chans = [zero, _minmax(img[..., 0]), _minmax(img[..., 1])]
    else:
        chans = [_minmax(img[..., i]) for i in range(3)]
        # ToRGB: backfill blanks
        if numpy.all(chans[1] == 0):
            chans[1] = chans[2].copy()
        if numpy.all(chans[2] == 0):
            chans[2] = chans[1].copy()
    return numpy.stack(chans, axis=0)


def _minmax(arr: numpy.ndarray) -> numpy.ndarray:
    lo = float(arr.min())
    hi = float(arr.max())
    if hi - lo < 1e-10:
        return numpy.zeros_like(arr, dtype=numpy.float32)
    return ((arr - lo) / (hi - lo)).astype(numpy.float32)


def _resize_chw(chw: numpy.ndarray, size: int) -> numpy.ndarray:
    """Bilinear resize a (3, H, W) array to (3, size, size)."""
    c, h, w = chw.shape
    if (h, w) == (size, size):
        return chw.astype(numpy.float32, copy=False)
    out = numpy.empty((c, size, size), dtype=numpy.float32)
    for i in range(c):
        out[i] = cv2.resize(chw[i], (size, size), interpolation=cv2.INTER_LINEAR)
    return out


def _sam_normalize(chw: numpy.ndarray) -> numpy.ndarray:
    """SAM mean/std normalise (per ImageNet statistics, /255)."""
    out = chw.astype(numpy.float32, copy=True)
    out -= SAM_PIXEL_MEAN[:, None, None]
    out /= SAM_PIXEL_STD[:, None, None]
    return out


def _cellfinder_preprocess(chw: numpy.ndarray) -> numpy.ndarray:
    """Apply CellSAM's bbox-detector preprocessing to a (3, 1024, 1024) image.

    Matches the chain used by upstream ``sam_bbox_preprocessing``:
      - PercentileThreshold on non-zero values (0 .. 99.5%)
      - ImageNet normalize
      - kornia normalize_min_max (per-tensor min-max to [0, 1])
      - ToRGB blank-channel back-fill
    """
    out = chw.astype(numpy.float32, copy=True)
    nonzero = out[out != 0]
    if nonzero.size:
        lo = float(nonzero.min())
        hi = float(numpy.percentile(nonzero, 99.5))
        rng = hi - lo
        if rng > 1e-10:
            mask = out != 0
            out[mask] = numpy.clip((out[mask] - lo) / rng, 0.0, 1.0)
    out -= IMAGENET_MEAN[:, None, None]
    out /= IMAGENET_STD[:, None, None]
    lo = float(out.min())
    hi = float(out.max())
    if hi - lo > 1e-10:
        out = (out - lo) / (hi - lo)
    if out[1].max() == 0:
        out[1] = out[2]
    if out[2].max() == 0:
        out[2] = out[1]
    return out


def _decode_boxes(
    pred_logits: numpy.ndarray,
    pred_boxes: numpy.ndarray,
    bbox_threshold: float,
) -> numpy.ndarray:
    """Filter cellfinder outputs into xyxy boxes at 1024 scale.

    pred_logits: (1, N, num_classes) raw logits; we sigmoid + take the per-row
    max class score and keep boxes whose score exceeds ``bbox_threshold``.

    pred_boxes: (1, N, 4) cxcywh in normalised coords [0, 1]; converted to
    xyxy at 1024 px.
    """
    logits = pred_logits[0]
    boxes = pred_boxes[0]
    scores = 1.0 / (1.0 + numpy.exp(-logits))
    keep = scores.max(axis=-1) > bbox_threshold
    sel = boxes[keep] * SAM_INPUT_SIZE
    cx, cy, bw, bh = sel[:, 0], sel[:, 1], sel[:, 2], sel[:, 3]
    x1 = cx - bw / 2.0
    y1 = cy - bh / 2.0
    x2 = cx + bw / 2.0
    y2 = cy + bh / 2.0
    return numpy.stack([x1, y1, x2, y2], axis=-1).astype(numpy.float32)


def _upscale_mask(low_res: numpy.ndarray, h: int, w: int, mask_threshold: float) -> numpy.ndarray:
    """Resize a 256x256 mask logit map to (h, w) and threshold to bool."""
    up = cv2.resize(low_res, (w, h), interpolation=cv2.INTER_LINEAR)
    if mask_threshold == 0.5:
        thr = 0.0  # logit 0 == sigmoid 0.5
    else:
        thr = float(numpy.log(mask_threshold / (1.0 - mask_threshold)))
    return up > thr


def _merge_into_label_map(masks: list[numpy.ndarray], h: int, w: int, min_size: int) -> numpy.ndarray:
    """Stack boolean masks into an int32 label map (later masks overwrite earlier)."""
    out = numpy.zeros((h, w), dtype=numpy.int32)
    label = 0
    for m in masks:
        if int(m.sum()) < min_size:
            continue
        label += 1
        out[m] = label
    return out


# ---------------------------------------------------------------------------
# Setup / process
# ---------------------------------------------------------------------------
def setup(
    weights_dir: str | None = None,
    device: int = 0,
    providers: list[str] | None = None,
    bbox_threshold: float = 0.4,
    iou_threshold: float = 0.5,
    mask_threshold: float = 0.5,
    min_size: int = 25,
) -> tuple[Callable, dict]:
    """Build three onnxruntime sessions and return (processor, info).

    Parameters
    ----------
    weights_dir : str, optional
        Where to look for / download the ONNX weights. Defaults to
        ``~/.cache/cellsam-onnx``.
    device : int
        CUDA device index used in the ``device_id`` provider option when
        CUDAExecutionProvider is available.
    providers : list[str], optional
        Explicit onnxruntime provider order. If unset we try CUDA first,
        then CPU.
    bbox_threshold : float
        Confidence cutoff for the cellfinder logits. Lower keeps more cells.
    iou_threshold : float
        Drop SAM-decoded masks below this IoU prediction.
    mask_threshold : float
        Sigmoid threshold for binarising the low-res mask logits.
    min_size : int
        Minimum mask area in pixels (otherwise dropped).
    """
    weights_path = Path(weights_dir).expanduser() if weights_dir else DEFAULT_WEIGHTS_DIR
    weights_path = _ensure_weights(weights_path)

    available = ort.get_available_providers()
    if providers is None:
        providers = []
        if "CUDAExecutionProvider" in available:
            providers.append(("CUDAExecutionProvider", {"device_id": int(device)}))
        providers.append("CPUExecutionProvider")

    sess_opts = ort.SessionOptions()
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    logger.info(f"Loading ONNX sessions from {weights_path} with providers={providers}")
    encoder = ort.InferenceSession(
        str(weights_path / "image_encoder.onnx"),
        sess_options=sess_opts,
        providers=providers,
    )
    cellfinder = ort.InferenceSession(
        str(weights_path / "cellfinder.onnx"),
        sess_options=sess_opts,
        providers=providers,
    )
    decoder = ort.InferenceSession(
        str(weights_path / "mask_decoder.onnx"),
        sess_options=sess_opts,
        providers=providers,
    )

    image_pe = numpy.load(str(weights_path / "image_pe.npy")).astype(numpy.float32)
    if image_pe.shape != (1, 256, 64, 64):
        raise RuntimeError(f"Unexpected image_pe shape {image_pe.shape}")

    resolved = encoder.get_providers()
    if "CUDAExecutionProvider" in resolved:
        device_str = f"cuda:{int(device)}"
    else:
        device_str = "cpu"

    info = {
        "device": device_str,
        "providers": resolved,
        "weights_dir": str(weights_path),
        "bbox_threshold": bbox_threshold,
        "iou_threshold": iou_threshold,
        "mask_threshold": mask_threshold,
        "min_size": min_size,
        "hf_repo_id": HF_REPO_ID,
    }

    processor = partial(
        process,
        encoder=encoder,
        cellfinder=cellfinder,
        decoder=decoder,
        image_pe=image_pe,
        bbox_threshold=bbox_threshold,
        iou_threshold=iou_threshold,
        mask_threshold=mask_threshold,
        min_size=min_size,
    )
    return processor, info


def _segment_one(
    img_hwc: numpy.ndarray,
    encoder: ort.InferenceSession,
    cellfinder: ort.InferenceSession,
    decoder: ort.InferenceSession,
    image_pe: numpy.ndarray,
    bbox_threshold: float,
    iou_threshold: float,
    mask_threshold: float,
    min_size: int,
) -> numpy.ndarray:
    """Full SAM-style chain on a single (H, W, C) image. Returns (H, W) int32."""
    h, w = img_hwc.shape[:2]
    chw = _to_chw_3(img_hwc)
    chw_1024 = _resize_chw(chw, SAM_INPUT_SIZE)
    enc_input = _sam_normalize(chw_1024)[None, ...]
    cf_input = _cellfinder_preprocess(chw_1024)[None, ...]

    enc_in_name = encoder.get_inputs()[0].name
    cf_in_name = cellfinder.get_inputs()[0].name

    embeddings = encoder.run(None, {enc_in_name: enc_input.astype(numpy.float32)})[0]
    pred_logits, pred_boxes = cellfinder.run(
        None, {cf_in_name: cf_input.astype(numpy.float32)}
    )

    boxes = _decode_boxes(pred_logits, pred_boxes, bbox_threshold)
    if boxes.size == 0:
        return numpy.zeros((h, w), dtype=numpy.int32)

    masks: list[numpy.ndarray] = []
    for box in boxes:
        feed = {
            "image_embeddings": embeddings.astype(numpy.float32),
            "box": box.reshape(1, 1, 4).astype(numpy.float32),
            "image_pe": image_pe,
        }
        mask_out, iou_out = decoder.run(None, feed)
        if float(iou_out[0, 0]) < iou_threshold:
            continue
        low_res = mask_out[0, 0]  # (256, 256)
        upscaled = _upscale_mask(low_res, h, w, mask_threshold)
        if upscaled.any():
            masks.append(upscaled)

    return _merge_into_label_map(masks, h, w, min_size)


def process(
    pixels: numpy.ndarray,
    encoder: ort.InferenceSession,
    cellfinder: ort.InferenceSession,
    decoder: ort.InferenceSession,
    image_pe: numpy.ndarray,
    bbox_threshold: float,
    iou_threshold: float,
    mask_threshold: float,
    min_size: int,
) -> numpy.ndarray:
    """Run CellSAM segmentation on a 5-D NCZYX numpy array.

    Z is squeezed (max projection if Z > 1). Returns an int32 instance label
    map of shape ``(N, H, W)`` (0 == background).
    """
    if pixels.ndim != 5:
        raise ValueError(f"Expected NCZYX (5D) array, got shape {pixels.shape}")
    n, c, z, h, w = pixels.shape
    if z != 1:
        pixels = pixels.max(axis=2, keepdims=True)
    arr = pixels[:, :, 0, :, :]  # (N, C, H, W)
    # to (N, H, W, C)
    arr = numpy.transpose(arr, (0, 2, 3, 1)).astype(numpy.float32)

    out = numpy.empty((n, h, w), dtype=numpy.int32)
    for i in range(n):
        out[i] = _segment_one(
            arr[i],
            encoder=encoder,
            cellfinder=cellfinder,
            decoder=decoder,
            image_pe=image_pe,
            bbox_threshold=bbox_threshold,
            iou_threshold=iou_threshold,
            mask_threshold=mask_threshold,
            min_size=min_size,
        )
    return out


async def main():
    with pynng.Rep0(listen=address, recv_timeout=300) as sock:
        print(f"CellSAM (ONNX) server listening on {address}", flush=True)
        async with trio.open_nursery() as nursery:
            nursery.start_soon(partial(responder, setup=setup), sock)


if __name__ == "__main__":
    try:
        trio.run(main)
    except KeyboardInterrupt:
        pass
