#!/usr/bin/env python3
"""End-to-end pretrained inference against the CellSAM OCI container."""

import json
import os
from pathlib import Path

os.environ.setdefault("NAHUAL_IPC_TIMEOUT_MS", "1800000")

import numpy as np
from nahual.process import dispatch_setup_process


def _input_image() -> tuple[np.ndarray, str, bool]:
    configured = os.environ.get("CELLSAM_TEST_IMAGE")
    candidates = [
        Path(configured) if configured else None,
        Path(__file__).resolve().parents[1] / "sample_imgs" / "ep_micro.png",
        Path(__file__).with_name("ep_micro.png"),
    ]
    for path in candidates:
        if path is not None and path.exists():
            from PIL import Image

            image = np.asarray(Image.open(path).convert("L"), dtype=np.float32)
            return image[None, None, None, :, :], str(path), True
    image = np.random.default_rng(42).random((1, 1, 1, 256, 256), dtype=np.float32)
    return image, "synthetic", False


def main() -> None:
    address = os.environ.get("NAHUAL_ADDRESS", "tcp://127.0.0.1:5555")
    device = os.environ.get("NAHUAL_DEVICE", "cpu")
    pixels, source, real_image = _input_image()
    setup, process = dispatch_setup_process("cellsam")
    info = setup(
        {
            "device": device,
            "bbox_threshold": 0.4 if real_image else 0.99,
            "iou_threshold": 0.5,
            "mask_threshold": 0.5,
            "min_size": 25,
        },
        address=address,
    )
    result = process(pixels, address=address)
    assert info["device"] == device, info
    assert info["hf_repo_id"] == "keejkrej/cellsam-onnx", info
    assert result.shape == (pixels.shape[0], pixels.shape[3], pixels.shape[4])
    assert np.issubdtype(result.dtype, np.integer), result.dtype
    assert (result >= 0).all()
    if real_image:
        assert result.max() > 0, "No cells detected in the example image"
    print(
        json.dumps(
            {
                "setup": info,
                "source": source,
                "shape": list(result.shape),
                "instances": int(result.max()),
            }
        )
    )


if __name__ == "__main__":
    main()
