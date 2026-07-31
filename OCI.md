# CellSAM Nahual OCI image

Build the reproducible archive and load it into Podman or Docker:

```console
nix build .#oci-image
podman load < result                         # or: docker load < result
```

The image is tagged `nahual/cellsam:local` and listens on TCP port 5555. The
public ONNX model is downloaded from `keejkrej/cellsam-onnx` on first setup, so
persist `/tmp/nahual` as a model cache:

```console
podman run --rm --device nvidia.com/gpu=all -p 5555:5555 \
  -v nahual-cellsam-cache:/tmp/nahual nahual/cellsam:local
```

For Docker, replace the CDI device option with `--gpus all`. CPU operation is
supported. With Nahual, NumPy, and Pillow installed on the host, run pretrained
end-to-end inference on the repository's example microscopy image:

```console
NAHUAL_DEVICE=cpu python oci/smoke_test.py
```

This ONNX edition uses public Hugging Face weights and does not require a
DeepCell access token.
