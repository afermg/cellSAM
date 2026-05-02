{
  lib,
  pkgs,
  python3Packages,
}:
# No third-party python packages need building from source any more — the
# ONNX edition uses only nixpkgs-provided libraries (onnxruntime, opencv,
# huggingface-hub, scipy, numpy). Kept as a stub so the existing
# `pkgs.callPackage ./nix { }` import sites elsewhere don't break.
{
}
