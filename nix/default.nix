{
  lib,
  pkgs,
  python3Packages,
}:
let
  callPackage = lib.callPackageWith (pkgs // packages // python3Packages);
  packages = {
    segment-anything = callPackage ./segment_anything.nix { };
    cellsam = callPackage ./cellsam.nix { };
  };
in
packages
