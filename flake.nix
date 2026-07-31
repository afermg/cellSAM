{
  description = "Nahual server for CellSAM (ONNX edition; no access token required)";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    systems.url = "github:nix-systems/default";
    flake-utils.url = "github:numtide/flake-utils";
    flake-utils.inputs.systems.follows = "systems";
    nahual-flake.url = "github:afermg/nahual";
    nahual-flake.inputs.nixpkgs.follows = "nixpkgs";
  };

  outputs = {
    self,
    nixpkgs,
    flake-utils,
    ...
  } @ inputs:
    flake-utils.lib.eachDefaultSystem (
      system: let
        pkgs = import nixpkgs {
          inherit system;
          config = {
            allowUnfree = true;
            cudaSupport = true;
          };
        };
        python_with_pkgs = pkgs.python3.withPackages (pp: [
          inputs.nahual-flake.packages.${system}.nahual
          pp.onnxruntime
          pp.numpy
          pp.opencv-python
          pp.scipy
          pp.huggingface-hub
          pp.loguru
        ]);
        runServer = pkgs.writeScriptBin "nahual-cellsam" ''
          #!${pkgs.bash}/bin/bash
          export CUDA_PATH=${pkgs.cudaPackages.cudatoolkit}
          export LD_LIBRARY_PATH=${pkgs.cudaPackages.cudatoolkit}/lib:${pkgs.cudaPackages.cudnn}/lib:''${LD_LIBRARY_PATH:-}
          exec ${python_with_pkgs}/bin/python ${self}/server.py \
            "''${1:-tcp://0.0.0.0:5555}"
        '';
        cellsamApp = {
          type = "app";
          program = "${runServer}/bin/nahual-cellsam";
        };
      in
        with pkgs; rec {
          packages = pkgs.lib.optionalAttrs pkgs.stdenv.hostPlatform.isLinux {
            oci-image = import ./nix/oci-image.nix {
              inherit pkgs;
              name = "cellsam";
              title = "Nahual CellSAM";
              description = "CellSAM ONNX instance segmentation served through Nahual";
              source = "https://github.com/afermg/cellSAM";
              revision = self.rev or self.dirtyRev or "unknown";
              server = runServer;
              entrypoint = cellsamApp.program;
            };
          };
          inherit python_with_pkgs;
          formatter = pkgs.alejandra;
          scripts.runServer = runServer;
          apps = rec {
            cellsam = cellsamApp;
            default = cellsam;
          };
          devShells.default = mkShell {
            packages = [
              python_with_pkgs
              pkgs.cudaPackages.cudatoolkit
              pkgs.cudaPackages.cudnn
              python3Packages.tifffile
              python3Packages.scikit-image
              python3Packages.pyyaml
            ];
            shellHook = ''
              export CUDA_PATH=${pkgs.cudaPackages.cudatoolkit}
              export LD_LIBRARY_PATH=${pkgs.cudaPackages.cudatoolkit}/lib:${pkgs.cudaPackages.cudnn}/lib:$LD_LIBRARY_PATH
              export PYTHONDONTWRITEBYTECODE=1
            '';
          };
        }
    );
}
