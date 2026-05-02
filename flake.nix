{
  description = "Nahual server for CellSAM (ONNX edition; no DEEPCELL_ACCESS_TOKEN required).";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    systems.url = "github:nix-systems/default";
    flake-utils.url = "github:numtide/flake-utils";
    flake-utils.inputs.systems.follows = "systems";
    pynng-flake.url = "github:afermg/pynng";
    pynng-flake.inputs.nixpkgs.follows = "nixpkgs";
  };

  outputs =
    {
      self,
      nixpkgs,
      flake-utils,
      systems,
      ...
    }@inputs:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = import nixpkgs {
          system = system;
          config = {
            allowUnfree = true;
            cudaSupport = true;
          };
        };
        nahualPkg = pkgs.python3.pkgs.callPackage ./nix/nahual.nix {
          pynng = inputs.pynng-flake.packages.${system}.pynng;
        };

        baseDeps = pp: [
          nahualPkg
          pp.onnxruntime
          pp.numpy
          pp.opencv-python
          pp.scipy
          pp.huggingface-hub
          pp.loguru
        ];
      in
      with pkgs;
      rec {
        formatter = pkgs.alejandra;

        packages = {
          nahual = nahualPkg;
        };

        apps.default =
          let
            python_with_pkgs = python3.withPackages baseDeps;
            runServer = pkgs.writeScriptBin "runserver.sh" ''
              #!${pkgs.bash}/bin/bash
              export CUDA_PATH=${pkgs.cudaPackages.cudatoolkit}
              export LD_LIBRARY_PATH=${pkgs.cudaPackages.cudatoolkit}/lib:${pkgs.cudaPackages.cudnn}/lib:$LD_LIBRARY_PATH
              ${python_with_pkgs}/bin/python ${self}/server.py ''${@:-"ipc:///tmp/cellsam.ipc"}
            '';
          in
          {
            type = "app";
            program = "${runServer}/bin/runserver.sh";
          };

        devShells = {
          default =
            let
              python_with_pkgs = python3.withPackages (
                pp:
                baseDeps pp
                ++ [
                  pp.tifffile
                  pp.scikit-image
                  pp.pyyaml
                ]
              );
            in
            mkShell {
              packages = [
                python_with_pkgs
                pkgs.cudaPackages.cudatoolkit
                pkgs.cudaPackages.cudnn
              ];
              shellHook = ''
                export CUDA_PATH=${pkgs.cudaPackages.cudatoolkit}
                export LD_LIBRARY_PATH=${pkgs.cudaPackages.cudatoolkit}/lib:${pkgs.cudaPackages.cudnn}/lib:$LD_LIBRARY_PATH
                export PYTHONPATH=${python_with_pkgs}/${python_with_pkgs.sitePackages}
                export PYTHONDONTWRITEBYTECODE=1
              '';
            };
        };
      }
    );
}
