{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    systems.url = "github:nix-systems/default";
    flake-utils.url = "github:numtide/flake-utils";
    flake-utils.inputs.systems.follows = "systems";
    nahual-flake.url = "github:afermg/nahual";
    nahual-flake.inputs.nixpkgs.follows = "nixpkgs";
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
        nahualPkg = inputs.nahual-flake.packages.${system}.nahual;
        modelPkgs = pkgs.callPackage ./nix { };
      in
      with pkgs;
      rec {
        formatter = pkgs.alejandra;

        packages = modelPkgs;

        apps.default =
          let
            python_with_pkgs = python3.withPackages (pp: [
              nahualPkg
              modelPkgs.cellsam
              pp.torch
              pp.torchvision
              pp.numpy
              pp.scikit-image
              pp.scikit-learn
              pp.scipy
              pp.pyyaml
              pp.tqdm
              pp.requests
              pp.kornia
              pp.dask
              pp.distributed
              pp.dask-image
              pp.pyarrow
              pp.loguru
            ]);
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
              python_with_pkgs = python3.withPackages (pp: [
                nahualPkg
                modelPkgs.cellsam
                pp.torch
                pp.torchvision
                pp.numpy
                pp.scikit-image
                pp.scikit-learn
                pp.scipy
                pp.pyyaml
                pp.tqdm
                pp.requests
                pp.kornia
                pp.dask
                pp.distributed
                pp.dask-image
                pp.pyarrow
                pp.loguru
                # Dev extras
                pp.tifffile
              ]);
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
