{
  lib,
  buildPythonPackage,
  setuptools,
  # Deps
  torch,
  torchvision,
  numpy,
  pyyaml,
  scikit-image,
  scikit-learn,
  scipy,
  kornia,
  requests,
  tqdm,
  dask,
  distributed,
  dask-image,
  segment-anything,
}:
buildPythonPackage {
  pname = "cellSAM";
  version = "0.0.dev1";

  # Local source. Build with --impure so flake can access it.
  src = ./..;

  pyproject = true;
  buildInputs = [
    setuptools
  ];

  dependencies = [
    torch
    torchvision
    numpy
    pyyaml
    scikit-image
    scikit-learn
    scipy
    kornia
    requests
    tqdm
    dask
    distributed
    dask-image
    segment-anything
  ];

  # The model weights are downloaded at runtime; nothing to import-test that
  # doesn't trigger a network call.
  pythonImportsCheck = [ ];

  pythonRuntimeDepsCheck = false;
  dontCheckRuntimeDeps = true;

  meta = {
    description = "Foundation Model for Cell Segmentation.";
    homepage = "https://github.com/vanvalenlab/cellSAM";
    license = lib.licenses.asl20;
  };
}
