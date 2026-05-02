{
  lib,
  buildPythonPackage,
  fetchPypi,
  setuptools,
  torch,
  torchvision,
}:

buildPythonPackage rec {
  pname = "segment_anything";
  version = "1.0";
  pyproject = true;

  src = fetchPypi {
    pname = pname;
    version = version;
    hash = "sha256-7Qyfb7B7vvnGI4pwKKE8gnLxumtjBcpz4+BkJmUDc2s=";
  };

  build-system = [
    setuptools
  ];

  dependencies = [
    torch
    torchvision
  ];

  pythonImportsCheck = [
    "segment_anything"
  ];

  meta = {
    description = "Segment Anything (Meta).";
    homepage = "https://github.com/facebookresearch/segment-anything";
    license = lib.licenses.asl20;
  };
}
