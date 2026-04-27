#!/usr/bin/env python3
"""Check whether the local environment can run the Geneformer foundation pipeline."""

from __future__ import annotations

import importlib.util
import os


os.environ.setdefault("NUMBA_DISABLE_JIT", "1")


def has_package(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def main() -> None:
    packages = ["geneformer", "transformers", "datasets", "scanpy", "anndata", "pyarrow", "torch"]
    for package in packages:
        print(f"{package}: {'ok' if has_package(package) else 'missing'}")

    try:
        import geneformer  # noqa: F401

        print("geneformer_import: ok")
    except Exception as exc:  # pragma: no cover - diagnostic script
        print(f"geneformer_import: failed ({type(exc).__name__}: {exc})")

    try:
        import torch

        print(f"torch_version: {torch.__version__}")
        print(f"torch_cuda_version: {torch.version.cuda}")
        print(f"cuda_available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"cuda_device: {torch.cuda.get_device_name(0)}")
            arch_list = torch.cuda.get_arch_list()
            print(f"torch_cuda_arch_list: {arch_list}")
            if "sm_120" not in arch_list:
                print("gpu_status: incompatible_torch_build_missing_sm_120")
            else:
                try:
                    value = torch.ones(1, device="cuda").item()
                    print(f"gpu_status: ok ({value})")
                except Exception as exc:  # pragma: no cover - diagnostic script
                    print(f"gpu_status: cuda_runtime_failed ({type(exc).__name__}: {exc})")
    except Exception as exc:  # pragma: no cover - diagnostic script
        print(f"torch_check: failed ({type(exc).__name__}: {exc})")


if __name__ == "__main__":
    main()
