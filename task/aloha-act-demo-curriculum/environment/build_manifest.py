#!/usr/bin/env python3
"""Record sha256 digests of every fixed asset and the installed package versions.

The Judge recomputes all of them before scoring (tests/bootstrap.py). The manifest
itself lives in the public directory so agents can see exactly what is fixed.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as md
import json
import os
from pathlib import Path

PACKAGES = ["lerobot", "gym-aloha", "mujoco", "dm-control", "torch", "torchvision", "torchcodec", "av", "numpy", "gymnasium", "pandas", "pyarrow", "aloha-curriculum"]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--roots", nargs="+", required=True, help="directories whose regular files are all recorded")
    p.add_argument("--output", required=True)
    p.add_argument("--lerobot-commit", required=True)
    a = p.parse_args()
    files: dict[str, str] = {}
    for root in a.roots:
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames.sort()
            if "__pycache__" in dirnames:
                dirnames.remove("__pycache__")
            for name in sorted(filenames):
                path = Path(dirpath) / name
                if path.is_symlink() or not path.is_file():
                    continue
                if path.resolve() == Path(a.output).resolve():
                    continue
                files[os.fspath(path)] = sha256_file(path)
    packages = {}
    for name in PACKAGES:
        try:
            packages[name] = md.version(name)
        except md.PackageNotFoundError:
            raise SystemExit(f"package {name} is not installed")
    manifest = {"files": files, "packages": packages, "lerobot_commit": a.lerobot_commit}
    Path(a.output).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"manifest: {len(files)} files, {len(packages)} packages -> {a.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
