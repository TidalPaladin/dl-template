#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import subprocess 
from setuptools import setup, find_packages

TORCH = "torch>=1.6.0,<=2.0.0"

extras = {}

install_requires = [
    TORCH,
    "pytorch-lightning>=1.0.0",
    "hydra-core>=1.0.0",
    "pynvml",
]

def write_version_info():
    # get version
    cwd = os.getcwd()
    version = open("version.txt", "r").read().strip()

    sha = "Unknown"
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=cwd).decode("ascii").strip()
    except Exception:
        pass

    if os.getenv("COMBUSTION_BUILD_VERSION"):
        version = os.getenv("COMBUSTION_BUILD_VERSION")
    elif sha != "Unknown":
        version += "+" + sha[:7]

    version_path = os.path.join(cwd, "src", "combustion", "version.py")
    with open(version_path, "w") as f:
        f.write("__version__ = '{}'\n".format(version))
        f.write("git_version = {}\n".format(repr(sha)))

    return version


def install(version):
    setup(
        name="dl-template",
        version=version,
        author="Scott Chase Waggener",
        author_email="tidalpaladin@gmail.com",
        description="Template for PyTorch model training/testing",
        keywords="deep learning pytorch",
        license="Apache",
        url="https://github.com/TidalPaladin/dl-template",
        package_dir={"": "src"},
        packages=find_packages("src"),
        install_requires=install_requires,
        extras_require=extras,
        python_requires=">=3.7.0",
        classifiers=[
            "Intended Audience :: Developers",
            "Intended Audience :: Education",
            "Intended Audience :: Science/Research",
            "Operating System :: OS Independent",
            "Programming Language :: Python :: 3.7",
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
        ],
    )


if __name__ == "__main__":
    version = write_version_info()
    install(version)
