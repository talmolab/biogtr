# DREEM Relates Every Entities' Motion

[![CI](https://github.com/talmolab/dreem/actions/workflows/ci.yml/badge.svg)](https://github.com/talmolab/dreem/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/talmolab/dreem/branch/main/graph/badge.svg?token=Sj8kIFl3pi)](https://codecov.io/gh/talmolab/dreem)
[![code](https://img.shields.io/github/stars/talmolab/dreem)](https://github.com/talmolab/dreem)
<!-- [![Release](https://img.shields.io/github/v/release/talmolab/dreem?label=Latest)](https://github.com/talmolab/dreem/releases/)
[![PyPI](https://img.shields.io/pypi/v/dreem?label=PyPI)](https://pypi.org/project/dreem)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/dreem) -->

Global Tracking Transformers for biological multi-object tracking.

## Installation

<!-- ### Basic
```bash
pip install dreem
```  
-->

### Development
#### Clone the repository:
```bash
git clone https://github.com/talmolab/dreem && cd dreem
```
#### Set up in a new conda environment:
##### Linux/Windows:
###### GPU-accelerated (requires CUDA/nvidia gpu)
```bash
conda env create -y -f environment.yml && conda activate dreem
```
###### CPU:
```bash
conda env create -y -f environment_cpu.yml && conda activate dreem
```
#### OSX (M chip)
```bash
conda env create -y -f environment_osx-arm.yml && conda activate dreem
```
### Uninstalling
```
conda env remove -n dreem
```