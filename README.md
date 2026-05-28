# 3D Point Cloud Anomaly Detection and Shape Completion

This repository contains two PyTorch experiments for learning from 3D point clouds with Point Cloud Transformer-style models.

The project covers two related tasks:

- **Anomaly detection/classification**: predict both the object category and anomaly/deformation type from an Anomaly-ShapeNet point cloud.
- **Shape completion**: reconstruct a missing local region from a partial ModelNet10 point cloud using a Chamfer-distance reconstruction loss.

## Repository Structure

```text
Anomaly_detection/
  Dual-head PCT classifier for Anomaly-ShapeNet.

completion_decoder/
  PCT-style encoder/decoder for missing-region point-cloud completion.
```

Each folder has its own README with dataset setup, run commands, smoke tests, and caveats:

- [Anomaly Detection README](Anomaly_detection/README.md)
- [Completion Decoder README](completion_decoder/README.md)

## Highlights

- Built custom PyTorch datasets for `.pcd` and `.off` point-cloud sources.
- Implemented point-cloud preprocessing including normalization, randomized sampling, local region removal, and farthest point sampling.
- Used Point Cloud Transformer-style self-attention blocks for point-cloud feature learning.
- Added a dual-head classifier for object category and anomaly type prediction.
- Added a completion decoder that predicts missing local geometry from partial point clouds.
- Improved reproducibility with configurable dataset paths, Windows-compatible data handling, smoke tests, and documentation.

## Environment

Use Python 3.11 or older. Python 3.14 is too new for parts of the PyTorch/CUDA stack used by this project.

The local working environment is expected to be:

```powershell
.\.venv311\Scripts\Activate.ps1
```

To recreate it:

```powershell
py -3.11 -m venv .venv311
.\.venv311\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Then install the requirements for the experiment you want to run:

```powershell
python -m pip install -r Anomaly_detection\requirements.txt
python -m pip install notebook
```

or:

```powershell
python -m pip install -r completion_decoder\requirements.txt
```

## Datasets

This repo does not include the full datasets.

- **Anomaly-ShapeNet** is used by `Anomaly_detection/`.
- **ModelNet10** is used by `completion_decoder/`.

See the experiment-specific READMEs for download links, expected folder layouts, and environment variables.

## Quick Smoke Checks

Anomaly classifier model check:

```powershell
.\.venv311\Scripts\Activate.ps1
cd Anomaly_detection
..\.venv311\Scripts\python.exe -c "import addict, torch; from model import Pct; args=addict.Dict(dropout=0.5); m=Pct(args, 3, 6).eval(); y=m(torch.randn(2,3,1024)); print(y[0].shape, y[1].shape)"
```

Completion decoder model/loss check:

```powershell
.\.venv311\Scripts\Activate.ps1
cd completion_decoder
..\.venv311\Scripts\python.exe -c "import torch; from model import Pct; from main import chamfer_distance; m=Pct(32,2).eval(); x=torch.randn(2,3,96); y=m(x); print(y.shape, float(chamfer_distance(y, torch.randn(2,32,3))))"
```

## Status

This is a research/prototype codebase. The anomaly detection notebook is the most complete original workflow, and the completion decoder has been updated into a runnable missing-region completion experiment. Full training still depends on downloading the appropriate datasets and, for efficient anomaly training, using a CUDA-compatible PyTorch setup with the local PointNet++ ops extension.
