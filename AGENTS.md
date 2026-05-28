# AGENTS.md

Guidance for coding agents working in this repository.

## Repository Overview

This repo contains two older point-cloud deep learning experiments:

- `Anomaly_detection/`: the main experiment. It trains a modified Point Cloud Transformer (PCT) on Anomaly-ShapeNet with two heads: object classification and anomaly/deformation classification.
- `completion_decoder/`: a separate point-cloud completion decoder on ModelNet10. It predicts synthetically removed local regions from partial point clouds.

Prefer working on `Anomaly_detection/` unless the user explicitly asks about the completion decoder.

## Environment

Use Python 3.11 or older. Python 3.14 is too new for the old PyTorch/CUDA stack.

The working local environment is expected to be:

```powershell
.\.venv311\Scripts\Activate.ps1
```

Recreate it from the repo root with:

```powershell
py -3.11 -m venv .venv311
.\.venv311\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r Anomaly_detection\requirements.txt
python -m pip install notebook
```

Do not commit virtual environments, downloaded datasets, caches, or notebook checkpoints.

## Dataset

The original Google Drive link for Anomaly-ShapeNet is stale. Use the current project sources:

- https://github.com/Chopper-233/Anomaly-ShapeNet
- https://huggingface.co/datasets/Chopper233/Anomaly-ShapeNet

The loader expects a dataset root containing:

```text
pcd\
new_pcd\    # optional, depending on release
```

Set the root with:

```powershell
$env:ANOMALY_SHAPENET_ROOT="C:\path\to\Anomaly-ShapeNet-v2\Anomaly-ShapeNet-v2\dataset"
```

If data is extracted into `Anomaly_detection\data\data\Anomaly-ShapeNet-v2\Anomaly-ShapeNet-v2\dataset`, the loader can find it automatically.

## Main Run Path

The main runnable workflow is the notebook:

```powershell
.\.venv311\Scripts\Activate.ps1
$env:ANOMALY_SHAPENET_ROOT="C:\path\to\Anomaly-ShapeNet-v2\Anomaly-ShapeNet-v2\dataset"
cd Anomaly_detection
jupyter notebook main.ipynb
```

`Anomaly_detection/train.py` is notebook-derived and is not a reliable standalone entry point.

## Smoke Tests

From `Anomaly_detection/`, check model imports and a CPU forward pass:

```powershell
..\.venv311\Scripts\python.exe -c "import addict, torch; from model import Pct; args=addict.Dict(dropout=0.5); m=Pct(args, 3, 6).eval(); y=m(torch.randn(2,3,1024)); print(y[0].shape, y[1].shape)"
```

Expected:

```text
torch.Size([2, 3]) torch.Size([2, 6])
```

Check dataset loading after extraction:

```powershell
..\.venv311\Scripts\python.exe -c "from data.dataset import AbnormalData; d=AbnormalData('test', device='cpu', num_points=64); print(len(d), d.classes, d.anomalies)"
```

If the dataset is missing, the loader should fail clearly with an `ANOMALY_SHAPENET_ROOT` message.

## CUDA Extension

The repo includes `pointnet2_ops_lib`. Full training should use the CUDA extension:

```powershell
cd Anomaly_detection
python -m pip install .\pointnet2_ops_lib
```

The code has a slow pure-PyTorch CPU fallback for farthest point sampling. Use it only for imports, small forward passes, and debugging.

## Known Caveats

- `Anomaly_detection/main_partseg.py` is stale and references missing segmentation code.
- `Anomaly_detection/train.py` contains notebook magic and calls undefined notebook-local helpers.
- Dataset labels are inferred from folder/file names in `Anomaly_detection/data/dataset.py`.
- Existing checkpoints and logs are historical experiment artifacts.
- `completion_decoder/` is a compact completion prototype, not a paper-quality model.

## Code Style

- Keep changes small and local to the relevant experiment folder.
- Preserve the research/prototype structure unless the user asks for a cleanup.
- Prefer configurable paths over hardcoded local machine paths.
- Do not remove checkpoints, logs, or old scripts unless explicitly asked.


from now on teh commits arein this form:
feat: = new validation behavior
refactor: = same behavior, cleaner structure
fix: = bug fix
test: = adding or updating tests
docs: chnage in docs
