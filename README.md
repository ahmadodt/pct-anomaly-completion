# PCT Point Cloud Transformer Experiments

This repository contains two older point-cloud deep learning experiments:

- `Anomaly_detection/`: the main project. It trains a modified Point Cloud Transformer (PCT) with two classification heads: object class and anomaly/deformation type.
- `completion_decoder/`: a separate point-cloud completion decoder that predicts synthetically removed local regions on ModelNet10.

The strongest runnable path is `Anomaly_detection/main.ipynb`. The Python scripts are useful for reference, but `Anomaly_detection/train.py` is notebook-derived and is not a clean standalone entry point.

## Recommended Environment

Use Python 3.11 or older. Python 3.14 is too new for this old PyTorch/CUDA stack.

On this machine, the working environment is:

```powershell
.\.venv311\Scripts\Activate.ps1
```

To recreate it:

```powershell
py -3.11 -m venv .venv311
.\.venv311\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r Anomaly_detection\requirements.txt
python -m pip install notebook
```

For GPU training, install a PyTorch build matching your CUDA version, then build the local PointNet++ ops extension:

```powershell
cd Anomaly_detection
python -m pip install .\pointnet2_ops_lib
```

The code has a slow CPU fallback for farthest point sampling, so imports and small smoke checks can run without the CUDA extension. Full training should use the CUDA extension.

## Dataset

The old Google Drive dataset link is stale. Download Anomaly-ShapeNet from the current project location:

- GitHub: https://github.com/Chopper-233/Anomaly-ShapeNet
- Hugging Face: https://huggingface.co/datasets/Chopper233/Anomaly-ShapeNet

Extract the archive so the dataset root contains:

```text
pcd\
new_pcd\    # optional, depending on the release
```

Then point the loader at that dataset root:

```powershell
$env:ANOMALY_SHAPENET_ROOT="C:\path\to\Anomaly-ShapeNet-v2\Anomaly-ShapeNet-v2\dataset"
```

If you extract it into `Anomaly_detection\data\data\Anomaly-ShapeNet-v2\Anomaly-ShapeNet-v2\dataset`, the loader will find it automatically.

## Quick Checks

From the repository root:

```powershell
.\.venv311\Scripts\Activate.ps1
cd Anomaly_detection
..\.venv311\Scripts\python.exe -c "import addict, torch; from model import Pct; args=addict.Dict(dropout=0.5); m=Pct(args, 3, 6).eval(); y=m(torch.randn(2,3,1024)); print(y[0].shape, y[1].shape)"
```

Expected output:

```text
torch.Size([2, 3]) torch.Size([2, 6])
```

To check the dataset after extraction:

```powershell
..\.venv311\Scripts\python.exe -c "from data.dataset import AbnormalData; d=AbnormalData('test', device='cpu', num_points=64); print(len(d), d.classes, d.anomalies)"
```

## Running

For the main experiment:

```powershell
.\.venv311\Scripts\Activate.ps1
$env:ANOMALY_SHAPENET_ROOT="C:\path\to\Anomaly-ShapeNet-v2\Anomaly-ShapeNet-v2\dataset"
cd Anomaly_detection
jupyter notebook main.ipynb
```

## Known Caveats

- `Anomaly_detection/train.py` contains notebook magic and calls `data_split()` without defining it.
- `Anomaly_detection/main_partseg.py` references stale segmentation code.
- Dataset paths were originally hardcoded; use `ANOMALY_SHAPENET_ROOT` for local data.
- CPU farthest point sampling is for small checks only and will be slow for full training.
- `completion_decoder/` is a compact research prototype that learns to reconstruct synthetically removed local regions from partial ModelNet10 point clouds.
