# Anomaly Detection Experiment

This is the main experiment in the repository. It trains a modified Point Cloud Transformer (PCT) on Anomaly-ShapeNet to predict:

- the object class, such as chair or table
- the anomaly/deformation type

The model uses one shared point-cloud backbone with two classification heads.

## Environment

Use Python 3.11 or older. The old `torch==1.13.1` pin does not install on newer Python versions such as Python 3.14.

```powershell
py -3.11 -m venv ..\.venv311
..\.venv311\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install notebook
```

For GPU training, install a PyTorch build that matches your CUDA version before building the local extension.

```powershell
python -m pip install ./pointnet2_ops_lib
```

The code now has a slow CPU fallback for farthest point sampling, so imports and small sanity checks can run without the CUDA extension. Full training should still use the extension.

## Dataset

The original Google Drive link is stale. Download the current Anomaly-ShapeNet release from the official project:

- GitHub: https://github.com/Chopper-233/Anomaly-ShapeNet
- Hugging Face: https://huggingface.co/datasets/Chopper233/Anomaly-ShapeNet

Extract the archive so the dataset root contains `pcd/` and, if present, `new_pcd/`.

Then set the dataset root before running the notebook:

```powershell
$env:ANOMALY_SHAPENET_ROOT="C:\path\to\Anomaly-ShapeNet-v2\Anomaly-ShapeNet-v2\dataset"
```

If you extract it into `Anomaly_detection\data\data\Anomaly-ShapeNet-v2\Anomaly-ShapeNet-v2\dataset`, the loader will find it automatically.

Expected layout:

```text
dataset\
  pcd\
    <class>\
      train\
      test\
  new_pcd\        # optional, depending on the release
```

## Run

Run `main.ipynb` from this `Anomaly_detection` directory. `train.py` is notebook-derived and is not a clean standalone entry point.

```powershell
..\.venv311\Scripts\Activate.ps1
$env:ANOMALY_SHAPENET_ROOT="C:\path\to\Anomaly-ShapeNet-v2\Anomaly-ShapeNet-v2\dataset"
jupyter notebook main.ipynb
```

## Smoke Tests

Check model imports and forward pass:

```powershell
..\.venv311\Scripts\python.exe -c "import addict, torch; from model import Pct; args=addict.Dict(dropout=0.5); m=Pct(args, 3, 6).eval(); y=m(torch.randn(2,3,1024)); print(y[0].shape, y[1].shape)"
```

Expected:

```text
torch.Size([2, 3]) torch.Size([2, 6])
```

Check the dataset after extraction:

```powershell
..\.venv311\Scripts\python.exe -c "from data.dataset import AbnormalData; d=AbnormalData('test', device='cpu', num_points=64); print(len(d), d.classes, d.anomalies)"
```

If the dataset is missing, the loader should fail with a clear `ANOMALY_SHAPENET_ROOT` message.

## Known Caveats

- `train.py` contains notebook magic and calls `data_split()` without defining it.
- `main_partseg.py` is an older/stale segmentation path.
- CPU farthest point sampling exists for sanity checks, not full training.
- Full training should use a CUDA PyTorch build and the local `pointnet2_ops_lib` extension.
