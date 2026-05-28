This experiment trains a dual-head Point Cloud Transformer on Anomaly-ShapeNet.

## Environment

Use Python 3.11 or older. The old `torch==1.13.1` pin does not install on newer Python versions such as Python 3.14.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
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

## Run

Run `main.ipynb` from this `Anomaly_detection` directory. `train.py` is notebook-derived and is not a clean standalone entry point.
