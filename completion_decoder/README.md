# Completion Decoder Experiment

This folder contains a separate experimental decoder. It is not part of the Anomaly-ShapeNet classifier workflow.

The idea is to load ModelNet10 `.off` meshes, sample them into point clouds, shuffle/normalize the points, and train a masked self-attention model to predict plausible later points in the shuffled point sequence.

This is exploratory rather than a standard point-cloud completion setup because point clouds are unordered and the "future point" target depends on the random shuffle.

## Environment

Use the repository Python 3.11 environment:

```powershell
..\.venv311\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Data

`data.py` attempts to download ModelNet10 from:

```text
http://3dvision.princeton.edu/projects/2014/3DShapeNets/ModelNet10.zip
```

The downloader uses Unix-style commands (`wget`, `unzip`, `mv`, `rm`), so it may not work on plain Windows PowerShell. If that happens, download and extract ModelNet10 manually so the layout is:

```text
completion_decoder\
  data\
    ModelNet10\
      bathtub\
        train\
        test\
      ...
```

## Run

From this folder:

```powershell
..\.venv311\Scripts\Activate.ps1
python main.py --epochs 10 --batch_size 8 --test_batch_size 8 --num_workers 0
```

Use `--num_workers 0` on Windows for fewer multiprocessing issues during quick checks.

## Caveats

- This experiment is independent from `Anomaly_detection/`.
- The downloader is not Windows-native.
- The training objective is experimental and sequence-dependent.
- `main.py` uses shell `cp` commands to back up files into `checkpoints/`, which may fail on Windows unless a Unix-like shell is available.
