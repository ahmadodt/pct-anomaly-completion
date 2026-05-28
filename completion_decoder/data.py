import glob
import os
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import open3d as o3d
import torch
import trimesh
from torch.utils.data import Dataset


MODELNET10_URL = "http://3dvision.princeton.edu/projects/2014/3DShapeNets/ModelNet10.zip"


def visualize_point_cloud(pcd):
    original_pcd = o3d.geometry.PointCloud()
    original_pcd.points = o3d.utility.Vector3dVector(pcd)
    o3d.visualization.draw_geometries([original_pcd], window_name="Point Cloud")


def read_OFF_mesh(file_path):
    with open(file_path, "r") as f:
        header = f.readline().strip()
        if header != "OFF":
            raise ValueError(f"{file_path} is not an OFF file")

        counts = f.readline().strip().split()
        while not counts:
            counts = f.readline().strip().split()

        num_vertices = int(counts[0])
        num_faces = int(counts[1])

        vertices = [list(map(float, f.readline().strip().split())) for _ in range(num_vertices)]
        faces = []
        for _ in range(num_faces):
            face = list(map(int, f.readline().strip().split()))
            faces.append(face[1:])

    return vertices, faces


def generate_point_cloud_from_mesh(file_path, num_points):
    vertices, faces = read_OFF_mesh(file_path)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    return np.array(mesh.sample(num_points), dtype=np.float32)


def data_dir(data_root=None):
    if data_root is not None:
        return Path(data_root)
    env_root = os.environ.get("MODELNET10_ROOT")
    if env_root:
        return Path(env_root)
    return Path(__file__).resolve().parent / "data" / "ModelNet10"


def download(data_root=None):
    target_dir = data_dir(data_root)
    if target_dir.exists():
        return target_dir

    archive_dir = target_dir.parent
    archive_dir.mkdir(parents=True, exist_ok=True)
    archive_path = archive_dir / "ModelNet10.zip"

    try:
        if not archive_path.exists():
            urllib.request.urlretrieve(MODELNET10_URL, archive_path)
        with zipfile.ZipFile(archive_path, "r") as zip_ref:
            zip_ref.extractall(archive_dir)
    except Exception as exc:
        raise RuntimeError(
            "ModelNet10 was not found and could not be downloaded. Download it manually from "
            f"{MODELNET10_URL}, extract it, then set MODELNET10_ROOT to the extracted ModelNet10 folder."
        ) from exc

    if not target_dir.exists():
        raise RuntimeError(f"Expected ModelNet10 at {target_dir}, but extraction did not create it.")

    return target_dir


def load_data(partition, num_points, data_root=None):
    root = download(data_root)
    all_data = []
    all_labels = []

    pattern = str(root / "*" / partition / "*.off")
    for off_path in glob.glob(pattern):
        pcd = generate_point_cloud_from_mesh(off_path, num_points)
        all_data.append(pcd)
        all_labels.append(Path(off_path).parts[-3])

    if not all_data:
        raise FileNotFoundError(f"No ModelNet10 .off files found for partition '{partition}' under {root}.")

    return all_data, all_labels


class ModelNet10(Dataset):
    def __init__(self, num_points, num_missing_points=128, partition="train", data_root=None):
        if num_missing_points <= 0:
            raise ValueError("num_missing_points must be positive")
        if num_missing_points >= num_points:
            raise ValueError("num_missing_points must be smaller than num_points")

        self.data, self.label = load_data(partition, num_points, data_root)
        self.num_points = num_points
        self.num_missing_points = num_missing_points
        self.partition = partition

    def normalize_pcd(self, points):
        points = points - torch.mean(points, dim=0, keepdim=True)
        furthest_distance = torch.max(torch.sqrt(torch.sum(points**2, dim=-1)))
        return points / furthest_distance.clamp_min(1e-6)

    def split_partial_missing(self, pcd_full):
        center_idx = torch.randint(0, pcd_full.shape[0], (1,)).item()
        center = pcd_full[center_idx].view(1, 3)
        distances = torch.sum((pcd_full - center) ** 2, dim=1)
        missing_idx = torch.topk(distances, self.num_missing_points, largest=False).indices

        keep_mask = torch.ones(pcd_full.shape[0], dtype=torch.bool)
        keep_mask[missing_idx] = False

        partial = pcd_full[keep_mask]
        missing = pcd_full[missing_idx]
        return partial, missing

    def __getitem__(self, item):
        pcd_full = torch.as_tensor(self.data[item], dtype=torch.float32)
        if pcd_full.shape[0] != self.num_points:
            raise ValueError(f"Expected {self.num_points} points, got {pcd_full.shape[0]}")

        pcd_full = self.normalize_pcd(pcd_full)
        partial, missing = self.split_partial_missing(pcd_full)

        partial = partial[torch.randperm(partial.shape[0])]
        missing = missing[torch.randperm(missing.shape[0])]
        full = pcd_full[torch.randperm(pcd_full.shape[0])]
        return partial, missing, full

    def __len__(self):
        return len(self.data)


if __name__ == "__main__":
    test_dataset = ModelNet10(1024, 128, "test")
    partial, missing, full = test_dataset[0]
    print(partial.shape, missing.shape, full.shape)
