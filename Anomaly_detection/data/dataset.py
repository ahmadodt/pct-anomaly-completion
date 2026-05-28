
import torch
from pathlib import Path
import numpy as np
import re
import pandas as pd
#from exercise_2.data.binvox_rw import read_as_3d_array
from pyntcloud import PyntCloud
#import open3d as o3d
import os

try:
    from pointnet2_ops_lib.pointnet2_ops.pointnet2_utils import FurthestPointSampling
except Exception:
    FurthestPointSampling = None


def _repo_dataset_root():
    env_root = os.environ.get("ANOMALY_SHAPENET_ROOT")
    if env_root:
        return Path(env_root)
    return Path(__file__).resolve().parent / "data" / "Anomaly-ShapeNet-v2" / "Anomaly-ShapeNet-v2" / "dataset"


def _dataset_paths():
    root = _repo_dataset_root()
    paths = []
    for name in ("pcd", "new_pcd"):
        path = root / name
        if path.exists():
            paths.append(path)
    if not paths:
        paths = [root / "pcd", root / "new_pcd"]
    return paths


def _furthest_point_sample_cpu(points, num_points):
    point_count = points.size(0)
    if point_count == 0:
        raise ValueError("Cannot sample an empty point cloud")

    sample_count = min(num_points, point_count)
    centroids = torch.zeros(sample_count, dtype=torch.long, device=points.device)
    distance = torch.full((point_count,), 1e10, device=points.device)
    farthest = torch.randint(0, point_count, (1,), device=points.device, dtype=torch.long).item()

    for i in range(sample_count):
        centroids[i] = farthest
        centroid = points[farthest].view(1, 3)
        dist = torch.sum((points - centroid) ** 2, dim=1)
        distance = torch.minimum(distance, dist)
        farthest = torch.max(distance, dim=0)[1].item()

    if sample_count < num_points:
        padding = centroids[torch.randint(0, sample_count, (num_points - sample_count,), device=points.device)]
        centroids = torch.cat([centroids, padding], dim=0)

    return centroids

class AbnormalData(torch.utils.data.Dataset):

    dataset_paths = _dataset_paths()

    def __init__(self, split="test", device="cuda", num_points=1024, dataset_root=None):

        super().__init__()
        assert split in ['train', 'test', 'GT']

        self.data = []
        self.classes = []
        self.anomalies = []

        self.split = split
        self.num_points = num_points
        self.device = device
        self.dataset_paths = _dataset_paths() if dataset_root is None else [
            path for path in (Path(dataset_root) / "pcd", Path(dataset_root) / "new_pcd") if path.exists()
        ]

        if not self.dataset_paths:
            raise FileNotFoundError(
                "Anomaly-ShapeNet data was not found. Set ANOMALY_SHAPENET_ROOT to the extracted "
                "directory that contains pcd/ and optionally new_pcd/."
            )
        if not any(path.exists() for path in self.dataset_paths):
            raise FileNotFoundError(
                "Anomaly-ShapeNet data was not found. Set ANOMALY_SHAPENET_ROOT to the extracted "
                "directory that contains pcd/ and optionally new_pcd/."
            )

        # Loop through the datasets paths and collect the data
        for self.dataset_path in self.dataset_paths:
          
            for root, dirs, files in os.walk(self.dataset_path, topdown=False):
                
                #if the root contain split then we are in the right directory
                if split in root:
                    
                    class_name = root.split(os.path.sep)[-2]
        
                    for filename in files:
                        if not filename.lower().endswith(".pcd"):
                            continue
                        # extracting the anomaly and class name from the filename
                        parts = filename.split("_")
                        anomaly = parts[1] if len(parts) > 1 else Path(filename).stem
                        anomaly = anomaly.split(".")[0]
                        anomaly = re.sub(r'[0-9]', '', anomaly)
                        class_name = re.sub(r'[0-9]', '', class_name)
    
                        if anomaly not in self.anomalies:
                            self.anomalies.append(anomaly)
    
                        if class_name not in self.classes:
                            self.classes.append(class_name)

                        # Add data entry to the list, repeating multiple times for augmentation
                        for i in range(5):
                            self.data.append({
                                                'pcd_path': os.path.join(root, filename),
                                                'class': self.classes.index(class_name),
                                                'anomaly': self.anomalies.index(anomaly)     
                                            })
                            
        # Create a DataFrame from the collected data and shuffle it
        self.data = pd.DataFrame(self.data)
        if self.data.empty:
            raise ValueError(
                f"No .pcd files found for split '{split}' under: "
                + ", ".join(str(path) for path in self.dataset_paths)
            )
        self.data = self.data.sample(frac=1).reset_index(drop=True)
        pass

    def __getitem__(self, index):

        pcd = self.data.iloc[index]['pcd_path']
        pcd = PyntCloud.from_file(pcd)
        points = torch.tensor(pcd.points.values).to(self.device)
        classes = torch.tensor(self.data.iloc[index]['class']).to(self.device)
        anomalies = torch.tensor(self.data.iloc[index]['anomaly']).to(self.device)

        # Ensure points tensor is contiguous(in one memory block) else the furthest points sampling gpu wont work
        points = points.contiguous()
        #randomize the indeces to result in a different output( by the furthest point sampling) for the same duplicte input in the data
        points = points[torch.randperm(points.size()[0])]
        # Apply furthest point sampling
        if FurthestPointSampling is not None and points.is_cuda:
            sampled_indices = FurthestPointSampling.apply(points.unsqueeze(0), self.num_points).squeeze()
        else:
            sampled_indices = _furthest_point_sample_cpu(points, self.num_points)
        sampled_points = points[sampled_indices]
        return sampled_points, classes, anomalies


    def __len__(self):
        return len(self.data)




#sp = AbnormalData("test")

#print(sp.data.head())
#item = sp.__getitem__(1)
#print(item[0].size())

#print(sp.__len__())
    

                        #add the same point multiple times to icrease the dataset to result in acceptable results
                        #those result wont be duplicates because we take random 1000 points from 50k points
