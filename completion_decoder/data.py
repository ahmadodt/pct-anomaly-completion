import glob
import os

import numpy as np
import open3d as o3d
import torch
import trimesh
from scipy.spatial.distance import cdist, pdist, squareform
from torch.utils.data import Dataset


def visualize_point_cloud(pcd):
    original_pcd = o3d.geometry.PointCloud()
    original_pcd.points = o3d.utility.Vector3dVector(pcd)
    o3d.visualization.draw_geometries([original_pcd], window_name="Point Cloud")


def read_OFF_mesh(file_path):
    """
    Read vertex information from an OFF file.
    """
    with open(file_path, "r") as f:
        lines = f.readlines()

    vertices = []
    faces = []
    for line in lines[2:]:
        data = list(map(float, line.strip().split()))
        if len(data) == 3:  # vertex
            vertices.append(data)
        elif len(data) == 4:  # face
            faces.append(data[1:])

    # Returns the vertices and faces as Python lists
    return vertices, faces


def generate_point_cloud_from_mesh(file_path, num_points):
    vertices, faces = read_OFF_mesh(file_path)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

    # Generate point cloud data from the mesh
    point_cloud = mesh.sample(num_points)

    # Output the generated point cloud data from mesh and only the vertexes
    # visualize_point_cloud(np.array(point_cloud), np.array(vertices))
    return np.array(point_cloud)


# Downloads the ModelNet dataset to the data folder and unzips it
def download():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, "data")
    if not os.path.exists(DATA_DIR):
        os.mkdir(DATA_DIR)
    if not os.path.exists(os.path.join(DATA_DIR, "ModelNet10")):
        www = "http://3dvision.princeton.edu/projects/2014/3DShapeNets/ModelNet10.zip"
        zipfile = os.path.basename(www)
        os.system("wget %s; unzip %s" % (www, zipfile))
        os.system("mv %s %s" % (zipfile[:-4], DATA_DIR))
        os.system("rm %s" % (zipfile))


def load_data(partition, num_points):
    download()
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, "data")
    all_data = []
    all_labels = []
    for off_path in glob.glob(os.path.join(DATA_DIR, "ModelNet10", f"*/{partition}/*.off")):
        pcd = generate_point_cloud_from_mesh(off_path, num_points)
        all_data.append(np.array(pcd))
        all_labels.append(off_path.split("/")[-3])
    return all_data, all_labels


def random_point_dropout(pc, max_dropout_ratio=0.875):
    """batch_pc: BxNx3"""
    # for b in range(batch_pc.shape[0]):
    dropout_ratio = np.random.random() * max_dropout_ratio  # 0~0.875
    drop_idx = np.where(np.random.random((pc.shape[0])) <= dropout_ratio)[0]
    # print ('use random drop', len(drop_idx))

    if len(drop_idx) > 0:
        pc[drop_idx, :] = pc[0, :]  # set to the first point
    return pc


def translate_pointcloud(pointcloud):
    xyz1 = np.random.uniform(low=2.0 / 3.0, high=3.0 / 2.0, size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])

    translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype("float32")
    return translated_pointcloud


def jitter_pointcloud(pointcloud, sigma=0.01, clip=0.02):
    N, C = pointcloud.shape
    pointcloud += np.clip(sigma * np.random.randn(N, C), -1 * clip, clip)
    return pointcloud


# original_pcd is one numpy array (Data of one .off file)
def filter_point_cloud(original_pcd):
    # Choose a random point
    random_point = np.random.choice(original_pcd, size=1, replace=False)
    radius = calculate_radius(original_pcd, 0.05)

    # Find points outside the sphere
    distances = cdist(original_pcd, random_point)
    points_outside_sphere = original_pcd[distances > radius]

    return points_outside_sphere, original_pcd


# original_pcd is one numpy array (Data of one .off file)
# if distance_percentage is 0.05, it means the 5% of the maximum distance
def calculate_radius(original_pcd, distance_percentage):
    # Compute pairwise distances between points
    pairwise_distances = squareform(pdist(original_pcd))

    # Find the farthest two points
    max_distance = np.max(pairwise_distances)

    # Calculate radius as a percentage of the maximum distance
    radius = distance_percentage * max_distance

    return radius


class ModelNet10(Dataset):
    def __init__(self, num_points, partition="train"):
        self.data, self.label = load_data(partition, num_points)
        self.num_points = num_points
        self.partition = partition

    def normalize_pcd(self, points):
        centroid = torch.mean(points, dim=0)
        points -= centroid
        furthest_distance = torch.max(torch.sqrt(torch.sum(torch.abs(points)**2,dim=-1)))
        points /= furthest_distance

        return points

    def __getitem__(self, item: int):
        pcd_full = torch.as_tensor(self.data[item], dtype=torch.float32)
        if pcd_full.shape[0] != self.num_points:
            raise ValueError("noo")
        
        pcd_full = self.normalize_pcd(pcd_full)

        new_indices = torch.randperm(pcd_full.shape[0])
        shuffled_pcd = pcd_full[new_indices]
        return shuffled_pcd

    def __len__(self):
        return len(self.data)


if __name__ == "__main__":
    test_dataset = ModelNet10(1024, "test")
    for pcd in test_dataset:
        print(pcd.shape)
        break
