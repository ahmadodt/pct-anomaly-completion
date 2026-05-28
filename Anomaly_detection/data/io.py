import trimesh
import numpy as np
from pathlib import Path

def read_off(filename):
    mesh = trimesh.load(Path(filename), process=False)
    return np.array(mesh.vertices)
    # return mesh

def write_off(filename, pointcloud):
    mesh = trimesh.Trimesh(vertices=pointcloud, process=False)
    mesh.export(filename, file_type='off')
    # trimesh.exchange.off.export_off(mesh, digits=10)