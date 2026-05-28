import torch
import random
import os
import data
import addict
import model
import h5py
import numpy as np
from torch.utils.data import Dataset

### pointclouds: torch.Tensor of size (N, 3)
def is_class_recognizable(
        pointcloud_normal: torch.Tensor,
        pointcloud_deformed: torch.Tensor, 
        class_model: torch.nn.Module, 
    ):

    Nn, _ = pointcloud_normal.shape
    Nd, _ = pointcloud_deformed.shape
    
    classes_prediction_normal : torch.Tensor = class_model(pointcloud_normal.view(1, Nn, 3))
    classes_prediction_deformed : torch.Tensor = class_model(pointcloud_deformed.view(1, Nd, 3))
    # MAX OF EACH AND COMPARE IT
    # INDEX OF THE MAXIMUM VALUE
    predict_normal = torch.argmax(classes_prediction_normal)
    predict_deformed = torch.argmax(classes_prediction_deformed)
    return predict_normal == predict_deformed

   
def remove_segment(pointcloud, segments, undesired_segment):
    # point cloud shape (N,x,y,z,segment)
    pointcloud_with_segments = torch.cat((pointcloud, segments.view(-1, 1)), dim=1)

    #return pointcloud[(segments != undesired_segment).nonzero().squeeze(1)]
    pointcloud_with_segments[(pointcloud_with_segments[:, 3] != undesired_segment).nonzero().squeeze(1)]
    # pointcloud_with_segments[:, :3] is the point cloud without the segments
    return pointcloud_with_segments[:, :3]

def is_deformed(
        pointcloud_normal: torch.Tensor,
        pointcloud_deformed: torch.Tensor
    ):
    # here it should return that they are different!! not should be added after the return
    return not torch.is_same_size(pointcloud_normal, pointcloud_deformed)

def deform(
        pointcloud: torch.Tensor, 
        segments: torch.Tensor, 
        class_model: torch.Tensor,
        max_num_deforms: int,
        min_points: int
    ) -> (torch.Tensor, bool):

    deformed_pointcloud = pointcloud
    deformed = False
    # why do we need to convert to set?
    segments_set = set(torch.unique(segments).tolist())
    for _ in range(max_num_deforms):
        if len(segments_set) == 0:
            return deformed_pointcloud, deformed
        undesired_segment = random.choice(segments_set)
        new_deformed_pointcloud = remove_segment(deformed_pointcloud, segments, undesired_segment)
        segments_set.remove(undesired_segment)
        if is_class_recognizable(deformed_pointcloud, new_deformed_pointcloud, class_model) and len(new_deformed_pointcloud) >= min_points:
            if is_deformed(deformed_pointcloud, new_deformed_pointcloud):
                deformed = True
            deformed_pointcloud = new_deformed_pointcloud
    return deformed_pointcloud, deformed

def create_abnormal_dataset(
        max_num_deforms: int = 5,
        num_original_points: int = 2048,
        min_points: int = 1024) -> None:
    original_dataset = data.ModelNet40(num_original_points)

    # Load the dataset
    original_dataset = data.ModelNet40(num_original_points)

    # Set up directories and paths
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')

    if not os.path.exists(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048_abnormal')):
        os.mkdir(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048_abnormal'))

    # Move the original point clouds and labels to GPU
    original_pointclouds, original_labels = original_dataset[:]
    original_pointclouds = original_pointclouds.to("cuda")
    original_pointclouds = original_pointclouds.permute(0, 2, 1)
    original_labels = original_labels.to("cuda")

    # Set up configuration parameters
    args = addict.Dict()
    args.exp_name = 'exp'
    args.dataset = 'modelnet40'
    args.batch_size = 16
    args.test_batch_size = 16
    args.epochs = 250
    args.use_sgd = True
    args.lr = 0.0001
    args.momentum = 0.9
    args.no_cuda = False
    args.seed = 1
    args.eval = True
    args.model_path = '../models/model.t7'
    args.dropout = 0.5
    args.num_points = 64

    # Instantiate PCT why do we need this? 
    class_model = model.Pct(args)
    class_model = class_model.to("cuda")
    class_model.eval()

    # Define a segmentation model ???? should be Pct_segmentation
    # thismodel should have the weights of the trained model PCT and the rest of the segmentation should 
    # be a random initialization adn be trained.
    segmentation_model = lambda x: x
    segmentation_model = segmentation_model.to("cuda")
    segmentation_model.eval()

    # Get segmentation results for original point clouds in shape B x N x S
    list_segments = segmentation_model(original_pointclouds)
    #LATER WE ARe using list_segments[i, :, :] for the ith point cloud we need to make sure that
    #the model is returning the same order of point clouds as the original dataset and especially if we use a data
    # loader we nee to set shuffle to false.

    # Initialize lists to store deformed point clouds and labels
    deformed_pointclouds = []
    deformed_labels = []

    # Apply deformations to each original point cloud
    for i in range(len(original_pointclouds)):
        original_pointcloud = original_pointclouds[i, :, :]
        deformed_pointcloud, deformed = deform(original_pointcloud, list_segments[i, :, :], class_model)
        if deformed:
            deformed_pointclouds.append(deformed_pointcloud)
            deformed_labels.append(original_labels[i])

    # Set up directory for saving deformed data
    deformed_data_dir = os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048_abnormal')

    if not os.path.exists(deformed_data_dir):
        os.mkdir(deformed_data_dir)

    # Save deformed point clouds and labels in HDF5 format
    deformed_file_path = os.path.join(deformed_data_dir, 'ply_data_abnormal.h5')
    label_file_path = os.path.join(deformed_data_dir, 'labels_abnormal.txt')

    with h5py.File(deformed_file_path, 'w') as f:
        with open(label_file_path, 'w') as label_file:
            for i in range(len(deformed_pointclouds)):
                deformed_pointcloud = deformed_pointclouds[i]
                label = deformed_labels[i].item()

                # Save point cloud data
                f.create_dataset(f'data_{i}', data=deformed_pointcloud.numpy().T, compression='gzip', compression_opts=4)

                # Save label information
                label_file.write(f"{label}\n")

    # Save mapping of labels to class names
    class_mapping_file_path = os.path.join(deformed_data_dir, 'class_mapping_abnormal.txt')
    with open(class_mapping_file_path, 'w') as class_mapping_file:
        for class_name, class_id in original_dataset.label_to_names.items():
            class_mapping_file.write(f"{class_id} {class_name}\n")
    ### END
            
class DeformedModelNet40(Dataset):
    def __init__(self, num_points, partition='train'):
        self.data, self.label = self.load_deformed_data(partition)
        self.num_points = num_points
        self.partition = partition

    def load_deformed_data(self, partition):
        deformed_data_dir = os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048_abnormal')
        label_file_path = os.path.join(deformed_data_dir, 'labels_abnormal.txt')

        with open(label_file_path, 'r') as label_file:
            labels = [int(line.strip()) for line in label_file]

        data = []
        for i in range(len(labels)):
            filename = os.path.join(deformed_data_dir, f'ply_data_abnormal_{i}.h5')
            with h5py.File(filename, 'r') as f:
                # Ensure the number of points is the same as the original dataset
                deformed_pointcloud = f[f'data_{i}'][:self.num_points].T
                data.append(deformed_pointcloud)

        return np.array(data), np.array(labels)

    def __getitem__(self, item):
        pointcloud = self.data[item]
        label = torch.Tensor(self.label[item])

        if self.partition == 'train':
            # Apply any desired transformations for training (e.g., random point dropout, translation, jitter)
            pointcloud = data.random_point_dropout(pointcloud)
            pointcloud = data.translate_pointcloud(pointcloud)
            np.random.shuffle(pointcloud)

        return torch.Tensor(pointcloud), label

    def __len__(self):
        return self.data.shape[0]
    
class CombinedModelNet40(Dataset):
    def __init__(self, num_points, partition='train'):
        self.original_dataset = data.ModelNet40(num_points, partition)
        self.deformed_dataset = DeformedModelNet40(num_points, partition)
        #add a column to the labels of the deformed dataset to indicate that it is deformed
        self.deformed_dataset.label = np.column_stack((self.deformed_dataset.label, np.ones(len(self.deformed_dataset.label))))
        #add a column to the labels of the original dataset to indicate that it is not deformed
        self.original_dataset.label = np.column_stack((self.original_dataset.label, np.zeros(len(self.original_dataset.label))))
        # Create a fixed order for alternating between original and deformed samples
        self.indices = [i for i in range(len(self.original_dataset) + len(self.deformed_dataset))]

    def __getitem__(self, item):
        # Use the fixed order to alternate between original and deformed point clouds
        index = self.indices[item]

        if index < len(self.original_dataset):
            return self.original_dataset[index]
        else:
            # Subtract the length of the original dataset to get the corresponding index in the deformed dataset
            deformed_index = index - len(self.original_dataset)
            return self.deformed_dataset[deformed_index]

    def __len__(self):
        # Return the total number of samples in both datasets
        return len(self.original_dataset) + len(self.deformed_dataset)