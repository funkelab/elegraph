import torch
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
import torch_geometric.transforms as T
import numpy as np
import glob
import os.path as osp

class PointCloudDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super(PointCloudDataset, self).__init__(root, transform, pre_transform, pre_filter)

    @property
    def raw_file_names(self):
        return sorted(glob.glob("/groups/funke/home/tame/data/seamcellcoordinates/*.csv"))
        # or "point_cloud.csv"
    
    @property
    def processed_file_names(self):
        return []

    def download(self):
        pass

    def process(self):
        for idx, raw_path in enumerate(self.raw_paths):
            data_array = np.genfromtxt(raw_path, delimiter=',')[1:, 1:4]
            pos = torch.tensor(data_array, dtype=torch.float)
            data = Data(pos=pos)
            if self.pre_transform is not None:
                data = self.pre_transform(data)
            torch.save(data, osp.join(self.processed_dir, f'data_{idx+20}.pt'))

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, f'data_{idx}.pt'))
        return data

# connected all of them?

root = "/groups/funke/home/tame/data/point_clouds/"
transformed_root = "/groups/funke/home/tame/data/t_point_clouds/"
transformation_tensor = torch.tensor([[1.0, 2.0, 3.0],
                                        [4.0, 5.0, 6.0],
                                        [7.0, 8.0, 9.0]])
pre_transform = T.Compose([T.RandomJitter(5), T.RandomFlip(0), T.RandomRotate(0.5), T.LinearTransformation(transformation_tensor)])


dataset = PointCloudDataset(root)
transformed_dataset = PointCloudDataset(transformed_root, pre_transform=pre_transform)
loader = DataLoader(dataset, batch_size=32)
transformed_loader = DataLoader(transformed_dataset, batch_size=32)