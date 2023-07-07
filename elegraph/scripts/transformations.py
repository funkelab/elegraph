import torch
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
import torch_geometric.transforms as T
import numpy as np
import glob
import os.path as osp
import csv

class RandomShift(T.BaseTransform):
    def __init__(self, range):
        self.low = range[0]
        self.high = range[1]
        assert self.low < self.high

    def forward(self, data: Data):
        shift = np.random.randint(low=self.low, high=self.high, size=(1,3))
        data.pos += shift
        return data

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
            # get tensor of seam cell coordinates (20/22 rows by x,y,z columns)
            data_array = np.genfromtxt(raw_path, delimiter=',')[1:, 1:4]
            pos = torch.tensor(data_array, dtype=torch.float)

            # create Data objects
            data = Data(pos=pos)
            transformed_data = Data(pos=pos)
            # data = Data(pos=pos, edge_index=self.set_edges(pos))
            # transformed_data = Data(pos=pos, edge_index=self.set_edges(pos))

            # normalize
            normalize = T.Compose([T.NormalizeScale()])
            data = normalize(data)
            transformed_data = normalize(transformed_data)

            # transform
            transformed_data = pre_transform(transformed_data)
            transformed_data = RandomShift((-2,2)).forward(transformed_data)

            # set all edge attributes to be euclidean distance between nodes
            # distance = T.Compose([T.Distance(norm=False)])
            # data = distance(data)
            # transformed_data = distance(transformed_data)

            # save graphs
            torch.save(data, osp.join(self.processed_dir, f'data_{idx+20}.pt'))
            torch.save(transformed_data, osp.join(self.processed_dir, f'transformed_data_{idx+20}.pt'))

    def len(self):
        return len(self.raw_paths) # TODO

    def get(self, idx):
        print("Index is {}".format(idx))
        data = torch.load(osp.join(self.processed_dir, f'data_{idx+20}.pt'))
        transformed_data = torch.load(osp.join(self.processed_dir, f'transformed_data_{idx+20}.pt'))
        print(data.pos.shape)
        print(transformed_data.pos.shape)

        if self.transform is not None:
            data = self.transform(data)
            transformed_data = self.transform(transformed_data)
            
        return data, transformed_data
    
    def set_edges(self, pos):
        """
        Connects all points with each other.
        """
        source = np.array([])
        target = np.array([])
        for s in range(pos.shape[0]):
            for t in range(pos.shape[0]):
                if s != t:
                    source = np.append(source, s)
                    target = np.append(target, t)
        source = np.array([source])
        target = np.array([target])
        return np.concatenate((source, target), axis=0)

#let's try creating this 22 by 5 nearest neighbors

transform = T.Compose([
    # T.Constant(),
    T.KNNGraph(k=5),
    T.Distance(),
])

root = "/groups/funke/home/tame/data/t_point_clouds/"
pre_transform = T.Compose([T.RandomFlip(0), T.RandomRotate(120), T.RandomScale((0,2))])
d = PointCloudDataset(root, pre_transform=pre_transform, transform=transform)
transformed_loader = DataLoader(d, batch_size=1)

for i, batch in enumerate(transformed_loader):
    data_knn, transformed_data_knn = batch
    if i == 0:
        print(data_knn.edge_attr)
        print(transformed_data_knn.edge_attr)

def write_gt():
    csv_path = "/groups/funke/home/tame/data/gt_matches.csv"
    path = sorted(glob.glob("/groups/funke/home/tame/data/point_clouds/processed/*.pt"))[:-2]
    path_transformed = sorted(glob.glob("/groups/funke/home/tame/data/t_point_clouds/processed/*.pt"))[:-2]
    with open(csv_path, "w") as f:
        writer = csv.writer(f, delimiter=' ')
        writer.writerow(["time", "x", "y", "z", "t_x", "t_y", "t_z"])
        for i in range(len(path)):
            graph = torch.load(path[i]).pos.numpy()
            t_graph = torch.load(path_transformed[i]).pos.numpy()
            for j in range(len(t_graph)):
                row = np.append(graph[j], t_graph[j])
                row = np.append(i+20, row)
                writer.writerow(row)

# write_gt()


