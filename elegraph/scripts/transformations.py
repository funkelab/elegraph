import torch
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
import torch_geometric.transforms as T
import numpy as np
import glob
import os.path as osp
import csv
import torch.nn.functional as F 
from simclr_loss import SimCLR_Loss
from scipy.optimize import linear_sum_assignment as lsa

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
        filenames =sorted(glob.glob("/groups/funke/home/lalitm/data/Raw/SeamCellCoordinates/*.csv"))
        #print("files found {}".format(len(filenames)))
        return filenames
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
        #print("Index is {}".format(idx))
        data = torch.load(osp.join(self.processed_dir, f'data_{idx+20}.pt'))
        transformed_data = torch.load(osp.join(self.processed_dir, f'transformed_data_{idx+20}.pt'))
        #print(data.pos.shape)
        #print(transformed_data.pos.shape)

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
    T.Constant(),
    T.KNNGraph(k=5),
    T.Distance(),
])

root = "/groups/funke/home/lalitm/data/t_point_clouds/"
pre_transform = T.Compose([T.RandomFlip(0), T.RandomRotate(120), T.RandomScale((0,2))])
point_cloud_dataset = PointCloudDataset(root, pre_transform=pre_transform, transform=transform)
point_cloud_loader = DataLoader(point_cloud_dataset, batch_size=1)

# create model
from torch_geometric.nn import GATConv

class GAT(torch.nn.Module):
    def __init__(self, num_node_features, num_output_features):
        super().__init__()
        self.num_node_features = num_node_features
        self.num_output_features = num_output_features
        self.conv1 = GATConv(num_node_features, 16)
        self.conv2 = GATConv(16, num_output_features)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index # TODO

        x = self.conv1(x.float(), edge_index) # TODO
        x = F.relu(x)
        #x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        #return F.log_softmax(x, dim=1)
        return x

model = GAT(num_node_features= 2, num_output_features = 32) # TODO
model = model.to('cuda')

# create criterion
criterion = torch.nn.CrossEntropyLoss().to('cuda') # TODO

simclr_loss = SimCLR_Loss(temperature =0.5)


# create optimizer
optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3, weight_decay =
        5e-4) # TODO



def train():
    model.train()
    loss_list = []
    accuracy_list = []
    for i, data in enumerate(point_cloud_loader):
        optimizer.zero_grad()
        data_orig = data[0].to('cuda') # TODO
        data_trans = data[1].to('cuda')
        print(data_orig.edge_attr)
        embeddings_orig = model(data_orig)
        embeddings_trans = model(data_trans)
        embeddings = torch.cat((embeddings_orig, embeddings_trans), 0)
        logits, labels = simclr_loss(embeddings)
        loss = criterion(logits, labels)
        logits_ = 1-logits[:logits.shape[0]//2, logits.shape[1]//2:]
        row_ind, col_ind = lsa(logits_.cpu().detach().numpy())
        accuracy_list.append((row_ind==col_ind).sum()/len(row_ind))
        loss_list.append(loss.item())
        loss.backward()
        optimizer.step()
    return np.mean(loss_list), np.mean(accuracy_list)

for epoch in range(1, 5000):
    loss, accuracy = train()
    print("At epoch {}, loss was {}, accuracy was {}".format(epoch, loss,
        accuracy))


#for i, batch in enumerate(transformed_loader):
#    data_knn, transformed_data_knn = batch
#    if i == 0:
#        print(data_knn.edge_attr)
#        print(transformed_data_knn.edge_attr)



