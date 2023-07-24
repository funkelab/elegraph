# sanity check with validation generated data to see if we still gte 70% acc
import argparse
import glob
import numpy as np
import os
import os.path as osp
import pandas as pd
import random
import torch
import torch_geometric.transforms as T
from math import comb
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment as lsa
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from tqdm import tqdm
from simclr_loss import SimCLR_Loss
from logger import Logger
from torch.nn import Linear as Lin
import torch.nn.functional as F
from torch_geometric.nn import SplineConv
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
from mid_spline import spline_graph
from deform import transform
from angle import set_curvatures, same_side

class RandomShift(T.BaseTransform):
    def __init__(self, range):
        self.low = range[0]
        self.high = range[1]
        assert self.low < self.high

    def forward(self, data: Data):
        shift = np.random.randint(low=self.low, high=self.high, size=(1,3))
        data.pos += shift
        return data

def graph(array, filename):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    x = array[:,0]
    y = array[:,1]
    z = array[:,2]
    ax.scatter(x, y, z, c='b', marker='o')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Point Cloud')
    plt.show()
    plt.savefig(filename)

class SplineCNN(torch.nn.Module):
    def __init__(self, in_channels, out_channels, dim, num_layers, cat=True,
                 lin=True, dropout=0.0):
        super(SplineCNN, self).__init__()
        self.in_channels = in_channels
        self.dim = dim
        self.num_layers = num_layers
        self.cat = cat
        self.lin = lin
        self.dropout = dropout
        self.convs = torch.nn.ModuleList()

        for _ in range(num_layers):
            conv = SplineConv(in_channels, out_channels, dim, kernel_size=5)
            self.convs.append(conv)
            in_channels = out_channels

        if self.cat:
            in_channels = self.in_channels + num_layers * out_channels
        else:
            in_channels = out_channels

        if self.lin:
            self.out_channels = out_channels
            self.final = Lin(in_channels, out_channels)
        else:
            self.out_channels = in_channels

        self.reset_parameters()

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        if self.lin:
            self.final.reset_parameters()

    def forward(self, x, edge_index, edge_attr, *args):
        """"""
        xs = [x]

        for conv in self.convs:
            xs += [F.relu(conv(xs[-1], edge_index, edge_attr))]

        x = torch.cat(xs, dim=-1) if self.cat else xs[-1]
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.final(x) if self.lin else x
        return x

    def __repr__(self):
        return ('{}({}, {}, dim={}, num_layers={}, cat={}, lin={}, '
                'dropout={})').format(self.__class__.__name__,
                                      self.in_channels, self.out_channels,
                                      self.dim, self.num_layers, self.cat,
                                      self.lin, self.dropout)

class WormValDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, type, transform):
        self.seam_cells_list = sorted(glob.glob(osp.join(data_dir, '*.csv')))
        self.real_size = len(self.seam_cells_list)
        self.transform = transform

    def __len__(self):
        return 54 #self.real_size-1 # TODO

    def map_ids_indices(self, gt_ids):
        d= {'H0L': 0, 'H0R':1, 'H1L':2, 'H1R':3, 'H2L':4, 'H2R':5, 'V1L':6,
                'V1R':7, 'V2L':8, 'V2R':9, 'V3L': 10, 'V3R':11, 'V4L':12,
                'V4R':13, 'V5L':14, 'V5R': 15, 'V6L':16, 'V6R': 17, 'TL':18,
                'TR': 19, 'QL': 20, 'QR': 21}

        gt_indices = []
        for gt_id in gt_ids:
            gt_indices.append(d[gt_id])
        return np.asarray(gt_indices)

    def __getitem__(self, idx):
        # moving
        moving_detections_df = pd.read_csv(self.seam_cells_list[idx], delimiter=',')
        moving_gt_ids = moving_detections_df['name']
        moving_gt_ids = self.map_ids_indices(moving_gt_ids)
        moving_positions = moving_detections_df.iloc[:, 1:4].to_numpy()

        # moving (no more shuffling)
        # ids_moving_permute = np.random.permutation(moving_positions.shape[0])
        # moving_gt_ids = moving_gt_ids[ids_moving_permute]
        # moving_positions = moving_positions[ids_moving_permute]

        # fixed
        fixed_detections_df = pd.read_csv(self.seam_cells_list[idx], delimiter=',')
        # fixed_detections_df = pd.read_csv(self.seam_cells_list[idx+1], delimiter=',') !!
        fixed_gt_ids = fixed_detections_df['name']
        fixed_gt_ids = self.map_ids_indices(fixed_gt_ids)
        fixed_positions = fixed_detections_df.iloc[:, 1:4].to_numpy()

        # fixed
        # ids_fixed_permute = np.random.permutation(fixed_positions.shape[0])
        # fixed_gt_ids = fixed_gt_ids[ids_fixed_permute]
        # fixed_positions = fixed_positions[ids_fixed_permute]

        y_moving = []
        y_fixed =[]
        for i, id in enumerate(moving_gt_ids):
            match = np.where(fixed_gt_ids == id)
            if len(match[0])==0:
                pass
            else:
                y_moving.append(i)
                y_fixed.append(match[0][0])

        y_moving = np.asarray(y_moving)
        y_fixed = np.asarray(y_fixed)

        y_moving = torch.from_numpy(y_moving)
        y_fixed = torch.from_numpy(y_fixed)
        
        # sanity check !!
        moving_detections = transform(moving_positions, 3).float()
        fixed_detections = transform(fixed_positions, 3).float()

        # moving_detections = torch.from_numpy(moving_positions).float() !!!
        # fixed_detections = torch.from_numpy(fixed_positions).float()

        data_s = Data(pos=moving_detections, y=y_moving)
        data_t = Data(pos=fixed_detections, y=y_fixed)

        if self.transform is not None:
            data_s = self.transform(data_s)
            data_t = self.transform(data_t)

        # create knn graph and establish distance attributes !!!!!!
        knn = T.Compose([T.KNNGraph(k=len(moving_positions) - 1)])
        data_s = knn(data_s)
        data_t = knn(data_t)
        
        # set curvatures as node attributes 
        # data_s.edge_index = same_side(data_s) hmmm, this is lowkey cheating cuz the model now knows what side is what side... we don't know this info during testing cuz 
        # data_t.edge_index = same_side(data_t)
        # set_curvatures(data_s, 6, 1)
        # set_curvatures(data_t, 6, 1)

        distance = T.Compose([T.Distance()])
        data_s = distance(data_s)
        data_t = distance(data_t)

        if idx == 0:
            graph(data_s.pos.detach().cpu().numpy(), "data_s_v.png")
            graph(data_t.pos.detach().cpu().numpy(), "data_t_v.png")

        data = Data(num_nodes=moving_detections.size(0))
        for key in data_s.keys:
            data['{}_s'.format(key)] = data_s[key]
        data['ids_s'] = torch.from_numpy(np.asarray(moving_gt_ids))

        for key in data_t.keys:
            data['{}_t'.format(key)] = data_t[key]
        data['ids_t'] = torch.from_numpy(np.asarray(fixed_gt_ids))
        return data

class WormTrainDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir='./', type='train', transform=None):
        self.seam_cells_list = sorted(
            glob.glob(osp.join(data_dir, '*.csv')))  # TODO

        self.real_size = len(self.seam_cells_list)
        print("Number of files in {} dataset is {}".format(type, self.real_size))
        self.transform = transform

    def __len__(self):
        return self.real_size

    def __getitem__(self, idx):
        detections_df = pd.read_csv(self.seam_cells_list[idx], delimiter=',')
        gt_ids = detections_df['name']
        gt_ids = np.arange(len(gt_ids)) # TODO
        positions = detections_df.iloc[:, 1:4].to_numpy()

        if idx == 0:
            graph(positions, "data_orig.png")
 
        #ids_fixed_permute = np.random.permutation(positions.shape[0])
        #fixed_gt_ids = gt_ids[ids_fixed_permute]
        #fixed_positions = positions[ids_fixed_permute]
        fixed_gt_ids = gt_ids
        fixed_positions = positions

        #ids_moving_permute = np.random.permutation(positions.shape[0])
        #moving_gt_ids = gt_ids[ids_moving_permute]
        #moving_positions = positions[ids_moving_permute]
        moving_gt_ids = gt_ids
        moving_positions = positions
        
        # for both moving and fixed, save the index of where each node is 
        y_moving = []
        y_fixed = []
        for i, id in enumerate(moving_gt_ids):
            match = np.where(fixed_gt_ids == id)
            if len(match[0]) ==0:
                pass
            else:
                y_moving.append(i)
                y_fixed.append(match[0][0])

        y_moving = np.asarray(y_moving)
        y_fixed = np.asarray(y_fixed)

        # map where each node is at for both moving and fixed
        y_moving = torch.from_numpy(y_moving) #0,1,2,3,..22
        y_fixed = torch.from_numpy(y_fixed) # 3,16,13, etc.

        # new spline transformation + in tensor form!
        moving_positions = transform(moving_positions, 3).float()
        fixed_positions = transform(fixed_positions, 3).float()

        # moving_detections = torch.from_numpy(moving_positions).float()
        # fixed_detections = torch.from_numpy(fixed_positions).float()

        data_s = Data(pos=moving_positions, y=y_moving)
        data_t = Data(pos=fixed_positions, y=y_fixed)

        if self.transform is not None:
            data_s = self.transform(data_s)
            data_t = self.transform(data_t)
        
        data_s = RandomShift((-2,2)).forward(data_s)
        data_t = RandomShift((-2,2)).forward(data_t)

        # spline transformation
        # data_s.pos = spline(data_s.pos.detach().cpu().numpy(), np.array([0,3,4,8]), 1)
        # data_t.pos = spline(data_t.pos.detach().cpu().numpy(), np.array([0,3,4,8]), 1)

        # create knn graph and establish distance attributes
        # distance = T.Compose([T.KNNGraph(k=len(moving_positions) - 1), T.Distance()])
        knn = T.Compose([T.KNNGraph(k=len(moving_positions) - 1)])
        data_s = knn(data_s)
        data_t = knn(data_t)

        # set curvatures as node attributes 
        # data_s.edge_index = same_side(data_s)
        # data_t.edge_index = same_side(data_t)
        # set_curvatures(data_s, 6, 1)
        # set_curvatures(data_t, 6, 1)

        distance = T.Compose([T.Distance()])
        data_s = distance(data_s)
        data_t = distance(data_t)
        
        if idx == 0:
            graph(data_s.pos.detach().cpu().numpy(), "data_s.png")
            graph(data_t.pos.detach().cpu().numpy(), "data_t.png")

        # aggregate all keys from data_s and data_t into one data object
        data = Data(num_nodes=moving_positions.size(0))
        for key in data_s.keys:
            data['{}_s'.format(key)] = data_s[key]
        data['ids_s'] = torch.from_numpy(np.asarray(moving_gt_ids))

        for key in data_t.keys:
            data['{}_t'.format(key)] = data_t[key]
        data['ids_t'] = torch.from_numpy(np.asarray(fixed_gt_ids))
        return data


def train():
    model.train()
    total_loss = total_examples = total_correct = 0
    for i, data in enumerate(train_loader):
        optimizer.zero_grad()
        data = data.to(device)
        # print(data.x_s.float())
        emb_s = model(data.x_s.float(), data.edge_index_s, data.edge_attr_s.float())
        emb_t = model(data.x_t.float(), data.edge_index_t, data.edge_attr_t.float())
        embeddings = torch.cat((emb_s, emb_t), 0)
        logits, labels, similarity_matrix = simclr_loss(embeddings, torch.cat((data.ids_s,
            data.ids_t), dim =0))
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        # calculate accuracy
        similarity_matrix_ = 1 - similarity_matrix[:similarity_matrix.shape[0]//2,
                similarity_matrix.shape[1]//2:]
        row_inds, col_inds = lsa(similarity_matrix_.cpu().detach().numpy())
        temp = (col_inds==data.y_t.cpu().detach().numpy()).sum()
        total_correct += temp
        total_examples += len(col_inds)

    return total_loss / len(train_loader), total_correct / total_examples

@torch.no_grad()
def val():
    model.eval()
    total_loss = total_examples = total_correct = 0
    for i, data in enumerate(val_loader):
        data = data.to(device)
        emb_s = model(data.x_s.float(), data.edge_index_s, data.edge_attr_s.float())
        emb_t = model(data.x_t.float(), data.edge_index_t, data.edge_attr_t.float())
        embeddings = torch.cat((emb_s, emb_t), 0)
        logits, labels, similarity_matrix = simclr_loss(embeddings, torch.cat((data.ids_s,
            data.ids_t), dim=0))
        loss = criterion(logits, labels)
        total_loss += loss.item()
        similarity_matrix_ = 1 - similarity_matrix[:data.x_s.shape[0],
                data.x_s.shape[0]:]
        row_inds, col_inds = lsa(similarity_matrix_.cpu().detach().numpy())
        total_correct += (col_inds==data.y_t.cpu().detach().numpy()).sum()
        total_examples += len(col_inds)
    return total_loss / len(train_loader), total_correct / total_examples
    
torch.backends.cudnn.benchmark = True
parser = argparse.ArgumentParser()
parser.add_argument('--isotropic', action='store_true')
parser.add_argument('--dim', type=int, default=32)
parser.add_argument('--rnd_dim', type=int, default=64) # TODO --> default : 64
parser.add_argument('--num_layers', type=int, default=4)  # TODO --> default : 2
parser.add_argument('--num_steps', type=int, default=10)  # TODO --> default 10 (during training)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--batch_size', type=int, default=1)  # TODO --> default 64
parser.add_argument('--epochs', type=int, default=10000)
parser.add_argument('--checkpoint_path', type=str, default='experiment/best_iou_model.pth')
parser.add_argument('--save', type=bool, default=True)
args = parser.parse_args()

#path = os.environ.get('path')
# path = '/groups/funke/home/tame/data/seamcellcoordinates'
train_path="/groups/funke/home/tame/data/Untwisted/SeamCellCoordinates"
val_path="/groups/funke/home/tame/data/seamcellcoordinates"

# logger
logger = Logger(['train_loss', 'train_accuracy', 'val_loss', 'val_accuracy'],
        'experiment6')

transform_train = T.Compose([
    T.RandomFlip(0),
    T.RandomRotate(120),
    T.RandomScale((0,2)),
    # T.NormalizeScale(),
    T.Constant(),
])

transform_val = T.Compose([
    T.NormalizeScale(),
    T.Constant(),
    T.KNNGraph(k=19),  # TODO - k = 8 (default)
    T.Distance(), # TODO - norm was set to True initially
])

train_path = osp.join(train_path)
train_dataset = WormTrainDataset(train_path, type='train',
        transform=transform_train)
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, follow_batch=['x_s', 'x_t'])

val_path = osp.join(val_path)
val_dataset = WormValDataset(train_path, type='val',
        transform=transform_val)
# val_dataset = WormValDataset(val_path, type='val',
#         transform=transform_val)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, follow_batch=['x_s', 'x_t'])

data = train_dataset[0]
# print(data.x_s.size())
# print(data.x_s[0])
# print(data.x_s[0].size(-1))
# print(data.x_s)
# print(data.edge_attr_s.size())
# print(data.edge_attr_s.size(-1))
# print(data.edge_attr_s)

from torch.nn import LayerNorm, Linear, ReLU
from torch_geometric.nn import DeepGCNLayer, GENConv
class DeeperGCN(torch.nn.Module):
    def __init__(self, hidden_channels, num_layers):
        super().__init__()
        self.layers = torch.nn.ModuleList()
        self.node_encoder = Linear(data.x_s.size(-1), hidden_channels)
        self.edge_encoder = Linear(data.edge_attr_s.size(-1), hidden_channels)
        for i in range(1, num_layers + 1):
            conv = GENConv(hidden_channels, hidden_channels, aggr='softmax',
                           t=1.0, learn_t=True, num_layers=2, norm='layer')
            norm = LayerNorm(hidden_channels, elementwise_affine=True)
            act = ReLU(inplace=False)
            layer = DeepGCNLayer(conv, norm, act, block='res+', dropout=0.0,
                                 ckpt_grad=i % 3)
            self.layers.append(layer)
        self.lin = Linear(hidden_channels, data.y_s.size(-1))
    
    def forward(self, x, edge_index, edge_attr):
        x = self.node_encoder(x)
        edge_attr = self.edge_encoder(edge_attr)
        # print(x[0].size(-1))
        # print(edge_attr.size(-1))
        x = self.layers[0].conv(x, edge_index, edge_attr)
        for layer in self.layers[1:]:
            x = layer(x, edge_index, edge_attr)
        x = self.layers[0].act(self.layers[0].norm(x))
        return self.lin(x)

# device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = "cpu"

# model = SplineCNN(1, args.dim, 1, args.num_layers, cat=False,
#         dropout=0.0).to(device)  # TODO dim was earlier 2

model = DeeperGCN(hidden_channels=64, num_layers=28).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

simclr_loss = SimCLR_Loss(temperature=0.1)
criterion = torch.nn.CrossEntropyLoss().to(device)
# load snapshot
if os.path.exists(args.checkpoint_path):
    state = torch.load(args.checkpoint_path)
    model.load_state_dict(state['model_state_dict'], strict=True)
    assert (True, 'checkpoint path found!')
else:
    assert (False, 'checkpoint_path {} does not exist!'.format(args.checkpoint_path))

EPS = 1e-8

for epoch in tqdm(range(1, args.epochs)):
    train_loss, train_acc = train()
    val_loss, val_acc = val()
    logger.add(key='train_loss', value=train_loss)
    logger.add(key='train_accuracy', value=train_acc)
    
    logger.add(key='val_loss', value=val_loss)
    logger.add(key='val_accuracy', value=val_acc)

    logger.write()
    logger.plot()
    print(f'Epoch: {epoch:02d}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}')

# fix angle.py so there's no same_side or abs(neighbor - node). ik there'll be some extra stuff, but j make the array longer? we'll see
# see how new data works with angle.py
# train with new data + validate with old data + angle
