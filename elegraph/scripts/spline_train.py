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
import random
from visual_matching import visualize_matches

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

def drop(n, moving_ids, moving_pos, fixed_ids, fixed_pos):
    assert n < len(moving_ids) and n < len(moving_pos) and n < len(fixed_ids) and n < len(fixed_pos)
    for i in range(n):
        assert len(moving_ids) == len(moving_pos) == len(fixed_ids) == len(fixed_pos)
        index = random.randint(0, len(moving_ids) - 1)
        moving_ids = np.delete(moving_ids, index)
        moving_pos = np.delete(moving_pos, index, axis=0)
        fixed_ids = np.delete(fixed_ids, index)
        fixed_pos = np.delete(fixed_pos, index, axis=0)
    return moving_ids, moving_pos, fixed_ids, fixed_pos

class WormValDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, type, transform):
        self.seam_cells_list = sorted(glob.glob(osp.join(data_dir, '*.csv')))
        self.real_size = len(self.seam_cells_list)
        self.transform = transform

    def __len__(self):
        return self.real_size-1

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
        if idx == 56:
            idx = 55
        # moving
        moving_detections_df = pd.read_csv(self.seam_cells_list[idx], delimiter=',')
        moving_gt_ids = moving_detections_df['name']
        moving_gt_ids = self.map_ids_indices(moving_gt_ids)
        moving_positions = moving_detections_df.iloc[:, 1:4].to_numpy()

        # fixed
        fixed_detections_df = pd.read_csv(self.seam_cells_list[idx+1], delimiter=',')
        #fixed_detections_df = pd.read_csv(self.seam_cells_list[idx], delimiter=',') # !! change back after sanity check
        fixed_gt_ids = fixed_detections_df['name']
        fixed_gt_ids = self.map_ids_indices(fixed_gt_ids)
        fixed_positions = fixed_detections_df.iloc[:, 1:4].to_numpy()

        y_moving = torch.from_numpy(moving_gt_ids)
        y_fixed = torch.from_numpy(fixed_gt_ids)

        # new spline transformation + in tensor form! # remove for new data !!! remove after sanity check
        # moving_positions = transform(moving_positions, 8).float()
        # fixed_positions = transform(fixed_positions, 8).float()

        moving_positions = torch.from_numpy(moving_positions).float() ###!!!!!! put this back in after sanity check
        fixed_positions = torch.from_numpy(fixed_positions).float()

        data_s = Data(pos=moving_positions, y=y_moving)
        data_t = Data(pos=fixed_positions, y=y_fixed)

        if self.transform is not None:
            data_s = self.transform(data_s)
            data_t = self.transform(data_t)

        # set curvatures as node attributes 
        set_curvatures(data_s, 3, 3)
        set_curvatures(data_t, 3, 3)

        distance = T.Compose([T.Distance()])
        data_s = distance(data_s)
        data_t = distance(data_t)

        # graph for reference
        # if idx == 0:
        #     graph(data_s.pos.detach().cpu().numpy(), "data_s_v.png")
        #     graph(data_t.pos.detach().cpu().numpy(), "data_t_v.png")

        data = Data(num_nodes=moving_positions.size(0))
        for key in data_s.keys:
            data['{}_s'.format(key)] = data_s[key]
        data['ids_s'] = torch.from_numpy(np.asarray(moving_gt_ids))

        for key in data_t.keys:
            data['{}_t'.format(key)] = data_t[key]
        data['ids_t'] = torch.from_numpy(np.asarray(fixed_gt_ids))
        return data

class WormTrainDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir='./', type='train', transform=None):
        cells = sorted(glob.glob(osp.join(data_dir, '*.csv')))
        #self.seam_cells_list = cells[49:] + cells[:49] # need to do this for new data!!
        self.seam_cells_list = cells
        print(self.seam_cells_list[0])
        self.real_size = len(self.seam_cells_list)
        print("Number of files in {} dataset is {}".format(type, self.real_size))
        self.transform = transform

    def __len__(self):
        return self.real_size - 1 # hmm need to allow for crossing one to another. or we can j skip 20-22 and only do 20-20 and 22-22 for now.

    def __getitem__(self, idx):
        # if idx == 67: # for new data
        #     idx = 66
        if idx == 56: # for untwisted data
            idx = 55
        moving_df = pd.read_csv(self.seam_cells_list[idx], delimiter=',')
        moving_gt_ids = np.arange(len(moving_df)) # TODO
        moving_positions = moving_df.iloc[:, 1:4].to_numpy()

        fixed_df = pd.read_csv(self.seam_cells_list[idx], delimiter=',') #maybe this should have been this for untwisted data all along?
        #fixed_df = pd.read_csv(self.seam_cells_list[idx+1], delimiter=',')
        fixed_gt_ids = np.arange(len(fixed_df))
        fixed_positions = fixed_df.iloc[:, 1:4].to_numpy()

        # randomly drop n nodes
        # n = 1
        # moving_gt_ids, moving_positions, fixed_gt_ids, fixed_positions = drop(n, moving_gt_ids, moving_positions, fixed_gt_ids, fixed_positions)

        y_moving = torch.from_numpy(moving_gt_ids)
        y_fixed = torch.from_numpy(fixed_gt_ids)

        # new spline transformation + in tensor form! # remove for new data
        moving_positions = transform(moving_positions, 8).float()
        fixed_positions = transform(fixed_positions, 8).float()

        #moving_positions = torch.from_numpy(moving_positions).float()
        #fixed_positions = torch.from_numpy(fixed_positions).float()

        data_s = Data(pos=moving_positions, y=y_moving)
        data_t = Data(pos=fixed_positions, y=y_fixed)

        if self.transform is not None:
            data_s = self.transform(data_s)
            data_t = self.transform(data_t)
        
        data_s = RandomShift((-2,2)).forward(data_s)
        data_t = RandomShift((-2,2)).forward(data_t)

        # create knn graph and establish distance attributes
        knn = T.Compose([T.KNNGraph(k=3)])
        data_s = knn(data_s)
        data_t = knn(data_t)
        
        # set curvatures as node attributes 
        set_curvatures(data_s, 3, 3)
        set_curvatures(data_t, 3, 3)

        distance = T.Compose([T.Distance()])
        data_s = distance(data_s)
        data_t = distance(data_t)

        # graph for reference
        # if idx == 0:
        #     graph(data_s.pos.detach().cpu().numpy(), "data_s.png")
        #     graph(data_t.pos.detach().cpu().numpy(), "data_t.png")

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
        # print(i)
        optimizer.zero_grad()
        data = data.to(device)
        emb_s = model(data.x_s.float(), data.edge_index_s, data.edge_attr_s.float())
        emb_t = model(data.x_t.float(), data.edge_index_t, data.edge_attr_t.float())
        embeddings = torch.cat((emb_s, emb_t), 0)
        logits, labels, similarity_matrix = simclr_loss(embeddings, torch.cat((data.ids_s,
            data.ids_t), dim=0))
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        # calculate accuracy
        similarity_matrix_ = 1 - similarity_matrix[:similarity_matrix.shape[0]//2,
                similarity_matrix.shape[1]//2:]
        row_inds, col_inds = lsa(similarity_matrix_.cpu().detach().numpy())
        # visualize matches
        visualize_matches(col_inds, "visual_match_train.png") # for some reason this overrides the graphic
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
        # calculate accuracy
        similarity_matrix_ = 1 - similarity_matrix[:data.x_s.shape[0],
                data.x_s.shape[0]:]
        row_inds, col_inds = lsa(similarity_matrix_.cpu().detach().numpy())
        # visualize matches
        visualize_matches(col_inds, "visual_match_val.png")
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
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--batch_size', type=int, default=1)  # TODO --> default 64
parser.add_argument('--epochs', type=int, default=10000)
parser.add_argument('--checkpoint_path', type=str, default='experiment/best_iou_model.pth')
parser.add_argument('--save', type=bool, default=True)
args = parser.parse_args()

#train_path="/groups/funke/home/tame/data/new_seq/Raw/SeamCellCoordinates"
train_path="/groups/funke/home/tame/data/Untwisted/SeamCellCoordinates"
val_path="/groups/funke/home/tame/data/seamcellcoordinates"
# val_path="/groups/funke/home/tame/data/Untwisted/SeamCellCoordinates"

# logger
logger = Logger(['train_loss', 'train_accuracy', 'val_loss', 'val_accuracy'],
        'experiment64')

transform_train = T.Compose([
    T.RandomFlip(0),
    T.RandomRotate(120),
    T.RandomScale((0,2)),
    T.NormalizeScale(),
    T.Constant(),
])

transform_val = T.Compose([
    T.NormalizeScale(),
    T.Constant(),
    T.KNNGraph(k=3),
])

train_path = osp.join(train_path)
train_dataset = WormTrainDataset(train_path, type='train',
        transform=transform_train)
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, follow_batch=['x_s', 'x_t'])

val_path = osp.join(val_path)
val_dataset = WormValDataset(val_path, type='val',
        transform=transform_val)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, follow_batch=['x_s', 'x_t'])

data = train_dataset[0]

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
        x = self.layers[0].conv(x, edge_index, edge_attr)
        for layer in self.layers[1:]:
            x = layer(x, edge_index, edge_attr)
        x = self.layers[0].act(self.layers[0].norm(x))
        return self.lin(x)

# device = 'cuda' if torch.cuda.is_available() else 'cpu'
#device = "cpu" #!!!
device = "cuda"
#64,28
model = DeeperGCN(hidden_channels=128, num_layers=56).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

simclr_loss = SimCLR_Loss(temperature=0.1, device=device)
criterion = torch.nn.CrossEntropyLoss().to(device)
# load snapshot
if os.path.exists(args.checkpoint_path):
    state = torch.load(args.checkpoint_path)
    model.load_state_dict(state['model_state_dict'], strict=True)
    assert (True, 'checkpoint path found!')
else:
    assert (False, 'checkpoint_path {} does not exist!'.format(args.checkpoint_path))

EPS = 1e-8
save_model_every = 1000

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
    
    # save model
    if epoch % save_model_every == 0:
        torch.save(model, "/experiments/model" + str(epoch) + ".pth")
