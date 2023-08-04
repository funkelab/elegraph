import numpy as np
import torch
from torch_geometric.data import Data
from angle_test import curvature_3d
import glob
import torch_geometric.transforms as T

def set_curvatures(data, k, t):
    """
    Finds lowest t curvatures by finding all possible paths at each node in data.
    Sets data.x attribute to these lowest t curvatures in a torch tensor.
    """
    node_positions = data.pos
    edge_indices = data.edge_index
    angles = [[] for i in range(data.num_nodes)] # stores top (lowest) number of curvatures 

    def dfs(node, path, visited, index):
        visited[node] = True
        path = np.vstack((path, data.pos[node]))

        if len(path) == k:
            angles[index].append(curvature_3d(path))
            return

        for neighbor in edge_indices[0, edge_indices[1] == node]:
            if not visited[neighbor]:
                dfs(neighbor, path.copy(), visited, index)
                visited[neighbor] = False
    
    path = np.empty((0, 3))
    visited = torch.zeros(data.num_nodes, dtype=torch.bool)
    for i in range(data.num_nodes):
        dfs(i, path.copy(), visited.clone(), i)
    
    # only take the top (lowest) t curvatures for each node 
    for i in range(len(angles)):
        angles[i] = sorted(angles[i])[:t]
        assert len(angles[i]) == len(angles[0])
    data.x = torch.tensor(angles)
    assert len(data.x) == data.num_nodes == len(data.pos)
    

if __name__ == "__main__":
    cells = sorted(glob.glob("/groups/funke/home/tame/data/seamcellcoordinates/*.csv"))
    node_positions = np.genfromtxt(cells[26], delimiter=',')[1:, 1:4]
    node_positions = torch.tensor(node_positions)
    data = Data(pos=node_positions)
    #distance = T.Compose([T.KNNGraph(k=node_positions.size(0) - 1), T.Distance()])
    distance = T.Compose([T.KNNGraph(k=3), T.Distance()])
    data = distance(data)
    # data.edge_index = same_side(data)
    print(data.edge_index)
    data.num_nodes = node_positions.size(0)
    set_curvatures(data, 4, 2)
    #set_curvatures(data, 5, 2)
    print(data.x)
    # print(np.mean(data.x.detach().cpu().numpy(), axis=1))
