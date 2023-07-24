import numpy as np
import torch
from torch_geometric.data import Data
from angle_test import curvature_3d
import glob
import torch_geometric.transforms as T

def same_side(data):
    edge_index = data.edge_index
    target = []
    source = []
    for i in range(len(edge_index[0])):
        # both even
        if edge_index[1][i] % 2 == 0 and edge_index[0][i] % 2 == 0:
            target.append(edge_index[0][i].item())
            source.append(edge_index[1][i].item())
        # both odd
        if edge_index[1][i] % 2 != 0 and edge_index[0][i] % 2 != 0:
            target.append(edge_index[0][i].item())
            source.append(edge_index[1][i].item())
    new_edge_index = np.empty((2, len(target)))
    new_edge_index[0] = target
    new_edge_index[1] = source
    new_edge_index = new_edge_index.astype('i')
    return torch.tensor(new_edge_index).type(torch.int64) 

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
            if not visited[neighbor] and abs(neighbor - node) == 2: 
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
    # Sample graph data with node positions and edge indices
    # node_positions = torch.tensor([[0,1,2],  
    #                                [1,2,3],  
    #                                [2,3,4],
    #                                [3,4,5],
    #                                [3,7,9]])
    
    # edge_indices = torch.tensor([[0, 0, 0, 1, 1, 2, 2, 3, 4, 4],  # source nodes
    #                              [1, 2, 3, 0, 4, 0, 4, 0, 1, 2]]) # target nodes

    cells = sorted(glob.glob("/groups/funke/home/tame/data/seamcellcoordinates/*.csv"))
    node_positions = np.genfromtxt(cells[44], delimiter=',')[1:, 1:4]
    node_positions = torch.tensor(node_positions)
    data = Data(pos=node_positions)
    distance = T.Compose([T.KNNGraph(k=node_positions.size(0) - 1), T.Distance()])
    data = distance(data)
    data.edge_index = same_side(data)
    print(data.edge_index)
    data.num_nodes = node_positions.size(0)

    set_curvatures(data, 6, 1)
    #set_curvatures(data, 5, 2)
    print(data.x)
    # print(np.mean(data.x.detach().cpu().numpy(), axis=1))
