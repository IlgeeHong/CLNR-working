import os
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid, Coauthor, Amazon
# from torch_geometric.loader import DataLoader
from torch_geometric.utils import from_scipy_sparse_matrix, to_undirected
from sklearn.neighbors import kneighbors_graph
from sklearn.datasets import make_moons, make_circles, make_swiss_roll
#from ogb.nodeproppred import PygNodePropPredDataset

def load(name, device):
    if name in ['Cora', 'CiteSeer', 'PubMed']:
        # Data = os.getcwd()+'/Planetoid'
        transform = T.Compose([T.NormalizeFeatures(),T.ToDevice(device)])                                                                                                          
        dataset = Planetoid(root = '/scratch/midway3/ilgee/SelfGCon/Planetoid', name=name, transform=transform)
        # loader = DataLoader(dataset, batch, shuffle=True)
        data = dataset[0]
        train_idx = data.train_mask 
        val_idx = data.val_mask 
        test_idx = data.test_mask  

    elif name in ['CS', 'Physics']:
        # Data = os.getcwd()+'/Coauthor'
        transform = T.Compose([T.ToDevice(device), T.RandomNodeSplit(split="train_rest", num_val = 0.1, num_test = 0.8)])
        dataset = Coauthor(name=name, root = '/scratch/midway3/ilgee/SelfGCon', transform=transform)
        # loader = DataLoader(dataset, batch, shuffle=True)
        data = dataset[0]
        train_idx = data.train_mask 
        val_idx = data.val_mask 
        test_idx = data.test_mask  

    elif name in ['Computers', 'Photo']:
        # Data = os.getcwd()+'/Amazon'
        transform = T.Compose([T.ToDevice(device), T.RandomNodeSplit(split="train_rest", num_val = 0.1, num_test = 0.8)])
        dataset = Amazon(name=name, root = '/scratch/midway3/ilgee/SelfGCon', transform=transform)
        # loader = DataLoader(dataset, batch, shuffle=True)
        data = dataset[0]
        train_idx = data.train_mask 
        val_idx = data.val_mask 
        test_idx = data.test_mask  
    
    # elif name in ['Swissroll','Moon','Circles']

    # elif name in ['ogbn-arxiv']:
    #     transform = T.Compose([T.ToDevice(device), T.ToUndirected()])
    #     dataset = PygNodePropPredDataset(name=name, root = '/scratch/midway3/ilgee/SelfGCon/dataset', transform=transform)
    #     data = dataset[0]
    #     split_idx = dataset.get_idx_split()
    #     train_idx = split_idx["train"]
    #     val_idx = split_idx["valid"]
    #     test_idx = split_idx["test"]

    return data, dataset, train_idx, val_idx, test_idx