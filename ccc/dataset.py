import os
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid, Coauthor, Amazon

def load(name, device):
    if name in ['Cora', 'CiteSeer', 'PubMed']:
        Data = os.getcwd()+'/Planetoid'
        transform = T.Compose([T.NormalizeFeatures(),T.ToDevice(device)])                                                                                                          
        dataset = Planetoid(root = 'Data', name=name, transform=transform)
        data = dataset[0]
        train_idx = data.train_mask 
        val_idx = data.val_mask 
        test_idx = data.test_mask  

    elif name in ['CS', 'Physics']:
        Data = os.getcwd()+'/Coauthor'
        transform = T.Compose([T.ToDevice(device), T.RandomNodeSplit(split="train_rest", num_val = 0.1, num_test = 0.8)])
        dataset = Coauthor(name=name, root = 'Data', transform=transform)
        data = dataset[0]
        train_idx = data.train_mask 
        val_idx = data.val_mask 
        test_idx = data.test_mask  

    elif name in ['Computers', 'Photo']:
        Data = os.getcwd()+'/Amazon'
        transform = T.Compose([T.ToDevice(device), T.RandomNodeSplit(split="train_rest", num_val = 0.1, num_test = 0.8)])
        dataset = Amazon(name=name, root = 'Data', transform=transform)
        data = dataset[0]
        train_idx = data.train_mask 
        val_idx = data.val_mask 
        test_idx = data.test_mask  

    return data, train_idx, val_idx, test_idx