import torch
import sys
import torch_geometric.transforms as T
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.datasets import Planetoid, Coauthor, Amazon

sys.path.append('/scratch/midway3/ilgee/SelfGCon')

def load(name, device):
    if name in ['Cora', 'CiteSeer', 'PubMed']:
        transform = T.Compose([T.NormalizeFeatures(),T.ToDevice(device)])                                                                                                          
        dataset = Planetoid(root = 'Planetoid', name=name, transform=transform)
        data = dataset[0]
        train_idx = data.train_mask 
        val_idx = data.val_mask 
        test_idx = data.test_mask  

    if name in ['CS', 'Physics']:
        transform = T.Compose([T.ToDevice(device), T.RandomNodeSplit(split="train_rest", num_val = 0.1, num_test = 0.8)])
        dataset = Coauthor(name=name, transform=transform)
        data = dataset[0]
        train_idx = data.train_mask 
        val_idx = data.val_mask 
        test_idx = data.test_mask  

    if name in ['Computers', 'Photo']:
        transform = T.Compose([T.ToDevice(device), T.RandomNodeSplit(split="train_rest", num_val = 0.1, num_test = 0.8)])
        dataset = Amazon(name=name, transform=transform)
        data = dataset[0]
        train_idx = data.train_mask 
        val_idx = data.val_mask 
        test_idx = data.test_mask  

    if name in ['ogbn-arxiv']:
        transform = T.Compose([T.ToDevice(device), T.ToUndirected()])
        dataset = PygNodePropPredDataset(name=name, root = 'dataset/', transform=transform)
        data = dataset[0]
        split_idx = dataset.get_idx_split()
        train_idx = split_idx["train"]
        val_idx = split_idx["valid"]
        test_idx = split_idx["test"] 

    
    return data, train_idx, val_idx, test_idx