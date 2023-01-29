import os
import torch
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid, Coauthor, Amazon
from torch_geometric.data import Data
from torch_geometric.utils import from_scipy_sparse_matrix, to_undirected
from sklearn.neighbors import kneighbors_graph
from sklearn.datasets import make_moons, make_circles, make_swiss_roll
from ogb.nodeproppred import PygNodePropPredDataset
from copy import deepcopy
from aug_perturbed import *
import random

def load(name, sigma, alpha, outlier):
    if name in ['Cora', 'CiteSeer', 'PubMed']:
        # Data = os.getcwd()+'/Planetoid'
        transform = T.Compose([T.NormalizeFeatures()]) #,T.ToDevice(device)
        dataset = Planetoid(root = '/scratch/midway3/ilgee/SelfGCon/Planetoid', name=name) #, transform=transform
        temp = dataset[0]
        if sigma is not None:
        #    new_feat, _ = mask_feature(temp.feat, alpha, mode='all') 
            noise = torch.normal(0, sigma, size=(temp.num_nodes, temp.num_features))
        feat = temp.x + noise
        if alpha is not None:
            new_edge_index, _ = add_random_edge(temp.edge_index, alpha, force_undirected=True)
        # if outlier == True:
        #     noise = torch.zeros(temp.num_nodes, temp.num_features)
        #     out = 10000 * torch.ones(1, temp.num_features)
        #     # out = torch.zeros(1, temp.num_features)
        #     ind = torch.LongTensor(random.sample(range(temp.train_mask.sum()), 10))
        #     noise[ind,:] = out
        #     feat = temp.x + noise

        data = deepcopy(temp)
        if sigma is not None:
            data.x = new_feat #feat
        if alpha is not None:
            data.edge_index = new_edge_index
        if outlier is True:
            data.x = feat
        data = transform(data)
        train_idx = data.train_mask 
        val_idx = data.val_mask 
        test_idx = data.test_mask  

    elif name in ['CS', 'Physics']:
        # Data = os.getcwd()+'/Coauthor'
        transform = T.Compose([T.RandomNodeSplit(split="train_rest", num_val = 0.1, num_test = 0.8)]) #T.ToDevice(device), 
        dataset = Coauthor(name=name, root = '/scratch/midway3/ilgee/SelfGCon', transform=transform)
        temp = dataset[0]
        if sigma is not None:
            noise = torch.normal(0, sigma, size=(temp.num_nodes, temp.num_features))
            feat = temp.x + noise
        if alpha is not None:
            new_edge_index, _ = add_random_edge(temp.edge_index, alpha, force_undirected=True)
        if outlier == True:
            noise = torch.zeros(temp.num_nodes, temp.num_features)
            out = 10000 * torch.ones(1, temp.num_features)
            ind = torch.LongTensor(random.sample(range(temp.num_nodes), 1))
            noise[ind,:] = out
            feat = temp.x + noise

        data = deepcopy(temp)
        if sigma is not None:
            data.x = feat
        if alpha is not None:
            data.edge_index = new_edge_index
        if outlier is True:
            data.x = feat
        data = transform(data)
        train_idx = data.train_mask 
        val_idx = data.val_mask 
        test_idx = data.test_mask

    elif name in ['Computers', 'Photo']:
        # Data = os.getcwd()+'/Amazon'
        transform = T.Compose([T.RandomNodeSplit(split="train_rest", num_val = 0.1, num_test = 0.8)]) #T.ToDevice(device), 
        dataset = Amazon(name=name, root = '/scratch/midway3/ilgee/SelfGCon', transform=transform)
        temp = dataset[0]
        if sigma is not None:
            noise = torch.normal(0, sigma, size=(temp.num_nodes, temp.num_features))
            feat = temp.x + noise
        if alpha is not None:
            new_edge_index, _ = add_random_edge(temp.edge_index, alpha, force_undirected=True)
        if outlier == True:
            noise = torch.zeros(temp.num_nodes, temp.num_features)
            out = 10000 * torch.ones(1, temp.num_features)
            ind = torch.LongTensor(random.sample(range(temp.num_nodes), 1))
            noise[ind,:] = out
            feat = temp.x + noise

        data = deepcopy(temp)
        if sigma is not None:
            data.x = feat
        if alpha is not None:
            data.edge_index = new_edge_index
        if outlier is True:
            data.x = feat
        data = transform(data)
        train_idx = data.train_mask 
        val_idx = data.val_mask 
        test_idx = data.test_mask
    
    elif name in ['Swissroll','Moon','Circles']:
        if name == 'Moon':
            XX, y = make_moons(n_samples=10000) #, noise=args.noise
        elif name == 'Swissroll':
            XX, y = make_swiss_roll(n_samples=10000)
        elif name == 'Circles':
            XX, y = make_circles(n_samples=10000, factor=0.4)
        A = kneighbors_graph(XX, 15, mode='distance', include_self=False)
        edge_index, edge_weights = from_scipy_sparse_matrix(A)
        edge_index, edge_weights = to_undirected(edge_index, edge_weights)
        transform = T.RandomNodeSplit(split="train_rest", num_val = 0.1, num_test = 0.8)
        data = Data(x=torch.eye(10000), edge_index=edge_index, edge_weight=edge_weights, y=torch.tensor(y))
        data = transform(data)
        train_idx = data.train_mask 
        val_idx = data.val_mask 
        test_idx = data.test_mask  

    elif name in ['ogbn-arxiv']:
        transform = T.Compose([T.ToUndirected()])
        dataset = PygNodePropPredDataset(name=name, root = '/scratch/midway3/ilgee/SelfGCon/dataset', transform=transform)
        temp = dataset[0]
        print(temp.x.shape)
        if sigma is not None:
            noise = torch.normal(0, sigma, size=(temp.num_nodes, temp.num_features))
            feat = temp.x + noise
        if alpha is not None:
            new_edge_index, _ = add_random_edge(temp.edge_index, alpha, force_undirected=True)
        if outlier == True:
            noise = torch.zeros(temp.num_nodes, temp.num_features)
            out = 10000 * torch.ones(1, temp.num_features)
            ind = torch.LongTensor(random.sample(range(temp.num_nodes), 1))
            noise[ind,:] = out
            feat = temp.x + noise

        data = deepcopy(temp)
        if sigma is not None:
            data.x = feat
        if alpha is not None:
            data.edge_index = new_edge_index
        if outlier is True:
            data.x = feat
        data = transform(data)
        split_idx = dataset.get_idx_split()
        train_idx = split_idx["train"]
        val_idx = split_idx["valid"]
        test_idx = split_idx["test"]

    return data, train_idx, val_idx, test_idx